"""Downloadable model weights: what is needed, what is on disk, and fetching it.

Echo's own weights are pulled by the vendored loader at startup. The optional
engines (Drama 3) and the tag-enhancer LLM are different: they are tens
of gigabytes, nobody should be made to fetch them just to run Echo, and the
user has to be able to see at a glance whether they are already present.

So each model lives in a predictable folder under ``models/`` and, once a
download has finished, gets a small manifest written next to it::

    models/drama3/candyecho-model.json

The manifest lists every file with its size. A model counts as *ready* only
when that manifest exists and every file in it is still on disk at the recorded
size, which is a handful of ``stat`` calls and needs no network. That is what
stops the UI from offering to download something that is already there, and
what notices a folder that was half-deleted by hand.

This module imports nothing heavy at import time (``huggingface_hub`` is loaded
only when a download actually starts) so it can be tested without the ML stack.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

MANIFEST_NAME = "candyecho-model.json"
MANIFEST_VERSION = 1

STATE_MISSING = "missing"
STATE_PARTIAL = "partial"
STATE_DOWNLOADING = "downloading"
STATE_READY = "ready"
STATE_ERROR = "error"

DEFAULT_ENHANCER_REPO = "Qwen/Qwen2.5-7B-Instruct"


@dataclass(frozen=True)
class RepoSpec:
    """One Hugging Face repository that is part of a model."""

    repo_id: str
    # Folder under the model's directory that the repo is downloaded into.
    subdir: str
    # Restrict the download to matching files (fnmatch patterns); None = all.
    allow_patterns: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    description: str
    repos: tuple[RepoSpec, ...]
    approx_gb: float
    license_name: str
    license_url: str


def default_specs(enhancer_repo: str | None = None) -> dict[str, ModelSpec]:
    """The models CandyEcho knows how to fetch."""
    enhancer_repo = enhancer_repo or os.environ.get(
        "CANDYECHO_ENHANCER_MODEL", DEFAULT_ENHANCER_REPO
    )
    specs = [
        ModelSpec(
            key="drama3",
            label="Fish Audio Drama 3",
            description="35B mixture-of-experts model, plus the S2-Pro codec it decodes with.",
            repos=(
                RepoSpec("fishaudio/drama-3", "drama-3"),
                # Drama 3 decodes with the S2-Pro codec; only that one file is needed.
                RepoSpec("fishaudio/s2-pro", "s2-pro", allow_patterns=("codec.pth",)),
            ),
            approx_gb=41.0,
            license_name="Fish Audio Community License",
            license_url="https://huggingface.co/fishaudio/drama-3",
        ),
        ModelSpec(
            key="enhancer",
            label="Tag enhancer LLM",
            description=f"{enhancer_repo}: writes Drama 3 direction tags into a script.",
            repos=(RepoSpec(enhancer_repo, enhancer_repo.rsplit("/", 1)[-1]),),
            approx_gb=15.5,
            license_name="See model card",
            license_url=f"https://huggingface.co/{enhancer_repo}",
        ),
    ]
    return {s.key: s for s in specs}


class Hub(Protocol):
    """The two Hugging Face operations the store needs (swappable in tests)."""

    def list_files(self, repo_id: str) -> tuple[str, dict[str, int]]:
        """Return (revision sha, {relative path: size in bytes})."""

    def download(
        self, repo_id: str, revision: str, local_dir: Path, allow_patterns: list[str] | None
    ) -> None:
        """Download the repo (or the matching files) into local_dir, resuming if possible."""


class HuggingFaceHub:
    """Real implementation, backed by ``huggingface_hub``."""

    def list_files(self, repo_id: str) -> tuple[str, dict[str, int]]:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True)
        files = {s.rfilename: int(s.size or 0) for s in info.siblings or []}
        return info.sha or "main", files

    def download(
        self, repo_id: str, revision: str, local_dir: Path, allow_patterns: list[str] | None
    ) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id,
            revision=revision,
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
        )


def _friendly_error(exc: BaseException, spec: ModelSpec) -> str:
    """Turn a download failure into something a person can act on."""
    name = type(exc).__name__
    text = str(exc)
    lowered = f"{name} {text}".lower()
    if "gated" in lowered or "401" in lowered or "403" in lowered:
        return (
            f"Hugging Face refused the download. Open {spec.license_url}, accept the "
            "licence while signed in, then run `hf auth login` (or set HF_TOKEN) and "
            "press the button again."
        )
    if "no space left" in lowered or "errno 28" in lowered:
        return f"The disk is full. {spec.label} needs about {spec.approx_gb:.0f} GB free."
    if any(k in lowered for k in ("connection", "timeout", "timed out", "resolve", "offline")):
        return (
            "The download was interrupted by a network problem. Press the button "
            "again to resume; finished files are kept."
        )
    return f"{name}: {text}" if text else name


@dataclass
class _Download:
    thread: threading.Thread | None = None
    started: float = 0.0
    bytes_total: int = 0
    error: str | None = None
    active: bool = False
    # Paths (relative to the model dir) expected by the download in flight.
    expected: dict[str, int] = field(default_factory=dict)


class ModelStore:
    """Tracks and downloads model weights under one root directory."""

    def __init__(
        self,
        root: Path | str = "models",
        specs: dict[str, ModelSpec] | None = None,
        hub: Hub | None = None,
        dirs: dict[str, Path] | None = None,
    ):
        self.root = Path(root)
        # Per-model overrides of where the weights live (default: root/<key>).
        self.dirs = {k: Path(v) for k, v in (dirs or {}).items()}
        self.specs = specs if specs is not None else default_specs()
        self.hub: Hub = hub or HuggingFaceHub()
        self._downloads: dict[str, _Download] = {}
        self._lock = threading.Lock()

    # -- paths ---------------------------------------------------------------

    def spec(self, key: str) -> ModelSpec:
        try:
            return self.specs[key]
        except KeyError:
            raise KeyError(f"Unknown model '{key}'") from None

    def model_dir(self, key: str) -> Path:
        self.spec(key)
        return self.dirs.get(key, self.root / key)

    def repo_dir(self, key: str, index: int = 0) -> Path:
        """Folder a given repo of the model is downloaded into."""
        return self.model_dir(key) / self.spec(key).repos[index].subdir

    def manifest_path(self, key: str) -> Path:
        return self.model_dir(key) / MANIFEST_NAME

    # -- status --------------------------------------------------------------

    def _read_manifest(self, key: str) -> dict | None:
        path = self.manifest_path(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("version") != MANIFEST_VERSION:
            return None
        if not isinstance(data.get("files"), dict) or not data["files"]:
            return None
        return data

    def _manifest_intact(self, key: str, manifest: dict) -> bool:
        base = self.model_dir(key)
        for rel, size in manifest["files"].items():
            try:
                if (base / rel).stat().st_size != int(size):
                    return False
            except OSError:
                return False
        return True

    def is_ready(self, key: str) -> bool:
        manifest = self._read_manifest(key)
        return manifest is not None and self._manifest_intact(key, manifest)

    def _bytes_on_disk(self, key: str) -> int:
        """Bytes present under the model dir, counting partial downloads too."""
        total = 0
        base = self.model_dir(key)
        if not base.exists():
            return 0
        for dirpath, _dirs, files in os.walk(base):
            for name in files:
                if name == MANIFEST_NAME:
                    continue
                try:
                    total += (Path(dirpath) / name).stat().st_size
                except OSError:
                    pass
        return total

    def status(self, key: str) -> dict[str, Any]:
        spec = self.spec(key)
        with self._lock:
            dl = self._downloads.get(key)
            active = bool(dl and dl.active)
            error = dl.error if dl else None
            total = dl.bytes_total if dl else 0

        result: dict[str, Any] = {
            "key": key,
            "label": spec.label,
            "description": spec.description,
            "approx_gb": spec.approx_gb,
            "license": spec.license_name,
            "license_url": spec.license_url,
            "path": str(self.model_dir(key)),
            "bytes_done": 0,
            "bytes_total": total,
            "error": None,
        }

        if active:
            result["state"] = STATE_DOWNLOADING
            done = self._bytes_on_disk(key)
            # Lock/metadata files can push the on-disk figure a hair past the
            # payload size; never report more than 100%.
            result["bytes_done"] = min(done, total) if total else done
            return result

        manifest = self._read_manifest(key)
        if manifest is not None and self._manifest_intact(key, manifest):
            size = sum(int(v) for v in manifest["files"].values())
            result.update(state=STATE_READY, bytes_done=size, bytes_total=size)
            return result

        if error:
            result.update(state=STATE_ERROR, error=error, bytes_done=self._bytes_on_disk(key))
            return result

        on_disk = self._bytes_on_disk(key)
        result["bytes_done"] = on_disk
        result["state"] = STATE_PARTIAL if on_disk > 0 else STATE_MISSING
        return result

    def all_status(self) -> list[dict[str, Any]]:
        return [self.status(key) for key in self.specs]

    # -- download ------------------------------------------------------------

    def start_download(self, key: str) -> bool:
        """Begin (or resume) downloading a model in the background.

        Returns False if a download of this model is already running or the
        model is already complete; True if one was started.
        """
        spec = self.spec(key)
        with self._lock:
            dl = self._downloads.get(key)
            if dl and dl.active:
                return False
            if self.is_ready(key):
                return False
            dl = _Download(started=time.time(), active=True)
            self._downloads[key] = dl
            dl.thread = threading.Thread(
                target=self._run_download, args=(spec, dl), name=f"download-{key}", daemon=True
            )
            dl.thread.start()
        return True

    def wait(self, key: str, timeout: float | None = None) -> None:
        """Block until the current download of ``key`` finishes (used by tests)."""
        with self._lock:
            dl = self._downloads.get(key)
            thread = dl.thread if dl else None
        if thread is not None:
            thread.join(timeout)

    def _run_download(self, spec: ModelSpec, dl: _Download) -> None:
        base = self.model_dir(spec.key)
        try:
            base.mkdir(parents=True, exist_ok=True)
            # A stale manifest must not outlive a re-download that fails midway.
            self.manifest_path(spec.key).unlink(missing_ok=True)

            plan: list[tuple[RepoSpec, str, dict[str, int]]] = []
            for repo in spec.repos:
                revision, files = self.hub.list_files(repo.repo_id)
                if repo.allow_patterns:
                    files = {
                        name: size
                        for name, size in files.items()
                        if any(fnmatch.fnmatch(name, pat) for pat in repo.allow_patterns)
                    }
                if not files:
                    raise RuntimeError(
                        f"{repo.repo_id} has none of the expected files "
                        f"({', '.join(repo.allow_patterns or ('*',))})"
                    )
                plan.append((repo, revision, files))

            expected: dict[str, int] = {}
            for repo, _revision, files in plan:
                for name, size in files.items():
                    expected[f"{repo.subdir}/{name}"] = size
            with self._lock:
                dl.expected = expected
                dl.bytes_total = sum(expected.values())

            for repo, revision, _files in plan:
                logger.info(f"Downloading {repo.repo_id} into {base / repo.subdir}")
                self.hub.download(
                    repo.repo_id,
                    revision,
                    base / repo.subdir,
                    list(repo.allow_patterns) if repo.allow_patterns else None,
                )

            missing = [
                rel for rel, size in expected.items()
                if not (base / rel).is_file() or (base / rel).stat().st_size != size
            ]
            if missing:
                raise RuntimeError(
                    f"Download finished but {len(missing)} file(s) are missing or the "
                    f"wrong size (first: {missing[0]})"
                )

            manifest = {
                "version": MANIFEST_VERSION,
                "key": spec.key,
                "repos": [
                    {"repo_id": repo.repo_id, "revision": revision, "subdir": repo.subdir}
                    for repo, revision, _files in plan
                ],
                "files": expected,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            tmp = self.manifest_path(spec.key).with_suffix(".tmp")
            tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(tmp, self.manifest_path(spec.key))
            logger.info(f"{spec.label} downloaded and verified")
            with self._lock:
                dl.error = None
        except BaseException as exc:  # noqa: BLE001 - reported to the UI, never raised
            logger.error(f"Download of {spec.label} failed: {exc}")
            with self._lock:
                dl.error = _friendly_error(exc, spec)
        finally:
            with self._lock:
                dl.active = False
