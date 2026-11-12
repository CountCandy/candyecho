"""The llama.cpp server that runs the tag-enhancer LLM from a GGUF file.

The enhancer is a quantised GGUF model (Qwen3.5-9B at Q8 by default). GGUF
needs llama.cpp to run it, and adding ``llama-cpp-python`` would mean compiling
CUDA code on the user's machine. llama.cpp publishes ready-built servers
instead, so on Windows CandyEcho fetches that build next to the model and runs
it as a child process, exactly as it does for Drama 3's server: started on
first use, reached over localhost with ``urllib``, and stopped to give the VRAM
back.

Two pieces live here:

* ``LlamaCppReleaseHub`` plugs into ``ModelStore`` so the server build is
  downloaded, verified and recorded in the same manifest as the model file.
* ``llama_server_config`` builds the launch recipe for ``EngineServer``.

Stdlib-only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable

from .engine_servers import EngineServerConfig

logger = logging.getLogger(__name__)

LLAMA_REPO = "ggml-org/llama.cpp"
RUNTIME_SUBDIR = "llama.cpp"
SERVER_KEY = "enhancer"

_WIN_CUDA_RE = re.compile(r"^llama-(?P<tag>.+)-bin-win-cuda-(?P<major>\d+)\.(?P<minor>\d+)-x64\.zip$")


def server_binary_name(windows: bool | None = None) -> str:
    windows = (os.name == "nt") if windows is None else windows
    return "llama-server.exe" if windows else "llama-server"


def managed_runtime_supported(windows: bool | None = None, machine: str | None = None) -> bool:
    """Whether CandyEcho can fetch a llama.cpp build itself on this machine.

    Only Windows x64 with an NVIDIA card, which is what the published CUDA
    builds target. Elsewhere the user installs llama.cpp (a package manager has
    it) and CandyEcho finds ``llama-server`` on PATH.
    """
    windows = (os.name == "nt") if windows is None else windows
    machine = (machine or platform.machine() or "").lower()
    return windows and machine in ("amd64", "x86_64")


def choose_windows_assets(assets: list[dict[str, Any]], cuda_major: int | None = None) -> list[dict[str, Any]]:
    """Pick the server zip and its CUDA runtime zip from a release's assets.

    Asset names carry a CUDA version that changes between releases
    (``...-win-cuda-13.3-x64.zip``, later ``13.5``), so they are matched by
    pattern rather than spelled out. CUDA 13 is preferred because Echo's own
    PyTorch already requires a driver new enough for it.
    """
    builds = []
    for asset in assets:
        match = _WIN_CUDA_RE.match(str(asset.get("name", "")))
        if match:
            builds.append((int(match["major"]), int(match["minor"]), asset))
    if cuda_major is not None:
        builds = [b for b in builds if b[0] == cuda_major]
    if not builds:
        wanted = f"CUDA {cuda_major}" if cuda_major is not None else "CUDA"
        raise RuntimeError(f"This llama.cpp release has no Windows x64 {wanted} build.")
    # Prefer CUDA 13, then the newest minor.
    builds.sort(key=lambda b: (b[0] == 13, b[0], b[1]), reverse=True)
    major, minor, server = builds[0]
    runtime_name = f"cudart-llama-bin-win-cuda-{major}.{minor}-x64.zip"
    runtime = next((a for a in assets if a.get("name") == runtime_name), None)
    if runtime is None:
        raise RuntimeError(f"This llama.cpp release is missing {runtime_name}.")
    return [server, runtime]


def _safe_extract(archive: Path, target: Path) -> None:
    """Unzip, refusing any entry that would land outside ``target``."""
    root = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            destination = (target / info.filename).resolve()
            if destination != root and root not in destination.parents:
                raise RuntimeError(f"{archive.name} contains an unsafe path: {info.filename}")
        zf.extractall(target)


class LlamaCppReleaseHub:
    """``ModelStore`` hub that installs a llama.cpp server build from GitHub.

    ``list_files`` reports the zips (so the progress bar has a total) and
    ``download`` fetches them, checks each against the size and SHA-256 GitHub
    publishes, unpacks them and returns the unpacked files, which is what the
    manifest then records.
    """

    API = "https://api.github.com/repos/{repo}/releases/{which}"
    # llama.cpp's "latest" release is a version marker (v0.9.0) holding only
    # this file, which names the build release (b12110) the binaries are in.
    POINTER_ASSET = "nightly-tag.txt"

    def __init__(
        self,
        release: str | None = None,
        cuda_major: int | None = None,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ):
        # A fixed tag ("b12283") or None for the latest release.
        self.release = release or os.environ.get("CANDYECHO_LLAMA_RELEASE") or None
        env_cuda = os.environ.get("CANDYECHO_LLAMA_CUDA", "").strip()
        self.cuda_major = cuda_major if cuda_major is not None else (int(env_cuda) if env_cuda.isdigit() else None)
        self._opener = opener
        self._assets: dict[str, dict[str, Any]] = {}

    def _get_json(self, url: str, missing: str | None = None) -> Any:
        request = urllib.request.Request(
            url, headers={"Accept": "application/vnd.github+json", "User-Agent": "candyecho"}
        )
        try:
            with self._opener(request, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404 and missing:
                raise RuntimeError(missing) from e
            raise RuntimeError(f"GitHub returned HTTP {e.code} for the llama.cpp release list.") from e
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"connection problem reaching GitHub: {e}") from e

    def _release_info(self, repo_id: str, tag: str | None = None) -> dict[str, Any]:
        tag = tag or self.release
        which = f"tags/{tag}" if tag else "latest"
        return self._get_json(
            self.API.format(repo=repo_id, which=which),
            missing=f"llama.cpp has no release named {tag}." if tag else None,
        )

    def _pointed_build(self, info: dict[str, Any]) -> str | None:
        """The build tag a version-marker release points at, if it is one."""
        pointer = next(
            (a for a in info.get("assets") or [] if a.get("name") == self.POINTER_ASSET), None
        )
        if pointer is None:
            return None
        request = urllib.request.Request(
            pointer["browser_download_url"], headers={"User-Agent": "candyecho"}
        )
        try:
            with self._opener(request, timeout=30) as resp:
                tag = resp.read(200).decode("utf-8", "replace").strip()
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"connection problem reaching GitHub: {e}") from e
        return tag if re.fullmatch(r"[A-Za-z0-9._-]{1,40}", tag) else None

    def _find_build(self, repo_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """The release to install and its two Windows assets.

        Tries, in order: the release asked for (or "latest"); the build that a
        version-marker release points at; then the newest published release
        that actually has a Windows CUDA build.
        """
        info = self._release_info(repo_id)
        try:
            return info, choose_windows_assets(info.get("assets") or [], self.cuda_major)
        except RuntimeError as first_error:
            pointed = self._pointed_build(info)
            if pointed:
                info = self._release_info(repo_id, pointed)
                try:
                    return info, choose_windows_assets(info.get("assets") or [], self.cuda_major)
                except RuntimeError:
                    pass
            if self.release:
                raise first_error       # the user pinned a release: do not wander
            recent = self._get_json(self.API.format(repo=repo_id, which="").rstrip("/") + "?per_page=30")
            for candidate in recent if isinstance(recent, list) else []:
                if candidate.get("draft"):
                    continue
                try:
                    return candidate, choose_windows_assets(candidate.get("assets") or [], self.cuda_major)
                except RuntimeError:
                    continue
            raise first_error

    def list_files(self, repo_id: str) -> tuple[str, dict[str, int]]:
        info, chosen = self._find_build(repo_id)
        self._assets = {a["name"]: a for a in chosen}
        return str(info.get("tag_name") or "latest"), {a["name"]: int(a["size"]) for a in chosen}

    def _fetch(self, asset: dict[str, Any], path: Path) -> None:
        size = int(asset["size"])
        digest = str(asset.get("digest") or "")
        expected_sha = digest.split(":", 1)[1].lower() if digest.startswith("sha256:") else None
        request = urllib.request.Request(asset["browser_download_url"], headers={"User-Agent": "candyecho"})
        sha = hashlib.sha256()
        written = 0
        try:
            with self._opener(request, timeout=60) as resp, open(path, "wb") as out:
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    sha.update(block)
                    written += len(block)
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"connection problem downloading {asset['name']}: {e}") from e
        if written != size or (expected_sha and sha.hexdigest() != expected_sha):
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"{asset['name']} arrived damaged ({written} of {size} bytes). Press the "
                "button again; if it keeps happening, set CANDYECHO_LLAMA_RELEASE to an "
                "older llama.cpp release tag."
            )

    def download(
        self, repo_id: str, revision: str, local_dir: Path, allow_patterns: list[str] | None
    ) -> dict[str, int]:
        if not self._assets:
            self.list_files(repo_id)
        local_dir = Path(local_dir)
        # Replace any earlier build wholesale: mixing DLLs from two releases
        # is the classic way to get a server that will not start.
        if local_dir.exists():
            shutil.rmtree(local_dir)
        local_dir.mkdir(parents=True)
        for name, asset in self._assets.items():
            archive = local_dir / name
            logger.info(f"Downloading {name} ({int(asset['size']) / 1e6:.0f} MB)")
            self._fetch(asset, archive)
            _safe_extract(archive, local_dir)
            archive.unlink()
        servers = sorted(local_dir.rglob("llama-server*"))
        if not servers:
            raise RuntimeError("The llama.cpp download did not contain llama-server.")
        # Windows loads DLLs from the program's own folder. Should a release
        # ever nest the server in a subfolder, bring the runtime DLLs to it.
        home = servers[0].parent
        if home != local_dir:
            for dll in local_dir.glob("*.dll"):
                if not (home / dll.name).exists():
                    shutil.copy2(dll, home / dll.name)
        (local_dir / "RELEASE.txt").write_text(f"{repo_id} {revision}\n", encoding="utf-8")
        return {
            path.relative_to(local_dir).as_posix(): path.stat().st_size
            for path in sorted(local_dir.rglob("*"))
            if path.is_file()
        }


def find_llama_server(runtime_dir: Path | str | None, windows: bool | None = None) -> Path | None:
    """Locate ``llama-server``: an explicit override, the managed build, then PATH."""
    override = os.environ.get("CANDYECHO_LLAMA_SERVER", "").strip()
    if override:
        path = Path(override)
        return path if path.is_file() else None
    name = server_binary_name(windows)
    if runtime_dir is not None:
        base = Path(runtime_dir)
        direct = base / name
        if direct.is_file():
            return direct
        if base.is_dir():
            found = sorted(base.rglob(name))
            if found:
                return found[0]
    on_path = shutil.which("llama-server")
    return Path(on_path) if on_path else None


def find_gguf(model_dir: Path | str, filename: str | None = None) -> Path | None:
    """The model file inside its download folder."""
    base = Path(model_dir)
    if filename and (base / filename).is_file():
        return base / filename
    candidates = sorted(p for p in base.glob("*.gguf") if not p.name.lower().startswith("mmproj"))
    return candidates[0] if candidates else None


def llama_server_config(
    server_path: Path | str,
    model_path: Path | str,
    url: str | None = None,
    context: int = 8192,
) -> EngineServerConfig:
    """Launch recipe for the enhancer's llama.cpp server.

    Reasoning is switched off at the server: the enhancer's job is to copy a
    script and insert tags, and a thinking model would spend most of its time
    and tokens deliberating before every paragraph.
    """
    url = (url or os.environ.get("CANDYECHO_ENHANCER_SERVER_URL") or "http://127.0.0.1:8090").rstrip("/")
    config = EngineServerConfig(
        key=SERVER_KEY,
        label="Tag enhancer",
        url=url,
        health_path="/health",
        startup_timeout=600.0,
        setup_hint="See logs/engine-enhancer.log for what llama.cpp reported.",
    )
    config.start_cmd = [
        str(server_path),
        "--model", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(config.port),
        "--gpu-layers", "all",
        "--ctx-size", str(context),
        "--parallel", "1",
        "--reasoning", "off",
        "--no-webui",
        "--alias", "candyecho-enhancer",
    ]
    return config
