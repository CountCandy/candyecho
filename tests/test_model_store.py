"""Model store: recognising finished downloads, resuming, and reporting errors."""
import json
from pathlib import Path

import pytest

from longecho.model_store import (
    MANIFEST_NAME,
    ModelSpec,
    ModelStore,
    RepoSpec,
    default_specs,
)


class FakeHub:
    """Stands in for Hugging Face: a dict of repo -> {file: bytes}."""

    def __init__(self, repos, fail_on=None, skip=()):
        self.repos = repos
        self.fail_on = fail_on
        self.skip = set(skip)
        self.downloads = []

    def list_files(self, repo_id):
        return f"sha-{repo_id}", {name: len(data) for name, data in self.repos[repo_id].items()}

    def download(self, repo_id, revision, local_dir, allow_patterns):
        self.downloads.append((repo_id, revision, allow_patterns))
        if self.fail_on == repo_id:
            raise ConnectionError("connection reset by peer")
        local_dir = Path(local_dir)
        for name, data in self.repos[repo_id].items():
            if allow_patterns and name not in allow_patterns:
                continue
            if name in self.skip:
                continue
            target = local_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


SPECS = {
    "toy": ModelSpec(
        key="toy",
        label="Toy model",
        description="",
        repos=(
            RepoSpec("org/main", "main"),
            RepoSpec("org/codec", "codec", allow_patterns=("codec.pth",)),
        ),
        approx_gb=0.1,
        license_name="Test",
        license_url="https://example.invalid/toy",
    )
}
REPOS = {
    "org/main": {"config.json": b"{}", "weights/model.safetensors": b"x" * 100},
    "org/codec": {"codec.pth": b"c" * 10, "huge-unneeded.bin": b"z" * 999},
}


def make_store(tmp_path, **hub_kwargs):
    hub = FakeHub(REPOS, **hub_kwargs)
    return ModelStore(tmp_path / "models", specs=SPECS, hub=hub), hub


def test_missing_then_ready_after_download(tmp_path):
    store, hub = make_store(tmp_path)
    assert store.status("toy")["state"] == "missing"
    assert not store.is_ready("toy")

    assert store.start_download("toy") is True
    store.wait("toy", timeout=10)

    status = store.status("toy")
    assert status["state"] == "ready"
    assert status["bytes_total"] == 2 + 100 + 10
    assert store.is_ready("toy")
    # Only the allowed file of the second repo was fetched.
    assert (tmp_path / "models/toy/codec/codec.pth").exists()
    assert not (tmp_path / "models/toy/codec/huge-unneeded.bin").exists()
    assert ("org/codec", "sha-org/codec", ["codec.pth"]) in hub.downloads


def test_ready_model_is_recognised_by_a_fresh_store_without_network(tmp_path):
    store, _ = make_store(tmp_path)
    store.start_download("toy")
    store.wait("toy", timeout=10)

    class NoNetwork:
        def list_files(self, repo_id):
            raise AssertionError("must not touch the network to check a finished model")

        def download(self, *a, **k):
            raise AssertionError("must not re-download a finished model")

    again = ModelStore(tmp_path / "models", specs=SPECS, hub=NoNetwork())
    assert again.status("toy")["state"] == "ready"
    assert again.start_download("toy") is False


def test_deleted_file_demotes_ready_to_partial(tmp_path):
    store, _ = make_store(tmp_path)
    store.start_download("toy")
    store.wait("toy", timeout=10)
    (tmp_path / "models/toy/main/weights/model.safetensors").unlink()
    assert store.status("toy")["state"] == "partial"
    assert not store.is_ready("toy")


def test_truncated_file_is_not_ready(tmp_path):
    store, _ = make_store(tmp_path)
    store.start_download("toy")
    store.wait("toy", timeout=10)
    (tmp_path / "models/toy/main/weights/model.safetensors").write_bytes(b"short")
    assert not store.is_ready("toy")


def test_files_without_manifest_are_partial_and_resumable(tmp_path):
    store, hub = make_store(tmp_path)
    target = tmp_path / "models/toy/main"
    target.mkdir(parents=True)
    (target / "config.json").write_bytes(b"{}")
    assert store.status("toy")["state"] == "partial"
    assert store.start_download("toy") is True
    store.wait("toy", timeout=10)
    assert store.status("toy")["state"] == "ready"


def test_network_failure_reports_error_and_can_be_retried(tmp_path):
    store, hub = make_store(tmp_path, fail_on="org/codec")
    store.start_download("toy")
    store.wait("toy", timeout=10)
    status = store.status("toy")
    assert status["state"] == "error"
    assert "resume" in status["error"].lower()
    assert not (tmp_path / "models/toy" / MANIFEST_NAME).exists()

    hub.fail_on = None
    assert store.start_download("toy") is True
    store.wait("toy", timeout=10)
    assert store.status("toy")["state"] == "ready"


def test_incomplete_download_is_never_marked_ready(tmp_path):
    store, _ = make_store(tmp_path, skip={"weights/model.safetensors"})
    store.start_download("toy")
    store.wait("toy", timeout=10)
    status = store.status("toy")
    assert status["state"] == "error"
    assert "missing" in status["error"]


def test_gated_repo_error_tells_the_user_what_to_do(tmp_path):
    class Gated(FakeHub):
        def list_files(self, repo_id):
            raise PermissionError("401 Client Error: gated repo")

    store = ModelStore(tmp_path / "models", specs=SPECS, hub=Gated(REPOS))
    store.start_download("toy")
    store.wait("toy", timeout=10)
    error = store.status("toy")["error"]
    assert "https://example.invalid/toy" in error and "hf auth login" in error


def test_corrupt_manifest_is_ignored(tmp_path):
    store, _ = make_store(tmp_path)
    folder = tmp_path / "models/toy"
    folder.mkdir(parents=True)
    (folder / MANIFEST_NAME).write_text("{not json")
    assert store.status("toy")["state"] == "missing"
    (folder / MANIFEST_NAME).write_text(json.dumps({"version": 1, "files": {}}))
    assert not store.is_ready("toy")


def test_unknown_model_raises(tmp_path):
    store, _ = make_store(tmp_path)
    with pytest.raises(KeyError):
        store.status("nope")


def test_default_specs_cover_drama3_and_the_enhancer(monkeypatch):
    for name in ("CANDYECHO_ENHANCER_URL", "CANDYECHO_ENHANCER_BACKEND", "CANDYECHO_LLAMA_SERVER",
                 "CANDYECHO_ENHANCER_GGUF_REPO", "CANDYECHO_ENHANCER_GGUF_FILE"):
        monkeypatch.delenv(name, raising=False)
    specs = default_specs()
    assert set(specs) == {"drama3", "enhancer"}
    drama = specs["drama3"]
    assert [r.repo_id for r in drama.repos] == ["fishaudio/drama-3", "fishaudio/s2-pro"]
    # Only the codec is taken from S2-Pro, not its 10+ GB of weights.
    assert drama.repos[1].allow_patterns == ("codec.pth",)


def test_enhancer_spec_is_one_gguf_file_plus_the_server_build_where_managed(monkeypatch):
    from longecho.model_store import SOURCE_LLAMA_CPP, enhancer_spec

    for name in ("CANDYECHO_ENHANCER_URL", "CANDYECHO_ENHANCER_BACKEND",
                 "CANDYECHO_ENHANCER_GGUF_REPO", "CANDYECHO_ENHANCER_GGUF_FILE"):
        monkeypatch.delenv(name, raising=False)

    managed = enhancer_spec(managed_runtime=True)
    assert managed.repos[0].repo_id == "tpnlabs/Qwen3.5-9B-heretic-v2-TPN-GGUF"
    # One quantisation, not the whole repo of them (nor its vision projector).
    assert managed.repos[0].allow_patterns == ("Qwen3.5-9B-heretic-v2-Q8_0.gguf",)
    assert managed.repos[1].source == SOURCE_LLAMA_CPP and managed.repos[1].subdir == "llama.cpp"

    # Elsewhere the user's own llama-server is used, so only the model is fetched.
    assert len(enhancer_spec(managed_runtime=False).repos) == 1

    # The stock model is two settings away.
    monkeypatch.setenv("CANDYECHO_ENHANCER_GGUF_REPO", "unsloth/Qwen3.5-9B-GGUF")
    monkeypatch.setenv("CANDYECHO_ENHANCER_GGUF_FILE", "Qwen3.5-9B-Q8_0.gguf")
    stock = enhancer_spec(managed_runtime=False).repos[0]
    assert (stock.repo_id, stock.subdir, stock.allow_patterns) == (
        "unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-GGUF", ("Qwen3.5-9B-Q8_0.gguf",))
    monkeypatch.delenv("CANDYECHO_ENHANCER_GGUF_REPO")
    monkeypatch.delenv("CANDYECHO_ENHANCER_GGUF_FILE")

    monkeypatch.setenv("CANDYECHO_ENHANCER_BACKEND", "transformers")
    monkeypatch.setenv("CANDYECHO_ENHANCER_MODEL", "acme/tiny-llm")
    old = enhancer_spec()
    assert old.repos[0].repo_id == "acme/tiny-llm" and old.repos[0].subdir == "tiny-llm"


def test_a_download_for_an_earlier_model_does_not_vouch_for_a_new_one(tmp_path):
    """Switching the enhancer must not leave the old weights looking ready."""
    old_spec = ModelSpec("enhancer", "LLM", "", (RepoSpec("acme/old-llm", "old-llm"),), 1.0, "x", "u")
    hub = FakeHub({"acme/old-llm": {"model.safetensors": b"o" * 500},
                   "acme/new-gguf": {"q8.gguf": b"8" * 300, "q4.gguf": b"4" * 100}})
    store = ModelStore(tmp_path, {"enhancer": old_spec}, hub=hub)
    store.start_download("enhancer"); store.wait("enhancer", timeout=10)
    assert store.is_ready("enhancer")

    new_spec = ModelSpec("enhancer", "LLM", "",
                         (RepoSpec("acme/new-gguf", "new-gguf", allow_patterns=("q8.gguf",)),), 1.0, "x", "u")
    store = ModelStore(tmp_path, {"enhancer": new_spec}, hub=hub)
    status = store.status("enhancer")
    # Not ready, and the old 500 bytes are not counted as progress on the new model.
    assert status["state"] == "missing" and status["bytes_done"] == 0
    store.start_download("enhancer"); store.wait("enhancer", timeout=10)
    assert store.is_ready("enhancer")

    # Another quantisation of the same repo is a different model too.
    other = ModelSpec("enhancer", "LLM", "",
                      (RepoSpec("acme/new-gguf", "new-gguf", allow_patterns=("q4.gguf",)),), 1.0, "x", "u")
    assert not ModelStore(tmp_path, {"enhancer": other}, hub=hub).is_ready("enhancer")


def test_a_hub_that_unpacks_records_the_unpacked_files(tmp_path):
    from longecho.model_store import SOURCE_LLAMA_CPP

    class UnpackingHub:
        def list_files(self, repo_id):
            return "b123", {"server.zip": 1000}

        def download(self, repo_id, revision, local_dir, allow_patterns):
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "llama-server.exe").write_bytes(b"x" * 40)
            (local_dir / "cuda.dll").write_bytes(b"x" * 60)
            return {"llama-server.exe": 40, "cuda.dll": 60}

    spec = ModelSpec("enhancer", "LLM", "", (
        RepoSpec("acme/new-gguf", "new-gguf", allow_patterns=("q8.gguf",)),
        RepoSpec("ggml-org/llama.cpp", "llama.cpp", source=SOURCE_LLAMA_CPP),
    ), 1.0, "x", "u")
    store = ModelStore(tmp_path, {"enhancer": spec}, hub=FakeHub({"acme/new-gguf": {"q8.gguf": b"8" * 300}}),
                       hubs={SOURCE_LLAMA_CPP: UnpackingHub()})
    store.start_download("enhancer"); store.wait("enhancer", timeout=10)
    assert store.status("enhancer")["state"] == "ready", store.status("enhancer")
    files = json.loads((tmp_path / "enhancer" / MANIFEST_NAME).read_text())["files"]
    assert files == {"new-gguf/q8.gguf": 300, "llama.cpp/llama-server.exe": 40, "llama.cpp/cuda.dll": 60}
    # Deleting the server binary is noticed like any other missing file.
    (tmp_path / "enhancer" / "llama.cpp" / "llama-server.exe").unlink()
    assert not store.is_ready("enhancer")


def test_a_model_can_live_outside_the_default_folder(tmp_path):
    """e.g. Drama 3's weights moved onto WSL's own disk."""
    elsewhere = tmp_path / "other-disk" / "toy"
    store = ModelStore(tmp_path / "models", specs=SPECS, hub=FakeHub(REPOS), dirs={"toy": elsewhere})
    assert store.model_dir("toy") == elsewhere
    store.start_download("toy")
    store.wait("toy", timeout=10)
    assert store.is_ready("toy")
    assert (elsewhere / "main" / "config.json").exists()
    assert not (tmp_path / "models" / "toy").exists()
