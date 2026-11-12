"""The llama.cpp server build: choosing, fetching, verifying, launching."""
import hashlib
import io
import json
import zipfile

import pytest

from longecho.llama_runtime import (
    LlamaCppReleaseHub,
    choose_windows_assets,
    find_gguf,
    find_llama_server,
    llama_server_config,
    managed_runtime_supported,
)


def asset(name, data=b""):
    return {
        "name": name, "size": len(data),
        "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
        "browser_download_url": "https://example.invalid/" + name,
        "_data": data,
    }


def zip_bytes(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


class FakeGitHub:
    """Stands in for urlopen: the release API and the asset downloads."""

    def __init__(self, assets, tag="b12283", corrupt=()):
        self.assets, self.tag, self.corrupt, self.urls = assets, tag, set(corrupt), []

    def __call__(self, request, timeout=None):
        url = request.full_url
        self.urls.append(url)
        if "api.github.com" in url:
            public = [{k: v for k, v in a.items() if k != "_data"} for a in self.assets]
            return io.BytesIO(json.dumps({"tag_name": self.tag, "assets": public}).encode())
        name = url.rsplit("/", 1)[-1]
        data = next(a["_data"] for a in self.assets if a["name"] == name)
        return io.BytesIO(data[:10] if name in self.corrupt else data)


def release_assets():
    server = zip_bytes({"llama-server.exe": b"MZ" + b"s" * 50, "ggml-cuda.dll": b"d" * 30})
    runtime = zip_bytes({"cudart64_13.dll": b"r" * 20})
    return [
        asset("llama-b12283-bin-win-cpu-x64.zip", b"cpu"),
        asset("llama-b12283-bin-win-cuda-12.4-x64.zip", b"old"),
        asset("cudart-llama-bin-win-cuda-12.4-x64.zip", b"oldrt"),
        asset("llama-b12283-bin-win-cuda-13.5-x64.zip", server),
        asset("cudart-llama-bin-win-cuda-13.5-x64.zip", runtime),
        asset("llama-b12283-bin-ubuntu-x64.tar.gz", b"linux"),
    ]


def test_cuda_13_build_is_chosen_whatever_its_minor_version():
    names = [a["name"] for a in choose_windows_assets(release_assets())]
    assert names == ["llama-b12283-bin-win-cuda-13.5-x64.zip", "cudart-llama-bin-win-cuda-13.5-x64.zip"]
    names = [a["name"] for a in choose_windows_assets(release_assets(), cuda_major=12)]
    assert names == ["llama-b12283-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]
    with pytest.raises(RuntimeError, match="no Windows x64"):
        choose_windows_assets([asset("llama-b1-bin-macos-arm64.tar.gz")])


def test_download_verifies_unpacks_and_reports_the_files(tmp_path):
    github = FakeGitHub(release_assets())
    hub = LlamaCppReleaseHub(opener=github)
    tag, files = hub.list_files("ggml-org/llama.cpp")
    assert tag == "b12283" and set(files) == {
        "llama-b12283-bin-win-cuda-13.5-x64.zip", "cudart-llama-bin-win-cuda-13.5-x64.zip"}
    assert github.urls[0].endswith("/releases/latest")

    target = tmp_path / "llama.cpp"
    target.mkdir()
    (target / "stale.dll").write_bytes(b"from an older release")
    installed = hub.download("ggml-org/llama.cpp", tag, target, None)
    assert installed["llama-server.exe"] == 52 and installed["cudart64_13.dll"] == 20
    # The archives and the previous build are gone; only the unpacked build remains.
    assert not list(target.glob("*.zip")) and not (target / "stale.dll").exists()
    assert find_llama_server(target, windows=True) == target / "llama-server.exe"


def test_a_truncated_download_is_rejected(tmp_path):
    github = FakeGitHub(release_assets(), corrupt={"cudart-llama-bin-win-cuda-13.5-x64.zip"})
    hub = LlamaCppReleaseHub(opener=github)
    with pytest.raises(RuntimeError, match="arrived damaged"):
        hub.download("ggml-org/llama.cpp", "b12283", tmp_path / "llama.cpp", None)


def test_a_pinned_release_is_requested_by_tag():
    github = FakeGitHub(release_assets(), tag="b11500")
    LlamaCppReleaseHub(release="b11500", opener=github).list_files("ggml-org/llama.cpp")
    assert github.urls[0].endswith("/releases/tags/b11500")


def test_zip_entries_cannot_escape_the_folder(tmp_path):
    evil = zip_bytes({"../outside.txt": b"x", "llama-server.exe": b"y"})
    assets = [asset("llama-b1-bin-win-cuda-13.5-x64.zip", evil),
              asset("cudart-llama-bin-win-cuda-13.5-x64.zip", zip_bytes({"a.dll": b"z"}))]
    hub = LlamaCppReleaseHub(opener=FakeGitHub(assets))
    with pytest.raises(RuntimeError, match="unsafe path"):
        hub.download("ggml-org/llama.cpp", "b1", tmp_path / "llama.cpp", None)
    assert not (tmp_path / "outside.txt").exists()


def test_server_is_found_in_a_subfolder_or_by_override(tmp_path, monkeypatch):
    monkeypatch.delenv("CANDYECHO_LLAMA_SERVER", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert find_llama_server(tmp_path, windows=True) is None
    nested = tmp_path / "build" / "bin"
    nested.mkdir(parents=True)
    (nested / "llama-server.exe").write_bytes(b"x")
    assert find_llama_server(tmp_path, windows=True) == nested / "llama-server.exe"

    mine = tmp_path / "my-llama-server"
    mine.write_bytes(b"x")
    monkeypatch.setenv("CANDYECHO_LLAMA_SERVER", str(mine))
    assert find_llama_server(tmp_path, windows=True) == mine


def test_model_file_is_found_and_projector_files_are_not_mistaken_for_it(tmp_path):
    (tmp_path / "mmproj-F16.gguf").write_bytes(b"x")
    assert find_gguf(tmp_path, "Qwen3.5-9B-Q8_0.gguf") is None
    (tmp_path / "Qwen3.5-9B-Q8_0.gguf").write_bytes(b"x")
    assert find_gguf(tmp_path, "Qwen3.5-9B-Q8_0.gguf").name == "Qwen3.5-9B-Q8_0.gguf"
    assert find_gguf(tmp_path).name == "Qwen3.5-9B-Q8_0.gguf"


def test_launch_recipe_turns_reasoning_off_and_stays_on_loopback(tmp_path):
    cfg = llama_server_config(tmp_path / "llama-server.exe", tmp_path / "m.gguf", url="http://127.0.0.1:8090")
    cmd = cfg.start_cmd
    assert cmd[0].endswith("llama-server.exe")
    assert cmd[cmd.index("--model") + 1].endswith("m.gguf")
    assert cmd[cmd.index("--reasoning") + 1] == "off"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1" and cmd[cmd.index("--port") + 1] == "8090"
    assert cmd[cmd.index("--gpu-layers") + 1] == "all"
    assert cfg.health_path == "/health" and cfg.stop_cmd is None


def test_only_windows_x64_gets_a_managed_build():
    assert managed_runtime_supported(windows=True, machine="AMD64")
    assert not managed_runtime_supported(windows=True, machine="ARM64")
    assert not managed_runtime_supported(windows=False, machine="x86_64")


def test_runtime_dlls_follow_a_server_nested_in_a_subfolder(tmp_path):
    assets = [asset("llama-b1-bin-win-cuda-13.5-x64.zip", zip_bytes({"build/bin/llama-server.exe": b"MZ"})),
              asset("cudart-llama-bin-win-cuda-13.5-x64.zip", zip_bytes({"cudart64_13.dll": b"r" * 20}))]
    hub = LlamaCppReleaseHub(opener=FakeGitHub(assets, tag="b1"))
    installed = hub.download("ggml-org/llama.cpp", "b1", tmp_path / "llama.cpp", None)
    assert installed["build/bin/cudart64_13.dll"] == 20
