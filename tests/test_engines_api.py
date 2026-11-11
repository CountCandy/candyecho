"""HTTP endpoints for engines, downloads, transcripts, the enhancer, and
generation routed through a remote engine."""
import io
import json
import wave
from unittest.mock import Mock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from longecho.engine_servers import EngineServer, EngineServerConfig
from longecho.engines import Engine
from longecho.model_store import ModelSpec, ModelStore, RepoSpec
from longecho.remote_engines import REMOTE_ID_BASE, EngineError, RemoteAudioGenerator
from longecho.voice_transcripts import TranscriptStore


class FakeHub:
    def list_files(self, repo_id):
        return "sha", {"config.json": 2}

    def download(self, repo_id, revision, local_dir, allow_patterns):
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_bytes(b"{}")


class FakeClient:
    key = "drama3"
    label = "Fish Audio Drama 3"
    sample_rate = 24000
    PARAMS = ("temperature", "top_p")

    def __init__(self):
        self.calls = []
        self.error = None

    def synthesize(self, text, reference, seed, params=None):
        self.calls.append({"text": text, "reference": reference, "seed": seed, "params": params})
        if self.error:
            raise EngineError(self.error)
        return torch.zeros(1, 1, 2400), self.sample_rate


class FakeProcess:
    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


def write_wav(path, seconds=1.0, rate=8000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))


def sse_events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """The app with Echo mocked out and one fake remote engine wired in."""
    monkeypatch.setenv("CANDYECHO_JOBS_DIR", str(tmp_path / "jobs"))
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    write_wav(voice_dir / "mina.wav")

    voice_manager = Mock()
    voice_manager.get_voice_names.return_value = ["mina"]
    voice_manager.get_voice_info.return_value = [{"name": "mina", "duration_seconds": 1.0}]
    voice_manager.get_voice.return_value = (Mock(), Mock())
    voice_manager.get_voice_path.side_effect = (
        lambda name: voice_dir / f"{name}.wav" if name == "mina" else None
    )

    specs = {
        key: ModelSpec(key, label, "", (RepoSpec(f"org/{key}", key),), 1.0, "Test licence",
                       f"https://example.invalid/{key}")
        for key, label in (("drama3", "Fish Audio Drama 3"), ("breeze", "Breeze TTS 2"),
                           ("enhancer", "Tag enhancer LLM"))
    }
    store = ModelStore(tmp_path / "models", specs=specs, hub=FakeHub())

    state = {"healthy": False, "launched": 0}

    def popen(cmd, **kwargs):
        state["launched"] += 1
        state["healthy"] = True  # the fake server is up as soon as it is launched
        return FakeProcess()

    client = FakeClient()
    server = EngineServer(
        EngineServerConfig("drama3", "Fish Audio Drama 3", "http://127.0.0.1:9", "/v1/health",
                           start_cmd=["run"], startup_timeout=5),
        log_dir=tmp_path / "logs", popen=popen, probe=lambda url: state["healthy"],
        poll_interval=0.01,
    )
    engine = Engine(
        key="drama3", label="Fish Audio Drama 3", blurb="", sample_rate=24000,
        model_key="drama3", server=server,
        generator=RemoteAudioGenerator(client, id_base=REMOTE_ID_BASE),
        params=FakeClient.PARAMS, supports_tags=True,
    )

    with patch("longecho.main.load_model_from_hf", return_value=Mock()), \
         patch("longecho.main.load_fish_ae_from_hf", return_value=Mock()), \
         patch("longecho.main.load_pca_state_from_hf", return_value=Mock()), \
         patch("longecho.main.VoiceManager", Mock(return_value=voice_manager)), \
         patch("longecho.main.AudioGenerator", Mock(return_value=Mock())), \
         patch("longecho.main.ModelStore", Mock(return_value=store)), \
         patch("longecho.main.build_engines", Mock(return_value={"drama3": engine})), \
         patch("longecho.main.TranscriptStore", Mock(return_value=TranscriptStore(voice_dir))), \
         patch("longecho.main._transcribe_voice", Mock(return_value="auto transcript")) as asr:
        from longecho.main import app

        with TestClient(app) as http:
            yield {
                "http": http, "app": app, "store": store, "client": client, "state": state,
                "voice_dir": voice_dir, "asr": asr, "engine": engine,
            }


def by_key(payload):
    return {e["key"]: e for e in payload["engines"]}


def test_engines_lists_echo_as_ready_and_remote_as_needing_download(env):
    data = env["http"].get("/engines").json()
    engines = by_key(data)
    assert engines["echo"]["ready"] is True and engines["echo"]["model"] is None
    drama = engines["drama3"]
    assert drama["model"]["state"] == "missing"
    assert drama["server"]["state"] == "stopped"
    assert drama["ready"] is False and drama["can_generate"] is False
    assert drama["needs_transcript"] is True and drama["supports_tags"] is True
    assert data["enhancer"]["backend"] == "local"


def test_download_then_start_makes_the_engine_ready_without_a_restart(env):
    http, store, state = env["http"], env["store"], env["state"]

    # Starting before the weights exist is refused.
    assert http.post("/engines/drama3/start").status_code == 409

    resp = http.post("/models/drama3/download")
    assert resp.status_code == 200 and resp.json()["started"] is True
    store.wait("drama3", timeout=10)
    assert by_key(http.get("/engines").json())["drama3"]["model"]["state"] == "ready"

    # A second click does not start a second download.
    assert http.post("/models/drama3/download").json()["started"] is False

    assert http.post("/engines/drama3/start").status_code == 200
    drama = by_key(http.get("/engines").json())["drama3"]
    assert drama["ready"] is True and drama["server"]["managed"] is True
    assert state["launched"] == 1

    assert http.post("/engines/drama3/stop").status_code == 200


def test_unknown_engine_and_model_are_404(env):
    assert env["http"].post("/engines/nope/start").status_code == 404
    assert env["http"].post("/models/nope/download").status_code == 404


def test_transcript_can_be_read_written_and_cleared(env):
    http = env["http"]
    assert http.get("/voices/mina/transcript").json()["transcript"] is None
    resp = http.put("/voices/mina/transcript", json={"text": "  Hello   there. "})
    assert resp.json()["transcript"] == "Hello there."
    assert (env["voice_dir"] / "mina.transcript.txt").exists()
    assert http.get("/voices/mina/transcript").json()["transcript"] == "Hello there."
    assert http.put("/voices/mina/transcript", json={"text": ""}).json()["transcript"] is None
    assert http.get("/voices/ghost/transcript").status_code == 404


def test_transcribe_endpoint_saves_the_result(env):
    resp = env["http"].post("/voices/mina/transcribe")
    assert resp.status_code == 200 and resp.json()["transcript"] == "auto transcript"
    assert env["http"].get("/voices/mina/transcript").json()["transcript"] == "auto transcript"


def test_generate_refuses_an_engine_that_is_not_downloaded(env):
    resp = env["http"].post("/generate", json={"text": "Hello.", "voice": "mina", "engine": "drama3"})
    events = sse_events(resp)
    assert events[-1]["type"] == "error"
    assert "not downloaded" in events[-1]["message"]
    assert env["client"].calls == []


def test_generate_with_remote_engine_transcribes_starts_and_streams(env):
    http, store, client = env["http"], env["store"], env["client"]
    http.post("/models/drama3/download")
    store.wait("drama3", timeout=10)

    resp = http.post("/generate", json={
        "text": "[calm] Hello there. This is a test.",
        "voice": "mina", "engine": "drama3", "seed": 11,
        "temperature": 0.5, "steps": 12,
    })
    events = sse_events(resp)
    types = [e["type"] for e in events]

    # The reference was transcribed and the server started on demand, with the
    # user told about both before the first chunk.
    assert types[:2] == ["status", "status"]
    assert "Transcribing" in events[0]["message"] and "Starting" in events[1]["message"]
    assert env["state"]["launched"] == 1
    assert "start" in types and "chunk" in types and types[-1] == "complete"

    start = next(e for e in events if e["type"] == "start")
    assert start["generation_id"] > REMOTE_ID_BASE

    call = client.calls[0]
    assert call["text"].startswith("[calm] Hello there.")
    assert call["reference"].transcript == "auto transcript"
    assert call["reference"].wav_bytes[:4] == b"RIFF"
    assert call["seed"] == 11
    # Only this engine's own parameters reach it; Echo's sampler knobs do not.
    assert call["params"] == {"temperature": 0.5}

    # The chunk on disk is at the engine's sample rate, and the job remembers it.
    job_id = next(e for e in events if e["type"] == "complete")["job_id"]
    job = http.get(f"/jobs/{job_id}").json()
    assert job["settings"]["engine"] == "drama3" and job["settings"]["sample_rate"] == 24000
    chunk = next(e for e in events if e["type"] == "chunk")
    import base64
    with wave.open(io.BytesIO(base64.b64decode(chunk["data"])), "rb") as w:
        assert w.getframerate() == 24000
    assert job["chunks"][0]["duration"] == pytest.approx(0.1)

    # Echo was never asked to generate.
    assert env["app"].state.audio_generator.generate_long_audio.call_count == 0


def test_saved_transcript_is_used_instead_of_transcribing_again(env):
    http, store = env["http"], env["store"]
    http.post("/models/drama3/download")
    store.wait("drama3", timeout=10)
    http.put("/voices/mina/transcript", json={"text": "My own words."})
    resp = http.post("/generate", json={"text": "Hi.", "voice": "mina", "engine": "drama3"})
    assert sse_events(resp)[-1]["type"] == "complete"
    assert env["asr"].call_count == 0
    assert env["client"].calls[0]["reference"].transcript == "My own words."


def test_engine_failure_mid_generation_is_reported(env):
    http, store = env["http"], env["store"]
    http.post("/models/drama3/download")
    store.wait("drama3", timeout=10)
    env["client"].error = "CUDA out of memory"
    resp = http.post("/generate", json={"text": "Hi.", "voice": "mina", "engine": "drama3"})
    last = sse_events(resp)[-1]
    assert last["type"] == "error" and "CUDA out of memory" in last["message"]


def test_unknown_engine_in_generate_is_an_error_event(env):
    resp = env["http"].post("/generate", json={"text": "Hi.", "voice": "mina", "engine": "zzz"})
    assert sse_events(resp)[-1]["type"] == "error"


def test_subtitles_drop_drama_markup(env):
    http, store = env["http"], env["store"]
    http.post("/models/drama3/download")
    store.wait("drama3", timeout=10)
    resp = http.post("/generate", json={
        "text": "[smug] Oh, <emphasis>darling</emphasis>.", "voice": "mina", "engine": "drama3",
    })
    job_id = sse_events(resp)[-1]["job_id"]
    srt = http.get(f"/jobs/{job_id}/subtitles").text
    assert "darling" in srt and "[smug]" not in srt and "<emphasis>" not in srt


def test_stop_routes_remote_ids_to_the_remote_generator(env):
    http, app = env["http"], env["app"]
    http.post(f"/stop?generation_id={REMOTE_ID_BASE + 5}")
    assert app.state.audio_generator.request_stop.call_count == 0
    http.post("/stop?generation_id=3")
    app.state.audio_generator.request_stop.assert_called_once_with(3)


def test_openai_endpoint_can_select_the_remote_engine(env):
    http, store = env["http"], env["store"]
    # Not downloaded yet: a clear 503 rather than a hang.
    resp = http.post("/v1/audio/speech", json={"input": "Hi.", "voice": "mina", "model": "drama3"})
    assert resp.status_code == 503 and "not downloaded" in resp.json()["detail"]

    http.post("/models/drama3/download")
    store.wait("drama3", timeout=10)
    resp = http.post("/v1/audio/speech", json={"input": "Hi there.", "voice": "mina", "model": "drama3"})
    assert resp.status_code == 200
    with wave.open(io.BytesIO(resp.content), "rb") as w:
        assert w.getframerate() == 24000
    assert len(env["client"].calls) == 1


def test_enhance_requires_the_model_then_uses_it(env):
    http, store, app = env["http"], env["store"], env["app"]
    app.state.enhancer = None
    resp = http.post("/enhance-text", json={"text": "Hello there."})
    assert resp.status_code == 409 and "not downloaded" in resp.json()["detail"]

    class FakeEnhancer:
        def enhance(self, text):
            from longecho.tag_enhancer import EnhanceResult
            return EnhanceResult(text="[warm] " + text, pieces=1, enhanced=1, tags_added=1)

    app.state.enhancer = FakeEnhancer()
    try:
        data = http.post("/enhance-text", json={"text": "Hello there."}).json()
        assert data["text"] == "[warm] Hello there." and data["tags_added"] == 1
    finally:
        app.state.enhancer = None
