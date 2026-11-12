"""HTTP endpoints for engines, downloads, transcripts, the enhancer, and
generation routed through a remote engine."""
import base64
import io
import json
import wave
from contextlib import contextmanager
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
    PARAMS = ("temperature", "top_p", "top_k")

    def __init__(self):
        self.calls = []
        self.error = None

    def synthesize(self, text, reference, seed, params=None):
        self.calls.append({"text": text, "reference": reference, "seed": seed, "params": params})
        if self.error:
            raise EngineError(self.error)
        # Level encodes how many clips were used, so trials are distinguishable.
        level = 0.1 * (len(reference.clips) - 1)
        return torch.full((1, 1, 2400), level), self.sample_rate


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


@contextmanager
def make_env(tmp_path, monkeypatch, preinstalled=False, autostart=False):
    """The app with Echo mocked out and one fake remote engine wired in."""
    monkeypatch.setenv("CANDYECHO_JOBS_DIR", str(tmp_path / "jobs"))
    settings_path = tmp_path / "candyecho-settings.json"
    monkeypatch.setenv("CANDYECHO_SETTINGS", str(settings_path))
    if autostart:
        settings_path.write_text(json.dumps({"autostart": ["drama3", "gone"]}))
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    names = ["mina", "mina2", "mina3"]
    for name in names:
        write_wav(voice_dir / f"{name}.wav")

    voice_manager = Mock()
    voice_manager.get_voice_names.return_value = names
    voice_manager.get_voice_info.return_value = [
        {"name": n, "duration_seconds": 1.0} for n in names
    ]
    voice_manager.get_voice.return_value = (Mock(), Mock())
    voice_manager.get_voice_path.side_effect = (
        lambda name: voice_dir / f"{name}.wav" if name in names else None
    )

    specs = {
        key: ModelSpec(key, label, "", (RepoSpec(f"org/{key}", key),), 1.0, "Test licence",
                       f"https://example.invalid/{key}")
        for key, label in (("drama3", "Fish Audio Drama 3"), ("enhancer", "Tag enhancer LLM"))
    }
    store = ModelStore(tmp_path / "models", specs=specs, hub=FakeHub())
    if preinstalled:
        store.start_download("drama3")
        store.wait("drama3", timeout=10)

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
        chunk_chars=(450, 700), max_extra_references=2,
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
                "settings_path": settings_path,
            }


@pytest.fixture
def env(tmp_path, monkeypatch):
    with make_env(tmp_path, monkeypatch) as e:
        yield e


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
    assert data["enhancer"]["backend"] == "llama"
    assert data["enhancer"]["loaded"] is False and data["enhancer"]["can_unload"] is True


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
        guidance = None

        def enhance(self, text, guidance=""):
            from longecho.tag_enhancer import EnhanceResult
            FakeEnhancer.guidance = guidance
            return EnhanceResult(text="[warm] " + text, pieces=1, enhanced=1, tags_added=1)

    app.state.enhancer = FakeEnhancer()
    try:
        data = http.post("/enhance-text", json={"text": "Hello there."}).json()
        assert data["text"] == "[warm] Hello there." and data["tags_added"] == 1
        assert FakeEnhancer.guidance == ""
        # The user's tagging notes are passed through, and over-long ones refused.
        http.post("/enhance-text", json={"text": "Hello there.", "guidance": "She is a smug vampire."})
        assert FakeEnhancer.guidance == "She is a smug vampire."
        assert http.post("/enhance-text", json={"text": "Hi.", "guidance": "x" * 5000}).status_code == 422
    finally:
        app.state.enhancer = None


# --- second round: fixes and additions from the first real run ------------

def ready(env):
    env["http"].post("/models/drama3/download")
    env["store"].wait("drama3", timeout=10)


def test_tags_survive_textbook_cleaning(env):
    """Regression: with cleaning on, <soft> reached the engine as words."""
    ready(env)
    text = ('[teasing, sarcastic] "Excuse me, did you get dressed in the dark, or is '
            "<soft>'dumpster chic'</soft> actually having a moment?\"")
    resp = env["http"].post("/generate", json={
        "text": text, "voice": "mina", "engine": "drama3",
        "cleaning": {"enabled": True, "preset": "textbook"},
    })
    assert sse_events(resp)[-1]["type"] == "complete"
    sent = env["client"].calls[0]["text"]
    assert "less than" not in sent and "greater than" not in sent
    assert sent.startswith("[teasing, sarcastic]")
    assert "<soft>'dumpster chic'</soft>" in sent


def test_short_dialogue_is_one_chunk_on_a_remote_engine(env):
    ready(env)
    text = "This is one sentence of a short piece of dialogue. " * 8  # ~410 chars: 2+ Echo chunks
    resp = env["http"].post("/generate", json={"text": text, "voice": "mina", "engine": "drama3"})
    assert sse_events(resp)[-1]["type"] == "complete"
    assert len(env["client"].calls) == 1


def test_best_of_n_uses_chunks_that_fit_the_verifier(env):
    ready(env)
    text = "This is one sentence of a short piece of dialogue. " * 8
    with patch("longecho.main.get_selector", return_value=None), \
         patch("longecho.main._reference_audio", return_value=None):
        resp = env["http"].post("/generate", json={
            "text": text, "voice": "mina", "engine": "drama3", "verify": True, "candidates": 1,
        })
    assert sse_events(resp)[-1]["type"] == "complete"
    assert len(env["client"].calls) == 2
    assert all(len(c["text"]) <= 330 for c in env["client"].calls)


def test_extra_voices_are_sent_as_additional_references(env):
    ready(env)
    env["http"].put("/voices/mina2/transcript", json={"text": "Second recording."})
    resp = env["http"].post("/generate", json={
        "text": "Hi.", "voice": "mina", "engine": "drama3", "top_k": 40,
        # mina itself, a duplicate and an unknown voice are ignored; the limit is 2.
        "extra_voices": ["mina2", "mina", "mina2", "ghost", "mina3"],
    })
    events = sse_events(resp)
    assert events[-1]["type"] == "complete"
    call = env["client"].calls[0]
    clips = call["reference"].clips
    assert [c.name for c in clips] == ["mina", "mina2", "mina3"]
    assert clips[1].transcript == "Second recording."
    assert clips[2].transcript == "auto transcript"   # transcribed on first use
    assert call["params"] == {"top_k": 40}
    job = env["http"].get(f"/jobs/{events[-1]['job_id']}").json()
    assert job["settings"]["extra_voices"] == ["mina2", "mina3"]


def test_saved_batches_carry_a_preview_of_their_first_words(env):
    ready(env)
    text = "[smug] One two three four five six seven eight nine ten eleven twelve."
    resp = env["http"].post("/generate", json={"text": text, "voice": "mina", "engine": "drama3"})
    job_id = sse_events(resp)[-1]["job_id"]
    summary = next(j for j in env["http"].get("/jobs").json()["jobs"] if j["id"] == job_id)
    assert summary["preview"] == "One two three four five six seven eight nine ten…"


def test_echo_can_be_unloaded_and_comes_back_on_demand(env):
    http, app = env["http"], env["app"]
    model = app.state.echo_model
    app.state.audio_generator.is_generating = False
    app.state.audio_generator.generate_long_audio.return_value = (
        1, (chunk for chunk in [torch.zeros(1, 1, 441)])
    )

    assert by_key(http.get("/engines").json())["echo"]["loaded"] is True
    assert http.post("/engines/echo/unload").json()["loaded"] is False
    model.to.assert_called_with("cpu")
    assert by_key(http.get("/engines").json())["echo"]["loaded"] is False

    # Generating with Echo brings it back first, and says so.
    resp = http.post("/generate", json={"text": "Hello.", "voice": "mina"})
    events = sse_events(resp)
    assert events[0]["type"] == "status" and "Loading Echo" in events[0]["message"]
    assert events[-1]["type"] == "complete"
    model.to.assert_called_with(app.state.echo_device)
    assert app.state.echo_loaded is True

    # Explicit load is a no-op when it is already loaded.
    calls = model.to.call_count
    assert http.post("/engines/echo/load").json()["loaded"] is True
    assert model.to.call_count == calls


def test_echo_is_not_unloaded_mid_generation(env):
    env["app"].state.audio_generator.is_generating = True
    assert env["http"].post("/engines/echo/unload").status_code == 409
    assert env["app"].state.echo_loaded is True


# --- third round: voice-first takes, chunk redo, audition, autostart -------

def test_verify_voice_is_passed_to_the_selector_and_saved(env):
    ready(env)
    seen = {}

    def fake_get_selector(app_state, threshold, prefer_voice=False):
        seen["prefer_voice"] = prefer_voice
        return None

    with patch("longecho.main.get_selector", side_effect=fake_get_selector), \
         patch("longecho.main._reference_audio", return_value=None):
        resp = env["http"].post("/generate", json={
            "text": "Hi.", "voice": "mina", "engine": "drama3",
            "verify": True, "verify_voice": True, "candidates": 1,
        })
    assert sse_events(resp)[-1]["type"] == "complete"
    assert seen["prefer_voice"] is True


def test_top_k_zero_is_rejected_before_it_reaches_the_server(env):
    resp = env["http"].post("/generate", json={
        "text": "Hi.", "voice": "mina", "engine": "drama3", "top_k": 0,
    })
    assert resp.status_code == 422


def make_batch(env, text="First sentence here. " * 40):
    resp = env["http"].post("/generate", json={
        "text": text, "voice": "mina", "engine": "drama3", "seed": 7,
        "temperature": 0.5, "extra_voices": ["mina2"],
    })
    events = sse_events(resp)
    assert events[-1]["type"] == "complete"
    return events[-1]["job_id"]


def test_redo_regenerates_one_chunk_in_place(env):
    ready(env)
    http, client = env["http"], env["client"]
    job_id = make_batch(env)
    job = http.get(f"/jobs/{job_id}").json()
    assert job["total"] >= 2
    calls_before = len(client.calls)
    original = http.get(f"/jobs/{job_id}/chunks/1/audio").content

    resp = http.post(f"/jobs/{job_id}/chunks/1/redo", json={"seed": 4242})
    assert resp.status_code == 200
    body = resp.json()
    assert body["index"] == 1 and body["seed"] == 4242

    # Exactly one request, for chunk 2's text, with the batch's own settings.
    assert len(client.calls) == calls_before + 1
    call = client.calls[-1]
    assert call["text"] == job["chunks"][1]["text"]
    assert call["seed"] == 4242 + 1000          # per-chunk seed spacing
    assert call["params"] == {"temperature": 0.5}
    assert [c.name for c in call["reference"].clips] == ["mina", "mina2"]

    after = http.get(f"/jobs/{job_id}").json()
    assert after["chunks"][1]["seed"] == 4242
    assert after["chunks"][0]["seed"] == job["chunks"][0]["seed"]   # others untouched
    assert after["complete"] is True
    assert http.get(f"/jobs/{job_id}/chunks/1/audio").content[:4] == b"RIFF"
    assert len(original) > 44


def test_redo_without_a_seed_picks_a_new_one(env):
    ready(env)
    job_id = make_batch(env)
    first = env["http"].post(f"/jobs/{job_id}/chunks/0/redo").json()["seed"]
    second = env["http"].post(f"/jobs/{job_id}/chunks/0/redo", json={}).json()["seed"]
    assert first != 7 and second != first


def test_redo_rejects_bad_targets(env):
    ready(env)
    job_id = make_batch(env)
    http = env["http"]
    assert http.post(f"/jobs/{job_id}/chunks/99/redo").status_code == 404
    assert http.post("/jobs/20260101-000000-abcdef/chunks/0/redo").status_code == 404
    assert http.get(f"/jobs/{job_id}/chunks/99/audio").status_code == 404


def test_redo_reports_an_engine_failure(env):
    ready(env)
    job_id = make_batch(env)
    env["client"].error = "CUDA out of memory"
    resp = env["http"].post(f"/jobs/{job_id}/chunks/0/redo")
    assert resp.status_code == 502 and "CUDA out of memory" in resp.json()["detail"]


def test_redo_on_an_echo_batch_continues_from_the_previous_chunk(env):
    http, app = env["http"], env["app"]
    generator = app.state.audio_generator
    generator.is_generating = False

    def fake_generate(texts, latent, mask, **kwargs):
        start = kwargs.get("start_index", 0)
        fake_generate.calls.append(kwargs)
        return 1, (torch.zeros(1, 1, 4410) for _ in texts[start:])

    fake_generate.calls = []
    generator.generate_long_audio.side_effect = fake_generate
    resp = http.post("/generate", json={"text": "Sentence number one is here. " * 20, "voice": "mina"})
    job_id = sse_events(resp)[-1]["job_id"]

    resp = http.post(f"/jobs/{job_id}/chunks/1/redo", json={"seed": 99})
    assert resp.status_code == 200
    kwargs = fake_generate.calls[-1]
    assert kwargs["start_index"] == 1 and kwargs["rng_seed"] == 99
    assert kwargs["initial_continuation_audio"] is not None
    assert kwargs["initial_previous_text"].startswith("Sentence number one")


class FakeSimilarity:
    """Embedding = the clip's level; closeness = 1 - distance."""

    def embed(self, audio, sample_rate):
        return float(audio.flatten()[0])

    def score(self, a, b):
        return 1.0 - abs(a - b)


def test_audition_ranks_clips_alone_and_together(env):
    ready(env)
    with patch("longecho.main.get_speaker_similarity", return_value=FakeSimilarity()):
        resp = env["http"].post("/audition", json={
            "engine": "drama3", "voices": ["mina", "mina2", "mina3"], "takes": 2, "seed": 5,
        })
    events = sse_events(resp)
    assert events[-1]["type"] == "complete"
    trials = [e for e in events if e["type"] == "trial"]
    labels = [t["label"] for t in trials]
    assert labels == [
        "mina alone", "mina2 alone", "mina3 alone", "All together",
        "All except mina", "All except mina2", "All except mina3",
    ]
    together = next(t for t in trials if t["label"] == "All together")
    assert together["voices"] == ["mina", "mina2", "mina3"] and len(together["takes"]) == 2
    assert base64.b64decode(together["audio"])[:4] == b"RIFF"

    # Every trial used the same seeds and the same test line.
    start = next(e for e in events if e["type"] == "start")
    assert start["trials"] == 7 and start["seed"] == 5
    used = env["client"].calls[-14:]
    assert {c["seed"] for c in used} == {5, 6}
    assert len({c["text"] for c in used}) == 1

    # With the fake scorer, fewer clips score higher, so singles lead the ranking.
    ranking = events[-1]["ranking"]
    assert ranking[0]["label"].endswith("alone")
    assert ranking[-1]["label"] == "All together"
    assert ranking == sorted(ranking, key=lambda r: r["score"], reverse=True)


def test_audition_with_two_clips_skips_leave_one_out(env):
    ready(env)
    with patch("longecho.main.get_speaker_similarity", return_value=FakeSimilarity()):
        resp = env["http"].post("/audition", json={"voices": ["mina", "mina2"], "takes": 1})
    labels = [e["label"] for e in sse_events(resp) if e["type"] == "trial"]
    assert labels == ["mina alone", "mina2 alone", "All together"]


def test_audition_rejects_unknown_voices_and_engines(env):
    http = env["http"]
    assert http.post("/audition", json={"voices": ["ghost"]}).status_code == 404
    assert http.post("/audition", json={"engine": "echo", "voices": ["mina"]}).status_code == 400
    assert http.post("/audition", json={"voices": []}).status_code == 422


def test_audition_reports_a_missing_download_as_an_error_event(env):
    resp = env["http"].post("/audition", json={"voices": ["mina", "mina2"]})
    last = sse_events(resp)[-1]
    assert last["type"] == "error" and "not downloaded" in last["message"]


def test_autostart_setting_is_saved_and_reported(env):
    http = env["http"]
    assert by_key(http.get("/engines").json())["drama3"]["autostart"] is False
    assert http.put("/engines/drama3/autostart", json={"enabled": True}).json()["autostart"] is True
    assert by_key(http.get("/engines").json())["drama3"]["autostart"] is True
    assert json.loads(env["settings_path"].read_text())["autostart"] == ["drama3"]
    http.put("/engines/drama3/autostart", json={"enabled": False})
    assert json.loads(env["settings_path"].read_text())["autostart"] == []
    assert http.put("/engines/nope/autostart", json={"enabled": True}).status_code == 404


def test_autostart_launches_a_downloaded_engine_at_startup(tmp_path, monkeypatch):
    with make_env(tmp_path, monkeypatch, preinstalled=True, autostart=True) as e:
        assert e["state"]["launched"] == 1
        drama = by_key(e["http"].get("/engines").json())["drama3"]
        assert drama["ready"] is True and drama["autostart"] is True
        assert drama["server"]["last_load_seconds"] is not None or drama["server"]["managed"]


def test_autostart_does_nothing_until_the_engine_is_downloaded(tmp_path, monkeypatch):
    with make_env(tmp_path, monkeypatch, preinstalled=False, autostart=True) as e:
        assert e["state"]["launched"] == 0


def test_generate_reports_how_closely_the_output_matches_the_sample(env, caplog):
    ready(env)
    with patch("longecho.main.get_speaker_similarity", return_value=FakeSimilarity()), \
            caplog.at_level("INFO", logger="longecho.main"):
        resp = env["http"].post("/generate", json={
            "text": "Hello there. " * 80, "voice": "mina", "engine": "drama3",
        })
    events = sse_events(resp)
    done = events[-1]
    assert done["type"] == "complete"
    match = done["voice_match"]
    assert match["chunks"] >= 2 and 0.0 <= match["lowest"] <= match["mean"] <= 1.0
    assert all("voice_match" in e for e in events if e["type"] == "progress" and "chunk" in e.get("message", ""))
    # Kept with the batch, and written to the terminal for every chunk.
    job = env["app"].state.job_manager.load(done["job_id"])
    assert all(c.voice_match is not None for c in job.chunks)
    assert "Chunk 1/" in caplog.text and "voice match to 'mina'" in caplog.text


def test_generate_without_the_speaker_model_still_completes(env):
    ready(env)
    resp = env["http"].post("/generate", json={"text": "Hello there.", "voice": "mina", "engine": "drama3"})
    done = sse_events(resp)[-1]
    assert done["type"] == "complete" and "voice_match" not in done


def test_selection_line_says_which_take_was_kept_and_why():
    from longecho.main import _describe_selection, _voice_match_summary

    verification = {
        "winner": 1, "acceptable": True, "chosen_for_voice": True,
        "candidates": [
            {"index": 0, "wer": 0.0, "speaker_similarity": 0.81},
            {"index": 1, "wer": 0.02, "speaker_similarity": 0.9},
        ],
    }
    line = _describe_selection(0, 2, verification, 0.9)
    assert line == ("Chunk 1/2 · kept take 2 of 2 (picked for voice) · word error 2.0% · "
                    "voice match 90.0% (takes 81.0-90.0%)")
    assert _describe_selection(1, 2, None, 0.875) == "Chunk 2/2 · voice match 87.5%"

    # Long chunks count for more than short ones.
    summary = _voice_match_summary([(0, 0.9, 30.0), (1, 0.6, 10.0)])
    assert summary == {"mean": 0.825, "lowest": 0.6, "lowest_chunk": 1, "chunks": 2}
    assert _voice_match_summary([]) is None


def test_enhancer_unload_stops_its_server_and_status_follows_the_health_check(env):
    http, app = env["http"], env["app"]

    class Server:
        healthy, stops = True, 0

        def is_healthy(self):
            return self.healthy

        def is_running(self):
            return self.healthy

        def stop(self):
            self.stops += 1
            self.healthy = False

    server = Server()
    app.state.enhancer, app.state.enhancer_server = None, server
    try:
        assert http.get("/engines").json()["enhancer"]["loaded"] is True
        assert http.post("/enhancer/unload").json() == {"status": "unloaded"}
        assert server.stops == 1
        assert http.get("/engines").json()["enhancer"]["loaded"] is False
    finally:
        app.state.enhancer_server = None
