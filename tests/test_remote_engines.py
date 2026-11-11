"""Remote engine clients and the chunk generator that drives them."""
import base64
import io
import json
import wave

import pytest
import torch

from longecho.remote_engines import (
    REMOTE_ID_BASE,
    BreezeClient,
    DramaClient,
    EngineError,
    RemoteAudioGenerator,
    pcm16_to_tensor,
    wav_to_tensor,
)
from longecho.voice_transcripts import VoiceReference

REF = VoiceReference(name="mina", wav_bytes=b"RIFF" + b"\x00" * 600, transcript="Hello there.")


def wav_bytes(samples, rate=44100, channels=1):
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(int(s).to_bytes(2, "little", signed=True) for s in samples))
    return out.getvalue()


def test_pcm16_to_tensor_shape_and_scale():
    data = (16384).to_bytes(2, "little", signed=True) + (-32768).to_bytes(2, "little", signed=True)
    audio = pcm16_to_tensor(data)
    assert audio.shape == (1, 1, 2)
    assert audio[0, 0].tolist() == pytest.approx([0.5, -1.0])


def test_pcm16_ignores_a_dangling_byte_and_rejects_empty():
    assert pcm16_to_tensor(b"\x00\x00\x01").shape == (1, 1, 1)
    with pytest.raises(EngineError):
        pcm16_to_tensor(b"")


def test_wav_to_tensor_reads_rate_and_channels():
    audio, rate = wav_to_tensor(wav_bytes([0, 100, 200, 300], rate=24000, channels=2))
    assert rate == 24000
    assert audio.shape == (1, 2, 2)


def test_drama_client_sends_reference_and_decodes_wav():
    seen = {}

    def post(url, body, headers, timeout):
        seen.update(url=url, body=json.loads(body), headers=headers)
        return wav_bytes([0] * 441), {}

    client = DramaClient("http://127.0.0.1:8080/", post=post)
    audio, rate = client.synthesize("[calm] Hi.", REF, seed=7, params={"temperature": 0.5, "steps": 40})
    assert seen["url"] == "http://127.0.0.1:8080/v1/tts"
    assert seen["headers"]["Content-Type"] == "application/json"
    body = seen["body"]
    assert body["text"] == "[calm] Hi." and body["seed"] == 7 and body["format"] == "wav"
    assert body["temperature"] == 0.5 and "steps" not in body and "top_p" not in body
    ref = body["references"][0]
    assert ref["text"] == "Hello there."
    assert base64.b64decode(ref["audio"]) == REF.wav_bytes
    assert rate == 44100 and audio.shape == (1, 1, 441)


def test_breeze_client_posts_multipart_and_reads_pcm():
    seen = {}

    def post(url, body, headers, timeout):
        seen.update(url=url, body=body, headers=headers)
        return b"\x00\x00" * 240, {"x-sample-rate": "24000"}

    client = BreezeClient("http://127.0.0.1:7860", post=post)
    audio, rate = client.synthesize("(sigh) Hi.", REF, seed=3)
    assert seen["url"] == "http://127.0.0.1:7860/v1/audio/speech"
    assert seen["headers"]["Content-Type"].startswith("multipart/form-data; boundary=")
    body = seen["body"]
    assert b'name="text"\r\n\r\n(sigh) Hi.' in body
    assert b'name="ref_text"\r\n\r\nHello there.' in body
    assert b'name="seed"\r\n\r\n3' in body
    assert b'name="ref_audio"; filename="reference.wav"' in body and REF.wav_bytes in body
    # No instruction: plain clone, CFG left to the server.
    assert b'name="instruction"' not in body and b'name="cfg_scale"' not in body
    assert rate == 24000 and audio.shape == (1, 1, 240)


def test_breeze_instruction_turns_on_direction_with_cfg_4():
    seen = {}

    def post(url, body, headers, timeout):
        seen["body"] = body
        return b"\x00\x00" * 4, {}

    BreezeClient("http://x", post=post).synthesize(
        "Hi.", REF, seed=1, params={"instruction": " Speak slowly. "}
    )
    assert b'name="instruction"\r\n\r\nSpeak slowly.' in seen["body"]
    assert b'name="cfg_scale"\r\n\r\n4.0' in seen["body"]


class FakeClient:
    key = "fake"
    label = "Fake"
    sample_rate = 24000
    PARAMS = ()

    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def synthesize(self, text, reference, seed, params=None):
        self.calls.append((text, seed, params))
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise EngineError("engine exploded")
        return torch.full((1, 1, 2400), float(len(self.calls))), self.sample_rate


def test_generator_yields_one_tensor_per_chunk_in_order():
    client = FakeClient()
    gen = RemoteAudioGenerator(client)
    gen_id, stream = gen.generate_long_audio(["a", "b", "c"], REF, rng_seed=10)
    assert gen.owns(gen_id) and gen_id > REMOTE_ID_BASE
    chunks = list(stream)
    assert [c.shape for c in chunks] == [(1, 1, 2400)] * 3
    assert [text for text, _s, _p in client.calls] == ["a", "b", "c"]
    # Seeds are spaced per chunk like Echo's.
    assert [seed for _t, seed, _p in client.calls] == [10, 1010, 2010]
    assert not gen.is_generating


def test_generator_resumes_from_start_index():
    client = FakeClient()
    _id, stream = RemoteAudioGenerator(client).generate_long_audio(["a", "b", "c"], REF, start_index=2)
    assert len(list(stream)) == 1
    assert client.calls[0][0] == "c"


def test_generator_applies_chunk_preparation():
    client = FakeClient()
    gen = RemoteAudioGenerator(client, prepare_chunks=lambda chunks: [c.upper() for c in chunks])
    list(gen.generate_long_audio(["a", "b"], REF)[1])
    assert [t for t, _s, _p in client.calls] == ["A", "B"]


def test_stop_ends_after_the_current_chunk():
    client = FakeClient()
    gen = RemoteAudioGenerator(client)
    gen_id, stream = gen.generate_long_audio(["a", "b", "c"], REF)
    next(stream)
    gen.request_stop(gen_id)
    assert list(stream) == []
    assert len(client.calls) == 1


def test_stop_all_does_not_affect_later_generations():
    client = FakeClient()
    gen = RemoteAudioGenerator(client)
    _id, first = gen.generate_long_audio(["a", "b"], REF)
    next(first)
    gen.request_stop()
    assert list(first) == []
    _id, second = gen.generate_long_audio(["c"], REF)
    assert len(list(second)) == 1


def test_stopping_another_generators_id_is_ignored():
    a = RemoteAudioGenerator(FakeClient(), id_base=REMOTE_ID_BASE)
    b = RemoteAudioGenerator(FakeClient(), id_base=REMOTE_ID_BASE * 2)
    id_a, stream_a = a.generate_long_audio(["x", "y"], REF)
    assert a.owns(id_a) and not b.owns(id_a)


def test_engine_error_propagates_to_the_caller():
    gen = RemoteAudioGenerator(FakeClient(fail_at=2))
    _id, stream = gen.generate_long_audio(["a", "b"], REF)
    next(stream)
    with pytest.raises(EngineError, match="exploded"):
        next(stream)
    assert not gen.is_generating


class FakeSelection:
    def __init__(self, winner, acceptable):
        self.winner = winner
        self.acceptable = acceptable

    def as_dict(self):
        return {"winner": self.winner, "candidates": []}


class FakeSelector:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.seen = []

    def select(self, candidates, text, speaker_audio):
        self.seen.append((len(candidates), text, [c.sample_rate for c in candidates]))
        return self.verdicts.pop(0)


def test_best_of_n_requests_takes_one_by_one_and_returns_the_winner():
    client = FakeClient()
    selector = FakeSelector([FakeSelection(winner=1, acceptable=True)])
    reports = []
    gen = RemoteAudioGenerator(client, spoken_text=lambda s: s.replace("[calm] ", ""))
    _id, stream = gen.generate_long_audio(
        ["[calm] hello"], REF, rng_seed=5, num_candidates=3, selector=selector, max_rounds=2,
        on_selection=lambda idx, payload: reports.append((idx, payload)),
    )
    chunk = next(stream)
    assert len(client.calls) == 3
    assert [seed for _t, seed, _p in client.calls] == [5, 6, 7]
    # The verifier is given the spoken words (no markup) at the engine's rate.
    assert selector.seen == [(3, "hello", [24000, 24000, 24000])]
    assert float(chunk[0, 0, 0]) == 2.0  # the second take won
    assert reports[0][0] == 0 and reports[0][1]["rounds"] == 2


def test_best_of_n_retries_when_no_take_is_acceptable():
    client = FakeClient()
    selector = FakeSelector([FakeSelection(0, False), FakeSelection(3, True)])
    gen = RemoteAudioGenerator(client)
    _id, stream = gen.generate_long_audio(
        ["hello"], REF, num_candidates=2, selector=selector, max_rounds=2
    )
    chunk = next(stream)
    assert len(client.calls) == 4
    assert [n for n, _t, _r in selector.seen] == [2, 4]
    assert float(chunk[0, 0, 0]) == 4.0


def test_drama_client_sends_every_reference_clip_and_integer_top_k():
    from longecho.voice_transcripts import ReferenceClip

    seen = {}

    def post(url, body, headers, timeout):
        seen["body"] = json.loads(body)
        return wav_bytes([0] * 10), {}

    ref = VoiceReference(
        "mina", b"M" * 400, "Main clip.",
        extras=(ReferenceClip(b"X" * 400, "Second clip.", "mina2"),
                ReferenceClip(b"Y" * 400, "Third clip.", "mina3")),
    )
    DramaClient("http://x", post=post).synthesize("Hi.", ref, seed=1, params={"top_k": 40.0})
    refs = seen["body"]["references"]
    assert [r["text"] for r in refs] == ["Main clip.", "Second clip.", "Third clip."]
    assert base64.b64decode(refs[1]["audio"]) == b"X" * 400
    assert seen["body"]["top_k"] == 40 and isinstance(seen["body"]["top_k"], int)


def test_breeze_client_uses_only_the_main_clip():
    from longecho.voice_transcripts import ReferenceClip

    seen = {}

    def post(url, body, headers, timeout):
        seen["body"] = body
        return b"\x00\x00" * 4, {}

    ref = VoiceReference("mina", b"MAIN" * 100, "Main.", extras=(ReferenceClip(b"EXTRA" * 100, "Extra."),))
    BreezeClient("http://x", post=post).synthesize("Hi.", ref, seed=1)
    assert b"MAIN" in seen["body"] and b"EXTRA" not in seen["body"]
