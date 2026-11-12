"""Reference clips and transcript sidecars."""
import io
import wave

import pytest

from longecho.voice_transcripts import TranscriptStore, reference_clip


def write_wav(path, seconds, rate=8000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(seconds * rate))


def duration(data):
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes() / w.getframerate()


def test_reference_clip_trims_long_audio(tmp_path):
    path = tmp_path / "long.wav"
    write_wav(path, 12)
    assert duration(reference_clip(path, max_seconds=5)) == pytest.approx(5.0)


def test_reference_clip_keeps_short_audio_whole(tmp_path):
    path = tmp_path / "short.wav"
    write_wav(path, 2)
    assert duration(reference_clip(path, max_seconds=30)) == pytest.approx(2.0)


def test_reference_clip_falls_back_to_whole_file_when_unparseable(tmp_path):
    path = tmp_path / "odd.wav"
    path.write_bytes(b"not really a wav file")
    assert reference_clip(path, max_seconds=5) == b"not really a wav file"


def test_transcript_round_trip_and_whitespace(tmp_path):
    store = TranscriptStore(tmp_path)
    assert store.get("alice") is None
    assert store.set("alice", "  Hello   there,\n friend. ") == "Hello there, friend."
    assert store.get("alice") == "Hello there, friend."
    assert (tmp_path / "alice.transcript.txt").exists()


def test_empty_transcript_clears_the_file(tmp_path):
    store = TranscriptStore(tmp_path)
    store.set("alice", "hi")
    store.set("alice", "   ")
    assert store.get("alice") is None
    assert not (tmp_path / "alice.transcript.txt").exists()


def test_rename_and_delete_follow_the_voice(tmp_path):
    store = TranscriptStore(tmp_path)
    store.set("old", "words")
    store.rename("old", "new")
    assert store.get("old") is None and store.get("new") == "words"
    store.delete("new")
    assert store.get("new") is None
    store.delete("never-existed")  # no error


@pytest.mark.parametrize("name", ["../evil", "a/b", "", ".hidden"])
def test_names_cannot_escape_the_voice_dir(tmp_path, name):
    store = TranscriptStore(tmp_path)
    with pytest.raises(ValueError):
        store.path(name)
    assert store.get(name) is None


# --- quiet-clip boost and multiple references -----------------------------

import struct

from longecho.voice_transcripts import ReferenceClip, VoiceReference


def write_tone(path, amplitude, seconds=0.5, rate=8000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", amplitude if i % 2 else -amplitude) for i in range(int(seconds * rate))
        )
        w.writeframes(frames)


def peak(data):
    with wave.open(io.BytesIO(data), "rb") as w:
        frames = w.readframes(w.getnframes())
    return max(abs(s[0]) for s in struct.iter_unpack("<h", frames)) / 32768.0


def test_quiet_reference_is_boosted(tmp_path):
    path = tmp_path / "quiet.wav"
    write_tone(path, amplitude=1000)  # about 3% of full scale
    assert peak(reference_clip(path, max_seconds=5)) == pytest.approx(0.89, abs=0.01)


def test_normal_reference_is_left_alone(tmp_path):
    path = tmp_path / "ok.wav"
    write_tone(path, amplitude=20000)
    assert peak(reference_clip(path, max_seconds=5)) == pytest.approx(20000 / 32768, abs=0.001)


def test_boost_can_be_turned_off(tmp_path, monkeypatch):
    path = tmp_path / "quiet.wav"
    write_tone(path, amplitude=1000)
    assert peak(reference_clip(path, max_seconds=5, boost_quiet=False)) == pytest.approx(1000 / 32768, abs=0.001)
    monkeypatch.setenv("CANDYECHO_REFERENCE_BOOST", "0")
    assert peak(reference_clip(path, max_seconds=5)) == pytest.approx(1000 / 32768, abs=0.001)


def test_silent_reference_does_not_divide_by_zero(tmp_path):
    path = tmp_path / "silent.wav"
    write_wav(path, 1)
    assert duration(reference_clip(path, max_seconds=5)) == pytest.approx(1.0)


def test_voice_reference_lists_main_clip_then_extras():
    ref = VoiceReference("mina", b"A", "one", extras=(ReferenceClip(b"B", "two", "mina2"),))
    assert [(c.wav_bytes, c.transcript) for c in ref.clips] == [(b"A", "one"), (b"B", "two")]
    assert VoiceReference("solo", b"A", "one").clips[0].name == "solo"
