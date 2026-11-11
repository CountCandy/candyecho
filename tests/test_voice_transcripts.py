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
