"""Reference clips and their transcripts, for engines that clone in-context.

Echo conditions on the reference *audio* alone. Drama 3 and Breeze instead
continue from the reference the way a language model continues a prompt, so
they need the exact words spoken in it, and they want a short clip (Fish
recommends 10-30 seconds) rather than Echo's up-to-five-minutes.

So for those engines a voice is reduced to:

* a **reference clip**: the first ``max_seconds`` of the voice's .wav, and
* a **transcript** of exactly that clip, kept in ``<voice>.transcript.txt``
  beside the .wav so it survives restarts and can be corrected by hand.

Everything here is stdlib-only.
"""

from __future__ import annotations

import io
import logging
import os
import wave
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

TRANSCRIPT_SUFFIX = ".transcript.txt"
DEFAULT_REFERENCE_SECONDS = 30.0
MAX_TRANSCRIPT_CHARS = 4000


@dataclass(frozen=True)
class VoiceReference:
    """What a remote engine needs to clone a voice."""

    name: str
    wav_bytes: bytes
    transcript: str


def reference_seconds() -> float:
    """Length of the reference clip sent to in-context engines."""
    try:
        value = float(os.environ.get("CANDYECHO_REFERENCE_SECONDS", DEFAULT_REFERENCE_SECONDS))
    except ValueError:
        return DEFAULT_REFERENCE_SECONDS
    return max(3.0, min(value, 120.0))


def reference_clip(wav_path: Path | str, max_seconds: float | None = None) -> bytes:
    """Return the first ``max_seconds`` of a .wav as WAV bytes.

    PCM files are trimmed with the stdlib ``wave`` module. Anything it cannot
    parse (float or compressed WAVs) is returned whole: a longer reference is
    better than failing, and the transcript is made from the same bytes, so the
    two always agree.
    """
    path = Path(wav_path)
    limit = reference_seconds() if max_seconds is None else max_seconds
    try:
        with wave.open(str(path), "rb") as src:
            params = src.getparams()
            frames = src.readframes(min(src.getnframes(), int(limit * src.getframerate())))
        out = io.BytesIO()
        with wave.open(out, "wb") as dst:
            dst.setnchannels(params.nchannels)
            dst.setsampwidth(params.sampwidth)
            dst.setframerate(params.framerate)
            dst.writeframes(frames)
        return out.getvalue()
    except (wave.Error, EOFError, OSError) as e:
        logger.info(f"Could not trim '{path.name}' ({e}); sending the whole file as reference")
        return path.read_bytes()


class TranscriptStore:
    """Sidecar transcripts stored next to the voice .wav files."""

    def __init__(self, voice_dir: Path | str = "voice_library"):
        self.voice_dir = Path(voice_dir)

    def path(self, voice_name: str) -> Path:
        # voice_name comes from the loaded-voice list, but guard anyway so this
        # can never be steered outside the voice directory.
        if not voice_name or Path(voice_name).name != voice_name or voice_name.startswith("."):
            raise ValueError(f"Invalid voice name: {voice_name!r}")
        return self.voice_dir / f"{voice_name}{TRANSCRIPT_SUFFIX}"

    def get(self, voice_name: str) -> str | None:
        try:
            text = self.path(voice_name).read_text(encoding="utf-8").strip()
        except (OSError, ValueError):
            return None
        return text or None

    def set(self, voice_name: str, text: str) -> str:
        cleaned = " ".join((text or "").split())[:MAX_TRANSCRIPT_CHARS]
        path = self.path(voice_name)
        if not cleaned:
            path.unlink(missing_ok=True)
            return ""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(cleaned + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return cleaned

    def delete(self, voice_name: str) -> None:
        try:
            self.path(voice_name).unlink(missing_ok=True)
        except ValueError:
            pass

    def rename(self, old_name: str, new_name: str) -> None:
        try:
            old, new = self.path(old_name), self.path(new_name)
        except ValueError:
            return
        if old.exists():
            try:
                os.replace(old, new)
            except OSError as e:
                logger.warning(f"Could not move transcript for '{old_name}': {e}")
