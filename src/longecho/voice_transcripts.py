"""Reference clips and their transcripts, for engines that clone in-context.

Echo conditions on the reference *audio* alone. Drama 3 instead continues from
the reference the way a language model continues a prompt, so it needs the
exact words spoken in it, and it wants a short clip (Fish recommends 10-30
seconds) rather than Echo's up-to-five-minutes.

So for such an engine a voice is reduced to:

* a **reference clip**: the first ``max_seconds`` of the voice's .wav, and
* a **transcript** of exactly that clip, kept in ``<voice>.transcript.txt``
  beside the .wav so it survives restarts and can be corrected by hand.

Everything here is stdlib-only.
"""

from __future__ import annotations

import array
import io
import logging
import os
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

TRANSCRIPT_SUFFIX = ".transcript.txt"
DEFAULT_REFERENCE_SECONDS = 30.0
MAX_TRANSCRIPT_CHARS = 4000


# Reference clips quieter than this (peak, as a fraction of full scale) are
# raised to REFERENCE_TARGET_PEAK. In-context engines copy the level of the
# reference along with the voice, so a quiet clip gives quiet, thin output.
REFERENCE_QUIET_PEAK = 0.5
REFERENCE_TARGET_PEAK = 0.89


@dataclass(frozen=True)
class ReferenceClip:
    """One reference recording and the words spoken in it."""

    wav_bytes: bytes
    transcript: str
    name: str = ""


@dataclass(frozen=True)
class VoiceReference:
    """What a remote engine needs to clone a voice.

    ``extras`` are further recordings of the same speaker. Engines that can
    condition on several references (Drama 3) get all of them; the others use
    the main clip only.
    """

    name: str
    wav_bytes: bytes
    transcript: str
    extras: tuple[ReferenceClip, ...] = ()

    @property
    def clips(self) -> list[ReferenceClip]:
        return [ReferenceClip(self.wav_bytes, self.transcript, self.name), *self.extras]


def reference_seconds() -> float:
    """Length of the reference clip sent to in-context engines."""
    try:
        value = float(os.environ.get("CANDYECHO_REFERENCE_SECONDS", DEFAULT_REFERENCE_SECONDS))
    except ValueError:
        return DEFAULT_REFERENCE_SECONDS
    return max(3.0, min(value, 120.0))


def _boost_quiet_pcm16(frames: bytes) -> bytes:
    """Raise a quiet 16-bit clip to a healthy level; leave a normal one alone."""
    samples = array.array("h")
    samples.frombytes(frames[: len(frames) - (len(frames) % 2)])
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return frames
    peak = max(max(samples), -min(samples)) / 32768.0
    if peak <= 0.0 or peak >= REFERENCE_QUIET_PEAK:
        return frames
    gain = REFERENCE_TARGET_PEAK / peak
    boosted = array.array(
        "h", (max(-32768, min(32767, int(s * gain))) for s in samples)
    )
    if sys.byteorder == "big":
        boosted.byteswap()
    logger.info(f"Reference clip was quiet (peak {peak:.2f}); raised by {gain:.1f}x")
    return boosted.tobytes()


def reference_clip(
    wav_path: Path | str, max_seconds: float | None = None, boost_quiet: bool | None = None
) -> bytes:
    """Return the first ``max_seconds`` of a .wav as WAV bytes.

    PCM files are trimmed with the stdlib ``wave`` module, and a 16-bit clip
    that is very quiet is raised to a normal level (turn off with
    ``CANDYECHO_REFERENCE_BOOST=0``). Anything ``wave`` cannot parse (float or
    compressed WAVs) is returned whole and untouched: a longer reference is
    better than failing, and the transcript is made from the same bytes, so the
    two always agree.
    """
    path = Path(wav_path)
    limit = reference_seconds() if max_seconds is None else max_seconds
    if boost_quiet is None:
        boost_quiet = os.environ.get("CANDYECHO_REFERENCE_BOOST", "1") != "0"
    try:
        with wave.open(str(path), "rb") as src:
            params = src.getparams()
            frames = src.readframes(min(src.getnframes(), int(limit * src.getframerate())))
        if boost_quiet and params.sampwidth == 2:
            frames = _boost_quiet_pcm16(frames)
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
