"""Build SRT/VTT subtitles from a finished job's chunks.

Timing comes from the measured duration of each generated chunk, which is exact
at chunk boundaries. Within a chunk, cues are split at sentence boundaries and
time is apportioned by character count -- an approximation, but a good one for
speech at a roughly even rate, and it needs no extra model pass.

Word-level timing would need the ASR to be re-run with timestamps requested,
which only applies when best-of-N verification is on; that is a later
refinement. Everything here works for every job.

Standard library only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = ["Cue", "build_cues", "to_srt", "to_vtt", "split_sentences"]

# Longest a single cue should run before it is split further.
MAX_CUE_SECONDS = 7.0
# Cues shorter than this get merged into their neighbour rather than flashing.
MIN_CUE_SECONDS = 0.6
MAX_CUE_CHARS = 90

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, further splitting anything overlong."""
    parts = [p.strip() for p in _SENTENCE_END.split(text.strip()) if p.strip()]
    out: list[str] = []
    for part in parts:
        if len(part) <= MAX_CUE_CHARS:
            out.append(part)
            continue
        # Too long for one cue: break at clause separators, then at spaces.
        pieces = re.split(r"(?<=[,;:])\s+", part)
        buffer = ""
        for piece in pieces:
            candidate = f"{buffer} {piece}".strip()
            if buffer and len(candidate) > MAX_CUE_CHARS:
                out.append(buffer)
                buffer = piece
            else:
                buffer = candidate
        if buffer:
            out.append(buffer)
    return out or ([text.strip()] if text.strip() else [])


def _apportion(text: str, start: float, duration: float) -> list[tuple[float, float, str]]:
    """Split one chunk's text across its duration, weighted by characters."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    total_chars = sum(len(s) for s in sentences) or 1

    spans: list[tuple[float, float, str]] = []
    cursor = start
    for i, sentence in enumerate(sentences):
        share = duration * (len(sentence) / total_chars)
        # Absorb rounding into the last cue so the chunk boundary stays exact.
        end = start + duration if i == len(sentences) - 1 else cursor + share
        spans.append((cursor, end, sentence))
        cursor = end
    return spans


def build_cues(chunks: Iterable[tuple[str, float]]) -> list[Cue]:
    """Turn (text, duration) pairs into timed cues.

    Args:
        chunks: One entry per generated chunk, in playback order.
    """
    spans: list[tuple[float, float, str]] = []
    clock = 0.0
    for text, duration in chunks:
        duration = max(0.0, float(duration or 0.0))
        if text and text.strip() and duration > 0:
            spans.extend(_apportion(text, clock, duration))
        clock += duration

    # Merge anything too brief to read, and split anything too long to follow.
    merged: list[tuple[float, float, str]] = []
    for start, end, text in spans:
        if merged and (end - start) < MIN_CUE_SECONDS:
            prev_start, _prev_end, prev_text = merged[-1]
            merged[-1] = (prev_start, end, f"{prev_text} {text}".strip())
            continue
        merged.append((start, end, text))

    cues: list[Cue] = []
    for start, end, text in merged:
        span = end - start
        if span > MAX_CUE_SECONDS and len(text) > 40:
            halves = max(2, int(span // MAX_CUE_SECONDS) + 1)
            words = text.split()
            per = max(1, len(words) // halves)
            step = span / halves
            for h in range(halves):
                piece = " ".join(words[h * per: (h + 1) * per] if h < halves - 1 else words[h * per:])
                if not piece:
                    continue
                cues.append(Cue(len(cues) + 1, start + h * step, start + (h + 1) * step, piece))
        else:
            cues.append(Cue(len(cues) + 1, start, end, text))

    for i, cue in enumerate(cues, 1):
        cue.index = i
    return cues


def _stamp(seconds: float, millis_sep: str) -> str:
    seconds = max(0.0, seconds)
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:  # rounding can tip over a whole second
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millis_sep}{millis:03d}"


def to_srt(cues: Sequence[Cue]) -> str:
    blocks = [
        f"{cue.index}\n{_stamp(cue.start, ',')} --> {_stamp(cue.end, ',')}\n{cue.text}"
        for cue in cues
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(cues: Sequence[Cue]) -> str:
    blocks = [
        f"{cue.index}\n{_stamp(cue.start, '.')} --> {_stamp(cue.end, '.')}\n{cue.text}"
        for cue in cues
    ]
    return "WEBVTT\n\n" + "\n\n".join(blocks) + ("\n" if blocks else "")
