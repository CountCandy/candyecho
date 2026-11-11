"""Drama 3 script markup: what it looks like and how to keep it valid.

Drama 3 is directed by markup written into the text itself:

* ``[warm, a little tired]``   a line direction, in force until the next one
* ``[sigh]`` ``[pause]``        a sound or a beat at that exact point
* ``<whisper>...</whisper>``    a paired tag around the words it applies to

Stdlib-only, so both the engine client and the tag enhancer can use it and it
can be tested without the ML stack.
"""

from __future__ import annotations

import re
from typing import List

PAIRED_TAGS = ("whisper", "emphasis", "soft", "fast", "slow", "stress", "forceful")

# Bracketed cues that are a sound or a beat at one point in the line, as
# opposed to a direction that holds until the next one.
_SOUND_RE = re.compile(
    r"\b(?:sigh|laugh|chuckl|giggl|breath|inhal|exhal|paus|beat\b|gasp|cough|sob|cry|"
    r"sniff|groan|yawn|gulp|hum(?:s|ming)?\b|scoff|snort|clears? (?:his |her |their )?throat|"
    r"tsk|silence|whimper|scream|shout)",
    re.IGNORECASE,
)

_BRACKET_RE = re.compile(r"\[([^\[\]]{1,200})\]")
_PAIRED_RE = re.compile(r"<(/?)(" + "|".join(PAIRED_TAGS) + r")>", re.IGNORECASE)


def is_sound_cue(tag_body: str) -> bool:
    """True for ``[sigh]``-style point events, False for ``[warm, steady]`` directions.

    A heuristic: sound cues are short, have no comma, and name a sound. Getting
    it wrong is mild either way (a direction is repeated once more than needed,
    or not carried into the next chunk).
    """
    body = tag_body.strip()
    if "," in body or len(body.split()) > 3:
        return False
    return bool(_SOUND_RE.search(body))


def prepare_drama_chunks(chunks: List[str]) -> List[str]:
    """Make each chunk stand on its own as a Drama 3 script.

    CandyEcho sends one request per chunk, but Drama 3's markup assumes one
    continuous script, so two things would otherwise break at chunk edges:

    * A line direction such as ``[calm]`` holds until the next direction. A
      later chunk that carries none would silently fall back to neutral, so the
      direction still in force is repeated at its start.
    * A paired tag such as ``<whisper>`` may be opened in one chunk and closed
      in the next; it is closed at the end of the first and reopened in the
      second so neither request sees a dangling tag.
    """
    prepared: list[str] = []
    active_direction: str | None = None
    open_tags: list[str] = []

    for chunk in chunks:
        text = chunk.strip()
        stripped_lead = text.lstrip()
        leads_with_direction = False
        lead = _BRACKET_RE.match(stripped_lead)
        if lead and not is_sound_cue(lead.group(1)):
            leads_with_direction = True

        prefix = ""
        if active_direction and not leads_with_direction:
            prefix += f"[{active_direction}] "
        prefix += "".join(f"<{tag}>" for tag in open_tags)

        # Track paired tags through this chunk to know what is still open.
        for match in _PAIRED_RE.finditer(text):
            closing, name = match.group(1), match.group(2).lower()
            if not closing:
                open_tags.append(name)
            elif name in open_tags:
                # Close the most recent matching opener.
                for i in range(len(open_tags) - 1, -1, -1):
                    if open_tags[i] == name:
                        del open_tags[i]
                        break
        suffix = "".join(f"</{tag}>" for tag in reversed(open_tags))

        for match in _BRACKET_RE.finditer(text):
            if not is_sound_cue(match.group(1)):
                active_direction = match.group(1).strip()

        prepared.append(f"{prefix}{text}{suffix}")
    return prepared


def strip_drama_markup(text: str) -> str:
    """The words that are actually spoken, with directions and tags removed."""
    text = _BRACKET_RE.sub(" ", text)
    text = _PAIRED_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()
