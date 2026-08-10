"""Clean book/textbook text extracted from PDFs, readers, or copy-paste.

Text pulled out of a print-formatted source carries a lot of things that must
never be spoken: running headers, page numbers, typesetter stamps, proof
watermarks, footnote reference digits. It also arrives *broken* -- a sentence
that spans a page boundary is split in two, with the page furniture sitting
between the halves.

Both matter, because :mod:`longecho.text_segmenter` turns a single newline into
a comma and a blank line into a period. Feed it raw extracted text and every
page break becomes a full stop in the middle of a sentence, with the header read
aloud after it.

This module runs *before* :class:`~longecho.text_normalizer.TextNormalizer`:

    extract -> clean (here) -> normalize -> segment

Every rule reports what it removed so the UI can show its work -- these are
heuristics, and a rule that eats real prose should be visible, not silent.

Stdlib only, deliberately: no new dependency needs to resolve through the
pinned CUDA torch index just to strip a page number.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable

__all__ = [
    "RULES",
    "PRESETS",
    "DEFAULT_PRESET",
    "Removal",
    "CleanResult",
    "clean_text",
    "resolve_rules",
]


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuleSpec:
    """A single cleaning rule, as advertised to the UI."""

    name: str
    label: str
    description: str


RULES: tuple[RuleSpec, ...] = (
    RuleSpec(
        "unicode",
        "Normalize characters",
        "Fold smart quotes, ligatures, dashes and odd spaces to plain ASCII "
        "equivalents; drop zero-width and control characters.",
    ),
    RuleSpec(
        "typesetter_stamps",
        "Typesetter stamps",
        "Remove print-production lines such as '9299_001.indd 2' and the "
        "'12/13/2016 1:58:29 PM' timestamps that sit in page margins.",
    ),
    RuleSpec(
        "proof_notices",
        "Proof/copyright notices",
        "Remove long ALL-CAPS notices such as 'PROPERTY OF THE MIT PRESS FOR "
        "PROOFREADING...' that are stamped across proof pages.",
    ),
    RuleSpec(
        "running_headers",
        "Running headers & footers",
        "Remove short lines that repeat across the document (chapter titles "
        "reprinted at the top of every page, section names, stray margin marks).",
    ),
    RuleSpec(
        "headings",
        "Chapter & section headings",
        "Terminate headings like 'Chapter 4' with a full stop so they are read "
        "as their own line instead of running into the paragraph below them.",
    ),
    RuleSpec(
        "page_numbers",
        "Page numbers",
        "Remove lines that are just a page number, and trailing page numbers "
        "left behind on header lines.",
    ),
    RuleSpec(
        "reflow",
        "Rejoin split paragraphs",
        "Rejoin sentences broken across page or line boundaries, instead of "
        "letting the segmenter turn the break into a comma or a full stop.",
    ),
    RuleSpec(
        "footnote_markers",
        "Footnote reference numbers",
        "Remove superscript footnote digits that flattened into the text, so "
        "'credits.2 In the election' is not read as 'credits point two'.",
    ),
    RuleSpec(
        "citations",
        "Inline citations",
        "Remove bracketed reference numbers like '[12]' and parenthetical "
        "author-year citations like '(Smith et al., 2019)'.",
    ),
    RuleSpec(
        "figures_tables",
        "Figure & table bodies",
        "Keep the caption line ('Figure 3.1: ...') but drop the table rows, "
        "axis labels and numeric data underneath it.",
    ),
    RuleSpec(
        "formula_lines",
        "Formula & symbol lines",
        "Drop lines that are mostly mathematical symbols or numbers rather "
        "than prose -- equations, axis scales, stray data rows.",
    ),
    RuleSpec(
        "symbols",
        "Speak or drop symbols",
        "Convert leftover symbols in prose to words ('≤' -> 'less than or "
        "equal to') so they are not read as gibberish.",
    ),
    RuleSpec(
        "urls",
        "URLs, DOIs & ISBNs",
        "Remove web addresses and identifiers, which are unlistenable when "
        "read character by character.",
    ),
    RuleSpec(
        "broken_compounds",
        "Repair broken compounds",
        "Repair hyphenated words that lost their hyphen when the source was "
        "de-hyphenated: 'upperincome' -> 'upper-income', 'lowand' -> 'low and'.",
    ),
    RuleSpec(
        "dot_letters",
        "Spaced initials",
        "Rewrite 'J.R.R.' as 'J R R' so the periods are not treated as "
        "sentence endings by the segmenter.",
    ),
    RuleSpec(
        "substitutions",
        "Pronunciation dictionary",
        "Apply your own word replacements, for jargon and names the model "
        "mispronounces.",
    ),
)

RULE_NAMES: frozenset[str] = frozenset(r.name for r in RULES)

# "textbook" is everything; "light" is only the safe, non-structural rules.
PRESETS: dict[str, frozenset[str]] = {
    "off": frozenset(),
    "light": frozenset({
        "unicode",
        "reflow",
        "urls",
        "dot_letters",
        "substitutions",
    }),
    "textbook": RULE_NAMES,
}

DEFAULT_PRESET = "textbook"


def resolve_rules(preset: str | None = None, overrides: dict[str, bool] | None = None) -> set[str]:
    """Resolve a preset name plus per-rule overrides into a set of enabled rules."""
    enabled = set(PRESETS.get((preset or DEFAULT_PRESET).lower(), PRESETS[DEFAULT_PRESET]))
    for name, on in (overrides or {}).items():
        if name not in RULE_NAMES:
            continue
        if on:
            enabled.add(name)
        else:
            enabled.discard(name)
    return enabled


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

MAX_SAMPLES = 6
SAMPLE_CHARS = 100


@dataclass
class Removal:
    """What one rule did, for the preview panel."""

    rule: str
    label: str
    count: int = 0
    samples: list[str] = field(default_factory=list)

    def record(self, sample: str) -> None:
        self.count += 1
        sample = " ".join(sample.split())
        if not sample:
            return
        if len(sample) > SAMPLE_CHARS:
            sample = sample[: SAMPLE_CHARS - 1] + "…"
        if len(self.samples) < MAX_SAMPLES:
            self.samples.append(sample)

    def as_dict(self) -> dict:
        return {
            "rule": self.rule,
            "label": self.label,
            "count": self.count,
            "samples": self.samples,
        }


@dataclass
class CleanResult:
    text: str
    report: list[Removal]
    chars_before: int
    chars_after: int

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "report": [r.as_dict() for r in self.report if r.count],
        }


class _Reporter:
    """Collects per-rule removals in registry order."""

    def __init__(self) -> None:
        self._labels = {r.name: r.label for r in RULES}
        self._removals: dict[str, Removal] = {}

    def get(self, rule: str) -> Removal:
        if rule not in self._removals:
            self._removals[rule] = Removal(rule=rule, label=self._labels.get(rule, rule))
        return self._removals[rule]

    def note(self, rule: str, sample: str) -> None:
        self.get(rule).record(sample)

    def report(self) -> list[Removal]:
        return [self._removals[r.name] for r in RULES if r.name in self._removals]


# ---------------------------------------------------------------------------
# Character-level normalization
# ---------------------------------------------------------------------------

_CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
    "–": "-", "—": ", ", "―": ", ", "−": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", "　": " ",
    "•": " ", "●": " ", "▪": " ", "⁃": " ", "·": " ",
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "­": "",  # soft hyphen
    "​": "", "‌": "", "‍": "", "﻿": "",
    "\f": "\n\n",
}

# Superscript digits, used as footnote markers in many extractions.
_SUPERSCRIPTS = "¹²³⁰⁴⁵⁶⁷⁸⁹"

_SYMBOL_WORDS = {
    "≈": " approximately ", "≠": " not equal to ",
    "≤": " less than or equal to ", "≥": " greater than or equal to ",
    "±": " plus or minus ", "×": " times ", "÷": " divided by ",
    "→": " to ", "←": " from ", "↔": " to ",
    "⇒": " implies ", "∑": " the sum of ", "∫": " the integral of ",
    "√": " the square root of ", "∞": " infinity ",
    "°": " degrees ", "‰": " per mille ",
    "½": " one half ", "¼": " one quarter ", "¾": " three quarters ",
    "<": " less than ", ">": " greater than ",
    "©": " copyright ", "®": "", "™": "",
}


def _rule_unicode(text: str, rep: _Reporter) -> str:
    before = text
    # Strip superscript footnote digits before NFKC folds them into plain
    # digits that are indistinguishable from real numbers.
    def _drop_sup(m: re.Match) -> str:
        rep.note("unicode", f"superscript {m.group(0)!r}")
        return ""

    text = re.sub(f"[{_SUPERSCRIPTS}]+", _drop_sup, text)

    for src, dst in _CHAR_MAP.items():
        if src in text:
            text = text.replace(src, dst)

    # Drop remaining control characters (keep tab/newline).
    text = "".join(
        ch for ch in text
        if ch in "\t\n" or unicodedata.category(ch) not in ("Cc", "Cf")
    )
    text = unicodedata.normalize("NFKC", text)
    # Collapse runs of spaces/tabs but preserve line structure for later rules.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)

    if text != before and not rep.get("unicode").count:
        rep.note("unicode", "folded quotes, dashes and spacing")
    return text


# ---------------------------------------------------------------------------
# Line-level furniture removal
# ---------------------------------------------------------------------------

# '9299_001.indd 2', 'ch01.indd  14'
_STAMP_INDD = re.compile(r"^\S*\.(?:indd|qxd|qxp|fm|dvi)\b.*$", re.IGNORECASE)
# '12/13/2016 1:58:29 PM', '2016-12-13 13:58'
_STAMP_TIME = re.compile(
    r"^\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?$"
)
_STAMP_TIME_ONLY = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?$")
# A run of 5+ ALL-CAPS words: proof stamps, copyright notices.
_CAPS_RUN = re.compile(
    r"\b[A-Z][A-Z'\-]+\b(?:[\s,.\-]+\b[A-Z][A-Z'\-]+\b){4,}"
)
_PAGE_NUMBER_LINE = re.compile(r"^[\[(]?(?:[0-9]{1,4}|[ivxlcdmIVXLCDM]{1,7})[\])]?$")
_TRAILING_PAGE_NUM = re.compile(r"\s+\d{1,4}\s*$")


def _rule_typesetter_stamps(lines: list[str], rep: _Reporter) -> list[str]:
    out = []
    for line in lines:
        s = line.strip()
        if s and (_STAMP_INDD.match(s) or _STAMP_TIME.match(s) or _STAMP_TIME_ONLY.match(s)):
            rep.note("typesetter_stamps", s)
            continue
        out.append(line)
    return out


def _rule_proof_notices(lines: list[str], rep: _Reporter) -> list[str]:
    """Strip long ALL-CAPS notices, keeping whatever else shares the line.

    Proof stamps frequently run into the running header, e.g.
    ``PROPERTY OF THE MIT PRESS ... ONLY Introduction 3`` -- so this removes the
    caps run and leaves ``Introduction 3`` for the header/page-number rules.
    """
    out = []
    for line in lines:
        s = line.strip()
        if not s:
            out.append(line)
            continue
        match = _CAPS_RUN.search(s)
        if match and len(match.group(0)) >= 30:
            rep.note("proof_notices", match.group(0))
            remainder = (s[: match.start()] + " " + s[match.end():]).strip()
            # A line that was mostly a proof stamp is furniture end to end; the
            # few words left over are the running header that shared the line.
            # Leaving them behind is actively harmful, because a stray header
            # can absorb the continuation of the sentence during reflow.
            if not remainder or (len(remainder) < 40 and not _looks_like_prose(remainder)):
                if remainder:
                    rep.note("proof_notices", remainder)
                continue
            s = remainder
        out.append(s)
    return out


def _header_signature(line: str) -> str:
    """Signature used to spot the same header repeated with a different number."""
    s = re.sub(r"\d+", "#", line.strip().lower())
    return re.sub(r"[^a-z#]+", " ", s).strip()


_TERMINAL = tuple(".!?…")


def _looks_like_prose(line: str) -> bool:
    """True if the line ends like a real sentence, so it is not furniture."""
    s = line.strip().rstrip("\"')]}")
    return s.endswith(_TERMINAL)


def _rule_running_headers(lines: list[str], rep: _Reporter) -> list[str]:
    """Drop short lines whose signature repeats across the document.

    A chapter title reprinted on every page, a section name, or a stray margin
    artifact ('Se') all repeat; a real sentence essentially never does. Lines
    that end in sentence punctuation are exempt, so repeated dialogue survives.

    Very short fragments are dropped whether or not they repeat: a one- or
    two-letter line is a cropped margin mark, not a sentence, and a single
    extracted page may only contain it once.
    """
    counts: Counter[str] = Counter()
    for line in lines:
        s = line.strip()
        if s and len(s) <= 100 and not _looks_like_prose(s):
            counts[_header_signature(s)] += 1

    out = []
    for line in lines:
        s = line.strip()
        if s and len(s) <= 100 and not _looks_like_prose(s):
            sig = _header_signature(s)
            is_repeated = bool(sig) and counts[sig] >= 2
            is_fragment = len(s) <= 3 and s.lower() not in ("i", "a", "no", "ok", "so")
            if is_repeated or is_fragment:
                rep.note("running_headers", s)
                continue
        out.append(line)
    return out


# Structural headings. A heading is not an unfinished sentence, but it looks
# like one to the reflow rule, which would otherwise glue it to the paragraph
# underneath ("Chapter" + "packages led to..." -> "Chapter packages led to...").
_HEADING = re.compile(
    r"^(chapter|part|section|appendix|book|volume|unit|lesson|epilogue|prologue"
    r"|preface|foreword|introduction|conclusion|afterword|glossary|index)\b"
    r"[\s:.\-]*([0-9]{1,3}|[ivxlcdm]{1,7})?\s*$",
    re.IGNORECASE,
)


def _rule_headings(lines: list[str], rep: _Reporter) -> list[str]:
    """Give a heading a full stop so it reads as its own line.

    Runs after the running-header rule, so a heading reprinted on every page is
    already gone; what reaches here appeared once and is a real heading worth
    speaking. Runs before the page-number rule so 'Chapter 4' keeps its number.
    """
    out = []
    for line in lines:
        s = line.strip()
        if s and len(s) <= 60 and not _looks_like_prose(s) and _HEADING.match(s):
            fixed = s.rstrip(" .:-") + "."
            rep.note("headings", f"{s} -> {fixed}")
            out.append(fixed)
            continue
        out.append(line)
    return out


def _rule_page_numbers(lines: list[str], rep: _Reporter) -> list[str]:
    out = []
    for line in lines:
        s = line.strip()
        if not s:
            out.append(line)
            continue
        if _PAGE_NUMBER_LINE.match(s):
            rep.note("page_numbers", s)
            continue
        # 'Introduction 3' -> drop the trailing page number on short header lines.
        if len(s) <= 60 and not _looks_like_prose(s):
            trimmed = _TRAILING_PAGE_NUM.sub("", s)
            if trimmed != s and trimmed:
                rep.note("page_numbers", s)
                s = trimmed
        out.append(s)
    return out


# --- figures / tables / formulas -------------------------------------------

_CAPTION = re.compile(
    r"^\s*((?:figure|fig\.?|table|chart|exhibit|plate|box|diagram|graph)\s*"
    r"[0-9ivxlc]+(?:[.\-][0-9]+)*)\s*[:.\-]?\s*(.*)$",
    re.IGNORECASE,
)
_SOURCE_LINE = re.compile(
    r"^\s*(source|sources|note|notes|adapted from|reprinted from|credit)\s*[:.]",
    re.IGNORECASE,
)
_NUMERIC_ROW = re.compile(r"^[^A-Za-z]*(?:[-+]?[\d.,%$()]+[\s|\t]+){2,}[-+]?[\d.,%$()]+[^A-Za-z]*$")


def _symbol_density(line: str) -> float:
    stripped = [c for c in line if not c.isspace()]
    if not stripped:
        return 0.0
    letters = sum(1 for c in stripped if c.isalpha())
    return 1.0 - (letters / len(stripped))


def _rule_figures_tables(lines: list[str], rep: _Reporter) -> list[str]:
    """Keep a figure/table caption; drop the data block that follows it.

    The caption is what orients a listener when the prose says "as shown in
    figure 3.1"; the rows and axis labels underneath are unlistenable. After a
    caption, lines are dropped until prose resumes (a line that reads like a
    sentence and is not itself numeric).
    """
    out: list[str] = []
    in_block = False
    for line in lines:
        s = line.strip()
        caption = _CAPTION.match(s) if s else None
        if caption:
            label, rest = caption.group(1), caption.group(2).strip()
            kept = f"{label}. {rest}".strip() if rest else f"{label}."
            out.append(kept)
            in_block = True
            continue
        if in_block:
            if not s:
                out.append(line)
                continue
            if _SOURCE_LINE.match(s) or _NUMERIC_ROW.match(s) or _symbol_density(s) > 0.4:
                rep.note("figures_tables", s)
                continue
            if _looks_like_prose(s) and len(s) > 80:
                in_block = False
                out.append(line)
                continue
            # Short non-prose line inside a figure block: axis label, legend.
            if len(s) <= 60 and not _looks_like_prose(s):
                rep.note("figures_tables", s)
                continue
            in_block = False
        out.append(line)
    return out


def _rule_formula_lines(lines: list[str], rep: _Reporter) -> list[str]:
    out = []
    for line in lines:
        s = line.strip()
        # Only judge short standalone lines; never drop a paragraph.
        if s and len(s) <= 120 and _symbol_density(s) > 0.55:
            rep.note("formula_lines", s)
            continue
        if s and _NUMERIC_ROW.match(s) and len(s) <= 200:
            rep.note("formula_lines", s)
            continue
        out.append(line)
    return out


# ---------------------------------------------------------------------------
# Reflow: rejoin sentences split across page/line boundaries
# ---------------------------------------------------------------------------

def _continues(prev: str, nxt: str) -> bool:
    """True if ``nxt`` looks like the continuation of an unfinished ``prev``."""
    if not prev or not nxt:
        return False
    if prev.endswith("-"):
        return True
    if _looks_like_prose(prev):
        return False
    if prev.rstrip().endswith(":"):
        return False
    # A short trailing fragment is a heading or a stray label, not the first
    # half of a severed sentence -- a real page split leaves most of a page
    # behind it. Refusing to join is the safe failure: worst case the segmenter
    # inserts a pause, rather than a heading swallowing the next paragraph.
    if len(prev.strip()) < 25:
        return False
    first = nxt.lstrip()
    if not first:
        return False
    ch = first[0]
    # A continuation resumes mid-sentence: lowercase, or an opening quote
    # followed by lowercase, or a conjunction-ish fragment.
    if ch.islower():
        return True
    if ch in "\"'(" and len(first) > 1 and first[1].islower():
        return True
    return False


def _join(prev: str, nxt: str) -> str:
    if prev.endswith("-"):
        # De-hyphenate only when a lowercase word continues the token.
        if nxt[:1].islower():
            return prev[:-1] + nxt
        return prev + nxt
    return prev + " " + nxt


def _rule_reflow(text: str, rep: _Reporter) -> str:
    """Rejoin wrapped lines within paragraphs, then across page breaks."""
    blocks = re.split(r"\n\s*\n+", text)
    rejoined_blocks: list[str] = []

    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        merged = lines[0]
        for line in lines[1:]:
            if _continues(merged, line):
                rep.note("reflow", f"…{merged[-40:]} + {line[:40]}…")
                merged = _join(merged, line)
            else:
                merged += "\n" + line
        rejoined_blocks.append(merged)

    # Across blank-line boundaries: a page break splits a sentence and leaves a
    # gap where the furniture used to be.
    out: list[str] = []
    for block in rejoined_blocks:
        if out:
            prev_tail = out[-1].split("\n")[-1]
            first_head = block.split("\n")[0]
            if _continues(prev_tail, first_head):
                rep.note("reflow", f"…{prev_tail[-40:]} + {first_head[:40]}…")
                lines = out[-1].split("\n")
                lines[-1] = _join(prev_tail, first_head)
                rest = block.split("\n")[1:]
                out[-1] = "\n".join(lines + rest)
                continue
        out.append(block)

    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Inline cleanup
# ---------------------------------------------------------------------------

# 'credits.2 In the election' / 'nation).3 To put'. Requires a letter or closing
# bracket before the punctuation so decimals like '39.6 percent' are untouched.
_FOOTNOTE_REF = re.compile(r"(?<=[A-Za-z\)\]\"'])([.!?,;:])(\d{1,3})(?=\s|$)")
_CITATION_BRACKET = re.compile(r"\s*\[\s*\d+(?:\s*[,\-–]\s*\d+)*\s*\]")
_CITATION_AUTHOR = re.compile(
    r"\s*\((?:see\s+)?(?:e\.g\.,?\s*)?[A-Z][A-Za-z'\-]+"
    r"(?:\s+(?:et\s+al\.?|and|&|,)\s*[A-Z]?[A-Za-z'\-]*)*,?\s*\d{4}[a-z]?"
    r"(?:\s*[;,]\s*[^()]{0,40}?\d{4}[a-z]?)*\s*\)"
)
# A dangling preposition is swallowed with the address, so "available at
# https://... and in the appendix" does not become "available at and in".
_URL = re.compile(
    r"(?:\b(?:at|from|via|on)\s+)?"
    r"(?:\b(?:https?://|www\.)\S+|\bdoi:\s*\S+|\b10\.\d{4,}/\S+)",
    re.IGNORECASE,
)
_ISBN = re.compile(r"\bISBN[- ]?(?:13|10)?:?\s*[\d\-Xx]{10,17}\b")
_DOT_LETTERS = re.compile(r"\b((?:[A-Z]\.){2,})(?=\s|$|[,;:])")

# Compounds that lost their hyphen when the source was de-hyphenated, e.g.
# 'upper-income' -> 'upperincome'. Splitting on prefixes alone is dangerous --
# 'under' + 'standing' is the real word 'understanding' -- so the tail list is
# restricted to words that essentially never form a solid compound with these
# prefixes, with an explicit exception set for the few that do.
_COMPOUND_PREFIXES = (
    "low", "high", "upper", "lower", "middle", "long", "short", "well",
    "cross", "multi", "single", "double", "self", "non", "anti", "pre",
    "post", "sub", "super", "inter", "intra", "over", "under", "tax",
)
_COMPOUND_TAILS = {
    "income": "-", "incomes": "-", "term": "-", "terms": "-", "run": "-",
    "based": "-", "cutting": "-", "wage": "-", "wages": "-", "skilled": "-",
    "earners": "-", "bracket": "-", "brackets": "-",
    "and": " ", "or": " ", "to": " ",
}
_COMPOUND_RE = re.compile(
    r"\b(" + "|".join(sorted(_COMPOUND_PREFIXES, key=len, reverse=True)) + r")"
    r"(" + "|".join(sorted(_COMPOUND_TAILS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
# The handful of prefix+tail pairs above that really are single English words.
_COMPOUND_EXCEPT = frozenset({
    "crosscutting", "overrun", "overruns", "underrun", "underruns",
    "longrun", "subterm", "interterm",
})


def _rule_footnote_markers(text: str, rep: _Reporter) -> str:
    def repl(m: re.Match) -> str:
        rep.note("footnote_markers", f"…{m.group(0)}")
        return m.group(1)

    return _FOOTNOTE_REF.sub(repl, text)


def _rule_citations(text: str, rep: _Reporter) -> str:
    def repl(m: re.Match) -> str:
        rep.note("citations", m.group(0))
        return ""

    text = _CITATION_BRACKET.sub(repl, text)
    return _CITATION_AUTHOR.sub(repl, text)


def _rule_urls(text: str, rep: _Reporter) -> str:
    def repl(m: re.Match) -> str:
        rep.note("urls", m.group(0))
        return ""

    return _ISBN.sub(repl, _URL.sub(repl, text))


def _rule_symbols(text: str, rep: _Reporter) -> str:
    for src, dst in _SYMBOL_WORDS.items():
        if src in text:
            rep.note("symbols", f"{src} -> {dst.strip() or '(removed)'}")
            text = text.replace(src, dst)
    # Greek letters read as gibberish byte sequences; name them.
    def greek(m: re.Match) -> str:
        try:
            name = unicodedata.name(m.group(0)).split()[-1].lower()
        except ValueError:
            return ""
        rep.note("symbols", f"{m.group(0)} -> {name}")
        return f" {name} "

    return re.sub(r"[Ͱ-Ͽ]", greek, text)


def _rule_dot_letters(text: str, rep: _Reporter) -> str:
    def repl(m: re.Match) -> str:
        spaced = " ".join(c for c in m.group(1) if c.isalpha())
        rep.note("dot_letters", f"{m.group(1)} -> {spaced}")
        return spaced

    return _DOT_LETTERS.sub(repl, text)


def _rule_broken_compounds(text: str, rep: _Reporter) -> str:
    def repl(m: re.Match) -> str:
        whole = m.group(0)
        if whole.lower() in _COMPOUND_EXCEPT:
            return whole
        joiner = _COMPOUND_TAILS[m.group(2).lower()]
        fixed = m.group(1) + joiner + m.group(2)
        rep.note("broken_compounds", f"{whole} -> {fixed}")
        return fixed

    return _COMPOUND_RE.sub(repl, text)


def _rule_substitutions(text: str, rep: _Reporter, subs: dict[str, str]) -> str:
    for src, dst in subs.items():
        if not src:
            continue
        pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
        if pattern.search(text):
            rep.note("substitutions", f"{src} -> {dst}")
            text = pattern.sub(dst, text)
    return text


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_LINE_RULES: tuple[tuple[str, Callable[[list[str], _Reporter], list[str]]], ...] = (
    ("typesetter_stamps", _rule_typesetter_stamps),
    ("proof_notices", _rule_proof_notices),
    # Repeated headers go first: the signature normalizes digits, so
    # 'Introduction 3' and 'Introduction 5' match each other without needing
    # the page number stripped beforehand.
    ("running_headers", _rule_running_headers),
    # Then headings, so a heading that appeared only once (a real one, not a
    # running header) keeps its number and gains a full stop before the page
    # number rule could strip the number off it.
    ("headings", _rule_headings),
    ("page_numbers", _rule_page_numbers),
    ("figures_tables", _rule_figures_tables),
    ("formula_lines", _rule_formula_lines),
)

_TEXT_RULES: tuple[tuple[str, Callable[[str, _Reporter], str]], ...] = (
    ("footnote_markers", _rule_footnote_markers),
    ("citations", _rule_citations),
    ("urls", _rule_urls),
    ("symbols", _rule_symbols),
    ("broken_compounds", _rule_broken_compounds),
    ("dot_letters", _rule_dot_letters),
)


def clean_text(
    text: str,
    preset: str | None = None,
    overrides: dict[str, bool] | None = None,
    substitutions: dict[str, str] | None = None,
) -> CleanResult:
    """Clean extracted book text for speech.

    Args:
        text: Raw extracted text.
        preset: ``"off"``, ``"light"`` or ``"textbook"`` (default).
        overrides: Per-rule ``{name: enabled}`` overrides on top of the preset.
        substitutions: Pronunciation dictionary applied last.

    Returns:
        A :class:`CleanResult` with the cleaned text and a per-rule report of
        what was removed.
    """
    chars_before = len(text)
    rep = _Reporter()
    enabled = resolve_rules(preset, overrides)

    if not text or not text.strip() or not enabled:
        return CleanResult(text=text, report=[], chars_before=chars_before, chars_after=chars_before)

    if "unicode" in enabled:
        text = _rule_unicode(text, rep)

    lines = text.split("\n")
    for name, fn in _LINE_RULES:
        if name in enabled:
            lines = fn(lines, rep)
    text = "\n".join(lines)

    # Reflow after furniture removal: the page break that split a sentence is
    # only adjacent once the header sitting between the halves is gone.
    if "reflow" in enabled:
        text = _rule_reflow(text, rep)

    for name, fn in _TEXT_RULES:
        if name in enabled:
            text = fn(text, rep)

    if "substitutions" in enabled and substitutions:
        text = _rule_substitutions(text, rep, substitutions)

    # Tidy the seams left by removals.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" ([.,!?;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(ln.strip() for ln in text.split("\n")).strip()

    return CleanResult(
        text=text,
        report=rep.report(),
        chars_before=chars_before,
        chars_after=len(text),
    )
