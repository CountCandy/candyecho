"""Adds Drama 3 performance directions to a plain script with a local LLM.

Drama 3 is steered by markup written into the text:

* ``[warm, a little tired]``   a line direction, in force until the next one
* ``[sigh]`` ``[pause]``        a sound or a beat at that exact point
* ``<whisper>...</whisper>``    a paired tag around the words it applies to

Writing that by hand for a long script is tedious, so this module asks an
instruct LLM to do it, the way the "enhance" button on Fish Audio's site does.

The LLM is never trusted with the words themselves. Every paragraph it returns
is checked: once the markup is stripped back out, the remaining words must be
exactly the words that went in, and the tags must be well formed. A paragraph
that fails is retried and then left as the user wrote it, so the worst outcome
is "no tags added here", never "the model rewrote my script".

The LLM backend is pluggable: a local ``transformers`` model (downloaded
through the model store) or any OpenAI-compatible endpoint such as Ollama or LM
Studio. Only the validation logic is imported at module load, so this file is
testable without the ML stack.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .drama_markup import PAIRED_TAGS, strip_drama_markup

logger = logging.getLogger(__name__)

# Paragraphs longer than this are split on sentence ends before being sent, so
# one LLM call stays short and a failure only costs a small piece of the text.
MAX_PIECE_CHARS = 900
MAX_ATTEMPTS = 2

SYSTEM_PROMPT = """You are a voice director preparing a script for the Fish Audio Drama 3 text-to-speech model. You add performance markup to the script. You never change, add, remove, reorder or correct any of the script's own words or punctuation.

Markup you may add:
1. Line directions in square brackets, written in plain English the way you would speak to an actor, placed before the words they apply to. A direction stays in force until the next one. Examples: [warm, unhurried] [smug, drawing it out] [voice cracking, close to tears] [flat and clipped, barely hiding anger]
2. Sounds and beats in square brackets at the exact point they happen: [sigh] [laugh] [breath] [pause] [gasp]
3. Paired tags around a few chosen words, only these seven: <whisper></whisper> <emphasis></emphasis> <soft></soft> <fast></fast> <slow></slow> <stress></stress> <forceful></forceful>

Rules:
- Copy every word and punctuation mark of the script exactly, in order. Only insert markup.
- Start with a line direction that fits the speaker and the moment.
- Add a new direction only where the emotion or delivery actually shifts.
- Use sounds and paired tags sparingly: at most one or two per sentence, and none where the text does not call for them.
- Keep any markup that is already present.
- Output only the marked-up script. No explanations, no quotation marks around it, no code fences."""

_EXAMPLE_IN = (
    "Oh, look at you, counting your coins like it's rent day. Wait... you have "
    "more than me? No, no, this is not fair, give it back right now!"
)
_EXAMPLE_OUT = (
    "[smug, drawing it out] Oh, look at you, counting your coins like it's rent day. "
    "[pause] [thrown, suddenly unsure] Wait... you have <emphasis>more</emphasis> than me? "
    "[furious, stamping tantrum] No, no, this is not fair, give it back "
    "<forceful>right now!</forceful>"
)


class LLM(Protocol):
    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str: ...


def build_messages(piece: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _EXAMPLE_IN},
        {"role": "assistant", "content": _EXAMPLE_OUT},
        {"role": "user", "content": piece},
    ]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_ANY_ANGLE_TAG_RE = re.compile(r"</?([A-Za-z][A-Za-z0-9_-]*)[^<>]*>")
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def spoken_words(text: str) -> list[str]:
    """Words of a script with all markup removed, normalised for comparison."""
    plain = strip_drama_markup(text).replace("’", "'").lower()
    return _WORD_RE.findall(plain)


def clean_llm_output(text: str) -> str:
    """Peel off the wrappers chat models like to add around an answer."""
    out = (text or "").strip()
    out = _FENCE_RE.sub("", out).strip()
    # A reply wrapped whole in one pair of quotes.
    if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'" and out.count(out[0]) == 2:
        out = out[1:-1].strip()
    return re.sub(r"[ \t]+", " ", out)


def validate(original: str, enhanced: str) -> str | None:
    """Return None if ``enhanced`` is a safe marked-up version of ``original``.

    Otherwise returns the reason it was rejected.
    """
    if not enhanced.strip():
        return "empty reply"

    for match in _ANY_ANGLE_TAG_RE.finditer(enhanced):
        if match.group(1).lower() not in PAIRED_TAGS:
            return f"unsupported tag <{match.group(1)}>"

    stack: list[str] = []
    for match in re.finditer(r"<(/?)([A-Za-z]+)>", enhanced):
        name = match.group(2).lower()
        if not match.group(1):
            stack.append(name)
        elif not stack or stack[-1] != name:
            return f"unbalanced </{name}>"
        else:
            stack.pop()
    if stack:
        return f"unclosed <{stack[-1]}>"

    if enhanced.count("[") != enhanced.count("]"):
        return "unbalanced square brackets"

    before, after = spoken_words(original), spoken_words(enhanced)
    if before != after:
        # Point at the first divergence; it makes the log useful.
        for i, (a, b) in enumerate(zip(before, after)):
            if a != b:
                return f"words changed at word {i + 1}: '{a}' became '{b}'"
        return f"word count changed ({len(before)} -> {len(after)})"

    if not after:
        return "no spoken words"
    return None


def split_pieces(text: str, limit: int = MAX_PIECE_CHARS) -> list[tuple[str, str]]:
    """Split text into (piece, separator-that-followed-it) pairs.

    Paragraph breaks are preserved exactly; an over-long paragraph is cut at
    sentence ends so each piece fits comfortably in one LLM call.
    """
    pieces: list[tuple[str, str]] = []
    parts = re.split(r"(\n\s*\n)", text)
    for i in range(0, len(parts), 2):
        paragraph = parts[i]
        separator = parts[i + 1] if i + 1 < len(parts) else ""
        if len(paragraph) <= limit:
            pieces.append((paragraph, separator))
            continue
        sentences = re.split(r"(?<=[.!?…])\s+", paragraph)
        current = ""
        for sentence in sentences:
            if current and len(current) + 1 + len(sentence) > limit:
                pieces.append((current, " "))
                current = sentence
            else:
                current = f"{current} {sentence}".strip() if current else sentence
        pieces.append((current, separator))
    return pieces


# ---------------------------------------------------------------------------
# Enhancer
# ---------------------------------------------------------------------------

@dataclass
class EnhanceResult:
    text: str
    pieces: int = 0
    enhanced: int = 0
    unchanged: int = 0
    tags_added: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "pieces": self.pieces,
            "enhanced": self.enhanced,
            "unchanged": self.unchanged,
            "tags_added": self.tags_added,
            "notes": self.notes,
        }


def _count_tags(text: str) -> int:
    return len(re.findall(r"\[[^\[\]]+\]", text)) + len(re.findall(r"<[A-Za-z]+>", text))


class TagEnhancer:
    def __init__(self, llm: LLM, max_attempts: int = MAX_ATTEMPTS):
        self.llm = llm
        self.max_attempts = max_attempts

    def enhance(self, text: str) -> EnhanceResult:
        result = EnhanceResult(text=text)
        out: list[str] = []
        for piece, separator in split_pieces(text):
            if not spoken_words(piece):
                out.append(piece + separator)
                continue
            result.pieces += 1
            marked = self._enhance_piece(piece.strip(), result)
            if marked is None:
                result.unchanged += 1
                out.append(piece + separator)
            else:
                result.enhanced += 1
                result.tags_added += max(0, _count_tags(marked) - _count_tags(piece))
                out.append(marked + separator)
        result.text = "".join(out)
        return result

    def _enhance_piece(self, piece: str, result: EnhanceResult) -> str | None:
        # Room for the words plus generous markup.
        budget = min(2048, int(len(piece) / 2.5) + 256)
        reason = "no attempt made"
        for attempt in range(self.max_attempts):
            try:
                reply = self.llm.generate(build_messages(piece), max_new_tokens=budget)
            except Exception as e:  # an LLM failure must not lose the script
                logger.error(f"Tag enhancer LLM failed: {e}")
                result.notes.append(f"The language model failed: {e}")
                return None
            candidate = clean_llm_output(reply)
            reason = validate(piece, candidate)
            if reason is None:
                return candidate
            logger.info(f"Tag enhancer attempt {attempt + 1} rejected: {reason}")
        preview = piece[:60] + ("…" if len(piece) > 60 else "")
        result.notes.append(f"Left unchanged ({reason}): “{preview}”")
        return None


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

class TransformersLLM:
    """A local chat model loaded with ``transformers`` (lazily, on first use)."""

    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "bfloat16"):
        self.model_path = str(model_path)
        self.device = device
        self.dtype = dtype
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available for the tag enhancer; using CPU (slow)")
            device = "cpu"
        dtype = getattr(torch, self.dtype) if device != "cpu" else torch.float32
        logger.info(f"Loading tag-enhancer LLM from {self.model_path}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=dtype)
        self._model.to(device)
        self._model.eval()
        self._device = device
        logger.info("Tag-enhancer LLM ready")

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        import torch

        with self._lock:
            if self._model is None:
                self._load()
            inputs = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(self._device)
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_tokens = output[0][inputs["input_ids"].shape[-1]:]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._tokenizer = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


class OpenAICompatLLM:
    """Any server speaking the OpenAI chat-completions API (Ollama, LM Studio, ...)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 300.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._opener = opener

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        url = self.base_url + ("" if self.base_url.endswith("/chat/completions") else "/chat/completions")
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": 0.6,
                "stream": False,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with self._opener(request, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"LLM endpoint returned HTTP {e.code}") from e
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"Could not reach the LLM endpoint at {url}: {e}") from e
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError("Unexpected response from the LLM endpoint") from e
