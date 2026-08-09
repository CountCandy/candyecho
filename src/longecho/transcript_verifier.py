"""Pick the best of several generated takes by listening back to them.

Echo occasionally drops or invents words. Generating a chunk N times and
transcribing each take with ASR lets us keep the one that actually says what the
text said. This matters more here than in a stateless pipeline: the winning take
is re-encoded into the continuation latent that seeds the *next* chunk, so a bad
take does not just sound wrong, it poisons everything after it.

Design notes:

* **Two ASR families, rank-averaged.** WhisperD (``jordand/whisper-d-v1a``) is
  fine-tuned on the same transcription format Echo was trained on, so it renders
  ``[S1]`` tags, disfluencies and ``(laughs)`` natively instead of scoring them
  as errors. A CTC model (wav2vec2 / HuBERT) is a genuinely different
  architecture whose errors decorrelate, and structurally it cannot emit the
  repetition loops that attention decoders hallucinate -- which is exactly the
  failure mode we are hunting. Ranking is averaged across families rather than
  trusting one verifier's absolute numbers.

* **Deletions are weighted above substitutions.** A missing word is the failure
  we care about; a substitution is often just the ASR mishearing.

* **WER is not enough.** Duration outliers, repetition loops and truncated
  endings are all real failures that leave word error untouched, so they are
  scored separately.

* **Absolute scores are noisy; rankings are not.** Any systematic disagreement
  between the reference text and an ASR's conventions (numbers especially) hits
  every candidate equally and cancels out when comparing them. Only the escalate
  threshold reads the absolute value, which is why it is generous by default.

The ASR backends need ``transformers``; install with the ``verify`` extra. Torch
is imported lazily so the scoring logic below is pure standard library: it can
be unit-tested without the CUDA stack or any model weights.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import torch

logger = logging.getLogger(__name__)

__all__ = [
    "Candidate",
    "CandidateScore",
    "SelectionResult",
    "Transcriber",
    "CandidateSelector",
    "WhisperTranscriber",
    "CTCTranscriber",
    "SpeakerSimilarity",
    "build_selector",
    "normalize_for_scoring",
    "word_error",
]

# Reference sample rate of everything Echo emits.
SAMPLE_RATE = 44_100
# Every ASR we use expects 16 kHz mono.
ASR_SAMPLE_RATE = 16_000

# Weighted word error. Deletions hurt most: dropped words are the failure the
# whole feature exists to catch.
DELETION_WEIGHT = 1.5
INSERTION_WEIGHT = 1.25
SUBSTITUTION_WEIGHT = 1.0

# Composite score weights.
CER_WEIGHT = 0.25
SPEAKER_WEIGHT = 0.5
DURATION_WEIGHT = 0.5
REPETITION_PENALTY = 0.4
TRUNCATION_PENALTY = 0.3

# Rough speech rate used to sanity-check a take's length (matches the segmenter).
CHARS_PER_SECOND = 13.0
# How far a take may deviate from its expected duration before it is penalised.
DURATION_TOLERANCE = 0.35


# ---------------------------------------------------------------------------
# Text normalization for scoring
# ---------------------------------------------------------------------------

_SPEAKER_TAG = re.compile(r"\[\s*S\d+\s*\]", re.IGNORECASE)
# WhisperD annotates non-speech events in parentheses; they are not words.
_NON_SPEECH = re.compile(r"\([^)]*\)")
_BRACKETED = re.compile(r"\[[^\]]*\]")
_PUNCT = re.compile(r"[^\w\s']")
_WHITESPACE = re.compile(r"\s+")

# Filler tokens WhisperD transcribes verbatim but that carry no content. They
# are dropped from both sides so a take is not punished for a natural "uh".
_FILLERS = frozenset({"uh", "um", "erm", "mm", "hmm", "ah", "eh", "er"})

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")]


def _int_to_words(num: int) -> str:
    """Spell an integer so both sides of a comparison agree on number wording."""
    if num < 0:
        return "minus " + _int_to_words(-num)
    if num < 20:
        return _ONES[num]
    if num < 100:
        rest = num % 10
        return _TENS[num // 10] + (" " + _ONES[rest] if rest else "")
    if num < 1000:
        rest = num % 100
        return _ONES[num // 100] + " hundred" + (" " + _int_to_words(rest) if rest else "")
    for scale, name in _SCALES:
        if num >= scale:
            rest = num % scale
            return _int_to_words(num // scale) + f" {name}" + (
                " " + _int_to_words(rest) if rest else ""
            )
    return str(num)


def _expand_numbers(text: str) -> str:
    def repl(m: re.Match) -> str:
        digits = m.group(0).replace(",", "")
        try:
            return _int_to_words(int(digits))
        except (ValueError, IndexError):
            return m.group(0)

    return re.sub(r"\d[\d,]*", repl, text)


def normalize_for_scoring(text: str) -> list[str]:
    """Reduce text to comparable word tokens.

    Applied identically to the reference and to every hypothesis, so speaker
    tags, non-speech annotations, punctuation, casing, fillers and number
    formatting cannot masquerade as word errors.
    """
    if not text:
        return []
    text = _SPEAKER_TAG.sub(" ", text)
    text = _NON_SPEECH.sub(" ", text)
    text = _BRACKETED.sub(" ", text)
    text = text.lower()
    text = _expand_numbers(text)
    text = text.replace("-", " ")
    text = _PUNCT.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return [t for t in text.split(" ") if t and t not in _FILLERS]


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

@dataclass
class ErrorCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0
    reference_length: int = 0

    @property
    def weighted_rate(self) -> float:
        if self.reference_length == 0:
            return 0.0 if self.insertions == 0 else 1.0
        weighted = (
            SUBSTITUTION_WEIGHT * self.substitutions
            + DELETION_WEIGHT * self.deletions
            + INSERTION_WEIGHT * self.insertions
        )
        return weighted / self.reference_length

    def as_dict(self) -> dict:
        return {
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "hits": self.hits,
            "reference_length": self.reference_length,
        }


def word_error(reference: Sequence[str], hypothesis: Sequence[str]) -> ErrorCounts:
    """Levenshtein alignment over tokens, returning the individual edit counts.

    Written out rather than pulled from ``jiwer`` because the composite score
    weights deletions and insertions differently, which needs the raw operation
    counts rather than a single rate.
    """
    ref, hyp = list(reference), list(hypothesis)
    n, m = len(ref), len(hyp)
    if n == 0:
        return ErrorCounts(insertions=m, reference_length=0)
    if m == 0:
        return ErrorCounts(deletions=n, reference_length=n)

    # dp[i][j] = (cost, subs, dels, ins, hits)
    dp: list[list[tuple[int, int, int, int, int]]] = [
        [(0, 0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)
    ]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                c, s, d, ins, h = dp[i - 1][j - 1]
                dp[i][j] = (c, s, d, ins, h + 1)
                continue
            sub_c, sub_s, sub_d, sub_i, sub_h = dp[i - 1][j - 1]
            del_c, del_s, del_d, del_i, del_h = dp[i - 1][j]
            ins_c, ins_s, ins_d, ins_i, ins_h = dp[i][j - 1]
            best = min(sub_c, del_c, ins_c)
            if best == sub_c:
                dp[i][j] = (sub_c + 1, sub_s + 1, sub_d, sub_i, sub_h)
            elif best == del_c:
                dp[i][j] = (del_c + 1, del_s, del_d + 1, del_i, del_h)
            else:
                dp[i][j] = (ins_c + 1, ins_s, ins_d, ins_i + 1, ins_h)

    _, subs, dels, ins, hits = dp[n][m]
    return ErrorCounts(
        substitutions=subs, deletions=dels, insertions=ins, hits=hits, reference_length=n
    )


def character_error(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    """Character error rate over the normalized token stream, used as a tiebreak."""
    ref, hyp = " ".join(reference), " ".join(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def repetition_score(tokens: Sequence[str], n: int = 3, threshold: int = 3) -> float:
    """Detect a stuck decoder repeating the same phrase.

    Returns the number of consecutive repeats beyond ``threshold``, as a
    fraction, so an occasional legitimate repeat scores zero.
    """
    if len(tokens) < n * threshold:
        return 0.0
    worst = 0
    i = 0
    while i + n <= len(tokens):
        gram = tuple(tokens[i:i + n])
        repeats = 1
        j = i + n
        while j + n <= len(tokens) and tuple(tokens[j:j + n]) == gram:
            repeats += 1
            j += n
        worst = max(worst, repeats)
        i += 1 if repeats == 1 else (j - i)
    return max(0.0, (worst - threshold + 1) / threshold) if worst >= threshold else 0.0


def truncation_score(reference: Sequence[str], hypothesis: Sequence[str], tail: int = 4) -> float:
    """Fraction of the reference's final words missing from the hypothesis tail.

    Catches a take that stops mid-sentence, which barely moves overall WER on a
    long chunk but is very audible.
    """
    if len(reference) < tail or not hypothesis:
        return 0.0
    ref_tail = reference[-tail:]
    hyp_tail = set(hypothesis[-(tail * 3):])
    missing = sum(1 for w in ref_tail if w not in hyp_tail)
    return missing / tail


# ---------------------------------------------------------------------------
# Candidates and scoring
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """One generated take of a single text chunk."""

    index: int
    seed: int
    audio: Any  # torch.Tensor [1, channels, samples] as produced by AudioGenerator
    sample_rate: int = SAMPLE_RATE

    @property
    def duration(self) -> float:
        return self.audio.shape[-1] / float(self.sample_rate)


@dataclass
class CandidateScore:
    index: int
    seed: int
    score: float = 0.0
    rank: float = 0.0
    wer: float = 0.0
    cer: float = 0.0
    duration: float = 0.0
    speaker_similarity: float | None = None
    penalties: dict[str, float] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)
    transcripts: dict[str, str] = field(default_factory=dict)
    per_verifier: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "seed": self.seed,
            "score": round(self.score, 4),
            "wer": round(self.wer, 4),
            "cer": round(self.cer, 4),
            "duration": round(self.duration, 2),
            "speaker_similarity": (
                round(self.speaker_similarity, 4)
                if self.speaker_similarity is not None else None
            ),
            "penalties": {k: round(v, 4) for k, v in self.penalties.items() if v},
            "errors": self.errors,
        }


@dataclass
class SelectionResult:
    winner: int
    scores: list[CandidateScore]
    acceptable: bool

    @property
    def best(self) -> CandidateScore:
        return next(s for s in self.scores if s.index == self.winner)

    def as_dict(self) -> dict:
        return {
            "winner": self.winner,
            "acceptable": self.acceptable,
            "candidates": [s.as_dict() for s in self.scores],
        }


class Transcriber(Protocol):
    """Anything that can turn a mono waveform into text."""

    name: str

    def transcribe(self, audio: Any, sample_rate: int) -> str:
        ...


def _to_mono_16k(audio: Any, sample_rate: int) -> Any:
    """Flatten to a 1-D 16 kHz mono waveform, which every ASR here expects."""
    import torch

    wave = audio.detach().to(torch.float32).cpu()
    while wave.dim() > 1:
        wave = wave.mean(dim=0)
    if sample_rate != ASR_SAMPLE_RATE:
        import torchaudio

        wave = torchaudio.functional.resample(wave, sample_rate, ASR_SAMPLE_RATE)
    return wave


class CandidateSelector:
    """Scores generated takes and picks the one that best matches the text."""

    def __init__(
        self,
        transcribers: Sequence[Transcriber],
        speaker_similarity: "SpeakerSimilarity | None" = None,
        accept_threshold: float = 0.10,
    ):
        if not transcribers:
            raise ValueError("CandidateSelector needs at least one transcriber")
        self.transcribers = list(transcribers)
        self.speaker_similarity = speaker_similarity
        self.accept_threshold = accept_threshold

    def select(
        self,
        candidates: Sequence[Candidate],
        reference_text: str,
        reference_audio: Any = None,
    ) -> SelectionResult:
        """Score every candidate and return the winner.

        Each transcriber ranks the candidates independently and the ranks are
        averaged, so no single ASR's idiosyncrasies decide the outcome.
        """
        if not candidates:
            raise ValueError("no candidates to select from")

        reference = normalize_for_scoring(reference_text)
        expected_duration = max(len(reference_text) / CHARS_PER_SECOND, 0.5)

        scores = [
            CandidateScore(index=c.index, seed=c.seed, duration=c.duration)
            for c in candidates
        ]
        by_index = {s.index: s for s in scores}

        # Per-transcriber composite, kept separately so ranks can be averaged.
        per_verifier: dict[str, dict[int, float]] = {}

        for transcriber in self.transcribers:
            verdicts: dict[int, float] = {}
            for candidate in candidates:
                entry = by_index[candidate.index]
                try:
                    text = transcriber.transcribe(candidate.audio, candidate.sample_rate)
                except Exception as e:
                    logger.warning(
                        f"{transcriber.name} failed on candidate {candidate.index}: {e}"
                    )
                    # A transcriber that cannot read a take must not hand it a
                    # winning score by default.
                    verdicts[candidate.index] = 1.0
                    continue

                hypothesis = normalize_for_scoring(text)
                counts = word_error(reference, hypothesis)
                wer = counts.weighted_rate
                cer = character_error(reference, hypothesis)
                composite = wer + CER_WEIGHT * cer

                entry.transcripts[transcriber.name] = text.strip()
                entry.per_verifier[transcriber.name] = round(composite, 4)
                # Report the metrics from the first transcriber that ran, so the
                # UI has a concrete WER rather than an ensemble abstraction.
                if not entry.errors:
                    entry.wer = wer
                    entry.cer = cer
                    entry.errors = counts.as_dict()

                repeats = repetition_score(hypothesis)
                truncated = truncation_score(reference, hypothesis)
                if repeats:
                    entry.penalties["repetition"] = max(
                        entry.penalties.get("repetition", 0.0), REPETITION_PENALTY * repeats
                    )
                if truncated:
                    entry.penalties["truncation"] = max(
                        entry.penalties.get("truncation", 0.0), TRUNCATION_PENALTY * truncated
                    )
                verdicts[candidate.index] = composite
            per_verifier[transcriber.name] = verdicts

        # Duration outliers: a take far off its expected length is rushed,
        # dragging or looping. WER alone will not always catch it.
        for candidate in candidates:
            entry = by_index[candidate.index]
            deviation = abs(candidate.duration - expected_duration) / expected_duration
            if deviation > DURATION_TOLERANCE:
                entry.penalties["duration"] = DURATION_WEIGHT * (deviation - DURATION_TOLERANCE)

        # Speaker drift: says the right words in the wrong voice.
        if self.speaker_similarity is not None and reference_audio is not None:
            for candidate in candidates:
                entry = by_index[candidate.index]
                try:
                    sim = self.speaker_similarity.compare(
                        reference_audio, candidate.audio, candidate.sample_rate
                    )
                except Exception as e:
                    logger.warning(f"speaker similarity failed: {e}")
                    continue
                entry.speaker_similarity = sim
                entry.penalties["speaker"] = SPEAKER_WEIGHT * max(0.0, 1.0 - sim)

        # Fold the penalties into each verifier's composite *before* ranking.
        # Duration, repetition, truncation and speaker drift are ASR-independent
        # signals, so they shift every verifier's view of a candidate equally --
        # but they have to be inside the ranking, or a take that every ASR reads
        # perfectly wins on word accuracy alone while being three times too long
        # or in the wrong voice.
        penalty_totals = {
            c.index: sum(by_index[c.index].penalties.values()) for c in candidates
        }

        rank_totals: dict[int, float] = {c.index: 0.0 for c in candidates}
        for verdicts in per_verifier.values():
            adjusted = {idx: verdicts[idx] + penalty_totals[idx] for idx in verdicts}
            ordered = sorted(adjusted, key=lambda idx: (adjusted[idx], idx))
            for rank, idx in enumerate(ordered):
                rank_totals[idx] += rank

        num_verifiers = max(1, len(per_verifier))
        for candidate in candidates:
            entry = by_index[candidate.index]
            entry.rank = rank_totals[candidate.index] / num_verifiers
            mean_composite = sum(
                v[candidate.index] for v in per_verifier.values()
            ) / num_verifiers
            entry.score = mean_composite + penalty_totals[candidate.index]

        # Mean rank is the primary key -- it is what makes the ensemble robust to
        # any single ASR's quirks. The composite score breaks ties within a rank.
        winner = min(scores, key=lambda s: (s.rank, s.score, s.index))
        scores.sort(key=lambda s: (s.rank, s.score, s.index))
        return SelectionResult(
            winner=winner.index,
            scores=scores,
            acceptable=winner.score <= self.accept_threshold,
        )


# ---------------------------------------------------------------------------
# ASR backends (require `transformers`)
# ---------------------------------------------------------------------------

def _require_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "Take verification needs the 'transformers' package, which should "
            "be installed automatically. Run 'uv sync' to repair the environment."
        ) from e


class WhisperTranscriber:
    """Whisper-family ASR.

    Defaults to WhisperD, which was fine-tuned on the transcription format Echo
    was trained against: it emits ``[S1]`` speaker tags, disfluencies and
    ``(laughs)``-style non-speech events. Those are stripped before scoring, but
    a model that renders them faithfully misreads far less of Echo's output than
    a stock Whisper does.
    """

    def __init__(
        self,
        model_id: str = "jordand/whisper-d-v1a",
        device: str = "cuda",
        dtype: str = "float16",
        name: str | None = None,
    ):
        _require_transformers()
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.name = name or model_id.rsplit("/", 1)[-1]
        self.device = device
        self.dtype = getattr(torch, dtype)
        logger.info(f"Loading ASR verifier '{self.name}' ({model_id})...")
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=self.dtype)
            .to(device)
            .eval()
        )

    def transcribe(self, audio: Any, sample_rate: int) -> str:
        import torch

        with torch.inference_mode():
            wave = _to_mono_16k(audio, sample_rate)
            features = self.processor(
                wave.numpy(), sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt"
            ).input_features.to(self.device, self.dtype)
            tokens = self.model.generate(features, language="en", task="transcribe")
            return self.processor.batch_decode(tokens, skip_special_tokens=True)[0]


class CTCTranscriber:
    """CTC ASR (wav2vec2 / HuBERT) used as the cross-family verifier.

    A CTC model emits one output per frame with no autoregressive decoder, so it
    physically cannot produce the runaway repetition that attention decoders
    hallucinate. When it and Whisper disagree about a take, that disagreement is
    informative rather than correlated noise.
    """

    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-large-960h-lv60-self",
        device: str = "cuda",
        dtype: str = "float32",
        name: str | None = None,
    ):
        _require_transformers()
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.name = name or model_id.rsplit("/", 1)[-1]
        self.device = device
        self.dtype = getattr(torch, dtype)
        logger.info(f"Loading ASR verifier '{self.name}' ({model_id})...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForCTC.from_pretrained(model_id, torch_dtype=self.dtype).to(device).eval()
        )

    def transcribe(self, audio: Any, sample_rate: int) -> str:
        import torch

        with torch.inference_mode():
            wave = _to_mono_16k(audio, sample_rate)
            inputs = self.processor(
                wave.numpy(), sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt"
            )
            values = inputs.input_values.to(self.device, self.dtype)
            logits = self.model(values).logits
            ids = torch.argmax(logits, dim=-1)
            return self.processor.batch_decode(ids)[0]


class SpeakerSimilarity:
    """Cosine similarity between speaker embeddings of the reference and a take.

    Guards against a candidate that says every word correctly in a voice that
    has drifted away from the reference -- Echo's known long-form weakness, and
    something word error is completely blind to.
    """

    def __init__(
        self,
        model_id: str = "microsoft/wavlm-base-plus-sv",
        device: str = "cuda",
    ):
        _require_transformers()
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.device = device
        logger.info(f"Loading speaker-similarity model ({model_id})...")
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = WavLMForXVector.from_pretrained(model_id).to(device).eval()

    def _embed(self, audio: Any, sample_rate: int) -> Any:
        import torch

        with torch.inference_mode():
            wave = _to_mono_16k(audio, sample_rate)
            inputs = self.extractor(
                wave.numpy(), sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt"
            )
            values = inputs.input_values.to(self.device)
            return self.model(values).embeddings[0]

    def compare(self, reference: Any, candidate: Any, sample_rate: int) -> float:
        import torch

        ref = self._embed(reference, sample_rate)
        cand = self._embed(candidate, sample_rate)
        sim = torch.nn.functional.cosine_similarity(ref, cand, dim=0)
        # Cosine runs [-1, 1]; rescale so the penalty term stays well behaved.
        return float((sim + 1.0) / 2.0)


def build_selector(
    whisper_model: str | None = "jordand/whisper-d-v1a",
    ctc_model: str | None = "facebook/wav2vec2-large-960h-lv60-self",
    speaker_model: str | None = "microsoft/wavlm-base-plus-sv",
    device: str = "cuda",
    accept_threshold: float = 0.10,
) -> CandidateSelector:
    """Build the default cross-family selector. Pass ``None`` to skip a model.

    Each model is loaded independently and a failure is logged and skipped
    rather than raised: one unreachable repo or a renamed model id should
    degrade verification, not abort a multi-hour book. Only a total failure --
    no transcriber at all -- is fatal, since there would be nothing to rank with.
    """
    transcribers: list[Transcriber] = []
    failures: list[str] = []

    for label, factory in (
        (whisper_model, lambda m: WhisperTranscriber(m, device=device)),
        (ctc_model, lambda m: CTCTranscriber(m, device=device)),
    ):
        if not label:
            continue
        try:
            transcribers.append(factory(label))
        except Exception as e:
            failures.append(f"{label}: {e}")
            logger.error(f"Could not load ASR verifier '{label}': {e}")

    if not transcribers:
        raise RuntimeError(
            "No ASR verifier could be loaded, so takes cannot be scored. "
            + (" | ".join(failures) if failures else "No verifier models configured.")
        )
    if len(transcribers) == 1:
        logger.warning(
            f"Only one ASR verifier loaded ({transcribers[0].name}); "
            "cross-family ranking is disabled for this session."
        )

    similarity = None
    if speaker_model:
        try:
            similarity = SpeakerSimilarity(speaker_model, device=device)
        except Exception as e:
            logger.error(
                f"Could not load speaker-similarity model '{speaker_model}': {e}. "
                "Continuing without the speaker-drift check."
            )

    return CandidateSelector(transcribers, similarity, accept_threshold=accept_threshold)
