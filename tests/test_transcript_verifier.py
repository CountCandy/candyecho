"""Tests for best-of-N candidate scoring and selection.

These exercise the scoring logic only -- no ASR weights and no torch. The
transcribers are fakes returning canned text, which is the point: the selection
policy is what decides whether a dropped word gets caught, and it should be
verifiable without a GPU.
"""
import pytest

from longecho.transcript_verifier import (
    Candidate,
    CandidateSelector,
    character_error,
    normalize_for_scoring,
    repetition_score,
    truncation_score,
    word_error,
)

REFERENCE = "The council met and agreed to proceed with the emergency levy on salt"


class FakeAudio:
    """Stands in for a torch tensor: selection only ever reads .shape."""

    def __init__(self, seconds: float, sample_rate: int = 44100):
        self.shape = (1, 1, int(seconds * sample_rate))


class FakeTranscriber:
    """Returns canned text per candidate index."""

    def __init__(self, name, transcripts, fails_on=()):
        self.name = name
        self.transcripts = transcripts
        self.fails_on = set(fails_on)
        self.calls = 0

    def transcribe(self, audio, sample_rate):
        self.calls += 1
        idx = getattr(audio, "candidate_index", None)
        if idx in self.fails_on:
            raise RuntimeError("transcription blew up")
        return self.transcripts[idx]


def make_candidate(index, seconds=5.0, seed=None):
    audio = FakeAudio(seconds)
    audio.candidate_index = index
    return Candidate(index=index, seed=seed if seed is not None else 1000 + index, audio=audio)


def build(transcripts_by_verifier, fails_on=(), **kwargs):
    transcribers = [
        FakeTranscriber(name, texts, fails_on=fails_on)
        for name, texts in transcripts_by_verifier.items()
    ]
    return CandidateSelector(transcribers, **kwargs)


# --- normalization ---------------------------------------------------------

class TestNormalization:
    def test_strips_speaker_tags(self):
        assert "s1" not in normalize_for_scoring("[S1] hello there")

    def test_strips_non_speech_events(self):
        # WhisperD emits these natively; they are not spoken words.
        assert normalize_for_scoring("hello (laughs) there") == ["hello", "there"]

    def test_strips_fillers(self):
        assert normalize_for_scoring("well uh yes um fine") == ["well", "yes", "fine"]

    def test_case_and_punctuation_insensitive(self):
        assert normalize_for_scoring("Hello, World!") == normalize_for_scoring("hello world")

    def test_expands_numbers_consistently(self):
        assert normalize_for_scoring("2016") == normalize_for_scoring("two thousand sixteen")

    def test_expands_numbers_with_separators(self):
        assert normalize_for_scoring("250,000") == normalize_for_scoring("two hundred fifty thousand")

    def test_hyphenated_words_split(self):
        assert normalize_for_scoring("upper-income") == ["upper", "income"]

    def test_empty(self):
        assert normalize_for_scoring("") == []


# --- error metrics ---------------------------------------------------------

class TestWordError:
    def test_identical_is_zero(self):
        ref = normalize_for_scoring(REFERENCE)
        assert word_error(ref, ref).weighted_rate == 0.0

    def test_counts_each_operation(self):
        counts = word_error(["a", "b", "c"], ["a", "x", "c"])
        assert (counts.substitutions, counts.deletions, counts.insertions) == (1, 0, 0)

        counts = word_error(["a", "b", "c"], ["a", "c"])
        assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 1, 0)

        counts = word_error(["a", "c"], ["a", "b", "c"])
        assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 0, 1)

    def test_deletions_weighted_above_substitutions(self):
        """A dropped word is the failure this feature exists to catch."""
        ref = ["a", "b", "c", "d"]
        deletion = word_error(ref, ["a", "c", "d"]).weighted_rate
        substitution = word_error(ref, ["a", "x", "c", "d"]).weighted_rate
        assert deletion > substitution

    def test_empty_hypothesis_is_all_deletions(self):
        counts = word_error(["a", "b"], [])
        assert counts.deletions == 2

    def test_empty_reference_with_hypothesis(self):
        counts = word_error([], ["a", "b"])
        assert counts.insertions == 2
        assert counts.weighted_rate == 1.0

    def test_both_empty(self):
        assert word_error([], []).weighted_rate == 0.0


class TestOtherMetrics:
    def test_character_error_identical(self):
        assert character_error(["hello"], ["hello"]) == 0.0

    def test_character_error_detects_near_miss(self):
        assert 0 < character_error(["hello"], ["helo"]) < 0.5

    def test_repetition_detects_stuck_decoder(self):
        assert repetition_score(["a", "b", "c"] * 5) > 0

    def test_repetition_ignores_normal_text(self):
        assert repetition_score(normalize_for_scoring(REFERENCE)) == 0.0

    def test_truncation_detects_missing_tail(self):
        ref = normalize_for_scoring(REFERENCE)
        assert truncation_score(ref, ref[:-4]) > 0

    def test_truncation_zero_for_complete(self):
        ref = normalize_for_scoring(REFERENCE)
        assert truncation_score(ref, ref) == 0.0


# --- selection -------------------------------------------------------------

class TestSelection:
    def test_picks_the_accurate_take(self):
        selector = build({"whisper": {
            0: "The council met and agreed to with the emergency levy on salt",  # dropped words
            1: REFERENCE,                                                        # perfect
            2: "The council met and agreed to proceed with the emergency tax on salt",
        }})
        result = selector.select([make_candidate(i) for i in range(3)], REFERENCE)
        assert result.winner == 1

    def test_perfect_take_is_acceptable(self):
        selector = build({"whisper": {0: REFERENCE}})
        result = selector.select([make_candidate(0)], REFERENCE)
        assert result.acceptable is True
        assert result.best.wer == 0.0

    def test_bad_take_is_not_acceptable(self):
        selector = build({"whisper": {0: "completely different words entirely"}})
        result = selector.select([make_candidate(0)], REFERENCE)
        assert result.acceptable is False

    def test_rank_ensemble_uses_both_verifiers(self):
        """Candidate 1 is not either verifier's favourite but wins on mean rank."""
        selector = build({
            "whisper": {
                0: REFERENCE,                                                  # rank 0
                1: "The council met and agreed to proceed with the emergency levy on sale",
                2: "utterly wrong text here",                                  # rank 2
            },
            "ctc": {
                0: "utterly wrong text here",                                   # rank 2
                1: "The council met and agreed to proceed with the emergency levy on sale",
                2: REFERENCE,                                                   # rank 0
            },
        })
        result = selector.select([make_candidate(i) for i in range(3)], REFERENCE)
        assert result.winner == 1

    def test_both_verifiers_are_consulted(self):
        selector = build({"whisper": {0: REFERENCE}, "ctc": {0: REFERENCE}})
        selector.select([make_candidate(0)], REFERENCE)
        assert all(t.calls == 1 for t in selector.transcribers)

    def test_failed_transcription_does_not_win(self):
        """A take the ASR could not read must not be handed a default victory."""
        selector = build(
            {"whisper": {0: REFERENCE, 1: REFERENCE}},
            fails_on=(1,),
        )
        result = selector.select([make_candidate(0), make_candidate(1)], REFERENCE)
        assert result.winner == 0

    def test_duration_outlier_is_penalised(self):
        """Same transcript, but one take runs far too long -- dragging or looping."""
        selector = build({"whisper": {0: REFERENCE, 1: REFERENCE}})
        expected = len(REFERENCE) / 13.0
        normal = make_candidate(0, seconds=expected)
        overlong = make_candidate(1, seconds=expected * 3)
        result = selector.select([normal, overlong], REFERENCE)

        assert result.winner == 0
        loser = next(s for s in result.scores if s.index == 1)
        assert loser.penalties.get("duration", 0) > 0

    def test_repetition_is_penalised(self):
        selector = build({"whisper": {
            0: REFERENCE,
            1: REFERENCE + " on salt on salt on salt on salt",
        }})
        result = selector.select([make_candidate(0), make_candidate(1)], REFERENCE)
        assert result.winner == 0

    def test_scores_carry_transcripts_and_errors(self):
        selector = build({"whisper": {0: "The council met"}})
        result = selector.select([make_candidate(0)], REFERENCE)
        best = result.best
        assert best.transcripts["whisper"] == "The council met"
        assert best.errors["deletions"] > 0
        assert best.seed == 1000

    def test_result_serializes_for_sse(self):
        selector = build({"whisper": {0: REFERENCE, 1: "wrong"}})
        payload = selector.select([make_candidate(0), make_candidate(1)], REFERENCE).as_dict()
        assert set(payload) == {"winner", "acceptable", "candidates"}
        assert len(payload["candidates"]) == 2
        for entry in payload["candidates"]:
            assert {"index", "seed", "score", "wer", "cer"} <= set(entry)

    def test_scores_returned_best_first(self):
        selector = build({"whisper": {0: "wrong entirely", 1: REFERENCE}})
        result = selector.select([make_candidate(0), make_candidate(1)], REFERENCE)
        assert result.scores[0].index == result.winner


class TestSpeakerSimilarity:
    class FakeSimilarity:
        def __init__(self, values):
            self.values = values

        def compare(self, reference, candidate, sample_rate):
            return self.values[candidate.candidate_index]

    def test_drifted_voice_loses_despite_matching_words(self):
        """Both takes say the right words; one has drifted off the reference voice."""
        selector = build(
            {"whisper": {0: REFERENCE, 1: REFERENCE}},
            speaker_similarity=self.FakeSimilarity({0: 0.55, 1: 0.98}),
        )
        result = selector.select(
            [make_candidate(0), make_candidate(1)], REFERENCE, reference_audio=FakeAudio(3.0)
        )
        assert result.winner == 1
        assert result.best.speaker_similarity == 0.98

    def test_similarity_skipped_without_reference_audio(self):
        selector = build(
            {"whisper": {0: REFERENCE}},
            speaker_similarity=self.FakeSimilarity({0: 0.1}),
        )
        result = selector.select([make_candidate(0)], REFERENCE)
        assert result.best.speaker_similarity is None
        assert "speaker" not in result.best.penalties


class TestGuards:
    def test_requires_a_transcriber(self):
        with pytest.raises(ValueError):
            CandidateSelector([])

    def test_requires_candidates(self):
        selector = build({"whisper": {}})
        with pytest.raises(ValueError):
            selector.select([], REFERENCE)

    def test_single_candidate_still_scored(self):
        selector = build({"whisper": {0: REFERENCE}})
        result = selector.select([make_candidate(0)], REFERENCE)
        assert result.winner == 0
        assert len(result.scores) == 1
