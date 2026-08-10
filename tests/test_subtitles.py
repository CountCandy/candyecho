"""Tests for SRT/VTT generation from chunk timings."""
import re

import pytest

from longecho.subtitles import Cue, build_cues, split_sentences, to_srt, to_vtt


class TestSplitSentences:
    def test_splits_on_sentence_boundaries(self):
        assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_keeps_short_text_whole(self):
        assert split_sentences("A single sentence") == ["A single sentence"]

    def test_breaks_overlong_sentence_at_clauses(self):
        long = (
            "the council met in the middle of the crisis, and agreed to proceed "
            "with the emergency levy, which the members had debated for weeks"
        )
        parts = split_sentences(long)
        assert len(parts) > 1
        assert all(len(p) <= 120 for p in parts)

    def test_empty(self):
        assert split_sentences("") == []


class TestBuildCues:
    def test_timings_follow_chunk_durations(self):
        cues = build_cues([("First chunk here.", 4.0), ("Second chunk here.", 6.0)])
        assert cues[0].start == 0.0
        assert cues[-1].end == pytest.approx(10.0)

    def test_chunk_boundaries_stay_exact(self):
        """Rounding must be absorbed inside a chunk, never leak across it."""
        cues = build_cues([("One. Two. Three.", 3.0), ("Four.", 1.0)])
        boundary = [c for c in cues if c.text.startswith("Four")][0]
        assert boundary.start == pytest.approx(3.0)

    def test_time_split_by_character_share(self):
        cues = build_cues([("Short. " + "x" * 100 + ".", 10.0)])
        assert cues[1].duration > cues[0].duration

    def test_cues_are_sequential_and_non_overlapping(self):
        cues = build_cues([("One. Two. Three.", 6.0), ("Four. Five.", 4.0)])
        for a, b in zip(cues, cues[1:]):
            assert a.end <= b.start + 1e-6

    def test_indices_are_contiguous_from_one(self):
        cues = build_cues([("One. Two. Three.", 6.0)])
        assert [c.index for c in cues] == list(range(1, len(cues) + 1))

    def test_silent_chunk_still_advances_the_clock(self):
        cues = build_cues([("", 5.0), ("After the gap.", 2.0)])
        assert cues[0].start == pytest.approx(5.0)

    def test_zero_duration_chunk_skipped(self):
        cues = build_cues([("Nothing generated.", 0.0), ("Real audio.", 2.0)])
        assert len(cues) == 1
        assert "Real" in cues[0].text

    def test_empty_input(self):
        assert build_cues([]) == []

    def test_long_chunk_is_broken_up(self):
        """A single very long cue is unreadable; it gets divided."""
        text = " ".join(["word"] * 120) + "."
        cues = build_cues([(text, 30.0)])
        assert len(cues) > 1
        assert all(c.duration <= 12.0 for c in cues)


class TestSrt:
    def test_format(self):
        srt = to_srt(build_cues([("Hello there.", 2.0)]))
        assert srt.startswith("1\n")
        assert "00:00:00,000 --> 00:00:02,000" in srt
        assert "Hello there." in srt

    def test_timestamps_use_comma(self):
        srt = to_srt(build_cues([("Hi.", 1.5)]))
        assert re.search(r"\d{2}:\d{2}:\d{2},\d{3}", srt)

    def test_hours_render(self):
        srt = to_srt([Cue(1, 3661.5, 3663.0, "Late.")])
        assert "01:01:01,500" in srt

    def test_empty_cue_list(self):
        assert to_srt([]) == ""


class TestVtt:
    def test_has_header(self):
        assert to_vtt(build_cues([("Hi.", 1.0)])).startswith("WEBVTT")

    def test_timestamps_use_period(self):
        vtt = to_vtt(build_cues([("Hi.", 1.5)]))
        assert re.search(r"\d{2}:\d{2}:\d{2}\.\d{3}", vtt)

    def test_empty_cue_list_still_valid(self):
        assert to_vtt([]).strip() == "WEBVTT"
