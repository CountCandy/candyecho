"""Tests for textbook/book text cleaning.

The fixture in ``fixtures/textbook_page_extract.txt`` is synthetic, but every
structural pattern in it was taken from a real page-extracted textbook: page
bodies as single long lines, sentences severed at page boundaries, print
furniture interleaved between pages, flattened footnote digits, and compounds
that lost their hyphen during de-hyphenation.
"""
from pathlib import Path

import pytest

from longecho.text_cleaner import (
    PRESETS,
    RULES,
    CleanResult,
    clean_text,
    resolve_rules,
)

FIXTURE = Path(__file__).parent / "fixtures" / "textbook_page_extract.txt"


@pytest.fixture
def extract() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture
def cleaned(extract: str) -> CleanResult:
    return clean_text(extract)


# --- rule registry ---------------------------------------------------------

class TestRuleRegistry:
    def test_presets_only_reference_known_rules(self):
        names = {r.name for r in RULES}
        for preset, rules in PRESETS.items():
            assert rules <= names, f"preset {preset} references unknown rules"

    def test_rule_names_are_unique(self):
        names = [r.name for r in RULES]
        assert len(names) == len(set(names))

    def test_resolve_defaults_to_textbook(self):
        assert resolve_rules() == PRESETS["textbook"]

    def test_resolve_applies_overrides(self):
        rules = resolve_rules("textbook", {"reflow": False})
        assert "reflow" not in rules
        assert "unicode" in rules

    def test_resolve_ignores_unknown_rule_names(self):
        assert resolve_rules("light", {"not_a_rule": True}) == PRESETS["light"]

    def test_off_preset_is_a_passthrough(self, extract):
        result = clean_text(extract, preset="off")
        assert result.text == extract
        assert result.report == []


# --- print furniture -------------------------------------------------------

class TestFurnitureRemoval:
    def test_typesetter_stamps_removed(self, cleaned):
        assert ".indd" not in cleaned.text
        assert "7714_004" not in cleaned.text

    def test_timestamps_removed(self, cleaned):
        assert "3:47:12" not in cleaned.text
        assert "11/02/2018" not in cleaned.text

    def test_proof_notice_removed(self, cleaned):
        assert "PROPERTY OF THE EXAMPLE PRESS" not in cleaned.text
        assert "PROOFREADING" not in cleaned.text

    def test_running_header_removed(self, cleaned):
        # 'Chapter 4' is reprinted at the top of every verso page.
        assert "Chapter 4" not in cleaned.text

    def test_margin_artifact_removed(self, cleaned):
        # The stray two-letter 'Se' mark repeats on every page.
        assert not any(line.strip() == "Se" for line in cleaned.text.split("\n"))

    def test_running_header_page_numbers_removed(self, cleaned):
        assert "Revenue 3" not in cleaned.text
        assert "Revenue 5" not in cleaned.text

    def test_body_prose_survives(self, cleaned):
        assert "harvests began to fail across the northern provinces" in cleaned.text
        assert "the political weather had shifted as well" in cleaned.text


# --- reflow ----------------------------------------------------------------

class TestReflow:
    def test_sentence_split_across_page_boundary_is_rejoined(self, cleaned):
        # 'Once in control' ended one page; 'of the chamber, though' began the next.
        assert "Once in control of the chamber, though" in cleaned.text

    def test_no_full_stop_inserted_at_the_page_break(self, cleaned):
        assert "Once in control." not in cleaned.text
        assert "Once in control," not in cleaned.text

    def test_hard_wrapped_lines_are_rejoined(self):
        wrapped = (
            "The mitochondrion is the primary site\n"
            "of ATP synthesis in eukaryotic cells."
        )
        result = clean_text(wrapped, preset="light")
        assert "primary site of ATP synthesis" in result.text

    def test_hyphenated_line_break_is_repaired(self):
        result = clean_text("an inter-\nnational agreement", preset="light")
        assert "international agreement" in result.text

    def test_real_paragraph_break_is_preserved(self):
        text = "First paragraph ends here.\n\nSecond paragraph starts here."
        result = clean_text(text, preset="light")
        assert "\n\n" in result.text

    def test_new_sentence_after_break_is_not_joined(self):
        text = "The first sentence is complete.\nA second one follows."
        result = clean_text(text, preset="light")
        assert "complete. A second" in result.text or "complete.\nA second" in result.text


# --- inline cleanup --------------------------------------------------------

class TestInlineCleanup:
    def test_footnote_digits_removed(self, cleaned):
        assert "credits.2" not in cleaned.text
        assert "seasons.3" not in cleaned.text
        assert "and credits. The joint panel" in cleaned.text

    def test_decimals_are_not_mistaken_for_footnotes(self, cleaned):
        assert "4.0 percent" in cleaned.text
        assert "18.1 percent" in cleaned.text
        assert "8.5 percent" in cleaned.text

    def test_bracketed_citation_removed(self, cleaned):
        assert "[14]" not in cleaned.text

    def test_author_year_citation_removed(self, cleaned):
        assert "Alderton" not in cleaned.text

    def test_url_removed(self, cleaned):
        assert "https://example.org" not in cleaned.text

    def test_curly_quotes_folded(self, cleaned):
        assert "’" not in cleaned.text
        assert "“" not in cleaned.text
        assert "bloc's promises" in cleaned.text


class TestBrokenCompounds:
    def test_repairs_lost_hyphens(self, cleaned):
        assert "tax-cutting" in cleaned.text
        assert "upper-income" in cleaned.text
        assert "low and middle-income" in cleaned.text

    @pytest.mark.parametrize(
        "word",
        ["taxpayers", "Understanding", "overrun", "crosscutting", "interest"],
    )
    def test_does_not_split_real_words(self, word):
        result = clean_text(f"The {word} matter here.", preset="textbook")
        assert word in result.text

    def test_real_words_in_fixture_survive(self, cleaned):
        assert "taxpayers facing the highest marginal rates" in cleaned.text
        assert "Understanding the full effect" in cleaned.text


# --- figures and tables ----------------------------------------------------

class TestFiguresAndTables:
    def test_caption_is_kept(self, cleaned):
        assert "Share of total receipts by source" in cleaned.text

    def test_table_rows_are_dropped(self, cleaned):
        assert "41.2" not in cleaned.text
        assert "39.8" not in cleaned.text

    def test_source_credit_dropped(self, cleaned):
        assert "reproduced with permission" not in cleaned.text

    def test_prose_after_the_table_resumes(self, cleaned):
        assert "The pattern in the figure is consistent" in cleaned.text


# --- reporting -------------------------------------------------------------

class TestReport:
    def test_report_counts_removals(self, cleaned):
        rules = {r.rule: r.count for r in cleaned.report}
        assert rules.get("typesetter_stamps", 0) >= 6
        assert rules.get("running_headers", 0) >= 3
        assert rules.get("reflow", 0) >= 1

    def test_report_carries_samples(self, cleaned):
        for removal in cleaned.report:
            assert removal.count > 0
            assert len(removal.samples) <= 6

    def test_as_dict_is_json_shaped(self, cleaned):
        payload = cleaned.as_dict()
        assert set(payload) == {"text", "chars_before", "chars_after", "report"}
        assert all(set(r) == {"rule", "label", "count", "samples"} for r in payload["report"])

    def test_cleaning_shrinks_the_text(self, cleaned):
        assert cleaned.chars_after < cleaned.chars_before


# --- substitutions and edge cases ------------------------------------------

class TestSubstitutionsAndEdges:
    def test_pronunciation_dictionary_applied(self):
        result = clean_text("The GDP deflator rose.", substitutions={"GDP": "G D P"})
        assert "G D P deflator" in result.text

    def test_substitution_is_case_insensitive(self):
        result = clean_text("the gdp figure", substitutions={"GDP": "G D P"})
        assert "G D P figure" in result.text

    @pytest.mark.parametrize("text", ["", "   ", "\n\n"])
    def test_empty_input_is_returned_unchanged(self, text):
        assert clean_text(text).text == text

    def test_plain_prose_is_left_alone(self):
        prose = "This is an ordinary sentence. So is this one."
        assert clean_text(prose).text == prose

    def test_dot_letters_spaced(self):
        result = clean_text("A book by J.R.R. Tolkien.")
        assert "J R R" in result.text

    def test_symbols_spoken(self):
        result = clean_text("The value is ≤ 5 percent.")
        assert "less than or equal to" in result.text
