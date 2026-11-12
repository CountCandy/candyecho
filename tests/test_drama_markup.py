"""Drama 3 markup helpers."""
import pytest

from longecho.drama_markup import is_sound_cue, prepare_drama_chunks, strip_drama_markup


@pytest.mark.parametrize("body", ["sigh", "pause", "shaky breath", "laughs softly", "clears throat"])
def test_sound_cues(body):
    assert is_sound_cue(body)


@pytest.mark.parametrize(
    "body",
    ["calm", "warm, steady", "upbeat", "humble and quiet", "breathless, rushing",
     "barely holding it together, forcing a smile"],
)
def test_directions_are_not_sound_cues(body):
    assert not is_sound_cue(body)


def test_direction_is_carried_into_later_chunks():
    out = prepare_drama_chunks(["[calm] One.", "Two.", "[sigh] Three."])
    assert out == ["[calm] One.", "[calm] Two.", "[calm] [sigh] Three."]


def test_new_direction_replaces_the_old_one():
    out = prepare_drama_chunks(["[calm] One. [furious] Two.", "Three.", "[gentle] Four.", "Five."])
    assert out[1] == "[furious] Three."
    assert out[2] == "[gentle] Four."
    assert out[3] == "[gentle] Five."


def test_paired_tag_spanning_chunks_is_closed_and_reopened():
    out = prepare_drama_chunks(["A <whisper>b", "c</whisper> d.", "e."])
    assert out == ["A <whisper>b</whisper>", "<whisper>c</whisper> d.", "e."]


def test_plain_text_is_untouched():
    chunks = ["No markup here.", "None here either."]
    assert prepare_drama_chunks(chunks) == chunks


def test_strip_markup_leaves_only_spoken_words():
    text = "[smug] Oh, <emphasis>darling</emphasis>. [pause] Really?"
    assert strip_drama_markup(text) == "Oh, darling. Really?"


# --- protecting markup through the text pipeline -------------------------

from longecho.drama_markup import protect_markup


def test_protect_and_restore_round_trips_exactly():
    text = "[teasing] Is <soft>'dumpster chic'</soft> having a moment? [sigh] Yes."
    masked, restore = protect_markup(text)
    assert "<" not in masked and "[" not in masked
    assert restore(masked) == text


def test_markup_survives_cleaning_and_normalizing():
    """Regression: with textbook cleaning on, <soft> was spelled out and spoken."""
    from longecho.text_cleaner import clean_text
    from longecho.text_normalizer import TextNormalizer
    from longecho.text_segmenter import segment_text

    text = (
        '[teasing, sarcastic] "Excuse me, did you get dressed in the dark, or is '
        "<soft>'dumpster chic'</soft> actually having a moment?\""
    )
    masked, restore = protect_markup(text)
    processed = TextNormalizer().normalize(clean_text(masked, preset="textbook").text)
    chunks = [restore(c) for c in segment_text(processed)]
    joined = " ".join(chunks)
    assert "less than" not in joined and "greater than" not in joined
    assert "[teasing, sarcastic]" in joined
    assert "<soft>'dumpster chic'</soft>" in joined


def test_restore_works_per_chunk_after_segmentation():
    from longecho.text_segmenter import segment_text

    text = "[calm] " + "This is a sentence that goes on for a while. " * 8 + "<whisper>The end.</whisper>"
    masked, restore = protect_markup(text)
    chunks = [restore(c) for c in segment_text(masked)]
    assert len(chunks) > 1
    assert chunks[0].startswith("[calm] This is")
    assert chunks[-1].endswith("<whisper>The end.</whisper>")
    assert not any("zqx" in c for c in chunks)


def test_numeric_brackets_are_left_for_the_cleaner():
    masked, restore = protect_markup("As shown [12] and [calm] here.")
    assert "[12]" in masked
    assert restore(masked) == "As shown [12] and [calm] here."


def test_many_tags_get_distinct_placeholders():
    text = " ".join(f"[mood {chr(97 + i % 26)}{chr(97 + i // 26)}] word." for i in range(60))
    masked, restore = protect_markup(text)
    assert restore(masked) == text
