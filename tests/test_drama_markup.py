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
