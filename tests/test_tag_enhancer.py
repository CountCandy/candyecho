"""Tag enhancer: the LLM may add markup but never change the words."""
import json

import pytest

from longecho import tag_enhancer as te


class ScriptedLLM:
    """Returns canned replies in order (the last one repeats)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, messages, max_new_tokens):
        self.calls.append(messages[-1]["content"])
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_built_in_example_is_valid():
    assert te.validate(te._EXAMPLE_IN, te._EXAMPLE_OUT) is None


@pytest.mark.parametrize(
    "enhanced, fragment",
    [
        ("[warm] Hello friend.", "words changed"),
        ("[warm] Hello.", "word count"),
        ("[warm] Hello there. And more.", "word count"),
        ("<b>Hello there.</b>", "unsupported tag"),
        ("<whisper>Hello there.", "unclosed"),
        ("Hello</whisper> there.", "unbalanced"),
        ("[warm Hello there.", "square brackets"),
        ("", "empty"),
    ],
)
def test_validate_rejects_unsafe_output(enhanced, fragment):
    reason = te.validate("Hello there.", enhanced)
    assert reason is not None and fragment in reason


def test_validate_accepts_markup_only_changes():
    assert te.validate(
        "Hello there. Come closer.",
        "[warm] Hello there. [pause] <whisper>Come closer.</whisper>",
    ) is None


def test_enhance_applies_valid_markup():
    llm = ScriptedLLM("[warm] Hello there.")
    result = te.TagEnhancer(llm).enhance("Hello there.")
    assert result.text == "[warm] Hello there."
    assert (result.pieces, result.enhanced, result.unchanged, result.tags_added) == (1, 1, 0, 1)


def test_rewritten_words_are_rejected_and_original_kept():
    llm = ScriptedLLM("[warm] Hi there, my friend!")
    result = te.TagEnhancer(llm, max_attempts=2).enhance("Hello there.")
    assert result.text == "Hello there."
    assert result.unchanged == 1 and result.enhanced == 0
    assert len(llm.calls) == 2  # retried once, then gave up
    assert "Left unchanged" in result.notes[0]


def test_retry_can_succeed():
    llm = ScriptedLLM("Sure! Here you go: Hello there.", "[calm] Hello there.")
    result = te.TagEnhancer(llm, max_attempts=2).enhance("Hello there.")
    assert result.text == "[calm] Hello there."


def test_paragraph_breaks_survive_and_failures_are_local():
    llm = ScriptedLLM("[bright] First part.", "totally different words", "totally different words")
    text = "First part.\n\nSecond part."
    result = te.TagEnhancer(llm, max_attempts=2).enhance(text)
    assert result.text == "[bright] First part.\n\nSecond part."
    assert (result.enhanced, result.unchanged) == (1, 1)


def test_llm_failure_never_loses_the_script():
    llm = ScriptedLLM(RuntimeError("out of memory"))
    result = te.TagEnhancer(llm).enhance("Keep me safe.")
    assert result.text == "Keep me safe."
    assert "out of memory" in result.notes[0]


def test_code_fences_and_quotes_are_peeled_off():
    assert te.clean_llm_output('```text\n[calm] Hi.\n```') == "[calm] Hi."
    assert te.clean_llm_output('"[calm] Hi."') == "[calm] Hi."
    # Quotes that belong to the script are left alone.
    assert te.clean_llm_output('"Hi," she said, "bye."') == '"Hi," she said, "bye."'


def test_long_paragraph_is_split_at_sentence_ends():
    paragraph = " ".join(f"Sentence number {i} goes here." for i in range(80))
    pieces = te.split_pieces(paragraph, limit=300)
    assert len(pieces) > 1
    assert all(len(p) <= 300 for p, _sep in pieces)
    assert " ".join(p for p, _sep in pieces) == paragraph


def test_existing_markup_is_counted_not_double_counted():
    llm = ScriptedLLM("[calm] Hello [pause] there.")
    result = te.TagEnhancer(llm).enhance("[calm] Hello there.")
    assert result.tags_added == 1


def test_openai_compatible_backend_builds_a_chat_request():
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "[calm] Hi."}}]}).encode()

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        seen["auth"] = request.get_header("Authorization")
        return Resp()

    llm = te.OpenAICompatLLM("http://localhost:11434/v1", "qwen", api_key="k", opener=opener)
    assert llm.generate([{"role": "user", "content": "Hi."}], 64) == "[calm] Hi."
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["body"]["model"] == "qwen" and seen["body"]["max_tokens"] == 64
    assert seen["auth"] == "Bearer k"
