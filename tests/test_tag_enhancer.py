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


# --- the user's tagging notes, and thinking models -------------------------

def test_guidance_is_added_to_the_system_prompt_and_capped():
    from longecho.tag_enhancer import MAX_GUIDANCE_CHARS, SYSTEM_PROMPT, build_messages

    plain = build_messages("Hello there.")
    assert plain[0]["content"] == SYSTEM_PROMPT

    noted = build_messages("Hello there.", "  The speaker is a smug vampire. Use whispers often. ")
    system = noted[0]["content"]
    assert system.startswith(SYSTEM_PROMPT)
    assert system.endswith("The speaker is a smug vampire. Use whispers often.")
    assert "never text to add to the script" in system
    # The script itself is still the last user turn, untouched.
    assert noted[-1] == {"role": "user", "content": "Hello there."}

    long = build_messages("Hi.", "x" * (MAX_GUIDANCE_CHARS + 500))[0]["content"]
    assert long.count("x" * MAX_GUIDANCE_CHARS) == 1 and "x" * (MAX_GUIDANCE_CHARS + 1) not in long


def test_guidance_reaches_the_llm_but_cannot_change_the_words():
    from longecho.tag_enhancer import TagEnhancer

    seen = []

    class Obedient:
        def generate(self, messages, max_new_tokens):
            seen.append(messages[0]["content"])
            # Follows a note that asked it to add a line: that must be rejected.
            return "[warm] Hello there. And goodbye."

    result = TagEnhancer(Obedient()).enhance("Hello there.", guidance="End every line with 'And goodbye.'")
    assert "And goodbye" in seen[0]
    assert result.text == "Hello there." and result.unchanged == 1


def test_thinking_is_removed_before_validation():
    from longecho.tag_enhancer import clean_llm_output, strip_reasoning, validate

    reply = "<think>\nThe speaker sounds tired, so a soft direction fits.\n</think>\n\n[tired] Hello there."
    assert clean_llm_output(reply) == "[tired] Hello there."
    assert validate("Hello there.", clean_llm_output(reply)) is None
    # Template opened the block, so only the closing tag is in the reply.
    assert strip_reasoning("still deciding...</think>[calm] Hi.").strip() == "[calm] Hi."
    # Cut off mid-thought: there is no answer to keep.
    assert strip_reasoning("<think>I should consider").strip() == ""


def test_openai_backend_sends_extra_fields_for_llama_cpp():
    import io, json as _json
    from longecho.tag_enhancer import LLAMA_REQUEST_EXTRAS, OpenAICompatLLM

    sent = {}

    def opener(request, timeout=None):
        sent.update(_json.loads(request.data.decode()))
        return io.BytesIO(_json.dumps({"choices": [{"message": {"content": "[calm] Hi."}}]}).encode())

    llm = OpenAICompatLLM("http://127.0.0.1:8090/v1", "candyecho-enhancer", opener=opener,
                          extra_body=LLAMA_REQUEST_EXTRAS)
    assert llm.generate([{"role": "user", "content": "Hi."}], 64) == "[calm] Hi."
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["top_k"] == 20 and "presence_penalty" not in sent


class _FakeServer:
    def __init__(self):
        self.healthy, self.starts, self.stops = False, 0, 0
        self.config = type("C", (), {"url": "http://127.0.0.1:8090"})()

    def is_healthy(self):
        return self.healthy

    def wait_ready(self):
        self.starts += 1
        self.healthy = True

    def stop(self):
        self.stops += 1
        self.healthy = False


class _FakeTimer:
    made = []

    def __init__(self, seconds, fn):
        self.seconds, self.fn, self.cancelled, self.daemon = seconds, fn, False, False
        _FakeTimer.made.append(self)

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True


def test_managed_server_starts_on_first_use_and_stops_when_idle():
    from longecho.tag_enhancer import ManagedLlamaLLM

    _FakeTimer.made.clear()

    class Client:
        def generate(self, messages, max_new_tokens):
            return "[calm] Hi."

    server = _FakeServer()
    llm = ManagedLlamaLLM(server, idle_seconds=600, client=Client(), timer=_FakeTimer)
    assert not llm.loaded

    assert llm.generate([], 10) == "[calm] Hi."
    assert server.starts == 1 and llm.loaded
    llm.generate([], 10)
    assert server.starts == 1                       # already up: not started again
    # Each use re-arms the idle timer and cancels the previous one.
    assert len(_FakeTimer.made) == 2 and _FakeTimer.made[0].cancelled and not _FakeTimer.made[1].cancelled
    assert _FakeTimer.made[1].seconds == 600

    _FakeTimer.made[1].fn()                         # ten quiet minutes later
    assert server.stops == 1 and not llm.loaded
    llm.generate([], 10)                            # and it comes back on demand
    assert server.starts == 2

    llm.unload()
    assert server.stops == 2 and _FakeTimer.made[-1].cancelled


def test_idle_stop_can_be_switched_off():
    from longecho.tag_enhancer import ManagedLlamaLLM

    _FakeTimer.made.clear()

    class Client:
        def generate(self, messages, max_new_tokens):
            return "x"

    ManagedLlamaLLM(_FakeServer(), idle_seconds=0, client=Client(), timer=_FakeTimer).generate([], 1)
    assert _FakeTimer.made == []
