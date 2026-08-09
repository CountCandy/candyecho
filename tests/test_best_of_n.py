"""Tests for best-of-N take generation and selection inside AudioGenerator.

The diffusion model, autoencoder and cropping are mocked; what is under test is
the batching (N takes from one sampler call), the selection wiring, the retry
escalation, and — most importantly — that the *winning* take is what seeds the
next chunk's continuation.
"""
import sys
from unittest.mock import Mock, patch

import pytest
import torch

sys.modules.setdefault('inference', Mock())
sys.modules.setdefault('inference_blockwise', Mock())

from longecho.audio_generator import AudioGenerator  # noqa: E402
from longecho.transcript_verifier import CandidateSelector  # noqa: E402

REFERENCE = (
    "the council met in the middle of the crisis and agreed to proceed with the "
    "emergency levy on salt, which the members had debated for several weeks"
)
# Size the fake audio to the duration the scorer expects from this much text,
# so the duration penalty stays at zero and does not trigger spurious retries.
LATENTS = int((len(REFERENCE) / 13.0) * 44100 / 2048)


class OrderedTranscriber:
    """Returns canned transcripts in call order, which matches candidate order."""

    def __init__(self, name, transcripts):
        self.name = name
        self.transcripts = list(transcripts)
        self.calls = 0

    def transcribe(self, audio, sample_rate):
        text = self.transcripts[min(self.calls, len(self.transcripts) - 1)]
        self.calls += 1
        return text


def make_selector(transcripts, accept_threshold=0.10):
    return CandidateSelector(
        [OrderedTranscriber("fake", transcripts)], accept_threshold=accept_threshold
    )


@pytest.fixture
def mocked_echo():
    """Patch the vendored inference calls with batch-aware fakes."""
    with patch('longecho.audio_generator.get_text_input_ids_and_mask') as text, \
         patch('longecho.audio_generator.sample_blockwise_euler_cfg_independent_guidances') as sample, \
         patch('longecho.audio_generator.ae_decode') as decode, \
         patch('longecho.audio_generator.crop_audio_to_flattening_point') as crop, \
         patch('longecho.audio_generator.get_speaker_latent_and_mask') as speaker:

        def _text(texts, **kwargs):
            n = len(texts)
            return torch.zeros(n, 32), torch.ones(n, 32, dtype=torch.bool)

        def _sample(**kwargs):
            # Mirror the real shape: continuation prefix plus the new block, so
            # trimming the continuation leaves LATENTS worth of new audio.
            n = kwargs['text_input_ids'].shape[0]
            continuation = kwargs.get('continuation_latent')
            prefix = continuation.shape[1] if continuation is not None else 0
            return torch.randn(n, prefix + LATENTS, 80)

        def _decode(fish_ae, pca_state, latents):
            return torch.randn(latents.shape[0], 1, latents.shape[1] * 2048)

        text.side_effect = _text
        sample.side_effect = _sample
        decode.side_effect = _decode
        crop.side_effect = lambda audio, latent: audio
        speaker.return_value = (torch.randn(1, 200, 80), torch.ones(1, 200, dtype=torch.bool))

        yield {"text": text, "sample": sample, "decode": decode, "crop": crop, "speaker": speaker}


@pytest.fixture
def generator():
    model = Mock()
    model.device = "cpu"
    return AudioGenerator(model, Mock(), Mock())


@pytest.fixture
def voice():
    return torch.randn(1, 200, 80), torch.ones(1, 200, dtype=torch.bool)


# --- batching --------------------------------------------------------------

class TestBatching:
    def test_candidates_share_one_sampler_call(self, generator, mocked_echo, voice):
        """N takes must cost one diffusion pass, not N of them."""
        latent, mask = voice
        _, gen = generator.generate_long_audio(
            [REFERENCE], latent, mask,
            num_candidates=3, selector=make_selector([REFERENCE] * 3),
        )
        list(gen)
        assert mocked_echo["sample"].call_count == 1

    def test_text_is_batched_to_candidate_count(self, generator, mocked_echo, voice):
        latent, mask = voice
        _, gen = generator.generate_long_audio(
            [REFERENCE], latent, mask,
            num_candidates=3, selector=make_selector([REFERENCE] * 3),
        )
        list(gen)
        texts = mocked_echo["text"].call_args[0][0]
        assert len(texts) == 3
        assert len(set(texts)) == 1  # same text, independent noise per row

    def test_conditioning_expanded_to_batch(self, generator, mocked_echo, voice):
        latent, mask = voice
        _, gen = generator.generate_long_audio(
            [REFERENCE], latent, mask,
            num_candidates=3, selector=make_selector([REFERENCE] * 3),
        )
        list(gen)
        kwargs = mocked_echo["sample"].call_args[1]
        assert kwargs["speaker_latent"].shape[0] == 3
        assert kwargs["speaker_mask"].shape[0] == 3

    def test_each_take_cropped_independently(self, generator, mocked_echo, voice):
        """Takes end at different flattening points; one shared crop would truncate."""
        latent, mask = voice
        _, gen = generator.generate_long_audio(
            [REFERENCE], latent, mask,
            num_candidates=3, selector=make_selector([REFERENCE] * 3),
        )
        list(gen)
        assert mocked_echo["crop"].call_count == 3

    def test_without_verification_stays_single_take(self, generator, mocked_echo, voice):
        """No selector -> previous behaviour exactly, one take per chunk."""
        latent, mask = voice
        _, gen = generator.generate_long_audio([REFERENCE], latent, mask, num_candidates=3)
        chunks = list(gen)
        assert len(chunks) == 1
        assert mocked_echo["text"].call_args[0][0] == [REFERENCE]
        assert mocked_echo["crop"].call_count == 1


# --- selection -------------------------------------------------------------

class TestSelection:
    def test_yields_the_winning_take(self, generator, mocked_echo, voice):
        """Take 2 is the accurate one; its audio must be what gets yielded."""
        latent, mask = voice
        selector = make_selector(["totally different words", REFERENCE, "wrong again"])
        _, gen = generator.generate_long_audio(
            [REFERENCE], latent, mask, num_candidates=3, selector=selector,
        )
        chunks = list(gen)
        assert len(chunks) == 1
        assert chunks[0].shape[0] == 1  # a single take, not the batch

    def test_selection_reported_via_callback(self, generator, mocked_echo, voice):
        latent, mask = voice
        seen = []
        selector = make_selector(["wrong", REFERENCE, "wrong"])
        _, gen = generator.generate_long_audio(
            [REFERENCE], latent, mask, num_candidates=3, selector=selector,
            on_selection=lambda idx, payload: seen.append((idx, payload)),
        )
        list(gen)

        assert len(seen) == 1
        chunk_index, payload = seen[0]
        assert chunk_index == 0
        assert payload["winner"] == 1
        assert len(payload["candidates"]) == 3

    def test_winner_seeds_the_next_continuation(self, generator, mocked_echo, voice):
        """The whole point: a caught error must not propagate to the next chunk."""
        latent, mask = voice
        selector = make_selector(["wrong words here", REFERENCE, "also wrong"] * 2)
        _, gen = generator.generate_long_audio(
            [REFERENCE, "a second chunk of text"], latent, mask,
            num_candidates=3, selector=selector,
        )
        list(gen)

        # Continuation is extracted once, between the two chunks, from the
        # winning take rather than from take 0.
        assert mocked_echo["speaker"].call_count == 1

    def test_verification_failure_falls_back_to_first_take(self, generator, mocked_echo, voice):
        """A broken selector must not abort a long generation."""
        latent, mask = voice
        broken = Mock()
        broken.select.side_effect = RuntimeError("verifier exploded")

        _, gen = generator.generate_long_audio(
            [REFERENCE], latent, mask, num_candidates=3, selector=broken,
        )
        chunks = list(gen)
        assert len(chunks) == 1


# --- retry escalation ------------------------------------------------------

class TestEscalation:
    def test_good_first_round_does_not_retry(self, generator, mocked_echo, voice):
        latent, mask = voice
        selector = make_selector([REFERENCE] * 3)
        _, gen = generator.generate_long_audio(
            [REFERENCE], latent, mask,
            num_candidates=3, selector=selector, max_rounds=3,
        )
        list(gen)
        assert mocked_echo["sample"].call_count == 1

    def test_poor_takes_trigger_another_round(self, generator, mocked_echo, voice):
        """Every take is bad, so a second batch is generated automatically."""
        latent, mask = voice
        selector = make_selector(["nothing like the text at all"] * 12)
        _, gen = generator.generate_long_audio(
            [REFERENCE], latent, mask,
            num_candidates=3, selector=selector, max_rounds=2,
        )
        list(gen)
        assert mocked_echo["sample"].call_count == 2

    def test_escalation_is_capped(self, generator, mocked_echo, voice):
        latent, mask = voice
        selector = make_selector(["nothing like the text at all"] * 30)
        _, gen = generator.generate_long_audio(
            [REFERENCE], latent, mask,
            num_candidates=2, selector=selector, max_rounds=3,
        )
        list(gen)
        assert mocked_echo["sample"].call_count == 3

    def test_later_rounds_use_different_seeds(self, generator, mocked_echo, voice):
        """A retry with the same seed would redraw identical noise."""
        latent, mask = voice
        selector = make_selector(["nothing like the text at all"] * 12)
        _, gen = generator.generate_long_audio(
            [REFERENCE], latent, mask,
            num_candidates=2, selector=selector, max_rounds=2,
        )
        list(gen)
        seeds = [c[1]["rng_seed"] for c in mocked_echo["sample"].call_args_list]
        assert len(set(seeds)) == len(seeds)

    def test_chunk_seeds_do_not_collide_across_rounds(self, generator, mocked_echo, voice):
        """Chunk stride must exceed the room retries can consume."""
        latent, mask = voice
        selector = make_selector(["nothing like the text at all"] * 40)
        _, gen = generator.generate_long_audio(
            [REFERENCE, "second chunk", "third chunk"], latent, mask,
            num_candidates=2, selector=selector, max_rounds=3,
        )
        list(gen)
        seeds = [c[1]["rng_seed"] for c in mocked_echo["sample"].call_args_list]
        assert len(set(seeds)) == len(seeds)
