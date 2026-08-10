import logging
import threading
import time
from typing import Callable, List, Generator, Tuple, Any

import torch

from .transcript_verifier import Candidate

from longecho._vendor.echo_tts import (
    get_text_input_ids_and_mask,
    ae_decode,
    crop_audio_to_flattening_point,
    get_speaker_latent_and_mask,
    sample_blockwise_euler_cfg_independent_guidances,
)

logger = logging.getLogger(__name__)

# Generation parameters from design
DEFAULT_PARAMS = {
    "num_steps": 40,
    "cfg_scale_text": 3.0,
    "cfg_scale_speaker": 8.0,
    "cfg_min_t": 0.5,
    "cfg_max_t": 1.0,
    "truncation_factor": 1.0,
    "rescale_k": 1.0,
    "rescale_sigma": 3.0,
    "speaker_kv_scale": None,
    "speaker_kv_max_layers": None,
    "speaker_kv_min_t": None,
}

# Max latents for continuation - if previous chunk exceeds this, we truncate
# With smaller chunks (~12-14s), full chunk should be ~280-320 latents
MAX_CONTINUATION_LATENTS = 400

# Seed layout for best-of-N. Chunks are spaced far enough apart that adding a
# round or a candidate can never collide with a neighbouring chunk's seeds.
SEED_STRIDE_CHUNK = 1000
SEED_STRIDE_ROUND = 100

# When "Force Speaker" KV scaling is enabled, the scaling is undone once the
# noise level falls below this point. Matches cfg_min_t: hold the speaker during
# the high-noise phase that fixes identity, then release it.
DEFAULT_SPEAKER_KV_MIN_T = 0.5


class AudioGenerator:
    """
    Generates long-form audio by chunking text and maintaining context.

    Uses Echo's blockwise inference to generate audio chunks with
    continuation from previous chunks for coherence.
    """

    def __init__(self, model: Any, fish_ae: Any, pca_state: Any):
        """
        Initialize audio generator.

        Args:
            model: Echo DiT model
            fish_ae: Fish autoencoder
            pca_state: PCA state
        """
        self.model = model
        self.fish_ae = fish_ae
        self.pca_state = pca_state
        self.device = model.device
        self._generation_id = 0  # Unique ID for each generation
        self._stop_generation_id = -1  # Which generation ID to stop (-1 = none)
        self._active_generations = 0  # Count of active generators (thread-safe via _id_lock)
        self._generation_lock = threading.Lock()  # Prevent concurrent generations
        self._id_lock = threading.Lock()  # Protect generation_id, stop_generation_id, active count

    @property
    def is_generating(self) -> bool:
        """Check if any generation is active (including between chunks).

        Uses a counter rather than a boolean so that overlapping generator
        lifecycles (one orphaned at yield, one active) don't race on the flag.
        """
        return self._active_generations > 0

    def request_stop(self, generation_id: int | None = None) -> None:
        """Request a specific generation to stop after the current chunk.

        Args:
            generation_id: The ID of the generation to stop. If None, stops all generations.
        """
        with self._id_lock:
            if generation_id is None:
                # Stop all generations (backward compat / emergency stop)
                self._stop_generation_id = self._generation_id
                logger.info(f"Stop requested for all generations (<= {self._generation_id})")
            else:
                # Stop only the specific generation
                # Only update if this would stop more generations than currently marked
                if generation_id > self._stop_generation_id:
                    self._stop_generation_id = generation_id
                logger.info(f"Stop requested for generation {generation_id}")

    def generate_long_audio(
        self,
        text_chunks: List[str],
        speaker_latent: torch.Tensor,
        speaker_mask: torch.Tensor,
        rng_seed: int = 0,
        gen_params: dict | None = None,
        num_candidates: int = 1,
        selector: Any = None,
        max_rounds: int = 1,
        speaker_audio: torch.Tensor | None = None,
        on_selection: Callable[[int, dict], None] | None = None,
    ) -> tuple[int, Generator[torch.Tensor, None, None]]:
        """
        Generate audio for multiple text chunks with continuation.

        Returns a tuple of (generation_id, generator). The generation_id can be used
        with request_stop(generation_id) to stop this specific generation.

        Args:
            text_chunks: List of text strings to generate
            speaker_latent: Speaker latent from voice
            speaker_mask: Speaker mask from voice
            rng_seed: Random seed for generation
            gen_params: Optional overrides for DEFAULT_PARAMS (e.g. num_steps,
                cfg_scale_text, cfg_scale_speaker); unknown/None keys fall back
                to the defaults.
            num_candidates: Takes to generate per chunk. All are produced in one
                batched diffusion pass, then the closest to the text is kept.
            selector: A CandidateSelector, or None to keep the first take.
            max_rounds: If the best take still scores above the selector's
                threshold, generate another batch, up to this many rounds.
            speaker_audio: Reference waveform, for the speaker-similarity term.
            on_selection: Called as (chunk_index, selection_dict) once per chunk
                when verification ran, for progress reporting. Invoked on the
                worker thread, before the chunk is yielded.

        Returns:
            Tuple of (generation_id, generator) where generator yields audio tensors
            for each chunk (shape: [1, 1, audio_length])
        """
        # Task 6: Validate that we have text to generate
        if not text_chunks:
            raise ValueError("No text to generate after normalization")

        # Assign unique ID to this generation for stop isolation (thread-safe)
        with self._id_lock:
            self._generation_id += 1
            my_generation_id = self._generation_id

            # Stop any previous generation that might be running
            if self._stop_generation_id < my_generation_id - 1:
                self._stop_generation_id = my_generation_id - 1
                logger.info(f"Auto-stopping generations <= {self._stop_generation_id} (new generation {my_generation_id} starting)")

        logger.info(f"Generation {my_generation_id} waiting for lock...")

        return my_generation_id, self._generate_chunks(
            text_chunks, speaker_latent, speaker_mask, rng_seed, my_generation_id, gen_params,
            num_candidates=num_candidates, selector=selector, max_rounds=max_rounds,
            speaker_audio=speaker_audio, on_selection=on_selection,
        )

    def _generate_chunks(
        self,
        text_chunks: List[str],
        speaker_latent: torch.Tensor,
        speaker_mask: torch.Tensor,
        rng_seed: int,
        my_generation_id: int,
        gen_params: dict | None = None,
        num_candidates: int = 1,
        selector: Any = None,
        max_rounds: int = 1,
        speaker_audio: torch.Tensor | None = None,
        on_selection: Callable[[int, dict], None] | None = None,
    ) -> Generator[torch.Tensor, None, None]:
        """Internal generator that yields audio chunks.

        The lock is acquired per-chunk (not for the entire generation) so that
        if the SSE consumer disconnects and this generator is orphaned at a
        yield point, the lock is already released and new generations can proceed.
        """

        try:
            with self._id_lock:
                self._active_generations += 1
            continuation_latent = None
            previous_chunk_text = ""  # Continuation text from previous chunk

            for i, chunk_text in enumerate(text_chunks):
                # Acquire lock per-chunk to protect the GPU
                # If generator is orphaned at yield, lock is already released
                self._generation_lock.acquire()
                try:
                    logger.info(f"Generation {my_generation_id} acquired lock for chunk {i+1}/{len(text_chunks)}")

                    # Check if we were stopped while waiting for the lock
                    if self._stop_generation_id >= my_generation_id:
                        logger.info(f"Generation {my_generation_id} stopped by user")
                        return

                    # Log chunk number and full text (encode to ASCII to avoid unicode errors in Windows console)
                    chunk_text_safe = chunk_text.encode('ascii', 'replace').decode('ascii')
                    logger.info(f"Generating chunk {i+1}/{len(text_chunks)}: {chunk_text_safe}")

                    # When using continuation, include portion of previous chunk's text
                    # (~last 60% of previous chunk, aligned to sentence boundary)
                    if previous_chunk_text:
                        # Encode to ASCII to avoid unicode errors
                        prev_preview = previous_chunk_text[:80].encode('ascii', 'replace').decode('ascii')
                        logger.debug(f"  Previous chunk text ({len(previous_chunk_text)} chars): {prev_preview}..." if len(previous_chunk_text) > 80 else f"  Previous chunk text: {prev_preview}")
                        full_text = previous_chunk_text + " " + chunk_text
                        full_preview = full_text[:150].encode('ascii', 'replace').decode('ascii')
                        logger.debug(f"  Full text for generation ({len(full_text)} chars): {full_preview}..." if len(full_text) > 150 else f"  Full text: {full_preview}")
                    else:
                        full_text = chunk_text

                    audio_chunk, latent_out, audio_full = self._best_take(
                        chunk_index=i,
                        full_text=full_text,
                        chunk_text=chunk_text,
                        speaker_latent=speaker_latent,
                        speaker_mask=speaker_mask,
                        continuation_latent=continuation_latent,
                        rng_seed=rng_seed,
                        gen_params=gen_params,
                        num_candidates=num_candidates,
                        selector=selector,
                        max_rounds=max_rounds,
                        speaker_audio=speaker_audio,
                        on_selection=on_selection,
                    )

                    # audio_chunk = NEW audio only (continuation removed, for yielding)
                    # audio_full = FULL cropped audio (includes continuation)

                    # Extract continuation from NEW audio only (not including previous continuation)
                    # This prevents continuation from growing each iteration
                    if i < len(text_chunks) - 1:  # Not the last chunk
                        continuation_latent = self._extract_continuation(latent_out, audio_chunk)

                        # Use full previous chunk text - continuation latents now cover most of the chunk
                        previous_chunk_text = chunk_text
                finally:
                    self._generation_lock.release()
                    logger.info(f"Generation {my_generation_id} released lock for chunk {i+1}/{len(text_chunks)}")

                # Yield OUTSIDE the lock - orphaned generators can't hold it
                yield audio_chunk
        finally:
            with self._id_lock:
                self._active_generations -= 1

    def _best_take(
        self,
        chunk_index: int,
        full_text: str,
        chunk_text: str,
        speaker_latent: torch.Tensor,
        speaker_mask: torch.Tensor,
        continuation_latent: torch.Tensor | None,
        rng_seed: int,
        gen_params: dict | None,
        num_candidates: int,
        selector: Any,
        max_rounds: int,
        speaker_audio: torch.Tensor | None,
        on_selection: Callable[[int, dict], None] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate takes of one chunk and return the one that matches the text.

        With no selector this is a single take, exactly as before. Otherwise
        every take is transcribed and scored, and the winner is returned -- which
        also makes it the take that seeds the next chunk's continuation, so an
        error is corrected instead of propagating forward.

        Scoring compares against ``chunk_text``, not ``full_text``: the audio we
        keep has the continuation trimmed off the front, so the previous chunk's
        text must not be part of the reference.
        """
        base_seed = rng_seed + chunk_index * SEED_STRIDE_CHUNK
        # takes[i] = (new_audio, latent, full_audio, seed)
        takes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
        selection = None
        rounds = max(1, max_rounds) if selector is not None else 1

        for round_index in range(rounds):
            round_seed = base_seed + round_index * SEED_STRIDE_ROUND
            produced = self._generate_chunk(
                full_text, speaker_latent, speaker_mask, continuation_latent,
                round_seed, gen_params,
                num_candidates=num_candidates if selector is not None else 1,
            )
            takes.extend((new, lat, full, round_seed) for new, lat, full in produced)

            if selector is None:
                break

            candidates = [
                Candidate(index=idx, seed=seed, audio=new)
                for idx, (new, _lat, _full, seed) in enumerate(takes)
            ]
            try:
                selection = selector.select(candidates, chunk_text, speaker_audio)
            except Exception as e:
                logger.error(f"Take verification failed, keeping the first take: {e}")
                selection = None
                break

            best = selection.best
            # Report the penalties too: a score well above the WER means the
            # take lost on duration, repetition or speaker drift, and without
            # this the retries look inexplicable in the log.
            detail = ", ".join(f"{k} {v:.2f}" for k, v in sorted(best.penalties.items()) if v)
            logger.info(
                f"  Chunk {chunk_index + 1}: kept take {best.index + 1}/{len(takes)} "
                f"(score {best.score:.3f}, WER {best.wer:.3f}"
                + (f", penalties: {detail}" if detail else "") + ")"
            )
            if selection.acceptable:
                break
            if round_index + 1 < rounds:
                logger.info(
                    f"  Chunk {chunk_index + 1}: best take still above threshold, "
                    f"generating another {num_candidates}"
                )

        if selection is not None:
            if on_selection is not None:
                try:
                    payload = selection.as_dict()
                    payload["rounds"] = rounds
                    on_selection(chunk_index, payload)
                except Exception as e:  # never let reporting break generation
                    logger.debug(f"on_selection callback failed: {e}")
            new_audio, latent_out, audio_full, _seed = takes[selection.winner]
            return new_audio, latent_out, audio_full

        new_audio, latent_out, audio_full, _seed = takes[0]
        return new_audio, latent_out, audio_full

    def _generate_chunk(
        self,
        text: str,
        speaker_latent: torch.Tensor,
        speaker_mask: torch.Tensor,
        continuation_latent: torch.Tensor | None,
        rng_seed: int,
        gen_params: dict | None = None,
        num_candidates: int = 1,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Generate one or more takes of a single audio chunk.

        Candidates ride a single batched diffusion pass: the sampler already
        derives its batch size from the text batch and draws independent noise
        per row, so N takes cost one pass rather than N sequential ones. Note
        that CFG internally triples the batch, so N takes means 3N rows.

        Args:
            text: Text to generate (includes previous text if continuation)
            speaker_latent: Speaker latent tensor
            speaker_mask: Speaker mask tensor
            continuation_latent: Optional continuation from previous chunk
            rng_seed: Random seed
            gen_params: Optional overrides for DEFAULT_PARAMS
            num_candidates: Number of independent takes to draw

        Returns:
            One tuple per take, each (new_audio, latent_output, full_audio):
                - new_audio: Audio with continuation removed (for yielding)
                - latent_output: Generated latent tensor
                - full_audio: Full cropped audio (for extracting next continuation)
        """
        start_time = time.time()
        num_candidates = max(1, num_candidates)

        # Encode text
        logger.debug(f"  [1/4] Encoding text ({len(text)} chars)...")
        # Encode to ASCII to avoid unicode errors in Windows console
        text_preview = text[:200].encode('ascii', 'replace').decode('ascii')
        logger.debug(f"  Text content: {text_preview}..." if len(text) > 200 else f"  Text content: {text_preview}")
        t0 = time.time()
        text_input_ids, text_mask = get_text_input_ids_and_mask(
            [text] * num_candidates,
            max_length=None,
            device=self.device,
        )
        logger.debug(f"  [1/4] Text encoded in {time.time() - t0:.2f}s")

        # Conditioning is shared by every take; expand it to match the batch.
        if num_candidates > 1:
            speaker_latent = speaker_latent.expand(num_candidates, -1, -1)
            speaker_mask = speaker_mask.expand(num_candidates, -1)
            if continuation_latent is not None:
                continuation_latent = continuation_latent.expand(num_candidates, -1, -1)

        # Determine block sizes
        if continuation_latent is None:
            block_sizes = [640]  # Full generation
            logger.debug(f"  [2/4] Starting fresh generation (block size: {block_sizes[0]})")
        else:
            # Account for continuation latent length
            cont_len = continuation_latent.shape[1]
            remaining = 640 - cont_len
            block_sizes = [remaining] if remaining > 0 else [640]
            logger.debug(f"  [2/4] Continuing from {cont_len} latents (block size: {block_sizes[0]})")

        # Merge any per-request overrides over the defaults (unknown keys ignored).
        params = dict(DEFAULT_PARAMS)
        if gen_params:
            params.update({k: v for k, v in gen_params.items() if k in DEFAULT_PARAMS and v is not None})

        # "Force Speaker" KV scaling needs a release point: the sampler compares
        # `t_next < speaker_kv_min_t` each step, which raises TypeError against
        # None. Supply the default rather than letting a scale-only request
        # crash mid-generation.
        if params.get("speaker_kv_scale") is not None and params.get("speaker_kv_min_t") is None:
            params["speaker_kv_min_t"] = DEFAULT_SPEAKER_KV_MIN_T
            logger.debug(
                f"  speaker_kv_scale set without a release point; "
                f"using speaker_kv_min_t={DEFAULT_SPEAKER_KV_MIN_T}"
            )

        # Generate latents
        logger.debug(f"  [2/4] Generating latents ({params['num_steps']} diffusion steps)...")
        t0 = time.time()
        latent_out = sample_blockwise_euler_cfg_independent_guidances(
            model=self.model,
            speaker_latent=speaker_latent,
            speaker_mask=speaker_mask,
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            rng_seed=rng_seed,
            block_sizes=block_sizes,
            continuation_latent=continuation_latent,
            **params,
        )
        logger.debug(f"  [2/4] Latents generated in {time.time() - t0:.2f}s")
        logger.debug(f"  Latent shape: {latent_out.shape}")

        # Decode to audio
        logger.debug(f"  [3/4] Decoding latents to audio...")
        t0 = time.time()
        audio_out = ae_decode(self.fish_ae, self.pca_state, latent_out)
        logger.debug(f"  [3/4] Audio decoded in {time.time() - t0:.2f}s")
        logger.debug(f"  Audio shape before crop: {audio_out.shape}")

        logger.debug(f"  [4/4] Cropping audio...")
        logger.debug(f"  Audio shape before crop: {audio_out.shape} = {audio_out.shape[-1] / 44100:.2f}s")
        logger.debug(f"  Latent shape: {latent_out.shape}")

        # Each take ends at its own flattening point, so cropping is per-take:
        # a single shared crop would truncate the longer takes.
        takes: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for i in range(num_candidates):
            latent_i = latent_out[i:i + 1]
            audio_i = audio_out[i:i + 1]

            audio_full_i = crop_audio_to_flattening_point(audio_i, latent_out[i])
            logger.debug(
                f"  Take {i + 1}/{num_candidates} cropped to {audio_full_i.shape[-1]} samples "
                f"({audio_full_i.shape[-1] / 44100:.2f}s)"
            )

            # Remove continuation portion if present (for yielding)
            # Use the fundamental relationship: 1 latent = 2048 audio samples
            if continuation_latent is not None:
                continuation_len = continuation_latent.shape[1]
                continuation_samples = continuation_len * 2048
                logger.debug(
                    f"  Removing continuation: {continuation_len} latents = "
                    f"{continuation_samples} samples ({continuation_samples / 44100:.2f}s)"
                )
                audio_new_i = audio_full_i[..., continuation_samples:]
            else:
                audio_new_i = audio_full_i

            takes.append((audio_new_i, latent_i, audio_full_i))

        total_time = time.time() - start_time
        logger.info(
            f"  Chunk complete in {total_time:.2f}s"
            + (f" ({num_candidates} takes)" if num_candidates > 1 else "")
        )
        return takes

    def _extract_continuation(self, latent_out: torch.Tensor, audio_out: torch.Tensor) -> torch.Tensor:
        """
        Extract continuation context from generated audio.

        Uses the FULL chunk audio as continuation (not just the last N seconds).
        This ensures perfect text-audio alignment when we pass full chunk text.

        Args:
            latent_out: Generated latent tensor (used to calculate expected latent count)
            audio_out: Generated audio tensor (cropped, the full chunk to use as continuation)

        Returns:
            Continuation latent tensor (re-encoded from full chunk audio)
        """
        # Use the FULL chunk audio as continuation
        continuation_audio = audio_out

        # Re-encode through Fish AE (same as Echo-TTS example)
        continuation_latent, continuation_mask = get_speaker_latent_and_mask(
            self.fish_ae,
            self.pca_state,
            continuation_audio[0],
        )

        # Trim to actual content (use mask)
        actual_latents = int(continuation_mask.sum().item())
        continuation_latent = continuation_latent[:, :actual_latents]

        # Cap at max to leave room for new generation (need at least 240 new latents)
        if continuation_latent.shape[1] > MAX_CONTINUATION_LATENTS:
            logger.warning(f"Continuation too long ({continuation_latent.shape[1]} latents), truncating to {MAX_CONTINUATION_LATENTS}")
            continuation_latent = continuation_latent[:, -MAX_CONTINUATION_LATENTS:]

        return continuation_latent
