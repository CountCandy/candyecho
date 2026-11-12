"""Drama 3, driven over its own HTTP server.

``RemoteAudioGenerator`` mirrors the interface of the Echo ``AudioGenerator``
closely enough that the streaming, job and best-of-N code in ``main.py`` can
drive either: it yields one ``[1, channels, samples]`` tensor per text chunk
and supports stopping after the current chunk.

What does *not* carry over is Echo's latent continuation. A remote engine
takes the reference clips and their transcripts on every request instead, so
each chunk is voiced from the same references rather than from the previous
chunk's audio.

HTTP is done with ``urllib`` on purpose: adding a client library would mean
touching the locked dependency set for one small POST request.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
import urllib.error
import urllib.request
import wave
from typing import Any, Callable, Generator, List

import torch

from .drama_markup import prepare_drama_chunks, strip_drama_markup  # noqa: F401 (re-exported)
from .transcript_verifier import Candidate
from .voice_transcripts import VoiceReference

logger = logging.getLogger(__name__)

# Generation ids are handed to the browser so it can stop "its" run. Each
# engine gets its own range so /stop can route an id to the right generator
# without the generators sharing state.
REMOTE_ID_BASE = 1_000_000
REMOTE_ID_SPAN = 1_000_000

# Same spacing as Echo's best-of-N, so a seed means the same thing everywhere.
SEED_STRIDE_CHUNK = 1000
SEED_STRIDE_ROUND = 100
MAX_SEED = 2_147_483_647


class EngineError(RuntimeError):
    """A remote engine refused or failed a request (message is user-facing)."""


# ---------------------------------------------------------------------------
# Audio decoding
# ---------------------------------------------------------------------------

def pcm16_to_tensor(data: bytes, channels: int = 1) -> torch.Tensor:
    """Signed 16-bit little-endian PCM -> float tensor ``[1, channels, samples]``."""
    usable = len(data) - (len(data) % (2 * channels))
    if usable <= 0:
        raise EngineError("The engine returned no audio.")
    samples = torch.frombuffer(bytearray(data[:usable]), dtype=torch.int16)
    audio = samples.to(torch.float32) / 32768.0
    return audio.reshape(-1, channels).transpose(0, 1).contiguous().unsqueeze(0)


def wav_to_tensor(data: bytes) -> tuple[torch.Tensor, int]:
    """WAV bytes -> (``[1, channels, samples]`` float tensor, sample rate)."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if wav.getsampwidth() != 2:
                raise wave.Error("not 16-bit PCM")
            rate, channels = wav.getframerate(), wav.getnchannels()
            frames = wav.readframes(wav.getnframes())
        if not frames:
            # A streamed WAV can carry a zero length in its header; the samples
            # are still everything after the 44-byte header.
            frames = data[44:]
        return pcm16_to_tensor(frames, channels), rate
    except (wave.Error, EOFError):
        pass

    # Float or otherwise unusual WAV: let torchaudio's decoder handle it.
    import tempfile
    from pathlib import Path

    import torchaudio

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        audio, rate = torchaudio.load(tmp_path)
    except Exception as e:
        raise EngineError(f"Could not decode the audio the engine returned: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return audio.to(torch.float32).unsqueeze(0), int(rate)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _http_post(
    url: str, body: bytes, headers: dict[str, str], timeout: float
) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.read(), {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace").strip()
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict):
                detail = str(parsed.get("detail") or parsed.get("message") or parsed.get("error") or detail)
        except ValueError:
            pass
        raise EngineError(f"The engine rejected the request (HTTP {e.code}): {detail[:400]}") from e
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise EngineError(f"Could not reach the engine at {url}: {reason}") from e


Poster = Callable[[str, bytes, dict[str, str], float], tuple[bytes, dict[str, str]]]


class DramaClient:
    """Fish Speech API server (``POST /v1/tts``) with Drama 3 loaded."""

    key = "drama3"
    label = "Fish Audio Drama 3"
    sample_rate = 44_100
    PARAMS = ("temperature", "top_p", "top_k")
    # Fields the server wants as integers.
    INT_PARAMS = frozenset({"top_k"})

    def __init__(self, url: str, post: Poster = _http_post, timeout: float = 900.0):
        self.url = url.rstrip("/")
        self._post = post
        self.timeout = timeout

    def synthesize(
        self, text: str, reference: VoiceReference, seed: int, params: dict | None = None
    ) -> tuple[torch.Tensor, int]:
        payload: dict[str, Any] = {
            "text": text,
            "format": "wav",
            "streaming": False,
            "seed": int(seed) % (MAX_SEED + 1),
            # Every clip of the speaker is sent: Drama 3 conditions on all of
            # them. The server accepts base64 for the bytes when the body is JSON.
            "references": [
                {"audio": base64.b64encode(clip.wav_bytes).decode("ascii"), "text": clip.transcript}
                for clip in reference.clips
            ],
        }
        for name in self.PARAMS:
            value = (params or {}).get(name)
            if value is not None:
                payload[name] = int(value) if name in self.INT_PARAMS else float(value)
        body, _headers = self._post(
            f"{self.url}/v1/tts",
            json.dumps(payload).encode("utf-8"),
            {"Content-Type": "application/json", "Accept": "audio/wav"},
            self.timeout,
        )
        return wav_to_tensor(body)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class RemoteAudioGenerator:
    """Chunk-by-chunk generation against a remote engine."""

    def __init__(
        self,
        client: Any,
        id_base: int = REMOTE_ID_BASE,
        prepare_chunks: Callable[[List[str]], List[str]] | None = None,
        spoken_text: Callable[[str], str] | None = None,
    ):
        self.client = client
        self.key = client.key
        self.sample_rate = client.sample_rate
        self._prepare_chunks = prepare_chunks
        self._spoken_text = spoken_text
        self._id_base = id_base
        self._next_id = id_base
        self._stopped: set[int] = set()
        self._stop_all_below = id_base - 1
        self._active = 0
        self._id_lock = threading.Lock()
        # These servers handle one request at a time; queue rather than collide.
        self._request_lock = threading.Lock()

    @property
    def is_generating(self) -> bool:
        return self._active > 0

    def owns(self, generation_id: int) -> bool:
        return self._id_base <= generation_id < self._id_base + REMOTE_ID_SPAN

    def request_stop(self, generation_id: int | None = None) -> None:
        with self._id_lock:
            if generation_id is None:
                self._stop_all_below = self._next_id
            else:
                self._stopped.add(generation_id)
        logger.info(f"Stop requested for {self.key} generation {generation_id or 'ALL'}")

    def _is_stopped(self, generation_id: int) -> bool:
        with self._id_lock:
            return generation_id in self._stopped or generation_id <= self._stop_all_below

    def synthesize_once(
        self,
        text: str,
        reference: VoiceReference,
        seed: int,
        params: dict | None = None,
    ) -> tuple[torch.Tensor, int]:
        """One standalone take (used for auditions), queued behind any run."""
        script = self._prepare_chunks([text])[0] if self._prepare_chunks else text
        with self._request_lock:
            return self.client.synthesize(script, reference, seed, params)

    def generate_long_audio(
        self,
        text_chunks: List[str],
        reference: VoiceReference,
        rng_seed: int = 0,
        gen_params: dict | None = None,
        num_candidates: int = 1,
        selector: Any = None,
        max_rounds: int = 1,
        speaker_audio: torch.Tensor | None = None,
        on_selection: Callable[[int, dict], None] | None = None,
        start_index: int = 0,
        **_unused: Any,
    ) -> tuple[int, Generator[torch.Tensor, None, None]]:
        """Same contract as ``AudioGenerator.generate_long_audio``.

        Continuation arguments are accepted and ignored: there is no latent to
        carry between chunks for these engines.
        """
        with self._id_lock:
            self._next_id += 1
            generation_id = self._next_id
        scripts = self._prepare_chunks(list(text_chunks)) if self._prepare_chunks else list(text_chunks)
        return generation_id, self._generate_chunks(
            scripts, reference, rng_seed, generation_id, gen_params,
            num_candidates, selector, max_rounds, speaker_audio, on_selection, start_index,
        )

    def _generate_chunks(
        self,
        scripts: List[str],
        reference: VoiceReference,
        rng_seed: int,
        generation_id: int,
        gen_params: dict | None,
        num_candidates: int,
        selector: Any,
        max_rounds: int,
        speaker_audio: torch.Tensor | None,
        on_selection: Callable[[int, dict], None] | None,
        start_index: int,
    ) -> Generator[torch.Tensor, None, None]:
        try:
            with self._id_lock:
                self._active += 1
            for i, script in enumerate(scripts):
                if i < start_index:
                    continue
                with self._request_lock:
                    if self._is_stopped(generation_id):
                        logger.info(f"{self.key} generation {generation_id} stopped by user")
                        return
                    logger.info(f"[{self.key}] chunk {i + 1}/{len(scripts)}")
                    audio = self._best_take(
                        i, script, reference, rng_seed, gen_params,
                        num_candidates, selector, max_rounds, speaker_audio, on_selection,
                    )
                yield audio
        finally:
            with self._id_lock:
                self._active -= 1
                self._stopped.discard(generation_id)

    def _best_take(
        self,
        chunk_index: int,
        script: str,
        reference: VoiceReference,
        rng_seed: int,
        gen_params: dict | None,
        num_candidates: int,
        selector: Any,
        max_rounds: int,
        speaker_audio: torch.Tensor | None,
        on_selection: Callable[[int, dict], None] | None,
    ) -> torch.Tensor:
        """One take, or the best of several when a selector is supplied.

        Unlike Echo, takes cannot share a batched pass, so N takes cost N
        requests.
        """
        base_seed = rng_seed + chunk_index * SEED_STRIDE_CHUNK
        rounds = max(1, max_rounds) if selector is not None else 1
        per_round = max(1, num_candidates) if selector is not None else 1
        # The verifier compares against what is spoken, not the markup.
        spoken = self._spoken_text(script) if self._spoken_text else script

        takes: list[tuple[torch.Tensor, int, int]] = []  # (audio, seed, sample_rate)
        selection = None
        for round_index in range(rounds):
            round_seed = base_seed + round_index * SEED_STRIDE_ROUND
            for n in range(per_round):
                seed = round_seed + n
                audio, rate = self.client.synthesize(script, reference, seed, gen_params)
                takes.append((audio, seed, rate))

            if selector is None:
                break
            candidates = [
                Candidate(index=idx, seed=seed, audio=audio, sample_rate=rate)
                for idx, (audio, seed, rate) in enumerate(takes)
            ]
            try:
                selection = selector.select(candidates, spoken, speaker_audio)
            except Exception as e:
                logger.error(f"Take verification failed, keeping the first take: {e}")
                selection = None
                break
            if selection.acceptable:
                break

        if selection is not None:
            if on_selection is not None:
                try:
                    payload = selection.as_dict()
                    payload["rounds"] = rounds
                    on_selection(chunk_index, payload)
                except Exception as e:  # never let reporting break generation
                    logger.debug(f"on_selection callback failed: {e}")
            return takes[selection.winner][0]
        return takes[0][0]
