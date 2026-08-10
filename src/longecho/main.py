import asyncio
import base64
import json
import logging
import os
import random
import re
import signal
import tempfile
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch
import torchaudio

from longecho._vendor.echo_tts import (
    load_model_from_hf,
    load_fish_ae_from_hf,
    load_pca_state_from_hf,
)
from .voice_manager import VoiceManager
from .audio_generator import AudioGenerator
from .text_extractor import extract_text_from_upload
from .text_segmenter import segment_text
from .text_cleaner import PRESETS as CLEANING_PRESETS, RULES as CLEANING_RULES, clean_text
from .text_normalizer import TextNormalizer, NormalizationLevel
from .voice_event_broadcaster import VoiceEventBroadcaster
from .file_watcher import FileWatcher
from .job_manager import STATUS_COMPLETE, STATUS_FAILED, STATUS_STOPPED, EtaTracker, JobManager
from .subtitles import build_cues, to_srt, to_vtt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and voices on startup, cleanup on shutdown"""
    # Startup
    logger.info("Loading Echo models...")
    model = load_model_from_hf()
    fish_ae = load_fish_ae_from_hf()
    pca_state = load_pca_state_from_hf()
    logger.info("Models loaded successfully")

    # Initialize voice manager and load voices
    logger.info("Loading voices...")
    app.state.voice_manager = VoiceManager(fish_ae, pca_state)
    app.state.voice_manager.load_voices()
    logger.info(f"Loaded {len(app.state.voice_manager.get_voice_names())} voices")

    # Initialize audio generator
    app.state.audio_generator = AudioGenerator(model, fish_ae, pca_state)
    logger.info("Audio generator initialized")

    # Initialize voice event broadcaster
    app.state.voice_broadcaster = VoiceEventBroadcaster()
    logger.info("Voice event broadcaster initialized")

    # Generation jobs are written to disk as they are produced, so a closed tab
    # or a crash no longer destroys hours of GPU work.
    app.state.job_manager = JobManager(root=Path(os.environ.get("CANDYECHO_JOBS_DIR", "jobs")))
    unfinished = [j for j in app.state.job_manager.list_jobs() if not j.complete]
    if unfinished:
        logger.info(f"{len(unfinished)} unfinished job(s) available to resume")

    # Create callback for new voice files
    async def on_new_voice_file(wav_path: Path):
        if not wav_path.exists():
            logger.warning(f"Voice file '{wav_path}' was deleted before processing")
            return

        voice_name = wav_path.stem
        logger.info(f"New voice file detected: {voice_name}")

        # Broadcast processing event
        app.state.voice_broadcaster.broadcast({
            "type": "processing",
            "voice": voice_name,
        })

        try:
            # Process the voice (runs in thread pool to avoid blocking)
            result = await asyncio.to_thread(
                app.state.voice_manager.add_voice, wav_path
            )

            if result is not None:
                # Broadcast ready event
                app.state.voice_broadcaster.broadcast({
                    "type": "ready",
                    "voice": voice_name,
                })
                logger.info(f"Voice '{voice_name}' ready")
            else:
                logger.info(f"Voice '{voice_name}' was already loaded")
        except Exception as e:
            logger.error(f"Failed to process voice '{voice_name}': {e}")
            app.state.voice_broadcaster.broadcast({
                "type": "error",
                "voice": voice_name,
                "reason": str(e),
            })

    # Create callback for deleted voice files
    async def on_deleted_voice_file(wav_path: Path):
        voice_name = wav_path.stem
        logger.info(f"Voice file deleted: {voice_name}")

        # Remove from loaded voices
        removed = app.state.voice_manager.remove_voice(voice_name)

        if removed:
            # Broadcast removed event
            app.state.voice_broadcaster.broadcast({
                "type": "removed",
                "voice": voice_name,
            })

    # Start file watcher
    app.state.file_watcher = FileWatcher(
        watch_dir=Path("voice_library"),
        on_new_wav=on_new_voice_file,
        on_deleted_wav=on_deleted_voice_file,
    )
    app.state.file_watcher.start(asyncio.get_running_loop())
    logger.info("File watcher started")

    # Set up Ctrl+C handler to stop generation gracefully
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum, frame):
        if app.state.audio_generator.is_generating:
            logger.info("Ctrl+C received - requesting generation stop")
            app.state.audio_generator.request_stop()
        else:
            logger.info("Ctrl+C received - shutting down")
            # Call the original handler to exit normally
            if callable(original_sigint_handler):
                original_sigint_handler(signum, frame)
            else:
                raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, handle_sigint)
        logger.info("Ctrl+C handler installed")
    except ValueError:
        # Signal handlers can only be set in the main thread
        logger.debug("Skipping signal handler (not in main thread)")

    yield

    # Shutdown (cleanup if needed)
    logger.info("Shutting down...")

    # Stop file watcher
    if hasattr(app.state, 'file_watcher'):
        app.state.file_watcher.stop()


# Initialize FastAPI app with lifespan
app = FastAPI(title="LongEcho", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Dependency injection functions
def get_voice_manager(request: Request) -> VoiceManager:
    """Get the VoiceManager instance from app state"""
    return request.app.state.voice_manager


def get_audio_generator(request: Request) -> AudioGenerator:
    """Get the AudioGenerator instance from app state"""
    return request.app.state.audio_generator


def get_voice_broadcaster(request: Request) -> VoiceEventBroadcaster:
    """Get the VoiceEventBroadcaster instance from app state"""
    return request.app.state.voice_broadcaster


@app.get("/")
async def root():
    """Serve the main page"""
    return FileResponse("static/index.html")


@app.get("/favicon.ico")
async def favicon():
    """Return 204 No Content for favicon requests"""
    return Response(status_code=204)


@app.get("/health")
async def health(voice_manager: VoiceManager = Depends(get_voice_manager)):
    """Health check endpoint"""
    return {
        "status": "ok",
        "voices_loaded": len(voice_manager.get_voice_names())
    }


@app.get("/voices")
async def get_voices(voice_manager: VoiceManager = Depends(get_voice_manager)):
    """List available voices with reference-audio durations."""
    return {
        "voices": voice_manager.get_voice_info()
    }


# Document import (.txt / .epub) for the "load from file" button.
MAX_TEXT_IMPORT_MB = 25
MAX_TEXT_IMPORT_BYTES = MAX_TEXT_IMPORT_MB * 1024 * 1024


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """Extract plain text from an uploaded .txt or .epub for the text box."""
    name = (file.filename or "").lower()
    if not name.endswith((".txt", ".epub")):
        raise HTTPException(status_code=400, detail="Only .txt and .epub files are supported")

    data = await file.read(MAX_TEXT_IMPORT_BYTES + 1)
    if len(data) > MAX_TEXT_IMPORT_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_TEXT_IMPORT_MB} MB limit")
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        text = await asyncio.to_thread(extract_text_from_upload, file.filename or "", data)
    except Exception as e:
        logger.error(f"Failed to extract text from '{file.filename}': {e}")
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="No readable text found in the file")
    return {"text": text, "chars": len(text)}


class CleaningOptions(BaseModel):
    """Textbook cleaning settings sent by the UI panel.

    Applied before text normalization, so page furniture and footnote markers
    are gone before the segmenter turns line breaks into pauses.
    """

    enabled: bool = Field(True)
    preset: str = Field("textbook")
    rules: dict[str, bool] = Field(default_factory=dict)
    substitutions: dict[str, str] = Field(default_factory=dict)

    def apply(self, text: str) -> str:
        if not self.enabled:
            return text
        result = clean_text(
            text,
            preset=self.preset,
            overrides=self.rules,
            substitutions=self.substitutions,
        )
        removed = sum(r.count for r in result.report)
        if removed:
            logger.info(
                f"Cleaning removed {removed} items "
                f"({result.chars_before} -> {result.chars_after} chars)"
            )
        return result.text


@app.get("/verification-models")
async def verification_models():
    """Report which ASR models best-of-N is configured to use, and which loaded.

    Loading is lazy, so before the first verified generation only the configured
    ids are known; afterwards this also reports what actually came up, which is
    how a model that failed to load becomes visible in the UI instead of only in
    the console.
    """
    selector = getattr(app.state, "selector", None)
    return {
        "configured": {
            "whisper": _model_or_none(_VERIFY_WHISPER),
            "ctc": _model_or_none(_VERIFY_CTC),
            "speaker": _model_or_none(_VERIFY_SPEAKER),
            "device": _VERIFY_DEVICE,
        },
        "ready": selector is not None,
        "loaded": [t.name for t in selector.transcribers] if selector else [],
        "speaker_loaded": bool(selector and selector.speaker_similarity),
    }


@app.get("/cleaning-rules")
async def cleaning_rules():
    """Describe the available cleaning rules and presets for the UI panel."""
    return {
        "rules": [
            {"name": r.name, "label": r.label, "description": r.description}
            for r in CLEANING_RULES
        ],
        "presets": {name: sorted(rules) for name, rules in CLEANING_PRESETS.items()},
    }


class CleanTextRequest(BaseModel):
    text: str = Field("", max_length=2_000_000)
    preset: str = Field("textbook")
    rules: dict[str, bool] = Field(default_factory=dict)
    substitutions: dict[str, str] = Field(default_factory=dict)


@app.post("/clean-text")
async def clean_text_endpoint(body: CleanTextRequest):
    """Preview cleaning: returns the cleaned text plus what each rule removed."""
    result = await asyncio.to_thread(
        clean_text,
        body.text,
        body.preset,
        body.rules,
        body.substitutions,
    )
    return result.as_dict()


# ---------------------------------------------------------------------------
# Best-of-N take verification
# ---------------------------------------------------------------------------

# Model choices are env-configurable so they can be swapped without editing code.
# Set any of these to an empty string (or "none") to drop that component.
_VERIFY_WHISPER = os.environ.get("CANDYECHO_VERIFY_WHISPER", "jordand/whisper-d-v1a")
_VERIFY_CTC = os.environ.get("CANDYECHO_VERIFY_CTC", "facebook/wav2vec2-large-960h-lv60-self")
_VERIFY_SPEAKER = os.environ.get("CANDYECHO_VERIFY_SPEAKER", "microsoft/wavlm-base-plus-sv")
_VERIFY_DEVICE = os.environ.get("CANDYECHO_VERIFY_DEVICE", "cuda")

_verifier_lock = threading.Lock()


def _model_or_none(value: str) -> str | None:
    value = (value or "").strip()
    return None if value.lower() in ("", "none", "off", "false") else value


def get_selector(app_state, accept_threshold: float):
    """Build the candidate selector once, on first use.

    Loading WhisperD plus a CTC model plus a speaker-similarity model costs
    several GB and a noticeable startup delay, so nobody pays for it unless they
    actually switch verification on.
    """
    selector = getattr(app_state, "selector", None)
    if selector is not None:
        selector.accept_threshold = accept_threshold
        return selector

    with _verifier_lock:
        selector = getattr(app_state, "selector", None)
        if selector is None:
            from .transcript_verifier import build_selector

            logger.info("Loading take-verification models (first use)...")
            selector = build_selector(
                whisper_model=_model_or_none(_VERIFY_WHISPER),
                ctc_model=_model_or_none(_VERIFY_CTC),
                speaker_model=_model_or_none(_VERIFY_SPEAKER),
                device=_VERIFY_DEVICE,
                accept_threshold=accept_threshold,
            )
            app_state.selector = selector
            logger.info(
                "Take verification ready: "
                + ", ".join(t.name for t in selector.transcribers)
            )
    selector.accept_threshold = accept_threshold
    return selector


def _reference_audio(voice_manager: VoiceManager, voice_name: str):
    """Load a voice's reference waveform for the speaker-similarity check."""
    path = voice_manager.get_voice_path(voice_name)
    if path is None:
        return None
    try:
        from longecho._vendor.echo_tts import load_audio

        return load_audio(str(path))
    except Exception as e:
        logger.warning(f"Could not load reference audio for '{voice_name}': {e}")
        return None


# Voice upload configuration
MAX_VOICE_UPLOAD_MB = 50
MAX_VOICE_UPLOAD_BYTES = MAX_VOICE_UPLOAD_MB * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024
# Voice names are used to build a path inside voice_library/, so restrict them
# to a conservative character set (no path separators, no leading dots).
_SAFE_VOICE_NAME_RE = re.compile(r"[^A-Za-z0-9 _-]")


def _sanitize_voice_name(filename: str) -> str:
    """Derive a safe voice name from an uploaded filename.

    Strips directory components (path traversal), drops the extension, and
    limits the result to a safe character set so it cannot escape
    voice_library/ or create hidden/dot files.
    """
    stem = Path(filename).name  # drop any directory components
    if stem.lower().endswith(".wav"):
        stem = stem[:-4]  # drop only a trailing .wav, so plain rename names survive
    stem = _SAFE_VOICE_NAME_RE.sub("_", stem).strip(" ._-")
    return stem


@app.post("/voices")
async def upload_voice(
    file: UploadFile = File(...),
    voice_manager: VoiceManager = Depends(get_voice_manager),
):
    """
    Upload a .wav voice sample via the web interface.

    The file is saved into voice_library/ and preprocessed immediately (Fish
    AE + PCA), after which it is selectable for generation - no need to place
    files in the folder before startup.

    The upload is written to a temp file and atomically moved into place, which
    the directory watcher sees as a "moved" event (ignored), so the voice is
    processed exactly once - here.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    voice_name = _sanitize_voice_name(filename)
    if not voice_name:
        raise HTTPException(status_code=400, detail="Invalid or empty voice name")

    if voice_name in voice_manager.get_voice_names():
        raise HTTPException(status_code=409, detail=f"Voice '{voice_name}' already exists")

    voice_dir = voice_manager.voice_dir
    voice_dir.mkdir(parents=True, exist_ok=True)
    dest = voice_dir / f"{voice_name}.wav"
    if dest.exists():
        raise HTTPException(status_code=409, detail=f"Voice '{voice_name}' already exists")

    # Stream to a temp file in the same directory (hidden, non-.wav so the
    # watcher ignores it), enforcing a size cap, then atomically move into place.
    fd, tmp_path = tempfile.mkstemp(dir=str(voice_dir), prefix=f".{voice_name}.", suffix=".upload")
    tmp = Path(tmp_path)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_VOICE_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_VOICE_UPLOAD_MB} MB limit",
                    )
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        os.replace(tmp, dest)
    except HTTPException:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as e:
        tmp.unlink(missing_ok=True)
        logger.error(f"Failed to save uploaded voice '{voice_name}': {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Preprocess off the event loop. Remove the .wav if it can't be processed so
    # a broken sample isn't left in the library.
    try:
        await asyncio.to_thread(voice_manager.add_voice, dest)
    except Exception as e:
        dest.unlink(missing_ok=True)
        logger.error(f"Failed to process uploaded voice '{voice_name}': {e}")
        raise HTTPException(status_code=400, detail=f"Could not process audio: {e}")

    logger.info(f"Voice '{voice_name}' uploaded and ready")
    return {"status": "ready", "voice": voice_name}


class RenameVoiceRequest(BaseModel):
    new_name: str = Field(..., min_length=1)


@app.get("/voices/{voice_name}/audio")
async def voice_audio(
    voice_name: str,
    voice_manager: VoiceManager = Depends(get_voice_manager),
):
    """Serve a loaded voice's reference .wav so it can be previewed in the UI."""
    path = voice_manager.get_voice_path(voice_name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
    return FileResponse(str(path), media_type="audio/wav", filename=f"{voice_name}.wav")


@app.post("/voices/{voice_name}/rename")
async def rename_voice_endpoint(
    voice_name: str,
    body: RenameVoiceRequest,
    voice_manager: VoiceManager = Depends(get_voice_manager),
    voice_broadcaster: VoiceEventBroadcaster = Depends(get_voice_broadcaster),
):
    """Rename a voice (its .wav + cache) and notify connected clients."""
    new_name = _sanitize_voice_name(body.new_name)
    if not new_name:
        raise HTTPException(status_code=400, detail="Invalid or empty voice name")
    if voice_name not in voice_manager.get_voice_names():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
    if new_name == voice_name:
        return {"status": "ok", "old": voice_name, "new": new_name}

    try:
        await asyncio.to_thread(voice_manager.rename_voice, voice_name, new_name)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    voice_broadcaster.broadcast({"type": "renamed", "old": voice_name, "new": new_name})
    logger.info(f"Voice '{voice_name}' renamed to '{new_name}'")
    return {"status": "ok", "old": voice_name, "new": new_name}


@app.delete("/voices/{voice_name}")
async def delete_voice_endpoint(
    voice_name: str,
    voice_manager: VoiceManager = Depends(get_voice_manager),
    voice_broadcaster: VoiceEventBroadcaster = Depends(get_voice_broadcaster),
):
    """Delete a voice (its .wav + cache) and notify connected clients."""
    if voice_name not in voice_manager.get_voice_names():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
    await asyncio.to_thread(voice_manager.delete_voice, voice_name)
    voice_broadcaster.broadcast({"type": "removed", "voice": voice_name})
    logger.info(f"Voice '{voice_name}' deleted")
    return {"status": "deleted", "voice": voice_name}


@app.get("/voice-events")
async def voice_events(
    voice_broadcaster: VoiceEventBroadcaster = Depends(get_voice_broadcaster),
):
    """
    SSE endpoint for voice library events.

    Streams events:
    - processing: New voice file detected, processing started
    - ready: Voice processed and available
    - error: Voice processing failed
    - removed: Voice file deleted and unloaded
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        queue = voice_broadcaster.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat to detect dead connections
                    yield ": heartbeat\n\n"
        finally:
            voice_broadcaster.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/stop")
async def stop_generation(
    generation_id: int | None = None,
    audio_generator: AudioGenerator = Depends(get_audio_generator)
):
    """Stop a specific audio generation or all generations if no ID provided."""
    audio_generator.request_stop(generation_id)
    return {"status": "stop requested", "generation_id": generation_id}


async def _stream_chunks(
    *,
    audio_generator: AudioGenerator,
    job,
    job_manager: JobManager,
    speaker_latent,
    speaker_mask,
    seed: int,
    gen_params: dict,
    selector,
    speaker_audio,
    num_candidates: int,
    max_rounds: int,
    normalize_volume: bool,
    clean_audio: bool,
    start_index: int = 0,
    initial_continuation_audio=None,
    initial_previous_text: str = "",
) -> AsyncGenerator[str, None]:
    """Drive a generation, persisting each chunk and reporting progress.

    Shared by a fresh /generate and by resuming an interrupted job, so both
    paths write the same manifest and emit the same events.
    """
    text_chunks = [c.text for c in job.chunks]

    # The generator runs on a worker thread, so selections land in a queue and
    # are drained here in order, after each chunk arrives.
    selections: deque = deque()

    generation_id, generator = audio_generator.generate_long_audio(
        text_chunks, speaker_latent, speaker_mask,
        rng_seed=seed, gen_params=gen_params or None,
        num_candidates=num_candidates if selector else 1,
        selector=selector,
        max_rounds=max_rounds if selector else 1,
        speaker_audio=speaker_audio,
        on_selection=(lambda idx, payload: selections.append((idx, payload))) if selector else None,
        start_index=start_index,
        initial_continuation_audio=initial_continuation_audio,
        initial_previous_text=initial_previous_text,
    )

    start_event = {
        'type': 'start',
        'generation_id': generation_id,
        'job_id': job.id,
        'chunks': len(text_chunks),
        'start_index': start_index,
        'seed': seed,
        'verify': bool(selector),
    }
    if selector is not None:
        # Report what actually loaded, so a verifier that failed to come up is
        # visible in the UI rather than only in the console.
        start_event['verifiers'] = [t.name for t in selector.transcribers]
        start_event['speaker_check'] = bool(selector.speaker_similarity)
    yield f"data: {json.dumps(start_event)}\n\n"

    tracker = EtaTracker(len(text_chunks), already_done=start_index)
    index = start_index
    stopped_early = False

    try:
        while True:
            started = time.monotonic()
            audio_chunk = await asyncio.to_thread(next, generator, None)
            if audio_chunk is None:
                stopped_early = index < len(text_chunks)
                break
            tracker.record(time.monotonic() - started)

            # Encode once, then reuse the same bytes for disk and for the wire.
            wav_bytes = await asyncio.to_thread(
                _chunk_to_wav_bytes, audio_chunk, normalize_volume, clean_audio
            )
            duration = audio_chunk.shape[-1] / 44100.0

            verification = None
            while selections:
                chunk_index, payload = selections.popleft()
                verification = payload
                yield f"data: {json.dumps({'type': 'verification', 'chunk': chunk_index, **payload})}\n\n"

            best = None
            if verification:
                best = next(
                    (c for c in verification["candidates"] if c["index"] == verification["winner"]),
                    None,
                )

            await asyncio.to_thread(
                job_manager.record_chunk,
                job, index, wav_bytes, duration,
                tracker.per_chunk[-1] if tracker.per_chunk else None,
                (best or {}).get("seed"),
                (best or {}).get("score"),
                (best or {}).get("wer"),
                len(verification["candidates"]) if verification else None,
            )

            progress = {
                'type': 'progress',
                'message': f"Generated chunk {index + 1}/{len(text_chunks)}",
                **tracker.as_dict(),
            }
            yield f"data: {json.dumps(progress)}\n\n"
            yield f"data: {json.dumps({'type': 'chunk', 'data': base64.b64encode(wav_bytes).decode('utf-8'), 'index': index})}\n\n"
            index += 1

        job_manager.finish(job, STATUS_STOPPED if stopped_early else STATUS_COMPLETE)
        yield f"data: {json.dumps({'type': 'complete', 'job_id': job.id, 'stopped': stopped_early})}\n\n"
    except Exception:
        job_manager.finish(job, STATUS_FAILED)
        raise
    finally:
        # Close the generator to clean up the active-generation count. The lock
        # is released per-chunk (before yield), so no lock leak here. If the
        # generator is still executing in the thread pool, close() raises
        # ValueError - in that case it stops on the next stop-flag check.
        try:
            generator.close()
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Jobs: listing, download, subtitles, resume
# ---------------------------------------------------------------------------

def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


def _load_job_or_404(job_manager: JobManager, job_id: str):
    try:
        job = job_manager.load(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job id")
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@app.get("/jobs")
async def list_jobs(job_manager: JobManager = Depends(get_job_manager)):
    """List saved generations, newest first."""
    return {"jobs": [j.summary() for j in job_manager.list_jobs()]}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, job_manager: JobManager = Depends(get_job_manager)):
    return _load_job_or_404(job_manager, job_id).as_dict()


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str, job_manager: JobManager = Depends(get_job_manager)):
    _load_job_or_404(job_manager, job_id)
    await asyncio.to_thread(job_manager.delete, job_id)
    return {"status": "deleted", "job": job_id}


@app.get("/jobs/{job_id}/audio")
async def job_audio(
    job_id: str,
    format: str = "wav",
    job_manager: JobManager = Depends(get_job_manager),
):
    """Assemble a job's chunks server-side.

    Doing this on the server is what lets a full book be downloaded at all: the
    browser cannot hold hours of decoded audio in memory.
    """
    job = _load_job_or_404(job_manager, job_id)
    paths = job_manager.done_chunk_paths(job)
    if not paths:
        raise HTTPException(status_code=404, detail="This job has no audio yet")

    def _assemble() -> tuple[bytes, str]:
        parts = [torchaudio.load(str(p))[0] for p in paths]
        audio = torch.cat(parts, dim=-1)
        return _encode_audio(audio, format)

    data, media_type = await asyncio.to_thread(_assemble)
    ext = format.lower() if format.lower() in ("wav", "mp3", "flac", "ogg") else "wav"
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="candyecho-{job_id}.{ext}"'},
    )


@app.get("/jobs/{job_id}/subtitles")
async def job_subtitles(
    job_id: str,
    format: str = "srt",
    job_manager: JobManager = Depends(get_job_manager),
):
    """Subtitles timed from the measured duration of each generated chunk."""
    job = _load_job_or_404(job_manager, job_id)
    pairs = [(c.text, c.duration or 0.0) for c in job.chunks if c.done]
    if not pairs:
        raise HTTPException(status_code=404, detail="This job has no audio yet")

    cues = build_cues(pairs)
    if format.lower() == "vtt":
        return Response(
            content=to_vtt(cues),
            media_type="text/vtt",
            headers={"Content-Disposition": f'attachment; filename="candyecho-{job_id}.vtt"'},
        )
    return Response(
        content=to_srt(cues),
        media_type="application/x-subrip",
        headers={"Content-Disposition": f'attachment; filename="candyecho-{job_id}.srt"'},
    )


@app.post("/jobs/{job_id}/resume")
async def resume_job(
    job_id: str,
    voice_manager: VoiceManager = Depends(get_voice_manager),
    audio_generator: AudioGenerator = Depends(get_audio_generator),
    job_manager: JobManager = Depends(get_job_manager),
):
    """Continue an interrupted job from its first unfinished chunk.

    The last completed chunk's audio is re-encoded into the continuation latent,
    so the join sounds the same as it would have in an uninterrupted run.
    """
    job = _load_job_or_404(job_manager, job_id)
    if job.complete:
        raise HTTPException(status_code=409, detail="This job is already finished")
    if job.voice not in voice_manager.get_voice_names():
        raise HTTPException(status_code=404, detail=f"Voice '{job.voice}' is no longer available")

    settings = job.settings or {}
    start_index = job.resume_index
    speaker_latent, speaker_mask = voice_manager.get_voice(job.voice)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            selector = None
            speaker_audio = None
            if settings.get("verify"):
                selector = await asyncio.to_thread(get_selector, app.state, 0.10)
                speaker_audio = await asyncio.to_thread(_reference_audio, voice_manager, job.voice)

            continuation_audio = None
            previous_text = ""
            if start_index > 0:
                previous = job.chunks[start_index - 1]
                previous_text = previous.text
                continuation_audio = await asyncio.to_thread(
                    _load_chunk_audio, job_manager, job, start_index - 1
                )
                if continuation_audio is None:
                    logger.warning(
                        f"Job {job.id}: previous chunk audio missing, resuming without continuation"
                    )
                    previous_text = ""

            job.status = "running"
            job_manager.save(job)

            async for event in _stream_chunks(
                audio_generator=audio_generator,
                job=job,
                job_manager=job_manager,
                speaker_latent=speaker_latent,
                speaker_mask=speaker_mask,
                seed=int(settings.get("seed", 0)),
                gen_params=settings.get("gen_params") or {},
                selector=selector,
                speaker_audio=speaker_audio,
                num_candidates=int(settings.get("candidates", 1)),
                max_rounds=int(settings.get("max_rounds", 1)),
                normalize_volume=bool(settings.get("normalize_volume", False)),
                clean_audio=bool(settings.get("clean_audio", False)),
                start_index=start_index,
                initial_continuation_audio=continuation_audio,
                initial_previous_text=previous_text,
            ):
                yield event
        except Exception as e:
            logger.error(f"Resume error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _load_chunk_audio(job_manager: JobManager, job, index: int):
    """Load a finished chunk as [1, 1, samples] for continuation seeding."""
    path = job_manager.chunk_path(job.id, index)
    if not path.exists():
        return None
    try:
        audio, _sr = torchaudio.load(str(path))
        return audio.unsqueeze(0)  # [channels, samples] -> [1, channels, samples]
    except Exception as e:
        logger.warning(f"Could not load chunk {index} of job {job.id}: {e}")
        return None


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000)
    voice: str = Field(..., min_length=1)
    normalization_level: NormalizationLevel = Field("moderate")
    normalize_volume: bool = Field(False)
    clean_audio: bool = Field(False)
    # Textbook cleaning (page furniture, footnote markers, split sentences).
    # Omitted entirely by older clients, in which case no cleaning is applied.
    cleaning: CleaningOptions | None = Field(None)
    # Optional advanced generation controls (Echo-TTS sampler knobs). Ranges are
    # validated here so a stray client value can't reach the model; omit any of
    # them to use the tuned defaults.
    seed: int | None = Field(None, ge=0, le=2_147_483_647)
    steps: int | None = Field(None, ge=8, le=64)
    cfg_text: float | None = Field(None, ge=1.0, le=8.0)
    cfg_speaker: float | None = Field(None, ge=1.0, le=15.0)
    # Echo has no sampler temperature (it is a deterministic Euler ODE solver);
    # truncation scales the initial noise, which is the nearest analogue. Lower
    # is safer and flatter, higher is more varied and more error-prone.
    truncation: float | None = Field(None, ge=0.5, le=1.5)
    # "Force Speaker" KV scaling. Upstream: "aim for the lowest scale that
    # produces the correct speaker", 1.0 being baseline and 1.5 usually enough.
    speaker_force: float | None = Field(None, ge=1.0, le=3.0)
    # Best-of-N: generate several takes per chunk and keep the one an ASR
    # ensemble says matches the text. Off by default so existing clients are
    # unaffected and nobody loads the verifier models by accident.
    verify: bool = Field(False)
    candidates: int = Field(3, ge=1, le=8)
    max_rounds: int = Field(2, ge=1, le=5)
    verify_threshold: float = Field(0.10, ge=0.0, le=1.0)


@app.post("/generate")
async def generate(
    body: GenerateRequest,
    voice_manager: VoiceManager = Depends(get_voice_manager),
    audio_generator: AudioGenerator = Depends(get_audio_generator),
):
    """
    Generate audio for long-form text with SSE streaming.

    Streams events:
    - progress: Generation progress updates
    - chunk: Base64-encoded audio chunks
    - complete: Generation finished
    - error: Error occurred
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Validate voice exists
            if body.voice not in voice_manager.get_voice_names():
                yield f"data: {json.dumps({'type': 'error', 'message': f'Voice {body.voice} not found'})}\n\n"
                return

            # Get voice data
            speaker_latent, speaker_mask = voice_manager.get_voice(body.voice)

            # Clean book/textbook text first: strip page furniture and rejoin
            # sentences split across pages, before the segmenter turns line
            # breaks into pauses.
            source_text = body.cleaning.apply(body.text) if body.cleaning else body.text

            # Normalize text for TTS (expand currencies, abbreviations, etc.)
            normalizer = TextNormalizer(level=body.normalization_level)
            normalized_text = normalizer.normalize(source_text)
            logger.info(f"Normalized text ({len(body.text)} -> {len(normalized_text)} chars)")

            # Segment text
            text_chunks = segment_text(normalized_text)
            logger.info(f"Created {len(text_chunks)} chunks")

            # Resolve advanced controls: an explicit seed reproduces a run; a
            # blank seed varies each time (and is reported back so it can be reused).
            seed = body.seed if body.seed is not None else random.randint(0, 2_147_483_647)
            gen_params = {}
            if body.steps is not None:
                gen_params["num_steps"] = body.steps
            if body.cfg_text is not None:
                gen_params["cfg_scale_text"] = body.cfg_text
            if body.cfg_speaker is not None:
                gen_params["cfg_scale_speaker"] = body.cfg_speaker
            if body.truncation is not None:
                gen_params["truncation_factor"] = body.truncation
            if body.speaker_force is not None and body.speaker_force > 1.0:
                gen_params["speaker_kv_scale"] = body.speaker_force

            # Best-of-N verification. Models load on first use, off the event loop.
            selector = None
            speaker_audio = None
            if body.verify:
                try:
                    # First use downloads several GB of ASR weights inside this
                    # request, so say so rather than letting the UI look hung.
                    if getattr(app.state, "selector", None) is None:
                        yield f"data: {json.dumps({'type': 'progress', 'message': 'Loading take-verification models (first run downloads several GB)...'})}\n\n"
                    selector = await asyncio.to_thread(
                        get_selector, app.state, body.verify_threshold
                    )
                    speaker_audio = await asyncio.to_thread(
                        _reference_audio, voice_manager, body.voice
                    )
                except Exception as e:
                    logger.error(f"Could not start take verification: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Take verification unavailable: {e}'})}\n\n"
                    return

            job = app.state.job_manager.create(
                text_chunks,
                voice=body.voice,
                settings={
                    "seed": seed,
                    "verify": bool(selector),
                    "candidates": body.candidates if selector else 1,
                    "max_rounds": body.max_rounds if selector else 1,
                    "normalize_volume": body.normalize_volume,
                    "clean_audio": body.clean_audio,
                    "gen_params": gen_params,
                },
            )

            async for event in _stream_chunks(
                audio_generator=audio_generator,
                job=job,
                job_manager=app.state.job_manager,
                speaker_latent=speaker_latent,
                speaker_mask=speaker_mask,
                seed=seed,
                gen_params=gen_params,
                selector=selector,
                speaker_audio=speaker_audio,
                num_candidates=body.candidates,
                max_rounds=body.max_rounds,
                normalize_volume=body.normalize_volume,
                clean_audio=body.clean_audio,
            ):
                yield event

        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# Target ~ -20 dBFS RMS with a peak safety limit, so quiet/loud voices land at a
# consistent perceived loudness when volume normalization is enabled.
_NORM_TARGET_RMS = 0.1
_NORM_MAX_GAIN = 8.0
_NORM_PEAK_LIMIT = 0.99


def _normalize_audio_tensor(audio_cpu: torch.Tensor) -> torch.Tensor:
    """RMS-normalize audio toward a consistent loudness, clamped to avoid clipping."""
    rms = audio_cpu.pow(2).mean().sqrt()
    if not torch.isfinite(rms) or float(rms) < 1e-6:
        return audio_cpu
    gain = min(float(_NORM_TARGET_RMS / rms), _NORM_MAX_GAIN)
    out = audio_cpu * gain
    peak = out.abs().max()
    if float(peak) > _NORM_PEAK_LIMIT:
        out = out * (_NORM_PEAK_LIMIT / peak)
    return out


# Audio cleanup: a high-pass to drop sub-bass rumble/DC, plus a gentle,
# noise-floor-adaptive downward expander that pulls down hiss and low-level
# tails between phrases without touching speech. It won't remove true reverb
# (that needs a dedicated model), but it tames the common TTS nuisances.
_CLEAN_HPF_HZ = 85.0
_CLEAN_GATE_FLOOR = 0.06          # residual gain in the quietest sections
_CLEAN_NOISE_MULT = 3.0           # gate a bit above the estimated noise floor


def _clean_audio_tensor(audio_cpu: torch.Tensor, sample_rate: int = 44100) -> torch.Tensor:
    """Reduce hiss/rumble/low-level artifacts. Returns the input unchanged on any error."""
    try:
        x = audio_cpu.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)

        try:
            x = torchaudio.functional.highpass_biquad(x, sample_rate, cutoff_freq=_CLEAN_HPF_HZ)
        except Exception:
            pass

        peak = float(x.abs().max())
        if peak > 1e-6:
            # Smoothed amplitude envelope (~12 ms window).
            win = max(1, int(sample_rate * 0.012))
            kernel = torch.ones(1, 1, win) / win
            env = x.abs().mean(dim=0, keepdim=True)
            env_s = torch.nn.functional.conv1d(env.unsqueeze(0), kernel, padding=win // 2)[0][:, : x.shape[-1]]

            # Estimate the noise floor from a low percentile of the envelope.
            flat = env_s.flatten()
            if flat.numel() > 1_000_000:
                flat = flat[:: (flat.numel() // 1_000_000 + 1)]
            noise_floor = float(torch.quantile(flat, 0.10))
            thresh = max(noise_floor * _CLEAN_NOISE_MULT, peak * 1e-4)

            ratio = torch.clamp(env_s / thresh, 0.0, 1.0)
            gain = _CLEAN_GATE_FLOOR + (1.0 - _CLEAN_GATE_FLOOR) * ratio * ratio  # steeper knee
            gain = torch.nn.functional.conv1d(gain.unsqueeze(0), kernel, padding=win // 2)[0][:, : x.shape[-1]]
            x = x * gain

        x = torch.clamp(torch.nan_to_num(x), -1.0, 1.0)
        return x.to(audio_cpu.dtype)
    except Exception as e:
        logger.warning(f"Audio cleanup failed, returning original: {e}")
        return audio_cpu


def _encode_audio(audio_cpu: torch.Tensor, fmt: str) -> tuple[bytes, str]:
    """Encode a [channels, samples] CPU tensor to audio bytes in the given format.

    Falls back to WAV for unknown formats. Compressed formats (mp3/flac/ogg)
    require the FFmpeg backend, which the project already depends on.
    """
    fmt = (fmt or "wav").lower()
    media_types = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
    }
    if fmt not in media_types:
        fmt = "wav"
    # TorchCodec/torchaudio backends write to a path, not BytesIO.
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        torchaudio.save(tmp_path, audio_cpu, 44100)
        with open(tmp_path, "rb") as f:
            return f.read(), media_types[fmt]
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _chunk_to_wav_bytes(audio_tensor: torch.Tensor, normalize: bool = False, clean: bool = False) -> bytes:
    """Encode a generated chunk ([batch, channels, samples]) to WAV bytes.

    Optionally noise-cleaned and/or volume-normalized (cleanup runs first). The
    bytes are reused for both the on-disk chunk and the SSE payload, so a long
    book is not encoded twice per chunk.
    """
    audio_cpu = audio_tensor[0].cpu()
    if clean:
        audio_cpu = _clean_audio_tensor(audio_cpu)
    if normalize:
        audio_cpu = _normalize_audio_tensor(audio_cpu)
    data, _ = _encode_audio(audio_cpu, "wav")
    return data


def _audio_to_base64_wav(audio_tensor: torch.Tensor, normalize: bool = False, clean: bool = False) -> str:
    """Convert an audio tensor to a base64 WAV string."""
    return base64.b64encode(
        _chunk_to_wav_bytes(audio_tensor, normalize=normalize, clean=clean)
    ).decode("utf-8")


# ---------------------------------------------------------------------------
# OpenAI-compatible TTS API (for SillyTavern and other tools)
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    """OpenAI /v1/audio/speech-compatible request body.

    `model` and `speed` are accepted for compatibility but unused (Echo-TTS has
    no speed control). `voice` must be one of the loaded voices.
    """
    input: str = Field(..., min_length=1, max_length=100000)
    voice: str = Field(..., min_length=1)
    model: str = Field("candyecho")
    response_format: str = Field("wav")
    speed: float = Field(1.0)
    normalization_level: NormalizationLevel = Field("moderate")
    normalize_volume: bool = Field(True)
    clean_audio: bool = Field(False)
    # Off by default here: this endpoint serves chat text, not book pages, and
    # textbook rules should not fire on it unless a client explicitly asks.
    cleaning: CleaningOptions | None = Field(None)


@app.get("/v1/audio/voices")
async def openai_list_voices(voice_manager: VoiceManager = Depends(get_voice_manager)):
    """List available voices (convenience endpoint for OpenAI-compatible clients)."""
    return {"voices": sorted(voice_manager.get_voice_names())}


@app.post("/v1/audio/speech")
async def openai_audio_speech(
    body: SpeechRequest,
    voice_manager: VoiceManager = Depends(get_voice_manager),
    audio_generator: AudioGenerator = Depends(get_audio_generator),
):
    """
    Non-streaming, OpenAI-compatible speech synthesis.

    Generates the full clip for `input` in the requested `voice` and returns it
    as one audio response (WAV by default; mp3/flac/ogg via FFmpeg). This is the
    endpoint SillyTavern's "OpenAI Compatible" TTS provider expects.
    """
    if body.voice not in voice_manager.get_voice_names():
        raise HTTPException(status_code=404, detail=f"Voice '{body.voice}' not found")

    speaker_latent, speaker_mask = voice_manager.get_voice(body.voice)

    source_text = body.cleaning.apply(body.input) if body.cleaning else body.input
    normalized_text = TextNormalizer(level=body.normalization_level).normalize(source_text)
    text_chunks = segment_text(normalized_text)
    if not text_chunks:
        raise HTTPException(status_code=400, detail="No speakable text in input")

    _gen_id, generator = audio_generator.generate_long_audio(
        text_chunks, speaker_latent, speaker_mask
    )

    parts: list[torch.Tensor] = []
    try:
        while True:
            chunk = await asyncio.to_thread(next, generator, None)
            if chunk is None:
                break
            parts.append(chunk[0].detach().cpu())  # [channels, samples]
    finally:
        try:
            generator.close()
        except ValueError:
            pass

    if not parts:
        raise HTTPException(status_code=500, detail="No audio was generated")

    audio = torch.cat(parts, dim=-1)
    if body.clean_audio:
        audio = _clean_audio_tensor(audio)
    if body.normalize_volume:
        audio = _normalize_audio_tensor(audio)

    data, media_type = _encode_audio(audio, body.response_format)
    return Response(content=data, media_type=media_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("longecho.main:app", host="127.0.0.1", port=8100, reload=True)
