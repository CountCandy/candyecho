import asyncio
import base64
import json
import logging
import os
import re
import signal
import tempfile
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
from .text_segmenter import segment_text
from .text_normalizer import TextNormalizer, NormalizationLevel
from .voice_event_broadcaster import VoiceEventBroadcaster
from .file_watcher import FileWatcher

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


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000)
    voice: str = Field(..., min_length=1)
    normalization_level: NormalizationLevel = Field("moderate")
    normalize_volume: bool = Field(False)


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

            # Normalize text for TTS (expand currencies, abbreviations, etc.)
            normalizer = TextNormalizer(level=body.normalization_level)
            normalized_text = normalizer.normalize(body.text)
            logger.info(f"Normalized text ({len(body.text)} -> {len(normalized_text)} chars)")

            # Segment text
            text_chunks = segment_text(normalized_text)
            logger.info(f"Created {len(text_chunks)} chunks")

            # Generate chunks - use thread pool to allow event loop to process other requests
            generation_id, generator = audio_generator.generate_long_audio(
                text_chunks, speaker_latent, speaker_mask
            )

            # Send generation_id first so frontend can use it for stop requests
            yield f"data: {json.dumps({'type': 'start', 'generation_id': generation_id, 'chunks': len(text_chunks)})}\n\n"

            i = 0
            try:
                while True:
                    # Run the blocking generator iteration in a thread pool
                    audio_chunk = await asyncio.to_thread(next, generator, None)
                    if audio_chunk is None:
                        break

                    # Send progress
                    progress_msg = f"Generated chunk {i+1}/{len(text_chunks)}"
                    yield f"data: {json.dumps({'type': 'progress', 'message': progress_msg})}\n\n"

                    # Convert audio to base64 WAV (optionally volume-normalized)
                    audio_base64 = _audio_to_base64_wav(audio_chunk, normalize=body.normalize_volume)

                    # Send chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'data': audio_base64, 'index': i})}\n\n"

                    i += 1

                # Send completion
                yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            finally:
                # Close the generator to clean up _is_generating flag.
                # The lock is released per-chunk (before yield), so no lock leak here.
                # If the generator is still executing in the thread pool, close() raises
                # ValueError - in that case it will stop on the next stop-flag check.
                try:
                    generator.close()
                except ValueError:
                    pass  # Generator still executing, will stop on next iteration

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


def _audio_to_base64_wav(audio_tensor: torch.Tensor, normalize: bool = False) -> str:
    """
    Convert an audio tensor (shape [batch, channels, samples]) to a base64 WAV,
    optionally volume-normalized.
    """
    audio_cpu = audio_tensor[0].cpu()
    if normalize:
        audio_cpu = _normalize_audio_tensor(audio_cpu)
    data, _ = _encode_audio(audio_cpu, "wav")
    return base64.b64encode(data).decode("utf-8")


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

    normalized_text = TextNormalizer(level=body.normalization_level).normalize(body.input)
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
    if body.normalize_volume:
        audio = _normalize_audio_tensor(audio)

    data, media_type = _encode_audio(audio, body.response_format)
    return Response(content=data, media_type=media_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("longecho.main:app", host="127.0.0.1", port=8100, reload=True)
