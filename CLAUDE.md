# LongEcho - Agent Guidelines

## Running Commands

Always use `uv run` prefix for all Python commands:
```bash
uv run python -m longecho.main      # Run server
uv run pytest                        # Run tests
uv run uvicorn longecho.main:app    # Run with uvicorn
```

Do NOT use bare `python` or `pytest` - they won't have the venv dependencies.

## Vendored Echo-TTS

Echo-TTS inference code is vendored at `src/longecho/_vendor/echo_tts/`.

**Imports:**
```python
# Correct
from longecho._vendor.echo_tts import load_model_from_hf, ae_decode

# Wrong - there is no external echo-tts package
from inference import load_model_from_hf
from echo_tts import load_model_from_hf
```

**Internal imports** within `_vendor/echo_tts/` use relative imports (`.inference`, `.model`, etc.).

## Dependencies

- PyTorch 2.11 (CUDA 13.0) installs from the `pytorch-cu130` index (`download.pytorch.org/whl/cu130`), configured in `pyproject.toml` — the CUDA wheels for Windows/Linux live only there, not on PyPI (PyPI's Windows torch is CPU-only)
- torch/torchaudio are pinned to the 2.11 line (torchaudio's final release is 2.11.0); torchcodec tracks it at 0.11
- CUDA 13 requires an NVIDIA driver R580+ on the GPU host
- `torchcodec` on Windows requires FFmpeg shared libraries in PATH (system dependency)
- Run `uv sync` to install/update dependencies

## Licensing

- Code: MIT (most files) and Apache-2.0 (autoencoder.py)
- Model weights: CC-BY-NC-SA-4.0 (non-commercial only)
- Generated audio inherits CC-BY-NC-SA-4.0 license
- Optional engines are not bundled; their weights carry their own licences
  (Drama 3: Fish Audio Community License; Breeze: research/non-commercial)

## Project Structure

```
src/longecho/
├── main.py                     # FastAPI server, model loading
├── audio_generator.py          # Chunk generation with context
├── voice_manager.py            # Voice preprocessing & caching
├── file_watcher.py             # voice_library/ auto-detection
├── voice_event_broadcaster.py  # SSE voice events
├── text_extractor.py           # .txt / .epub import (stdlib only)
├── text_cleaner.py             # Textbook cleaning (page furniture, reflow)
├── text_normalizer.py          # Currency, abbreviations, etc.
├── text_segmenter.py           # Text chunking logic
├── engines.py                  # Registry of the optional engines (Drama 3, Breeze)
├── engine_servers.py           # Launch/stop/health of their servers (stdlib only)
├── remote_engines.py           # HTTP clients + RemoteAudioGenerator
├── model_store.py              # Weight downloads, manifest, status (stdlib only)
├── voice_transcripts.py        # Reference clips + transcripts (stdlib only)
├── drama_markup.py             # Drama 3 tags (stdlib only)
├── tag_enhancer.py             # LLM that writes Drama 3 tags
└── _vendor/echo_tts/           # Vendored inference code (don't modify unless necessary)
```

## Optional Engines

Drama 3 and Breeze run in their own servers (inside WSL on Windows) because they
pin a different torch/transformers than Echo. Rules that follow from that:

- **Never add their dependencies to `pyproject.toml`.** CI installs with
  `uv sync --locked`; the engines live in `~/candyecho-engines`, set up by
  `scripts/setup_engines.sh`.
- **HTTP to the engines uses `urllib`**, for the same reason: no new dependency.
- **Echo stays the default and untouched.** `engine` defaults to `"echo"` in
  every request model, and `AudioGenerator` is not modified. Remote engines go
  through `RemoteAudioGenerator`, which mirrors its interface but takes a
  `VoiceReference` (clip + transcript) instead of speaker latents.
- **Sample rate is per engine** (Breeze is 24 kHz). Pass it through; do not
  hard-code 44100 in new code.
- **Engine markup must be hidden from the text pipeline.** The cleaner and
  normalizer spell `<soft>` out as words. `_prepare_text_chunks` in `main.py`
  wraps the pipeline with `protect_markup` for engines that have script tags;
  route any new entry point through it rather than calling the stages directly.
- **Chunk size is per engine** (`Engine.chunk_limits`). Echo's 220-character
  limit comes from its 30-second cap and does not apply to the others, except
  with best-of-N on, where takes must fit the verifier's 30-second window.
- **A server that answers its health check is "ready"**, however it was
  started. Do not track readiness any other way.
- **A model is "downloaded" only if `models/<key>/candyecho-model.json` exists
  and every file in it is present at the recorded size.**
- Tests fake the engine servers and Hugging Face; none need a GPU or network:

```bash
uv run pytest tests/test_model_store.py tests/test_engine_servers.py tests/test_remote_engines.py tests/test_engines_api.py tests/test_tag_enhancer.py
```

## Text Pipeline

Order matters — each stage assumes the previous one ran:

```
extract → clean → normalize → segment → generate
```

`text_cleaner` must run before `text_segmenter`, because the segmenter turns a
single newline into a comma and a blank line into a period. Page furniture left
in place is therefore *spoken*, and a sentence split across a page break is
severed permanently. `text_cleaner` and `text_normalizer` are stdlib-only and
import no torch, so they can be tested without the CUDA stack:

```bash
uv run pytest tests/test_text_cleaner.py tests/test_text_normalizer.py tests/test_text_segmenter.py
```

## Testing

- Tests are in `tests/` directory
- Use `uv run pytest -v` for verbose output
- Tests mock the heavy ML components - they don't require GPU

## Voice Library

- Voice `.wav` files go in `voice_library/`
- Voices can be added at runtime: upload a `.wav` via the web UI (`POST /voices`) or drop one into `voice_library/` (directory watcher)
- Preprocessed voices are cached as `.pkl` files
- Cache invalidates automatically if `.wav` file changes
