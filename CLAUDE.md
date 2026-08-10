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
└── _vendor/echo_tts/           # Vendored inference code (don't modify unless necessary)
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
