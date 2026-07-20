# CandyEcho: Sweet-Talking, Long-Form TTS 🍬

Generate coherent audio for arbitrary-length text using Echo-TTS, which has a ~30-second generation limit per inference call. CandyEcho wraps it in a friendly web UI and an OpenAI-compatible API, so you can also use it as a TTS backend for other apps.

## Features

- Generate audio for arbitrary-length text
- Maintains voice coherence across chunks using blockwise inference
- Real-time streaming — audio plays as it generates
- Web interface with dark, light, and 🍬 candy themes
- Voice panel: preview, rename, and favorite/organize your voices — plus drag-and-drop upload
- Optional volume normalization (even out voices that come out too quiet or loud)
- Download generated audio as WAV or MP3
- OpenAI-compatible TTS API — use CandyEcho as a backend for SillyTavern and other apps
- One-click `run.bat` launcher on Windows (no terminal needed)
- Voice library with automatic preprocessing and caching
- Text normalization for better TTS output (currencies, abbreviations, etc.)

## Quick Start

### 1. Install Dependencies

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable package management.

**Install uv (if not already installed):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install project dependencies:**
```bash
uv sync
```

This installs PyTorch 2.11 with CUDA 13.0 support from the PyTorch index (`download.pytorch.org/whl/cu130`). CUDA wheels for Windows and Linux are only published there — PyPI's Windows `torch` is CPU-only.

#### Windows: FFmpeg

On Windows, `torchcodec` loads FFmpeg's **shared** libraries (the `av*.dll` files) at import time, and it supports only **FFmpeg 4–8** (`avutil-56.dll` … `avutil-60.dll`). Without a compatible one the server won't start.

Download a **shared FFmpeg 8** build — the `ffmpeg-n8.x-latest-win64-gpl-shared` asset from [BtbN's builds](https://github.com/BtbN/FFmpeg-Builds/releases) (it ships `avutil-60.dll`, `avcodec-62.dll`, …). Extract it and add its `bin\` folder to your `PATH`, then open a new terminal.

> Pitfalls:
> - **FFmpeg 9 is too new.** `git-master` / FFmpeg 9 builds ship `avutil-61.dll`, which torchcodec 0.11 can't load. Verify with `ffmpeg -version`: you want `libavutil 60.x` (or 56–59), **not** 61.
> - **Static builds don't work.** The default / "essentials" / "full" builds (and `winget install ffmpeg`) are static (`ffmpeg.exe` only, no DLLs). You need the *shared* build.
>
> LongEcho adds FFmpeg's `bin` from your PATH to the DLL search at startup (Python 3.8+ no longer searches PATH for a DLL's dependencies), so having the shared build on PATH is enough — no need to copy DLLs.

### 2. Add Voice Samples

Place `.wav` files in the `voice_library/` directory:

```bash
cp path/to/your/voice.wav voice_library/
```

The first time you run the app, it will preprocess these files and cache them as `.pkl` files for fast loading.

You can also add voices at runtime from the web interface — drop a `.wav` onto the upload area (or click it to browse). The voice is preprocessed and ready to use without restarting, so there's no need to pre-populate `voice_library/`.

### 3. Run the Server

**Windows:** double-click **`run.bat`** — it starts the server and opens your browser (no terminal needed). Just run `uv sync` once first.

Or from a terminal:
```bash
uv run python -m longecho.main
```

Or with uvicorn directly (use `--host 0.0.0.0` to expose on your local network — needed to reach it from another machine, e.g. SillyTavern):
```bash
uv run uvicorn longecho.main:app --port 8100 --reload
```

### 4. Open Web Interface

Visit http://localhost:8100 in your browser.

1. Enter your text (any length)
2. Select a voice from the dropdown
3. Click "Generate Audio"
4. Audio streams as it generates
5. Download the result as WAV or MP3

## How It Works

### Text Normalization

Before generation, text is normalized for better TTS output:

- **Currencies**: `$5M` → "5 million dollars", `$99.99` → "99 dollars and 99 cents"
- **Abbreviations**: `Dr.` → "Doctor", `etc.` → "etcetera", `vs.` → "versus"
- **Parentheses**: Content is preserved but parens removed — Echo-TTS uses WhisperD format where text in parentheses denotes sound effects rather than speech

Two normalization levels available via API:
- `moderate` (default): Normalize currencies and abbreviations, preserve plain numbers
- `full`: Also convert numbers to words

### Text Segmentation

Text is split into ~160-220 character chunks at natural boundaries:

1. Sentence boundaries (`.`, `!`, `?`)
2. Clause separators (`,`, `;`)
3. Word boundaries (spaces)
4. Hard cut if no boundary found

### Contextual Generation

Text is chunked to ~12-15 seconds of audio each, so that a previous chunk plus a new chunk fit within Echo-TTS's ~30-second (640-latent) generation window.

1. **First chunk**: Generated fresh with the selected voice reference
2. **Subsequent chunks**: The full previous chunk's audio is re-encoded through the Fish autoencoder and passed as a continuation latent, seeding the diffusion process. The previous chunk's text is also prepended so the model sees the text-audio alignment. After generation, the continuation portion is trimmed so only new audio is emitted.
3. **Streaming**: Each chunk's new audio is sent to the browser via SSE as soon as it's ready

### Voice Management

- Voices are preprocessed using Fish autoencoder + PCA
- Results cached as `.pkl` files for fast loading
- Cache automatically invalidates if `.wav` file changes
- New `.wav` files are detected automatically via file watcher
- Voices can also be uploaded directly from the web interface (drag-and-drop or click to browse)

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 13.0+ (driver R580 or newer)
- 8GB+ VRAM recommended
- [uv](https://docs.astral.sh/uv/) package manager
- **Windows only:** FFmpeg shared libraries in PATH (see installation instructions)

## API Endpoints

- `GET /` - Web UI
- `GET /voices` - List available voices
- `POST /voices` - Upload a `.wav` voice sample (multipart form field `file`); it's saved to `voice_library/`, preprocessed, and added to the voice list
- `GET /voices/{name}/audio` - Download/preview a voice's reference `.wav`
- `POST /voices/{name}/rename` - Rename a voice. JSON body: `{"new_name": "..."}`
- `POST /generate` - Generate audio (SSE stream). JSON body: `{"text": "...", "voice": "...", "normalization_level": "moderate", "normalize_volume": false}`
- `POST /stop` - Stop an in-progress generation. Optional query param: `generation_id`
- `GET /voice-events` - SSE stream of voice library changes (processing, ready, removed, renamed, error)
- `GET /health` - Health check

### OpenAI-compatible TTS API

- `POST /v1/audio/speech` - Non-streaming synthesis returning a single audio file. JSON body: `{"input": "...", "voice": "<voice name>", "response_format": "wav|mp3|flac|ogg"}` (`model` and `speed` are accepted but ignored).
- `GET /v1/audio/voices` - List voices.

**Use with SillyTavern:** Extensions → TTS → set **TTS Provider** to **OpenAI Compatible**. Provider Endpoint `http://localhost:8100/v1/audio/speech`, Model `candyecho`, API Key `not-needed`, then enter your voice names under Available Voices. (The same steps are in the app's "Use CandyEcho as a TTS backend" panel.) If SillyTavern runs on another machine, start CandyEcho with `--host 0.0.0.0` and use this PC's LAN IP.

## Development

### Project Structure

```
longecho/
├── src/
│   └── longecho/
│       ├── __init__.py
│       ├── main.py              # FastAPI server
│       ├── text_normalizer.py   # TTS text preprocessing
│       ├── text_segmenter.py    # Text chunking
│       ├── voice_manager.py     # Voice preprocessing & caching
│       ├── audio_generator.py   # Audio generation with continuation
│       ├── file_watcher.py      # Voice file auto-detection
│       ├── voice_event_broadcaster.py  # SSE voice events
│       └── _vendor/
│           └── echo_tts/        # Vendored Echo-TTS inference code
├── voice_library/               # Voice .wav files (user-provided)
├── static/
│   ├── index.html               # Web UI
│   ├── style.css                # Styles
│   ├── app.js                   # Client-side logic
│   └── lamejs.min.js            # Vendored MP3 encoder
├── tests/                       # Test suite
└── pyproject.toml
```

### Running Tests

```bash
uv run pytest
```

Or with verbose output:
```bash
uv run pytest -v
```

## License

MIT — see [LICENSE](LICENSE).

## Credits

Built with [Echo-TTS](https://github.com/jordandare/echo-tts) by Jordan Darefsky.

### Third-Party Licenses

This project includes vendored code from Echo-TTS:

- **Echo-TTS** by Jordan Darefsky - MIT License
- **autoencoder.py** - Apache-2.0 License (derived from Fish Speech)
- **Model weights** - CC-BY-NC-SA-4.0 (non-commercial use only)

See `src/longecho/_vendor/echo_tts/LICENSE` for full license text.

- **[lamejs](https://github.com/zhuker/lamejs)** by zhuker - LGPL-3.0 License (client-side MP3 encoding)

See `static/LICENSE-lamejs` for details.
