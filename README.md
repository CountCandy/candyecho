# CandyEcho: Sweet-Talking, Long-Form TTS 🍬

Generate coherent audio for arbitrary-length text using Echo-TTS, which has a ~30-second generation limit per inference call. CandyEcho wraps it in a friendly web UI and an OpenAI-compatible API, so you can also use it as a TTS backend for other apps.

## Features

- Generate audio for arbitrary-length text
- Maintains voice coherence across chunks using blockwise inference
- Real-time streaming — audio plays as it generates
- Web interface with dark, light, and 🍬 candy themes
- Organize voices into **Sweet Treats** (favorites, expanded) and **Unwrapped Candy** (a space-saving dropdown), with preview, rename, durations, and drag-and-drop upload
- Optional volume normalization (even out voices that come out too quiet or loud)
- Download generated audio as WAV or MP3
- OpenAI-compatible TTS API — use CandyEcho as a backend for SillyTavern and other apps
- One-click `run.bat` launcher on Windows (no terminal needed)
- Voice library with automatic preprocessing and caching
- Text normalization for better TTS output (currencies, abbreviations, etc.)
- **Textbook cleaning** — strips page headers, typesetter stamps and footnote markers from page-extracted text, and rejoins sentences split across page breaks

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

#### Choosing reference audio

Echo-TTS's own guidance is: *"You can condition on up to 5 minutes of reference
audio, but shorter clips (e.g., 10 seconds or shorter) work well too."* The
5-minute figure is the supported ceiling, and the code agrees exactly —
`max_speaker_latent_length` is 6400 latents, which at 2048 samples each is 297
seconds.

**Clean beats long.** Upstream does not claim longer references are better, and
the community tooling around Echo emphasises the opposite: clip, de-noise, and
isolate the vocal first. Everything in the reference — room tone, music, a second
speaker, an interviewer — is averaged into the conditioning, so a noisy minute
will usually lose to a clean fifteen seconds.

**Longer clips are not free at generation time.** Only the `.pkl` preprocessing
is cached. `get_kv_cache_speaker` runs once per chunk, is tripled for CFG, and is
attended at every diffusion step, so the reference length multiplies with the
best-of-N batch size. At `speaker_patch_size=4`:

| Reference | Latents | Speaker tokens |
| --- | --- | --- |
| 15 s | 323 | 80 |
| 60 s | 1,292 | 323 |
| 5 min | 6,400 | 1,600 |

Worth knowing: the conditioning is a latent *sequence* the model attends over,
not a pooled speaker embedding, so in principle a longer clip exposes more of the
speaker's prosody and not just their timbre. Whether that measurably improves
style transfer is untested — treat it as an experiment rather than a rule, and
match the reference to the material either way (for a textbook, measured
narration rather than animated conversation).

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

### Textbook Cleaning

Text copied out of a PDF, an e-reader, or a print proof carries things that must
never be spoken, and it arrives broken. A page break severs the sentence that
spans it, and the page furniture lands in the gap:

```
... the tax staple of most of the rest of the world. Once in control

Se

9299_001.indd 2

12/13/2016 1:58:29 PM

PROPERTY OF THE MIT PRESS FOR PROOFREADING ... ONLY  Introduction 3

of the House, though, Ryan, the new chair of the Budget Committee, ...
```

Fed to the segmenter directly, every one of those blank lines becomes a full
stop, so the reader hears *"Once in control. Se. 9299_001.indd 2. 12/13/2016
1:58:29 PM. PROPERTY OF THE MIT PRESS…"* — and the real sentence never
reassembles.

The **📖 Textbook cleaning** panel in the web UI fixes this before anything else
runs. Rules can be toggled individually, or set with a preset:

| Preset | What it does |
| --- | --- |
| `textbook` (default) | Everything below |
| `light` | Only the safe, non-structural rules: character folding, reflow, URLs, spaced initials, dictionary |
| `off` | Pass-through, nothing is changed |

Rules include: typesetter stamps (`9299_001.indd 2`, timestamps), ALL-CAPS proof
notices, running headers and page numbers (detected by repetition across the
document), rejoining split sentences and hard-wrapped lines, footnote reference
digits (`credits.2` → `credits.`), inline citations (`[12]`,
`(Smith et al., 2019)`), figure and table bodies (the caption is kept, the data
rows dropped), formula lines, URLs/DOIs/ISBNs, character folding (smart quotes,
ligatures, dashes), repair of compounds that lost their hyphen
(`upperincome` → `upper-income`), spaced initials (`J.R.R.` → `J R R`), and a
per-project pronunciation dictionary.

These are heuristics, so the panel has a **Preview cleaning** button that reports
what each rule removed, with samples, and can apply the result to the text box.
Worth a look before committing to a long book.

### Expressiveness, emotion and sampler controls

Echo has **no temperature**. It is a flow-matching model sampled with a
deterministic Euler ODE solver, so there is no softmax to heat up — all the
randomness lives in the initial noise draw. The advanced panel exposes the
controls that do exist:

| Control | Parameter | Notes |
| --- | --- | --- |
| Quality | `num_steps` (40) | More steps, slower, marginal gains past ~40 |
| Text guidance | `cfg_scale_text` (3.0) | Higher sticks to the words |
| Voice guidance | `cfg_scale_speaker` (8.0) | Upstream's own quick-start value |
| Expressiveness | `truncation_factor` (1.0) | The nearest thing to temperature |
| Force speaker | `speaker_kv_scale` (off) | Pulls a drifting voice back |

**Expressiveness** scales the starting noise. Below 1.0 is safer, steadier and
flatter; above 1.0 gives more varied delivery but drops and invents more words.
Echo's own README puts the same tradeoff a different way: *"Exclamation points
(and other non-bland punctuation) may lead to increased expressiveness but also
potentially lower quality on occasion."* This is exactly what Best-of-N buys
back — with the verifier catching errors you can afford to push expressiveness up.

**Force speaker** applies KV scaling to the speaker conditioning. Upstream:
*"Aim for the lowest scale that produces the correct speaker: 1.0 is baseline,
1.5 is the default when enabled and will usually force the speaker, but lower
values (e.g., 1.3, 1.1) may suffice."* It constrains delivery as it climbs, so
leave it off unless the voice is actually drifting.

#### Directing emotion in the text

Echo was trained on [WhisperD](https://huggingface.co/jordand/whisper-d-v1a)
transcriptions, whose format annotates non-speech events in **parentheses** —
Darefsky's own example is `[S1] Hey! [S2] (sighs) Um, how's it going?`. So
delivery is directed in the text:

```
She read the last line. (sighs) It had all been for nothing.
```

Recognized events keep their parentheses through normalization; ordinary
parenthetical prose is still flattened into commas, so `(see chapter four)` is
spoken normally. Disable with `keep_sound_tags: false` if you want the old
behaviour.

**There is no official list of supported events.** Only `(laughs)`, `(coughs)`
and `(sighs)` appear in the upstream write-ups. CandyEcho recognizes a wider set
of plausible neighbours — `(chuckles)`, `(gasps)`, `(clears throat)`,
`(whispers)`, `(groans)`, `(yawns)`, `(scoffs)`, `(sobs)` and others (see
`SOUND_TAGS` in `text_normalizer.py`) — but these are *suggestions to test*, not
a guaranteed vocabulary. A tag the model never saw in training will simply be
ignored or, worse, voiced. Test one before sprinkling it through a book.

Punctuation is the more reliable lever, and it always works.

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

### Best-of-N Takes (accuracy check)

Echo occasionally drops or invents a word. With **🎧 Best-of-N takes** enabled in
the advanced panel, each chunk is generated several times in a single batched
diffusion pass, every take is transcribed, and the one that actually matches the
text is kept.

This matters more here than in a stateless pipeline: the winning take is
re-encoded into the continuation latent that seeds the *next* chunk, so a bad
take does not merely sound wrong, it poisons everything after it. Catching it
stops the error cascading.

How a take is judged:

- **Two ASR families, rank-averaged.** WhisperD (`jordand/whisper-d-v1a`) was
  fine-tuned on the same transcription format Echo was trained against, so it
  renders `[S1]` tags, disfluencies and `(laughs)` natively rather than scoring
  them as errors. A CTC model (wav2vec2) is a different architecture whose errors
  decorrelate, and which structurally cannot produce the runaway repetition that
  attention decoders hallucinate. Neither model's absolute numbers are trusted —
  only the rankings they agree on.
- **Deletions weigh more than substitutions.** A dropped word is the failure
  worth catching; a substitution is often just the ASR mishearing.
- **Beyond word error:** duration outliers (rushed, dragging or looping),
  repetition loops, truncated endings, and speaker drift — a take that says every
  word correctly in a voice that has wandered off the reference.
- **Automatic retries.** If the best take is still above threshold, another batch
  is generated, up to the round limit. No prompting, no manual re-rolls.

Nothing extra to install — `transformers` ships as a normal dependency. The model
**weights** are what's large (roughly 4–5 GB across the three models) and they
download on first use, inside your first verified generation, so expect that one
to sit at "Loading take-verification models" for a while. Setting `HF_TOKEN`
gives you faster, rate-limit-free downloads from Hugging Face.

Models are configurable by environment variable, and any of them can be switched
off by setting it to `none`. If one fails to load it is skipped with a warning
rather than failing the generation — losing the speaker check, or falling back to
a single ASR, still beats aborting a multi-hour book:

| Variable | Default |
| --- | --- |
| `CANDYECHO_VERIFY_WHISPER` | `jordand/whisper-d-v1a` |
| `CANDYECHO_VERIFY_CTC` | `facebook/wav2vec2-large-960h-lv60-self` |
| `CANDYECHO_VERIFY_SPEAKER` | `microsoft/wavlm-base-plus-sv` |
| `CANDYECHO_VERIFY_DEVICE` | `cuda` |

They load on first use, not at startup, so you only pay for them when the option
is on. Verification is off by default on `/v1/audio/speech`, which serves
interactive chat rather than long reads.

**Expressiveness.** Accuracy and expressiveness trade off directly, which is what
makes this feature worth its cost: with the verifier catching errors, you can
raise `truncation_factor` for livelier delivery instead of playing it safe.

### Saved Batches, Resume and Subtitles

Every generation is written to disk as it runs — one file per chunk plus a
manifest under `jobs/<id>/`. Closing the tab, refreshing, or losing power no
longer destroys the work, and assembling the final file happens server-side, so
a multi-hour book is no longer limited by what the browser can hold in memory.

The **🗄️ Saved batches** panel lists everything with a download for WAV, MP3 or
SRT. An unfinished batch gets a **Resume** button: generation restarts at the
first chunk that never completed, and the last finished chunk's audio is
re-encoded into the continuation latent, so the join sounds the same as it would
have in an uninterrupted run.

Subtitles are timed from each chunk's measured duration — exact at chunk
boundaries, with cues split at sentence boundaries and time apportioned by
character count within a chunk.

```
GET    /jobs                       list saved batches
GET    /jobs/{id}                  full manifest, per-chunk scores and seeds
GET    /jobs/{id}/audio?format=    wav | mp3 | flac | ogg
GET    /jobs/{id}/subtitles?format= srt | vtt
POST   /jobs/{id}/resume           continue an interrupted batch (SSE)
DELETE /jobs/{id}                  delete the batch and its audio
```

Set `CANDYECHO_JOBS_DIR` to store them somewhere other than `jobs/`.

Progress events carry a **remaining-time estimate**, shown next to the progress
bar. It averages a trailing window of chunk times rather than the whole run,
because best-of-N retries make individual chunks lumpy and a long book's early
chunks stop being representative.

### Measuring quality changes

`scripts/ab_quality.py` turns "is this better?" into a number. It generates the
same text under each variant with a single take per chunk — best-of-N is
deliberately off, since the point is to measure the setting itself rather than
how well the verifier compensates — then scores every chunk with the same ASR
ensemble the live path uses.

```bash
# Does a longer reference clip actually help?
uv run python scripts/ab_quality.py --text page.txt --voices Ranni15s,Ranni60s,Ranni3m

# What does expressiveness cost in accuracy?
uv run python scripts/ab_quality.py --text page.txt --voice Ranni60s \
    --vary truncation=0.8,1.0,1.2 --repeats 3 --csv results.csv

# Is Force Speaker worth it?
uv run python scripts/ab_quality.py --text page.txt --voice Ranni60s \
    --vary speaker_force=1.0,1.2,1.5
```

Reports mean and worst WER, CER, speaker similarity, and generation time per
second of audio, with optional per-chunk CSV. Use `--repeats` for anything
marginal — a single run of a diffusion model is a noisy sample.

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
- `GET /voices` - List available voices (each with its reference-audio duration)
- `POST /voices` - Upload a `.wav` voice sample (multipart form field `file`); it's saved to `voice_library/`, preprocessed, and added to the voice list
- `GET /voices/{name}/audio` - Download/preview a voice's reference `.wav`
- `POST /voices/{name}/rename` - Rename a voice. JSON body: `{"new_name": "..."}`
- `GET /cleaning-rules` - List the textbook-cleaning rules and presets
- `POST /clean-text` - Preview cleaning. JSON body: `{"text": "...", "preset": "textbook", "rules": {}, "substitutions": {}}`; returns the cleaned text plus a per-rule report of what was removed
- `POST /generate` - Generate audio (SSE stream). JSON body: `{"text": "...", "voice": "...", "normalization_level": "moderate", "normalize_volume": false, "cleaning": {"enabled": true, "preset": "textbook"}}` (`cleaning` is optional; omit it to skip cleaning entirely)
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
│       ├── text_cleaner.py      # Textbook/page-extract cleaning
│       ├── text_extractor.py    # .txt / .epub import
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
