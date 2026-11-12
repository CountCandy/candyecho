# CandyEcho: Sweet-Talking, Long-Form TTS 🍬

Generate coherent audio for arbitrary-length text using Echo-TTS, which has a ~30-second generation limit per inference call. CandyEcho wraps it in a friendly web UI and an OpenAI-compatible API, so you can also use it as a TTS backend for other apps. Fish Audio Drama 3 can be added as an [optional second engine](#drama-3-an-optional-second-engine) and chosen per generation.

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
- **Optional second engine** — Fish Audio Drama 3, downloaded and started from the web UI, with multi-clip cloning, a clip audition that measures which references clone best, and a tag enhancer that writes directions into your script
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

**Favour voice match.** By default Best-of-N keeps the take that says the text
most accurately. Tick *Favour voice match* and accuracy becomes a gate instead:
every take that says the text well enough is eligible, and the one that sounds
most like your reference voice is kept. A take with wrong words is never chosen
for its voice, and if no take passes the gate the normal ranking applies. The
progress line shows the kept take's voice match and how far the takes ranged,
so you can see what the choice bought. This is most useful with Drama 3, whose
takes differ more from one another than Echo's.

**Voice match of the finished audio.** Every generation, with or without
Best-of-N, is rated against the selected sample by the same speaker model
(`microsoft/wavlm-base-plus-sv`, about 400 MB, fetched on first use). When the
batch finishes, a line under the progress bar gives the length-weighted average
and the lowest chunk; each chunk's figure is shown in the saved batch's chunk
list and written to the terminal as the chunk completes, along with which take
was kept:

```
Chunk 1/2 · kept take 2 of 3 (picked for voice) · word error 0.0% · voice match 87.3% (takes 81.2-87.3%)
Job 20261111-191825-8c140f: voice match to 'EC1_best' 86.1% over 2 chunk(s), lowest 84.4% (chunk 2)
```

Read the figure as a comparison between runs, clips and settings rather than
as a grade. It is cosine similarity mapped onto 0–100%, measured against the
selected voice only (not the extra reference clips), and short chunks score
less steadily than long ones. Set `CANDYECHO_VERIFY_SPEAKER=none` to switch it
off.

The models load once and stay in memory, so their loading messages (including
a `masked_spec_embed | MISSING` report from wav2vec2, which is expected and
harmless) appear only on the first run after starting CandyEcho.

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

Each saved batch is listed with its first ten words, so batches are easy to
tell apart, and a **▶ Play** button that plays the assembled audio in the page.
The list shows the five most recent batches; a button underneath reveals the
older ones. Nothing is deleted automatically.

**Redoing one chunk.** If a single chunk came out badly, there is no need to
regenerate the batch. Press **✂ Chunks** on the batch to list its chunks, **▶**
to hear one, and **↻ Redo** to generate that chunk again with a new seed. The
new take replaces the old one in the batch, so Play and the WAV/MP3 downloads
use it straight away. The same engine, voice, extra clips and sampler settings
as the original run are used. With Echo the new take continues from the chunk
before it, as the original did; the chunk after it was generated to follow the
old take, so that one join can be slightly less smooth.

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

## Drama 3: an optional second engine

Echo is built in and is always the default. Fish Audio Drama 3 can be added and
picked per generation from the **Engine** menu at the top of the page.

| Engine | Good at | Output | VRAM | Weights licence |
| --- | --- | --- | --- | --- |
| **Echo-TTS** (built in) | Voice match from audio alone; seamless long reads | 44.1 kHz | 8 GB+ | CC-BY-NC-SA-4.0 |
| **Fish Audio Drama 3** | Directed, emotional delivery via tags in the script | 44.1 kHz | 48 GB+ | Fish Audio Community License |

### How it is run

Drama 3 cannot be loaded into CandyEcho's own process: it pins a different
PyTorch and `transformers` than Echo, and is Linux-only. It ships its own HTTP
server, so CandyEcho starts that server as a child process (inside **WSL** on
Windows) and sends it requests over localhost. Everything else — text cleaning,
chunking, saved batches, resume, subtitles, Best-of-N and the OpenAI-compatible
endpoint — works the same for both engines.

### One-time setup

1. **Windows only:** install WSL if you have not (`wsl --install` in an
   administrator terminal, then reboot).
2. Install the engine's code. In a WSL terminal (or any terminal on Linux),
   from the CandyEcho folder:

   ```bash
   bash scripts/setup_engines.sh
   ```

   This puts the engine and its Python environment in `~/candyecho-engines`.
   It asks for your Linux password once, to install FFmpeg and audio headers.

That is the only terminal step. It does not download any model weights.

### Downloading and using the model

Pick **Fish Audio Drama 3** in the **Engine** menu. Its card shows two steps:

1. **Model files** — press **Download**. Progress is shown in the card. If the
   download is interrupted or CandyEcho is closed, press the button again and
   it carries on from where it stopped; finished files are kept.
2. **Engine** — once the files are complete the server is started for you and
   the badge turns to *ready*. No restart of CandyEcho is needed. If you start
   a generation while the engine is stopped, it is started on demand.

The engine's status is checked when you switch engine, after a generation, and
while a download or start is in progress. Press **↻ Check status** to check at
any other time.

Weights are stored under `models/` (override with `CANDYECHO_MODELS_DIR`):

```
models/
├── drama3/
│   ├── candyecho-model.json     # written when the download is complete
│   ├── drama-3/                 # fishaudio/drama-3
│   └── s2-pro/codec.pth         # the codec Drama 3 decodes with
└── enhancer/
    ├── <model repo name>/       # the tag-enhancer model file
    └── llama.cpp/               # the server that runs it (Windows)
```

`candyecho-model.json` lists every file and its size. A model counts as
downloaded only when that file exists and every listed file is still on disk at
the recorded size, so CandyEcho never offers to re-download something that is
there, and notices if part of it was deleted. If you already fetched the weights
yourself into these folders, press **Download** once: existing files are
verified and skipped rather than fetched again.

If Hugging Face refuses a download, open the model page linked in the card,
accept its licence while signed in, run `hf auth login` (or set `HF_TOKEN`), and
press the button again.

### Start-up time

Drama 3 takes a while to become ready: the server loads about 38 GB of weights
and then runs a warm-up generation. The card shows how long the last start took.
Two things help:

- **Start when CandyEcho opens.** Tick this on the engine's card and the server
  is launched as soon as CandyEcho starts, so the model loads while Echo's
  voices and the page come up instead of when you first press Generate.
- **Leave it running.** Switching the Engine menu back to Echo does not stop
  Drama 3. With enough VRAM both can stay loaded, so the wait is paid once per
  session. Use **Stop** on the card only when you need the memory back.

On Windows the weights are read from a Windows drive through WSL's file bridge,
which is slower than WSL's own disk. If start-up is dominated by reading the
files (watch whether VRAM fills slowly), keeping the weights on WSL's disk may
help. Copy them once, from a WSL terminal:

```bash
mkdir -p ~/candyecho-models
cp -r /mnt/c/path/to/candyecho/models/drama3 ~/candyecho-models/
wslpath -w ~/candyecho-models/drama3      # prints the Windows form of the path
```

then tell CandyEcho where they are in `candyecho-settings.json` (next to
`run.bat`; create it if it does not exist), using the path `wslpath` printed
with every backslash doubled:

```json
{ "model_dirs": { "drama3": "\\\\wsl.localhost\\Ubuntu\\home\\you\\candyecho-models\\drama3" } }
```

This needs about 41 GB free on the drive that holds WSL's disk (usually `C:`).
If VRAM already fills quickly and the wait comes afterwards, the time is going
into the model's own initialisation and warm-up, and moving the files will not
change it.

### Reference transcripts

Echo clones from the reference audio alone. Drama 3 also needs the exact words
spoken in the reference, and wants a short clip, so for it a voice is reduced to
the **first 30 seconds** of its `.wav` (`CANDYECHO_REFERENCE_SECONDS`) plus a
transcript of that clip.

The transcript box appears under the voice list when Drama 3 is selected. Leave
it empty and the clip is transcribed with Whisper the first time you generate;
or press **Transcribe automatically** and correct any mistakes. Transcripts are
saved as `voice_library/<voice>.transcript.txt` and follow the voice when it is
renamed or deleted. An accurate transcript noticeably improves the clone, so it
is worth a glance.

A reference clip that is very quiet (peak below half of full scale) is raised
to a normal level before it is sent, because the engine copies the level of the
reference along with the voice. Set `CANDYECHO_REFERENCE_BOOST=0` to send clips
untouched.

### Extra reference clips

Drama 3 can clone from several recordings of a speaker at once, and usually
matches the voice better when it has more to go on. Under the transcript box,
**Extra reference clips** shows the clips you have added as removable chips and
a search box for adding more: type part of a voice name and click a match (or
press Enter for the first one). Voices named like the selected one (`mina`,
`mina 2`, `mina_angry`) are offered without typing. Nothing else in the library
is listed, so the box stays the same size however many voices you have. Up to 8
clips of the **same speaker** can be added; each needs its own transcript,
filled in automatically on first use like the main one. The choice is
remembered per voice.

This is the way to give Drama 3 more than 30 seconds of a speaker: several
short clips, rather than one long one.

### What each engine wants in a sample

The same guide is in the app under **What makes a good sample**, and follows
the selected engine.

| | Echo | Drama 3 |
|---|---|---|
| Length | 5–15 seconds. Up to about five minutes is accepted, but longer rarely sounds better | 10–30 seconds per clip; the first 30 seconds of each is used |
| How many | One clip per voice | Two or three, ideally 15–20 seconds each |
| Tone | Consistent, and already in the delivery you want: there are no tags to change it afterwards | Steady inside each clip; use separate clips to cover different moods. Direction tags add emotion later |
| Transcript | Not used | Must match the audio word for word |
| Recording | One speaker, no music, little room echo, no clipping | The same, plus steady volume and short pauses between sentences |
| If it drifts | Raise **Force speaker** | Add clips, then **Audition** them |

The Echo figures come from its author's notes (clips of ten seconds or less
work well) and from users' reports that very long clips do not usually help.
The Drama 3 figures are Fish Audio's guidance for its hosted service, which
runs the same family of models. Fish's own pages disagree about tone: the
best-practices guide lists consistent tone and emotion as a must-have, the SDK
reference asks for varied intonation, and the feature page says several samples
capture more range. The table's reading of that (steady within a clip, varied
across clips) is an interpretation, not something Fish states.

### Do temperature, top-p and top-k do anything?

Yes. Drama 3's sampler divides the model's scores by the temperature, then
keeps only the top-k most likely audio tokens and, within those, the ones that
fit inside the top-p share of probability, and draws from what is left. Two
details limit how much you hear:

- The single most likely token is always kept, so very low settings converge on
  the same steady reading rather than breaking.
- When a draw repeats a token used in the last 16 frames, the server redraws
  it with its own fixed settings (temperature 1.0, top-k 30, top-p 0.95) to
  avoid stuck sounds. Your settings do not apply to those redraws.

The sliders apply to every layer of audio tokens, so they shape wording, rhythm
and intonation as well as fine detail. Lower values give a steadier, flatter
read; higher values give more varied delivery and more slips. They are not a
"closer to the clip" control, though lowering them can make the voice more
stable from line to line. How closely it matches comes mostly from the reference
clips.

### Auditioning clips

Not every clip clones equally well, and more clips is not always better. With
at least one extra clip ticked, **🎧 Audition these clips** finds out which to
use. It generates the same two-sentence line:

- from each clip on its own,
- from all of them together, and
- with three or more clips, from all-but-one in turn, which shows up a clip
  that drags the set down.

Every result is scored by how close it sounds to the speaker, using the same
speaker-embedding model as Best-of-N, measured against **all** the clips rather
than only the ones it was cloned from. Each trial uses the same seeds, so only
the clips differ. Results are listed best first, each with a **▶** to hear its
best take and a **Use this set** button that selects those clips.

Up to six clips are compared at a time. Three clips means seven trials of two
takes each, so allow a few minutes. The score is an automatic proxy: differences
under about one point are noise, so listen before deciding.

### What differs from Echo

- **No chunk-to-chunk continuation.** Echo feeds each chunk's audio into the
  next so joins are seamless. Drama 3 voices every chunk from the same
  references instead, so joins on long reads can be slightly less smooth.
- **Best-of-N costs time, not memory.** Takes are separate requests, so three
  takes take three times as long.
- **Sampler controls are per engine.** Echo's steps/guidance/truncation sliders
  are replaced by temperature, top-p and top-k, which are the sampling controls
  the Drama 3 server accepts per request.
- **Chunks are larger.** Echo is limited to about 220 characters a chunk by its
  30-second cap. Drama 3 gets up to 700 (its server splits long requests itself
  and carries context across the pieces), so short passages are not cut up and
  long ones have fewer joins. With Best-of-N on, it drops to 330 so each take
  fits the verifier's 30-second window.

### Freeing VRAM: unloading Echo

Each engine keeps its model in VRAM while it is loaded (roughly 48 GB for
Drama 3, plus Echo, the verifier models and the tag-enhancer LLM). Use **Stop**
on Drama 3's card to free its memory.

Echo can be parked too: with Echo selected, press **Unload**. The model moves
to system RAM and its VRAM is released; it moves back in a few seconds the next
time you generate with Echo, or when you press **Load now**.

### Drama 3 direction tags and the tag enhancer

Drama 3 is directed by markup written into the script:

| You write | What happens |
| --- | --- |
| `[warm, unhurried]` | A line direction in plain words; holds until the next direction |
| `[sigh]` `[laugh]` `[breath]` `[pause]` | A sound or beat at that point |
| `<whisper>come alone</whisper>` | A paired tag changes only the words inside it |

Paired tags: `<whisper>` `<emphasis>` `<soft>` `<fast>` `<slow>` `<stress>` `<forceful>`.

Because long text is sent one chunk at a time, CandyEcho repeats the direction
still in force at the start of each chunk, and closes and reopens a paired tag
that spans a chunk edge, so the markup behaves as if the script were sent whole.
The markup is also shielded from text cleaning and normalisation, which would
otherwise spell `<soft>` out as words.

The **✨ Add direction tags** button (shown when Drama 3 is selected) has a
local language model write this markup for you, like the enhance button on
Fish Audio's site. Your words are protected: every passage the model returns is
checked, and if anything other than markup changed it is retried and then left
exactly as you wrote it. Review the result in the text box, edit freely, or
press **Undo**.

**Tagging notes.** Open **📝 Tagging notes** under the button to tell the model
how to direct the script, in plain words: who is speaking, the mood, how heavy
a hand to use. For example: *"The speaker is a bratty, theatrical young woman
with a posh accent. Use whispers for asides and very few sounds."* The notes
are added to the model's instructions for every passage and remembered until
you clear them. They steer the choice of tags only: a note that asks for words
to be added or changed has no effect, because the result is still checked
against your script word for word.

**The model.** The enhancer runs an abliterated `Qwen3.5-9B` at 8-bit:
`trohrbaugh/Qwen3.5-9B-heretic-v2`, as the `Qwen3.5-9B-heretic-v2-Q8_0.gguf`
file from `tpnlabs/Qwen3.5-9B-heretic-v2-TPN-GGUF` (about 9.5 GB, Apache-2.0),
on [llama.cpp](https://github.com/ggml-org/llama.cpp)'s server. Abliterated
means the model's tendency to refuse has been removed, so a script with
violence or adult content is tagged like any other instead of being refused or
quietly softened (which the word-for-word check would then reject, leaving the
passage untagged). Of the abliterated Qwen3.5-9B builds compared in an
independent benchmark, this one stayed closest to the original model. To use
the stock model instead, set `CANDYECHO_ENHANCER_GGUF_REPO` to
`unsloth/Qwen3.5-9B-GGUF` and `CANDYECHO_ENHANCER_GGUF_FILE` to
`Qwen3.5-9B-Q8_0.gguf`.
The **Download tag model** button fetches the model file and, on Windows, the
current llama.cpp CUDA build (about 0.5 GB, MIT) into `models/enhancer/`; both
are checked against their published sizes, and the build against its SHA-256.
On Linux or macOS install llama.cpp yourself so that `llama-server` is on PATH.

CandyEcho starts that server the first time you add tags and talks to it on
`127.0.0.1:8090`. It uses about 10 GB of VRAM while running and is stopped
again after ten minutes without use, or at once with **⏏ Unload tag model**, so
it does not sit in VRAM between sessions of tagging. Its output goes to
`logs/engine-enhancer.log`.

Qwen3.5 is a thinking model. Thinking is switched off, both when the server is
launched (`--reasoning off`) and in each request, because the job is to copy a
script and insert tags and deliberating first only makes it slower. Anything
that still arrives inside `<think>` tags is discarded.

| Variable | Effect |
| --- | --- |
| `CANDYECHO_ENHANCER_GGUF_REPO` / `CANDYECHO_ENHANCER_GGUF_FILE` | A different GGUF model or quantisation: the Hugging Face repo and the one file to take from it |
| `CANDYECHO_ENHANCER_IDLE_MINUTES` | Minutes unused before the server is stopped (default 10; 0 keeps it loaded) |
| `CANDYECHO_ENHANCER_SERVER_URL` | Address the managed server listens on (default `http://127.0.0.1:8090`) |
| `CANDYECHO_LLAMA_SERVER` | Full path to your own `llama-server`; the managed build is then not downloaded |
| `CANDYECHO_LLAMA_RELEASE` | Pin the llama.cpp release to download, e.g. `b12283` (default: latest) |
| `CANDYECHO_LLAMA_CUDA` | Use the build for this CUDA major version (default: 13 when published, else the newest) |
| `CANDYECHO_ENHANCER_URL` | Use an OpenAI-compatible server you run yourself instead (e.g. `http://localhost:11434/v1` for Ollama); nothing is downloaded |
| `CANDYECHO_ENHANCER_ENDPOINT_MODEL` / `CANDYECHO_ENHANCER_API_KEY` | Model name and bearer token for that server |
| `CANDYECHO_ENHANCER_BACKEND=transformers` | The earlier in-process backend: downloads `CANDYECHO_ENHANCER_MODEL` (default `Qwen/Qwen2.5-7B-Instruct`, about 15 GB) and runs it with `transformers` on `CANDYECHO_ENHANCER_DEVICE` |

If you used the tag enhancer before this change, the old model is still in
`models/enhancer/Qwen2.5-7B-Instruct` (about 15 GB). It is no longer used and
can be deleted, as can `models/enhancer/Qwen3.5-9B-GGUF` if you downloaded the
stock model before the default changed.

### Changing how the engine is launched

The built-in launch command matches what `scripts/setup_engines.sh` installs. To
run the engine differently — Docker, another machine, a different install
location — create `engines.json` in the CandyEcho folder. It can override
`url`, `health_path`, `start_cmd`, `stop_cmd` and `startup_timeout`;
`{model_dir}` and `{model_dir_wsl}` expand to the engine's weights folder.

```json
{
  "drama3": {
    "start_cmd": ["docker", "run", "--rm", "--name", "candyecho-drama3", "--gpus", "device=0",
                  "-v", "{model_dir}:/app/checkpoints",
                  "-e", "LLAMA_CHECKPOINT_PATH=checkpoints/drama-3",
                  "-e", "DECODER_CHECKPOINT_PATH=checkpoints/s2-pro/codec.pth",
                  "-p", "8080:8080", "fishaudio/fish-speech:server-cuda"],
    "stop_cmd": ["docker", "stop", "candyecho-drama3"]
  }
}
```

`"start_cmd": null` means "I run this server myself": CandyEcho only checks
whether it answers. A server that is already running at the configured address
is always used as-is, however it was started. `CANDYECHO_DRAMA3_URL` changes
just the address.

When the server fails to start, the last lines of its output are shown in the
engine card; the full log is in `logs/engine-drama3.log`.

> **RTX 50-series / RTX PRO Blackwell:** these cards need a PyTorch built for
> CUDA 12.8 or newer. The setup script installs Drama 3 with `cu129` for that
> reason. Fish's prebuilt Docker image is documented as CUDA 12.6 by default, so
> if you take the Docker route on one of these cards, build the image yourself
> with `CUDA_VER=12.9.0 UV_EXTRA=cu129` as described in the fish-speech docs.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 13.0+ (driver R580 or newer)
- 8GB+ VRAM recommended
- [uv](https://docs.astral.sh/uv/) package manager
- **Windows only:** FFmpeg shared libraries in PATH (see installation instructions)
- **Drama 3 (optional):** WSL 2 on Windows (it is Linux-only) and 48 GB+ VRAM, on top of what Echo uses if both are loaded at once

## API Endpoints

- `GET /` - Web UI
- `GET /voices` - List available voices (each with its reference-audio duration)
- `POST /voices` - Upload a `.wav` voice sample (multipart form field `file`); it's saved to `voice_library/`, preprocessed, and added to the voice list
- `GET /voices/{name}/audio` - Download/preview a voice's reference `.wav`
- `POST /voices/{name}/rename` - Rename a voice. JSON body: `{"new_name": "..."}`
- `GET /cleaning-rules` - List the textbook-cleaning rules and presets
- `POST /clean-text` - Preview cleaning. JSON body: `{"text": "...", "preset": "textbook", "rules": {}, "substitutions": {}}`; returns the cleaned text plus a per-rule report of what was removed
- `POST /generate` - Generate audio (SSE stream). JSON body: `{"text": "...", "voice": "...", "normalization_level": "moderate", "normalize_volume": false, "cleaning": {"enabled": true, "preset": "textbook"}}` (`cleaning` is optional; omit it to skip cleaning entirely)
  - Add `"engine": "drama3"` to use Drama 3 (default `"echo"`). It also takes `temperature`, `top_p`, `top_k` and `extra_voices` (a list of further voice names used as additional references).
  - With `"verify": true`, add `"verify_voice": true` to keep the take closest to the reference voice among those that say the text correctly.
- `POST /stop` - Stop an in-progress generation. Optional query param: `generation_id`
- `GET /engines` - Every engine with the state of its weights (`missing`, `partial`, `downloading`, `ready`, `error`) and of its server (`stopped`, `starting`, `ready`, `failed`)
- `POST /models/{key}/download` - Start or resume downloading weights (`drama3`, `enhancer`); returns at once, progress is in `GET /engines`
- `POST /engines/{key}/start`, `POST /engines/{key}/stop` - Start or stop an engine's server
- `POST /engines/echo/unload`, `POST /engines/echo/load` - Park Echo off the GPU to free its VRAM, or bring it back
- `PUT /engines/{key}/autostart` (`{"enabled": true}`) - Start the engine's server whenever CandyEcho launches
- `POST /audition` - Compare reference clips (SSE stream). JSON body: `{"engine": "drama3", "voices": ["a", "b"], "takes": 2}`; emits one `trial` event per combination with its voice-match score and best take, then a ranking
- `GET /jobs/{id}/chunks/{index}/audio` - One finished chunk of a saved batch
- `POST /jobs/{id}/chunks/{index}/redo` - Generate that chunk again with a new seed (or `{"seed": N}`) and swap it into the batch
- `GET /voices/{name}/transcript`, `PUT /voices/{name}/transcript` (`{"text": "..."}`) - Read or set a voice's reference transcript
- `POST /voices/{name}/transcribe` - Transcribe a voice's reference clip with Whisper and save it
- `POST /enhance-text` - Add Drama 3 direction tags. JSON body: `{"text": "..."}`; returns the tagged text and counts of passages changed or left alone
- `GET /voice-events` - SSE stream of voice library changes (processing, ready, removed, renamed, error)
- `GET /health` - Health check

### OpenAI-compatible TTS API

- `POST /v1/audio/speech` - Non-streaming synthesis returning a single audio file. JSON body: `{"input": "...", "voice": "<voice name>", "response_format": "wav|mp3|flac|ogg"}` (`model` picks the engine: `drama3` uses Drama 3, anything else uses Echo; `speed` is accepted but ignored).
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
│       ├── engines.py           # Registry of the optional engines
│       ├── engine_servers.py    # Start/stop/health of engine servers
│       ├── remote_engines.py    # Drama 3 HTTP client + generator
│       ├── model_store.py       # Weights: download, manifest, status
│       ├── settings.py          # Autostart and weight-folder preferences
│       ├── voice_transcripts.py # Reference clips and transcripts
│       ├── drama_markup.py      # Drama 3 tag handling
│       ├── tag_enhancer.py      # LLM that writes Drama 3 tags
│       ├── llama_runtime.py     # llama.cpp server for the enhancer's GGUF model
│       ├── file_watcher.py      # Voice file auto-detection
│       ├── voice_event_broadcaster.py  # SSE voice events
│       └── _vendor/
│           └── echo_tts/        # Vendored Echo-TTS inference code
├── voice_library/               # Voice .wav files (user-provided)
├── static/
│   ├── index.html               # Web UI
│   ├── style.css                # Styles
│   ├── app.js                   # Client-side logic
│   ├── engines.js               # Engine picker, downloads, transcripts, enhancer
│   └── lamejs.min.js            # Vendored MP3 encoder
├── scripts/
│   └── setup_engines.sh         # One-time install of the optional engines
├── models/                      # Downloaded weights (created on demand)
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

Drama 3 is not bundled: its code is installed by `scripts/setup_engines.sh`
and its weights are downloaded on request, under their own licence.

- **Fish Audio Drama 3** (and the S2-Pro codec) - Fish Audio Community License. Read it before any commercial use.
- **Tag-enhancer LLM** - the licence of whichever model you choose; the default (an abliterated `Qwen3.5-9B`) is Apache-2.0, and llama.cpp, which runs it, is MIT.
