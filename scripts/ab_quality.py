#!/usr/bin/env python
"""A/B quality harness: measure what a setting or a voice sample actually costs.

Once take verification exists, questions that were previously arguments become
measurements. Does a 60-second reference beat a 15-second one? Does raising
expressiveness to 1.2 lose words? Does Force Speaker at 1.3 help or just
flatten the delivery? Run it and read the numbers.

Each variant generates the same text with a single take per chunk -- best-of-N
is deliberately off, because the point is to measure the raw quality of the
setting, not how well the verifier papers over it. Every chunk is then scored by
the same ASR ensemble the live path uses.

Examples
--------
Compare voice samples (e.g. the same speaker at different reference lengths)::

    uv run python scripts/ab_quality.py --text page.txt --voices Ranni15s,Ranni60s,Ranni3m

Sweep a sampler parameter::

    uv run python scripts/ab_quality.py --text page.txt --voice Ranni60s \\
        --vary truncation=0.8,1.0,1.2 --repeats 3

Sweep Force Speaker::

    uv run python scripts/ab_quality.py --text page.txt --voice Ranni60s \\
        --vary speaker_force=1.0,1.2,1.5

Results print as a table and are written to CSV with ``--csv out.csv``.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Allow running straight from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from longecho.text_cleaner import clean_text  # noqa: E402
from longecho.text_normalizer import TextNormalizer  # noqa: E402
from longecho.text_segmenter import segment_text  # noqa: E402

# Parameters that map onto Echo's sampler, and the gen_params key each one sets.
VARIABLE_PARAMS = {
    "truncation": "truncation_factor",
    "steps": "num_steps",
    "cfg_text": "cfg_scale_text",
    "cfg_speaker": "cfg_scale_speaker",
    "speaker_force": "speaker_kv_scale",
}


@dataclass
class Measurement:
    variant: str
    voice: str
    repeat: int
    chunk: int
    wer: float
    cer: float
    score: float
    similarity: float | None
    duration: float
    seconds: float


@dataclass
class Variant:
    label: str
    voice: str
    gen_params: dict = field(default_factory=dict)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", required=True, type=Path, help="Text file to generate")
    p.add_argument("--voice", help="Voice name (single-voice mode)")
    p.add_argument("--voices", help="Comma-separated voice names to compare")
    p.add_argument(
        "--vary",
        action="append",
        default=[],
        metavar="NAME=V1,V2",
        help=f"Sweep a parameter. One of: {', '.join(sorted(VARIABLE_PARAMS))}",
    )
    p.add_argument("--repeats", type=int, default=1, help="Runs per variant (default 1)")
    p.add_argument("--seed", type=int, default=1234, help="Base seed; repeats offset from it")
    p.add_argument("--max-chunks", type=int, default=0, help="Cap chunks per run (0 = all)")
    p.add_argument("--no-clean", action="store_true", help="Skip textbook cleaning")
    p.add_argument("--csv", type=Path, help="Write per-chunk measurements here")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def build_variants(args: argparse.Namespace) -> list[Variant]:
    voices = [v.strip() for v in (args.voices or args.voice or "").split(",") if v.strip()]
    if not voices:
        raise SystemExit("Give --voice or --voices")

    sweeps: list[tuple[str, list[float]]] = []
    for spec in args.vary:
        if "=" not in spec:
            raise SystemExit(f"--vary needs NAME=V1,V2 (got {spec!r})")
        name, raw = spec.split("=", 1)
        name = name.strip()
        if name not in VARIABLE_PARAMS:
            raise SystemExit(f"Unknown --vary parameter {name!r}. Options: {sorted(VARIABLE_PARAMS)}")
        sweeps.append((name, [float(v) for v in raw.split(",") if v.strip()]))

    variants: list[Variant] = []
    for voice in voices:
        if not sweeps:
            variants.append(Variant(label=voice, voice=voice))
            continue
        # One sweep at a time keeps the comparison readable; combine by running
        # the tool twice rather than exploding into a grid.
        for name, values in sweeps:
            for value in values:
                key = VARIABLE_PARAMS[name]
                params = {} if (name == "speaker_force" and value <= 1.0) else {key: value}
                label = f"{voice} {name}={value:g}" if len(voices) > 1 else f"{name}={value:g}"
                variants.append(Variant(label=label, voice=voice, gen_params=params))
    return variants


def prepare_chunks(args: argparse.Namespace) -> list[str]:
    raw = args.text.read_text(encoding="utf-8")
    text = raw if args.no_clean else clean_text(raw).text
    chunks = segment_text(TextNormalizer().normalize(text))
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    return chunks


def summarize(rows: list[Measurement]) -> dict:
    def mean(values):
        values = [v for v in values if v is not None]
        return statistics.fmean(values) if values else None

    sims = [r.similarity for r in rows if r.similarity is not None]
    return {
        "chunks": len(rows),
        "wer": mean([r.wer for r in rows]),
        "cer": mean([r.cer for r in rows]),
        "score": mean([r.score for r in rows]),
        "similarity": mean(sims) if sims else None,
        "worst_wer": max((r.wer for r in rows), default=None),
        "audio": sum(r.duration for r in rows),
        "seconds": sum(r.seconds for r in rows),
    }


def print_table(results: dict[str, dict]) -> None:
    if not results:
        print("No results.")
        return
    width = max(len(k) for k in results) + 2
    header = (
        f"{'variant'.ljust(width)}{'WER':>8}{'worst':>8}{'CER':>8}"
        f"{'score':>8}{'spk sim':>9}{'audio s':>9}{'gen s':>8}{'xRT':>7}"
    )
    print()
    print(header)
    print("-" * len(header))

    best = min((v["wer"] for v in results.values() if v["wer"] is not None), default=None)
    for label, stats in results.items():
        sim = f"{stats['similarity']:.3f}" if stats["similarity"] is not None else "-"
        rt = stats["seconds"] / stats["audio"] if stats["audio"] else 0.0
        marker = "  <-- best" if best is not None and stats["wer"] == best else ""
        print(
            f"{label.ljust(width)}"
            f"{stats['wer']:>8.4f}{stats['worst_wer']:>8.4f}{stats['cer']:>8.4f}"
            f"{stats['score']:>8.4f}{sim:>9}{stats['audio']:>9.1f}"
            f"{stats['seconds']:>8.1f}{rt:>7.2f}{marker}"
        )
    print()
    print("WER/CER: lower is better. spk sim: higher means closer to the reference voice.")
    print("xRT: generation seconds per second of audio.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    variants = build_variants(args)
    chunks = prepare_chunks(args)
    if not chunks:
        raise SystemExit("Nothing to generate after cleaning/segmentation")

    print(f"{len(chunks)} chunks x {len(variants)} variant(s) x {args.repeats} repeat(s)")

    # Imports are deferred so --help works without the CUDA stack loaded.
    from longecho._vendor.echo_tts import (
        load_fish_ae_from_hf,
        load_model_from_hf,
        load_pca_state_from_hf,
    )
    from longecho.audio_generator import AudioGenerator
    from longecho.transcript_verifier import build_selector
    from longecho.voice_manager import VoiceManager

    print("Loading Echo models...")
    model = load_model_from_hf(device=args.device)
    fish_ae = load_fish_ae_from_hf(device=args.device)
    pca_state = load_pca_state_from_hf(device=args.device)

    voice_manager = VoiceManager(fish_ae, pca_state)
    voice_manager.load_voices()
    available = set(voice_manager.get_voice_names())
    missing = {v.voice for v in variants} - available
    if missing:
        raise SystemExit(f"Unknown voice(s): {sorted(missing)}. Available: {sorted(available)}")

    print("Loading verification models...")
    selector = build_selector(device=args.device)
    generator = AudioGenerator(model, fish_ae, pca_state)

    from longecho._vendor.echo_tts import load_audio

    reference_cache: dict[str, object] = {}

    def reference_for(voice: str):
        if voice not in reference_cache:
            path = voice_manager.get_voice_path(voice)
            reference_cache[voice] = load_audio(str(path)) if path else None
        return reference_cache[voice]

    rows: list[Measurement] = []
    for variant in variants:
        speaker_latent, speaker_mask = voice_manager.get_voice(variant.voice)
        reference = reference_for(variant.voice)

        for repeat in range(args.repeats):
            collected: list[tuple[int, dict]] = []
            started = time.monotonic()
            _gid, gen = generator.generate_long_audio(
                chunks, speaker_latent, speaker_mask,
                rng_seed=args.seed + repeat * 7919,
                gen_params=variant.gen_params or None,
                # One take per chunk: we are measuring the setting, not how well
                # best-of-N compensates for it.
                num_candidates=1,
                selector=selector,
                max_rounds=1,
                speaker_audio=reference,
                on_selection=lambda idx, payload: collected.append((idx, payload)),
            )
            durations = [chunk.shape[-1] / 44100.0 for chunk in gen]
            elapsed = time.monotonic() - started

            per_chunk = elapsed / max(1, len(durations))
            for index, payload in collected:
                best = next(
                    (c for c in payload["candidates"] if c["index"] == payload["winner"]), None
                )
                if best is None:
                    continue
                rows.append(Measurement(
                    variant=variant.label,
                    voice=variant.voice,
                    repeat=repeat,
                    chunk=index,
                    wer=best["wer"],
                    cer=best["cer"],
                    score=best["score"],
                    similarity=best.get("speaker_similarity"),
                    duration=durations[index] if index < len(durations) else 0.0,
                    seconds=per_chunk,
                ))
            print(
                f"  {variant.label} repeat {repeat + 1}/{args.repeats}: "
                f"{len(collected)} chunks in {elapsed:.1f}s"
            )

    results = {}
    for variant in variants:
        subset = [r for r in rows if r.variant == variant.label]
        if subset:
            results[variant.label] = summarize(subset)
    print_table(results)

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["variant", "voice", "repeat", "chunk", "wer", "cer", "score",
                 "similarity", "audio_seconds", "gen_seconds"]
            )
            for r in rows:
                writer.writerow([
                    r.variant, r.voice, r.repeat, r.chunk,
                    f"{r.wer:.6f}", f"{r.cer:.6f}", f"{r.score:.6f}",
                    "" if r.similarity is None else f"{r.similarity:.6f}",
                    f"{r.duration:.3f}", f"{r.seconds:.3f}",
                ])
        print(f"Per-chunk measurements written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
