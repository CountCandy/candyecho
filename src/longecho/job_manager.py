"""On-disk generation jobs: persistence, resume, and time estimates.

A long read is hours of GPU time. Before this, all of it lived in the browser
tab's memory: closing the tab, a refresh, or a crash lost the lot, and the client
held every decoded chunk in RAM at once. Chunks are now written to disk as they
are produced, with a manifest recording what was generated and how, so a job can
be listened to, downloaded, or picked up again later.

Deliberately torch-free. Audio arrives already encoded as bytes, which keeps the
manifest and scheduling logic testable without the CUDA stack.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

__all__ = ["JobChunk", "Job", "JobManager", "EtaTracker", "estimate_remaining"]

# Job ids are used to build paths, so they are restricted to a shape that cannot
# escape the jobs directory.
_JOB_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")

STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"


@dataclass
class JobChunk:
    index: int
    text: str
    status: str = "pending"          # pending | done
    duration: float | None = None    # seconds of audio
    seconds: float | None = None     # wall time spent generating it
    seed: int | None = None
    score: float | None = None
    wer: float | None = None
    takes: int | None = None

    @property
    def done(self) -> bool:
        return self.status == "done"


@dataclass
class Job:
    id: str
    created_at: float
    voice: str
    status: str = STATUS_RUNNING
    settings: dict = field(default_factory=dict)
    chunks: list[JobChunk] = field(default_factory=list)

    # --- derived ---------------------------------------------------------

    @property
    def done_count(self) -> int:
        return sum(1 for c in self.chunks if c.done)

    @property
    def total(self) -> int:
        return len(self.chunks)

    @property
    def audio_duration(self) -> float:
        return sum(c.duration or 0.0 for c in self.chunks if c.done)

    @property
    def resume_index(self) -> int:
        """First chunk not yet generated.

        Chunks are produced strictly in order because each seeds the next, so
        the first pending index is where a resumed run must restart.
        """
        for chunk in self.chunks:
            if not chunk.done:
                return chunk.index
        return len(self.chunks)

    @property
    def complete(self) -> bool:
        return self.total > 0 and self.done_count == self.total

    def summary(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "voice": self.voice,
            "status": self.status,
            "total": self.total,
            "done": self.done_count,
            "audio_duration": round(self.audio_duration, 2),
            "resume_index": self.resume_index,
            "complete": self.complete,
            "settings": self.settings,
        }

    def as_dict(self) -> dict:
        payload = self.summary()
        payload["chunks"] = [asdict(c) for c in self.chunks]
        return payload


class JobManager:
    """Stores jobs under ``jobs/<id>/`` as a manifest plus one file per chunk."""

    def __init__(self, root: Path | str = "jobs", audio_ext: str = "wav"):
        self.root = Path(root)
        self.audio_ext = audio_ext

    # --- paths -----------------------------------------------------------

    def _validate(self, job_id: str) -> str:
        if not _JOB_ID_RE.match(job_id or ""):
            raise ValueError(f"Invalid job id: {job_id!r}")
        return job_id

    def job_dir(self, job_id: str) -> Path:
        return self.root / self._validate(job_id)

    def chunk_path(self, job_id: str, index: int) -> Path:
        return self.job_dir(job_id) / f"chunk_{index:05d}.{self.audio_ext}"

    def manifest_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "manifest.json"

    # --- lifecycle -------------------------------------------------------

    def create(self, text_chunks: Iterable[str], voice: str, settings: dict | None = None) -> Job:
        job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        job = Job(
            id=job_id,
            created_at=time.time(),
            voice=voice,
            settings=dict(settings or {}),
            chunks=[JobChunk(index=i, text=t) for i, t in enumerate(text_chunks)],
        )
        self.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        self.save(job)
        logger.info(f"Job {job_id} created ({job.total} chunks, voice '{voice}')")
        return job

    def save(self, job: Job) -> None:
        """Write the manifest atomically, so a crash mid-write cannot corrupt it."""
        path = self.manifest_path(job.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job.as_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def record_chunk(
        self,
        job: Job,
        index: int,
        audio_bytes: bytes,
        duration: float,
        seconds: float | None = None,
        seed: int | None = None,
        score: float | None = None,
        wer: float | None = None,
        takes: int | None = None,
    ) -> None:
        """Persist one finished chunk and update the manifest."""
        path = self.chunk_path(job.id, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(audio_bytes)
        os.replace(tmp, path)

        chunk = job.chunks[index]
        chunk.status = "done"
        chunk.duration = duration
        chunk.seconds = seconds
        chunk.seed = seed
        chunk.score = score
        chunk.wer = wer
        chunk.takes = takes
        self.save(job)

    def finish(self, job: Job, status: str = STATUS_COMPLETE) -> None:
        job.status = status
        self.save(job)
        logger.info(f"Job {job.id} {status} ({job.done_count}/{job.total} chunks)")

    # --- reading ---------------------------------------------------------

    def load(self, job_id: str) -> Job | None:
        path = self.manifest_path(job_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Could not read manifest for job {job_id}: {e}")
            return None
        chunks = [JobChunk(**c) for c in data.get("chunks", [])]
        return Job(
            id=data["id"],
            created_at=data.get("created_at", 0.0),
            voice=data.get("voice", ""),
            status=data.get("status", STATUS_RUNNING),
            settings=data.get("settings", {}),
            chunks=chunks,
        )

    def list_jobs(self, limit: int = 50) -> list[Job]:
        if not self.root.exists():
            return []
        jobs = []
        for entry in self.root.iterdir():
            if not entry.is_dir() or not _JOB_ID_RE.match(entry.name):
                continue
            job = self.load(entry.name)
            if job is not None:
                jobs.append(job)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def done_chunk_paths(self, job: Job) -> list[Path]:
        """Paths of the finished chunks, in order, skipping any gone missing."""
        paths = []
        for chunk in job.chunks:
            if not chunk.done:
                continue
            path = self.chunk_path(job.id, chunk.index)
            if path.exists():
                paths.append(path)
            else:
                logger.warning(f"Job {job.id}: chunk {chunk.index} missing from disk")
        return paths

    def delete(self, job_id: str) -> bool:
        directory = self.job_dir(job_id)
        if not directory.exists():
            return False
        shutil.rmtree(directory, ignore_errors=True)
        logger.info(f"Job {job_id} deleted")
        return True


# ---------------------------------------------------------------------------
# Time estimates
# ---------------------------------------------------------------------------

def estimate_remaining(
    per_chunk_seconds: list[float],
    remaining_chunks: int,
    window: int = 5,
) -> float | None:
    """Seconds of work left, from the pace of the chunks generated so far.

    Averages a trailing window rather than the whole run: best-of-N escalation
    makes individual chunks lumpy (a chunk that retries costs roughly double),
    and a long book's early chunks stop being representative once settings or
    text density change. Returns None until there is something to go on.
    """
    if not per_chunk_seconds or remaining_chunks <= 0:
        return None
    sample = per_chunk_seconds[-window:]
    return (sum(sample) / len(sample)) * remaining_chunks


class EtaTracker:
    """Tracks chunk pace and reports a remaining-time estimate."""

    def __init__(self, total_chunks: int, already_done: int = 0, window: int = 5):
        self.total = max(0, total_chunks)
        self.done = max(0, already_done)
        self.window = window
        self.per_chunk: list[float] = []
        self.started = time.monotonic()

    def record(self, seconds: float) -> None:
        self.per_chunk.append(max(0.0, seconds))
        self.done += 1

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.done)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def eta(self) -> float | None:
        return estimate_remaining(self.per_chunk, self.remaining, self.window)

    def as_dict(self) -> dict:
        eta = self.eta()
        return {
            "done": self.done,
            "total": self.total,
            "elapsed_seconds": round(self.elapsed, 1),
            "eta_seconds": round(eta, 1) if eta is not None else None,
            "seconds_per_chunk": (
                round(sum(self.per_chunk[-self.window:]) / len(self.per_chunk[-self.window:]), 2)
                if self.per_chunk else None
            ),
        }
