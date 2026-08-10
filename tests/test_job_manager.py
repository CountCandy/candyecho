"""Tests for job persistence, resume points, and time estimates."""
import json

import pytest

from longecho.job_manager import (
    STATUS_COMPLETE,
    STATUS_RUNNING,
    EtaTracker,
    Job,
    JobManager,
    estimate_remaining,
)

CHUNKS = ["first chunk", "second chunk", "third chunk"]


@pytest.fixture
def manager(tmp_path):
    return JobManager(root=tmp_path / "jobs")


@pytest.fixture
def job(manager):
    return manager.create(CHUNKS, voice="Peppermint", settings={"verify": True})


class TestCreation:
    def test_creates_manifest_on_disk(self, manager, job):
        assert manager.manifest_path(job.id).exists()

    def test_manifest_is_valid_json(self, manager, job):
        data = json.loads(manager.manifest_path(job.id).read_text())
        assert data["id"] == job.id
        assert len(data["chunks"]) == 3

    def test_records_every_chunk_as_pending(self, job):
        assert job.total == 3
        assert job.done_count == 0
        assert all(not c.done for c in job.chunks)

    def test_ids_are_unique(self, manager):
        ids = {manager.create(CHUNKS, voice="v").id for _ in range(5)}
        assert len(ids) == 5

    def test_settings_round_trip(self, manager, job):
        assert manager.load(job.id).settings == {"verify": True}


class TestChunkRecording:
    def test_chunk_written_to_disk(self, manager, job):
        manager.record_chunk(job, 0, b"RIFFfake", duration=4.0)
        assert manager.chunk_path(job.id, 0).read_bytes() == b"RIFFfake"

    def test_manifest_updated_after_each_chunk(self, manager, job):
        manager.record_chunk(job, 0, b"a", duration=4.0, seconds=3.1, wer=0.02)
        reloaded = manager.load(job.id)
        assert reloaded.chunks[0].done
        assert reloaded.chunks[0].duration == 4.0
        assert reloaded.chunks[0].seconds == 3.1
        assert reloaded.chunks[0].wer == 0.02

    def test_audio_duration_accumulates(self, manager, job):
        manager.record_chunk(job, 0, b"a", duration=4.0)
        manager.record_chunk(job, 1, b"b", duration=6.5)
        assert job.audio_duration == 10.5

    def test_done_chunk_paths_in_order(self, manager, job):
        manager.record_chunk(job, 0, b"a", duration=1.0)
        manager.record_chunk(job, 1, b"b", duration=1.0)
        paths = manager.done_chunk_paths(job)
        assert [p.name for p in paths] == ["chunk_00000.wav", "chunk_00001.wav"]

    def test_missing_file_is_skipped_not_fatal(self, manager, job):
        manager.record_chunk(job, 0, b"a", duration=1.0)
        manager.record_chunk(job, 1, b"b", duration=1.0)
        manager.chunk_path(job.id, 0).unlink()
        assert len(manager.done_chunk_paths(job)) == 1


class TestResume:
    def test_resume_index_is_first_pending(self, manager, job):
        assert job.resume_index == 0
        manager.record_chunk(job, 0, b"a", duration=1.0)
        assert job.resume_index == 1

    def test_resume_index_at_end_when_complete(self, manager, job):
        for i in range(3):
            manager.record_chunk(job, i, b"x", duration=1.0)
        assert job.resume_index == 3
        assert job.complete

    def test_interrupted_job_reloads_with_progress(self, manager, job):
        manager.record_chunk(job, 0, b"a", duration=4.0)
        manager.record_chunk(job, 1, b"b", duration=4.0)
        # Simulate a crash: reload from disk only.
        reloaded = manager.load(job.id)
        assert reloaded.done_count == 2
        assert reloaded.resume_index == 2
        assert not reloaded.complete

    def test_resumed_job_keeps_chunk_text(self, manager, job):
        manager.record_chunk(job, 0, b"a", duration=1.0)
        reloaded = manager.load(job.id)
        assert [c.text for c in reloaded.chunks] == CHUNKS


class TestListingAndDeletion:
    def test_lists_newest_first(self, manager):
        first = manager.create(CHUNKS, voice="a")
        second = manager.create(CHUNKS, voice="b")
        second.created_at = first.created_at + 100
        manager.save(second)
        assert [j.id for j in manager.list_jobs()][0] == second.id

    def test_ignores_unrelated_directories(self, manager, job):
        (manager.root / "not-a-job").mkdir(parents=True, exist_ok=True)
        assert [j.id for j in manager.list_jobs()] == [job.id]

    def test_empty_root(self, tmp_path):
        assert JobManager(root=tmp_path / "nothing").list_jobs() == []

    def test_delete_removes_everything(self, manager, job):
        manager.record_chunk(job, 0, b"a", duration=1.0)
        assert manager.delete(job.id) is True
        assert not manager.job_dir(job.id).exists()

    def test_delete_missing_job_is_false(self, manager):
        assert manager.delete("20260101-000000-abcdef") is False

    def test_load_missing_job_is_none(self, manager):
        assert manager.load("20260101-000000-abcdef") is None


class TestSafety:
    @pytest.mark.parametrize(
        "bad", ["../etc", "..", "job/../..", "", "not-an-id", "20260101-000000-XYZ"]
    )
    def test_path_traversal_rejected(self, manager, bad):
        with pytest.raises(ValueError):
            manager.job_dir(bad)

    def test_corrupt_manifest_returns_none(self, manager, job):
        manager.manifest_path(job.id).write_text("{ not json")
        assert manager.load(job.id) is None

    def test_finish_sets_status(self, manager, job):
        manager.finish(job, STATUS_COMPLETE)
        assert manager.load(job.id).status == STATUS_COMPLETE

    def test_default_status_is_running(self, job):
        assert job.status == STATUS_RUNNING


# --- ETA -------------------------------------------------------------------

class TestEstimateRemaining:
    def test_no_history_gives_no_estimate(self):
        assert estimate_remaining([], 10) is None

    def test_nothing_remaining_gives_no_estimate(self):
        assert estimate_remaining([3.0], 0) is None

    def test_simple_projection(self):
        assert estimate_remaining([4.0, 4.0], 5) == 20.0

    def test_uses_trailing_window(self):
        """Early chunks stop being representative; recent pace wins."""
        history = [20.0] * 10 + [4.0] * 5
        assert estimate_remaining(history, 5, window=5) == 20.0

    def test_window_larger_than_history(self):
        assert estimate_remaining([2.0], 3, window=5) == 6.0


class TestEtaTracker:
    def test_starts_without_an_estimate(self):
        assert EtaTracker(10).eta() is None

    def test_estimate_after_first_chunk(self):
        tracker = EtaTracker(10)
        tracker.record(5.0)
        assert tracker.eta() == 45.0

    def test_counts_down(self):
        tracker = EtaTracker(3)
        tracker.record(2.0)
        assert tracker.remaining == 2
        tracker.record(2.0)
        assert tracker.remaining == 1

    def test_no_estimate_once_finished(self):
        tracker = EtaTracker(1)
        tracker.record(3.0)
        assert tracker.remaining == 0
        assert tracker.eta() is None

    def test_resumed_job_accounts_for_existing_progress(self):
        """A job resumed at chunk 8 of 10 has two left, not ten."""
        tracker = EtaTracker(10, already_done=8)
        tracker.record(5.0)
        assert tracker.remaining == 1
        assert tracker.eta() == 5.0

    def test_serialized_shape(self):
        tracker = EtaTracker(4)
        tracker.record(3.0)
        payload = tracker.as_dict()
        assert payload["done"] == 1
        assert payload["total"] == 4
        assert payload["eta_seconds"] == 9.0
        assert payload["seconds_per_chunk"] == 3.0

    def test_negative_durations_clamped(self):
        tracker = EtaTracker(2)
        tracker.record(-5.0)
        assert tracker.eta() == 0.0
