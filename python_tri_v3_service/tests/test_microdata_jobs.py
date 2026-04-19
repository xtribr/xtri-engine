"""Tests for persistent local microdata jobs."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from app.microdata.cache import CacheKey
from app.microdata.jobs import JOB_DONE, JOB_ERROR, JOB_RUNNING, JobManager


@pytest.fixture
def temp_jobs_root(tmp_path):
    jobs_root = tmp_path / "jobs"
    with mock.patch("app.runtime.local_paths.JOBS_ROOT", jobs_root):
        yield jobs_root


class TestJobManagerPersistence:
    @pytest.mark.unit
    def test_persiste_job_em_disco(self, temp_jobs_root):
        manager = JobManager()
        job = manager.create(year=2024, sample_size=1000, rng_seed=42)
        manager.mark_running(job.job_id)
        manager.update_progress(job.job_id, progress=0.4, stage="streaming CSV")

        payload = json.loads(
            (temp_jobs_root / "microdata" / f"{job.job_id}.json").read_text(encoding="utf-8")
        )
        assert payload["job_id"] == job.job_id
        assert payload["status"] == JOB_RUNNING
        assert payload["stage"] == "streaming CSV"
        assert payload["progress"] == 0.4

    @pytest.mark.unit
    def test_recarrega_jobs_e_normaliza_execucao_interrompida(self, temp_jobs_root):
        manager = JobManager()
        job = manager.create(year=2024, sample_size=2000, rng_seed=7)
        manager.mark_running(job.job_id)

        restored = JobManager()
        count = restored.reload_from_disk()
        loaded = restored.get(job.job_id)

        assert count == 1
        assert loaded is not None
        assert loaded.status == JOB_ERROR
        assert loaded.stage == "interrupted"
        assert loaded.error is not None

    @pytest.mark.unit
    def test_mark_done_persiste_result_key(self, temp_jobs_root):
        manager = JobManager()
        job = manager.create(year=2024, sample_size=500, rng_seed=11)
        key = CacheKey(year=2024, source_hash="abcd1234abcd1234", filters_hash="deadbeef", stage="calibrated")

        manager.mark_done(job.job_id, result_key=key)

        restored = JobManager()
        restored.reload_from_disk()
        loaded = restored.get(job.job_id)

        assert loaded is not None
        assert loaded.status == JOB_DONE
        assert loaded.result_key is not None
        assert loaded.result_key.stage == "calibrated"

    @pytest.mark.unit
    def test_clear_remove_sidecars(self, temp_jobs_root):
        manager = JobManager()
        job = manager.create(year=2024, sample_size=100, rng_seed=1)

        removed = manager.clear()

        assert removed == 1
        assert not (temp_jobs_root / "microdata" / f"{job.job_id}.json").exists()
