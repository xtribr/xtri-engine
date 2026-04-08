"""
In-memory job manager for the microdata pipeline.

Single-user only — stores jobs in a process-local dict guarded by a Lock.
No Redis, no database, no persistence across restarts. This is a deliberate
simplification for the Tier 1 native Mac app.

A job represents one full load + calibrate pipeline run. The frontend kicks
it off via ``POST /api/v5/microdata/prepare``, then polls
``GET /api/v5/microdata/job/{job_id}`` until status flips to ``done`` (or
``error``), then fetches the rendered dashboards.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.microdata.cache import CacheKey

logger = logging.getLogger(__name__)


# Job status state machine.
JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_ERROR = "error"


@dataclass
class JobState:
    """Mutable state of one pipeline job."""

    job_id: str
    year: int
    sample_size: int
    rng_seed: int
    status: str = JOB_QUEUED
    progress: float = 0.0  # 0.0..1.0
    stage: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    result_key: Optional[CacheKey] = None  # the calibrated cache key on success

    @property
    def elapsed_seconds(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at


class JobManager:
    """Thread-safe in-memory job registry."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def create(self, *, year: int, sample_size: int, rng_seed: int) -> JobState:
        """Allocate a new job_id and register a queued job."""
        job_id = uuid.uuid4().hex[:12]
        job = JobState(job_id=job_id, year=year, sample_size=sample_size, rng_seed=rng_seed)
        with self._lock:
            self._jobs[job_id] = job
        logger.info("Job created: %s (year=%d, sample=%d)", job_id, year, sample_size)
        return job

    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            return self._jobs.get(job_id)

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JOB_RUNNING
            job.started_at = time.time()
            job.stage = "starting"
            job.progress = 0.0

    def update_progress(
        self, job_id: str, *, progress: float, stage: Optional[str] = None
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.progress = max(0.0, min(1.0, progress))
            if stage is not None:
                job.stage = stage

    def mark_done(self, job_id: str, *, result_key: CacheKey) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JOB_DONE
            job.progress = 1.0
            job.stage = "done"
            job.finished_at = time.time()
            job.result_key = result_key
        logger.info("Job %s done", job_id)

    def mark_error(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JOB_ERROR
            job.error = message
            job.finished_at = time.time()
        logger.warning("Job %s errored: %s", job_id, message)

    def clear(self) -> int:
        """Remove all jobs from the registry. Returns the count removed.

        Mainly used by tests; production code rarely needs to call this.
        """
        with self._lock:
            n = len(self._jobs)
            self._jobs.clear()
            return n


# Module-level singleton — single-user app, one process.
job_manager = JobManager()
