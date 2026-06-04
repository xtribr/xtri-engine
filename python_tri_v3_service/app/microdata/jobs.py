"""
Job manager local do pipeline de microdados.

Mantém o estado em memória para acesso rápido, mas também persiste sidecars
JSON em `Application Support/jobs/` para o app desktop não perder o histórico
quando reinicia no meio de uma calibração.
"""

from __future__ import annotations

import logging
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from app.microdata.cache import CacheKey
from app.runtime import local_paths

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

    def to_payload(self) -> dict:
        """Serializa o job para persistência em JSON."""

        payload = asdict(self)
        if self.result_key is not None:
            payload["result_key"] = asdict(self.result_key)
        return payload

    @classmethod
    def from_payload(cls, payload: dict) -> "JobState":
        """Reconstrói um `JobState` a partir do sidecar persistido."""

        result_key = payload.get("result_key")
        if result_key is not None:
            payload = dict(payload)
            payload["result_key"] = CacheKey(**result_key)
        return cls(**payload)


def _jobs_dir() -> Path:
    """Diretório persistente dos jobs locais do pipeline."""

    return local_paths.JOBS_ROOT / "microdata"


def _job_file(job_id: str) -> Path:
    """Caminho do sidecar JSON persistido para um job."""

    return _jobs_dir() / f"{job_id}.json"


def _write_json(path: Path, payload: dict) -> None:
    """Escreve JSON de forma atômica para evitar truncamento."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class JobManager:
    """Registry thread-safe com persistência local para jobs do pipeline."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def _persist_locked(self, job: JobState) -> None:
        _write_json(_job_file(job.job_id), job.to_payload())

    def reload_from_disk(self) -> int:
        """Recarrega jobs persistidos e normaliza execuções interrompidas."""

        jobs_dir = _jobs_dir()
        jobs_dir.mkdir(parents=True, exist_ok=True)
        restored: dict[str, JobState] = {}
        now = time.time()

        for path in jobs_dir.glob("*.json"):
            try:
                job = JobState.from_payload(_read_json(path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read persisted job %s: %s", path, exc)
                continue

            if job.status in {JOB_QUEUED, JOB_RUNNING}:
                job.status = JOB_ERROR
                job.stage = "interrupted"
                job.finished_at = now
                if not job.error:
                    job.error = "Job interrompido por reinício do app"
                _write_json(path, job.to_payload())
            restored[job.job_id] = job

        with self._lock:
            self._jobs = restored
        return len(restored)

    def create(self, *, year: int, sample_size: int, rng_seed: int) -> JobState:
        """Allocate a new job_id and register a queued job."""
        job_id = uuid.uuid4().hex[:12]
        job = JobState(job_id=job_id, year=year, sample_size=sample_size, rng_seed=rng_seed)
        with self._lock:
            self._jobs[job_id] = job
            self._persist_locked(job)
        logger.info("Job created: %s (year=%d, sample=%d)", job_id, year, sample_size)
        return job

    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self) -> list[JobState]:
        """Retorna uma cópia rasa dos jobs atuais para endpoints de status."""

        with self._lock:
            return list(self._jobs.values())

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JOB_RUNNING
            job.started_at = time.time()
            job.stage = "starting"
            job.progress = 0.0
            self._persist_locked(job)

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
            self._persist_locked(job)

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
            self._persist_locked(job)
        logger.info("Job %s done", job_id)

    def mark_error(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JOB_ERROR
            job.error = message
            job.finished_at = time.time()
            self._persist_locked(job)
        logger.warning("Job %s errored: %s", job_id, message)

    def clear(self) -> int:
        """Remove all jobs from the registry. Returns the count removed.

        Mainly used by tests; production code rarely needs to call this.
        """
        with self._lock:
            n = len(self._jobs)
            self._jobs.clear()
        for path in _jobs_dir().glob("*.json"):
            try:
                path.unlink()
            except OSError:
                pass
        return n


# Module-level singleton — single-user app, one process.
job_manager = JobManager()
