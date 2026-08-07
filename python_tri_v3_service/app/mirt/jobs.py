"""Fila persistente de calibrações MIRT executadas pelo worker R."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
import uuid
from typing import Any, Callable, Optional

from ..runtime import local_paths


logger = logging.getLogger(__name__)

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_ERROR = "error"

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xtri-mirt-fit")
_futures_lock = threading.Lock()
_futures: dict[str, Future[Any]] = {}
MAX_ACTIVE_JOBS = max(1, int(os.getenv("XTRI_MIRT_MAX_ACTIVE_JOBS", "1")))


class MIRTActiveJobError(RuntimeError):
    """Já existe uma calibração queued/running para o mesmo escopo."""

    def __init__(self, job_id: str):
        super().__init__(f"Já existe uma calibração ativa para este escopo: {job_id}")
        self.job_id = job_id


class MIRTQueueFullError(RuntimeError):
    """A fila global atingiu o limite configurado de payloads pesados."""


def _jobs_dir() -> Path:
    return local_paths.JOBS_ROOT / "mirt"


def _job_file(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{16}", job_id):
        raise ValueError("job_id MIRT inválido")
    return _jobs_dir() / f"{job_id}.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class MIRTJobState:
    """Estado serializável de uma calibração MIRT."""

    job_id: str
    simulado_id: str
    area: str
    ano_referencia: int
    n_alunos: int
    n_itens: int
    status: str = JOB_QUEUED
    progress: float = 0.0
    stage: Optional[str] = "queued"
    model_id: Optional[str] = None
    artifact_id: Optional[str] = None
    fit_summary: Optional[dict[str, Any]] = None
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def elapsed_seconds(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return round(end - self.started_at, 3)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    def to_api_payload(self) -> dict[str, Any]:
        payload = self.to_payload()
        payload.pop("created_at", None)
        payload.pop("started_at", None)
        payload.pop("finished_at", None)
        payload["elapsed_seconds"] = self.elapsed_seconds
        return payload


class MIRTJobManager:
    """Registry em disco, legível por qualquer worker Uvicorn."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def _active_for_scope_unlocked(
        self, *, simulado_id: str, area: str, ano_referencia: int
    ) -> Optional[MIRTJobState]:
        for path in _jobs_dir().glob("*.json"):
            try:
                job = MIRTJobState(**_read_json(path))
            except Exception:  # noqa: BLE001
                continue
            if (
                job.simulado_id == simulado_id
                and job.area == area.upper()
                and job.ano_referencia == ano_referencia
                and job.status in {JOB_QUEUED, JOB_RUNNING}
            ):
                return job
        return None

    def _count_active_unlocked(self) -> int:
        count = 0
        for path in _jobs_dir().glob("*.json"):
            try:
                job = MIRTJobState(**_read_json(path))
            except Exception:  # noqa: BLE001
                continue
            if job.status in {JOB_QUEUED, JOB_RUNNING}:
                count += 1
        return count

    def create(
        self,
        *,
        simulado_id: str,
        area: str,
        ano_referencia: int,
        n_alunos: int,
        n_itens: int,
        require_scope_available: bool = False,
    ) -> MIRTJobState:
        with self._lock:
            if require_scope_available:
                active = self._active_for_scope_unlocked(
                    simulado_id=simulado_id,
                    area=area,
                    ano_referencia=ano_referencia,
                )
                if active is not None:
                    raise MIRTActiveJobError(active.job_id)
            if self._count_active_unlocked() >= MAX_ACTIVE_JOBS:
                raise MIRTQueueFullError(
                    f"Limite global de {MAX_ACTIVE_JOBS} calibração(ões) MIRT ativa(s) atingido"
                )
            job = MIRTJobState(
                job_id=uuid.uuid4().hex[:16],
                simulado_id=simulado_id,
                area=area.upper(),
                ano_referencia=ano_referencia,
                n_alunos=n_alunos,
                n_itens=n_itens,
            )
            _write_json(_job_file(job.job_id), job.to_payload())
        return job

    def get(self, job_id: str) -> Optional[MIRTJobState]:
        try:
            path = _job_file(job_id)
        except ValueError:
            return None
        if not path.is_file():
            return None
        try:
            return MIRTJobState(**_read_json(path))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            logger.exception("Falha ao ler job MIRT %s", job_id)
            return None

    def _update(self, job_id: str, **changes: Any) -> Optional[MIRTJobState]:
        with self._lock:
            job = self.get(job_id)
            if job is None:
                return None
            for key, value in changes.items():
                setattr(job, key, value)
            _write_json(_job_file(job_id), job.to_payload())
            return job

    def mark_running(self, job_id: str) -> None:
        self._update(
            job_id,
            status=JOB_RUNNING,
            progress=0.1,
            stage="preparing_input",
            started_at=time.time(),
            error=None,
        )

    def update_progress(self, job_id: str, *, progress: float, stage: str) -> None:
        self._update(
            job_id,
            progress=max(0.0, min(1.0, progress)),
            stage=stage,
        )

    def mark_done(
        self,
        job_id: str,
        *,
        model_id: str,
        artifact_id: str,
        fit_summary: dict[str, Any],
        warnings: list[str],
    ) -> None:
        self._update(
            job_id,
            status=JOB_DONE,
            progress=1.0,
            stage="done",
            model_id=model_id,
            artifact_id=artifact_id,
            fit_summary=fit_summary,
            warnings=warnings,
            finished_at=time.time(),
        )

    def mark_error(self, job_id: str, message: str) -> None:
        self._update(
            job_id,
            status=JOB_ERROR,
            stage="error",
            error=message,
            finished_at=time.time(),
        )
        logger.warning("Calibração MIRT %s falhou: %s", job_id, message)

    def reload_from_disk(self) -> int:
        """Marca jobs interrompidos por reinício e retorna o total encontrado."""

        jobs_dir = _jobs_dir()
        jobs_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for path in jobs_dir.glob("*.json"):
            count += 1
            try:
                job = MIRTJobState(**_read_json(path))
            except Exception:  # noqa: BLE001
                logger.exception("Sidecar MIRT inválido: %s", path)
                continue
            if job.status in {JOB_QUEUED, JOB_RUNNING}:
                job.status = JOB_ERROR
                job.stage = "interrupted"
                job.error = "Calibração interrompida pelo reinício do serviço"
                job.finished_at = time.time()
                _write_json(path, job.to_payload())
                artifact_base = (
                    local_paths.MIRT_MODELS_ROOT
                    / str(job.ano_referencia)
                    / job.area
                )
                for sensitive_name in ("response_matrix.csv", "factor_scores.csv"):
                    for sensitive_path in artifact_base.glob(
                        f"*/{job.job_id}/{sensitive_name}"
                    ):
                        try:
                            sensitive_path.unlink(missing_ok=True)
                        except OSError:
                            logger.warning(
                                "Falha ao limpar dado sensível de job interrompido: %s",
                                sensitive_path,
                            )
        return count

    def active_for_scope(
        self, *, simulado_id: str, area: str, ano_referencia: int
    ) -> Optional[MIRTJobState]:
        """Localiza calibração queued/running para o mesmo escopo."""

        with self._lock:
            return self._active_for_scope_unlocked(
                simulado_id=simulado_id,
                area=area,
                ano_referencia=ano_referencia,
            )

    def clear(self) -> int:
        paths = list(_jobs_dir().glob("*.json"))
        for path in paths:
            try:
                path.unlink()
            except OSError:
                pass
        return len(paths)


def submit_mirt_job(job_id: str, runner: Callable[[str], None]) -> Future[Any]:
    """Enfileira uma calibração no executor R dedicado."""

    future = _executor.submit(runner, job_id)
    with _futures_lock:
        _futures[job_id] = future

    def _cleanup(_future: Future[Any]) -> None:
        with _futures_lock:
            _futures.pop(job_id, None)

    future.add_done_callback(_cleanup)
    return future


job_manager = MIRTJobManager()
