"""
Dispatcher dedicado para calibrações pesadas em background.

Evita usar o executor implícito do event loop e isola a fila de trabalho do
pipeline de microdados em um pool próprio do runtime local.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import threading
from typing import Callable


_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="xtri-calibration",
)
_lock = threading.Lock()
_futures: dict[str, Future] = {}


def submit_microdata_job(
    runner: Callable[[str, int, int, int, bool], None],
    *,
    job_id: str,
    year: int,
    sample_size: int,
    rng_seed: int,
    force_rerun: bool,
) -> Future:
    """Enfileira um job de microdados no executor dedicado."""

    future = _executor.submit(runner, job_id, year, sample_size, rng_seed, force_rerun)
    with _lock:
        _futures[job_id] = future

    def _cleanup(done_future: Future) -> None:
        with _lock:
            _futures.pop(job_id, None)

    future.add_done_callback(_cleanup)
    return future


def active_job_ids() -> list[str]:
    """Lista jobs ainda registrados no executor dedicado."""

    with _lock:
        return sorted(_futures.keys())
