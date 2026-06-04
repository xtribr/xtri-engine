"""
Streaming loader for the official ENEM microdata CSV.

This module is the entry point for the microdata analysis pipeline. It:

1. Locates the CSV at a hardcoded path
   (``/Volumes/KINGSTON/apps/MIRT/microdata/MICRODADOS_ENEM_<year>.csv``).
2. Reads it in chunks via ``pd.read_csv(chunksize=100_000)`` with a column
   projection (``usecols=``) to keep peak memory under ~1 GB.
3. Filters each chunk to valid first-application students using
   :mod:`app.microdata.filters`.
4. Concatenates surviving rows and draws a reproducible random sample
   (default 200K).

Downstream modules (Phase 3+) consume the returned DataFrame for V3 Local
calibration, dashboard generation, and drill-down aggregation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from app.microdata.filters import is_primeira_aplicacao, is_valid_student
from app.microdata.schema import AREAS, MicrodataSchema, detect_schema

logger = logging.getLogger(__name__)

# Hardcoded path per Q1 decision (single-user, stable KINGSTON location).
# Caller can override by passing ``csv_path=`` explicitly.
MICRODATA_DIR = Path("/Volumes/KINGSTON/apps/MIRT/microdata")
FILENAME_TEMPLATE = "MICRODADOS_ENEM_{year}.csv"

# Chunk size for streaming read. Tuned for ~1 GB peak RAM on a 16 GB Mac.
DEFAULT_CHUNKSIZE = 100_000

# Default random-sample size. Q4 default per user confirmation.
DEFAULT_SAMPLE_SIZE = 200_000

# INEP microdata is always semicolon-delimited latin-1.
ENCODING = "latin-1"
DELIMITER = ";"


def default_csv_path(year: int) -> Path:
    """Return the canonical microdata CSV path for a given year."""
    return MICRODATA_DIR / FILENAME_TEMPLATE.format(year=year)


def load_primeira_aplicacao_provas(
    year: int, area: Optional[str] = None
) -> Optional[frozenset[int]]:
    """Fetch the primeira-aplicação co_prova whitelist from Supabase.

    Queries ``enem_codigos_prova`` for rows where ``is_reaplicacao == False``
    (and optionally ``ano == year`` and ``area == area``). Returns the set of
    ``co_prova`` values.

    Returns ``None`` if Supabase is unconfigured, unreachable, or returns no
    matching rows. Callers should treat ``None`` as "no filter available" and
    decide whether to proceed unfiltered or raise.

    Args:
        year: ENEM year, used to scope the whitelist. Required because
            co_prova codes are reused across years.
        area: Optional area filter (e.g. "MT"). If ``None``, returns codes
            for all areas.

    Returns:
        A frozenset of int co_prova codes, or ``None`` on failure.
    """
    try:
        from app.db.supabase_client import get_supabase, is_supabase_configured  # noqa: PLC0415
    except ImportError:
        logger.warning("app.db.supabase_client not available")
        return None

    if not is_supabase_configured():
        logger.warning(
            "Supabase not configured — cannot load primeira-aplicação whitelist. "
            "Set SUPABASE_URL and SUPABASE_KEY in ~/Library/Application Support/"
            "XTRI Engine/.env"
        )
        return None

    client = get_supabase()
    if client is None:
        return None

    try:
        query = client.table("enem_codigos_prova").select("co_prova").eq("ano", year).eq(
            "is_reaplicacao", False
        )
        if area is not None:
            query = query.eq("area", area.upper())
        response = query.execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to query enem_codigos_prova: %s", exc)
        return None

    rows = response.data or []
    if not rows:
        logger.warning(
            "enem_codigos_prova returned 0 rows for year=%d area=%s — whitelist empty",
            year,
            area or "ALL",
        )
        return None

    codes = frozenset(int(r["co_prova"]) for r in rows if r.get("co_prova") is not None)
    logger.info(
        "Loaded %d primeira-aplicação co_prova codes for year=%d area=%s",
        len(codes),
        year,
        area or "ALL",
    )
    return codes


def discover_available_years() -> list[int]:
    """Scan :data:`MICRODATA_DIR` and return years with a CSV present.

    Returns a sorted list of ints. Empty if the directory doesn't exist yet or
    contains no matching files.
    """
    if not MICRODATA_DIR.is_dir():
        return []
    years: list[int] = []
    for path in MICRODATA_DIR.glob("MICRODADOS_ENEM_*.csv"):
        stem = path.stem  # e.g. "MICRODADOS_ENEM_2024"
        suffix = stem.replace("MICRODADOS_ENEM_", "", 1)
        try:
            years.append(int(suffix))
        except ValueError:
            continue
    return sorted(years)


def load_microdata(
    year: int,
    *,
    csv_path: Optional[Path | str] = None,
    sample_size: Optional[int] = DEFAULT_SAMPLE_SIZE,
    rng_seed: int = 42,
    provas_primeira: Optional[Iterable[int]] = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
    progress_callback: Optional[callable] = None,
) -> pd.DataFrame:
    """Load, filter, and sample microdata for a given ENEM year.

    Args:
        year: ENEM year (e.g. 2024). Used to locate the default CSV path.
        csv_path: Override for the microdata CSV location. Defaults to
            :func:`default_csv_path(year)`.
        sample_size: Number of valid students to keep after filtering. If
            ``None``, returns all valid rows (memory-heavy — use for testing
            only). Defaults to 200_000.
        rng_seed: Random seed for reproducible sampling.
        provas_primeira: Optional whitelist of primeira-aplicação co_prova
            codes. If provided, enables strict filtering (drops reaplicação
            rows). If ``None``, falls back to ``IN_TREINEIRO == 0``.
        chunksize: Rows per ``pd.read_csv`` chunk. Smaller = lower peak RAM,
            slower wall time. 100K is a good balance on 16 GB Macs.
        progress_callback: Optional callable invoked as
            ``progress_callback(rows_read, rows_kept)`` once per chunk.

    Returns:
        A DataFrame of at most ``sample_size`` valid first-application
        students. Columns are whatever :meth:`MicrodataSchema.all_needed_columns`
        returned. Row order is randomized by the sample step.

    Raises:
        FileNotFoundError: If the resolved CSV path doesn't exist.
        RuntimeError: If zero valid students are found (usually means the
            filter is wrong for this year's schema).
    """
    path = Path(csv_path) if csv_path is not None else default_csv_path(year)
    if not path.is_file():
        raise FileNotFoundError(
            f"Microdata CSV for year {year} not found at {path}. "
            f"Download MICRODADOS_ENEM_{year}.csv from INEP and copy to {MICRODATA_DIR}/"
        )

    schema = detect_schema(path, encoding=ENCODING)
    usecols = schema.all_needed_columns()
    logger.info("Loading microdata for year %d from %s", year, path)
    logger.info("Projecting %d columns: %s...", len(usecols), usecols[:5])

    t0 = time.time()
    rows_read = 0
    kept_chunks: list[pd.DataFrame] = []

    # Read in chunks, filter each, concat survivors.
    reader = pd.read_csv(
        path,
        sep=DELIMITER,
        encoding=ENCODING,
        usecols=usecols,
        dtype=str,  # keep everything as strings; filters coerce as needed
        chunksize=chunksize,
        on_bad_lines="skip",
        low_memory=False,
    )

    for chunk in reader:
        rows_read += len(chunk)
        mask = is_valid_student(chunk, schema) & is_primeira_aplicacao(
            chunk, schema, provas_primeira=provas_primeira
        )
        kept = chunk.loc[mask].copy()
        if not kept.empty:
            kept_chunks.append(kept)
        if progress_callback is not None:
            progress_callback(rows_read, sum(len(k) for k in kept_chunks))

    if not kept_chunks:
        raise RuntimeError(
            f"Zero valid students found in {path} after filtering. "
            f"This usually means the schema detection is wrong for year {year}."
        )

    valid_df = pd.concat(kept_chunks, ignore_index=True)
    n_valid = len(valid_df)
    elapsed = time.time() - t0
    logger.info(
        "Read %s rows in %.1fs, kept %s valid (%.1f%%)",
        f"{rows_read:,}",
        elapsed,
        f"{n_valid:,}",
        100.0 * n_valid / rows_read if rows_read else 0.0,
    )

    # Random sample step. If sample_size is None or larger than available,
    # return all valid rows (shuffled for downstream repeatability).
    if sample_size is None or sample_size >= n_valid:
        return valid_df.sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)
    return (
        valid_df.sample(n=sample_size, random_state=rng_seed).reset_index(drop=True)
    )
