"""
Parquet cache layer for filtered microdata DataFrames.

Saves the expensive result of :func:`app.microdata.loader.load_microdata`
(which streams a 1.5 GB CSV in ~60s) as a Parquet file so subsequent loads
complete in <2s.

Cache location:
    ``~/Library/Application Support/XTRI Engine/cache/microdata/<year>/``

Legacy compatibility:
    Reads also from ``~/Library/Application Support/XTRI Engine/microdata_cache/``
    while o runtime desktop migra para o layout novo.

Each cache entry is:
    ``<source_hash[:16]>_<filters_hash[:8]>.parquet``    — the DataFrame
    ``<source_hash[:16]>_<filters_hash[:8]>.meta.json``  — invalidation metadata

The ``source_hash`` is derived from the source CSV's path, size, and mtime
(NOT a full content hash — that would take 10+ seconds on a 1.5 GB file). If
the source file changes size or mtime, the cache key changes and the old
Parquet is silently ignored (still on disk for cleanup later).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..runtime.local_paths import LEGACY_MICRODATA_CACHE_ROOT, MICRODATA_CACHE_ROOT

logger = logging.getLogger(__name__)

CACHE_ROOT = MICRODATA_CACHE_ROOT
LEGACY_CACHE_ROOT = LEGACY_MICRODATA_CACHE_ROOT


@dataclass(frozen=True)
class CacheKey:
    """Stable identifier for a cache entry.

    Combines a source-file identity (``source_hash``) with a filter-config
    identity (``filters_hash``) and a processing stage. The stage lets
    downstream phases (e.g. Phase 3 calibration) cache their output
    independently from the raw-sample cache without collision.
    """

    year: int
    source_hash: str  # hex, first 16 chars
    filters_hash: str  # hex, first 8 chars
    stage: str = "raw"  # "raw" for the filtered sample, "calibrated" for V3 results, etc.

    @property
    def stem(self) -> str:
        if self.stage == "raw":
            return f"{self.source_hash}_{self.filters_hash}"
        return f"{self.source_hash}_{self.filters_hash}_{self.stage}"

    @property
    def parquet_path(self) -> Path:
        return CACHE_ROOT / str(self.year) / f"{self.stem}.parquet"

    @property
    def meta_path(self) -> Path:
        return CACHE_ROOT / str(self.year) / f"{self.stem}.meta.json"

    def with_stage(self, stage: str) -> "CacheKey":
        """Return a sibling key for a different processing stage."""
        return CacheKey(
            year=self.year,
            source_hash=self.source_hash,
            filters_hash=self.filters_hash,
            stage=stage,
        )


@dataclass(frozen=True)
class CacheMeta:
    """Sidecar metadata written alongside each Parquet.

    Used primarily for human inspection and sanity-checking — the actual
    invalidation happens via the cache key changing (new mtime → new hash).
    """

    year: int
    source_path: str
    source_size: int
    source_mtime: float
    n_rows: int
    filters_hash: str
    filters_description: str
    created_at: float


def _source_hash(source_path: Path) -> str:
    """Stable identity for a source file without reading its content.

    Combines the absolute path, size (bytes), and mtime (to nanosecond
    precision). This is ~1000x faster than hashing the file itself and is
    sufficient for cache invalidation on a single-user system where the file
    isn't being edited in place.
    """
    if not source_path.is_file():
        raise FileNotFoundError(f"Source file not found for cache key: {source_path}")
    stat = source_path.stat()
    blob = f"{source_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _filters_hash(
    sample_size: Optional[int],
    rng_seed: int,
    provas_primeira: Optional[frozenset[int]],
) -> str:
    """Stable identity for the filter + sampling configuration."""
    parts = [
        f"sample={sample_size}",
        f"seed={rng_seed}",
        f"provas={'strict' if provas_primeira else 'loose'}",
    ]
    if provas_primeira:
        # Sort so the hash is order-independent.
        parts.append(f"provas_set={','.join(str(p) for p in sorted(provas_primeira))}")
    blob = "|".join(parts).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def cache_key(
    source_path: Path | str,
    year: int,
    *,
    sample_size: Optional[int],
    rng_seed: int,
    provas_primeira: Optional[frozenset[int]] = None,
) -> CacheKey:
    """Derive a :class:`CacheKey` from the source file and filter config."""
    return CacheKey(
        year=year,
        source_hash=_source_hash(Path(source_path)),
        filters_hash=_filters_hash(sample_size, rng_seed, provas_primeira),
    )


def load_cached(key: CacheKey) -> Optional[pd.DataFrame]:
    """Try to load a cached DataFrame. Returns ``None`` on miss or error.

    Gracefully handles the case where ``pyarrow`` is missing — logs a warning
    and returns ``None`` so the caller falls back to re-reading the CSV.
    """
    candidate_paths = [
        key.parquet_path,
        LEGACY_CACHE_ROOT / str(key.year) / key.parquet_path.name,
    ]
    for candidate in candidate_paths:
        if not candidate.is_file():
            continue
        try:
            df = pd.read_parquet(candidate)
            logger.info("Cache HIT: %s (%s rows)", candidate.name, f"{len(df):,}")
            return df
        except ImportError:
            logger.warning(
                "pyarrow not installed — cannot read Parquet cache. "
                "Install pyarrow to enable caching."
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache read failed for %s: %s", candidate, exc)
            return None
    return None


def save_cached(
    df: pd.DataFrame,
    key: CacheKey,
    *,
    source_path: Path | str,
    filters_description: str = "",
) -> bool:
    """Write a DataFrame to the Parquet cache.

    Returns True on success, False on error (missing pyarrow, permission, etc).
    Creates the cache directory if needed.
    """
    try:
        key.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(key.parquet_path, engine="pyarrow", compression="snappy")
    except ImportError:
        logger.warning(
            "pyarrow not installed — cannot write Parquet cache. "
            "Install pyarrow to enable caching."
        )
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache write failed for %s: %s", key.parquet_path, exc)
        return False

    # Write the sidecar metadata too (best-effort, not fatal if it fails).
    try:
        source_path = Path(source_path)
        stat = source_path.stat()
        import time  # noqa: PLC0415

        meta = CacheMeta(
            year=key.year,
            source_path=str(source_path.resolve()),
            source_size=stat.st_size,
            source_mtime=stat.st_mtime,
            n_rows=len(df),
            filters_hash=key.filters_hash,
            filters_description=filters_description,
            created_at=time.time(),
        )
        with open(key.meta_path, "w") as f:
            json.dump(asdict(meta), f, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache meta write failed for %s: %s", key.meta_path, exc)

    logger.info("Cache SAVE: %s (%s rows)", key.parquet_path.name, f"{len(df):,}")
    return True


def clear_year_cache(year: int) -> int:
    """Remove all cache entries for a given year. Returns number of files deleted."""
    count = 0
    for year_dir in (CACHE_ROOT / str(year), LEGACY_CACHE_ROOT / str(year)):
        if not year_dir.is_dir():
            continue
        for entry in year_dir.iterdir():
            if entry.is_file():
                try:
                    entry.unlink()
                    count += 1
                except OSError:
                    pass
    return count


def cache_size_bytes() -> int:
    """Total bytes consumed by the microdata cache directory."""
    total = 0
    for cache_root in (CACHE_ROOT, LEGACY_CACHE_ROOT):
        if not cache_root.is_dir():
            continue
        for path in cache_root.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
    return total
