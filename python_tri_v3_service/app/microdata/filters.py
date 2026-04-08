"""
Vectorized predicates for filtering microdata rows.

These are pandas-aware versions of the row-by-row filters used in the existing
scripts (``validacao_massa.py``, ``calcular_transformacao.py``). They operate
on a full DataFrame chunk at once for ~100x speedup over Python-level loops.

The "valid student" definition matches the existing scripts exactly:
    TP_PRESENCA_LC == "1"
    TP_PRESENCA_CH == "1"
    TP_PRESENCA_CN == "1"
    TP_PRESENCA_MT == "1"
    TP_STATUS_REDACAO == "1"
    NU_NOTA_{area} is not null for all 4 areas
    NU_NOTA_REDACAO is not null

The "primeira aplicação" filter is strict-only. The caller must supply a
whitelist of first-application ``co_prova`` codes (typically loaded from the
Supabase ``enem_codigos_prova`` table where ``is_reaplicacao == False``).
If ``provas_primeira`` is ``None``, no filter is applied — log a clear
message so the caller knows the sample is unfiltered.

NB: INEP removed ``IN_TREINEIRO`` from ENEM 2024+ microdata, so the old
"drop treineiros via flag" approach no longer works. Strict whitelist is the
only reliable filter.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from app.microdata.schema import AREAS, MicrodataSchema

logger = logging.getLogger(__name__)


def is_valid_student(df: pd.DataFrame, schema: MicrodataSchema) -> pd.Series:
    """Return a boolean mask of rows matching the valid-student definition.

    Args:
        df: Chunk of microdata as read by ``pd.read_csv`` (string dtypes).
        schema: Column-name accessor for the detected year's layout.

    Returns:
        A boolean Series aligned with ``df.index`` where True means the row is
        a valid student (present in all 4 areas + valid redação).
    """
    # Presence flags: must all equal the string "1". Microdata stores everything
    # as strings when loaded with default dtypes.
    presenca_mask = pd.Series(True, index=df.index)
    for area in AREAS:
        col = schema.tp_presenca[area]
        presenca_mask &= df[col].astype(str).str.strip() == "1"
    presenca_mask &= df[schema.tp_status_redacao].astype(str).str.strip() == "1"

    # Notas must be parseable numbers > 0 for all 4 areas + redação.
    notas_mask = pd.Series(True, index=df.index)
    for area in AREAS:
        col = schema.nu_nota[area]
        nota = _parse_nota_column(df[col])
        notas_mask &= nota.notna() & (nota > 0)
    red_nota = _parse_nota_column(df[schema.nu_nota_redacao])
    notas_mask &= red_nota.notna()  # redação can legitimately be 0, only require not-null

    return presenca_mask & notas_mask


def is_primeira_aplicacao(
    df: pd.DataFrame,
    schema: MicrodataSchema,
    *,
    provas_primeira: Optional[Iterable[int]] = None,
) -> pd.Series:
    """Return a boolean mask for primeira-aplicação rows (strict whitelist only).

    A row passes iff its ``CO_PROVA_{area}`` value for EVERY area is in the
    ``provas_primeira`` whitelist. This matches how
    ``calcular_transformacao.processar_microdados`` uses
    ``enem_codigos_prova.is_reaplicacao == False`` loaded from Supabase.

    If ``provas_primeira`` is ``None``, no filter is applied — a warning is
    logged so the caller is aware the sample is unfiltered. INEP removed
    ``IN_TREINEIRO`` from ENEM 2024+ microdata, so there is no reliable
    fallback without the Supabase whitelist.

    Args:
        df: Chunk of microdata.
        schema: Column-name accessor.
        provas_primeira: Iterable of primeira-aplicação co_prova codes loaded
            from Supabase ``enem_codigos_prova``. If ``None``, no filter is
            applied.

    Returns:
        A boolean Series aligned with ``df.index``.
    """
    if provas_primeira is None:
        logger.warning(
            "is_primeira_aplicacao called without provas_primeira whitelist — "
            "no primeira-aplicação filter applied (sample includes reaplicação)"
        )
        return pd.Series(True, index=df.index)

    whitelist = set(int(x) for x in provas_primeira)
    mask = pd.Series(True, index=df.index)
    for area in AREAS:
        col = schema.co_prova[area]
        mask &= df[col].apply(_safe_int).isin(whitelist)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_nota_column(series: pd.Series) -> pd.Series:
    """Convert a string-typed NU_NOTA column to float, handling BR decimal format.

    INEP microdata uses ``,`` as decimal separator (e.g. ``"723,4"``). This
    helper converts to float and returns NaN for empty/invalid cells.
    """
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _safe_int(value: object) -> Optional[int]:
    """Best-effort int conversion for CO_PROVA columns."""
    try:
        return int(str(value).strip())
    except (ValueError, AttributeError, TypeError):
        return None
