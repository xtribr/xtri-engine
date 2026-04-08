"""
Per-group (UF / Município / Escola) aggregation for the microdata drill-down.

The microdata page is strictly INEP-only, so every stat here is computed from
``nota_ref_*`` (the official NU_NOTA_* columns) — we never touch V3 Local
columns. This mirrors the contract of :mod:`app.microdata.dashboard_data`.

Output shape (one row per group):
    {
        "key": "SP",                 # UF sigla / city name / school code
        "label": "SP",               # human-readable label (same as key for UF)
        "uf": "SP",                  # UF sigla (always filled for context)
        "municipio": "São Paulo",    # city name (only for municipio/escola)
        "dependencia": "Estadual",   # only for escola
        "localizacao": "Urbana",     # only for escola
        "n": 12345,
        "mean": 582.3,
        "median": 580.1,
        "p25": 510.0,
        "p75": 640.0,
        "min": 340.0,
        "max": 920.0,
        "mean_acertos": 28.4,
    }

Schools that have fewer than ``min_n`` students in the sample are dropped —
per-school stats on n<5 are statistical noise and mislead drill-down users.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from app.microdata.schema import AREAS

GroupBy = Literal["uf", "municipio", "escola"]

# Minimum students per group. Schools with <5 in a 200K sample are noise.
DEFAULT_MIN_N: dict[GroupBy, int] = {"uf": 1, "municipio": 5, "escola": 5}

# Max rows returned per breakdown call. UF is tiny (27) so no cap there; the
# others cap at this value to keep the JSON payload manageable.
DEFAULT_LIMIT: dict[GroupBy, int | None] = {
    "uf": None,
    "municipio": 500,
    "escola": 1000,
}

# TP_DEPENDENCIA_ADM_ESC mapping per INEP dictionary.
_DEPENDENCIA_LABELS: dict[str, str] = {
    "1": "Federal",
    "2": "Estadual",
    "3": "Municipal",
    "4": "Privada",
}

# TP_LOCALIZACAO_ESC mapping per INEP dictionary.
_LOCALIZACAO_LABELS: dict[str, str] = {
    "1": "Urbana",
    "2": "Rural",
}


def compute_breakdown(
    df: pd.DataFrame,
    area: str,
    *,
    group_by: GroupBy,
    min_n: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Aggregate INEP nota_ref stats per UF / Município / Escola.

    Args:
        df: Calibrated DataFrame (we only read ``nota_ref_*``, ``acertos_v3_*``,
            and the geo columns).
        area: ENEM area code (LC, CH, CN, MT).
        group_by: Grouping dimension.
        min_n: Drop groups with fewer students than this. Defaults per
            :data:`DEFAULT_MIN_N` (1 for UF, 5 for municipio/escola).
        limit: Max rows to return, sorted by median desc. ``None`` = no limit.
            Defaults per :data:`DEFAULT_LIMIT`.

    Returns:
        A list of dicts (one per group) sorted by median desc. Empty if the
        requested geo column is absent from the DataFrame (older years).
    """
    if area not in AREAS:
        raise ValueError(f"Unknown area: {area}")
    if group_by not in ("uf", "municipio", "escola"):
        raise ValueError(f"Unknown group_by: {group_by}")

    al = area.lower()
    nota_col = f"nota_ref_{al}"
    ac_col = f"acertos_v3_{al}"

    # Pick the grouping key column per dimension. If it's missing we return
    # empty so the frontend can render "not available for this year".
    key_col = _key_column_for(group_by)
    if key_col not in df.columns or nota_col not in df.columns:
        return []

    effective_min_n = min_n if min_n is not None else DEFAULT_MIN_N[group_by]
    effective_limit = limit if limit is not None else DEFAULT_LIMIT[group_by]

    # Base mask: must have nota + key + acertos
    cols_needed = [key_col, nota_col]
    if ac_col in df.columns:
        cols_needed.append(ac_col)
    # Context columns (always include what exists)
    for ctx in ("SG_UF_ESC", "NO_MUNICIPIO_ESC", "TP_DEPENDENCIA_ADM_ESC", "TP_LOCALIZACAO_ESC"):
        if ctx in df.columns and ctx not in cols_needed:
            cols_needed.append(ctx)

    sub = df.loc[df[nota_col].notna() & df[key_col].notna(), cols_needed].copy()
    if sub.empty:
        return []

    # Groupby + aggregate in one pass
    grouped = sub.groupby(key_col, sort=False)
    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        n = len(group)
        if n < effective_min_n:
            continue
        notas = group[nota_col].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "key": str(key),
            "label": str(key),
            "n": int(n),
            "mean": round(float(notas.mean()), 1),
            "median": round(float(np.median(notas)), 1),
            "p25": round(float(np.percentile(notas, 25)), 1),
            "p75": round(float(np.percentile(notas, 75)), 1),
            "min": round(float(notas.min()), 1),
            "max": round(float(notas.max()), 1),
        }
        if ac_col in group.columns:
            row["mean_acertos"] = round(float(group[ac_col].astype(float).mean()), 1)
        else:
            row["mean_acertos"] = None

        _enrich_context(row, group, group_by)
        rows.append(row)

    # Sort by median desc (primary), then n desc as tiebreaker
    rows.sort(key=lambda r: (-(r["median"] or 0.0), -r["n"]))

    if effective_limit is not None and len(rows) > effective_limit:
        rows = rows[:effective_limit]
    return rows


def _key_column_for(group_by: GroupBy) -> str:
    if group_by == "uf":
        return "SG_UF_ESC"
    if group_by == "municipio":
        return "NO_MUNICIPIO_ESC"
    return "CO_ESCOLA"


def _enrich_context(row: dict[str, Any], group: pd.DataFrame, group_by: GroupBy) -> None:
    """Add uf/municipio/dependencia/localizacao columns when available."""
    # UF tag (always when the column exists)
    if "SG_UF_ESC" in group.columns:
        uf_vals = group["SG_UF_ESC"].dropna()
        if not uf_vals.empty:
            row["uf"] = str(uf_vals.iloc[0])

    # Município tag (only meaningful for escola; for municipio rows the key
    # IS the city, so we don't duplicate it)
    if group_by == "escola" and "NO_MUNICIPIO_ESC" in group.columns:
        muni_vals = group["NO_MUNICIPIO_ESC"].dropna()
        if not muni_vals.empty:
            row["municipio"] = str(muni_vals.iloc[0])

    # Dependência / localização: only makes sense at the school level
    # (city-level would be a mode, state-level irrelevant)
    if group_by == "escola":
        if "TP_DEPENDENCIA_ADM_ESC" in group.columns:
            dep_vals = group["TP_DEPENDENCIA_ADM_ESC"].dropna()
            if not dep_vals.empty:
                code = str(dep_vals.iloc[0])
                row["dependencia"] = _DEPENDENCIA_LABELS.get(code, code)
        if "TP_LOCALIZACAO_ESC" in group.columns:
            loc_vals = group["TP_LOCALIZACAO_ESC"].dropna()
            if not loc_vals.empty:
                code = str(loc_vals.iloc[0])
                row["localizacao"] = _LOCALIZACAO_LABELS.get(code, code)
