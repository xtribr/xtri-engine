"""
Compute the JSON payload for the interactive Plotly dashboards.

The microdata page is INEP-only (no V3 Local references). Per area, we serve:
- Summary stats (mean, median, P25, P75, min, max, n)
- Histogram (bin_edges + counts) of NU_NOTA
- Scatter (acertos, nota, acert_dif) for the colored gradient panel
- Boxplot data per acertos count (n, q1, median, q3, min, max)

The frontend feeds these arrays directly into Plotly.newPlot().
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.microdata.schema import AREAS

# Maximum scatter points to ship to the browser. With ~200K students per area,
# Plotly handles this fine but the JSON gets large. 30K balances density and
# payload size; same value used by the matplotlib renderer.
SCATTER_MAX = 30_000

# Number of histogram bins. Matches the matplotlib default density.
HIST_BINS = 40


def compute_dashboard_data(df: pd.DataFrame, area: str) -> dict[str, Any]:
    """Build the per-area JSON payload for the Plotly frontend.

    Args:
        df: Calibrated DataFrame from :func:`calibrate_all_areas` (we only use
            the ``nota_ref_*``, ``acertos_v3_*``, and ``acert_dif_*`` columns).
        area: ENEM area code (LC, CH, CN, MT).

    Returns:
        A nested dict with keys ``summary``, ``histogram``, ``scatter``,
        ``boxplot``. Empty (n=0) if no data for the area.
    """
    if area not in AREAS:
        raise ValueError(f"Unknown area: {area}")

    al = area.lower()
    nota_col = f"nota_ref_{al}"
    ac_col = f"acertos_v3_{al}"
    dif_col = f"acert_dif_{al}"

    if nota_col not in df.columns or ac_col not in df.columns:
        return _empty_payload(area)

    mask = df[nota_col].notna() & df[ac_col].notna()
    if mask.sum() == 0:
        return _empty_payload(area)

    sub = df.loc[mask, [nota_col, ac_col]].copy()
    sub[ac_col] = sub[ac_col].astype(int)
    if dif_col in df.columns:
        sub[dif_col] = df.loc[mask, dif_col].fillna(0.0)
    else:
        sub[dif_col] = 0.0

    nota = sub[nota_col].to_numpy(dtype=float)
    acertos = sub[ac_col].to_numpy(dtype=int)
    acert_dif = sub[dif_col].to_numpy(dtype=float)

    # ── Summary stats ──
    summary = {
        "n": int(len(nota)),
        "mean": round(float(nota.mean()), 1),
        "median": round(float(np.median(nota)), 1),
        "p25": round(float(np.percentile(nota, 25)), 1),
        "p75": round(float(np.percentile(nota, 75)), 1),
        "min": round(float(nota.min()), 1),
        "max": round(float(nota.max()), 1),
    }

    # ── Histogram ──
    hist_counts, hist_edges = np.histogram(nota, bins=HIST_BINS)
    histogram = {
        "bin_edges": [round(float(e), 1) for e in hist_edges],
        "counts": [int(c) for c in hist_counts],
    }

    # ── Scatter (subsampled if too dense) ──
    n_total = len(nota)
    if n_total > SCATTER_MAX:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, SCATTER_MAX, replace=False)
        scatter_acertos = acertos[idx]
        scatter_nota = nota[idx]
        scatter_acert_dif = acert_dif[idx]
    else:
        scatter_acertos = acertos
        scatter_nota = nota
        scatter_acert_dif = acert_dif

    scatter = {
        "acertos": [int(a) for a in scatter_acertos],
        "nota": [round(float(n), 1) for n in scatter_nota],
        "acert_dif": [round(float(d), 4) for d in scatter_acert_dif],
        "n_displayed": int(len(scatter_acertos)),
        "n_total": n_total,
    }

    # ── Boxplot per acertos count ──
    boxplot: list[dict[str, Any]] = []
    for ac, group in sub.groupby(ac_col):
        notas_group = group[nota_col].to_numpy()
        if len(notas_group) == 0:
            continue
        boxplot.append(
            {
                "acertos": int(ac),
                "n": int(len(notas_group)),
                "min": round(float(notas_group.min()), 1),
                "q1": round(float(np.percentile(notas_group, 25)), 1),
                "median": round(float(np.median(notas_group)), 1),
                "q3": round(float(np.percentile(notas_group, 75)), 1),
                "max": round(float(notas_group.max()), 1),
            }
        )
    boxplot.sort(key=lambda r: r["acertos"])

    return {
        "area": area,
        "summary": summary,
        "histogram": histogram,
        "scatter": scatter,
        "boxplot": boxplot,
    }


def _empty_payload(area: str) -> dict[str, Any]:
    return {
        "area": area,
        "summary": {"n": 0, "mean": None, "median": None, "p25": None, "p75": None, "min": None, "max": None},
        "histogram": {"bin_edges": [], "counts": []},
        "scatter": {"acertos": [], "nota": [], "acert_dif": [], "n_displayed": 0, "n_total": 0},
        "boxplot": [],
    }
