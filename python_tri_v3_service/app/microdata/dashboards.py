"""
Server-side matplotlib dashboards for microdata V3 Local validation.

Reproduces the user's reference layout from ``mente_mat_resultados.png``:
a per-area row of three panels — V3-vs-Reference scatter (with correlation),
distribution histogram (V3 vs reference, with mean lines), and Acertos-vs-Nota
scatter — plus a difficulty table function used by the frontend.

All chart functions return PNG bytes (raw, not base64). Callers in Phase 5
endpoints will base64-encode them for inline `<img src="data:image/png;...">`
embedding in the frontend.

CRITICAL: ``matplotlib.use('Agg')`` is set at module-import time before any
pyplot import. The FastAPI worker thread has no GUI backend (no Cocoa, no Tk),
and an interactive backend would crash with "main thread is not in main loop".
"""

from __future__ import annotations

import io
import logging
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # MUST come before pyplot import — no GUI backend in workers

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.microdata.schema import AREAS  # noqa: E402

logger = logging.getLogger(__name__)


# Per-area color palette matching the user's reference image (mente_mat_resultados.png).
# CN = blue (sky), MT = orange (warm), CH = green (forest), LC = purple (royal).
AREA_COLORS: dict[str, str] = {
    "LC": "#9b59b6",  # purple
    "CH": "#27ae60",  # green
    "CN": "#3498db",  # blue
    "MT": "#e67e22",  # orange
}

AREA_LABELS_PT: dict[str, str] = {
    "LC": "Linguagens",
    "CH": "Ciências Humanas",
    "CN": "Ciências da Natureza",
    "MT": "Matemática",
}


# ─────────────────────────────────────────────────────────────────────────────
# Difficulty table (cohort-empirical, mirrors frontend renderRefTable for microdata)
# ─────────────────────────────────────────────────────────────────────────────


def compute_difficulty_table(df: pd.DataFrame, area: str) -> list[dict[str, Any]]:
    """Per-acertos empirical Min/Med/Max table from the OFFICIAL INEP scores.

    Mirrors the structure of the user's reference PDF tables (Acertos / Freq /
    Min / Med / Max). Each row aggregates the official ``NU_NOTA_{area}``
    values (already present in the microdata, computed by INEP using the real
    item parameters) over all students in the sample with that acertos count.

    Why ``nota_ref_*`` and not ``nota_v3_*``: V3 Local uses cohort-anchored
    z-score scoring, which compresses the within-band spread to near-zero for
    most acertos counts. INEP uses item-level parameters that DIFFERENTIATE
    students with the same acertos but different response patterns (Primi
    "differential scoring"), producing a real Min<Med<Max range. The user
    explicitly wants this spread visible (matching the PDF tables they shared).

    Only acertos counts present in the cohort are included — no phantom rows.

    Args:
        df: Calibrated DataFrame from :func:`calibrate_all_areas`.
        area: ENEM area code (LC, CH, CN, MT).

    Returns:
        List of dicts ``[{"acertos": int, "freq": int, "tri_min": float,
        "tri_med": float, "tri_max": float}, ...]`` sorted by acertos ascending.
        Empty if neither the V3 acertos column nor the reference nota column
        is present.
    """
    al = area.lower()
    nota_col = f"nota_ref_{al}"  # OFFICIAL INEP score from the microdata
    ac_col = f"acertos_v3_{al}"  # V3 acertos count (same as official, computed in calibration)
    if nota_col not in df.columns or ac_col not in df.columns:
        return []

    mask = df[nota_col].notna() & df[ac_col].notna()
    if mask.sum() == 0:
        return []

    sub = df.loc[mask, [ac_col, nota_col]].copy()
    sub[ac_col] = sub[ac_col].astype(int)

    grouped = sub.groupby(ac_col)[nota_col].agg(["count", "min", "median", "max"])
    rows: list[dict[str, Any]] = []
    for ac, row in grouped.iterrows():
        rows.append(
            {
                "acertos": int(ac),
                "freq": int(row["count"]),
                "tri_min": round(float(row["min"]), 1),
                "tri_med": round(float(row["median"]), 1),
                "tri_max": round(float(row["max"]), 1),
            }
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Per-panel chart helpers
# ─────────────────────────────────────────────────────────────────────────────


def _v3_vs_ref_scatter(ax, df: pd.DataFrame, area: str, color: str) -> None:
    """Left panel: V3 Local nota vs official reference nota with diagonal."""
    al = area.lower()
    v3 = df[f"nota_v3_{al}"]
    ref = df[f"nota_ref_{al}"]
    mask = v3.notna() & ref.notna()
    v3m = v3[mask].to_numpy()
    refm = ref[mask].to_numpy()

    if len(v3m) == 0:
        ax.set_title(f"{area} — V3 vs Referência (sem dados)", fontsize=11)
        ax.set_xlabel("Nota Ref CSV")
        ax.set_ylabel("Nota V3 (Científico)")
        return

    # Identity line
    lo = float(min(v3m.min(), refm.min()))
    hi = float(max(v3m.max(), refm.max()))
    ax.plot([lo, hi], [lo, hi], color="#7f8c8d", linestyle="--", linewidth=1, label="Identidade")

    # Subsample if very large to keep PNG size + render time sane
    if len(v3m) > 30000:
        idx = np.random.default_rng(42).choice(len(v3m), 30000, replace=False)
        v3m_plot = v3m[idx]
        refm_plot = refm[idx]
    else:
        v3m_plot = v3m
        refm_plot = refm

    ax.scatter(refm_plot, v3m_plot, s=4, alpha=0.35, color=color, edgecolors="none")

    r = float(np.corrcoef(v3m, refm)[0, 1]) if len(v3m) > 1 else float("nan")
    ax.set_title(f"{area} — V3 vs Referência (r={r:.3f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Nota Ref CSV")
    ax.set_ylabel("Nota V3 (Científico)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)


def _distribution_histogram(ax, df: pd.DataFrame, area: str, color: str) -> None:
    """Center panel: overlaid V3 vs reference histograms with mean lines."""
    al = area.lower()
    v3 = df[f"nota_v3_{al}"].dropna().to_numpy()
    ref = df[f"nota_ref_{al}"].dropna().to_numpy()

    if len(v3) == 0 and len(ref) == 0:
        ax.set_title(f"{area} — Distribuição de Notas (sem dados)", fontsize=11)
        return

    # Common bin range so the two histograms align
    all_notas = np.concatenate([v3, ref]) if len(v3) and len(ref) else (v3 if len(v3) else ref)
    bins = np.linspace(all_notas.min(), all_notas.max(), 40)

    if len(ref) > 0:
        ax.hist(ref, bins=bins, alpha=0.55, color="#95a5a6", label="Nota Ref CSV", edgecolor="white", linewidth=0.3)
    if len(v3) > 0:
        ax.hist(v3, bins=bins, alpha=0.55, color=color, label="V3 Científico", edgecolor="white", linewidth=0.3)

    if len(v3):
        v3_mean = float(v3.mean())
        ax.axvline(v3_mean, color=color, linestyle="--", linewidth=1.5, label=f"Média V3: {v3_mean:.0f}")
    if len(ref):
        ref_mean = float(ref.mean())
        ax.axvline(ref_mean, color="#34495e", linestyle="--", linewidth=1.5, label=f"Média Ref: {ref_mean:.0f}")

    ax.set_title(f"{area} — Distribuição de Notas", fontsize=11, fontweight="bold")
    ax.set_xlabel("Nota TRI")
    ax.set_ylabel("Frequência")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)


def _acertos_vs_nota_scatter(ax, df: pd.DataFrame, area: str, color: str) -> None:
    """Right panel: acertos × nota scatter, V3 + reference overlaid."""
    al = area.lower()
    v3 = df[f"nota_v3_{al}"]
    ref = df[f"nota_ref_{al}"]
    ac = df[f"acertos_v3_{al}"]

    mask = v3.notna() & ref.notna() & ac.notna()
    if mask.sum() == 0:
        ax.set_title(f"{area} — Acertos vs Nota (sem dados)", fontsize=11)
        return

    acm = ac[mask].astype(int).to_numpy()
    v3m = v3[mask].to_numpy()
    refm = ref[mask].to_numpy()

    # Subsample for plot density
    if len(acm) > 30000:
        idx = np.random.default_rng(7).choice(len(acm), 30000, replace=False)
        acm_p, v3m_p, refm_p = acm[idx], v3m[idx], refm[idx]
    else:
        acm_p, v3m_p, refm_p = acm, v3m, refm

    ax.scatter(acm_p, refm_p, s=4, alpha=0.20, color="#7f8c8d", label="Nota Ref CSV", edgecolors="none")
    ax.scatter(acm_p, v3m_p, s=4, alpha=0.35, color=color, label="V3 Científico", edgecolors="none")

    ax.set_title(f"{area} — Acertos vs Nota", fontsize=11, fontweight="bold")
    ax.set_xlabel("Acertos")
    ax.set_ylabel("Nota TRI")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Public dashboard renderers
# ─────────────────────────────────────────────────────────────────────────────


def render_area_dashboard(df: pd.DataFrame, area: str, *, dpi: int = 110) -> bytes:
    """Render a 1×3 panel dashboard for a single area.

    Args:
        df: Calibrated DataFrame from :func:`calibrate_all_areas`.
        area: ENEM area code (LC, CH, CN, MT).
        dpi: Output DPI. Default 110 = ~1320×440 pixel image.

    Returns:
        Raw PNG bytes.
    """
    if area not in AREAS:
        raise ValueError(f"Unknown area: {area}")
    color = AREA_COLORS[area]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    _v3_vs_ref_scatter(axes[0], df, area, color)
    _distribution_histogram(axes[1], df, area, color)
    _acertos_vs_nota_scatter(axes[2], df, area, color)

    al = area.lower()
    n = int(df[f"nota_v3_{al}"].notna().sum())
    fig.suptitle(
        f"{AREA_LABELS_PT[area]} ({area}) — V3 Local vs INEP   |   N = {n:,} alunos",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_full_report(
    df: pd.DataFrame,
    *,
    dpi: int = 110,
    title: str = "MICRODADOS ENEM — V3 Local (Motor Científico) vs INEP Oficial",
) -> bytes:
    """Render a 4×3 panel report with all areas stacked vertically.

    This matches the layout of ``mente_mat_resultados.png`` but with all
    four areas (the reference image had only CN + MT). One row per area.

    Args:
        df: Calibrated DataFrame from :func:`calibrate_all_areas`.
        dpi: Output DPI. Default 110 = ~1320×1760 pixel image.
        title: Top title for the figure.

    Returns:
        Raw PNG bytes.
    """
    fig, axes = plt.subplots(4, 3, figsize=(13, 14))

    for row, area in enumerate(AREAS):
        color = AREA_COLORS[area]
        _v3_vs_ref_scatter(axes[row, 0], df, area, color)
        _distribution_histogram(axes[row, 1], df, area, color)
        _acertos_vs_nota_scatter(axes[row, 2], df, area, color)

    # Top title with sample size summary
    n_lc = int(df["nota_v3_lc"].notna().sum()) if "nota_v3_lc" in df.columns else 0
    n_total = len(df)
    fig.suptitle(
        f"{title}\nN sample = {n_total:,}   |   V3 calibrado: LC {n_lc:,} (similar para CH/CN/MT)",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_difficulty_table_png(df: pd.DataFrame, area: str, *, dpi: int = 110) -> bytes:
    """Render the cohort-empirical difficulty table as a PNG (text-only).

    Useful for embedding alongside the chart panels in a single export.
    Three columns mirroring the PDF layout.
    """
    rows = compute_difficulty_table(df, area)
    if not rows:
        fig = plt.figure(figsize=(10, 1.5))
        plt.text(0.5, 0.5, f"{area}: sem dados", ha="center", va="center", fontsize=12)
        plt.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    color = AREA_COLORS[area]
    n_rows = len(rows)
    cols_per_table = max(1, (n_rows + 2) // 3)
    chunks = [rows[i : i + cols_per_table] for i in range(0, n_rows, cols_per_table)]

    fig, axes = plt.subplots(1, len(chunks), figsize=(4 * len(chunks), max(3, n_rows / 3)))
    if len(chunks) == 1:
        axes = [axes]
    headers = ["Ac", "Freq", "Min", "Med", "Max"]

    for ax, chunk in zip(axes, chunks):
        ax.axis("off")
        cell_text = [[r["acertos"], r["freq"], r["tri_min"], r["tri_med"], r["tri_max"]] for r in chunk]
        table = ax.table(
            cellText=cell_text,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            colColours=[color] * len(headers),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        # White text on the colored header row
        for j in range(len(headers)):
            cell = table[0, j]
            cell.set_text_props(color="white", fontweight="bold")

    fig.suptitle(
        f"{AREA_LABELS_PT[area]} ({area}) — Tabela de Dificuldades (INEP empírico)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
