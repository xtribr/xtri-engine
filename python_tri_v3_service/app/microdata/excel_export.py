"""
Multi-sheet Excel export for the microdata validation report.

Generates a single .xlsx with:
    Sheet 1 — Resumo: per-area metrics + run metadata
    Sheet 2 — Relatorio Completo: full 4×3 PNG embedded
    Sheet 3-6 — One per area (LC, CH, CN, MT): area dashboard PNG +
                INEP-empirical difficulty table
    Sheet 7 — Alunos: full 200K-row per-student data with diff columns
                (NU_SEQUENCIAL, geo, V3 nota, REF nota, diff per area)

Uses openpyxl, which is already a bundled dependency. PNG images are produced
by ``app.microdata.dashboards`` and embedded via ``openpyxl.drawing.image.Image``.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from typing import Any

import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as OpenpyxlImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from app.microdata.dashboards import (
    AREA_COLORS,
    AREA_LABELS_PT,
    compute_difficulty_table,
    render_area_dashboard,
    render_full_report,
)
from app.microdata.schema import AREAS

logger = logging.getLogger(__name__)


# Header style helpers ----------------------------------------------------------

WHITE = Font(color="FFFFFF", bold=True)
HEADER_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=True)
CENTER = Alignment(horizontal="center", vertical="center")


def _hex_to_argb(hex_color: str) -> str:
    """Convert ``#9b59b6`` to ``FF9B59B6`` (openpyxl ARGB format)."""
    return "FF" + hex_color.lstrip("#").upper()


def _area_fill(area: str) -> PatternFill:
    return PatternFill(
        start_color=_hex_to_argb(AREA_COLORS[area]),
        end_color=_hex_to_argb(AREA_COLORS[area]),
        fill_type="solid",
    )


HEADER_FILL_DARK = PatternFill(start_color="FF1E3C72", end_color="FF1E3C72", fill_type="solid")
HEADER_FILL_LIGHT = PatternFill(start_color="FFE8F0FE", end_color="FFE8F0FE", fill_type="solid")


# ─────────────────────────────────────────────────────────────────────────────
# Sheet builders
# ─────────────────────────────────────────────────────────────────────────────


def _build_resumo_sheet(ws, year: int, df: pd.DataFrame, summaries: dict) -> None:
    """Top-level summary sheet: title, run metadata, INEP-only per-area stats.

    The microdata section is INEP-only (Phase 8 P6 design decision: V3 Local
    is reserved for the V3 Local CSV processing on the main page and the
    /validator V3-vs-V4 comparison; the microdata page uses the official
    NU_NOTA_* directly). So this sheet shows descriptive statistics of the
    OFFICIAL INEP scores, not V3-vs-INEP comparisons.
    """
    ws.title = "Resumo"

    # Title block
    ws["A1"] = "XTRI Engine — Análise de Microdados ENEM (INEP oficial)"
    ws["A1"].font = Font(bold=True, size=16, color="FF1E3C72")
    ws.merge_cells("A1:H1")
    ws["A1"].alignment = CENTER

    ws["A2"] = f"Ano ENEM: {year}    |    Amostra: {len(df):,} alunos    |    Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    ws["A2"].font = Font(italic=True, color="FF555555", size=10)
    ws.merge_cells("A2:H2")
    ws["A2"].alignment = CENTER

    # Per-area INEP descriptive stats (computed inline from nota_ref_*)
    headers = [
        "Área",
        "Nome",
        "N alunos",
        "Mean",
        "Median",
        "P25",
        "P75",
        "Min",
        "Max",
    ]
    for col, h in enumerate(headers, start=1):
        cell = ws.cell(row=4, column=col, value=h)
        cell.font = WHITE
        cell.fill = HEADER_FILL_DARK
        cell.alignment = HEADER_ALIGN

    for row_idx, area in enumerate(AREAS, start=5):
        al = area.lower()
        nota_col = f"nota_ref_{al}"
        if nota_col in df.columns:
            notas = df[nota_col].dropna()
            n = int(len(notas))
            mean_v = round(float(notas.mean()), 1) if n else None
            median_v = round(float(notas.median()), 1) if n else None
            p25 = round(float(notas.quantile(0.25)), 1) if n else None
            p75 = round(float(notas.quantile(0.75)), 1) if n else None
            min_v = round(float(notas.min()), 1) if n else None
            max_v = round(float(notas.max()), 1) if n else None
        else:
            n, mean_v, median_v, p25, p75, min_v, max_v = 0, None, None, None, None, None, None

        ws.cell(row=row_idx, column=1, value=area).fill = _area_fill(area)
        ws.cell(row=row_idx, column=1).font = WHITE
        ws.cell(row=row_idx, column=1).alignment = CENTER
        ws.cell(row=row_idx, column=2, value=AREA_LABELS_PT[area])
        ws.cell(row=row_idx, column=3, value=n)
        ws.cell(row=row_idx, column=4, value=mean_v)
        ws.cell(row=row_idx, column=5, value=median_v)
        ws.cell(row=row_idx, column=6, value=p25)
        ws.cell(row=row_idx, column=7, value=p75)
        ws.cell(row=row_idx, column=8, value=min_v)
        ws.cell(row=row_idx, column=9, value=max_v)

    # Footer note
    ws["A11"] = (
        "Estatísticas calculadas diretamente sobre NU_NOTA_* (oficial INEP) "
        "a partir da amostra de microdados. Sem referência a V3 Local."
    )
    ws.merge_cells("A11:I11")
    ws["A11"].font = Font(italic=True, color="FF777777", size=9)
    ws["A11"].alignment = Alignment(wrap_text=True)

    widths = {1: 8, 2: 22, 3: 12, 4: 11, 5: 11, 6: 10, 7: 10, 8: 10, 9: 10}
    for col_idx, w in widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = w


def _build_full_report_sheet(ws, year: int, df: pd.DataFrame) -> None:
    """Sheet with the embedded 4×3 full report PNG."""
    ws.title = "Relatorio Completo"
    ws["A1"] = f"Relatório Completo — ENEM {year}"
    ws["A1"].font = Font(bold=True, size=14, color="FF1E3C72")
    ws.merge_cells("A1:M1")
    ws["A1"].alignment = CENTER

    png_bytes = render_full_report(df)
    img = OpenpyxlImage(io.BytesIO(png_bytes))
    # Scale down for sheet display (~85% of native size)
    img.width = int(img.width * 0.85)
    img.height = int(img.height * 0.85)
    ws.add_image(img, "A3")

    # Pre-size columns/rows so the image renders cleanly
    for col_idx in range(1, 14):
        ws.column_dimensions[get_column_letter(col_idx)].width = 12


def _build_area_sheet(ws, area: str, df: pd.DataFrame) -> None:
    """One sheet per area: dashboard PNG + INEP empirical difficulty table."""
    ws.title = AREA_LABELS_PT[area][:31]  # Excel sheet names max 31 chars

    # Header
    ws["A1"] = f"{AREA_LABELS_PT[area]} ({area}) — V3 Local vs INEP"
    ws["A1"].font = Font(bold=True, size=14, color=_hex_to_argb(AREA_COLORS[area]).replace("FF", "", 1))
    ws.merge_cells("A1:H1")
    ws["A1"].alignment = CENTER

    # Embedded dashboard PNG
    png_bytes = render_area_dashboard(df, area)
    img = OpenpyxlImage(io.BytesIO(png_bytes))
    img.width = int(img.width * 0.85)
    img.height = int(img.height * 0.85)
    ws.add_image(img, "A3")

    # Difficulty table starts well below the image (PNG is ~440px tall ≈ 22 rows)
    table_start_row = 28
    ws.cell(row=table_start_row, column=1, value="Tabela de Dificuldades — INEP empírico (NU_NOTA real)").font = Font(
        bold=True, size=12, color="FF1E3C72"
    )
    ws.merge_cells(start_row=table_start_row, start_column=1, end_row=table_start_row, end_column=5)

    headers = ["Acertos", "Freq", "Min (INEP)", "Med (INEP)", "Max (INEP)"]
    for col_idx, h in enumerate(headers, start=1):
        cell = ws.cell(row=table_start_row + 1, column=col_idx, value=h)
        cell.font = WHITE
        cell.fill = _area_fill(area)
        cell.alignment = HEADER_ALIGN

    rows = compute_difficulty_table(df, area)
    for offset, row in enumerate(rows, start=2):
        r = table_start_row + offset
        ws.cell(row=r, column=1, value=row["acertos"]).alignment = CENTER
        ws.cell(row=r, column=2, value=row["freq"]).alignment = CENTER
        ws.cell(row=r, column=3, value=row["tri_min"]).alignment = CENTER
        ws.cell(row=r, column=4, value=row["tri_med"]).font = Font(bold=True)
        ws.cell(row=r, column=4).alignment = CENTER
        ws.cell(row=r, column=5, value=row["tri_max"]).alignment = CENTER

    # Freeze the table header
    ws.freeze_panes = ws.cell(row=table_start_row + 2, column=1)

    # Column widths
    for col_idx, w in [(1, 10), (2, 10), (3, 12), (4, 12), (5, 12)]:
        ws.column_dimensions[get_column_letter(col_idx)].width = w


def _build_alunos_sheet(ws, df: pd.DataFrame) -> None:
    """Per-student INEP-only data sheet (no V3 Local columns).

    Per Phase 8 P6 the microdata page is INEP-only, so the Alunos sheet
    matches: NU_SEQUENCIAL + geo + per area (acertos, nota_ref, acert_dif).
    """
    ws.title = "Alunos"

    # Build the columns we want to export
    base_cols = ["NU_SEQUENCIAL"]
    geo_cols = []
    for c in ("SG_UF_ESC", "NO_MUNICIPIO_ESC", "CO_ESCOLA"):
        if c in df.columns:
            geo_cols.append(c)

    headers: list[str] = base_cols + geo_cols
    for area in AREAS:
        al = area.lower()
        headers.extend([f"acertos_{al}", f"nota_ref_{al}", f"acert_dif_{al}"])

    # Write header row
    for col_idx, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=h)
        cell.font = WHITE
        cell.fill = HEADER_FILL_DARK
        cell.alignment = HEADER_ALIGN

    export_df = df[base_cols + geo_cols].copy()
    for area in AREAS:
        al = area.lower()
        export_df[f"acertos_{al}"] = df.get(f"acertos_v3_{al}")
        export_df[f"nota_ref_{al}"] = df.get(f"nota_ref_{al}")
        export_df[f"acert_dif_{al}"] = df.get(f"acert_dif_{al}")

    # Write data rows. ws.append() is ~5-10x faster than per-cell ws.cell()
    # because it skips coordinate lookups for each cell. Critical for 200K rows.
    # Pre-clean each tuple to plain Python types (openpyxl rejects numpy scalars).
    for row_tuple in export_df[headers].itertuples(index=False, name=None):
        cleaned = [
            None
            if (v is None or (isinstance(v, float) and pd.isna(v)))
            else (v.item() if hasattr(v, "item") else v)
            for v in row_tuple
        ]
        ws.append(cleaned)

    # Freeze the header row
    ws.freeze_panes = "A2"

    # Auto-size key columns (best effort, openpyxl doesn't have true autosize)
    width_map: dict[int, int] = {}
    for col_idx, h in enumerate(headers, start=1):
        if "NU_SEQUENCIAL" in h:
            width_map[col_idx] = 14
        elif "MUNICIPIO" in h:
            width_map[col_idx] = 24
        elif "ESCOLA" in h:
            width_map[col_idx] = 12
        elif "acert_dif" in h:
            width_map[col_idx] = 11
        else:
            width_map[col_idx] = 11
    for col_idx, w in width_map.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = w


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def build_microdata_excel(
    df: pd.DataFrame,
    year: int,
    summaries: dict,
    *,
    include_students: bool = False,
) -> bytes:
    """Build the multi-sheet Excel report and return raw .xlsx bytes.

    Args:
        df: Calibrated DataFrame from :func:`calibrate_all_areas`.
        year: ENEM year (for the title).
        summaries: Output of :func:`calibration_summary` keyed by lowercase area.
        include_students: If True, append a 200K-row per-student "Alunos" sheet.
            Defaults to False because openpyxl serialization of that many rows
            takes ~50s and produces a 30+ MB file. The lightweight Excel
            (without Alunos) builds in ~10s and ships in ~5 MB. Use the
            separate ``/api/v5/microdata/export-csv/{year}`` endpoint for the
            full per-student data — CSV streaming is ~10x faster than Excel
            for that volume.

    Returns:
        Raw .xlsx bytes ready to write to disk or stream over HTTP.
    """
    wb = Workbook()

    # Sheet 1: Resumo (replaces the default empty sheet)
    _build_resumo_sheet(wb.active, year=year, df=df, summaries=summaries)

    # Sheet 2: Full report
    _build_full_report_sheet(wb.create_sheet(), year=year, df=df)

    # Sheets 3-6: One per area
    for area in AREAS:
        _build_area_sheet(wb.create_sheet(), area=area, df=df)

    # Optional Sheet 7: Per-student data (heavy — opt in)
    if include_students:
        _build_alunos_sheet(wb.create_sheet(), df=df)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def build_microdata_csv(df: pd.DataFrame) -> bytes:
    """Build the per-student INEP-only CSV. ~5x faster than the old V3 export.

    Returns CSV bytes (UTF-8 with BOM for Excel compatibility) covering all
    students in the calibrated DataFrame, with one row per student and these
    columns:
        NU_SEQUENCIAL, [SG_UF_ESC,] [NO_MUNICIPIO_ESC,] [CO_ESCOLA,]
        per area: acertos, nota_ref, acert_dif

    V3 Local columns (theta_v3, nota_v3, diff, estado_v3) were intentionally
    dropped — the microdata page is INEP-only per the Phase 8 P6 design.
    Per area we keep:
      - acertos: number of items correct
      - nota_ref: official NU_NOTA from INEP
      - acert_dif: Primi metric (proportion of hard items correct)
    """
    base_cols = ["NU_SEQUENCIAL"]
    geo_cols = [
        c for c in ("SG_UF_ESC", "NO_MUNICIPIO_ESC", "CO_ESCOLA") if c in df.columns
    ]
    headers: list[str] = base_cols + geo_cols
    for area in AREAS:
        al = area.lower()
        headers.extend([f"acertos_{al}", f"nota_ref_{al}", f"acert_dif_{al}"])

    export_df = df[base_cols + geo_cols].copy()
    for area in AREAS:
        al = area.lower()
        export_df[f"acertos_{al}"] = df.get(f"acertos_v3_{al}")
        export_df[f"nota_ref_{al}"] = df.get(f"nota_ref_{al}")
        export_df[f"acert_dif_{al}"] = df.get(f"acert_dif_{al}")

    # pandas to_csv with float_format='%.1f' for nota columns (faster than
    # default repr). UTF-8 BOM lets Excel auto-detect encoding.
    csv_str = export_df[headers].to_csv(
        index=False, sep=";", float_format="%.1f", na_rep=""
    )
    return ("\ufeff" + csv_str).encode("utf-8")
