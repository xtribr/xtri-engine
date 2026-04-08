"""Tests for :mod:`app.microdata.dashboards` — Phase 8 P4."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.microdata.dashboards import (
    AREA_COLORS,
    AREA_LABELS_PT,
    compute_difficulty_table,
    render_area_dashboard,
    render_difficulty_table_png,
    render_full_report,
)
from app.microdata.schema import AREAS


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def calibrated_synthetic_df():
    """Build a 1000-row DataFrame matching the calibrate_all_areas output schema.

    Each student gets simulated V3 Local notas + reference notas with controlled
    correlation so the dashboards have realistic-looking data to render.
    """
    rng = np.random.default_rng(11)
    n = 1000
    data: dict = {}
    for area in AREAS:
        al = area.lower()
        # Simulate acertos uniformly 0..45
        acertos = rng.integers(0, 46, size=n)
        # Simulate ref nota with linear function of acertos + noise
        nota_ref = 200 + 13 * acertos + rng.normal(0, 25, size=n)
        nota_ref = np.clip(nota_ref, 200, 1000)
        # V3 nota = ref + small bias + noise → r ~ 0.95
        nota_v3 = nota_ref + rng.normal(-5, 30, size=n)
        nota_v3 = np.clip(nota_v3, 200, 1000)
        # Simulate theta as a z-score
        theta_v3 = (nota_v3 - nota_v3.mean()) / nota_v3.std()

        data[f"acertos_v3_{al}"] = acertos
        data[f"theta_v3_{al}"] = theta_v3
        data[f"se_v3_{al}"] = np.full(n, 0.3)
        data[f"nota_v3_{al}"] = nota_v3
        data[f"nota_ref_{al}"] = nota_ref
        data[f"estado_v3_{al}"] = np.where(nota_v3 >= 600, "PROFICIENTE", "BASICO")
        data[f"acert_dif_{al}"] = rng.uniform(0, 1, size=n)

    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# compute_difficulty_table
# ─────────────────────────────────────────────────────────────────────────────


class TestDifficultyTable:
    @pytest.mark.unit
    def test_returns_sorted_rows_with_required_keys(self, calibrated_synthetic_df):
        rows = compute_difficulty_table(calibrated_synthetic_df, "MT")
        assert len(rows) > 0
        # Sorted ascending by acertos
        acertos_seq = [r["acertos"] for r in rows]
        assert acertos_seq == sorted(acertos_seq)
        # Each row has all 5 keys
        for r in rows:
            assert set(r.keys()) == {"acertos", "freq", "tri_min", "tri_med", "tri_max"}
            assert r["freq"] >= 1
            assert r["tri_min"] <= r["tri_med"] <= r["tri_max"]

    @pytest.mark.unit
    def test_empty_when_no_v3_data(self):
        df = pd.DataFrame({"nota_v3_mt": [None, None], "acertos_v3_mt": [None, None]})
        assert compute_difficulty_table(df, "MT") == []

    @pytest.mark.unit
    def test_missing_columns_returns_empty(self):
        df = pd.DataFrame({"foo": [1, 2, 3]})
        assert compute_difficulty_table(df, "MT") == []


# ─────────────────────────────────────────────────────────────────────────────
# render_area_dashboard / render_full_report — PNG smoke tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRenderDashboards:
    @pytest.mark.unit
    @pytest.mark.parametrize("area", list(AREAS))
    def test_area_dashboard_returns_valid_png(self, calibrated_synthetic_df, area):
        png = render_area_dashboard(calibrated_synthetic_df, area)
        assert isinstance(png, bytes)
        assert png.startswith(PNG_MAGIC)
        assert len(png) > 5_000  # non-trivial image

    @pytest.mark.unit
    def test_unknown_area_raises(self, calibrated_synthetic_df):
        with pytest.raises(ValueError, match="Unknown area"):
            render_area_dashboard(calibrated_synthetic_df, "XX")

    @pytest.mark.unit
    def test_full_report_returns_valid_png(self, calibrated_synthetic_df):
        png = render_full_report(calibrated_synthetic_df)
        assert png.startswith(PNG_MAGIC)
        assert len(png) > 20_000  # 4 rows × 3 panels → bigger image

    @pytest.mark.unit
    def test_difficulty_table_png_returns_valid_png(self, calibrated_synthetic_df):
        png = render_difficulty_table_png(calibrated_synthetic_df, "MT")
        assert png.startswith(PNG_MAGIC)
        assert len(png) > 3_000

    @pytest.mark.unit
    def test_empty_dataframe_does_not_crash(self):
        empty = pd.DataFrame(
            {
                f"{prefix}_{area.lower()}": [None, None]
                for area in AREAS
                for prefix in ("nota_v3", "nota_ref", "acertos_v3")
            }
        )
        # Should still produce a PNG (with "sem dados" labels)
        png = render_area_dashboard(empty, "MT")
        assert png.startswith(PNG_MAGIC)


# ─────────────────────────────────────────────────────────────────────────────
# Color/label coverage
# ─────────────────────────────────────────────────────────────────────────────


class TestPalette:
    @pytest.mark.unit
    def test_all_areas_have_color_and_label(self):
        for area in AREAS:
            assert area in AREA_COLORS
            assert area in AREA_LABELS_PT
            assert AREA_COLORS[area].startswith("#")
            assert len(AREA_LABELS_PT[area]) > 0
