"""Tests for app.microdata.dashboard_data — Phase 8 P6 Plotly rewrite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.microdata.cache import cache_key
from app.microdata.dashboard_data import compute_dashboard_data
from app.microdata.loader import default_csv_path
from app.microdata.schema import AREAS


@pytest.fixture
def calibrated_synthetic_df():
    rng = np.random.default_rng(123)
    n = 1500
    data: dict = {"NU_SEQUENCIAL": [str(2_000_000 + i) for i in range(n)]}
    for area in AREAS:
        al = area.lower()
        ac = rng.integers(0, 46, size=n)
        ref = 200 + 13 * ac + rng.normal(0, 25, size=n)
        ref = np.clip(ref, 200, 1000)
        data[f"acertos_v3_{al}"] = ac
        data[f"nota_ref_{al}"] = ref
        data[f"acert_dif_{al}"] = rng.uniform(0, 1, size=n)
        # V3 columns also present so the calibration cache is realistic;
        # compute_dashboard_data must NOT use them.
        data[f"nota_v3_{al}"] = ref + rng.normal(0, 30, size=n)
        data[f"theta_v3_{al}"] = rng.normal(0, 1, size=n)
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# compute_dashboard_data
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeDashboardData:
    @pytest.mark.unit
    def test_returns_expected_keys(self, calibrated_synthetic_df):
        d = compute_dashboard_data(calibrated_synthetic_df, "MT")
        assert set(d.keys()) == {"area", "summary", "histogram", "scatter", "boxplot"}
        assert d["area"] == "MT"

    @pytest.mark.unit
    def test_summary_uses_nota_ref_not_v3(self, calibrated_synthetic_df):
        """Critical: must aggregate the OFFICIAL nota_ref, not nota_v3."""
        d = compute_dashboard_data(calibrated_synthetic_df, "MT")
        s = d["summary"]
        assert s["n"] == len(calibrated_synthetic_df)
        # Compute expected mean from nota_ref directly
        expected_mean = round(float(calibrated_synthetic_df["nota_ref_mt"].mean()), 1)
        assert abs(s["mean"] - expected_mean) < 0.5
        # Sanity: P25 < median < P75 < max
        assert s["min"] <= s["p25"] <= s["median"] <= s["p75"] <= s["max"]

    @pytest.mark.unit
    def test_histogram_shape(self, calibrated_synthetic_df):
        d = compute_dashboard_data(calibrated_synthetic_df, "MT")
        h = d["histogram"]
        assert len(h["bin_edges"]) == len(h["counts"]) + 1
        assert sum(h["counts"]) == len(calibrated_synthetic_df)

    @pytest.mark.unit
    def test_scatter_no_subsample_when_under_threshold(self, calibrated_synthetic_df):
        d = compute_dashboard_data(calibrated_synthetic_df, "MT")
        sc = d["scatter"]
        # 1500 < 30K → no subsample
        assert sc["n_displayed"] == sc["n_total"] == len(calibrated_synthetic_df)
        assert len(sc["acertos"]) == len(sc["nota"]) == len(sc["acert_dif"])
        # acert_dif values are in [0, 1]
        assert all(0 <= v <= 1 for v in sc["acert_dif"])

    @pytest.mark.unit
    def test_boxplot_sorted_and_complete(self, calibrated_synthetic_df):
        d = compute_dashboard_data(calibrated_synthetic_df, "MT")
        bp = d["boxplot"]
        assert len(bp) > 0
        acertos = [r["acertos"] for r in bp]
        assert acertos == sorted(acertos)
        # Each row has the 6 expected fields
        for r in bp:
            assert set(r.keys()) == {"acertos", "n", "min", "q1", "median", "q3", "max"}
            assert r["min"] <= r["q1"] <= r["median"] <= r["q3"] <= r["max"]

    @pytest.mark.unit
    def test_unknown_area_raises(self, calibrated_synthetic_df):
        with pytest.raises(ValueError, match="Unknown area"):
            compute_dashboard_data(calibrated_synthetic_df, "XX")

    @pytest.mark.unit
    def test_empty_when_nota_ref_missing(self):
        df = pd.DataFrame({"acertos_v3_mt": [1, 2, 3]})
        d = compute_dashboard_data(df, "MT")
        assert d["summary"]["n"] == 0
        assert d["histogram"]["counts"] == []


# ─────────────────────────────────────────────────────────────────────────────
# /api/v5/microdata/dashboard-data/{year} endpoint
# ─────────────────────────────────────────────────────────────────────────────


class TestDashboardDataEndpoint:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.mark.api
    def test_invalid_area_returns_400(self, client):
        r = client.get("/api/v5/microdata/dashboard-data/2024?area=XX")
        assert r.status_code == 400

    @pytest.mark.api
    def test_missing_csv_returns_404(self, client):
        r = client.get("/api/v5/microdata/dashboard-data/9999?area=MT")
        assert r.status_code == 404

    @pytest.mark.api
    def test_real_data(self, client):
        csv_path = default_csv_path(2024)
        if not csv_path.is_file():
            pytest.skip("MICRODADOS_ENEM_2024.csv not on disk")
        raw_key = cache_key(csv_path, year=2024, sample_size=200_000, rng_seed=42)
        if not raw_key.with_stage("calibrated").parquet_path.is_file():
            pytest.skip("calibrated cache not present")

        r = client.get("/api/v5/microdata/dashboard-data/2024?area=MT")
        assert r.status_code == 200
        d = r.json()
        assert d["area"] == "MT"
        assert d["year"] == 2024
        assert d["summary"]["n"] > 100_000
        assert d["summary"]["mean"] > 200
        assert d["summary"]["max"] > 800
        # Scatter is subsampled
        assert d["scatter"]["n_displayed"] <= 30_000
        assert d["scatter"]["n_total"] > 100_000
        # Boxplot has many acertos rows
        assert len(d["boxplot"]) > 30
