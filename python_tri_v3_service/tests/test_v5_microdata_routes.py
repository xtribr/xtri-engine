"""Tests for /api/v5/microdata/* routes — Phase 8 P5."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.microdata.cache import CACHE_ROOT, cache_key
from app.microdata.jobs import job_manager
from app.microdata.loader import default_csv_path

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_jobs():
    """Clear the in-memory job registry between tests."""
    job_manager.clear()
    yield
    job_manager.clear()


# ─────────────────────────────────────────────────────────────────────────────
# anos-disponiveis
# ─────────────────────────────────────────────────────────────────────────────


class TestAvailability:
    @pytest.mark.api
    def test_returns_list_and_dir(self, client):
        r = client.get("/api/v5/microdata/anos-disponiveis")
        assert r.status_code == 200
        data = r.json()
        assert "anos" in data
        assert "microdata_dir" in data
        assert isinstance(data["anos"], list)
        assert all(isinstance(y, int) for y in data["anos"])


# ─────────────────────────────────────────────────────────────────────────────
# status/{year}
# ─────────────────────────────────────────────────────────────────────────────


class TestStatus:
    @pytest.mark.api
    def test_status_for_unknown_year_is_200_with_csv_exists_false(self, client):
        # Use a far-future year that won't exist on disk
        r = client.get("/api/v5/microdata/status/9999")
        assert r.status_code == 200
        data = r.json()
        assert data["year"] == 9999
        assert data["csv_exists"] is False
        assert data["cache_raw_present"] is False
        assert data["cache_calibrated_present"] is False

    @pytest.mark.api
    def test_status_for_2024_real_csv(self, client):
        # 2024 CSV is on disk per the smoke tests; assert basic shape
        r = client.get("/api/v5/microdata/status/2024")
        assert r.status_code == 200
        data = r.json()
        assert data["year"] == 2024
        # csv_exists may be True (real file present) or False (test environment)
        # — both are acceptable, just check the field types
        assert isinstance(data["csv_exists"], bool)
        if data["csv_exists"]:
            assert data["csv_size_mb"] is not None
            assert data["csv_size_mb"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# prepare + job polling
# ─────────────────────────────────────────────────────────────────────────────


class TestPrepareAndJob:
    @pytest.mark.api
    def test_prepare_for_missing_csv_returns_404(self, client):
        r = client.post(
            "/api/v5/microdata/prepare",
            json={"year": 9999, "sample_size": 100},
        )
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()

    @pytest.mark.api
    def test_unknown_job_id_returns_404(self, client):
        r = client.get("/api/v5/microdata/job/nonexistent-id-12345")
        assert r.status_code == 404
        assert "Unknown job_id" in r.json()["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# dashboard / full-report
# ─────────────────────────────────────────────────────────────────────────────


class TestDashboard:
    @pytest.mark.api
    def test_dashboard_invalid_area_returns_400(self, client):
        r = client.get("/api/v5/microdata/dashboard/2024?area=XX")
        assert r.status_code == 400
        assert "Invalid area" in r.json()["detail"]

    @pytest.mark.api
    def test_dashboard_for_missing_csv_returns_404(self, client):
        r = client.get("/api/v5/microdata/dashboard/9999?area=MT")
        assert r.status_code == 404

    @pytest.mark.api
    def test_full_report_for_missing_csv_returns_404(self, client):
        r = client.get("/api/v5/microdata/full-report/9999")
        assert r.status_code == 404

    @pytest.mark.api
    def test_dashboard_with_calibrated_cache(self, client, tmp_path):
        """If the calibrated cache is on disk for 2024, the dashboard endpoint
        returns a valid base64 PNG. This is a real-data smoke test that only
        runs if the user has executed Phase 3 against MICRODADOS_ENEM_2024.csv.
        """
        csv_path = default_csv_path(2024)
        if not csv_path.is_file():
            pytest.skip("MICRODADOS_ENEM_2024.csv not on disk")
        raw_key = cache_key(csv_path, year=2024, sample_size=200_000, rng_seed=42)
        calib_path = raw_key.with_stage("calibrated").parquet_path
        if not calib_path.is_file():
            pytest.skip("calibrated cache not present (run smoke_calibration first)")

        r = client.get("/api/v5/microdata/dashboard/2024?area=MT")
        assert r.status_code == 200
        data = r.json()
        assert data["year"] == 2024
        assert data["area"] == "MT"
        assert data["n_students"] > 100_000
        assert data["summary"]["correlation"] is not None
        assert data["summary"]["correlation"] > 0.85
        # PNG base64 must decode and start with the PNG magic
        png_bytes = base64.b64decode(data["dashboard_png_base64"])
        assert png_bytes.startswith(PNG_MAGIC)
        assert len(png_bytes) > 5_000
        # Difficulty table is non-empty
        assert len(data["difficulty_table"]) > 0
        first_row = data["difficulty_table"][0]
        assert "acertos" in first_row
        assert "tri_min" in first_row
        assert "tri_med" in first_row
        assert "tri_max" in first_row

    @pytest.mark.api
    def test_full_report_with_calibrated_cache(self, client):
        csv_path = default_csv_path(2024)
        if not csv_path.is_file():
            pytest.skip("MICRODADOS_ENEM_2024.csv not on disk")
        raw_key = cache_key(csv_path, year=2024, sample_size=200_000, rng_seed=42)
        calib_path = raw_key.with_stage("calibrated").parquet_path
        if not calib_path.is_file():
            pytest.skip("calibrated cache not present")

        r = client.get("/api/v5/microdata/full-report/2024")
        assert r.status_code == 200
        data = r.json()
        assert data["year"] == 2024
        assert data["n_students"] == 200_000
        # All four area summaries present
        assert set(data["summaries"].keys()) == {"lc", "ch", "cn", "mt"}
        for area in ("lc", "ch", "cn", "mt"):
            s = data["summaries"][area]
            assert s["n"] > 100_000
            assert s["correlation"] > 0.85
        # Full-report PNG is bigger than per-area
        png_bytes = base64.b64decode(data["full_report_png_base64"])
        assert png_bytes.startswith(PNG_MAGIC)
        assert len(png_bytes) > 100_000
