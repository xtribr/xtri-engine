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
from app.storage.artifacts import save_unidim_artifact
from app.storage.model_registry import set_active_model

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

    @pytest.mark.api
    def test_job_lifecycle_reflects_status_transitions(self, client):
        """GET /job/{id} surfaces queued → running → done transitions."""
        job = job_manager.create(year=2024, sample_size=1000, rng_seed=7)

        # queued
        r = client.get(f"/api/v5/microdata/job/{job.job_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["job_id"] == job.job_id
        assert data["status"] == "queued"
        assert data["progress"] == 0.0
        assert data["started_at"] is None

        # running
        job_manager.mark_running(job.job_id)
        job_manager.update_progress(job.job_id, progress=0.42, stage="calibrating_LC")
        r = client.get(f"/api/v5/microdata/job/{job.job_id}")
        data = r.json()
        assert data["status"] == "running"
        assert data["progress"] == pytest.approx(0.42)
        assert data["stage"] == "calibrating_LC"
        assert data["started_at"] is not None
        assert data["elapsed_seconds"] is not None and data["elapsed_seconds"] >= 0

        # done
        from app.microdata.cache import CacheKey
        fake_key = CacheKey(
            year=2024,
            source_hash="abcd1234abcd1234",
            filters_hash="deadbeef",
            stage="calibrated",
        )
        job_manager.mark_done(job.job_id, result_key=fake_key)
        r = client.get(f"/api/v5/microdata/job/{job.job_id}")
        data = r.json()
        assert data["status"] == "done"
        assert data["progress"] == 1.0
        assert data["stage"] == "done"
        assert data["error"] is None

    @pytest.mark.api
    def test_job_error_path_surfaces_message(self, client):
        """mark_error surfaces the error message via the job endpoint."""
        job = job_manager.create(year=2024, sample_size=500, rng_seed=1)
        job_manager.mark_running(job.job_id)
        job_manager.mark_error(job.job_id, "CSV malformed at row 42")

        r = client.get(f"/api/v5/microdata/job/{job.job_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "error"
        assert data["error"] == "CSV malformed at row 42"

    @pytest.mark.api
    def test_prepare_envia_pipeline_para_dispatcher_dedicado(self, client, tmp_path):
        csv_path = tmp_path / "MICRODADOS_ENEM_2024.csv"
        csv_path.write_text("NU_INSCRICAO,TP_PRESENCA_CN\n1,1\n", encoding="utf-8")

        with mock.patch("app.main.default_csv_path", return_value=csv_path), mock.patch(
            "app.main.submit_microdata_job"
        ) as submit_mock:
            r = client.post(
                "/api/v5/microdata/prepare",
                json={"year": 2024, "sample_size": 1234, "rng_seed": 99},
            )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "queued"
        submit_mock.assert_called_once()


class TestRuntimeStatus:
    @pytest.mark.api
    def test_runtime_status_explica_fallback_sem_modelo(self, client, temp_model_storage):
        r = client.get("/api/v5/runtime/status?ano_referencia=2024&simulado_id=sim-sem-modelo")
        assert r.status_code == 200
        data = r.json()
        assert data["app_support_dir"] == str(temp_model_storage)
        assert data["directories"]["models"]["exists"] is True
        assert data["status_areas"]["MT"] == "fallback_heuristico"

    @pytest.mark.api
    def test_runtime_status_lista_modelo_e_job(self, client, temp_model_storage):
        manifest = save_unidim_artifact(
            simulado_id="sim-modelo",
            area="MT",
            ano_referencia=2024,
            c_fixo=0.20,
            parametros_itens=[{"a": 1.2, "b": 0.3, "c": 0.2, "taxa_acerto": 0.55, "saturado": False}],
            transformacao={"mu": 500.0, "sigma": 100.0, "metodo": "regressao_linear_lote"},
            diagnostico={"n_alunos": 100, "n_itens": 45},
            n_alunos=100,
            n_itens=45,
        )
        set_active_model(manifest)

        job = job_manager.create(year=2024, sample_size=1000, rng_seed=42)
        job_manager.mark_running(job.job_id)

        r = client.get("/api/v5/runtime/status?ano_referencia=2024&simulado_id=sim-modelo")
        assert r.status_code == 200
        data = r.json()

        assert data["job_counts"]["running"] >= 1
        assert any(model["artifact_id"] == manifest["artifact_id"] for model in data["models"])
        assert data["status_areas"]["MT"] == "pronto"
        assert data["directories"]["cache_microdata"]["path"].endswith("/cache/microdata")


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
