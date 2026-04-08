"""Tests for :mod:`app.microdata.breakdown` — Phase 8 P7 drill-down."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.microdata.breakdown import compute_breakdown
from app.microdata.cache import cache_key
from app.microdata.loader import default_csv_path
from app.microdata.schema import AREAS


@pytest.fixture
def geo_synthetic_df():
    """Realistic DataFrame with geo columns for drill-down unit tests."""
    rng = np.random.default_rng(77)
    n = 1_000
    ufs = ["SP", "RJ", "MG", "BA", "CE", "PE", "PR"]
    cities = {
        "SP": ["São Paulo", "Campinas", "Santos"],
        "RJ": ["Rio de Janeiro", "Niterói"],
        "MG": ["Belo Horizonte", "Uberlândia"],
        "BA": ["Salvador", "Feira de Santana"],
        "CE": ["Fortaleza"],
        "PE": ["Recife"],
        "PR": ["Curitiba", "Londrina"],
    }
    rows: list[dict] = []
    for i in range(n):
        uf = rng.choice(ufs)
        city = rng.choice(cities[uf])
        # ~40% students have school info, ~60% have none (matches real INEP)
        has_school = rng.random() < 0.4
        row: dict = {"NU_SEQUENCIAL": str(3_000_000 + i)}
        if has_school:
            row["SG_UF_ESC"] = uf
            row["NO_MUNICIPIO_ESC"] = city
            row["CO_ESCOLA"] = str(35_000_000 + rng.integers(0, 200))
            row["TP_DEPENDENCIA_ADM_ESC"] = str(rng.choice([1, 2, 3, 4]))
            row["TP_LOCALIZACAO_ESC"] = str(rng.choice([1, 2]))
        else:
            row["SG_UF_ESC"] = None
            row["NO_MUNICIPIO_ESC"] = None
            row["CO_ESCOLA"] = None
            row["TP_DEPENDENCIA_ADM_ESC"] = None
            row["TP_LOCALIZACAO_ESC"] = None
        for area in AREAS:
            al = area.lower()
            ac = int(rng.integers(0, 46))
            ref = float(np.clip(200 + 13 * ac + rng.normal(0, 25), 200, 1000))
            row[f"acertos_v3_{al}"] = ac
            row[f"nota_ref_{al}"] = ref
            row[f"acert_dif_{al}"] = float(rng.uniform(0, 1))
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# compute_breakdown — unit tests
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeBreakdown:
    @pytest.mark.unit
    def test_uf_returns_expected_rows(self, geo_synthetic_df):
        rows = compute_breakdown(geo_synthetic_df, "MT", group_by="uf")
        assert len(rows) > 0
        keys = {r["key"] for r in rows}
        assert keys <= {"SP", "RJ", "MG", "BA", "CE", "PE", "PR"}
        # Sorted by median desc
        medians = [r["median"] for r in rows]
        assert medians == sorted(medians, reverse=True)

    @pytest.mark.unit
    def test_uf_row_shape(self, geo_synthetic_df):
        rows = compute_breakdown(geo_synthetic_df, "MT", group_by="uf")
        r0 = rows[0]
        assert set(r0.keys()) >= {
            "key",
            "label",
            "n",
            "mean",
            "median",
            "p25",
            "p75",
            "min",
            "max",
            "mean_acertos",
            "uf",
        }
        assert r0["min"] <= r0["p25"] <= r0["median"] <= r0["p75"] <= r0["max"]
        assert r0["n"] > 0

    @pytest.mark.unit
    def test_municipio_returns_rows_with_uf_context(self, geo_synthetic_df):
        rows = compute_breakdown(
            geo_synthetic_df, "LC", group_by="municipio", min_n=1
        )
        assert len(rows) > 0
        for r in rows:
            # Municipio rows carry UF for context
            assert "uf" in r
            assert r["uf"] in {"SP", "RJ", "MG", "BA", "CE", "PE", "PR"}

    @pytest.mark.unit
    def test_escola_includes_municipio_and_dependencia(self, geo_synthetic_df):
        rows = compute_breakdown(
            geo_synthetic_df, "CH", group_by="escola", min_n=1
        )
        assert len(rows) > 0
        # At least one school should carry municipio + dependencia + localizacao
        with_dep = [r for r in rows if "dependencia" in r]
        with_muni = [r for r in rows if "municipio" in r]
        with_loc = [r for r in rows if "localizacao" in r]
        assert with_dep, "expected dependencia in at least one school row"
        assert with_muni, "expected municipio in at least one school row"
        assert with_loc, "expected localizacao in at least one school row"
        # Dependência labels are humanized
        dep_values = {r["dependencia"] for r in with_dep}
        assert dep_values <= {"Federal", "Estadual", "Municipal", "Privada"}

    @pytest.mark.unit
    def test_uses_nota_ref_not_v3(self, geo_synthetic_df):
        """Drill-down must aggregate nota_ref_*, never nota_v3_*.

        We inject an extreme nota_v3_mt value for SP students and verify it
        does NOT leak into the stats.
        """
        df = geo_synthetic_df.copy()
        df["nota_v3_mt"] = 9_999.0  # obviously wrong value
        rows = compute_breakdown(df, "MT", group_by="uf")
        for r in rows:
            assert r["max"] < 1100, "nota_v3 leaked into stats"

    @pytest.mark.unit
    def test_min_n_filter_drops_small_groups(self, geo_synthetic_df):
        rows_loose = compute_breakdown(
            geo_synthetic_df, "MT", group_by="escola", min_n=1
        )
        rows_tight = compute_breakdown(
            geo_synthetic_df, "MT", group_by="escola", min_n=10
        )
        assert len(rows_tight) <= len(rows_loose)
        for r in rows_tight:
            assert r["n"] >= 10

    @pytest.mark.unit
    def test_limit_respected(self, geo_synthetic_df):
        rows = compute_breakdown(
            geo_synthetic_df, "MT", group_by="escola", min_n=1, limit=5
        )
        assert len(rows) <= 5

    @pytest.mark.unit
    def test_missing_geo_column_returns_empty(self):
        df = pd.DataFrame(
            {
                "NU_SEQUENCIAL": ["1", "2", "3"],
                "nota_ref_mt": [500.0, 600.0, 700.0],
                "acertos_v3_mt": [20, 25, 30],
            }
        )
        assert compute_breakdown(df, "MT", group_by="uf") == []
        assert compute_breakdown(df, "MT", group_by="municipio") == []
        assert compute_breakdown(df, "MT", group_by="escola") == []

    @pytest.mark.unit
    def test_unknown_area_raises(self, geo_synthetic_df):
        with pytest.raises(ValueError, match="Unknown area"):
            compute_breakdown(geo_synthetic_df, "XX", group_by="uf")

    @pytest.mark.unit
    def test_unknown_group_by_raises(self, geo_synthetic_df):
        with pytest.raises(ValueError, match="Unknown group_by"):
            compute_breakdown(geo_synthetic_df, "MT", group_by="foo")  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# /api/v5/microdata/breakdown/{year} endpoint
# ─────────────────────────────────────────────────────────────────────────────


class TestBreakdownEndpoint:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.mark.api
    def test_invalid_area_returns_400(self, client):
        r = client.get("/api/v5/microdata/breakdown/2024?area=XX&by=uf")
        assert r.status_code == 400

    @pytest.mark.api
    def test_invalid_by_returns_400(self, client):
        r = client.get("/api/v5/microdata/breakdown/2024?area=MT&by=planet")
        assert r.status_code == 400

    @pytest.mark.api
    def test_missing_csv_returns_404(self, client):
        r = client.get("/api/v5/microdata/breakdown/9999?area=MT&by=uf")
        assert r.status_code == 404

    @pytest.mark.api
    def test_real_data_uf(self, client):
        csv_path = default_csv_path(2024)
        if not csv_path.is_file():
            pytest.skip("MICRODADOS_ENEM_2024.csv not on disk")
        raw_key = cache_key(csv_path, year=2024, sample_size=200_000, rng_seed=42)
        if not raw_key.with_stage("calibrated").parquet_path.is_file():
            pytest.skip("calibrated cache not present")

        r = client.get("/api/v5/microdata/breakdown/2024?area=MT&by=uf")
        assert r.status_code == 200
        body = r.json()
        assert body["year"] == 2024
        assert body["area"] == "MT"
        assert body["by"] == "uf"
        assert body["count"] >= 20  # Brazil has 27 UFs, most should show up
        first = body["rows"][0]
        assert first["min"] <= first["median"] <= first["max"]
        # UF row carries its own key as label
        assert first["key"] == first["label"]

    @pytest.mark.api
    def test_real_data_escola_has_context(self, client):
        csv_path = default_csv_path(2024)
        if not csv_path.is_file():
            pytest.skip("MICRODADOS_ENEM_2024.csv not on disk")
        raw_key = cache_key(csv_path, year=2024, sample_size=200_000, rng_seed=42)
        if not raw_key.with_stage("calibrated").parquet_path.is_file():
            pytest.skip("calibrated cache not present")

        r = client.get(
            "/api/v5/microdata/breakdown/2024?area=MT&by=escola&limit=50"
        )
        assert r.status_code == 200
        body = r.json()
        assert body["count"] <= 50
        # At least one school carries full context
        contexts = [r for r in body["rows"] if "municipio" in r and "uf" in r]
        assert len(contexts) > 0
