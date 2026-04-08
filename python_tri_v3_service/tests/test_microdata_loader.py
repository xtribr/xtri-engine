"""Tests for :mod:`app.microdata` — Phase 7 P1/P2."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from app.microdata import detect_schema, load_microdata
from app.microdata.cache import (
    CACHE_ROOT,
    cache_key,
    cache_size_bytes,
    clear_year_cache,
    load_cached,
    save_cached,
)
from app.microdata.filters import is_primeira_aplicacao, is_valid_student
from app.microdata.schema import AREAS

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "microdata_tiny.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Schema detection
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaDetection:
    @pytest.mark.unit
    def test_detects_nu_sequencial_as_id(self):
        schema = detect_schema(FIXTURE_PATH)
        assert schema.id_col == "NU_SEQUENCIAL"

    @pytest.mark.unit
    def test_all_areas_have_presence_columns(self):
        schema = detect_schema(FIXTURE_PATH)
        for area in AREAS:
            assert schema.tp_presenca[area] == f"TP_PRESENCA_{area}"
            assert schema.tx_respostas[area] == f"TX_RESPOSTAS_{area}"
            assert schema.co_prova[area] == f"CO_PROVA_{area}"
            assert schema.nu_nota[area] == f"NU_NOTA_{area}"

    @pytest.mark.unit
    def test_detects_optional_demographic_columns(self):
        schema = detect_schema(FIXTURE_PATH)
        # INEP removed IN_TREINEIRO and NO_ENTIDADE from ENEM 2024+ microdata,
        # so the schema no longer tracks them. School names are resolved via
        # the Supabase inep_escolas table in Phase 7.
        assert schema.co_uf_esc == "CO_UF_ESC"
        assert schema.sg_uf_esc == "SG_UF_ESC"
        assert schema.no_municipio_esc == "NO_MUNICIPIO_ESC"
        assert schema.co_escola == "CO_ESCOLA"

    @pytest.mark.unit
    def test_all_needed_columns_includes_optional_when_present(self):
        schema = detect_schema(FIXTURE_PATH)
        cols = schema.all_needed_columns()
        assert "NU_SEQUENCIAL" in cols
        assert "CO_UF_ESC" in cols
        assert "CO_ESCOLA" in cols
        assert "TP_STATUS_REDACAO" in cols
        # ID + redação fields + lingua + 4 areas × 5 columns (presence,
        # respostas, co_prova, nota, gabarito) + 7 optional demographic = ~28
        assert len(cols) >= 25

    @pytest.mark.unit
    def test_missing_csv_raises(self):
        with pytest.raises(FileNotFoundError):
            detect_schema("/nonexistent/microdados.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Filter predicates
# ─────────────────────────────────────────────────────────────────────────────


class TestFilters:
    @pytest.fixture
    def df_and_schema(self):
        schema = detect_schema(FIXTURE_PATH)
        df = pd.read_csv(
            FIXTURE_PATH,
            sep=";",
            encoding="latin-1",
            dtype=str,
        )
        return df, schema

    @pytest.mark.unit
    def test_is_valid_student_drops_absent_and_invalid_redacao(self, df_and_schema):
        df, schema = df_and_schema
        mask = is_valid_student(df, schema)
        # Fixture: 8 rows. 1000003 (absent LC) and 1000004 (no redação) fail.
        assert mask.sum() == 6
        # Valid IDs — check against the NU_SEQUENCIAL column.
        valid_ids = set(df.loc[mask, "NU_SEQUENCIAL"].astype(int))
        assert valid_ids == {1000001, 1000002, 1000005, 1000006, 1000007, 1000008}

    @pytest.mark.unit
    def test_primeira_aplicacao_none_passes_all_rows(self, df_and_schema, caplog):
        df, schema = df_and_schema
        with caplog.at_level("WARNING"):
            mask = is_primeira_aplicacao(df, schema, provas_primeira=None)
        # Unfiltered mode returns all-True and logs a warning
        assert mask.sum() == len(df)
        assert "no primeira-aplicação filter applied" in caplog.text

    @pytest.mark.unit
    def test_primeira_aplicacao_strict_drops_reaplicacao(self, df_and_schema):
        df, schema = df_and_schema
        whitelist = {1291, 1357, 1415, 1473}
        mask = is_primeira_aplicacao(df, schema, provas_primeira=whitelist)
        # Strict mode: drops 1000006 (uses co_prova 9991-9994 outside whitelist).
        dropped = df.loc[~mask, "NU_SEQUENCIAL"].astype(int).tolist()
        assert 1000006 in dropped

    @pytest.mark.unit
    def test_combined_valid_plus_unfiltered_gives_6(self, df_and_schema):
        df, schema = df_and_schema
        mask = is_valid_student(df, schema) & is_primeira_aplicacao(
            df, schema, provas_primeira=None
        )
        # valid_student keeps {1,2,5,6,7,8}; unfiltered passes all; result = {1,2,5,6,7,8}
        assert mask.sum() == 6
        kept = set(df.loc[mask, "NU_SEQUENCIAL"].astype(int))
        assert kept == {1000001, 1000002, 1000005, 1000006, 1000007, 1000008}

    @pytest.mark.unit
    def test_combined_valid_plus_strict_gives_5(self, df_and_schema):
        df, schema = df_and_schema
        whitelist = {1291, 1357, 1415, 1473}
        mask = is_valid_student(df, schema) & is_primeira_aplicacao(
            df, schema, provas_primeira=whitelist
        )
        # valid_student keeps {1,2,5,6,7,8}; strict drops 6; result = {1,2,5,7,8}
        assert mask.sum() == 5
        kept = set(df.loc[mask, "NU_SEQUENCIAL"].astype(int))
        assert kept == {1000001, 1000002, 1000005, 1000007, 1000008}


# ─────────────────────────────────────────────────────────────────────────────
# Loader integration
# ─────────────────────────────────────────────────────────────────────────────


class TestLoader:
    @pytest.mark.integration
    def test_loads_filters_and_samples(self):
        df = load_microdata(
            year=2024,
            csv_path=FIXTURE_PATH,
            sample_size=None,  # keep all valid rows
            provas_primeira=None,  # unfiltered (no primeira-aplicação filter)
        )
        # valid-student filter alone keeps 6 students from the fixture.
        assert len(df) == 6
        # Sample order is randomized — check as a set.
        ids = set(df["NU_SEQUENCIAL"].astype(int))
        assert ids == {1000001, 1000002, 1000005, 1000006, 1000007, 1000008}

    @pytest.mark.integration
    def test_strict_mode_drops_reaplicacao(self):
        df = load_microdata(
            year=2024,
            csv_path=FIXTURE_PATH,
            sample_size=None,
            provas_primeira={1291, 1357, 1415, 1473},
        )
        assert len(df) == 5
        ids = set(df["NU_SEQUENCIAL"].astype(int))
        assert 1000006 not in ids  # reaplicação dropped (co_prova 9991-9994 not in whitelist)
        assert 1000005 in ids  # kept (its co_prova IS in the whitelist)

    @pytest.mark.integration
    def test_sampling_reproducibility(self):
        df1 = load_microdata(year=2024, csv_path=FIXTURE_PATH, sample_size=3, rng_seed=42)
        df2 = load_microdata(year=2024, csv_path=FIXTURE_PATH, sample_size=3, rng_seed=42)
        assert list(df1["NU_SEQUENCIAL"]) == list(df2["NU_SEQUENCIAL"])

    @pytest.mark.integration
    def test_missing_file_raises_with_helpful_message(self):
        with pytest.raises(FileNotFoundError, match="Download MICRODADOS_ENEM"):
            load_microdata(year=9999, csv_path="/nonexistent/file.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Cache layer
# ─────────────────────────────────────────────────────────────────────────────


class TestCache:
    @pytest.fixture
    def temp_cache_root(self, tmp_path):
        """Patch CACHE_ROOT to a temp directory so tests don't pollute the user's real cache."""
        temp_root = tmp_path / "microdata_cache"
        with mock.patch("app.microdata.cache.CACHE_ROOT", temp_root):
            yield temp_root

    @pytest.mark.unit
    def test_cache_key_is_stable_for_same_inputs(self, temp_cache_root):
        k1 = cache_key(FIXTURE_PATH, year=2024, sample_size=100, rng_seed=42)
        k2 = cache_key(FIXTURE_PATH, year=2024, sample_size=100, rng_seed=42)
        assert k1 == k2
        assert k1.year == 2024

    @pytest.mark.unit
    def test_cache_key_changes_with_sample_size(self, temp_cache_root):
        k1 = cache_key(FIXTURE_PATH, year=2024, sample_size=100, rng_seed=42)
        k2 = cache_key(FIXTURE_PATH, year=2024, sample_size=200, rng_seed=42)
        assert k1.filters_hash != k2.filters_hash

    @pytest.mark.unit
    def test_cache_key_changes_with_provas_whitelist(self, temp_cache_root):
        k1 = cache_key(FIXTURE_PATH, year=2024, sample_size=100, rng_seed=42, provas_primeira=None)
        k2 = cache_key(
            FIXTURE_PATH,
            year=2024,
            sample_size=100,
            rng_seed=42,
            provas_primeira=frozenset({1291, 1357, 1415, 1473}),
        )
        assert k1.filters_hash != k2.filters_hash

    @pytest.mark.integration
    def test_save_and_load_round_trip(self, temp_cache_root):
        df = load_microdata(
            year=2024,
            csv_path=FIXTURE_PATH,
            sample_size=None,
            provas_primeira=None,
        )
        key = cache_key(FIXTURE_PATH, year=2024, sample_size=None, rng_seed=42)
        assert save_cached(df, key, source_path=FIXTURE_PATH, filters_description="test")
        assert key.parquet_path.is_file()
        assert key.meta_path.is_file()
        # Round-trip
        df_back = load_cached(key)
        assert df_back is not None
        assert len(df_back) == len(df)
        assert set(df_back["NU_SEQUENCIAL"]) == set(df["NU_SEQUENCIAL"])

    @pytest.mark.unit
    def test_load_cached_miss_returns_none(self, temp_cache_root):
        key = cache_key(FIXTURE_PATH, year=2024, sample_size=999999, rng_seed=777)
        assert load_cached(key) is None

    @pytest.mark.integration
    def test_clear_year_cache_removes_files(self, temp_cache_root):
        df = load_microdata(year=2024, csv_path=FIXTURE_PATH, sample_size=None)
        key = cache_key(FIXTURE_PATH, year=2024, sample_size=None, rng_seed=42)
        save_cached(df, key, source_path=FIXTURE_PATH)
        assert cache_size_bytes() > 0
        count = clear_year_cache(2024)
        assert count >= 2  # parquet + meta.json
