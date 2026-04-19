"""Tests for :mod:`app.microdata.report` — Phase 8 P9 Claude narrative."""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.microdata.report import (
    DEFAULT_MODEL,
    MAX_TOKENS,
    _pick_difficulty_highlights,
    build_payload,
    build_prompt,
    generate_report,
)
from app.microdata.schema import AREAS


@pytest.fixture
def calibrated_df_with_geo():
    """Calibrated DataFrame with realistic geo for prompt assembly."""
    rng = np.random.default_rng(2024)
    n = 1_200
    ufs = ["SP", "RJ", "MG", "BA", "CE", "PE", "RS", "PR"]
    cities = {
        "SP": ["São Paulo", "Campinas"],
        "RJ": ["Rio de Janeiro", "Niterói"],
        "MG": ["Belo Horizonte"],
        "BA": ["Salvador"],
        "CE": ["Fortaleza"],
        "PE": ["Recife"],
        "RS": ["Porto Alegre"],
        "PR": ["Curitiba"],
    }
    rows: list[dict[str, Any]] = []
    for i in range(n):
        uf = rng.choice(ufs)
        city = rng.choice(cities[uf])
        row: dict[str, Any] = {"NU_SEQUENCIAL": str(4_000_000 + i)}
        if rng.random() < 0.5:
            row["SG_UF_ESC"] = uf
            row["NO_MUNICIPIO_ESC"] = city
            row["CO_ESCOLA"] = str(35_000_000 + int(rng.integers(0, 100)))
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
# Payload + prompt assembly
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildPayload:
    @pytest.mark.unit
    def test_payload_has_all_sections(self, calibrated_df_with_geo):
        payload = build_payload(calibrated_df_with_geo, 2024, "MT")
        assert payload.year == 2024
        assert payload.area == "MT"
        assert payload.summary["n"] > 0
        assert payload.summary["median"] is not None
        assert len(payload.top_ufs) > 0
        assert len(payload.top_municipios) > 0
        assert len(payload.difficulty_highlights) > 0

    @pytest.mark.unit
    def test_payload_unknown_area_raises(self, calibrated_df_with_geo):
        with pytest.raises(ValueError, match="Unknown area"):
            build_payload(calibrated_df_with_geo, 2024, "XX")


class TestBuildPrompt:
    @pytest.mark.unit
    def test_prompt_contains_grounding_instruction(self, calibrated_df_with_geo):
        payload = build_payload(calibrated_df_with_geo, 2024, "MT")
        prompt = build_prompt(payload)
        assert "EXCLUSIVAMENTE" in prompt
        assert "não invente" in prompt.lower()

    @pytest.mark.unit
    def test_prompt_includes_summary_numbers(self, calibrated_df_with_geo):
        payload = build_payload(calibrated_df_with_geo, 2024, "MT")
        prompt = build_prompt(payload)
        # Summary numbers must appear verbatim so Claude can quote them
        assert str(payload.summary["median"]) in prompt
        assert str(payload.summary["mean"]) in prompt

    @pytest.mark.unit
    def test_prompt_includes_top_ufs(self, calibrated_df_with_geo):
        payload = build_payload(calibrated_df_with_geo, 2024, "MT")
        prompt = build_prompt(payload)
        # Every UF label must be mentioned
        for uf_row in payload.top_ufs:
            assert uf_row["label"] in prompt

    @pytest.mark.unit
    def test_prompt_includes_section_headers(self, calibrated_df_with_geo):
        payload = build_payload(calibrated_df_with_geo, 2024, "MT")
        prompt = build_prompt(payload)
        assert "Estatísticas Descritivas" in prompt
        assert "Tabela de Dificuldades" in prompt
        assert "Top 10 Estados" in prompt
        assert "Top 10 Municípios" in prompt
        assert "Top 10 Escolas" in prompt

    @pytest.mark.unit
    def test_pick_difficulty_highlights_returns_small_sample(self):
        rows = [
            {"acertos": i, "freq": 100, "tri_min": 200 + i, "tri_med": 210 + i, "tri_max": 220 + i}
            for i in range(46)
        ]
        picked = _pick_difficulty_highlights(rows, n_target=8)
        assert 4 <= len(picked) <= 12  # ~8 ± stride
        assert picked[0]["acertos"] == 0
        assert picked[-1]["acertos"] == 45

    @pytest.mark.unit
    def test_pick_difficulty_highlights_pass_through_small(self):
        rows = [{"acertos": i, "freq": 10, "tri_min": 1, "tri_med": 2, "tri_max": 3} for i in range(5)]
        assert _pick_difficulty_highlights(rows, n_target=8) == rows


# ─────────────────────────────────────────────────────────────────────────────
# generate_report — mocked Anthropic
# ─────────────────────────────────────────────────────────────────────────────


def _fake_anthropic_response(text: str):
    block = types.SimpleNamespace(text=text, type="text")
    usage = types.SimpleNamespace(input_tokens=1234, output_tokens=567)
    return types.SimpleNamespace(content=[block], usage=usage, id="fake")


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs: dict[str, Any] = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeAnthropic:
    def __init__(self, api_key: str, response):
        self.api_key = api_key
        self.messages = _FakeMessages(response)


class TestGenerateReport:
    @pytest.mark.unit
    def test_raises_runtime_error_without_api_key(self, calibrated_df_with_geo, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Also stub out dotenv loader path so we don't read a real .env
        with patch("app.microdata.report._load_api_key", return_value=None):
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                generate_report(calibrated_df_with_geo, 2024, "MT")

    @pytest.mark.unit
    def test_calls_anthropic_with_expected_args(
        self, calibrated_df_with_geo, monkeypatch
    ):
        fake = _fake_anthropic_response("# Resumo\n\nTexto do relatório.")
        instance_holder: dict[str, Any] = {}

        def _fake_ctor(api_key: str):
            instance = _FakeAnthropic(api_key, fake)
            instance_holder["last"] = instance
            return instance

        fake_module = types.SimpleNamespace(Anthropic=_fake_ctor)
        monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_module)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-xxx")

        result = generate_report(calibrated_df_with_geo, 2024, "MT")

        assert result["year"] == 2024
        assert result["area"] == "MT"
        assert result["model"] == DEFAULT_MODEL
        assert result["prompt_tokens"] == 1234
        assert result["output_tokens"] == 567
        assert "Resumo" in result["markdown"]

        # Verify the call used the default model + our prompt
        instance = instance_holder["last"]
        kwargs = instance.messages.last_kwargs
        assert kwargs["model"] == DEFAULT_MODEL
        assert kwargs["max_tokens"] == MAX_TOKENS
        assert len(kwargs["messages"]) == 1
        user_msg = kwargs["messages"][0]
        assert user_msg["role"] == "user"
        assert "ENEM 2024" in user_msg["content"]
        assert "MT" in user_msg["content"]

    @pytest.mark.unit
    def test_respects_model_override_env(self, calibrated_df_with_geo, monkeypatch):
        fake = _fake_anthropic_response("x")
        instance_holder: dict[str, Any] = {}

        def _fake_ctor(api_key: str):
            instance = _FakeAnthropic(api_key, fake)
            instance_holder["last"] = instance
            return instance

        fake_module = types.SimpleNamespace(Anthropic=_fake_ctor)
        monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_module)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("XTRI_REPORT_MODEL", "claude-haiku-4-5-20251001")

        result = generate_report(calibrated_df_with_geo, 2024, "MT")
        assert result["model"] == "claude-haiku-4-5-20251001"
        assert (
            instance_holder["last"].messages.last_kwargs["model"]
            == "claude-haiku-4-5-20251001"
        )


# ─────────────────────────────────────────────────────────────────────────────
# /api/v5/microdata/report/{year} endpoint
# ─────────────────────────────────────────────────────────────────────────────


class TestReportEndpoint:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.mark.api
    def test_invalid_area_returns_400(self, client):
        r = client.post("/api/v5/microdata/report/2024?area=XX")
        assert r.status_code == 400

    @pytest.mark.api
    def test_missing_csv_returns_404(self, client):
        r = client.post("/api/v5/microdata/report/9999?area=MT")
        assert r.status_code == 404

    @pytest.mark.api
    def test_missing_api_key_returns_501(self, client, monkeypatch):
        from app.microdata.cache import cache_key
        from app.microdata.loader import default_csv_path

        csv_path = default_csv_path(2024)
        if not csv_path.is_file():
            pytest.skip("MICRODADOS_ENEM_2024.csv not on disk")
        raw_key = cache_key(csv_path, year=2024, sample_size=200_000, rng_seed=42)
        if not raw_key.with_stage("calibrated").parquet_path.is_file():
            pytest.skip("calibrated cache not present")

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("app.microdata.report._load_api_key", return_value=None):
            r = client.post("/api/v5/microdata/report/2024?area=MT")
        assert r.status_code == 501
        assert "ANTHROPIC_API_KEY" in r.json()["detail"]
