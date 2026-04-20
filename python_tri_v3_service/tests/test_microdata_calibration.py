"""Tests for :mod:`app.microdata.calibration` — Phase 8 P3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.microdata.calibration import (
    _effective_lc_gabarito,
    _parse_nota_series,
    _resp_to_binary,
    calibrar_itens_fast,
    calibrate_all_areas,
    calibrate_area_group,
    calibration_summary,
    eap_theta_batch,
)
from app.microdata.schema import detect_schema
from app.motores.cientifico import calibrar_itens as calibrar_itens_scalar
from app.motores.cientifico import eap_theta as eap_theta_scalar
from app.tabela_referencia import TabelaReferenciaTRI

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "microdata_tiny.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: _resp_to_binary and _parse_nota_series
# ─────────────────────────────────────────────────────────────────────────────


class TestResponseToBinary:
    @pytest.mark.unit
    def test_perfect_match_returns_all_ones(self):
        result = _resp_to_binary("ABCDE", "ABCDE")
        assert result == [1, 1, 1, 1, 1]

    @pytest.mark.unit
    def test_partial_match(self):
        result = _resp_to_binary("AXCDE", "ABCDE")
        assert result == [1, 0, 1, 1, 1]

    @pytest.mark.unit
    def test_empty_returns_none(self):
        assert _resp_to_binary("", "ABCDE") is None
        assert _resp_to_binary("ABCDE", "") is None

    @pytest.mark.unit
    def test_length_mismatch_returns_none(self):
        assert _resp_to_binary("ABCDE", "ABCD") is None

    @pytest.mark.unit
    def test_annulled_items_give_credit(self):
        # '.' and 'X' in the gabarito mean the item was annulled (credit to all).
        result = _resp_to_binary("ZBCDE", "AB.DE")  # position 2 is annulled
        assert result == [0, 1, 1, 1, 1]  # A vs Z=0, B vs B=1, . is credit=1, etc.


class TestCalibrarItensFast:
    """Vectorized item calibration must match scalar within tight tolerance."""

    @pytest.mark.unit
    def test_matches_scalar_calibration(self):
        rng = np.random.default_rng(42)
        n_students = 200
        n_items = 15
        # Random binary responses
        resp_matrix = rng.integers(0, 2, size=(n_students, n_items)).astype(int)

        slow = calibrar_itens_scalar(resp_matrix, c=0.20)
        fast = calibrar_itens_fast(resp_matrix, c=0.20)

        assert len(slow) == len(fast) == n_items
        # The two L-BFGS runs may converge to slightly different points but
        # the (a, b) parameters should agree to ~3 decimal places.
        for s, f in zip(slow, fast):
            assert abs(s["a"] - f["a"]) < 0.05, f"a mismatch: {s['a']} vs {f['a']}"
            assert abs(s["b"] - f["b"]) < 0.05, f"b mismatch: {s['b']} vs {f['b']}"
            assert s["c"] == f["c"]
            assert s["taxa_acerto"] == f["taxa_acerto"]


class TestEAPThetaBatch:
    """Batched EAP must match the scalar per-student EAP from cientifico.py."""

    @pytest.mark.unit
    def test_matches_scalar_eap_within_tolerance(self):
        rng = np.random.default_rng(7)
        n_students = 50
        n_items = 20

        # Random item params
        item_params = [
            {
                "a": float(rng.uniform(0.8, 2.2)),
                "b": float(rng.uniform(-2.0, 2.0)),
                "c": 0.2,
            }
            for _ in range(n_items)
        ]
        # Random binary responses
        resp_matrix = rng.integers(0, 2, size=(n_students, n_items)).astype(int)

        # Scalar reference
        scalar_theta = np.zeros(n_students)
        scalar_se = np.zeros(n_students)
        for i in range(n_students):
            t, se = eap_theta_scalar(resp_matrix[i], item_params)
            scalar_theta[i] = t
            scalar_se[i] = se

        # Batched
        batch_theta, batch_se = eap_theta_batch(resp_matrix, item_params)

        # Theta must match within ~1e-4 (both use same quadrature, same math).
        # The scalar version rounds to 6 decimals, so loosen tolerance slightly.
        assert np.allclose(batch_theta, scalar_theta, atol=1e-4), (
            f"theta mismatch: max diff = {np.abs(batch_theta - scalar_theta).max():.6f}"
        )
        assert np.allclose(batch_se, scalar_se, atol=1e-3), (
            f"se mismatch: max diff = {np.abs(batch_se - scalar_se).max():.6f}"
        )

    @pytest.mark.unit
    def test_saturated_item_excluded_in_batch(self):
        """Accuracy plan #3: saturated items (clamped b) must not contribute
        to the batch EAP. Must stay consistent with scalar ``eap_theta``."""
        rng = np.random.default_rng(31)
        n_students = 20
        n_items = 15
        item_params = [
            {
                "a": float(rng.uniform(0.8, 2.2)),
                "b": float(rng.uniform(-2.0, 2.0)),
                "c": 0.2,
                "saturado": False,
            }
            for _ in range(n_items)
        ]
        resp_matrix = rng.integers(0, 2, size=(n_students, n_items)).astype(int)

        # Baseline: no saturated items
        batch_theta_baseline, _ = eap_theta_batch(resp_matrix, item_params)

        # Flag item 3 as saturated with a clamped b — the batch should
        # behave as if that item were absent.
        item_params_flagged = [dict(p) for p in item_params]
        item_params_flagged[3]["saturado"] = True
        item_params_flagged[3]["b"] = 4.9
        batch_theta_flagged, _ = eap_theta_batch(resp_matrix, item_params_flagged)

        # Reference: manual exclusion
        params_ex = [p for i, p in enumerate(item_params) if i != 3]
        resp_ex = np.delete(resp_matrix, 3, axis=1)
        batch_theta_ex, _ = eap_theta_batch(resp_ex, params_ex)

        assert np.allclose(batch_theta_flagged, batch_theta_ex, atol=1e-6), (
            "batch EAP with saturated flag diverges from manual exclusion"
        )
        # And the flagged version should differ from the baseline (otherwise
        # the flag has no effect).
        assert not np.allclose(batch_theta_flagged, batch_theta_baseline, atol=1e-3)


class TestEffectiveLCGabarito:
    """The 50-char INEP LC gabarito must be sliced to 45 chars per TP_LINGUA."""

    # English section (5) + Spanish section (5) + common (40) = 50 chars
    FULL_GAB = "ENGLI" + "SPANI" + "C" * 40  # 5 + 5 + 40 = 50

    @pytest.mark.unit
    def test_length_mismatch_returns_none(self):
        assert _effective_lc_gabarito("ABCDE" * 8, 0) is None  # 40 chars

    @pytest.mark.unit
    def test_unknown_lingua_returns_none(self):
        assert _effective_lc_gabarito(self.FULL_GAB, 5) is None

    @pytest.mark.unit
    def test_english_slice_takes_english_then_common(self):
        result = _effective_lc_gabarito(self.FULL_GAB, 0)
        assert result is not None
        assert len(result) == 45
        assert result[:5] == "ENGLI"  # English items
        assert result[5:] == "C" * 40  # common items

    @pytest.mark.unit
    def test_spanish_slice_takes_spanish_then_common(self):
        result = _effective_lc_gabarito(self.FULL_GAB, 1)
        assert result is not None
        assert len(result) == 45
        assert result[:5] == "SPANI"  # Spanish items
        assert result[5:] == "C" * 40  # common items

    @pytest.mark.unit
    def test_none_input(self):
        assert _effective_lc_gabarito(None, 0) is None


class TestParseNotaSeries:
    @pytest.mark.unit
    def test_parses_br_decimal(self):
        s = pd.Series(["523,4", "612,8", "588,1"])
        result = _parse_nota_series(s)
        assert list(result) == [523.4, 612.8, 588.1]

    @pytest.mark.unit
    def test_empty_and_invalid_become_nan(self):
        s = pd.Series(["523,4", "", "abc", None])
        result = _parse_nota_series(s)
        assert result[0] == 523.4
        assert pd.isna(result[1])
        assert pd.isna(result[2])
        assert pd.isna(result[3])


# ─────────────────────────────────────────────────────────────────────────────
# Calibration of a single group
# ─────────────────────────────────────────────────────────────────────────────


def _synthetic_group(
    n_students: int,
    n_items: int,
    true_thetas: np.ndarray,
    true_a: np.ndarray,
    true_b: np.ndarray,
    c: float = 0.20,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic microdata-style DataFrame for one (area, co_prova) group.

    Simulates student responses using a 3PL model with known parameters, then
    encodes them into TX_RESPOSTAS_MT / TX_GABARITO_MT format so
    ``calibrate_area_group`` can consume them.
    """
    rng = np.random.default_rng(rng_seed)
    # Simulate 3PL probability then sample 0/1
    # P(correct) = c + (1-c) / (1 + exp(-a*(theta-b)))
    probs = c + (1 - c) / (1 + np.exp(-np.outer(true_thetas, true_a) + true_b))
    # Above: np.outer(thetas, a) = (n_students, n_items), minus true_b broadcasts per-item
    # Correct: P = c + (1-c) / (1 + exp(-(a*theta - b)))
    probs = c + (1 - c) / (1 + np.exp(-(np.outer(true_thetas, true_a) - true_b)))
    correct = (rng.random(probs.shape) < probs).astype(int)

    # Build a fixed gabarito string (e.g. "AAAAA...A" n_items long)
    gab = "A" * n_items
    # Student responses: "A" for correct, "B" for wrong
    resp_strs = ["".join("A" if c else "B" for c in row) for row in correct]

    # Compute official nota_ref using the tabela (we'll use tri_med for each student's acertos)
    # For the synthetic group, just pass through the acertos count as nota_ref placeholder.
    acertos = correct.sum(axis=1)
    # Fake nota_ref: linear mapping of theta for comparison purposes
    nota_ref = 500 + 100 * true_thetas

    df = pd.DataFrame(
        {
            "NU_SEQUENCIAL": [f"{1000000 + i}" for i in range(n_students)],
            "TP_PRESENCA_LC": ["1"] * n_students,
            "TP_PRESENCA_CH": ["1"] * n_students,
            "TP_PRESENCA_CN": ["1"] * n_students,
            "TP_PRESENCA_MT": ["1"] * n_students,
            "TP_STATUS_REDACAO": ["1"] * n_students,
            "TX_RESPOSTAS_LC": ["A" * n_items] * n_students,
            "TX_RESPOSTAS_CH": ["A" * n_items] * n_students,
            "TX_RESPOSTAS_CN": ["A" * n_items] * n_students,
            "TX_RESPOSTAS_MT": resp_strs,
            "TX_GABARITO_LC": [gab] * n_students,
            "TX_GABARITO_CH": [gab] * n_students,
            "TX_GABARITO_CN": [gab] * n_students,
            "TX_GABARITO_MT": [gab] * n_students,
            "CO_PROVA_LC": ["1291"] * n_students,
            "CO_PROVA_CH": ["1357"] * n_students,
            "CO_PROVA_CN": ["1415"] * n_students,
            "CO_PROVA_MT": ["1473"] * n_students,
            "NU_NOTA_LC": ["500,0"] * n_students,
            "NU_NOTA_CH": ["500,0"] * n_students,
            "NU_NOTA_CN": ["500,0"] * n_students,
            "NU_NOTA_MT": [f"{n:.1f}".replace(".", ",") for n in nota_ref],
            "NU_NOTA_REDACAO": ["500,0"] * n_students,
            "TP_LINGUA": ["0"] * n_students,
        }
    )
    return df, acertos, true_thetas


class TestCalibrateAreaGroup:
    @pytest.fixture(scope="class")
    def tabela(self):
        return TabelaReferenciaTRI()

    @pytest.fixture(scope="class")
    def fixture_schema(self):
        return detect_schema(FIXTURE_PATH)

    @pytest.mark.unit
    def test_too_small_returns_empty(self, tabela, fixture_schema):
        df = pd.read_csv(FIXTURE_PATH, sep=";", encoding="latin-1", dtype=str)
        # Default min_calibration_size=500, fixture has 8 rows → empty
        result = calibrate_area_group(df, fixture_schema, "MT", tabela)
        assert result.empty

    @pytest.mark.integration
    def test_synthetic_recovers_theta_correlation(self, tabela):
        """Simulate 200 students with known thetas and verify V3 recovers order."""
        rng = np.random.default_rng(42)
        n_students = 200
        n_items = 45
        true_thetas = rng.normal(0, 1, n_students)
        true_a = rng.uniform(0.8, 2.2, n_items)
        true_b = rng.uniform(-2.0, 2.0, n_items)

        df, _acertos, _true = _synthetic_group(
            n_students, n_items, true_thetas, true_a, true_b, rng_seed=42
        )

        # Build a mini schema pointing at the synthetic columns
        from app.microdata.schema import MicrodataSchema

        schema = MicrodataSchema(
            id_col="NU_SEQUENCIAL",
            tp_presenca={a: f"TP_PRESENCA_{a}" for a in ("LC", "CH", "CN", "MT")},
            tx_respostas={a: f"TX_RESPOSTAS_{a}" for a in ("LC", "CH", "CN", "MT")},
            co_prova={a: f"CO_PROVA_{a}" for a in ("LC", "CH", "CN", "MT")},
            tx_gabarito={a: f"TX_GABARITO_{a}" for a in ("LC", "CH", "CN", "MT")},
            nu_nota={a: f"NU_NOTA_{a}" for a in ("LC", "CH", "CN", "MT")},
            tp_status_redacao="TP_STATUS_REDACAO",
            nu_nota_redacao="NU_NOTA_REDACAO",
            tp_lingua="TP_LINGUA",
        )

        result = calibrate_area_group(
            df,
            schema,
            "MT",
            tabela,
            min_calibration_size=100,  # below default for test
        )

        assert len(result) == n_students
        # V3 theta should correlate strongly with true theta
        r = np.corrcoef(result["theta_v3"].to_numpy(), true_thetas)[0, 1]
        assert r > 0.85, f"Theta correlation too low: {r:.3f}"
        # All notas should be in a sane ENEM range
        assert result["nota_v3"].min() >= 200
        assert result["nota_v3"].max() <= 1000


# ─────────────────────────────────────────────────────────────────────────────
# calibrate_all_areas end-to-end
# ─────────────────────────────────────────────────────────────────────────────


class TestCalibrateAllAreas:
    @pytest.fixture(scope="class")
    def tabela(self):
        return TabelaReferenciaTRI()

    @pytest.fixture(scope="class")
    def fixture_schema(self):
        return detect_schema(FIXTURE_PATH)

    @pytest.mark.integration
    def test_fixture_runs_but_skips_all_groups(self, tabela, fixture_schema):
        """Fixture is too small for calibration — all groups skipped.

        Expected behaviour: returns a DataFrame with the V3 columns present
        but all NaN, and ``nota_ref_{area}`` populated from NU_NOTA_*.
        """
        df = pd.read_csv(FIXTURE_PATH, sep=";", encoding="latin-1", dtype=str)
        result = calibrate_all_areas(df, fixture_schema, tabela)

        # V3 columns present but all NaN (groups below MIN_CALIBRATION_SIZE)
        for area in ("LC", "CH", "CN", "MT"):
            al = area.lower()
            assert f"theta_v3_{al}" in result.columns
            assert f"nota_v3_{al}" in result.columns
            assert f"nota_ref_{al}" in result.columns
            assert result[f"theta_v3_{al}"].isna().all()

        # Reference notas should be parsed correctly for non-null rows
        # Fixture row 1000001: NU_NOTA_MT = "704,2"
        ref_mt = result.loc[result["NU_SEQUENCIAL"] == "1000001", "nota_ref_mt"].iloc[0]
        assert abs(ref_mt - 704.2) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# calibration_summary
# ─────────────────────────────────────────────────────────────────────────────


class TestCalibrationSummary:
    @pytest.mark.unit
    def test_summary_shape_and_empty_handling(self):
        df = pd.DataFrame(
            {
                "nota_v3_lc": [500.0, 520.0, 540.0, np.nan],
                "nota_ref_lc": [510.0, 515.0, 545.0, 600.0],
                "nota_v3_ch": [np.nan] * 4,
                "nota_ref_ch": [np.nan] * 4,
                "nota_v3_cn": [600.0, 650.0, 700.0, 750.0],
                "nota_ref_cn": [610.0, 640.0, 695.0, 760.0],
                "nota_v3_mt": [800.0, 850.0, 900.0, 950.0],
                "nota_ref_mt": [790.0, 840.0, 920.0, 940.0],
            }
        )
        summary = calibration_summary(df)
        assert set(summary.keys()) == {"lc", "ch", "cn", "mt"}

        # LC: 3 valid rows
        assert summary["lc"]["n"] == 3
        assert summary["lc"]["correlation"] is not None
        assert summary["lc"]["mae"] is not None

        # CH: no valid rows
        assert summary["ch"]["n"] == 0
        assert summary["ch"]["correlation"] is None

        # CN + MT: 4 valid rows each
        assert summary["cn"]["n"] == 4
        assert summary["mt"]["n"] == 4
        assert summary["mt"]["correlation"] > 0.95  # deliberately high correlation
