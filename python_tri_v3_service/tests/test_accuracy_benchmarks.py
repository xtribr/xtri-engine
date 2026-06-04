"""Accuracy-regression benchmarks for the ENEM IRT engine.

Goal: pin MAE / correlation floors so future refactors of the calibration or
scoring layers can't silently degrade the engine without breaking CI. Two
layers:

1. **Synthetic ground truth** (fast, deterministic, no external data) —
   generates a cohort with known θ and known item params, runs the full
   pipeline, and asserts that the recovered nota correlates tightly with the
   ideal nota computed from the true θ. Catches regressions in the math core.

2. **Real INEP benchmark** (slow, skip-if-missing) — loads the calibrated
   2024 microdata parquet produced by the Phase 8 pipeline and asserts that
   the V3 Local nota stays within published MAE / correlation bands vs the
   official ``NU_NOTA_*`` columns. Pins the April-2026 smoke-test baseline
   from ``task_plan.md`` (LC r=0.9495/MAE=43.5, CH r=0.9711/MAE=43.7,
   CN r=0.9300/MAE=35.3, MT r=0.9489/MAE=37.1). The floors are deliberately
   conservative — they catch ~5-10% regressions without flagging
   single-digit drift from the noisy real-world numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.irt.transform import ancorar_nota_individual
from app.motores.cientifico import pipeline_cientifico
from app.tabela_referencia import TabelaReferenciaTRI


@pytest.fixture(scope="module")
def tabela_multiano():
    t = TabelaReferenciaTRI()
    t.validar()
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic benchmark (always runs — no external data)
# ─────────────────────────────────────────────────────────────────────────────


def _synthetic_responses(
    n_students: int,
    n_itens: int,
    *,
    seed: int,
    c: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate (responses, true_thetas, true_a, true_b) from the 3PL model."""
    rng = np.random.default_rng(seed)
    true_thetas = rng.normal(0, 1, n_students)
    true_a = rng.uniform(0.8, 2.2, n_itens)
    true_b = rng.uniform(-2.0, 2.0, n_itens)
    probs = c + (1 - c) / (
        1 + np.exp(-true_a[None, :] * (true_thetas[:, None] - true_b[None, :]))
    )
    resp = (rng.uniform(size=probs.shape) < probs).astype(int)
    # Reject perfectly constant rows (they'd be filtered from calibration
    # anyway); we want the benchmark to exercise the informative path.
    keep = (resp.sum(axis=1) > 0) & (resp.sum(axis=1) < n_itens)
    return resp[keep], true_thetas[keep], true_a, true_b


class TestSyntheticAccuracy:
    """Guards the IRT pipeline against regressions at the math-core level.

    The test generates a synthetic cohort where the "ground-truth nota" is the
    reference-table anchor of the *true* θ, then runs the full calibration +
    scoring pipeline on the noisy response matrix. Correlation and MAE floors
    are set comfortably above typical sampling noise so transient flakes are
    unlikely, but still tight enough that the drops caused by plan items
    #1-#3 would be detectable.
    """

    @pytest.mark.integration
    def test_mt_500_students_recovers_nota(self, tabela_multiano):
        n_students = 500
        n_itens = 45
        resp, true_thetas, _, _ = _synthetic_responses(
            n_students, n_itens, seed=2024
        )

        # Ideal nota per student: anchor the *true* θ via the same table the
        # engine would use. This removes the scale contribution from the
        # benchmark and measures ONLY the θ recovery error.
        true_acertos = resp.sum(axis=1).astype(int)
        ideal_notas = np.array([
            ancorar_nota_individual(
                theta=float(true_thetas[i]),
                area="MT",
                acertos=int(true_acertos[i]),
                tabela=tabela_multiano,
            )
            for i in range(resp.shape[0])
        ])

        alunos = [
            {"aluno_id": f"S{i:04d}", "nome": "", "respostas": resp[i].tolist()}
            for i in range(resp.shape[0])
        ]
        resultado = pipeline_cientifico(alunos=alunos, area="MT", tabela=tabela_multiano)
        v3_notas = np.array([r["nota_ancorada"] for r in resultado["resultados_alunos"]])

        # Correlation: should be very high — the only noise is 3PL sampling.
        r = float(np.corrcoef(v3_notas, ideal_notas)[0, 1])
        mae = float(np.mean(np.abs(v3_notas - ideal_notas)))
        rmse = float(np.sqrt(np.mean((v3_notas - ideal_notas) ** 2)))

        # Floors tightened after the review pointed out that r>0.90/MAE<45
        # gave ~2x headroom. With post-#1/#2/#3 actuals of r≈0.98/MAE≈20,
        # these tighter floors still have ~30-40% margin over observed noise.
        assert r > 0.95, f"MT correlation regressed: r={r:.4f} < 0.95"
        assert mae < 30.0, f"MT MAE regressed: {mae:.1f} >= 30.0"
        assert rmse < 45.0, f"MT RMSE regressed: {rmse:.1f} >= 45.0"

    # Hard-coded per-area seeds — Python's built-in hash() is randomised per
    # process (PYTHONHASHSEED), so `hash(area) & 0xFFFF` would give different
    # seeds on every CI run. Fixed values make the test fully reproducible.
    _AREA_SEED = {"LC": 1001, "CH": 1002, "CN": 1003, "MT": 1004}

    @pytest.mark.integration
    @pytest.mark.parametrize("area", ["LC", "CH", "CN", "MT"])
    def test_all_areas_min_correlation(self, tabela_multiano, area):
        """Each of the 4 ENEM areas must recover θ with r > 0.85 minimum."""
        n_students = 300
        n_itens = 44 if area == "LC" else 45  # LC goes 0..44 by ENEM convention
        resp, true_thetas, _, _ = _synthetic_responses(
            n_students, n_itens, seed=self._AREA_SEED[area],
        )

        alunos = [
            {"aluno_id": f"{area}{i:04d}", "nome": "", "respostas": resp[i].tolist()}
            for i in range(resp.shape[0])
        ]
        resultado = pipeline_cientifico(alunos=alunos, area=area, tabela=tabela_multiano)
        v3_thetas = np.array([r["theta"] for r in resultado["resultados_alunos"]])
        r = float(np.corrcoef(v3_thetas, true_thetas)[0, 1])
        assert r > 0.85, f"{area} θ correlation regressed: r={r:.4f} < 0.85"


# ─────────────────────────────────────────────────────────────────────────────
# Real INEP benchmark (skip-if-missing)
# ─────────────────────────────────────────────────────────────────────────────


def _calibrated_cache_path(year: int = 2024):
    """Locate the calibrated parquet from Phase 8's cache layer, or None."""
    try:
        from app.microdata.cache import cache_key
        from app.microdata.loader import default_csv_path
    except ImportError:
        return None
    csv_path = default_csv_path(year)
    if not csv_path.is_file():
        return None
    raw_key = cache_key(csv_path, year=year, sample_size=200_000, rng_seed=42)
    calib_path = raw_key.with_stage("calibrated").parquet_path
    if not calib_path.is_file():
        return None
    return calib_path


# Published floors from task_plan.md Phase 8 P3 smoke test (2026-04-08) with a
# 10% tolerance band — the numbers below are the ceilings the CI will accept
# before flagging a regression. Improvements from plan items #1/#2/#3 should
# push the actuals well inside these floors.
_INEP_BENCHMARK = {
    "LC": {"min_r": 0.90, "max_mae": 50.0},
    "CH": {"min_r": 0.92, "max_mae": 50.0},
    "CN": {"min_r": 0.88, "max_mae": 45.0},
    "MT": {"min_r": 0.90, "max_mae": 45.0},
}


class TestRealINEPBenchmark:
    """End-to-end accuracy floors against the real INEP 2024 sample.

    These tests are the single most important guardrail against silent
    accuracy drift. If they start failing, revert before investigating —
    a 5-10% MAE degradation across all 4 areas is never a wanted outcome.
    """

    @pytest.fixture(scope="class")
    def calibrated_df(self):
        path = _calibrated_cache_path(2024)
        if path is None:
            pytest.skip(
                "Cache 2024 calibrado ausente. Gere com: "
                "`python -m app.evaluation --year 2024` (ou o pipeline V5 'prepare'). "
                "Sem ele, o guardrail de acuracia real NAO roda."
            )
        return pd.read_parquet(path)

    @pytest.mark.integration
    @pytest.mark.parametrize("area", ["LC", "CH", "CN", "MT"])
    def test_mae_and_correlation_within_floor(self, calibrated_df, area):
        al = area.lower()
        nota_v3_col = f"nota_v3_{al}"
        nota_ref_col = f"nota_ref_{al}"
        if nota_v3_col not in calibrated_df or nota_ref_col not in calibrated_df:
            pytest.skip(f"{area}: calibration columns missing in parquet")

        mask = calibrated_df[nota_v3_col].notna() & calibrated_df[nota_ref_col].notna()
        if int(mask.sum()) < 5000:
            pytest.skip(f"{area}: only {int(mask.sum())} aligned rows")

        v3 = calibrated_df.loc[mask, nota_v3_col].to_numpy()
        ref = calibrated_df.loc[mask, nota_ref_col].to_numpy()

        r = float(np.corrcoef(v3, ref)[0, 1])
        mae = float(np.mean(np.abs(v3 - ref)))
        rmse = float(np.sqrt(np.mean((v3 - ref) ** 2)))

        floor = _INEP_BENCHMARK[area]
        assert r >= floor["min_r"], (
            f"{area}: correlation {r:.4f} below floor {floor['min_r']}"
        )
        assert mae <= floor["max_mae"], (
            f"{area}: MAE {mae:.1f} above floor {floor['max_mae']} "
            f"(rmse={rmse:.1f}, n={int(mask.sum())})"
        )
