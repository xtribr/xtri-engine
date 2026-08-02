import pytest
import numpy as np
from pathlib import Path


@pytest.mark.unit
def test_microdata_dir_has_no_double_apps():
    from app.microdata.loader import MICRODATA_DIR, default_csv_path

    assert "apps/apps" not in str(MICRODATA_DIR), (
        f"MICRODATA_DIR ainda tem o bug de caminho: {MICRODATA_DIR}"
    )
    assert MICRODATA_DIR == Path("/Volumes/KINGSTON/apps/MIRT/microdata")
    p = default_csv_path(2024)
    assert p.name == "MICRODADOS_ENEM_2024.csv"
    assert p.parent == MICRODATA_DIR


@pytest.mark.unit
def test_mae_rmse_pearson_exact():
    from app.evaluation.metrics import mae, rmse, pearson_r

    pred = np.array([1.0, 2.0, 3.0, 4.0])
    target = np.array([1.0, 2.0, 3.0, 5.0])
    assert mae(pred, target) == pytest.approx(0.25)
    assert rmse(pred, target) == pytest.approx(0.5)
    assert pearson_r(pred, target) == pytest.approx(0.9827076, abs=1e-5)


@pytest.mark.unit
def test_bootstrap_ci_contains_true_mean():
    from app.evaluation.metrics import bootstrap_ci

    rng = np.random.default_rng(0)
    values = rng.normal(50.0, 5.0, size=2000)
    lo, hi = bootstrap_ci(values, n_boot=500, seed=42)
    assert lo < 50.0 < hi
    assert hi - lo < 2.0


@pytest.mark.unit
def test_paired_diff_ci_detects_improvement():
    from app.evaluation.metrics import paired_diff_ci

    err_baseline = np.full(1000, 40.0)
    err_candidate = np.full(1000, 35.0)
    mean_diff, lo, hi = paired_diff_ci(err_candidate, err_baseline, seed=42)
    assert mean_diff == pytest.approx(-5.0)
    assert hi < 0


@pytest.mark.unit
def test_param_recovery_stats_perfect():
    from app.evaluation.metrics import param_recovery_stats

    ours = np.array([0.5, 1.0, 1.5, 2.0])
    stats = param_recovery_stats(ours, ours)
    assert stats["rmse"] == pytest.approx(0.0)
    assert stats["r"] == pytest.approx(1.0)
    assert stats["n"] == 4


@pytest.mark.unit
def test_mean_sigma_recovers_known_transform():
    from app.evaluation.linking import mean_sigma_linking

    rng = np.random.default_rng(1)
    b_ours = rng.normal(0.0, 1.0, size=45)
    A_true, B_true = 1.3, -0.4
    b_official = A_true * b_ours + B_true
    link = mean_sigma_linking(b_ours, b_official)
    assert link.A == pytest.approx(A_true, abs=1e-6)
    assert link.B == pytest.approx(B_true, abs=1e-6)


@pytest.mark.unit
def test_apply_linking_transforms_a_and_b():
    from app.evaluation.linking import Linking, apply_linking

    params = [{"a": 2.0, "b": 1.0, "c": 0.2}]
    out = apply_linking(params, Linking(A=2.0, B=0.5))
    assert out[0]["b"] == pytest.approx(2.0 * 1.0 + 0.5)
    assert out[0]["a"] == pytest.approx(2.0 / 2.0)
    assert out[0]["c"] == pytest.approx(0.2)


@pytest.mark.unit
def test_stocking_lord_recovers_known_transform():
    from app.evaluation.linking import stocking_lord_linking

    rng = np.random.default_rng(2)
    a_f = rng.uniform(0.8, 2.2, size=45)
    b_f = rng.uniform(-2.0, 2.0, size=45)
    c_f = np.full(45, 0.2)
    official = [{"a": a_f[j], "b": b_f[j], "c": c_f[j]} for j in range(45)]
    A_true, B_true = 1.25, 0.3
    ours = [
        {"a": a_f[j] * A_true, "b": (b_f[j] - B_true) / A_true, "c": c_f[j]}
        for j in range(45)
    ]
    link = stocking_lord_linking(ours, official)
    assert link.A == pytest.approx(A_true, abs=0.05)
    assert link.B == pytest.approx(B_true, abs=0.05)


@pytest.mark.unit
def test_official_items_and_coprova():
    from app.evaluation.ground_truth import (
        OFFICIAL_PARAMS_PATH,
        load_official_params,
        official_coprova,
        official_items,
    )

    if not OFFICIAL_PARAMS_PATH.is_file():
        pytest.skip(f"params_oficiais.json ausente em {OFFICIAL_PARAMS_PATH}")

    params = load_official_params()
    assert official_coprova(params, 2024, "MT") == 1323
    assert official_coprova(params, 2024, "CN") == 1333
    items = official_items(params, 2024, "MT")
    assert len(items) == 45
    assert set(items[0].keys()) >= {"a", "b", "c", "posicao", "gabarito"}


@pytest.mark.unit
def test_load_official_params_missing_file_has_actionable_error(tmp_path):
    from app.evaluation.ground_truth import (
        OfficialParamsNotFoundError,
        load_official_params,
    )

    with pytest.raises(OfficialParamsNotFoundError, match="params_oficiais.json"):
        load_official_params(tmp_path / "params_oficiais.json")


@pytest.mark.unit
def test_align_official_to_gabarito_matches_and_skips_annulled():
    from app.evaluation.ground_truth import align_official_to_gabarito

    official = [
        {"a": 1.0, "b": 0.0, "c": 0.2, "posicao": 91, "gabarito": "A"},
        {"a": 1.2, "b": 0.5, "c": 0.2, "posicao": 92, "gabarito": "B"},
        {"a": 0.9, "b": -0.3, "c": 0.2, "posicao": 93, "gabarito": "C"},
    ]
    valid_cols, aligned = align_official_to_gabarito(official, "AXC")
    assert valid_cols == [0, 2]
    assert [p["posicao"] for p in aligned] == [91, 93]


@pytest.mark.unit
def test_align_raises_on_gabarito_mismatch():
    from app.evaluation.ground_truth import align_official_to_gabarito

    official = [{"a": 1.0, "b": 0.0, "c": 0.2, "posicao": 91, "gabarito": "A"}]
    with pytest.raises(ValueError, match="gabarito"):
        align_official_to_gabarito(official, "B")


@pytest.mark.unit
def test_kfold_indices_partition_and_no_leak():
    from app.evaluation.kfold import _kfold_indices

    folds = _kfold_indices(100, k=5, seed=42)
    assert len(folds) == 5
    allidx = np.concatenate(folds)
    assert len(allidx) == 100  # sem repetição (disjunção real)
    assert sorted(allidx.tolist()) == list(range(100))

    # n não-divisível por k: partição ainda exata (sem off-by-one)
    folds2 = _kfold_indices(103, k=5, seed=42)
    allidx2 = np.concatenate(folds2)
    assert len(allidx2) == 103
    assert sorted(allidx2.tolist()) == list(range(103))


@pytest.mark.unit
def test_anchor_batch_matches_individual():
    from app.evaluation.kfold import anchor_batch
    from app.irt.transform import ancorar_nota_individual
    from app.tabela_referencia import TabelaReferenciaTRI

    tabela = TabelaReferenciaTRI()
    theta = np.array([0.5, -0.2, 1.1])
    acertos = np.array([30, 20, 38])
    got = anchor_batch(theta, acertos, "MT", tabela)
    expect = np.array([
        ancorar_nota_individual(theta=0.5, area="MT", acertos=30, tabela=tabela, tolerancia_pct=0.0),
        ancorar_nota_individual(theta=-0.2, area="MT", acertos=20, tabela=tabela, tolerancia_pct=0.0),
        ancorar_nota_individual(theta=1.1, area="MT", acertos=38, tabela=tabela, tolerancia_pct=0.0),
    ])
    assert np.allclose(got, expect)


@pytest.mark.unit
def test_kfold_oof_predictions_shape_and_range():
    from app.evaluation.kfold import kfold_oof_predictions
    from app.tabela_referencia import TabelaReferenciaTRI

    rng = np.random.default_rng(3)
    theta_true = rng.normal(0, 1, size=600)
    b = np.linspace(-2, 2, 45)
    p = 0.2 + 0.8 / (1 + np.exp(-(1.3) * (theta_true[:, None] - b[None, :])))
    resp = (rng.uniform(size=p.shape) < p).astype(int)
    tabela = TabelaReferenciaTRI()
    oof = kfold_oof_predictions(resp, "CN", tabela, k=5, seed=42)
    assert oof.shape == (600,)
    assert np.all(~np.isnan(oof))
    assert np.all(oof >= 100.0) and np.all(oof <= 1000.0)


@pytest.mark.unit
def test_score_with_official_params_shapes():
    from app.evaluation.attribution import score_with_params
    from app.tabela_referencia import TabelaReferenciaTRI

    rng = np.random.default_rng(4)
    resp = (rng.uniform(size=(50, 45)) < 0.6).astype(int)
    params = [{"a": 1.2, "b": 0.0, "c": 0.2, "saturado": False} for _ in range(45)]
    tabela = TabelaReferenciaTRI()
    notas = score_with_params(resp, params, "CN", tabela)
    assert notas.shape == (50,)
    assert np.all(notas >= 100.0) and np.all(notas <= 1000.0)


@pytest.mark.unit
def test_decompose_error_attributes_calibration():
    from app.evaluation.attribution import decompose_error

    ref = np.array([500.0, 600.0, 700.0])
    our_pred = np.array([540.0, 560.0, 760.0])
    official_pred = np.array([510.0, 590.0, 710.0])
    out = decompose_error(our_pred, official_pred, ref)
    assert out["mae_full"] == pytest.approx(np.mean([40, 40, 60]))
    assert out["mae_anchor"] == pytest.approx(np.mean([10, 10, 10]))
    assert out["calib_contrib_mean"] == pytest.approx(out["mae_full"] - out["mae_anchor"])
    assert "calib_contrib_ci" in out


@pytest.mark.unit
def test_build_and_render_report(tmp_path):
    from app.evaluation.report import build_report, render_text_table, write_report

    end_to_end = {
        "LC": {"area": "LC", "n": 1000, "mae": 38.2, "rmse": 47.1, "r": 0.91,
               "mae_ci": (37.0, 39.4)},
        "MT": {"area": "MT", "n": 1000, "mae": 41.0, "rmse": 52.0, "r": 0.90,
               "mae_ci": (39.5, 42.5)},
    }
    attribution = {
        "MT": {"n": 1000, "mae_full": 41.0, "mae_anchor": 22.0,
               "calib_contrib_mean": 19.0, "calib_contrib_ci": (17.0, 21.0)},
    }
    recovery = {
        "MT": {
            "2024": {
                "mean_sigma": {"a": {"r": 0.85, "rmse": 0.3, "mean_diff": 0.0, "n": 45},
                               "b": {"r": 0.93, "rmse": 0.25, "mean_diff": 0.0, "n": 45}},
                "stocking_lord": {"a": {"r": 0.86, "rmse": 0.28, "mean_diff": 0.0, "n": 45},
                                  "b": {"r": 0.94, "rmse": 0.22, "mean_diff": 0.0, "n": 45}},
            }
        }
    }
    report = build_report(year=2024, end_to_end=end_to_end,
                          attribution=attribution, recovery=recovery)
    assert report["year"] == 2024
    assert report["end_to_end"]["MT"]["mae"] == 41.0

    table = render_text_table(report)
    assert "MT" in table and "MAE" in table

    out = tmp_path / "report.json"
    write_report(report, out)
    assert out.is_file()
    import json
    loaded = json.loads(out.read_text())
    assert loaded["year"] == 2024


@pytest.mark.unit
def test_run_recovery_for_group_end_to_end():
    """Recuperação de parâmetro num grupo sintético com gabarito casado."""
    from app.evaluation.__main__ import run_recovery_for_group

    rng = np.random.default_rng(5)
    gabarito = "A" * 45
    b_official = np.linspace(-2, 2, 45)
    official_items = [
        {"a": 1.2, "b": float(b_official[j]), "c": 0.2, "posicao": 91 + j, "gabarito": "A"}
        for j in range(45)
    ]
    theta = rng.normal(0, 1, size=800)
    p = 0.2 + 0.8 / (1 + np.exp(-1.3 * (theta[:, None] - b_official[None, :])))
    resp = (rng.uniform(size=p.shape) < p).astype(int)

    rec = run_recovery_for_group(resp, gabarito, official_items)
    assert set(rec.keys()) == {"mean_sigma", "stocking_lord"}
    assert rec["stocking_lord"]["b"]["r"] > 0.9


@pytest.mark.unit
def test_benchmark_cache_path_uses_corrected_dir():
    from app.microdata.loader import default_csv_path

    p = default_csv_path(2024)
    assert "apps/apps" not in str(p)


@pytest.mark.unit
def test_load_fredao_matrix_synthetic(tmp_path):
    import pandas as pd
    from app.evaluation.fredao_recovery import load_fredao_matrix

    cols = {"ID": [f"S{i}" for i in range(10)]}
    rng = np.random.default_rng(7)
    for pos in range(91, 136):  # 45 itens CN em ordem de posição
        cols[pos] = rng.integers(0, 2, size=10)
    df = pd.DataFrame(cols)
    p = tmp_path / "fredao.xlsx"
    df.to_excel(p, index=False)

    mat = load_fredao_matrix(p)
    assert mat.shape == (10, 45)
    assert set(np.unique(mat)).issubset({0, 1})


@pytest.mark.integration
def test_fredao_recovery_real_data_reports_alignment():
    """O pipeline Fredão roda e reporta sanidade de alinhamento honestamente.

    Cada área deve trazer `sanity_corr_acerto_b` (float) e `aligned` (bool); e a
    recuperação só é preenchida quando alinhado (senão None) — nunca números
    enganosos sob desalinhamento.
    """
    from app.evaluation.fredao_recovery import FREDAO_DIR, run_fredao_recovery
    from app.evaluation.ground_truth import OFFICIAL_PARAMS_PATH

    if not OFFICIAL_PARAMS_PATH.is_file():
        pytest.skip(f"params_oficiais.json ausente em {OFFICIAL_PARAMS_PATH}")
    if not FREDAO_DIR.is_dir():
        pytest.skip(f"Simulado Fredão não encontrado em {FREDAO_DIR}")
    rec = run_fredao_recovery(2017)
    if not rec:
        pytest.skip("Nenhum arquivo Fredão CN/MT presente")
    for area, entry in rec.items():
        assert "sanity_corr_acerto_b" in entry
        assert isinstance(entry["aligned"], bool)
        # invariante de honestidade: recuperação preenchida sse alinhado
        assert (entry["2017"] is not None) == entry["aligned"]
        if entry["aligned"]:
            assert entry["2017"]["stocking_lord"]["b"]["r"] > 0.5
