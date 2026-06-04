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
