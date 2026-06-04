# app/evaluation/metrics.py
"""Métricas de acurácia e intervalos de confiança por bootstrap."""
from __future__ import annotations

import numpy as np


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    return float(np.mean(np.abs(pred - target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(pred) < 2 or np.std(pred) == 0 or np.std(target) == 0:
        return float("nan")
    return float(np.corrcoef(pred, target)[0, 1])


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    stat: str = "mean",
) -> tuple[float, float]:
    """IC percentil por bootstrap de uma estatística (mean|median) de `values`."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    fn = {"mean": np.mean, "median": np.median}[stat]
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = fn(values[rng.integers(0, n, n)])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_diff_ci(
    err_candidate: np.ndarray,
    err_baseline: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """IC bootstrap da diferença PAREADA de erro (candidato − baseline).

    Negativo = candidato erra menos. Se `hi < 0`, a melhora é significativa
    (o IC exclui 0). Pareado por aluno: mais sensível que comparar dois ICs.
    """
    rng = np.random.default_rng(seed)
    diff = np.asarray(err_candidate, dtype=float) - np.asarray(err_baseline, dtype=float)
    n = len(diff)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = diff[rng.integers(0, n, n)].mean()
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(diff.mean()), lo, hi


def param_recovery_stats(ours: np.ndarray, official: np.ndarray) -> dict:
    """Compara um vetor de parâmetros calibrados vs. oficiais (já linkados)."""
    ours = np.asarray(ours, dtype=float)
    official = np.asarray(official, dtype=float)
    return {
        "n": int(len(ours)),
        "r": pearson_r(ours, official),
        "rmse": rmse(ours, official),
        "mean_diff": float(np.mean(ours - official)),
    }
