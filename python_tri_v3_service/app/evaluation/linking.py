# app/evaluation/linking.py
"""Equating de escala IRT: transforma nossos params para a métrica INEP.

Convenção: θ_oficial = A·θ_nosso + B. Para itens: b' = A·b + B, a' = a/A
(c invariante). Linking é necessário porque nossa calibração usa prior N(0,1)
fresco por grupo, enquanto os params oficiais vivem na métrica histórica INEP.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class Linking:
    A: float
    B: float


def _p3pl(theta: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return c + (1.0 - c) / (1.0 + np.exp(-a * (theta - b)))


def mean_sigma_linking(b_ours: np.ndarray, b_official: np.ndarray) -> Linking:
    """Linking pelos momentos das dificuldades dos itens comuns."""
    b_ours = np.asarray(b_ours, dtype=float)
    b_official = np.asarray(b_official, dtype=float)
    A = float(b_official.std(ddof=1) / b_ours.std(ddof=1))
    B = float(b_official.mean() - A * b_ours.mean())
    return Linking(A=A, B=B)


def stocking_lord_linking(
    params_ours: list[dict],
    params_official: list[dict],
    *,
    theta_grid: np.ndarray | None = None,
) -> Linking:
    """Linking que minimiza a diferença das curvas características do teste (TCC)."""
    if theta_grid is None:
        theta_grid = np.linspace(-4.0, 4.0, 81)

    a_o = np.array([p["a"] for p in params_ours], dtype=float)
    b_o = np.array([p["b"] for p in params_ours], dtype=float)
    c_o = np.array([p["c"] for p in params_ours], dtype=float)
    a_f = np.array([p["a"] for p in params_official], dtype=float)
    b_f = np.array([p["b"] for p in params_official], dtype=float)
    c_f = np.array([p["c"] for p in params_official], dtype=float)

    tcc_target = sum(_p3pl(theta_grid, a_f[j], b_f[j], c_f[j]) for j in range(len(a_f)))

    def loss(x: np.ndarray) -> float:
        A, B = float(x[0]), float(x[1])
        if A <= 1e-6:
            return 1e12
        tcc = np.zeros_like(theta_grid)
        for j in range(len(a_o)):
            tcc = tcc + _p3pl(theta_grid, a_o[j] / A, A * b_o[j] + B, c_o[j])
        return float(np.mean((tcc - tcc_target) ** 2))

    res = minimize(loss, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    return Linking(A=float(res.x[0]), B=float(res.x[1]))


def apply_linking(params: list[dict], link: Linking) -> list[dict]:
    """Retorna NOVOS dicts com a/b transformados para a métrica alvo."""
    return [
        {**p, "a": p["a"] / link.A, "b": link.A * p["b"] + link.B}
        for p in params
    ]
