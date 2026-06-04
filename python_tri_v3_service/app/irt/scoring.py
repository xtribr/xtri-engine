"""
Funções de scoring do núcleo IRT.

Este módulo concentra apenas funções puras de probabilidade, estimação EAP e
classificação pedagógica. A calibração conjunta entra na próxima fase.
"""

from typing import Sequence

import numpy as np
from scipy.stats import norm

from .contracts import ItemParameter


def P(theta: np.ndarray, a: float, b: float, c: float = 0.20) -> np.ndarray:
    """Probabilidade de acerto no modelo logístico com `c` fixo."""

    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))


def eap_theta(
    responses: np.ndarray,
    item_params: Sequence[ItemParameter],
    n_quad: int = 61,
) -> tuple[float, float]:
    """
    Estima θ via EAP com prior N(0,1).

    Grade ampliada para `[-5, 5]` com 61 pontos (cf. accuracy audit 2026-04)
    para cobrir alunos de ability extrema sem clipar no limite anterior de
    ±4σ. Mantém compatibilidade de assinatura: callers podem forçar 41 pontos
    se precisarem reproduzir um baseline antigo.

    Items flagged ``saturado=True`` are excluded from the likelihood: their
    b estimate was clamped at the optimizer bound (|b| >= 4.8) so using it
    injects a bias roughly equal to the clamp distance. If every item is
    saturated, the posterior collapses to the prior (θ≈0, SE≈1).
    """

    theta_grid = np.linspace(-5, 5, n_quad)
    prior = norm.pdf(theta_grid, 0, 1)

    lk = np.ones(n_quad)
    for j, p in enumerate(item_params):
        if p.get("saturado"):
            continue  # unreliable b — skip to avoid biasing theta
        prob = P(theta_grid, p["a"], p["b"], p["c"])
        prob = np.clip(prob, 1e-9, 1 - 1e-9)
        lk *= np.where(responses[j] == 1, prob, 1 - prob)

    posterior = lk * prior
    s = posterior.sum()
    if s < 1e-300:
        return 0.0, 1.0
    posterior /= s
    theta_eap = float(np.sum(theta_grid * posterior))
    se_eap = float(np.sqrt(np.sum((theta_grid - theta_eap) ** 2 * posterior)))
    return round(theta_eap, 6), round(se_eap, 4)


def estado_tri(nota: float) -> str:
    """Classifica a nota TRI em nível pedagógico."""

    if nota >= 700:
        return "AVANCADO"
    if nota >= 600:
        return "PROFICIENTE"
    if nota >= 500:
        return "BASICO"
    if nota >= 400:
        return "ELEMENTAR"
    return "INICIANTE"
