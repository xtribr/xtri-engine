"""
Calibração unidimensional conjunta via MML/EM.

Esta implementação substitui a calibração item-a-item do motor científico por
uma estimação conjunta com grade fixa de quadratura. Mantém o contrato atual de
parâmetros `{a, b, c, taxa_acerto, saturado}` para não quebrar o restante do
projeto.
"""

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .contracts import ItemParameter
from .scoring import P


def _theta_grid_and_weights(n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    """Retorna a grade de quadratura e os pesos do prior N(0,1)."""

    theta_grid = np.linspace(-4, 4, n_quad)
    weights = norm.pdf(theta_grid, 0, 1)
    weights /= weights.sum()
    return theta_grid, weights


def _init_item_params(resp_matrix: np.ndarray, c: float) -> list[ItemParameter]:
    """Inicializa `a` e `b` a partir da taxa observada de acerto."""

    params: list[ItemParameter] = []
    for j in range(resp_matrix.shape[1]):
        responses = resp_matrix[:, j]
        p_obs = float(np.clip(responses.mean(), c + 0.01, 0.99))
        b_init = -np.log((1 - c) / (p_obs - c) - 1) if p_obs > c else 2.0
        params.append(
            {
                "a": 1.0,
                "b": round(float(b_init), 4),
                "c": c,
                "taxa_acerto": round(float(responses.mean()), 4),
                "saturado": False,
            }
        )
    return params


def _item_probabilities(
    item_params: Sequence[ItemParameter],
    theta_grid: np.ndarray,
) -> np.ndarray:
    """Calcula `P(theta_k, item_j)` em shape `(K, J)`."""

    a_arr = np.array([p["a"] for p in item_params], dtype=float)
    b_arr = np.array([p["b"] for p in item_params], dtype=float)
    c_arr = np.array([p["c"] for p in item_params], dtype=float)
    prob = P(theta_grid[:, None], a_arr[None, :], b_arr[None, :], c_arr[None, :])
    return np.clip(prob, 1e-9, 1 - 1e-9)


def _expected_counts(
    resp_matrix: np.ndarray,
    item_params: Sequence[ItemParameter],
    theta_grid: np.ndarray,
    weights: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Executa o E-step em chunks.

    Retorna:
    - `n_k`: número esperado de alunos em cada ponto da quadratura
    - `r_jk`: acertos esperados por item e ponto da quadratura
    - `loglik`: log-verossimilhança marginal total
    """

    prob = _item_probabilities(item_params, theta_grid)
    log_p = np.log(prob)
    log_1p = np.log(1 - prob)
    log_weights = np.log(np.clip(weights, 1e-300, None))

    n_students, n_items = resp_matrix.shape
    n_quad = theta_grid.shape[0]
    n_k = np.zeros(n_quad, dtype=float)
    r_jk = np.zeros((n_items, n_quad), dtype=float)
    loglik = 0.0

    for start in range(0, n_students, chunk_size):
        chunk = resp_matrix[start:start + chunk_size].astype(float)
        inv_chunk = 1.0 - chunk

        log_lk = np.einsum("bi,ki->bk", chunk, log_p) + np.einsum(
            "bi,ki->bk", inv_chunk, log_1p
        )
        log_post = log_lk + log_weights[None, :]

        max_log = log_post.max(axis=1, keepdims=True)
        post_unnorm = np.exp(log_post - max_log)
        denom = post_unnorm.sum(axis=1, keepdims=True)
        posterior = post_unnorm / np.clip(denom, 1e-300, None)

        n_k += posterior.sum(axis=0)
        r_jk += np.einsum("bi,bk->ik", chunk, posterior)
        loglik += float(np.sum(np.log(np.clip(denom[:, 0], 1e-300, None)) + max_log[:, 0]))

    return n_k, r_jk, loglik


def _neg_expected_log_lik_item(
    params: np.ndarray,
    expected_correct: np.ndarray,
    expected_total: np.ndarray,
    c: float,
    theta_grid: np.ndarray,
) -> float:
    """Parte do M-step para um item."""

    a, b = params
    if a <= 0.01:
        return 1e10

    prob = P(theta_grid, a, b, c)
    prob = np.clip(prob, 1e-9, 1 - 1e-9)
    ll = expected_correct @ np.log(prob) + (expected_total - expected_correct) @ np.log(1 - prob)
    return -float(ll)


def calibrar_itens_mml_em(
    resp_matrix: np.ndarray,
    c: float = 0.20,
    n_quad: int = 41,
    max_iter: int = 25,
    tol: float = 1e-3,
    chunk_size: int = 4096,
) -> list[ItemParameter]:
    """
    Calibra itens conjuntamente via MML/EM.

    O E-step usa chunks de alunos para reduzir consumo de memória em grupos
    grandes. O M-step otimiza cada item com as contagens esperadas da grade.
    """

    if resp_matrix.ndim != 2 or resp_matrix.size == 0:
        return []

    theta_grid, weights = _theta_grid_and_weights(n_quad)
    params = _init_item_params(resp_matrix, c)
    prev_loglik: Optional[float] = None

    for _ in range(max_iter):
        n_k, r_jk, loglik = _expected_counts(
            resp_matrix,
            params,
            theta_grid,
            weights,
            chunk_size,
        )

        new_params: list[ItemParameter] = []
        max_delta = 0.0
        for j, current in enumerate(params):
            res = minimize(
                _neg_expected_log_lik_item,
                x0=[current["a"], current["b"]],
                args=(r_jk[j], n_k, c, theta_grid),
                method="L-BFGS-B",
                bounds=[(0.3, 4.0), (-4.0, 4.0)],
                options={"maxiter": 200, "ftol": 1e-9},
            )
            a_est, b_est = res.x
            max_delta = max(
                max_delta,
                abs(float(a_est) - current["a"]),
                abs(float(b_est) - current["b"]),
            )
            new_params.append(
                {
                    "a": round(float(a_est), 4),
                    "b": round(float(b_est), 4),
                    "c": c,
                    "taxa_acerto": round(float(resp_matrix[:, j].mean()), 4),
                    "saturado": bool(b_est >= 3.9),
                }
            )

        params = new_params
        if prev_loglik is not None and abs(loglik - prev_loglik) < tol and max_delta < 1e-2:
            break
        prev_loglik = loglik

    return params
