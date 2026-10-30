"""
Calibração unidimensional conjunta via MML/EM.

Esta implementação substitui a calibração item-a-item do motor científico por
uma estimação conjunta com grade fixa de quadratura. Mantém o contrato atual de
parâmetros `{a, b, c, taxa_acerto, saturado}` para não quebrar o restante do
projeto.
"""

import logging
from typing import Any, Optional, Sequence, Union

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .contracts import ItemParameter
from .scoring import P

logger = logging.getLogger(__name__)


def _theta_grid_and_weights(n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    """Retorna a grade de quadratura e os pesos do prior N(0,1).

    A grade vai de -5 a +5 (cobre ~99.99994% da massa do prior) para evitar
    clipar alunos de ability extrema. Cf. accuracy audit 2026-04 — a grade
    anterior em [-4, 4] truncava ~0.006% da cauda, o que se traduzia em
    ~2-3 pontos de MAE contra o INEP nos alunos acima do percentil 99.
    """

    theta_grid = np.linspace(-5, 5, n_quad)
    weights = norm.pdf(theta_grid, 0, 1)
    weights /= weights.sum()
    return theta_grid, weights


def _init_item_params(resp_matrix: np.ndarray, c: float) -> list[ItemParameter]:
    """Inicializa `a` e `b` a partir da taxa observada de acerto."""

    params: list[ItemParameter] = []
    for j in range(resp_matrix.shape[1]):
        responses = resp_matrix[:, j]
        p_obs = float(np.clip(responses.mean(), c + 0.01, 0.99))
        # Com theta = 0 e a = 1: p = c + (1 - c) / (1 + e^b), logo
        # b = ln((1 - c) / (p - c) - 1). Item fácil começa com b negativo.
        b_init = np.log((1 - c) / (p_obs - c) - 1) if p_obs > c else 2.0
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
    """Parte do M-step para um item (c fixo)."""

    a, b = params
    if a <= 0.01:
        return 1e10

    prob = P(theta_grid, a, b, c)
    prob = np.clip(prob, 1e-9, 1 - 1e-9)
    ll = expected_correct @ np.log(prob) + (expected_total - expected_correct) @ np.log(1 - prob)
    return -float(ll)


# Beta(α=10, β=40) prior on c: mean=0.20, mode≈0.19, std≈0.055. Matches the
# design choice "5-alternative MCQ" with modest shrinkage toward 0.20. The
# values are hand-tuned so the penalty becomes negligible for N >= 1000 but
# strong enough to keep c stable on smaller cohorts.
_C_PRIOR_ALPHA = 10.0
_C_PRIOR_BETA = 40.0
# Below this informative-row count, the Bayes path falls back to fixed c.
# Rationale: with too few students, the prior dominates and per-item
# estimation mostly adds noise. 1000 informative rows gives ~25 students
# per item on a 45-item test — enough signal to justify the extra dof.
_BAYES_C_MIN_N = 1000


def _neg_penalized_log_lik_item_bayes_c(
    params: np.ndarray,
    expected_correct: np.ndarray,
    expected_total: np.ndarray,
    theta_grid: np.ndarray,
) -> float:
    """M-step for one item with Beta(α, β) prior on c (per-item estimation).

    Penalized log-likelihood: ll(data | a, b, c) + log Beta(c; α, β).
    The Beta normalizing constant is independent of c so we drop it. Bounds
    are enforced by L-BFGS-B; the internal guards here just prevent NaN at
    the boundary.
    """
    a, b, c = params
    if a <= 0.01 or c <= 0.001 or c >= 0.999:
        return 1e10
    prob = P(theta_grid, a, b, c)
    prob = np.clip(prob, 1e-9, 1 - 1e-9)
    ll = expected_correct @ np.log(prob) + (expected_total - expected_correct) @ np.log(1 - prob)
    log_prior = (_C_PRIOR_ALPHA - 1.0) * np.log(c) + (_C_PRIOR_BETA - 1.0) * np.log(1.0 - c)
    return -float(ll + log_prior)


def _informative_rows_mask(resp_matrix: np.ndarray) -> np.ndarray:
    """Boolean mask of rows that have at least one 1 AND at least one 0.

    Rows with all-zeros or all-ones (row_sum == 0 or row_sum == n_items) carry
    no within-student discriminating signal: their posterior theta collapses
    to the quadrature edge regardless of item parameters, and including them
    in the M-step biases item difficulties (b) toward the prior mean.
    Filtering them out of calibration — but NOT out of scoring — addresses
    the ~2-3 MAE-point poisoning flagged in the Apr-2026 accuracy audit.
    """
    if resp_matrix.ndim != 2 or resp_matrix.size == 0:
        return np.zeros((0,), dtype=bool)
    n_items = resp_matrix.shape[1]
    row_sums = resp_matrix.sum(axis=1)
    return (row_sums > 0) & (row_sums < n_items)


def calibrar_itens_mml_em(
    resp_matrix: np.ndarray,
    c: float = 0.20,
    n_quad: int = 61,
    max_iter: int = 500,
    tol: float = 1e-4,
    chunk_size: int = 4096,
    c_strategy: str = "fixed",
    bayes_min_n: Optional[int] = None,
    return_info: bool = False,
) -> Union[list[ItemParameter], tuple[list[ItemParameter], dict[str, Any]]]:
    """
    Calibra itens conjuntamente via MML/EM.

    O E-step usa chunks de alunos para reduzir consumo de memória em grupos
    grandes. O M-step otimiza cada item com as contagens esperadas da grade.
    Alunos com resposta constante (tudo-0 ou tudo-N) são excluídos da
    calibração — cf. :func:`_informative_rows_mask` — mas ainda aparecem no
    ``taxa_acerto`` reportado por item, pois esse campo é diagnóstico.

    Args:
        c_strategy: ``"fixed"`` (default) keeps ``c`` constant across items
            at the scalar value passed via ``c``. ``"bayes"`` estimates
            ``c`` per item under a Beta(10, 40) prior (mean 0.20, std ~0.055)
            whenever the informative-row count is at least ``_BAYES_C_MIN_N``
            — below that threshold the Bayes path gracefully falls back to
            fixed ``c``. Expected MAE improvement: 5-10 points on cohorts
            where true item guessing rates vary (per the Apr-2026 audit).
        bayes_min_n: limiar de alunos informativos para estimar ``c`` no
            caminho ``"bayes"``. ``None`` usa ``_BAYES_C_MIN_N``.
        return_info: com ``True``, devolve ``(params, info)``, em que
            ``info = {"iteracoes": n, "convergiu": bool}``. O padrão
            ``False`` mantém o retorno antigo (só a lista de parâmetros).
    """

    if resp_matrix.ndim != 2 or resp_matrix.size == 0:
        return ([], {"iteracoes": 0, "convergiu": False}) if return_info else []

    # Calibrate on the informative subset, but keep the full matrix for the
    # per-item taxa_acerto diagnostic (answered by the whole cohort).
    informative_mask = _informative_rows_mask(resp_matrix)
    n_informative = int(informative_mask.sum())
    if n_informative >= 10:
        resp_for_em = resp_matrix[informative_mask]
    else:
        # Too few informative rows — fall back to the full matrix. Better to
        # calibrate with bias than to refuse to calibrate at all, but the
        # resulting item params are likely to hit the L-BFGS-B bounds.
        logger.warning(
            "calibrar_itens_mml_em: only %d informative rows (of %d); "
            "falling back to full matrix — item params may saturate at bounds.",
            n_informative,
            resp_matrix.shape[0],
        )
        resp_for_em = resp_matrix

    theta_grid, weights = _theta_grid_and_weights(n_quad)
    params = _init_item_params(resp_for_em, c)
    prev_loglik: Optional[float] = None

    # Decide the c strategy once, before the EM loop. Bayesian per-item c is
    # only safe when there are enough informative students (_BAYES_C_MIN_N);
    # below that the Beta prior dominates anyway and the extra degree of
    # freedom just introduces noise.
    limiar_bayes = _BAYES_C_MIN_N if bayes_min_n is None else int(bayes_min_n)
    use_bayes_c = (
        c_strategy == "bayes" and resp_for_em.shape[0] >= limiar_bayes
    )
    if c_strategy == "bayes" and not use_bayes_c:
        logger.info(
            "calibrar_itens_mml_em: c_strategy='bayes' requested but only "
            "%d informative rows (< %d); falling back to fixed c=%s.",
            resp_for_em.shape[0],
            limiar_bayes,
            c,
        )

    iteracoes = 0
    convergiu = False
    for _ in range(max_iter):
        iteracoes += 1
        n_k, r_jk, loglik = _expected_counts(
            resp_for_em,
            params,
            theta_grid,
            weights,
            chunk_size,
        )

        new_params: list[ItemParameter] = []
        max_delta = 0.0
        for j, current in enumerate(params):
            if use_bayes_c:
                res = minimize(
                    _neg_penalized_log_lik_item_bayes_c,
                    x0=[current["a"], current["b"], current["c"]],
                    args=(r_jk[j], n_k, theta_grid),
                    method="L-BFGS-B",
                    bounds=[(0.3, 4.0), (-5.0, 5.0), (0.05, 0.40)],
                    options={"maxiter": 200, "ftol": 1e-9},
                )
                a_est, b_est, c_est = res.x
            else:
                res = minimize(
                    _neg_expected_log_lik_item,
                    x0=[current["a"], current["b"]],
                    args=(r_jk[j], n_k, c, theta_grid),
                    method="L-BFGS-B",
                    bounds=[(0.3, 4.0), (-5.0, 5.0)],
                    options={"maxiter": 200, "ftol": 1e-9},
                )
                a_est, b_est = res.x
                c_est = c
            # c-delta has to join the convergence check whenever the Bayes
            # path is active — otherwise c can oscillate after a/b have
            # settled and the EM terminates prematurely.
            max_delta = max(
                max_delta,
                abs(float(a_est) - current["a"]),
                abs(float(b_est) - current["b"]),
            )
            if use_bayes_c:
                max_delta = max(max_delta, abs(float(c_est) - current["c"]))
            new_params.append(
                {
                    "a": round(float(a_est), 4),
                    "b": round(float(b_est), 4),
                    "c": round(float(c_est), 4),
                    "taxa_acerto": round(float(resp_matrix[:, j].mean()), 4),
                    # Saturation threshold tied to the L-BFGS-B bound (5.0).
                    # Reaching within 0.2 of the bound means the optimizer likely
                    # clamped there, i.e. the item is harder than we can reliably
                    # estimate with this sample size.
                    "saturado": bool(b_est >= 4.8),
                }
            )

        params = new_params
        if prev_loglik is not None and abs(loglik - prev_loglik) < tol and max_delta < 1e-2:
            convergiu = True
            break
        prev_loglik = loglik

    if not convergiu:
        logger.warning(
            "calibrar_itens_mml_em: EM terminou sem convergir após %d iterações "
            "(max_iter=%d, tol=%g); parâmetros podem não estar estáveis.",
            iteracoes,
            max_iter,
            tol,
        )
    if return_info:
        return params, {"iteracoes": iteracoes, "convergiu": convergiu}
    return params
