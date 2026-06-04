# app/evaluation/attribution.py
"""Decompõe o erro de nota em parcela de ANCORAGEM vs. CALIBRAÇÃO.

Pontuar com params OFICIAIS isola a ancoragem (params são "verdadeiros");
a diferença para o erro com NOSSOS params atribui a parcela de calibração.
Só CN/MT/AZUL de 2024 (único ano com NU_NOTA e params oficiais).
"""
from __future__ import annotations

import numpy as np

from app.evaluation.kfold import anchor_batch
from app.evaluation.metrics import mae, paired_diff_ci
from app.microdata.calibration import eap_theta_batch
from app.tabela_referencia import TabelaReferenciaTRI


def score_with_params(
    resp_matrix: np.ndarray,
    item_params: list[dict],
    area: str,
    tabela: TabelaReferenciaTRI,
) -> np.ndarray:
    """EAP com params FIXOS + ancoragem. `resp_matrix` deve alinhar a `item_params`."""
    theta, _ = eap_theta_batch(resp_matrix, item_params)
    acertos = resp_matrix.sum(axis=1)
    return anchor_batch(theta, acertos, area, tabela)


def decompose_error(
    our_pred: np.ndarray, official_pred: np.ndarray, nota_ref: np.ndarray
) -> dict:
    """Atribui erro de nota a ancoragem (oficial) vs. calibração (resíduo)."""
    err_full = np.abs(np.asarray(our_pred, float) - np.asarray(nota_ref, float))
    err_anchor = np.abs(np.asarray(official_pred, float) - np.asarray(nota_ref, float))
    mean_diff, lo, hi = paired_diff_ci(err_full, err_anchor)
    return {
        "n": int(len(nota_ref)),
        "mae_full": mae(our_pred, nota_ref),
        "mae_anchor": mae(official_pred, nota_ref),
        "calib_contrib_mean": mean_diff,
        "calib_contrib_ci": (lo, hi),
    }
