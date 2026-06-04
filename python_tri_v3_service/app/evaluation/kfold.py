# app/evaluation/kfold.py
"""Hold-out k-fold sobre alunos: calibra no treino, pontua no teste.

Mede generalização (não memorização). Reusa calibração e EAP existentes;
nunca mistura provas/áreas num modelo (regra inviolável).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.irt.transform import ancorar_nota_individual
from app.microdata.calibration import (
    _effective_lc_gabarito,
    _parse_nota_series,
    _resp_to_binary,
    calibrar_itens_fast,
    eap_theta_batch,
)
from app.microdata.schema import MicrodataSchema
from app.tabela_referencia import TabelaReferenciaTRI

MIN_ITEMS_PER_STUDENT = 30


def _kfold_indices(n: int, k: int, seed: int) -> list[np.ndarray]:
    """k folds de teste (partição exata dos índices 0..n-1)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [np.sort(perm[i::k]) for i in range(k)]


def anchor_batch(
    theta_arr: np.ndarray,
    acertos_arr: np.ndarray,
    area: str,
    tabela: TabelaReferenciaTRI,
) -> np.ndarray:
    return np.array(
        [
            ancorar_nota_individual(
                theta=float(t), area=area, acertos=int(ac),
                tabela=tabela, tolerancia_pct=0.0,
            )
            for t, ac in zip(theta_arr, acertos_arr)
        ],
        dtype=float,
    )


def build_group_matrix(
    df_group: pd.DataFrame, schema: MicrodataSchema, area: str
) -> tuple[np.ndarray, np.ndarray]:
    """Constrói (matriz 0/1, nota_ref) de um grupo homogêneo (mesma prova)."""
    resp_arr = df_group[schema.tx_respostas[area]].astype(str).to_numpy()
    gab_arr = df_group[schema.tx_gabarito[area]].astype(str).to_numpy()
    nota_ref = _parse_nota_series(df_group[schema.nu_nota[area]]).to_numpy()
    lingua_arr = (
        df_group[schema.tp_lingua].astype(str).to_numpy() if area == "LC" else None
    )
    rows: list[list[int]] = []
    notas: list[float] = []
    for i in range(len(df_group)):
        gab = gab_arr[i]
        if area == "LC" and len(gab) == 50:
            try:
                tp = int(lingua_arr[i].strip())
            except (ValueError, AttributeError):
                continue
            sliced = _effective_lc_gabarito(gab, tp)
            if sliced is None:
                continue
            gab = sliced
        binary = _resp_to_binary(resp_arr[i], gab)
        if binary is None or len(binary) < MIN_ITEMS_PER_STUDENT:
            continue
        rows.append(binary)
        notas.append(float(nota_ref[i]))
    if not rows or len({len(r) for r in rows}) > 1:
        return np.empty((0, 0), dtype=int), np.empty((0,), dtype=float)
    return np.array(rows, dtype=int), np.array(notas, dtype=float)


def kfold_oof_predictions(
    resp_matrix: np.ndarray,
    area: str,
    tabela: TabelaReferenciaTRI,
    *,
    k: int = 5,
    seed: int = 42,
    c_strategy: str = "fixed",
) -> np.ndarray:
    """Notas out-of-fold: para cada fold, calibra no resto e pontua o fold."""
    n = len(resp_matrix)
    oof = np.full(n, np.nan, dtype=float)
    acertos = resp_matrix.sum(axis=1)
    folds = _kfold_indices(n, k, seed)
    all_idx = np.arange(n)
    for test_idx in folds:
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        params = calibrar_itens_fast(resp_matrix[train_idx], c_strategy=c_strategy)
        theta, _ = eap_theta_batch(resp_matrix[test_idx], params)
        oof[test_idx] = anchor_batch(theta, acertos[test_idx], area, tabela)
    return oof


def run_kfold_area(
    df: pd.DataFrame,
    schema: MicrodataSchema,
    area: str,
    tabela: TabelaReferenciaTRI,
    *,
    k: int = 5,
    seed: int = 42,
    c_strategy: str = "fixed",
    min_group: int = 500,
) -> dict:
    """Agrega predições OOF por (co_prova[, tp_lingua]) e empilha por área."""
    co_col = schema.co_prova[area]
    group_cols = [co_col, schema.tp_lingua] if area == "LC" else [co_col]
    preds: list[np.ndarray] = []
    refs: list[np.ndarray] = []
    for _, g in df.groupby(group_cols, dropna=False):
        resp, nota_ref = build_group_matrix(g, schema, area)
        if len(resp) < min_group:
            continue
        oof = kfold_oof_predictions(resp, area, tabela, k=k, seed=seed, c_strategy=c_strategy)
        mask = ~np.isnan(oof) & ~np.isnan(nota_ref) & (nota_ref > 0)
        preds.append(oof[mask])
        refs.append(nota_ref[mask])
    if not preds:
        return {"area": area, "n": 0, "pred": np.array([]), "ref": np.array([])}
    return {
        "area": area,
        "n": int(sum(len(p) for p in preds)),
        "pred": np.concatenate(preds),
        "ref": np.concatenate(refs),
    }
