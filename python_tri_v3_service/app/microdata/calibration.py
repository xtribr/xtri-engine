"""
V3 Local calibration pipeline for ENEM microdata samples.

For each area (LC, CH, CN, MT), this module:

1. Groups the sampled DataFrame by ``CO_PROVA_{area}`` (plus ``TP_LINGUA`` for LC,
   because English-vs-Spanish students received different language items in
   positions 0-4 of the 45-item response string).
2. For each homogeneous group with at least :data:`MIN_CALIBRATION_SIZE` students,
   builds a binary response matrix by comparing each student's
   ``TX_RESPOSTAS_{area}`` against their own ``TX_GABARITO_{area}`` char-by-char.
3. Feeds the matrix to :func:`app.motores.cientifico.pipeline_cientifico`, which
   calibrates item parameters (MML + L-BFGS-B), estimates per-student theta via
   EAP, and anchors notas using the fixed-theta `MIN-MED-MAX` reference rule.
4. Merges V3 Local results back into the original DataFrame alongside the
   official reference scores (``NU_NOTA_{area}``).

The result is a single enriched DataFrame ready for Phase 4 dashboards:
comparison scatters, distribution histograms, difficulty tables, and correlation
metrics against the INEP ground truth.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from app.irt.unidim_em import calibrar_itens_mml_em
from app.microdata.schema import AREAS, MicrodataSchema
from app.motores.cientifico import ancorar_notas, estado_tri
from app.tabela_referencia import TabelaReferenciaTRI

logger = logging.getLogger(__name__)

# Minimum number of students required to calibrate a (area, co_prova [, tp_lingua])
# group. Below this, IRT calibration becomes statistically unreliable and the
# group is skipped. Can be lowered for testing.
MIN_CALIBRATION_SIZE = 500

# Minimum binary-vector length (number of items with a valid gabarito) required
# to estimate theta for a single student. Matches ``validacao_massa.py:209``.
MIN_ITEMS_PER_STUDENT = 30


def calibrar_itens_fast(
    resp_matrix: np.ndarray,
    c: float = 0.20,
    n_quad: int = 41,
) -> list[dict[str, Any]]:
    """
    Wrapper do estimador conjunto usado no motor principal.

    O nome `fast` foi mantido para preservar o contrato do módulo de microdados,
    mas a implementação agora delega para a calibração conjunta em chunks.
    """

    return calibrar_itens_mml_em(resp_matrix, c=c, n_quad=n_quad)


def eap_theta_batch(
    resp_matrix: np.ndarray,
    item_params: list[dict[str, float]],
    n_quad: int = 41,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized EAP theta estimation for ALL students at once.

    Equivalent to calling ``app.motores.cientifico.eap_theta`` once per row of
    ``resp_matrix``, but ~100-1000x faster because it replaces the Python
    per-student loop with a single NumPy matmul over the log-likelihood.

    The math:
        For each quadrature point θ_k and each item j:
            P(θ_k, j) = c_j + (1 - c_j) / (1 + exp(-a_j * (θ_k - b_j)))
        For each student i, log-likelihood at θ_k is:
            log_lk[i, k] = sum_j resp[i, j] * log(P[k, j]) +
                           (1 - resp[i, j]) * log(1 - P[k, j])
        This is a single matmul: resp @ log_p.T + (1 - resp) @ log_1p.T
        Then posterior ∝ exp(log_lk) * prior, θ_EAP = sum(θ_k * posterior_k).

    Args:
        resp_matrix: ``(N_students, N_items)`` int array of 0/1 responses.
        item_params: List of N_items dicts with keys ``a``, ``b``, ``c``.
        n_quad: Quadrature points (default 41, matches ``eap_theta``).

    Returns:
        ``(theta, se)`` each shape ``(N_students,)`` as float64 arrays.
    """
    theta_grid = np.linspace(-4, 4, n_quad)  # (K,)
    prior = norm.pdf(theta_grid, 0, 1)  # (K,)

    a_arr = np.array([p["a"] for p in item_params], dtype=float)  # (J,)
    b_arr = np.array([p["b"] for p in item_params], dtype=float)  # (J,)
    c_arr = np.array([p["c"] for p in item_params], dtype=float)  # (J,)

    # P_kj: shape (K, J) — 3PL probability at each quadrature point for each item.
    # z[k, j] = a_j * (θ_k - b_j)
    z = a_arr[None, :] * (theta_grid[:, None] - b_arr[None, :])
    prob = c_arr[None, :] + (1 - c_arr[None, :]) / (1 + np.exp(-z))
    prob = np.clip(prob, 1e-9, 1 - 1e-9)

    log_p = np.log(prob)  # (K, J)
    log_1p = np.log(1 - prob)  # (K, J)

    resp = resp_matrix.astype(float)  # (N, J)
    one_minus_resp = 1.0 - resp  # (N, J)

    # log_lk[i, k] = sum_j resp[i, j] * log_p[k, j] + (1 - resp[i, j]) * log_1p[k, j]
    # Matmul form: (N, J) @ (J, K) = (N, K)
    log_lk = resp @ log_p.T + one_minus_resp @ log_1p.T  # (N, K)

    # Numerically stable softmax × prior
    max_log = log_lk.max(axis=1, keepdims=True)  # (N, 1)
    lk = np.exp(log_lk - max_log) * prior[None, :]  # (N, K)
    total = lk.sum(axis=1, keepdims=True)  # (N, 1)
    total = np.where(total < 1e-300, 1.0, total)
    posterior = lk / total  # (N, K)

    theta_eap = (theta_grid[None, :] * posterior).sum(axis=1)  # (N,)
    var_eap = ((theta_grid[None, :] - theta_eap[:, None]) ** 2 * posterior).sum(axis=1)
    se_eap = np.sqrt(np.maximum(var_eap, 0.0))

    return theta_eap, se_eap


def _pipeline_cientifico_batch(
    alunos: list[dict[str, Any]],
    area: str,
    tabela: TabelaReferenciaTRI,
    c_fixo: float = 0.20,
) -> dict[str, Any]:
    """Batched replacement for :func:`app.motores.cientifico.pipeline_cientifico`.

    Same inputs, same output shape. The only difference is that per-student
    EAP theta estimation uses :func:`eap_theta_batch` instead of the original
    Python loop — ~100-1000x speedup for large groups (tens of thousands of
    students per prova group from the microdata).
    """
    resp_matrix = np.array([a["respostas"] for a in alunos], dtype=int)
    n_alunos, n_itens = resp_matrix.shape

    # 1. Calibrate items via the FAST O(quadrature) likelihood (no per-student loop).
    item_params = calibrar_itens_fast(resp_matrix, c=c_fixo)

    b_vals = [p["b"] for p in item_params]
    n_saturados = sum(1 for b in b_vals if b >= 3.9)

    # 2. Primi difficulty subgroups.
    idx_faceis = [j for j, p in enumerate(item_params) if p["b"] < -1.5]
    idx_medios = [j for j, p in enumerate(item_params) if -1.5 <= p["b"] <= 1.5]
    idx_dificeis = [j for j, p in enumerate(item_params) if p["b"] > 1.5]
    n_dificeis = len(idx_dificeis)

    # 3. VECTORIZED theta EAP for all students at once.
    theta_arr, se_arr = eap_theta_batch(resp_matrix, item_params)

    # Per-student Primi sub-scores via vectorized boolean indexing.
    acertos_arr = resp_matrix.sum(axis=1)
    acertos_dificeis_arr = (
        resp_matrix[:, idx_dificeis].sum(axis=1) if idx_dificeis else np.zeros(n_alunos, dtype=int)
    )
    acert_dif_arr = (
        acertos_dificeis_arr / n_dificeis if n_dificeis > 0 else np.zeros(n_alunos)
    )

    # 4. Build the thetas list that ancorar_notas expects.
    thetas: list[dict[str, Any]] = []
    for i, aluno in enumerate(alunos):
        thetas.append(
            {
                "aluno_id": aluno["aluno_id"],
                "nome": aluno.get("nome", ""),
                "acertos": int(acertos_arr[i]),
                "theta": round(float(theta_arr[i]), 6),
                "se": round(float(se_arr[i]), 4),
                "acertos_dificeis": int(acertos_dificeis_arr[i]),
                "acert_dif": round(float(acert_dif_arr[i]), 4),
            }
        )

    # 5. Anchor notas via the existing cohort-anchored function.
    resultados = ancorar_notas(thetas, tabela, area)
    for r in resultados:
        r["estado_tri"] = estado_tri(r["nota_ancorada"])

    # 6. Diagnostics.
    notas = [r["nota_ancorada"] for r in resultados]
    diagnostico = {
        "n_alunos": n_alunos,
        "n_itens": n_itens,
        "b_medio": round(float(np.mean(b_vals)), 3) if b_vals else 0.0,
        "b_desvio": round(float(np.std(b_vals)), 3) if b_vals else 0.0,
        "itens_saturados": n_saturados,
        "nota_media": round(float(np.mean(notas)), 1) if notas else 0.0,
        "itens_dificeis": n_dificeis,
    }

    return {
        "parametros_itens": item_params,
        "resultados_alunos": resultados,
        "diagnostico": diagnostico,
    }


def _resp_to_binary(resp: str, gab: str) -> Optional[list[int]]:
    """Compare a student's response string against their own gabarito.

    Returns a list of 0/1 values (1 = matched the key), or ``None`` if the
    strings are empty/mismatched in length.
    """
    if not resp or not gab:
        return None
    if len(resp) != len(gab):
        return None
    binary: list[int] = []
    for r, g in zip(resp, gab):
        # Annulled items (gabarito = '.', 'X', '*') give credit to all.
        if g in "X.*":
            binary.append(1)
        elif r == g:
            binary.append(1)
        else:
            binary.append(0)
    return binary


def _effective_lc_gabarito(full_gab: str, tp_lingua: int) -> Optional[str]:
    """Slice the 50-char LC gabarito down to the 45-char effective key.

    INEP 2024+ stores the LC answer key as a 50-char string:
        positions 0-4  → 5 English language items
        positions 5-9  → 5 Spanish language items
        positions 10-49 → 40 common items

    Students only take 45 items (either 5 English OR 5 Spanish, plus 40 common),
    and their ``TX_RESPOSTAS_LC`` is only 45 chars long. To score correctly we
    need to slice the right language items from the full gabarito based on
    ``TP_LINGUA`` (0 = English, 1 = Spanish).

    Args:
        full_gab: The 50-char raw gabarito from the microdata.
        tp_lingua: 0 for English, 1 for Spanish.

    Returns:
        A 45-char effective gabarito matching the student's response layout,
        or ``None`` if the input is not 50 chars or tp_lingua is unknown.
    """
    if full_gab is None or len(full_gab) != 50:
        return None
    if tp_lingua == 0:
        return full_gab[0:5] + full_gab[10:50]
    if tp_lingua == 1:
        return full_gab[5:10] + full_gab[10:50]
    return None


def _parse_nota_series(series: pd.Series) -> pd.Series:
    """Convert a BR-decimal NU_NOTA string column to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def calibrate_area_group(
    df: pd.DataFrame,
    schema: MicrodataSchema,
    area: str,
    tabela: TabelaReferenciaTRI,
    *,
    c_fixo: float = 0.20,
    min_calibration_size: int = MIN_CALIBRATION_SIZE,
) -> pd.DataFrame:
    """Run V3 Local calibration on a homogeneous (area, co_prova [, tp_lingua]) group.

    All students in ``df`` must have taken the same prova (same gabarito). The
    caller is responsible for grouping upstream.

    Args:
        df: Sub-DataFrame of students who took the same prova.
        schema: Column accessor for the microdata.
        area: ENEM area code ("LC", "CH", "CN", or "MT").
        tabela: Reference table for nota anchoring.
        c_fixo: Fixed guessing parameter (0.20 for 5 alternatives).
        min_calibration_size: Minimum group size to attempt calibration.

    Returns:
        DataFrame indexed by the original ``df.index`` values, with columns
        ``theta_v3``, ``se_v3``, ``nota_v3``, ``acertos_v3``, ``estado_v3``,
        ``acert_dif``. Empty if the group is too small or calibration failed.
    """
    if len(df) < min_calibration_size:
        return pd.DataFrame()

    resp_col = schema.tx_respostas[area]
    gab_col = schema.tx_gabarito[area]
    id_col = schema.id_col
    lingua_col = schema.tp_lingua if area == "LC" else None

    # Extract columns as NumPy arrays up front. `df.iterrows()` is ~1000x slower
    # than numpy array access because of pandas' per-row dtype coercion overhead;
    # the tight Python loop below is fast enough for 100K+ students per group.
    resp_arr = df[resp_col].astype(str).to_numpy()
    gab_arr = df[gab_col].astype(str).to_numpy()
    id_arr = df[id_col].astype(str).to_numpy()
    idx_arr = df.index.to_numpy()
    lingua_arr = (
        df[lingua_col].astype(str).to_numpy()
        if lingua_col is not None and lingua_col in df.columns
        else None
    )

    # Build the list of student dicts expected by the batched pipeline. Use a
    # synthetic aluno_id that encodes the original index so we can merge
    # results back without relying on the microdata ID being unique within the group.
    alunos: list[dict] = []
    original_indices: list = []
    n_rows = len(df)
    for i in range(n_rows):
        resp = resp_arr[i]
        gab = gab_arr[i]
        if not resp or not gab:
            continue
        # LC special case: INEP stores a 50-char gabarito that covers both
        # English and Spanish language items. Slice it down to the 45-char
        # effective key for the student's TP_LINGUA.
        if area == "LC" and lingua_arr is not None and len(gab) == 50:
            try:
                tp_lingua = int(lingua_arr[i].strip())
            except (ValueError, AttributeError):
                continue
            sliced = _effective_lc_gabarito(gab, tp_lingua)
            if sliced is None:
                continue
            gab = sliced
        binary = _resp_to_binary(resp, gab)
        if binary is None or len(binary) < MIN_ITEMS_PER_STUDENT:
            continue
        alunos.append(
            {
                "aluno_id": f"idx_{idx_arr[i]}",
                "nome": id_arr[i],
                "respostas": binary,
            }
        )
        original_indices.append(idx_arr[i])

    if len(alunos) < min_calibration_size:
        return pd.DataFrame()

    # All students in a homogeneous group took the same N items. Verify.
    lens = {len(a["respostas"]) for a in alunos}
    if len(lens) > 1:
        logger.warning(
            "Group has mixed response lengths %s in area=%s — skipping", lens, area
        )
        return pd.DataFrame()

    # Run the BATCHED pipeline (~1000x faster than pipeline_cientifico for
    # large groups because the per-student EAP loop is vectorized).
    try:
        result = _pipeline_cientifico_batch(alunos, area, tabela, c_fixo=c_fixo)
    except Exception as exc:  # noqa: BLE001
        logger.warning("_pipeline_cientifico_batch failed for area=%s: %s", area, exc)
        return pd.DataFrame()

    # Map pipeline results back to the original DataFrame index.
    results_by_synth_id = {r["aluno_id"]: r for r in result["resultados_alunos"]}
    rows: list[dict] = []
    for original_idx, aluno in zip(original_indices, alunos):
        res = results_by_synth_id.get(aluno["aluno_id"])
        if res is None:
            continue
        rows.append(
            {
                "_orig_idx": original_idx,
                "theta_v3": res["theta"],
                "se_v3": res.get("se", 0.0),
                "nota_v3": res["nota_ancorada"],
                "acertos_v3": res["acertos"],
                "estado_v3": res["estado_tri"],
                "acert_dif": res.get("acert_dif", 0.0),
            }
        )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("_orig_idx")
    out.index.name = None
    return out


def calibrate_all_areas(
    df: pd.DataFrame,
    schema: MicrodataSchema,
    tabela: TabelaReferenciaTRI,
    *,
    c_fixo: float = 0.20,
    min_calibration_size: int = MIN_CALIBRATION_SIZE,
    progress_callback: Optional[callable] = None,
) -> pd.DataFrame:
    """Run V3 Local calibration across all 4 ENEM areas.

    For each area, groups the sample by the relevant CO_PROVA (and TP_LINGUA
    for LC), calibrates each group independently, and merges the V3 Local
    results back alongside the reference ``NU_NOTA_*`` columns.

    Args:
        df: Sample DataFrame from :func:`app.microdata.loader.load_microdata`.
        schema: Column accessor for the detected microdata year.
        tabela: Reference table for nota anchoring.
        c_fixo: Fixed guessing parameter (0.20 for 5 alternatives).
        min_calibration_size: Minimum group size. Defaults to 500 for
            production runs; tests pass a smaller value.
        progress_callback: Optional callable ``(area, group_idx, n_groups,
            n_students_done)`` invoked once per group for progress display.

    Returns:
        Copy of ``df`` with these new columns per area (lowercase suffix):
            ``theta_v3_{lc,ch,cn,mt}``
            ``se_v3_{lc,ch,cn,mt}``
            ``nota_v3_{lc,ch,cn,mt}``
            ``acertos_v3_{lc,ch,cn,mt}``
            ``estado_v3_{lc,ch,cn,mt}``
            ``acert_dif_{lc,ch,cn,mt}``
            ``nota_ref_{lc,ch,cn,mt}``  (parsed from NU_NOTA_*)
    """
    out = df.copy()

    # Initialize V3 columns and extract reference notas up front.
    for area in AREAS:
        al = area.lower()
        out[f"theta_v3_{al}"] = np.nan
        out[f"se_v3_{al}"] = np.nan
        out[f"nota_v3_{al}"] = np.nan
        out[f"acertos_v3_{al}"] = np.nan
        out[f"estado_v3_{al}"] = None
        out[f"acert_dif_{al}"] = np.nan
        out[f"nota_ref_{al}"] = _parse_nota_series(out[schema.nu_nota[area]])

    for area in AREAS:
        al = area.lower()
        co_prova_col = schema.co_prova[area]

        # LC has language-specific items in positions 0-4 of the response string,
        # so English and Spanish takers must be calibrated separately even when
        # they share a CO_PROVA code.
        if area == "LC" and schema.tp_lingua in df.columns:
            group_cols = [co_prova_col, schema.tp_lingua]
        else:
            group_cols = [co_prova_col]

        logger.info("Calibrating area=%s (grouped by %s)", area, group_cols)
        t0 = time.time()
        grouped = df.groupby(group_cols, dropna=False)
        n_groups_total = grouped.ngroups
        n_students_calibrated = 0
        n_groups_calibrated = 0

        for group_idx, (_group_keys, group_df) in enumerate(grouped):
            results = calibrate_area_group(
                group_df,
                schema,
                area,
                tabela,
                c_fixo=c_fixo,
                min_calibration_size=min_calibration_size,
            )
            if results.empty:
                if progress_callback is not None:
                    progress_callback(area, group_idx + 1, n_groups_total, n_students_calibrated)
                continue

            # Merge back by index.
            col_map = {
                "theta_v3": f"theta_v3_{al}",
                "se_v3": f"se_v3_{al}",
                "nota_v3": f"nota_v3_{al}",
                "acertos_v3": f"acertos_v3_{al}",
                "estado_v3": f"estado_v3_{al}",
                "acert_dif": f"acert_dif_{al}",
            }
            for src, tgt in col_map.items():
                out.loc[results.index, tgt] = results[src]
            n_students_calibrated += len(results)
            n_groups_calibrated += 1
            if progress_callback is not None:
                progress_callback(area, group_idx + 1, n_groups_total, n_students_calibrated)

        elapsed = time.time() - t0
        logger.info(
            "Area %s: %d/%d groups calibrated, %d students scored in %.1fs",
            area,
            n_groups_calibrated,
            n_groups_total,
            n_students_calibrated,
            elapsed,
        )

    return out


def calibration_summary(df: pd.DataFrame) -> dict:
    """Compute correlation / MAE / RMSE of V3 Local vs reference per area.

    Used by Phase 4 dashboards and the smoke test. Returns a dict keyed by
    lowercase area code with sub-dicts ``{n, correlation, mae, rmse,
    mean_v3, mean_ref, mean_shift}``. Only counts rows where both V3 Local
    and reference notas are present.
    """
    summary: dict = {}
    for area in AREAS:
        al = area.lower()
        v3 = df[f"nota_v3_{al}"]
        ref = df[f"nota_ref_{al}"]
        mask = v3.notna() & ref.notna()
        if mask.sum() < 2:
            summary[al] = {
                "n": int(mask.sum()),
                "correlation": None,
                "mae": None,
                "rmse": None,
                "mean_v3": None,
                "mean_ref": None,
                "mean_shift": None,
            }
            continue
        v3m = v3[mask].to_numpy()
        rm = ref[mask].to_numpy()
        diff = v3m - rm
        summary[al] = {
            "n": int(mask.sum()),
            "correlation": float(np.corrcoef(v3m, rm)[0, 1]),
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "mean_v3": float(np.mean(v3m)),
            "mean_ref": float(np.mean(rm)),
            "mean_shift": float(np.mean(diff)),
        }
    return summary
