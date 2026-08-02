"""Fredão Sim2 — MT relatório 4 colunas — V4 CONTÍNUO (sem saturação).

FIX do bug "mesmos acertos → mesma TRI": substitui ancoragem por tabela
(que satura no tri_max da faixa de acertos) por transformação contínua
σ·θ+μ derivada da Test Characteristic Curve (TCC) INEP.

Por ano:
- 2024: usa σ, μ OFICIAIS INEP da enem_transformacoes (já calibrados em 5K alunos).
- 2017: deriva σ, μ via inversão da TCC sobre params INEP 2017 +
  regressão linear contra tri_med da tabela INEP 2017.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.irt.scoring import eap_theta  # noqa: E402
from app.tabela_referencia import (  # noqa: E402
    TRI_MAXIMA_OFICIAL, TRI_MINIMA_OFICIAL, TabelaReferenciaTRI,
)

MT_PATH = Path("/Users/home/Library/CloudStorage/Dropbox/Fredao Simulados/Matemática - Simulado 2.xlsx")
PARAMS_JSON = Path(__file__).parent / "data" / "params_oficiais.json"
MT_ANULADA_IDX = 12  # Q148

# σ, μ OFICIAL INEP 2024 (enem_transformacoes, 5000 alunos do microdata real)
SIGMA_MT_2024_OFICIAL, MU_MT_2024_OFICIAL = 129.6484, 500.0186

OUT_DIR = Path("/Users/home/Downloads/XTRI Engine")


# ────────────────────────────────────────────────────────────────────────────
# Test Characteristic Curve (TCC) — invertendo para derivar σ, μ de 2017
# ────────────────────────────────────────────────────────────────────────────

def prob_acerto_3pl(theta: float, a: float, b: float, c: float) -> float:
    """P(acerto | θ) no modelo 3PL."""
    return c + (1.0 - c) / (1.0 + np.exp(-a * (theta - b)))


def tcc_acertos_esperados(theta: float, params: List[Dict]) -> float:
    """E[acertos | θ] somando P para todos os itens."""
    return float(sum(prob_acerto_3pl(theta, p["a"], p["b"], p["c"]) for p in params))


def inverter_tcc(k_target: float, params: List[Dict],
                 theta_min: float = -5.0, theta_max: float = 5.0,
                 n_iter: int = 80) -> float:
    """Bisseção para encontrar θ tal que TCC(θ) = k."""
    lo, hi = theta_min, theta_max
    # Garantir que k_target está no domínio
    e_lo = tcc_acertos_esperados(lo, params)
    e_hi = tcc_acertos_esperados(hi, params)
    if k_target <= e_lo:
        return lo
    if k_target >= e_hi:
        return hi
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        if tcc_acertos_esperados(mid, params) < k_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def derivar_sigma_mu_via_tcc(
    params: List[Dict],
    tabela: TabelaReferenciaTRI,
    area: str,
    n_itens: int,
    trim_extremos: bool = True,
) -> Tuple[float, float, Dict]:
    """Regressão tri_med[k] = σ·θ_k + μ onde TCC(θ_k) = k.

    Por padrão usa apenas k ∈ [n_itens*0.2, n_itens*0.8] (centro da curva TCC)
    para minimizar viés do achatamento sigmoidal nos extremos.
    """
    theta_k_list, nota_k_list, k_list = [], [], []
    for k in range(0, n_itens + 1):
        theta_k = inverter_tcc(k, params)
        # Pular saturações da TCC nos extremos
        if -4.99 < theta_k < 4.99:
            theta_k_list.append(theta_k)
            nota_k_list.append(tabela.obter(area, k)["tri_med"])
            k_list.append(k)
    theta_arr = np.array(theta_k_list)
    nota_arr = np.array(nota_k_list)
    k_arr = np.array(k_list)

    if trim_extremos:
        # Recortar para o centro da curva (onde a relação θ ↔ nota é linear)
        lo_k = int(np.percentile(k_arr, 20))
        hi_k = int(np.percentile(k_arr, 80))
        mask = (k_arr >= lo_k) & (k_arr <= hi_k)
        theta_fit = theta_arr[mask]
        nota_fit = nota_arr[mask]
    else:
        theta_fit = theta_arr
        nota_fit = nota_arr

    sigma = float(np.cov(theta_fit, nota_fit, bias=True)[0, 1] / np.var(theta_fit))
    mu = float(np.mean(nota_fit) - sigma * np.mean(theta_fit))
    debug = {
        "theta_range_full": (theta_arr.min(), theta_arr.max()),
        "theta_range_fit": (theta_fit.min(), theta_fit.max()),
        "nota_range_fit": (nota_fit.min(), nota_fit.max()),
        "n_pontos_total": len(theta_arr),
        "n_pontos_fit": len(theta_fit),
        "k_range_fit": (k_arr[mask][0] if trim_extremos else 0,
                         k_arr[mask][-1] if trim_extremos else n_itens),
        "method": "TCC_inversion + linear_regression (centro)" if trim_extremos
                  else "TCC_inversion + linear_regression (full)",
    }
    return sigma, mu, debug


# ────────────────────────────────────────────────────────────────────────────
# Pipeline
# ────────────────────────────────────────────────────────────────────────────

def carregar_mt() -> Tuple[List[str], np.ndarray]:
    wb = openpyxl.load_workbook(MT_PATH, data_only=True)
    ws = wb.worksheets[0]
    ids, rows = [], []
    for r in range(2, ws.max_row + 1):
        aluno = ws.cell(r, 1).value
        if not aluno:
            continue
        ids.append(str(aluno))
        rows.append([int(ws.cell(r, c).value or 0) for c in range(2, 47)])
    return ids, np.array(rows, dtype=int)


def params_para_dicts(plist: List[List]) -> List[Dict]:
    return [{"a": float(p[1]), "b": float(p[2]), "c": float(p[3])} for p in plist]


def estimar_thetas(resp: np.ndarray, params: List[Dict]) -> np.ndarray:
    n = len(resp)
    thetas = np.zeros(n)
    for i, row in enumerate(resp):
        t, _ = eap_theta(row, params)
        thetas[i] = t
    return thetas


def main():
    t0 = time.time()
    print("FREDÃO Sim2 — MT relatório SIMPLES V2 (V4 contínuo via TCC)")
    print("=" * 70)

    ids, mat_full = carregar_mt()
    keep = [j for j in range(mat_full.shape[1]) if j != MT_ANULADA_IDX]
    mat = mat_full[:, keep]
    acertos = mat.sum(axis=1).astype(int)
    print(f"  Alunos: {len(ids)} | itens válidos: {mat.shape[1]} (Q148 excluída)")

    with open(PARAMS_JSON) as f:
        params_of = json.load(f)

    params_2017_full = params_para_dicts(params_of["2017"]["MT"])
    params_2024_full = params_para_dicts(params_of["2024"]["MT"])
    params_2017 = [params_2017_full[j] for j in keep]
    params_2024 = [params_2024_full[j] for j in keep]

    tabelas = TabelaReferenciaTRI.carregar_multiano()
    n_itens = mat.shape[1]
    tri_minima = TRI_MINIMA_OFICIAL["MT"]
    tri_maxima = TRI_MAXIMA_OFICIAL["MT"]

    # ── 2017: derivar σ, μ via TCC + regressão ──
    sigma_2017, mu_2017, dbg_2017 = derivar_sigma_mu_via_tcc(
        params_2017, tabelas[2017], "MT", n_itens,
    )
    print(f"\n  2017 σ, μ derivados via TCC:")
    print(f"    σ = {sigma_2017:.4f}, μ = {mu_2017:.4f}")
    print(f"    k_range_fit: {dbg_2017['k_range_fit']} (n={dbg_2017['n_pontos_fit']}/{dbg_2017['n_pontos_total']})")
    print(f"    θ range fit [{dbg_2017['theta_range_fit'][0]:.2f}, {dbg_2017['theta_range_fit'][1]:.2f}]")
    print(f"    nota range fit [{dbg_2017['nota_range_fit'][0]:.1f}, {dbg_2017['nota_range_fit'][1]:.1f}]")

    # ── 2024: SANITY CHECK derivado vs oficial ──
    sigma_2024_derivado, mu_2024_derivado, dbg_2024 = derivar_sigma_mu_via_tcc(
        params_2024, tabelas[2024], "MT", n_itens,
    )
    print(f"\n  2024 σ, μ via TCC (sanity check):")
    print(f"    Derivado: σ = {sigma_2024_derivado:.4f}, μ = {mu_2024_derivado:.4f}")
    print(f"    Oficial:  σ = {SIGMA_MT_2024_OFICIAL}, μ = {MU_MT_2024_OFICIAL}")
    print(f"    DIFF: σ Δ={sigma_2024_derivado - SIGMA_MT_2024_OFICIAL:+.2f}, "
          f"μ Δ={mu_2024_derivado - MU_MT_2024_OFICIAL:+.2f}")

    # ── Estimar θs ──
    print("\n  Estimando θ com params INEP 2017...")
    thetas_2017 = estimar_thetas(mat, params_2017)
    print(f"    θ μ={thetas_2017.mean():+.3f} σ={thetas_2017.std():.3f} "
          f"min={thetas_2017.min():+.2f} max={thetas_2017.max():+.2f}")

    print("  Estimando θ com params INEP 2024...")
    thetas_2024 = estimar_thetas(mat, params_2024)
    print(f"    θ μ={thetas_2024.mean():+.3f} σ={thetas_2024.std():.3f} "
          f"min={thetas_2024.min():+.2f} max={thetas_2024.max():+.2f}")

    # ── Aplicar σ·θ+μ ──
    notas_2017 = np.clip(sigma_2017 * thetas_2017 + mu_2017, tri_minima, tri_maxima)
    notas_2024 = np.clip(
        SIGMA_MT_2024_OFICIAL * thetas_2024 + MU_MT_2024_OFICIAL, tri_minima, tri_maxima,
    )

    print(f"\n  Notas TRI 2017: μ={notas_2017.mean():.1f} σ={notas_2017.std():.1f} "
          f"min={notas_2017.min():.0f} max={notas_2017.max():.0f}")
    print(f"  Notas TRI 2024: μ={notas_2024.mean():.1f} σ={notas_2024.std():.1f} "
          f"min={notas_2024.min():.0f} max={notas_2024.max():.0f}")

    # ── Sanity: alunos com mesmo nº de acertos devem ter notas DIFERENTES ──
    print("\n  Sanity check — distinct notas para mesmo acertos:")
    df_check = pd.DataFrame({"acertos": acertos, "TRI 2017": notas_2017, "TRI 2024": notas_2024})
    for k in [40, 41, 42, 43]:
        sub = df_check[df_check.acertos == k]
        if len(sub) > 1:
            n_distinct_17 = sub["TRI 2017"].nunique()
            n_distinct_24 = sub["TRI 2024"].nunique()
            print(f"    {k} acertos ({len(sub)} alunos): "
                  f"TRI 2017 distintas={n_distinct_17}, TRI 2024 distintas={n_distinct_24}")

    # ── DataFrame final ──
    df = pd.DataFrame({
        "Nome": ids,
        "Acertos": acertos,
        "TRI 2017": np.round(notas_2017, 1),
        "TRI 2024": np.round(notas_2024, 1),
    })
    df = df.sort_values("TRI 2024", ascending=False).reset_index(drop=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    xlsx_path = OUT_DIR / f"xtri_FREDAO_sim2_MT_simples_{ts}.xlsx"
    df.to_excel(xlsx_path, sheet_name="MT", index=False)

    # Formatação
    wb = openpyxl.load_workbook(xlsx_path)
    from openpyxl.styles import Font, Alignment, PatternFill
    ws = wb["MT"]
    bold = Font(bold=True, color="FFFFFF")
    fill = PatternFill(start_color="2E86AB", end_color="2E86AB", fill_type="solid")
    centro = Alignment(horizontal="center", vertical="center")
    for cell in ws[1]:
        cell.font = bold
        cell.alignment = centro
        cell.fill = fill
    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["B"].width = 12
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 14
    for r in range(2, ws.max_row + 1):
        for c in range(1, 5):
            ws.cell(r, c).alignment = centro
    ws.freeze_panes = "A2"
    wb.save(xlsx_path)

    print(f"\n  XLSX: {xlsx_path}")
    print(f"\nTOTAL: {time.time()-t0:.1f}s")

    print("\n── Top 8 ──")
    print(df.head(8).to_string(index=False))
    print("\n── Bottom 5 ──")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
