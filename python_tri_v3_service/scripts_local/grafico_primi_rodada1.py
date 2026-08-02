"""Gera gráfico PRIMI a partir de MAT_-_rodada1.xlsx (V4 puro 2024).

Lê arquivo com Nome, Acertos, TRI (já calculados), cruza com respostas
binárias do Fredão Sim2 MT e params INEP 2024 para calcular acert_dif PRIMI.

4 painéis:
  (0,0) acertos × TRI com banda INEP 2024 sobreposta
  (0,1) acert_dif × TRI (PRIMI difficulty-weighted score)
  (1,0) histograma TRI com faixas pedagógicas
  (1,1) ac_dif × acert_dif (acerto fáceis vs difíceis — coerência)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.tabela_referencia import TabelaReferenciaTRI  # noqa: E402

# Inputs
RODADA1_PATH = Path("/Users/home/Downloads/XTRI Engine/MAT_-_rodada1.xlsx")
FREDAO_MT_PATH = Path("/Users/home/Library/CloudStorage/Dropbox/Fredao Simulados/Matemática - Simulado 2.xlsx")
PARAMS_JSON = Path(__file__).parent / "data" / "params_oficiais.json"
MT_ANULADA_IDX = 12
ANO_REFERENCIA = 2024  # MAT_rodada1 = V4 puro 2024 (bate σ·θ+μ oficial)

OUT_DIR = Path("/Users/home/Downloads/XTRI Engine")

NIVEIS_CORES = {
    "AVANÇADO": "#1B7340",
    "PROFICIENTE": "#56AE57",
    "BÁSICO": "#F5C84C",
    "ELEMENTAR": "#E58A3D",
    "INICIANTE": "#C1392B",
}


def nivel(nota: float) -> str:
    if nota >= 650: return "AVANÇADO"
    if nota >= 580: return "PROFICIENTE"
    if nota >= 480: return "BÁSICO"
    if nota >= 380: return "ELEMENTAR"
    return "INICIANTE"


def carregar_rodada1() -> pd.DataFrame:
    wb = openpyxl.load_workbook(RODADA1_PATH, data_only=True)
    ws = wb.worksheets[0]
    rows = []
    for r in range(2, ws.max_row + 1):
        nome = ws.cell(r, 1).value
        ac = ws.cell(r, 2).value
        tri = ws.cell(r, 3).value
        if nome is None:
            continue
        rows.append((str(nome), int(ac), float(tri)))
    return pd.DataFrame(rows, columns=["Nome", "Acertos", "TRI"])


def carregar_fredao_mt() -> Tuple[List[str], np.ndarray]:
    wb = openpyxl.load_workbook(FREDAO_MT_PATH, data_only=True)
    ws = wb.worksheets[0]
    ids, rows = [], []
    for r in range(2, ws.max_row + 1):
        aluno = ws.cell(r, 1).value
        if not aluno:
            continue
        ids.append(str(aluno))
        rows.append([int(ws.cell(r, c).value or 0) for c in range(2, 47)])
    return ids, np.array(rows, dtype=int)


def desenhar_zonas_pedagogicas(ax, x_min, x_max, eixo="y", show_labels=True,
                                 y_for_labels=None):
    faixas = [
        (650, 1000, "AVANÇADO"),
        (580, 650, "PROFICIENTE"),
        (480, 580, "BÁSICO"),
        (380, 480, "ELEMENTAR"),
        (0, 380, "INICIANTE"),
    ]
    for lo, hi, nv in faixas:
        if eixo == "y":
            ax.axhspan(lo, hi, alpha=0.10, color=NIVEIS_CORES[nv], zorder=0)
            if show_labels:
                ax.text(x_max * 0.98, (lo + hi) / 2, nv,
                        fontsize=8, ha="right", va="center",
                        color=NIVEIS_CORES[nv], fontweight="bold", alpha=0.8)
        else:  # eixo == "x"
            ax.axvspan(lo, hi, alpha=0.10, color=NIVEIS_CORES[nv], zorder=0)
            if show_labels and y_for_labels is not None:
                ax.text((lo + hi) / 2, y_for_labels, nv,
                        fontsize=8, ha="center", va="top",
                        color=NIVEIS_CORES[nv], fontweight="bold", alpha=0.85)


def main():
    print("Gráfico PRIMI — MAT_rodada1.xlsx")
    print("=" * 60)

    df_r1 = carregar_rodada1()
    print(f"  rodada1: {len(df_r1)} alunos")

    ids_fredao, mat_full = carregar_fredao_mt()
    keep = [j for j in range(mat_full.shape[1]) if j != MT_ANULADA_IDX]
    mat = mat_full[:, keep]
    print(f"  Fredão MT: {len(ids_fredao)} alunos, {mat.shape[1]} itens válidos")

    # Cruzar
    fredao_idx_by_id = {aid: i for i, aid in enumerate(ids_fredao)}
    df_r1["fredao_idx"] = df_r1["Nome"].map(fredao_idx_by_id)
    missing = df_r1[df_r1.fredao_idx.isna()]
    if len(missing) > 0:
        print(f"  AVISO: {len(missing)} alunos da rodada1 não encontrados no Fredão")
    df_r1 = df_r1.dropna(subset=["fredao_idx"]).copy()
    df_r1["fredao_idx"] = df_r1["fredao_idx"].astype(int)
    print(f"  Cruzado: {len(df_r1)} alunos")

    # Params INEP 2024 (CO_PROVA 1323 AZUL regular)
    with open(PARAMS_JSON) as f:
        params_of = json.load(f)
    params_2024 = [{"a": float(p[1]), "b": float(p[2]), "c": float(p[3])}
                    for p in params_of[str(ANO_REFERENCIA)]["MT"]]
    params_2024 = [params_2024[j] for j in keep]
    print(f"  Params INEP {ANO_REFERENCIA}: {len(params_2024)} itens "
          f"(b μ={np.mean([p['b'] for p in params_2024]):.2f})")

    # Classificação PRIMI: b<-1.5 fáceis, b>1.5 difíceis
    idx_f = [j for j, p in enumerate(params_2024) if p["b"] < -1.5]
    idx_m = [j for j, p in enumerate(params_2024) if -1.5 <= p["b"] <= 1.5]
    idx_d = [j for j, p in enumerate(params_2024) if p["b"] > 1.5]
    print(f"  PRIMI: {len(idx_f)} fáceis + {len(idx_m)} médios + {len(idx_d)} difíceis (b>1.5)")

    # Calcular acert_dif e acert_fac para cada aluno
    acert_dif_arr = np.zeros(len(df_r1))
    acert_fac_arr = np.zeros(len(df_r1))
    acert_med_arr = np.zeros(len(df_r1))
    for i, fi in enumerate(df_r1["fredao_idx"].values):
        respostas = mat[fi]
        ad = respostas[idx_d].sum() if idx_d else 0
        af = respostas[idx_f].sum() if idx_f else 0
        am = respostas[idx_m].sum() if idx_m else 0
        acert_dif_arr[i] = ad / max(1, len(idx_d))
        acert_fac_arr[i] = af / max(1, len(idx_f)) if idx_f else 0.0
        acert_med_arr[i] = am / max(1, len(idx_m))

    df_r1["acert_dif"] = acert_dif_arr
    df_r1["acert_med"] = acert_med_arr
    df_r1["nivel"] = df_r1["TRI"].apply(nivel)

    print(f"\n  Acertos μ={df_r1['Acertos'].mean():.1f}")
    print(f"  TRI μ={df_r1['TRI'].mean():.1f} σ={df_r1['TRI'].std():.1f}")
    print(f"  acert_dif μ={df_r1['acert_dif'].mean():.3f}")

    # Stats por nível
    print("\n  Por nível pedagógico:")
    for nv in ["AVANÇADO", "PROFICIENTE", "BÁSICO", "ELEMENTAR", "INICIANTE"]:
        sub = df_r1[df_r1["nivel"] == nv]
        if len(sub) > 0:
            print(f"    {nv:12s} n={len(sub):4d}  TRI μ={sub['TRI'].mean():6.1f}  "
                  f"acert_dif μ={sub['acert_dif'].mean():.3f}")

    # ── Gráfico 4 painéis ──
    tabelas = TabelaReferenciaTRI.carregar_multiano()
    n_itens = mat.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=150)

    # (0,0) acertos × TRI com banda INEP 2024
    ax = axes[0, 0]
    ac_axis = np.arange(0, n_itens + 1)
    tri_min = np.array([tabelas[ANO_REFERENCIA].obter("MT", a)["tri_min"] for a in ac_axis])
    tri_med = np.array([tabelas[ANO_REFERENCIA].obter("MT", a)["tri_med"] for a in ac_axis])
    tri_max_band = np.array([tabelas[ANO_REFERENCIA].obter("MT", a)["tri_max"] for a in ac_axis])
    ax.fill_between(ac_axis, tri_min, tri_max_band, alpha=0.18, color="#888",
                     label=f"Banda INEP {ANO_REFERENCIA} (tri_min..tri_max)", zorder=1)
    ax.plot(ac_axis, tri_med, "--", color="#444", linewidth=1.5,
             label=f"Mediana INEP {ANO_REFERENCIA}", zorder=2)
    desenhar_zonas_pedagogicas(ax, 0, n_itens, eixo="y")
    cores = [NIVEIS_CORES[n] for n in df_r1["nivel"]]
    ax.scatter(df_r1["Acertos"], df_r1["TRI"], s=22, c=cores,
                edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
    ax.set_xlim(-0.5, n_itens + 0.5)
    ax.set_ylim(200, 1000)
    ax.set_xlabel(f"Acertos MT (0 a {n_itens})")
    ax.set_ylabel("Nota TRI V4 INEP 2024")
    ax.set_title("Acertos × TRI com banda de referência INEP 2024",
                  fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.25, zorder=0)

    # (0,1) acert_dif PRIMI × TRI
    ax = axes[0, 1]
    desenhar_zonas_pedagogicas(ax, 0, 1, eixo="y")
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.005, 0.005, size=len(df_r1))
    ax.scatter(df_r1["acert_dif"] + jitter, df_r1["TRI"], s=22, c=cores,
                edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
    coef = np.polyfit(df_r1["acert_dif"], df_r1["TRI"], 1)
    xr = np.linspace(0, 1, 50)
    ax.plot(xr, np.polyval(coef, xr), "--", color="#222", linewidth=1.4,
             alpha=0.75, label=f"Tendência (β={coef[0]:.0f} TRI/100% difíceis)", zorder=2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(200, 1000)
    ax.set_xlabel(f"acert_dif: % itens DIFÍCEIS acertados (b>1.5 INEP)\n"
                   f"{len(idx_d)} itens classificados como difíceis")
    ax.set_ylabel("Nota TRI V4 INEP 2024")
    ax.set_title("PRIMI — acert_dif × TRI (coerência pedagógica)",
                  fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.25, zorder=0)

    # (1,0) histograma TRI por faixa pedagógica
    ax = axes[1, 0]
    bins = np.linspace(200, 1000, 41)  # 20pt por bin
    # Construir histograma empilhado por nível pra cores corretas
    hist_data = []
    hist_labels = []
    hist_cores = []
    for nv in ["INICIANTE", "ELEMENTAR", "BÁSICO", "PROFICIENTE", "AVANÇADO"]:
        sub = df_r1[df_r1["nivel"] == nv]
        if len(sub) > 0:
            hist_data.append(sub["TRI"].values)
            hist_labels.append(f"{nv} ({len(sub)})")
            hist_cores.append(NIVEIS_CORES[nv])
    n_vals, _, _ = ax.hist(hist_data, bins=bins, color=hist_cores, stacked=True,
                            edgecolor="white", linewidth=0.5, label=hist_labels, alpha=0.95)
    ymax = float(np.max([np.max(h) for h in n_vals])) * 1.15
    # Pintar zonas pedagógicas no EIXO X (nota) com Y dinâmico
    desenhar_zonas_pedagogicas(ax, 0, 0, eixo="x", show_labels=True, y_for_labels=ymax * 0.97)
    # Marcadores μ e mediana
    media = df_r1["TRI"].mean()
    mediana = df_r1["TRI"].median()
    ax.axvline(media, color="black", linestyle="-", linewidth=1.4, alpha=0.7,
                label=f"μ = {media:.1f}")
    ax.axvline(mediana, color="black", linestyle="--", linewidth=1.2, alpha=0.6,
                label=f"mediana = {mediana:.1f}")
    ax.set_xlim(200, 1020)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Nota TRI V4 INEP 2024")
    ax.set_ylabel("Nº alunos")
    ax.set_title(f"Distribuição TRI — μ={media:.1f}  σ={df_r1['TRI'].std():.1f}  "
                  f"mediana={mediana:.1f}",
                  fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.grid(True, alpha=0.25, axis="y")

    # (1,1) acert_med × acert_dif (coerência fáceis vs difíceis)
    ax = axes[1, 1]
    ax.scatter(df_r1["acert_med"], df_r1["acert_dif"], s=22, c=cores,
                edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
    ax.plot([0, 1], [0, 1], "--", color="#999", alpha=0.6, label="Linha 45° (acerto proporcional)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(f"acert_med: % itens MÉDIOS acertados (-1.5 ≤ b ≤ 1.5)\n"
                   f"{len(idx_m)} itens médios INEP 2024")
    ax.set_ylabel(f"acert_dif: % itens DIFÍCEIS acertados (b>1.5)\n"
                   f"{len(idx_d)} itens difíceis INEP 2024")
    corr = df_r1[["acert_med", "acert_dif"]].corr().iloc[0, 1]
    ax.set_title(f"PRIMI — coerência itens médios × difíceis (ρ={corr:.3f})",
                  fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.suptitle(
        f"MAT rodada1 — análise PRIMI — V4 INEP OFICIAL 2024 (CO_PROVA AZUL 1323)\n"
        f"{len(df_r1)} alunos | 44 itens válidos | acertos μ={df_r1['Acertos'].mean():.1f} | "
        f"TRI μ={df_r1['TRI'].mean():.1f}",
        fontsize=14, fontweight="bold", y=1.00,
    )
    plt.tight_layout()

    out_png = OUT_DIR / "MAT_rodada1_PRIMI.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  PNG: {out_png}")


if __name__ == "__main__":
    main()
