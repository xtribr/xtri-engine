"""Tabela MT rodada1 no estilo INEP: Acerto | Freq | Min | Med | Max.

Para cada nº de acertos observado, mostra:
- Freq: quantos alunos com esse nº de acertos
- Min: menor TRI dentre eles
- Med: TRI mediana
- Max: maior TRI

Renderiza em 3 sub-tabelas (0-15, 16-30, 31-45) lado a lado, igual ao INEP.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd

RODADA1_PATH = Path("/Users/home/Downloads/XTRI Engine/MAT_-_rodada1.xlsx")
OUT_DIR = Path("/Users/home/Downloads/XTRI Engine")

# Cores INEP-style
VERDE_HEADER = "#7FC97F"  # verde médio do cabeçalho
VERDE_ZEBRA = "#E8F5E1"   # verde clarinho da zebra
BRANCO = "#FFFFFF"

N_ITENS = 44  # MT Fredão tem 44 itens válidos (Q148 anulada excluída)


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


def agregar_por_acertos(df: pd.DataFrame, max_acertos: int) -> List[Tuple]:
    """Retorna lista de (acertos, freq, min, med, max). Acertos sem aluno: tudo zerado."""
    grouped = df.groupby("Acertos")["TRI"].agg(
        freq="count", v_min="min", v_med="median", v_max="max",
    ).reset_index()
    out = []
    for k in range(0, max_acertos + 1):
        row = grouped[grouped["Acertos"] == k]
        if len(row) == 0:
            out.append((k, 0, None, None, None))
        else:
            r = row.iloc[0]
            out.append((k, int(r["freq"]), float(r["v_min"]),
                        float(r["v_med"]), float(r["v_max"])))
    return out


def render_subtabela(ax, dados: List[Tuple], col_start_idx: int):
    """Renderiza uma sub-tabela em um eixo matplotlib."""
    ax.axis("off")
    headers = ["Acerto", "Freq", "Min", "Med", "Max"]
    n_rows = len(dados)
    n_cols = 5

    # Larguras relativas das colunas
    col_widths = [0.16, 0.22, 0.20, 0.20, 0.22]
    x_positions = [0.0]
    for w in col_widths:
        x_positions.append(x_positions[-1] + w)

    row_height = 1.0 / (n_rows + 1)

    # Cabeçalho
    y_top = 1.0 - row_height
    for c, head in enumerate(headers):
        x = x_positions[c]
        ax.add_patch(plt.Rectangle((x, y_top), col_widths[c], row_height,
                                    facecolor=VERDE_HEADER, edgecolor="white",
                                    linewidth=1.5))
        ax.text(x + col_widths[c] / 2, y_top + row_height / 2, head,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="#1a3a1a")

    # Linhas
    for i, (ac, freq, vmin, vmed, vmax) in enumerate(dados):
        y = y_top - (i + 1) * row_height
        # zebra
        zebra = (i % 2 == 0)
        bg = VERDE_ZEBRA if zebra else BRANCO

        for c in range(n_cols):
            x = x_positions[c]
            ax.add_patch(plt.Rectangle((x, y), col_widths[c], row_height,
                                        facecolor=bg, edgecolor="white",
                                        linewidth=0.8))

        # Conteúdo
        vals = [
            str(ac),
            str(freq) if freq > 0 else "—",
            f"{vmin:.1f}" if vmin is not None else "—",
            f"{vmed:.1f}" if vmed is not None else "—",
            f"{vmax:.1f}" if vmax is not None else "—",
        ]
        for c, v in enumerate(vals):
            x = x_positions[c]
            # Coluna Acerto centralizada, demais ligeiramente à direita
            ha = "center" if c == 0 else "right"
            x_text = x + col_widths[c] * (0.5 if ha == "center" else 0.92)
            ax.text(x_text, y + row_height / 2, v,
                    ha=ha, va="center", fontsize=9.5,
                    color="#222" if freq > 0 else "#888")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def main():
    print("Tabela MT rodada1 estilo INEP")
    print("=" * 60)

    df = carregar_rodada1()
    print(f"  Alunos: {len(df)}")
    print(f"  Acertos μ={df['Acertos'].mean():.1f}, range=[{df['Acertos'].min()}, {df['Acertos'].max()}]")
    print(f"  TRI μ={df['TRI'].mean():.1f} σ={df['TRI'].std():.1f}")

    # Agregar por acertos (vai de 0 até N_ITENS)
    dados = agregar_por_acertos(df, N_ITENS)
    print(f"\n  Acertos com pelo menos 1 aluno: "
          f"{sum(1 for d in dados if d[1] > 0)}/{N_ITENS + 1}")

    # Distribuir em 3 sub-tabelas
    # 0-14 (15), 15-29 (15), 30-44 (15) → 15 linhas cada
    grupos = [dados[0:15], dados[15:30], dados[30:45]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=180,
                              gridspec_kw={"wspace": 0.05})
    fig.patch.set_facecolor("#F5F5F5")

    for ax, sub in zip(axes, grupos):
        render_subtabela(ax, sub, 0)

    # Título e info
    fig.suptitle("MATEMÁTICA — Fredão Simulado 2 (rodada 1)",
                  fontsize=18, fontweight="bold", y=0.96)
    info = (f"V4 INEP OFICIAL 2024 (CO_PROVA AZUL 1323) | "
            f"{len(df)} alunos | 44 itens válidos (Q148 anulada excluída) | "
            f"TRI μ={df['TRI'].mean():.1f} σ={df['TRI'].std():.1f} | "
            f"Acertos μ={df['Acertos'].mean():.1f}")
    fig.text(0.5, 0.92, info, ha="center", fontsize=10, color="#444")

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])

    out_png = OUT_DIR / "MAT_rodada1_tabela_INEP_style.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="#F5F5F5")
    plt.close()
    print(f"\n  PNG: {out_png}")

    # CSV também pra facilitar processamento
    df_out = pd.DataFrame(dados, columns=["Acerto", "Freq", "Min", "Med", "Max"])
    csv_path = OUT_DIR / "MAT_rodada1_tabela_INEP_style.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    # Print resumo
    print("\n  Top acertos (com alunos):")
    for d in dados[-10:]:
        if d[1] > 0:
            print(f"    {d[0]:2d} acertos: freq={d[1]:3d}  min={d[2]:.1f}  "
                  f"med={d[3]:.1f}  max={d[4]:.1f}")


if __name__ == "__main__":
    main()
