"""Fredão Sim2 — análise CONSOLIDADA CN + MT × escalas 2017 e 2024.

Reutiliza a mesma calibração local de CN (45 itens) e MT (44 itens, Q148 excluída),
ancora nas escalas pedagógicas 2017 e 2024, e gera:
- 1 Excel com 9 abas (resumo, alunos consolidado, por nível por ano, migração, atípicos, gráficos)
- 2 PNGs (um por ano) com 4 painéis: hist CN, hist MT, hist média, scatter CN×MT
"""

from __future__ import annotations

import sys
import time
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

from app.irt.scoring import eap_theta  # noqa: E402
from app.irt.transform import ancorar_notas  # noqa: E402
from app.irt.unidim_em import calibrar_itens_mml_em  # noqa: E402
from app.tabela_referencia import TabelaReferenciaTRI  # noqa: E402

DROPBOX = Path("/Users/home/Library/CloudStorage/Dropbox/Fredao Simulados")
CN_PATH = DROPBOX / "Natureza - Simulado 2.xlsx"
MT_PATH = DROPBOX / "Matemática - Simulado 2.xlsx"

MT_ANULADA_IDX = 12
ANOS = [2017, 2024]

OUT_DIR = Path("/Users/home/Downloads/XTRI Engine")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def carregar(path: Path) -> Tuple[List[str], np.ndarray]:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    ids = []
    rows = []
    for r in range(2, ws.max_row + 1):
        aluno = ws.cell(r, 1).value
        if not aluno:
            continue
        ids.append(str(aluno))
        rows.append([int(ws.cell(r, c).value or 0) for c in range(2, 47)])
    return ids, np.array(rows, dtype=int)


def nivel_pedagogico(nota: float) -> str:
    if nota >= 650:
        return "AVANÇADO"
    if nota >= 580:
        return "PROFICIENTE"
    if nota >= 480:
        return "BÁSICO"
    if nota >= 380:
        return "ELEMENTAR"
    return "INICIANTE"


def calibrar_estimar(area: str, mat: np.ndarray) -> Dict:
    """Calibra + estima θ. Retorna dict com arrays."""
    print(f"  [{area}] calibrando...")
    t0 = time.time()
    params = calibrar_itens_mml_em(mat, c=0.20, n_quad=61)
    a_vals = [p["a"] for p in params]
    b_vals = [p["b"] for p in params]
    n_alunos = mat.shape[0]
    thetas = np.zeros(n_alunos)
    ses = np.zeros(n_alunos)
    acertos = mat.sum(axis=1)
    thetas_list = []
    for i, row in enumerate(mat):
        t, se = eap_theta(row, params)
        thetas[i] = t
        ses[i] = se
        thetas_list.append({
            "aluno_id": f"row_{i}",
            "nome": f"row_{i}",
            "acertos": int(acertos[i]),
            "theta": float(t),
            "se": float(se),
            "acertos_faceis": 0,
            "acertos_medios": 0,
            "acertos_dificeis": 0,
            "acert_dif": 0.0,
        })
    print(f"  [{area}] done in {time.time()-t0:.1f}s | "
          f"a={np.mean(a_vals):.2f}±{np.std(a_vals):.2f}, "
          f"b={np.mean(b_vals):.2f}±{np.std(b_vals):.2f}, "
          f"θ={thetas.mean():+.3f}±{thetas.std():.3f}")
    return {
        "params": params, "thetas": thetas, "ses": ses,
        "acertos": acertos, "thetas_list": thetas_list,
        "n_itens": mat.shape[1],
    }


def ancorar(area: str, thetas_list: List[Dict], tab: TabelaReferenciaTRI) -> np.ndarray:
    res = ancorar_notas([dict(t) for t in thetas_list], tab, area)
    return np.array([r["nota_ancorada"] for r in res])


def gerar_grafico_consolidado(df: pd.DataFrame, ano: int, png_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    bins = np.linspace(0, 1000, 51)

    cn_col = f"NOTA CN {ano}"
    mt_col = f"NOTA MT {ano}"
    med_col = f"MÉDIA {ano}"

    # Histograma CN
    ax = axes[0, 0]
    ax.hist(df[cn_col], bins=bins, color="#2E86AB", edgecolor="white", alpha=0.85)
    ax.set_title(f"CN — μ={df[cn_col].mean():.1f}  σ={df[cn_col].std():.1f}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Nota CN")
    ax.set_ylabel("Nº alunos")
    for lim, lab in [(650, "AV"), (580, "PR"), (480, "BA"), (380, "EL")]:
        ax.axvline(lim, color="gray", linestyle="--", alpha=0.5)
        ax.text(lim, ax.get_ylim()[1] * 0.95, lab, fontsize=8,
                ha="center", color="gray")

    # Histograma MT
    ax = axes[0, 1]
    ax.hist(df[mt_col], bins=bins, color="#A23B72", edgecolor="white", alpha=0.85)
    ax.set_title(f"MT — μ={df[mt_col].mean():.1f}  σ={df[mt_col].std():.1f}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Nota MT")
    ax.set_ylabel("Nº alunos")
    for lim, lab in [(650, "AV"), (580, "PR"), (480, "BA"), (380, "EL")]:
        ax.axvline(lim, color="gray", linestyle="--", alpha=0.5)
        ax.text(lim, ax.get_ylim()[1] * 0.95, lab, fontsize=8,
                ha="center", color="gray")

    # Histograma média CN+MT
    ax = axes[1, 0]
    ax.hist(df[med_col], bins=bins, color="#F18F01", edgecolor="white", alpha=0.85)
    ax.set_title(f"MÉDIA CN+MT — μ={df[med_col].mean():.1f}  σ={df[med_col].std():.1f}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Nota média")
    ax.set_ylabel("Nº alunos")
    for lim, lab in [(650, "AV"), (580, "PR"), (480, "BA"), (380, "EL")]:
        ax.axvline(lim, color="gray", linestyle="--", alpha=0.5)
        ax.text(lim, ax.get_ylim()[1] * 0.95, lab, fontsize=8,
                ha="center", color="gray")

    # Scatter CN × MT
    ax = axes[1, 1]
    ax.scatter(df[cn_col], df[mt_col], s=18, alpha=0.55,
               color="#264653", edgecolor="white", linewidth=0.4)
    ax.plot([0, 1000], [0, 1000], "--", color="gray", linewidth=1, alpha=0.7)
    lim_min = min(df[cn_col].min(), df[mt_col].min()) - 20
    lim_max = max(df[cn_col].max(), df[mt_col].max()) + 20
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Nota CN")
    ax.set_ylabel("Nota MT")
    corr = df[[cn_col, mt_col]].corr().iloc[0, 1]
    ax.set_title(f"CN × MT — ρ={corr:.3f}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"FREDÃO Simulado 2 — análise consolidada — escala pedagógica {ano}",
                 fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    t0 = time.time()
    print("FREDÃO Sim2 — CONSOLIDADO CN + MT × escalas 2017/2024")
    print("=" * 70)

    # === CN ===
    print("\n=== CN ===")
    ids_cn, mat_cn = carregar(CN_PATH)
    print(f"  Alunos: {len(ids_cn)} | itens: {mat_cn.shape[1]}")
    cn = calibrar_estimar("CN", mat_cn)

    # === MT ===
    print("\n=== MT ===")
    ids_mt, mat_mt_full = carregar(MT_PATH)
    keep = [j for j in range(mat_mt_full.shape[1]) if j != MT_ANULADA_IDX]
    mat_mt = mat_mt_full[:, keep]
    print(f"  Alunos: {len(ids_mt)} | itens válidos: {mat_mt.shape[1]} (Q148 excluída)")
    mt = calibrar_estimar("MT", mat_mt)

    # Verifica que os IDs batem
    if ids_cn != ids_mt:
        raise RuntimeError("IDs CN e MT não coincidem")
    ids = ids_cn

    # === Ancoragem por ano ===
    tabelas = TabelaReferenciaTRI.carregar_multiano()
    notas_cn = {ano: ancorar("CN", cn["thetas_list"], tabelas[ano]) for ano in ANOS}
    notas_mt = {ano: ancorar("MT", mt["thetas_list"], tabelas[ano]) for ano in ANOS}

    for ano in ANOS:
        print(f"\n  Escala {ano}:")
        print(f"    CN μ={notas_cn[ano].mean():.1f} σ={notas_cn[ano].std():.1f}")
        print(f"    MT μ={notas_mt[ano].mean():.1f} σ={notas_mt[ano].std():.1f}")
        media = (notas_cn[ano] + notas_mt[ano]) / 2
        print(f"    MÉDIA μ={media.mean():.1f} σ={media.std():.1f}")

    # === DataFrame consolidado ===
    rows = []
    for i, aluno_id in enumerate(ids):
        row = {
            "ID": aluno_id,
            "ACERTOS CN": int(cn["acertos"][i]),
            "θ CN": round(float(cn["thetas"][i]), 3),
            "SE θ CN": round(float(cn["ses"][i]), 3),
            "ACERTOS MT": int(mt["acertos"][i]),
            "θ MT": round(float(mt["thetas"][i]), 3),
            "SE θ MT": round(float(mt["ses"][i]), 3),
        }
        for ano in ANOS:
            ncn = float(notas_cn[ano][i])
            nmt = float(notas_mt[ano][i])
            med = (ncn + nmt) / 2
            row[f"NOTA CN {ano}"] = round(ncn, 1)
            row[f"NOTA MT {ano}"] = round(nmt, 1)
            row[f"MÉDIA {ano}"] = round(med, 1)
            row[f"NÍVEL CN {ano}"] = nivel_pedagogico(ncn)
            row[f"NÍVEL MT {ano}"] = nivel_pedagogico(nmt)
            row[f"NÍVEL MÉDIO {ano}"] = nivel_pedagogico(med)
        row["DIFF MÉDIA 2024-2017"] = round(row["MÉDIA 2024"] - row["MÉDIA 2017"], 1)
        # flags
        flags = []
        if abs(cn["thetas"][i]) > 2.5:
            flags.append(f"θ_CN={cn['thetas'][i]:+.2f}")
        if abs(mt["thetas"][i]) > 2.5:
            flags.append(f"θ_MT={mt['thetas'][i]:+.2f}")
        if cn["ses"][i] > 0.6:
            flags.append(f"SE_CN={cn['ses'][i]:.2f}")
        if mt["ses"][i] > 0.6:
            flags.append(f"SE_MT={mt['ses'][i]:.2f}")
        # divergência forte entre CN e MT na escala 2024 (referência mais recente)
        diff_cn_mt_2024 = abs(row["NOTA CN 2024"] - row["NOTA MT 2024"])
        if diff_cn_mt_2024 > 250:
            flags.append(f"|CN-MT|={diff_cn_mt_2024:.0f}")
        row["ATÍPICO"] = "; ".join(flags)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["RANK MÉDIA 2024"] = df["MÉDIA 2024"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("RANK MÉDIA 2024").reset_index(drop=True)

    # === Por Nível (média) ===
    def por_nivel(df: pd.DataFrame, ano: int) -> pd.DataFrame:
        col_n = f"NÍVEL MÉDIO {ano}"
        ordem = ["AVANÇADO", "PROFICIENTE", "BÁSICO", "ELEMENTAR", "INICIANTE"]
        faixas = {"AVANÇADO": "≥650", "PROFICIENTE": "≥580", "BÁSICO": "≥480",
                  "ELEMENTAR": "≥380", "INICIANTE": "<380"}
        rows_n = []
        total = len(df)
        for nv in ordem:
            sub = df[df[col_n] == nv]
            n = len(sub)
            if n == 0:
                rows_n.append({"NÍVEL": nv, "FAIXA": faixas[nv], "N": 0, "%": 0.0,
                               "CN MED": None, "MT MED": None, "MÉDIA MED": None,
                               "ACERTOS CN MED": None, "ACERTOS MT MED": None})
                continue
            rows_n.append({
                "NÍVEL": nv,
                "FAIXA": faixas[nv],
                "N": n,
                "%": round(100 * n / total, 1),
                "CN MED": round(sub[f"NOTA CN {ano}"].median(), 1),
                "MT MED": round(sub[f"NOTA MT {ano}"].median(), 1),
                "MÉDIA MED": round(sub[f"MÉDIA {ano}"].median(), 1),
                "ACERTOS CN MED": int(sub["ACERTOS CN"].median()),
                "ACERTOS MT MED": int(sub["ACERTOS MT"].median()),
            })
        return pd.DataFrame(rows_n)

    df_niv_2017 = por_nivel(df, 2017)
    df_niv_2024 = por_nivel(df, 2024)

    # === Migração de nível MÉDIO 2017 → 2024 ===
    ord_full = ["AVANÇADO", "PROFICIENTE", "BÁSICO", "ELEMENTAR", "INICIANTE"]
    comparacao = pd.crosstab(
        df["NÍVEL MÉDIO 2017"], df["NÍVEL MÉDIO 2024"],
        rownames=["Nível 2017 ↓"], colnames=["Nível 2024 →"], dropna=False,
    ).reindex(index=ord_full, columns=ord_full, fill_value=0)

    # === Resumo Executivo ===
    elapsed_total = time.time() - t0
    n_atip = int((df["ATÍPICO"] != "").sum())
    resumo_rows = [
        ("Simulado", "Fredão — Simulado 2"),
        ("Análise", "Consolidada CN + MT × escalas 2017 e 2024"),
        ("Alunos", len(ids)),
        ("Itens CN", cn["n_itens"]),
        ("Itens MT (válidos)", mt["n_itens"]),
        ("Anuladas", "MT Q148 (excluída da calibração)"),
        ("Motor", "V3 cientifico — MML/EM 2PL, c=0.20 fixo"),
        ("Método", "Ancoragem por tabela de referência (tri_min/tri_med/tri_max)"),
        ("Tempo total", f"{elapsed_total:.1f}s"),
        ("", ""),
    ]
    for ano in ANOS:
        resumo_rows += [
            (f"── Escala {ano} ──", ""),
            ("CN nota μ ± σ", f"{notas_cn[ano].mean():.1f} ± {notas_cn[ano].std():.1f}"),
            ("MT nota μ ± σ", f"{notas_mt[ano].mean():.1f} ± {notas_mt[ano].std():.1f}"),
            ("MÉDIA CN+MT μ ± σ",
                f"{df[f'MÉDIA {ano}'].mean():.1f} ± {df[f'MÉDIA {ano}'].std():.1f}"),
            ("MÉDIA min/max",
                f"{df[f'MÉDIA {ano}'].min():.0f} / {df[f'MÉDIA {ano}'].max():.0f}"),
            ("Correlação CN × MT",
                f"{df[[f'NOTA CN {ano}', f'NOTA MT {ano}']].corr().iloc[0,1]:.3f}"),
            ("", ""),
        ]
    resumo_rows += [
        ("── Comparação 2017 vs 2024 ──", ""),
        ("DIFF MÉDIA 2024-2017 μ", f"{df['DIFF MÉDIA 2024-2017'].mean():+.1f}"),
        ("DIFF MÉDIA 2024-2017 σ", f"{df['DIFF MÉDIA 2024-2017'].std():.1f}"),
        ("DIFF min/max",
            f"{df['DIFF MÉDIA 2024-2017'].min():+.1f} / {df['DIFF MÉDIA 2024-2017'].max():+.1f}"),
        ("Atípicos sinalizados", n_atip),
    ]
    resumo = pd.DataFrame(resumo_rows, columns=["Métrica", "Valor"])

    df_atip = df[df["ATÍPICO"] != ""].copy().reset_index(drop=True)

    # === Gráficos ===
    ts = time.strftime("%Y%m%d_%H%M%S")
    png_2017 = OUT_DIR / f"xtri_FREDAO_sim2_consolidado_escala2017_{ts}.png"
    png_2024 = OUT_DIR / f"xtri_FREDAO_sim2_consolidado_escala2024_{ts}.png"
    gerar_grafico_consolidado(df, 2017, png_2017)
    gerar_grafico_consolidado(df, 2024, png_2024)
    print(f"\n  PNG 2017: {png_2017}")
    print(f"  PNG 2024: {png_2024}")

    # === Excel ===
    xlsx_path = OUT_DIR / f"xtri_FREDAO_sim2_consolidado_{ts}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        resumo.to_excel(writer, sheet_name="Resumo Executivo", index=False)
        df.to_excel(writer, sheet_name="Alunos", index=False)
        df_niv_2017.to_excel(writer, sheet_name="Por Nivel 2017", index=False)
        df_niv_2024.to_excel(writer, sheet_name="Por Nivel 2024", index=False)
        comparacao.to_excel(writer, sheet_name="Migracao 2017-2024")
        df_atip.to_excel(writer, sheet_name="Atipicos - Revisar", index=False)
        wb = writer.book
        from openpyxl.drawing.image import Image as XLImage
        ws_g17 = wb.create_sheet("Grafico 2017")
        ws_g17.add_image(XLImage(str(png_2017)), "B2")
        ws_g24 = wb.create_sheet("Grafico 2024")
        ws_g24.add_image(XLImage(str(png_2024)), "B2")

    # Formatação
    wb = openpyxl.load_workbook(xlsx_path)
    from openpyxl.styles import PatternFill, Font, Alignment
    rosa = PatternFill(start_color="FFE6F0", end_color="FFE6F0", fill_type="solid")
    bold = Font(bold=True)
    centro = Alignment(horizontal="center")

    ws_al = wb["Alunos"]
    headers = [c.value for c in ws_al[1]]
    atip_col = headers.index("ATÍPICO") + 1
    for r in range(2, ws_al.max_row + 1):
        if ws_al.cell(r, atip_col).value:
            for c in range(1, ws_al.max_column + 1):
                ws_al.cell(r, c).fill = rosa

    for sheet in ["Resumo Executivo", "Alunos", "Por Nivel 2017",
                  "Por Nivel 2024", "Migracao 2017-2024", "Atipicos - Revisar"]:
        ws = wb[sheet]
        for cell in ws[1]:
            cell.font = bold
            cell.alignment = centro
        for col in ws.columns:
            max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 30)

    wb.save(xlsx_path)
    print(f"\n  XLSX: {xlsx_path}")
    print(f"\nTOTAL: {elapsed_total:.1f}s")

    print("\n=== Por Nível MÉDIO escala 2017 ===")
    print(df_niv_2017.to_string(index=False))
    print("\n=== Por Nível MÉDIO escala 2024 ===")
    print(df_niv_2024.to_string(index=False))
    print("\n=== Migração de nível MÉDIO ===")
    print(comparacao.to_string())
    print(f"\nAtípicos: {n_atip}")
    print(f"\nDIFF MÉDIA 2024-2017: μ={df['DIFF MÉDIA 2024-2017'].mean():+.1f}, "
          f"min={df['DIFF MÉDIA 2024-2017'].min():+.1f}, "
          f"max={df['DIFF MÉDIA 2024-2017'].max():+.1f}")


if __name__ == "__main__":
    main()
