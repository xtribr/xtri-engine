"""Fredao Simulado 2 - Correcao TRI V3 ancorada na escala ENEM 2017.

Usa pipeline_cientifico (MML/EM 2PL, c=0.20) + ancoragem na tabela
oficial multi-ano (tri_tabela_referencia_multiano.json[2017]).

- CN: 45 itens (Q91-135), 0 NaN, 0 anuladas
- MT: 45 itens (Q136-180), Q148 ANULADA -> excluida (44 itens validos)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
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

from app.motores.cientifico import pipeline_cientifico  # noqa: E402
from app.tabela_referencia import TabelaReferenciaTRI  # noqa: E402

ANO_REFERENCIA = 2017

DROPBOX = Path("/Volumes/Kingston/Library/CloudStorage/Dropbox/Fredao Simulados")
CN_PATH = DROPBOX / "Natureza - Simulado 2.xlsx"
MT_PATH = DROPBOX / "Matemática - Simulado 2.xlsx"
OUT_DIR = DROPBOX


@dataclass(frozen=True)
class AreaResult:
    area: str
    ids: List[str]
    resp_matrix: np.ndarray
    item_params: List[Dict]
    thetas: np.ndarray
    ses: np.ndarray
    notas: np.ndarray
    acertos: np.ndarray
    diagnostico: Dict
    # Numeros reais das questoes do gabarito (ex: 91..135 ou 136..180).
    # Mesmo comprimento da matriz ORIGINAL (antes de excluir anuladas).
    q_numbers_full: List[int]
    # Indice (na matriz original) do item anulado, ou -1 se nenhum
    anulada_idx: int


def carregar_planilha(path: Path, area: str) -> Tuple[List[str], np.ndarray, List[int]]:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    # primeira col = ID, demais (col 2..46) = questoes
    header = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    raw = header[1:46]
    q_numbers: List[int] = []
    for i, h in enumerate(raw):
        if isinstance(h, (int, float)):
            q_numbers.append(int(h))
        else:
            # Header nao numerico (ex: "Anulada") -> inferir pelo vizinho.
            anterior = next((int(x) for x in reversed(raw[:i])
                             if isinstance(x, (int, float))), None)
            posterior = next((int(x) for x in raw[i + 1:]
                              if isinstance(x, (int, float))), None)
            if anterior is not None:
                q_numbers.append(anterior + 1)
            elif posterior is not None:
                q_numbers.append(posterior - 1)
            else:
                q_numbers.append(-1)
    ids: List[str] = []
    rows: List[List[int]] = []
    for r in range(2, ws.max_row + 1):
        aluno_id = ws.cell(r, 1).value
        if not aluno_id:
            continue
        ids.append(str(aluno_id))
        row = [int(ws.cell(r, c).value or 0) for c in range(2, 47)]
        rows.append(row)
    mat = np.array(rows, dtype=int)
    print(f"  [{area}] alunos={len(ids)} | itens={mat.shape[1]} "
          f"(Q{q_numbers[0]}-Q{q_numbers[-1]}) | "
          f"acertos medio={mat.sum(axis=1).mean():.1f}/{mat.shape[1]}")
    return ids, mat, q_numbers


def processar_area(area: str, path: Path, tabela: TabelaReferenciaTRI) -> AreaResult:
    print(f"\n=== {area} ===")
    ids, mat_full, q_numbers_full = carregar_planilha(path, area)

    # Auto-detecta anulada pelo header da planilha
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    header_raw = [ws.cell(1, c).value for c in range(2, 47)]
    anulada_idx = next(
        (i for i, h in enumerate(header_raw)
         if isinstance(h, str) and h.strip().lower().startswith("anul")),
        -1,
    )

    if anulada_idx >= 0:
        keep = [j for j in range(mat_full.shape[1]) if j != anulada_idx]
        mat = mat_full[:, keep]
        q_anulada = q_numbers_full[anulada_idx]
        print(f"  [{area}] EXCLUINDO Q{q_anulada} (anulada, idx {anulada_idx}) "
              f"-> matriz {mat.shape}")
    else:
        mat = mat_full

    alunos = [{"aluno_id": ids[i], "nome": ids[i], "respostas": mat[i].tolist()}
              for i in range(len(ids))]

    t0 = time.time()
    out = pipeline_cientifico(alunos=alunos, area=area, tabela=tabela, c_fixo=0.20)
    elapsed = time.time() - t0

    diag = out["diagnostico"]
    print(f"  [{area}] done in {elapsed:.1f}s | "
          f"b mu={diag['b_medio']:+.2f} sd={diag['b_desvio']:.2f} | "
          f"saturados={diag['itens_saturados']}/{diag['n_itens']}")
    print(f"  [{area}] Notas mu={diag['nota_media']:.1f} sd={diag['nota_desvio']:.1f} | "
          f"min={diag['nota_min']:.0f} max={diag['nota_max']:.0f} | "
          f"dentro_faixa_ref={diag['pct_dentro_intervalo']}%")

    resultados = out["resultados_alunos"]
    thetas = np.array([r["theta"] for r in resultados])
    ses = np.array([r["se"] for r in resultados])
    notas = np.array([r["nota_ancorada"] for r in resultados])
    acertos = np.array([r["acertos"] for r in resultados])

    return AreaResult(area=area, ids=ids, resp_matrix=mat,
                      item_params=out["parametros_itens"],
                      thetas=thetas, ses=ses, notas=notas, acertos=acertos,
                      diagnostico=diag,
                      q_numbers_full=q_numbers_full,
                      anulada_idx=anulada_idx)


def nivel_pedagogico(nota: float) -> str:
    if nota >= 650:
        return "AVANCADO"
    if nota >= 580:
        return "PROFICIENTE"
    if nota >= 480:
        return "BASICO"
    if nota >= 380:
        return "ELEMENTAR"
    return "INICIANTE"


def flag_atipico(r_cn: AreaResult, r_mt: AreaResult, i: int) -> str:
    flags = []
    if abs(r_cn.thetas[i]) > 2.5:
        flags.append(f"theta_CN={r_cn.thetas[i]:+.2f}")
    if abs(r_mt.thetas[i]) > 2.5:
        flags.append(f"theta_MT={r_mt.thetas[i]:+.2f}")
    if r_cn.ses[i] > 0.6:
        flags.append(f"SE_CN={r_cn.ses[i]:.2f}")
    if r_mt.ses[i] > 0.6:
        flags.append(f"SE_MT={r_mt.ses[i]:.2f}")
    diff = abs(r_cn.notas[i] - r_mt.notas[i])
    if diff > 250:
        flags.append(f"DIFF_CN_MT={diff:.0f}")
    return "; ".join(flags)


def gerar_alunos_df(r_cn: AreaResult, r_mt: AreaResult) -> pd.DataFrame:
    rows = []
    for i, aluno_id in enumerate(r_cn.ids):
        nota_cn = float(r_cn.notas[i])
        nota_mt = float(r_mt.notas[i])
        media = (nota_cn + nota_mt) / 2
        rows.append({
            "ID": aluno_id,
            "ACERTOS CN": int(r_cn.acertos[i]),
            "theta CN": round(float(r_cn.thetas[i]), 3),
            "SE CN": round(float(r_cn.ses[i]), 3),
            "NOTA CN": round(nota_cn, 1),
            "NIVEL CN": nivel_pedagogico(nota_cn),
            "ACERTOS MT": int(r_mt.acertos[i]),
            "theta MT": round(float(r_mt.thetas[i]), 3),
            "SE MT": round(float(r_mt.ses[i]), 3),
            "NOTA MT": round(nota_mt, 1),
            "NIVEL MT": nivel_pedagogico(nota_mt),
            "MEDIA CN+MT": round(media, 1),
            "NIVEL MEDIO": nivel_pedagogico(media),
            "ATIPICO": flag_atipico(r_cn, r_mt, i),
        })
    df = pd.DataFrame(rows)
    df["RANK MEDIA"] = df["MEDIA CN+MT"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("RANK MEDIA").reset_index(drop=True)
    return df


def gerar_por_nivel(df: pd.DataFrame) -> pd.DataFrame:
    ordem = ["AVANCADO", "PROFICIENTE", "BASICO", "ELEMENTAR", "INICIANTE"]
    faixas = {"AVANCADO": ">=650", "PROFICIENTE": ">=580", "BASICO": ">=480",
              "ELEMENTAR": ">=380", "INICIANTE": "<380"}
    rows = []
    total = len(df)
    for nivel in ordem:
        sub = df[df["NIVEL MEDIO"] == nivel]
        n = len(sub)
        if n == 0:
            rows.append({"NIVEL": nivel, "FAIXA": faixas[nivel], "N": 0, "%": 0.0,
                         "NOTA CN MED": None, "NOTA MT MED": None, "MEDIA MED": None,
                         "ACERTOS CN MED": None, "ACERTOS MT MED": None})
            continue
        rows.append({
            "NIVEL": nivel,
            "FAIXA": faixas[nivel],
            "N": n,
            "%": round(100 * n / total, 1),
            "NOTA CN MED": round(sub["NOTA CN"].median(), 1),
            "NOTA MT MED": round(sub["NOTA MT"].median(), 1),
            "MEDIA MED": round(sub["MEDIA CN+MT"].median(), 1),
            "ACERTOS CN MED": int(sub["ACERTOS CN"].median()),
            "ACERTOS MT MED": int(sub["ACERTOS MT"].median()),
        })
    return pd.DataFrame(rows)


NIVEIS_CORES = {
    "AVANCADO": "#1B7340",
    "PROFICIENTE": "#56AE57",
    "BASICO": "#F5C84C",
    "ELEMENTAR": "#E58A3D",
    "INICIANTE": "#C1392B",
}
NIVEIS_ORDEM = ["INICIANTE", "ELEMENTAR", "BASICO", "PROFICIENTE", "AVANCADO"]


def _suavizar_banda_referencia(y: np.ndarray, threshold_pct: float = 0.15) -> np.ndarray:
    """Remove picos isolados na banda de referencia INEP para visualizacao.

    Algumas tabelas ENEM (ex: 2017 CN, acertos=9 com tri_max=733 vs vizinhos ~580)
    tem outliers de pequena amostra. Esta funcao substitui valores que excedem
    AMBOS os vizinhos em mais de `threshold_pct` pela media dos dois vizinhos.

    Aplica APENAS na visualizacao da banda -- a tabela original continua sendo
    a fonte de verdade para o calculo de notas dos alunos.
    """
    y = np.array(y, dtype=float).copy()
    n = len(y)
    # itera ate convergir (cobre picos consecutivos)
    for _ in range(3):
        for i in range(1, n - 1):
            ref = max(y[i - 1], y[i + 1])
            if ref > 0 and y[i] > ref * (1 + threshold_pct):
                y[i] = (y[i - 1] + y[i + 1]) / 2
    return y


def desenhar_zonas_pedagogicas(ax, x_min, x_max, eixo="y",
                                show_labels=True, y_for_labels=None):
    faixas = [
        (650, 1000, "AVANCADO"),
        (580, 650, "PROFICIENTE"),
        (480, 580, "BASICO"),
        (380, 480, "ELEMENTAR"),
        (0, 380, "INICIANTE"),
    ]
    for lo, hi, nv in faixas:
        if eixo == "y":
            ax.axhspan(lo, hi, alpha=0.10, color=NIVEIS_CORES[nv], zorder=0)
            if show_labels:
                ax.text(x_max * 0.98, (lo + hi) / 2, nv,
                        fontsize=8, ha="right", va="center",
                        color=NIVEIS_CORES[nv], fontweight="bold", alpha=0.85)
        else:
            ax.axvspan(lo, hi, alpha=0.10, color=NIVEIS_CORES[nv], zorder=0)
            if show_labels and y_for_labels is not None:
                ax.text((lo + hi) / 2, y_for_labels, nv,
                        fontsize=8, ha="center", va="top",
                        color=NIVEIS_CORES[nv], fontweight="bold", alpha=0.85)


def _classificar_itens(item_params: List[Dict]) -> Tuple[List[int], List[int], List[int], float, float]:
    """PRIMI-tercis: classifica itens em faceis/medios/dificeis pelos tercis
    empiricos do parametro b (exclui saturados). Retorna tambem os cortes
    (b_q33, b_q66) para anotacao no grafico.

    Justificativa: thresholds absolutos (b<-1.5, b>1.5) funcionam quando ha
    parametros oficiais INEP (escala estavel entre anos). Em calibracoes
    locais V3 (escala do proprio lote), os b's tem media ~0 e poucos itens
    extremos -- usar tercis garante 3 grupos balanceados e interpretaveis.
    """
    bs_validos = [(j, p["b"]) for j, p in enumerate(item_params)
                  if not p.get("saturado")]
    if not bs_validos:
        return [], [], [], 0.0, 0.0
    bs = np.array([b for _, b in bs_validos])
    q33 = float(np.quantile(bs, 1 / 3))
    q66 = float(np.quantile(bs, 2 / 3))
    idx_f = [j for j, b in bs_validos if b <= q33]
    idx_m = [j for j, b in bs_validos if q33 < b <= q66]
    idx_d = [j for j, b in bs_validos if b > q66]
    return idx_f, idx_m, idx_d, q33, q66


def _calcular_primi_metricas(r: AreaResult):
    idx_f, idx_m, idx_d, q33, q66 = _classificar_itens(r.item_params)
    mat = r.resp_matrix
    n_alunos = mat.shape[0]
    af = mat[:, idx_f].sum(axis=1) / max(1, len(idx_f)) if idx_f else np.zeros(n_alunos)
    am = mat[:, idx_m].sum(axis=1) / max(1, len(idx_m)) if idx_m else np.zeros(n_alunos)
    ad = mat[:, idx_d].sum(axis=1) / max(1, len(idx_d)) if idx_d else np.zeros(n_alunos)
    df = pd.DataFrame({
        "ID": r.ids,
        "acertos": r.acertos,
        "theta": r.thetas,
        "nota": r.notas,
        "acert_fac": af,
        "acert_med": am,
        "acert_dif": ad,
        "nivel": [nivel_pedagogico(n) for n in r.notas],
    })
    return df, len(idx_f), len(idx_m), len(idx_d), q33, q66


def gerar_primi_area(r: AreaResult, area_label: str, area_cor: str,
                     tabela: TabelaReferenciaTRI, png_path: Path):
    """4 paineis estilo PRIMI para uma area: banda ref, coerencia dificil,
    distribuicao por nivel, coerencia fac vs dif."""
    df_p, n_f, n_m, n_d, q33, q66 = _calcular_primi_metricas(r)
    n_itens = r.resp_matrix.shape[1]
    cores = [NIVEIS_CORES[n] for n in df_p["nivel"]]
    classif_legenda = (f"Tercis empiricos do b calibrado: "
                       f"T1 b<={q33:+.2f}, T3 b>{q66:+.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=150)

    # (0,0) Acertos x Nota com banda INEP do ano
    ax = axes[0, 0]
    ac_axis = np.arange(0, n_itens + 1)
    tri_min = np.array([tabela.obter(area_label, a)["tri_min"] for a in ac_axis])
    tri_med = np.array([tabela.obter(area_label, a)["tri_med"] for a in ac_axis])
    tri_max = np.array([tabela.obter(area_label, a)["tri_max"] for a in ac_axis])
    # Suaviza picos isolados na banda (artefatos de pequena amostra INEP)
    tri_min = _suavizar_banda_referencia(tri_min)
    tri_med = _suavizar_banda_referencia(tri_med)
    tri_max = _suavizar_banda_referencia(tri_max)
    ax.fill_between(ac_axis, tri_min, tri_max, alpha=0.20, color="#888",
                    label=f"Banda INEP {ANO_REFERENCIA} (min..max)", zorder=1)
    ax.plot(ac_axis, tri_med, "--", color="#333", linewidth=1.6,
            label=f"Mediana INEP {ANO_REFERENCIA}", zorder=2)
    desenhar_zonas_pedagogicas(ax, 0, n_itens, eixo="y")
    ax.scatter(df_p["acertos"], df_p["nota"], s=24, c=cores,
               edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
    ax.set_xlim(-0.5, n_itens + 0.5)
    ax.set_ylim(200, 1000)
    ax.set_xlabel(f"Acertos {area_label} (0 a {n_itens})")
    ax.set_ylabel(f"Nota TRI ({ANO_REFERENCIA})")
    ax.set_title(f"Acertos x Nota com banda de referencia INEP {ANO_REFERENCIA}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, zorder=0)

    # (0,1) acert_dif x Nota (PRIMI coerencia)
    ax = axes[0, 1]
    desenhar_zonas_pedagogicas(ax, 0, 1, eixo="y")
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.008, 0.008, size=len(df_p))
    ax.scatter(df_p["acert_dif"] + jitter, df_p["nota"], s=24, c=cores,
               edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
    if n_d > 0 and df_p["acert_dif"].std() > 1e-6:
        coef = np.polyfit(df_p["acert_dif"], df_p["nota"], 1)
        xr = np.linspace(0, 1, 50)
        ax.plot(xr, np.polyval(coef, xr), "--", color="#222", linewidth=1.6,
                alpha=0.8, label=f"Tendencia (β={coef[0]:.0f} pts/100% dificeis)",
                zorder=2)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(200, 1000)
    ax.set_xlabel(f"acert_dif: % itens DIFICEIS (T3) acertados\n"
                  f"{n_d} dificeis | {n_m} medios | {n_f} faceis  |  {classif_legenda}")
    ax.set_ylabel(f"Nota TRI ({ANO_REFERENCIA})")
    ax.set_title("PRIMI - coerencia pedagogica (dificeis x nota)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25, zorder=0)

    # (1,0) Histograma TRI empilhado por nivel
    ax = axes[1, 0]
    bins = np.linspace(200, 1000, 41)
    hist_data, hist_labels, hist_cores = [], [], []
    for nv in NIVEIS_ORDEM:
        sub = df_p[df_p["nivel"] == nv]
        if len(sub) > 0:
            hist_data.append(sub["nota"].values)
            hist_labels.append(f"{nv} ({len(sub)})")
            hist_cores.append(NIVEIS_CORES[nv])
    n_vals, _, _ = ax.hist(hist_data, bins=bins, color=hist_cores, stacked=True,
                           edgecolor="white", linewidth=0.5, label=hist_labels, alpha=0.95)
    ymax = float(np.max([np.max(h) for h in n_vals])) * 1.18 if len(n_vals) else 10.0
    desenhar_zonas_pedagogicas(ax, 0, 0, eixo="x",
                                show_labels=True, y_for_labels=ymax * 0.97)
    media = df_p["nota"].mean()
    mediana = df_p["nota"].median()
    ax.axvline(media, color="black", linestyle="-", linewidth=1.4, alpha=0.75,
               label=f"mu = {media:.1f}")
    ax.axvline(mediana, color="black", linestyle="--", linewidth=1.2, alpha=0.65,
               label=f"mediana = {mediana:.1f}")
    ax.set_xlim(200, 1020)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(f"Nota TRI {area_label} ({ANO_REFERENCIA})")
    ax.set_ylabel("N alunos")
    ax.set_title(f"Distribuicao - mu={media:.1f}  sd={df_p['nota'].std():.1f}  "
                 f"mediana={mediana:.1f}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.grid(True, alpha=0.25, axis="y")

    # (1,1) acert_med x acert_dif (coerencia fac vs dif)
    ax = axes[1, 1]
    ax.scatter(df_p["acert_med"], df_p["acert_dif"], s=24, c=cores,
               edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
    ax.plot([0, 1], [0, 1], "--", color="#999", alpha=0.6,
            label="Linha 45 (proporcional)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(f"acert_med: % itens MEDIOS (T2 - {q33:+.2f} < b <= {q66:+.2f})\n{n_m} itens")
    ax.set_ylabel(f"acert_dif: % itens DIFICEIS (T3 - b > {q66:+.2f})\n{n_d} itens")
    if df_p["acert_med"].std() > 1e-6 and df_p["acert_dif"].std() > 1e-6:
        corr = df_p[["acert_med", "acert_dif"]].corr().iloc[0, 1]
        titulo = f"PRIMI - coerencia medios x dificeis (rho={corr:.3f})"
    else:
        titulo = "PRIMI - coerencia medios x dificeis"
    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.suptitle(
        f"FREDAO Sim 2 - {area_label} - V3 + escala ENEM {ANO_REFERENCIA}\n"
        f"{len(df_p)} alunos | {n_itens} itens calibrados | "
        f"acertos mu={df_p['acertos'].mean():.1f} | TRI mu={df_p['nota'].mean():.1f}",
        fontsize=14, fontweight="bold", y=1.00,
    )
    plt.tight_layout()
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close()
    return df_p


def gerar_cnxmt_heatmap(df_cn_primi: pd.DataFrame, df_mt_primi: pd.DataFrame,
                        png_path: Path):
    """Comparativo CN x MT: hexbin density + scatter colorido + transformacao."""
    df = pd.DataFrame({
        "ID": df_cn_primi["ID"],
        "NOTA CN": df_cn_primi["nota"].values,
        "NOTA MT": df_mt_primi["nota"].values,
        "NIVEL CN": df_cn_primi["nivel"].values,
        "NIVEL MT": df_mt_primi["nivel"].values,
    })
    df["MEDIA"] = (df["NOTA CN"] + df["NOTA MT"]) / 2
    df["NIVEL MEDIO"] = df["MEDIA"].apply(nivel_pedagogico)
    cores_medio = [NIVEIS_CORES[n] for n in df["NIVEL MEDIO"]]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), dpi=150)

    # (0) Hexbin density
    ax = axes[0]
    desenhar_zonas_pedagogicas(ax, 200, 900, eixo="y", show_labels=False)
    desenhar_zonas_pedagogicas(ax, 200, 900, eixo="x", show_labels=False)
    hb = ax.hexbin(df["NOTA CN"], df["NOTA MT"], gridsize=28,
                   cmap="YlOrRd", mincnt=1, alpha=0.9, zorder=2)
    ax.plot([200, 1000], [200, 1000], "--", color="#444", alpha=0.6,
            linewidth=1.2, zorder=3)
    ax.set_xlim(300, 950)
    ax.set_ylim(300, 1000)
    ax.set_xlabel(f"Nota CN ({ANO_REFERENCIA})")
    ax.set_ylabel(f"Nota MT ({ANO_REFERENCIA})")
    cb = plt.colorbar(hb, ax=ax, label="N alunos")
    cb.ax.tick_params(labelsize=9)
    corr = df[["NOTA CN", "NOTA MT"]].corr().iloc[0, 1]
    ax.set_title(f"Densidade CN x MT (hexbin)  -  rho={corr:.3f}",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, zorder=1)

    # (1) Scatter colorido por nivel medio
    ax = axes[1]
    desenhar_zonas_pedagogicas(ax, 200, 900, eixo="y", show_labels=False)
    desenhar_zonas_pedagogicas(ax, 200, 900, eixo="x", show_labels=False)
    for nv in NIVEIS_ORDEM:
        sub = df[df["NIVEL MEDIO"] == nv]
        if len(sub) > 0:
            ax.scatter(sub["NOTA CN"], sub["NOTA MT"], s=26,
                       c=NIVEIS_CORES[nv], edgecolor="white", linewidth=0.5,
                       alpha=0.85, label=f"{nv} (n={len(sub)})", zorder=3)
    ax.plot([200, 1000], [200, 1000], "--", color="#444", alpha=0.6,
            linewidth=1.2, zorder=2)
    ax.set_xlim(300, 950)
    ax.set_ylim(300, 1000)
    ax.set_xlabel(f"Nota CN ({ANO_REFERENCIA})")
    ax.set_ylabel(f"Nota MT ({ANO_REFERENCIA})")
    ax.set_title("CN x MT por NIVEL MEDIO (CN+MT)/2",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.grid(True, alpha=0.2, zorder=1)

    # (2) Histograma da MEDIA CN+MT
    ax = axes[2]
    bins = np.linspace(200, 1000, 41)
    hist_data, hist_labels, hist_cores = [], [], []
    for nv in NIVEIS_ORDEM:
        sub = df[df["NIVEL MEDIO"] == nv]
        if len(sub) > 0:
            hist_data.append(sub["MEDIA"].values)
            hist_labels.append(f"{nv} ({len(sub)})")
            hist_cores.append(NIVEIS_CORES[nv])
    n_vals, _, _ = ax.hist(hist_data, bins=bins, color=hist_cores, stacked=True,
                           edgecolor="white", linewidth=0.5, label=hist_labels, alpha=0.95)
    ymax = float(np.max([np.max(h) for h in n_vals])) * 1.18 if len(n_vals) else 10.0
    desenhar_zonas_pedagogicas(ax, 0, 0, eixo="x",
                                show_labels=True, y_for_labels=ymax * 0.97)
    media = df["MEDIA"].mean()
    mediana = df["MEDIA"].median()
    ax.axvline(media, color="black", linestyle="-", linewidth=1.4, alpha=0.75,
               label=f"mu = {media:.1f}")
    ax.axvline(mediana, color="black", linestyle="--", linewidth=1.2, alpha=0.65,
               label=f"mediana = {mediana:.1f}")
    ax.set_xlim(200, 1020)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(f"Media (CN+MT)/2 ({ANO_REFERENCIA})")
    ax.set_ylabel("N alunos")
    ax.set_title(f"Distribuicao da media - mu={media:.1f}  sd={df['MEDIA'].std():.1f}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.grid(True, alpha=0.25, axis="y")

    plt.suptitle(f"FREDAO Sim 2 - CN x MT comparativo - escala ENEM {ANO_REFERENCIA}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close()


def gerar_resumo(r_cn: AreaResult, r_mt: AreaResult, df: pd.DataFrame, t0: float) -> pd.DataFrame:
    elapsed = time.time() - t0
    corr = df[["NOTA CN", "NOTA MT"]].corr().iloc[0, 1]
    return pd.DataFrame([
        ("Simulado", "Fredao - Simulado 2"),
        ("Ancoragem", f"Tabela oficial ENEM {ANO_REFERENCIA} (multiano)"),
        ("Motor", "V3 cientifico - MML/EM 2PL, c=0.20 fixo, EAP, N(0,1) prior"),
        ("Data processamento", time.strftime("%Y-%m-%d %H:%M")),
        ("Tempo total", f"{elapsed:.1f}s"),
        ("Alunos processados", len(df)),
        ("Itens CN", r_cn.resp_matrix.shape[1]),
        ("Itens MT (Q148 anulada excluida)", r_mt.resp_matrix.shape[1]),
        ("CN b medio +/- sd", f"{r_cn.diagnostico['b_medio']:+.2f} +/- {r_cn.diagnostico['b_desvio']:.2f}"),
        ("CN itens saturados", r_cn.diagnostico["itens_saturados"]),
        ("CN nota mu", f"{df['NOTA CN'].mean():.1f}"),
        ("CN nota sd", f"{df['NOTA CN'].std():.1f}"),
        ("CN nota min/max", f"{df['NOTA CN'].min():.0f} / {df['NOTA CN'].max():.0f}"),
        ("CN % dentro faixa ref", f"{r_cn.diagnostico['pct_dentro_intervalo']}%"),
        ("MT b medio +/- sd", f"{r_mt.diagnostico['b_medio']:+.2f} +/- {r_mt.diagnostico['b_desvio']:.2f}"),
        ("MT itens saturados", r_mt.diagnostico["itens_saturados"]),
        ("MT nota mu", f"{df['NOTA MT'].mean():.1f}"),
        ("MT nota sd", f"{df['NOTA MT'].std():.1f}"),
        ("MT nota min/max", f"{df['NOTA MT'].min():.0f} / {df['NOTA MT'].max():.0f}"),
        ("MT % dentro faixa ref", f"{r_mt.diagnostico['pct_dentro_intervalo']}%"),
        ("Correlacao CN x MT", f"{corr:.3f}"),
        ("Atipicos sinalizados", int((df['ATIPICO'] != '').sum())),
    ], columns=["Metrica", "Valor"])


def gerar_itens_df(r: AreaResult) -> pd.DataFrame:
    """Itens com numero REAL da questao (Q91-Q135 ou Q136-Q180).

    Insere uma linha com 'STATUS = ANULADA' na posicao do item excluido,
    sem parametros (a/b/c em branco) para nao confundir com item real.
    """
    # Mapeia indice na matriz CALIBRADA -> param. anulada_idx esta fora.
    calib_iter = iter(r.item_params)
    rows = []
    for j_full, q_num in enumerate(r.q_numbers_full):
        if j_full == r.anulada_idx:
            rows.append({
                "Questao": f"Q{q_num}",
                "Pos. prova": j_full + 1,
                "a": None, "b": None, "c": None,
                "Saturado": None,
                "Status": "ANULADA",
            })
            continue
        p = next(calib_iter)
        rows.append({
            "Questao": f"Q{q_num}",
            "Pos. prova": j_full + 1,
            "a": round(float(p["a"]), 3),
            "b": round(float(p["b"]), 3),
            "c": round(float(p.get("c", 0.20)), 3),
            "Saturado": bool(p.get("saturado", False)),
            "Status": "Calibrado",
        })
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    print(f"FREDAO - Simulado 2 - pipeline V3 ancorado na escala ENEM {ANO_REFERENCIA}")
    print("=" * 70)

    tabelas = TabelaReferenciaTRI.carregar_multiano()
    if ANO_REFERENCIA not in tabelas:
        raise SystemExit(f"Ano {ANO_REFERENCIA} ausente na tabela multiano.")
    tabela = tabelas[ANO_REFERENCIA]
    print(f"Tabela ENEM {ANO_REFERENCIA} carregada. Areas: {tabela.areas}")
    print(f"  CN range nota mediana: {tabela.obter('CN', 0)['tri_med']:.1f} .. "
          f"{tabela.obter('CN', 45)['tri_med']:.1f}")
    print(f"  MT range nota mediana: {tabela.obter('MT', 0)['tri_med']:.1f} .. "
          f"{tabela.obter('MT', 45)['tri_med']:.1f}")

    r_cn = processar_area("CN", CN_PATH, tabela)
    r_mt = processar_area("MT", MT_PATH, tabela)

    print("\n=== Consolidando ===")
    df = gerar_alunos_df(r_cn, r_mt)
    df_nivel = gerar_por_nivel(df)
    df_atip = df[df["ATIPICO"] != ""].sort_values("MEDIA CN+MT", ascending=False).reset_index(drop=True)
    df_resumo = gerar_resumo(r_cn, r_mt, df, t0)
    df_itens_cn = gerar_itens_df(r_cn)
    df_itens_mt = gerar_itens_df(r_mt)

    ts = time.strftime("%Y%m%d_%H%M%S")
    base = OUT_DIR / f"xtri_FREDAO_sim2_ENEM{ANO_REFERENCIA}_{ts}"
    xlsx_path = base.with_suffix(".xlsx")
    png_cn = base.parent / (base.name + "_PRIMI_CN.png")
    png_mt = base.parent / (base.name + "_PRIMI_MT.png")
    png_cnxmt = base.parent / (base.name + "_CNxMT_heatmap.png")
    csv_alunos = base.parent / (base.name + "_alunos.csv")

    df_cn_primi = gerar_primi_area(r_cn, "CN", "#2E86AB", tabela, png_cn)
    df_mt_primi = gerar_primi_area(r_mt, "MT", "#A23B72", tabela, png_mt)
    gerar_cnxmt_heatmap(df_cn_primi, df_mt_primi, png_cnxmt)
    print(f"\n  PNG CN PRIMI : {png_cn}")
    print(f"  PNG MT PRIMI : {png_mt}")
    print(f"  PNG CN x MT  : {png_cnxmt}")
    df.to_csv(csv_alunos, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_resumo.to_excel(writer, sheet_name="Resumo Executivo", index=False)
        df.to_excel(writer, sheet_name="Alunos", index=False)
        df_nivel.to_excel(writer, sheet_name="Por Nivel", index=False)
        df_atip.to_excel(writer, sheet_name="Atipicos", index=False)
        df_itens_cn.to_excel(writer, sheet_name="Itens CN", index=False)
        df_itens_mt.to_excel(writer, sheet_name="Itens MT", index=False)
        # Abas placeholder para imagens (criadas depois)
        wb_w = writer.book
        for nm in ("Grafico CN", "Grafico MT", "Grafico CN x MT"):
            wb_w.create_sheet(nm)

    # Estilo
    from openpyxl.styles import PatternFill, Font, Alignment
    wb = openpyxl.load_workbook(xlsx_path)
    bold = Font(bold=True)
    centro = Alignment(horizontal="center")
    rosa = PatternFill(start_color="FFE6F0", end_color="FFE6F0", fill_type="solid")

    ws_al = wb["Alunos"]
    headers = [c.value for c in ws_al[1]]
    atip_col = headers.index("ATIPICO") + 1
    for row in range(2, ws_al.max_row + 1):
        if ws_al.cell(row, atip_col).value:
            for col in range(1, ws_al.max_column + 1):
                ws_al.cell(row, col).fill = rosa

    # Destaque amarelo nas linhas com Status = ANULADA nas abas de itens
    amarelo = PatternFill(start_color="FFF3B0", end_color="FFF3B0", fill_type="solid")
    for sheet_itens in ("Itens CN", "Itens MT"):
        ws = wb[sheet_itens]
        hdrs = [c.value for c in ws[1]]
        status_col = hdrs.index("Status") + 1
        for row in range(2, ws.max_row + 1):
            if ws.cell(row, status_col).value == "ANULADA":
                for col in range(1, ws.max_column + 1):
                    ws.cell(row, col).fill = amarelo
                    ws.cell(row, col).font = Font(bold=True)

    # Embute imagens nas abas de grafico
    from openpyxl.drawing.image import Image as XLImage
    for sheet_nm, png in [("Grafico CN", png_cn),
                           ("Grafico MT", png_mt),
                           ("Grafico CN x MT", png_cnxmt)]:
        ws_g = wb[sheet_nm]
        img = XLImage(str(png))
        img.width = int(img.width * 0.55)
        img.height = int(img.height * 0.55)
        ws_g.add_image(img, "A1")

    for sheet in wb.sheetnames:
        ws = wb[sheet]
        for cell in ws[1]:
            cell.font = bold
            cell.alignment = centro
        for col in ws.columns:
            max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 28)

    wb.save(xlsx_path)

    print(f"\n  XLSX:  {xlsx_path}")
    print(f"  CSV :  {csv_alunos}")
    print(f"\nTOTAL: {time.time() - t0:.1f}s")
    print("\nDistribuicao (NIVEL MEDIO):")
    print(df_nivel.to_string(index=False))
    print(f"\nAtipicos: {len(df_atip)} alunos")

    return xlsx_path, csv_alunos


if __name__ == "__main__":
    main()
