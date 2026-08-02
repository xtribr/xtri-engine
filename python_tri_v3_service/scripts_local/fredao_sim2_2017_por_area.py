"""Fredao Simulado 2 - correcao TRI por area, ancorada na escala ENEM 2017.

Le os dois arquivos binarios da pasta `Simulado Fredao ` (Natureza e Matematica),
roda o motor cientifico (MML/EM 2PL, c=0.20, EAP, prior N(0,1)) e ancora cada
nota na tabela de referencia oficial ENEM 2017 (tri_min/tri_med/tri_max por
acertos). Gera DUAS planilhas SEPARADAS, uma por area, cada uma auto-contida:

  - Resumo            metadados da correcao + distribuicao por nivel
  - Alunos            ID, acertos, theta, SE, nota TRI 2017, nivel, rank, atipico
  - Por Nivel         contagem/%/medianas por faixa pedagogica
  - Itens             a, b, c por questao real (Q148 destacada como ANULADA)
  - Atipicos          alunos sinalizados (|theta|>2.5 ou SE>0.6)

CN: 45 itens (Q91-135), 0 anuladas.
MT: 45 colunas (Q136-180), Q148 ANULADA -> excluida da calibracao (44 validos).
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

from app.motores.cientifico import pipeline_cientifico  # noqa: E402
from app.irt.scoring import P  # noqa: E402
from app.tabela_referencia import TabelaReferenciaTRI, TRI_MINIMA_OFICIAL  # noqa: E402

ANO_REFERENCIA = 2017

# --- Configuracao da rodada -------------------------------------------------
# C_FIXO = 0.0  -> modelo 2PL puro (sem acerto casual). Remove o piso de chute.
# C_FIXO = 0.20 -> modelo 3PL (acerto casual de 1/5, padrao ENEM).
C_FIXO = 0.0
# ANCHOR = "faixa" -> nota ancorada na faixa [tri_min,tri_med,tri_max] do acerto
#                     exato (amplitude completa, fiel ao ENEM 2017, todos os
#                     niveis povoados; teto por faixa pode comprimir o topo).
# ANCHOR = "tcc"   -> theta -> E[acertos] -> interpola tri_med (monotonica em
#                     theta, mas comprime a escala com calibracao local).
ANCHOR = "faixa"
SUFFIX = "_2PL" if C_FIXO == 0.0 else "_3PL"

_MODELO = "2PL (c=0, sem acerto casual)" if C_FIXO == 0.0 else f"3PL (c={C_FIXO} fixo)"
METODO_ANCORAGEM = (
    "Faixa por acertos (tri_min/tri_med/tri_max do acerto exato)"
    if ANCHOR == "faixa"
    else "TCC<->tabela 2017 (continua, monotonica em theta)"
)

PASTA = Path(
    "/Volumes/Kingston/Library/Mobile Documents/com~apple~CloudDocs/APPS/MIRT/"
    "Simulado Fredão "
)
AREAS = {
    "CN": {"arquivo": "Natureza - Simulado 2.xlsx", "saida": "Natureza_Simulado2_TRI_ENEM2017.xlsx"},
    "MT": {"arquivo": "Matemática - Simulado 2.xlsx", "saida": "Matematica_Simulado2_TRI_ENEM2017.xlsx"},
}

# Faixas pedagogicas usadas em todas as rodadas Fredao (consistencia com o
# historico do simulado; o corte de ancoragem da NOTA vem da tabela INEP 2017).
NIVEIS_FAIXAS = {
    "AVANCADO": ">=650",
    "PROFICIENTE": ">=580",
    "BASICO": ">=480",
    "ELEMENTAR": ">=380",
    "INICIANTE": "<380",
}
NIVEIS_ORDEM = ["AVANCADO", "PROFICIENTE", "BASICO", "ELEMENTAR", "INICIANTE"]
NIVEIS_CORES = {
    "AVANCADO": "#1B7340",
    "PROFICIENTE": "#56AE57",
    "BASICO": "#F5C84C",
    "ELEMENTAR": "#E58A3D",
    "INICIANTE": "#C1392B",
}
AREA_COR = {"CN": "#2E86AB", "MT": "#A23B72"}


def tcc_acertos_esperados(theta: float, params: List[Dict]) -> float:
    """E[acertos | theta] = soma das probabilidades 3PL sobre todos os itens."""
    th = np.array([float(theta)])
    return float(sum(P(th, p["a"], p["b"], p.get("c", 0.20))[0] for p in params))


def nota_via_tcc_tabela(theta: float, params: List[Dict],
                        tabela: TabelaReferenciaTRI, area: str, n_itens: int) -> float:
    """theta -> E[acertos] -> interpola tri_med 2017 entre acertos adjacentes.

    Monotonica em theta porque a TCC e estritamente crescente. Nao satura por
    faixa de acertos exatos (resolve a compressao/inversao do metodo anterior).
    """
    e_ac = max(0.0, min(float(n_itens), tcc_acertos_esperados(theta, params)))
    k_lo = int(np.floor(e_ac))
    k_hi = min(n_itens, k_lo + 1)
    frac = (e_ac - k_lo) if k_hi > k_lo else 0.0
    nota_lo = tabela.obter(area, k_lo)["tri_med"]
    nota_hi = tabela.obter(area, k_hi)["tri_med"]
    return nota_lo + frac * (nota_hi - nota_lo)


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


def carregar_planilha(path: Path) -> Tuple[List[str], np.ndarray, List[int], int]:
    """Le ID (col 1) + 45 colunas de respostas binarias (cols 2..46).

    Retorna (ids, matriz_full 45 colunas, numeros_reais_das_questoes, idx_anulada).
    Numeros nao numericos no header (ex.: 'Anulada') sao inferidos pelo vizinho.
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    raw = [ws.cell(1, c).value for c in range(2, 47)]  # 45 headers

    q_numbers: List[int] = []
    anulada_idx = -1
    for i, h in enumerate(raw):
        if isinstance(h, (int, float)):
            q_numbers.append(int(h))
        else:
            if isinstance(h, str) and h.strip().lower().startswith("anul"):
                anulada_idx = i
            anterior = next((int(x) for x in reversed(raw[:i]) if isinstance(x, (int, float))), None)
            posterior = next((int(x) for x in raw[i + 1:] if isinstance(x, (int, float))), None)
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
        if aluno_id in (None, ""):
            continue
        ids.append(str(aluno_id))
        rows.append([int(ws.cell(r, c).value or 0) for c in range(2, 47)])
    return ids, np.array(rows, dtype=int), q_numbers, anulada_idx


def gerar_alunos_df(ids, acertos, thetas, ses, notas, acert_dif) -> pd.DataFrame:
    rows = []
    for i, aid in enumerate(ids):
        nota = float(notas[i])
        rows.append({
            "ID": aid,
            "ACERTOS": int(acertos[i]),
            "theta": round(float(thetas[i]), 3),
            "SE theta": round(float(ses[i]), 3),
            "NOTA TRI 2017": round(nota, 1),
            "NIVEL": nivel_pedagogico(nota),
            "% DIFICEIS (T3)": round(float(acert_dif[i]), 3),
            "ATIPICO": atipico_flag(thetas[i], ses[i]),
        })
    df = pd.DataFrame(rows)
    df["RANK"] = df["NOTA TRI 2017"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("RANK").reset_index(drop=True)
    cols = ["RANK", "ID", "ACERTOS", "theta", "SE theta", "NOTA TRI 2017",
            "NIVEL", "% DIFICEIS (T3)", "ATIPICO"]
    return df[cols]


def atipico_flag(theta: float, se: float) -> str:
    flags = []
    if abs(theta) > 2.5:
        flags.append(f"theta={theta:+.2f}")
    if se > 0.6:
        flags.append(f"SE={se:.2f}")
    return "; ".join(flags)


def gerar_por_nivel(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = []
    for nv in NIVEIS_ORDEM:
        sub = df[df["NIVEL"] == nv]
        n = len(sub)
        rows.append({
            "NIVEL": nv,
            "FAIXA": NIVEIS_FAIXAS[nv],
            "N": n,
            "%": round(100 * n / total, 1) if total else 0.0,
            "NOTA MEDIANA": round(float(sub["NOTA TRI 2017"].median()), 1) if n else None,
            "ACERTOS MEDIANA": int(sub["ACERTOS"].median()) if n else None,
        })
    return pd.DataFrame(rows)


def gerar_itens_df(item_params: List[Dict], q_numbers: List[int], anulada_idx: int) -> pd.DataFrame:
    """Itens com numero REAL da questao; linha ANULADA sem parametros."""
    calib_iter = iter(item_params)
    rows = []
    for j_full, q_num in enumerate(q_numbers):
        if j_full == anulada_idx:
            rows.append({"Questao": f"Q{q_num}", "Pos. prova": j_full + 1,
                         "a": None, "b": None, "c": None, "Saturado": None, "Status": "ANULADA"})
            continue
        p = next(calib_iter)
        rows.append({
            "Questao": f"Q{q_num}", "Pos. prova": j_full + 1,
            "a": round(float(p["a"]), 3), "b": round(float(p["b"]), 3),
            "c": round(float(p.get("c", 0.20)), 3),
            "Saturado": bool(p.get("saturado", False)), "Status": "Calibrado",
        })
    return pd.DataFrame(rows)


def gerar_resumo(area, area_nome, ids, q_numbers, anulada_idx, diag, df, elapsed) -> pd.DataFrame:
    q_anulada = q_numbers[anulada_idx] if anulada_idx >= 0 else None
    notas = df["NOTA TRI 2017"]
    n_distintas = notas.round(1).nunique()
    rows = [
        ("Simulado", "Fredao - Simulado 2"),
        ("Area", f"{area} - {area_nome}"),
        ("Escala de referencia", f"Tabela oficial ENEM {ANO_REFERENCIA}"),
        ("Modelo TRI", f"{_MODELO} - MML/EM, EAP, prior N(0,1)"),
        ("Metodo ancoragem", METODO_ANCORAGEM),
        ("Ordenacao monotonica em theta",
         "SIM (por construcao da TCC)" if ANCHOR == "tcc"
         else "Nao estrita (teto por faixa de acertos)"),
        ("Data processamento", time.strftime("%Y-%m-%d %H:%M")),
        ("Tempo (s)", round(elapsed, 1)),
        ("Alunos corrigidos", len(ids)),
        ("Itens da prova", len(q_numbers)),
        ("Itens validos (calibrados)", diag["n_itens"]),
        ("Item anulado", f"Q{q_anulada} (excluido da calibracao)" if q_anulada else "Nenhum"),
        ("b medio +/- desvio", f"{diag['b_medio']:+.2f} +/- {diag.get('b_desvio', 0):.2f}"),
        ("Itens saturados", diag["itens_saturados"]),
        ("Nota media", f"{notas.mean():.1f}"),
        ("Nota desvio-padrao", f"{notas.std():.1f}"),
        ("Nota mediana", f"{notas.median():.1f}"),
        ("Nota min / max", f"{notas.min():.0f} / {notas.max():.0f}"),
        ("Notas distintas", f"{n_distintas} / {len(notas)}"),
        ("Acertos medio", f"{df['ACERTOS'].mean():.1f} / {diag['n_itens']}"),
        ("Atipicos sinalizados", int((df["ATIPICO"] != "").sum())),
    ]
    return pd.DataFrame(rows, columns=["Metrica", "Valor"])


def processar(area: str, area_nome: str, tabela: TabelaReferenciaTRI) -> Tuple[Path, pd.DataFrame]:
    cfg = AREAS[area]
    path = PASTA / cfg["arquivo"]
    print(f"\n=== {area} ({area_nome}) ===")
    ids, mat_full, q_numbers, anulada_idx = carregar_planilha(path)
    print(f"  alunos={len(ids)} | colunas={mat_full.shape[1]} "
          f"(Q{q_numbers[0]}-Q{q_numbers[-1]})")

    if anulada_idx >= 0:
        keep = [j for j in range(mat_full.shape[1]) if j != anulada_idx]
        mat = mat_full[:, keep]
        print(f"  Q{q_numbers[anulada_idx]} ANULADA (idx {anulada_idx}) -> "
              f"{mat.shape[1]} itens validos")
    else:
        mat = mat_full
        print("  sem itens anulados")

    alunos = [{"aluno_id": ids[i], "nome": ids[i], "respostas": mat[i].tolist()}
              for i in range(len(ids))]

    t0 = time.time()
    out = pipeline_cientifico(alunos=alunos, area=area, tabela=tabela, c_fixo=C_FIXO)
    elapsed = time.time() - t0
    res = out["resultados_alunos"]
    diag = out["diagnostico"]
    item_params = out["parametros_itens"]
    n_itens = diag["n_itens"]

    thetas = np.array([r["theta"] for r in res])
    ses = np.array([r["se"] for r in res])
    acertos = np.array([r["acertos"] for r in res])

    # theta -> nota conforme ANCHOR escolhido.
    piso = TRI_MINIMA_OFICIAL.get(area, 200.0)
    if ANCHOR == "tcc":
        notas = np.array([
            min(1000.0, max(piso, nota_via_tcc_tabela(float(th), item_params, tabela, area, n_itens)))
            for th in thetas
        ])
    else:  # "faixa": usa a ancoragem do pipeline (faixa do acerto exato)
        notas = np.array([float(r["nota_ancorada"]) for r in res])

    # Metricas PRIMI por tercis do b calibrado (usadas na coluna e no grafico)
    af, am, ad, n_f, n_m, n_d, q33, q66 = primi_metricas(mat, item_params)

    n_distintas = len(set(round(float(x), 1) for x in notas))
    print(f"  done in {elapsed:.1f}s | b mu={diag['b_medio']:+.2f} "
          f"saturados={diag['itens_saturados']} | nota mu={notas.mean():.1f} "
          f"sd={notas.std():.1f} min={notas.min():.0f} max={notas.max():.0f} | "
          f"distintas={n_distintas}/{len(notas)} | PRIMI tercis f/m/d={n_f}/{n_m}/{n_d}")

    df_alunos = gerar_alunos_df(ids, acertos, thetas, ses, notas, ad)
    df_nivel = gerar_por_nivel(df_alunos)
    df_itens = gerar_itens_df(out["parametros_itens"], q_numbers, anulada_idx)
    df_atip = df_alunos[df_alunos["ATIPICO"] != ""].reset_index(drop=True)
    df_resumo = gerar_resumo(area, area_nome, ids, q_numbers, anulada_idx, diag, df_alunos, elapsed)

    saida = PASTA / cfg["saida"].replace(".xlsx", f"{SUFFIX}.xlsx")
    png = PASTA / cfg["saida"].replace(".xlsx", f"{SUFFIX}_PRIMI.png")
    niveis = [nivel_pedagogico(float(x)) for x in notas]
    gerar_primi_chart(area, area_nome, mat, item_params, acertos, notas, niveis,
                      af, am, ad, n_f, n_m, n_d, q33, q66, tabela, png)
    escrever_xlsx(saida, df_resumo, df_alunos, df_nivel, df_itens, df_atip, primi_png=png)
    return saida, df_nivel


# ── PRIMI charts ─────────────────────────────────────────────────────────────

def _suavizar_banda(y: np.ndarray, threshold_pct: float = 0.15) -> np.ndarray:
    """Remove picos isolados na banda de referencia INEP (somente visualizacao).

    Tabelas ENEM tem outliers de pequena amostra (ex.: CN 2017 acertos=9). Troca
    valores que excedem AMBOS os vizinhos em >threshold_pct pela media deles.
    A tabela original (usada no calculo das notas) NAO e alterada.
    """
    y = np.array(y, dtype=float).copy()
    n = len(y)
    for _ in range(3):
        for i in range(1, n - 1):
            ref = max(y[i - 1], y[i + 1])
            if ref > 0 and y[i] > ref * (1 + threshold_pct):
                y[i] = (y[i - 1] + y[i + 1]) / 2
    return y


def _zonas(ax, x_max, eixo="y", labels=True, y_for_labels=None):
    faixas = [(650, 1000, "AVANCADO"), (580, 650, "PROFICIENTE"),
              (480, 580, "BASICO"), (380, 480, "ELEMENTAR"), (0, 380, "INICIANTE")]
    for lo, hi, nv in faixas:
        if eixo == "y":
            ax.axhspan(lo, hi, alpha=0.10, color=NIVEIS_CORES[nv], zorder=0)
            if labels:
                ax.text(x_max * 0.98, (lo + hi) / 2, nv, fontsize=8, ha="right",
                        va="center", color=NIVEIS_CORES[nv], fontweight="bold", alpha=0.85)
        else:
            ax.axvspan(lo, hi, alpha=0.10, color=NIVEIS_CORES[nv], zorder=0)
            if labels and y_for_labels is not None:
                ax.text((lo + hi) / 2, y_for_labels, nv, fontsize=8, ha="center",
                        va="top", color=NIVEIS_CORES[nv], fontweight="bold", alpha=0.85)


def _classificar_tercis(item_params):
    """Tercis empiricos do b calibrado (exclui saturados). Retorna idx + cortes.

    Em calibracao local (escala do proprio lote) os b's tem media ~0; tercis
    garantem 3 grupos balanceados, mais interpretaveis que cortes absolutos.
    """
    bs = [(j, p["b"]) for j, p in enumerate(item_params) if not p.get("saturado")]
    if not bs:
        return [], [], [], 0.0, 0.0
    arr = np.array([b for _, b in bs])
    q33, q66 = float(np.quantile(arr, 1 / 3)), float(np.quantile(arr, 2 / 3))
    idx_f = [j for j, b in bs if b <= q33]
    idx_m = [j for j, b in bs if q33 < b <= q66]
    idx_d = [j for j, b in bs if b > q66]
    return idx_f, idx_m, idx_d, q33, q66


def primi_metricas(resp_matrix, item_params):
    """% de acerto por tercil de dificuldade, por aluno."""
    idx_f, idx_m, idx_d, q33, q66 = _classificar_tercis(item_params)
    n = resp_matrix.shape[0]
    af = resp_matrix[:, idx_f].sum(axis=1) / max(1, len(idx_f)) if idx_f else np.zeros(n)
    am = resp_matrix[:, idx_m].sum(axis=1) / max(1, len(idx_m)) if idx_m else np.zeros(n)
    ad = resp_matrix[:, idx_d].sum(axis=1) / max(1, len(idx_d)) if idx_d else np.zeros(n)
    return af, am, ad, len(idx_f), len(idx_m), len(idx_d), q33, q66


def gerar_primi_chart(area, area_nome, resp_matrix, item_params, acertos, notas,
                      niveis, af, am, ad, n_f, n_m, n_d, q33, q66,
                      tabela, png_path, acertos_x=None, acertos_x_label=None,
                      titulo_acertos=None) -> None:
    """4 paineis estilo PRIMI: banda INEP, coerencia dificil, distribuicao, fac x dif.

    acertos_x: se informado, usa esse vetor no eixo x do painel (0,0) em vez dos
    acertos brutos (ex.: acertos esperados E[acertos|theta], para a nota encaixar
    sobre a banda). NAO altera nenhuma nota -- so muda o eixo da visualizacao.
    """
    n_itens = resp_matrix.shape[1]
    cor = AREA_COR.get(area, "#2E86AB")
    cores = [NIVEIS_CORES[n] for n in niveis]
    x_panel = acertos_x if acertos_x is not None else acertos
    x_label = acertos_x_label or f"Acertos {area} (0 a {n_itens})"
    titulo0 = titulo_acertos or f"Acertos x Nota com banda de referencia INEP {ANO_REFERENCIA}"
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=130)

    # (0,0) Acertos x Nota com banda INEP do ano
    ax = axes[0, 0]
    ac_axis = np.arange(0, n_itens + 1)
    tri_min = _suavizar_banda(np.array([tabela.obter(area, a)["tri_min"] for a in ac_axis]))
    tri_med = _suavizar_banda(np.array([tabela.obter(area, a)["tri_med"] for a in ac_axis]))
    tri_max = _suavizar_banda(np.array([tabela.obter(area, a)["tri_max"] for a in ac_axis]))
    ax.fill_between(ac_axis, tri_min, tri_max, alpha=0.20, color="#888",
                    label=f"Banda INEP {ANO_REFERENCIA} (min..max)", zorder=1)
    ax.plot(ac_axis, tri_med, "--", color="#333", linewidth=1.6,
            label=f"Mediana INEP {ANO_REFERENCIA}", zorder=2)
    _zonas(ax, n_itens, eixo="y")
    ax.scatter(x_panel, notas, s=24, c=cores, edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
    ax.set_xlim(-0.5, n_itens + 0.5); ax.set_ylim(200, 1000)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Nota TRI ({ANO_REFERENCIA})")
    ax.set_title(titulo0, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9); ax.grid(True, alpha=0.25, zorder=0)

    # (0,1) acert_dif x Nota (coerencia pedagogica)
    ax = axes[0, 1]
    _zonas(ax, 1, eixo="y")
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.008, 0.008, size=len(notas))
    ax.scatter(ad + jitter, notas, s=24, c=cores, edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
    if n_d > 0 and float(np.std(ad)) > 1e-6:
        coef = np.polyfit(ad, notas, 1)
        xr = np.linspace(0, 1, 50)
        ax.plot(xr, np.polyval(coef, xr), "--", color="#222", linewidth=1.6, alpha=0.8,
                label=f"Tendencia (b={coef[0]:.0f} pts/100% dificeis)", zorder=2)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(200, 1000)
    ax.set_xlabel(f"acert_dif: % itens DIFICEIS (T3, b>{q66:+.2f}) acertados\n"
                  f"{n_d} dificeis | {n_m} medios | {n_f} faceis")
    ax.set_ylabel(f"Nota TRI ({ANO_REFERENCIA})")
    ax.set_title("PRIMI - coerencia pedagogica (dificeis x nota)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25, zorder=0)

    # (1,0) Histograma TRI empilhado por nivel
    ax = axes[1, 0]
    bins = np.linspace(200, 1000, 41)
    data, labels, cols = [], [], []
    for nv in NIVEIS_ORDEM:
        sub = notas[np.array(niveis) == nv]
        if len(sub) > 0:
            data.append(sub); labels.append(f"{nv} ({len(sub)})"); cols.append(NIVEIS_CORES[nv])
    n_vals, _, _ = ax.hist(data, bins=bins, color=cols, stacked=True,
                           edgecolor="white", linewidth=0.5, label=labels, alpha=0.95)
    ymax = float(np.max([np.max(h) for h in n_vals])) * 1.18 if len(n_vals) else 10.0
    _zonas(ax, 0, eixo="x", labels=True, y_for_labels=ymax * 0.97)
    media, mediana = float(np.mean(notas)), float(np.median(notas))
    ax.axvline(media, color="black", linewidth=1.4, alpha=0.75, label=f"mu = {media:.1f}")
    ax.axvline(mediana, color="black", linestyle="--", linewidth=1.2, alpha=0.65, label=f"mediana = {mediana:.1f}")
    ax.set_xlim(200, 1020); ax.set_ylim(0, ymax)
    ax.set_xlabel(f"Nota TRI {area} ({ANO_REFERENCIA})"); ax.set_ylabel("N alunos")
    ax.set_title(f"Distribuicao - mu={media:.1f} sd={float(np.std(notas)):.1f} mediana={mediana:.1f}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9); ax.grid(True, alpha=0.25, axis="y")

    # (1,1) acert_med x acert_dif (coerencia fac vs dif)
    ax = axes[1, 1]
    ax.scatter(am, ad, s=24, c=cores, edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
    ax.plot([0, 1], [0, 1], "--", color="#999", alpha=0.6, label="Linha 45 (proporcional)")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(f"acert_med: % itens MEDIOS (T2) acertados\n{n_m} itens")
    ax.set_ylabel(f"acert_dif: % itens DIFICEIS (T3) acertados\n{n_d} itens")
    if float(np.std(am)) > 1e-6 and float(np.std(ad)) > 1e-6:
        rho = float(np.corrcoef(am, ad)[0, 1])
        ax.set_title(f"PRIMI - coerencia medios x dificeis (rho={rho:.3f})", fontsize=13, fontweight="bold")
    else:
        ax.set_title("PRIMI - coerencia medios x dificeis", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9); ax.grid(True, alpha=0.25)

    plt.suptitle(
        f"FREDAO Sim 2 - {area_nome} ({area}) - 2PL + escala ENEM {ANO_REFERENCIA}\n"
        f"{len(notas)} alunos | {n_itens} itens calibrados | "
        f"acertos mu={float(np.mean(acertos)):.1f} | TRI mu={float(np.mean(notas)):.1f}",
        fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close()


def escrever_xlsx(path, df_resumo, df_alunos, df_nivel, df_itens, df_atip, primi_png=None) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_resumo.to_excel(writer, sheet_name="Resumo", index=False)
        df_alunos.to_excel(writer, sheet_name="Alunos", index=False)
        df_nivel.to_excel(writer, sheet_name="Por Nivel", index=False)
        df_itens.to_excel(writer, sheet_name="Itens", index=False)
        df_atip.to_excel(writer, sheet_name="Atipicos", index=False)

    from openpyxl.styles import PatternFill, Font, Alignment
    wb = openpyxl.load_workbook(path)
    bold = Font(bold=True)
    centro = Alignment(horizontal="center")
    rosa = PatternFill(start_color="FFE6F0", end_color="FFE6F0", fill_type="solid")
    amarelo = PatternFill(start_color="FFF3B0", end_color="FFF3B0", fill_type="solid")

    # Atipicos em rosa na aba Alunos
    ws_al = wb["Alunos"]
    hdr = [c.value for c in ws_al[1]]
    atip_col = hdr.index("ATIPICO") + 1
    for r in range(2, ws_al.max_row + 1):
        if ws_al.cell(r, atip_col).value:
            for c in range(1, ws_al.max_column + 1):
                ws_al.cell(r, c).fill = rosa

    # Linha ANULADA em amarelo negrito na aba Itens
    ws_it = wb["Itens"]
    hdr_it = [c.value for c in ws_it[1]]
    status_col = hdr_it.index("Status") + 1
    for r in range(2, ws_it.max_row + 1):
        if ws_it.cell(r, status_col).value == "ANULADA":
            for c in range(1, ws_it.max_column + 1):
                ws_it.cell(r, c).fill = amarelo
                ws_it.cell(r, c).font = Font(bold=True)

    # Cabecalhos em negrito + larguras
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        for cell in ws[1]:
            cell.font = bold
            cell.alignment = centro
        for col in ws.columns:
            max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 30)
        ws.freeze_panes = "A2"

    # Embute o grafico PRIMI numa aba propria (4 paineis)
    if primi_png is not None and Path(primi_png).is_file():
        from openpyxl.drawing.image import Image as XLImage
        ws_g = wb.create_sheet("Grafico PRIMI")
        img = XLImage(str(primi_png))
        img.width = int(img.width * 0.62)
        img.height = int(img.height * 0.62)
        ws_g.add_image(img, "A1")

    wb.save(path)


def main():
    t0 = time.time()
    print(f"FREDAO - Simulado 2 - correcao TRI por area | escala ENEM {ANO_REFERENCIA}")
    print("=" * 72)
    tabelas = TabelaReferenciaTRI.carregar_multiano()
    if ANO_REFERENCIA not in tabelas:
        raise SystemExit(f"Ano {ANO_REFERENCIA} ausente na tabela multiano.")
    tabela = tabelas[ANO_REFERENCIA]

    saidas = []
    for area, nome in [("CN", "Ciencias da Natureza"), ("MT", "Matematica")]:
        saida, df_nivel = processar(area, nome, tabela)
        saidas.append(saida)
        print(f"  -> {saida.name}")
        print(df_nivel.to_string(index=False))

    print(f"\nTOTAL: {time.time() - t0:.1f}s")
    print("Planilhas geradas:")
    for s in saidas:
        print(f"  {s}")


if __name__ == "__main__":
    main()
