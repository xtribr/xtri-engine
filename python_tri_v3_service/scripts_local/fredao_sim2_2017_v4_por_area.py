"""Fredao Simulado 2 - correcao V4 (parametros OFICIAIS INEP 2017) + metodo C.

V4: usa a,b,c OFICIAIS do INEP (prova AZUL aplicacao regular: CN co_prova 391,
MT co_prova 403), estima theta por EAP na escala real do ENEM 2017 (populacao
de referencia 2009, N(0,1)), e ancora a nota pelo metodo C (TCC<->tabela 2017):
theta -> E[acertos|theta] -> interpola a curva tri_med 2017. Monotonica em theta
(theta maior => nota maior, sempre), sem as inversoes do metodo de faixa.

Q148 anulada pelo cursinho -> excluida da prova de MT (44 itens validos).

Premissa: o Simulado 2 reaplicou os itens do ENEM 2017 AZUL nas mesmas posicoes
(CN Q91-135, MT Q136-180). Sem isso, parametros oficiais nao se aplicam.

Reaproveita os helpers puros (graficos PRIMI, abas, niveis) de
fredao_sim2_2017_por_area.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # para importar o modulo irmao

from app.irt.scoring import eap_theta, P  # noqa: E402
from app.tabela_referencia import TabelaReferenciaTRI, TRI_MINIMA_OFICIAL  # noqa: E402

# Helpers puros reaproveitados (graficos, abas, niveis) ----------------------
import fredao_sim2_2017_por_area as L  # noqa: E402

ANO_REFERENCIA = 2017
PASTA = L.PASTA
PARAMS_JSON = Path(__file__).resolve().parent / "data" / "params_oficiais.json"

AREAS = {
    "CN": {"arquivo": "Natureza - Simulado 2.xlsx", "nome": "Ciencias da Natureza",
           "saida": "Natureza_Simulado2_TRI_ENEM2017_V4.xlsx"},
    "MT": {"arquivo": "Matemática - Simulado 2.xlsx", "nome": "Matematica",
           "saida": "Matematica_Simulado2_TRI_ENEM2017_V4.xlsx"},
}
CO_PROVA = {"CN": 391, "MT": 403, "_cor": "AZUL (aplicacao regular)"}


def eap_theta_oficial(resp_matrix: np.ndarray, params: List[Dict]):
    thetas = np.zeros(len(resp_matrix))
    ses = np.zeros(len(resp_matrix))
    for i, row in enumerate(resp_matrix):
        t, se = eap_theta(row, params)
        thetas[i] = t
        ses[i] = se
    return thetas, ses


def nota_metodo_c(theta: float, params: List[Dict], tabela, area: str, n_itens: int) -> float:
    """theta -> E[acertos|theta] (TCC com params oficiais) -> interpola tri_med 2017."""
    e_ac = float(sum(P(np.array([theta]), p["a"], p["b"], p["c"])[0] for p in params))
    e_ac = max(0.0, min(float(n_itens), e_ac))
    lo = int(np.floor(e_ac))
    hi = min(n_itens, lo + 1)
    frac = (e_ac - lo) if hi > lo else 0.0
    a = tabela.obter(area, lo)["tri_med"]
    b = tabela.obter(area, hi)["tri_med"]
    nota = a + frac * (b - a)
    return min(1000.0, max(TRI_MINIMA_OFICIAL.get(area, 200.0), nota))


def gerar_resumo_v4(area, nome, co_prova, ids, q_numbers, anulada_idx,
                    params, df, elapsed) -> pd.DataFrame:
    q_anulada = q_numbers[anulada_idx] if anulada_idx >= 0 else None
    notas = df["NOTA TRI 2017"]
    b = np.array([p["b"] for p in params]); a = np.array([p["a"] for p in params])
    rows = [
        ("Simulado", "Fredao - Simulado 2"),
        ("Area", f"{area} - {nome}"),
        ("Modo", "V4 - parametros OFICIAIS INEP"),
        ("Prova oficial usada", f"{co_prova} - {CO_PROVA['_cor']}"),
        ("Escala de referencia", f"Tabela oficial ENEM {ANO_REFERENCIA}"),
        ("Modelo TRI", "3PL com a,b,c oficiais INEP - EAP, prior N(0,1)"),
        ("Metodo ancoragem", "Metodo C: TCC<->tabela 2017 (continua, monotonica em theta)"),
        ("Ordenacao monotonica em theta", "SIM (theta maior => nota maior, sem inversoes)"),
        ("Data processamento", time.strftime("%Y-%m-%d %H:%M")),
        ("Tempo (s)", round(elapsed, 1)),
        ("Alunos corrigidos", len(ids)),
        ("Itens da prova", len(q_numbers)),
        ("Itens validos (scorados)", len(params)),
        ("Item anulado", f"Q{q_anulada} (excluido do scoring)" if q_anulada else "Nenhum"),
        ("a oficial medio (discriminacao)", f"{a.mean():.2f} [{a.min():.2f}, {a.max():.2f}]"),
        ("b oficial medio (dificuldade)", f"{b.mean():+.2f} [{b.min():+.2f}, {b.max():+.2f}]"),
        ("theta medio +/- dp", f"{df['theta'].mean():+.2f} +/- {df['theta'].std():.2f}"),
        ("Nota media", f"{notas.mean():.1f}"),
        ("Nota desvio-padrao", f"{notas.std():.1f}"),
        ("Nota mediana", f"{notas.median():.1f}"),
        ("Nota min / max", f"{notas.min():.0f} / {notas.max():.0f}"),
        ("Notas distintas", f"{notas.round(1).nunique()} / {len(notas)}"),
        ("Acertos medio", f"{df['ACERTOS'].mean():.1f} / {len(params)}"),
        ("Atipicos sinalizados", int((df['ATIPICO'] != '').sum())),
        ("Premissa", "Simulado reaplica itens do ENEM 2017 AZUL nas mesmas posicoes"),
    ]
    return pd.DataFrame(rows, columns=["Metrica", "Valor"])


def processar(area: str, tabela) -> Path:
    cfg = AREAS[area]
    nome = cfg["nome"]
    print(f"\n=== {area} ({nome}) - V4 oficial prova {CO_PROVA[area]} ===")
    ids, mat_full, q_numbers, anulada_idx = L.carregar_planilha(PASTA / cfg["arquivo"])

    # Parametros oficiais alinhados as colunas (posicao). Drop do item anulado.
    plist = json.load(open(PARAMS_JSON))[str(ANO_REFERENCIA)][area]
    if len(plist) != mat_full.shape[1]:
        raise SystemExit(f"{area}: {len(plist)} params oficiais != {mat_full.shape[1]} colunas")
    if anulada_idx >= 0:
        keep = [j for j in range(mat_full.shape[1]) if j != anulada_idx]
        mat = mat_full[:, keep]
        plist = [plist[j] for j in keep]
        print(f"  Q{q_numbers[anulada_idx]} ANULADA -> {mat.shape[1]} itens scorados")
    else:
        mat = mat_full
    params = [{"a": float(p[1]), "b": float(p[2]), "c": float(p[3])} for p in plist]
    n_itens = len(params)

    t0 = time.time()
    thetas, ses = eap_theta_oficial(mat, params)
    acertos = mat.sum(axis=1)
    notas = np.array([nota_metodo_c(float(t), params, tabela, area, n_itens) for t in thetas])
    # Acertos ESPERADOS E[acertos|theta] (TCC) - escala que de fato gera a nota.
    # Plotar a nota contra esta escala faz a correcao encaixar sobre a banda 2017
    # (resíduo ~0 ate a mediana). Nao muda nenhuma nota.
    acertos_esp = np.array([
        float(sum(P(np.array([float(t)]), p["a"], p["b"], p["c"])[0] for p in params))
        for t in thetas
    ])
    elapsed = time.time() - t0

    # Metricas PRIMI por tercis do b oficial
    af, am, ad, n_f, n_m, n_d, q33, q66 = L.primi_metricas(mat, params)
    niveis = [L.nivel_pedagogico(float(x)) for x in notas]
    print(f"  theta {thetas.min():+.2f}..{thetas.max():+.2f} | nota {notas.min():.0f}..{notas.max():.0f} "
          f"(mu={notas.mean():.1f}) | distintas={len(set(round(x,1) for x in notas))}/{len(notas)} "
          f"| PRIMI tercis f/m/d={n_f}/{n_m}/{n_d}")

    df = L.gerar_alunos_df(ids, acertos, thetas, ses, notas, ad)
    df_nivel = L.gerar_por_nivel(df)
    df_itens = L.gerar_itens_df(params, q_numbers, anulada_idx)
    df_atip = df[df["ATIPICO"] != ""].reset_index(drop=True)
    df_resumo = gerar_resumo_v4(area, nome, CO_PROVA[area], ids, q_numbers,
                                anulada_idx, params, df, elapsed)

    saida = PASTA / cfg["saida"]
    png = PASTA / cfg["saida"].replace(".xlsx", "_PRIMI.png")
    L.gerar_primi_chart(
        area, nome, mat, params, acertos, notas, niveis,
        af, am, ad, n_f, n_m, n_d, q33, q66, tabela, png,
        acertos_x=acertos_esp,
        acertos_x_label=f"Acertos esperados E[acertos|theta] (TCC) - {area}",
        titulo_acertos=f"Acertos esperados x Nota encaixando na banda INEP {ANO_REFERENCIA}",
    )
    L.escrever_xlsx(saida, df_resumo, df, df_nivel, df_itens, df_atip, primi_png=png)
    png.unlink(missing_ok=True)  # figura ja embutida no xlsx
    print(f"  -> {saida.name}")
    print(df_nivel.to_string(index=False))
    return saida


def main():
    t0 = time.time()
    print(f"FREDAO Sim 2 - V4 (parametros oficiais INEP) + metodo C | escala ENEM {ANO_REFERENCIA}")
    print("=" * 78)
    tabela = TabelaReferenciaTRI.carregar_multiano()[ANO_REFERENCIA]
    saidas = [processar(area, tabela) for area in ("CN", "MT")]
    print(f"\nTOTAL: {time.time() - t0:.1f}s\nPlanilhas V4 geradas:")
    for s in saidas:
        print(f"  {s}")


if __name__ == "__main__":
    main()
