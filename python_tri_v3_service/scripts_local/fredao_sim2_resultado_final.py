"""Resultado Final - planilha consolidada com graficos separados.

Le as notas V4 (parametros oficiais INEP 2017 + metodo C TCC<->tabela) ja
calculadas e monta UMA planilha unica com:

  - Resumo            metricas por area
  - Alunos            CN+MT consolidado (id, acertos, theta, nota, nivel, media)
  - Dispersao         acertos x nota com banda INEP 2017 (CN | MT)
  - PRIMI             acertos e estimativas E[acertos|theta] (raw | esperado, CN+MT)
  - Histograma        distribuicao da nota TRI por nivel (CN | MT)

Nao altera nenhuma nota. Apenas reorganiza/apresenta o que ja existe.
"""

from __future__ import annotations

import json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.irt.scoring import eap_theta, P  # noqa: E402
from app.tabela_referencia import TabelaReferenciaTRI, TRI_MINIMA_OFICIAL  # noqa: E402
import fredao_sim2_2017_por_area as L  # noqa: E402

ANO = 2017
PASTA = L.PASTA
PARAMS_JSON = Path(__file__).resolve().parent / "data" / "params_oficiais.json"

AREAS = {
    "CN": {"arq": "Natureza - Simulado 2.xlsx", "drop": None, "co_prova": 391,
           "nome": "Ciencias da Natureza"},
    "MT": {"arq": "Matemática - Simulado 2.xlsx", "drop": 12, "co_prova": 403,
           "nome": "Matematica"},
}
OUT = PASTA / "Resultado Final.xlsx"


def carregar_e_scorear(area: str, tabela) -> Dict:
    cfg = AREAS[area]
    ids, mat_full, q_numbers, anulada_idx = L.carregar_planilha(PASTA / cfg["arq"])
    plist = json.load(open(PARAMS_JSON))[str(ANO)][area]
    if cfg["drop"] is not None:
        keep = [j for j in range(45) if j != cfg["drop"]]
        mat = mat_full[:, keep]
        plist = [plist[j] for j in keep]
    else:
        mat = mat_full
    params = [{"a": float(p[1]), "b": float(p[2]), "c": float(p[3])} for p in plist]
    n_itens = len(params)

    thetas = np.zeros(len(ids)); ses = np.zeros(len(ids))
    for i, row in enumerate(mat):
        t, s = eap_theta(row, params); thetas[i] = t; ses[i] = s
    acertos = mat.sum(axis=1)
    # Acertos esperados E[acertos|theta] via TCC
    Eac = np.array([float(sum(P(np.array([t]), p["a"], p["b"], p["c"])[0] for p in params))
                    for t in thetas])
    # Nota metodo C: tri_med(E[acertos])
    def C(e):
        e = max(0.0, min(float(n_itens), e))
        lo = int(np.floor(e)); hi = min(n_itens, lo + 1); f = (e - lo) if hi > lo else 0
        a = tabela.obter(area, lo)["tri_med"]; b = tabela.obter(area, hi)["tri_med"]
        return min(1000.0, max(TRI_MINIMA_OFICIAL[area], a + f * (b - a)))
    notas = np.array([C(e) for e in Eac])
    niveis = [L.nivel_pedagogico(float(x)) for x in notas]
    return dict(area=area, nome=cfg["nome"], co_prova=cfg["co_prova"],
                ids=ids, mat=mat, params=params, n_itens=n_itens,
                acertos=acertos, thetas=thetas, ses=ses, Eac=Eac,
                notas=notas, niveis=niveis, q_numbers=q_numbers,
                anulada_idx=anulada_idx)


def df_alunos_consolidado(cn: Dict, mt: Dict) -> pd.DataFrame:
    """Une CN e MT por ID. Espera as duas listas alinhadas."""
    rows = []
    for i, aid in enumerate(cn["ids"]):
        nc = float(cn["notas"][i]); nm = float(mt["notas"][i])
        med = (nc + nm) / 2
        rows.append({
            "ID": aid,
            "ACERTOS CN": int(cn["acertos"][i]),
            "theta CN": round(float(cn["thetas"][i]), 3),
            "NOTA CN": round(nc, 1),
            "NIVEL CN": L.nivel_pedagogico(nc),
            "ACERTOS MT": int(mt["acertos"][i]),
            "theta MT": round(float(mt["thetas"][i]), 3),
            "NOTA MT": round(nm, 1),
            "NIVEL MT": L.nivel_pedagogico(nm),
            "MEDIA CN+MT": round(med, 1),
            "NIVEL MEDIO": L.nivel_pedagogico(med),
        })
    df = pd.DataFrame(rows)
    df["RANK MEDIA"] = df["MEDIA CN+MT"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("RANK MEDIA").reset_index(drop=True)
    cols = ["RANK MEDIA", "ID", "ACERTOS CN", "theta CN", "NOTA CN", "NIVEL CN",
            "ACERTOS MT", "theta MT", "NOTA MT", "NIVEL MT", "MEDIA CN+MT", "NIVEL MEDIO"]
    return df[cols]


FREDAO_DECISAO = (
    "Por mim, pode mandar bala nessa daí mesmo! Até acho que o nível de dificuldade da "
    "prova foi similar a 2017 e poderíamos 'forçar' a jogar os alunos mais pro meio das "
    "bandas de 2017, ao invés da porção mais baixa da banda, já que essa nossa amostra "
    "sempre é viesada hahahahaha. Mas prefiro deixar pra fazer isso, se necessário, mais "
    "pra frente, segurando um pouco as notas agora pra que eles continuem dando um gás legal."
)


def df_resumo_final(cn: Dict, mt: Dict) -> pd.DataFrame:
    """Aba de assinatura: decisao pedagogica + caminhos para a proxima rodada."""
    rows = [
        ("STATUS", "APROVADO - V4 oficial INEP 2017 + método C (TCC<->tabela), sem alterações"),
        ("", ""),
        ("Decisão do professor Fredão", FREDAO_DECISAO),
        ("", ""),
        ("-- Diagnóstico técnico --", ""),
        ("Por que as notas tendem à porção inferior da banda",
         f"Coorte do cursinho tem theta médio ~+{mt['thetas'].mean():.1f} (MT) na escala 2009 "
         "(referência theta=0). A EAP encolhe theta-chapéu rumo ao prior N(0,1), subestimando "
         "o theta real desta coorte. Resultado: E[acertos|theta-chapéu] fica 3-5 acertos abaixo "
         "do bruto, e a nota (método C, ancorada na mediana) cai ~48 pts abaixo da mediana INEP "
         "indexada pelos acertos brutos. Não é defeito: é a resposta correta do modelo TRI "
         "aplicado a uma coorte acima do prior."),
        ("Por que está aprovado assim",
         "Decisão pedagógica deliberada do professor: nota conservadora preserva motivação "
         "(\"dando um gás\"). A coorte sabe que pode subir; o piso baixo evita inflar expectativas."),
        ("", ""),
        ("-- Caminhos para próxima rodada --",
         "(caso o professor queira empurrar as notas para o meio da banda)"),
        ("Opção A — sigma*theta + mu derivado dos parâmetros oficiais 2017",
         "Trocar o método C por nota = sigma * theta-chapéu + mu, com (sigma, mu) derivados via "
         "regressão TCC<->tabela 2017 nos parâmetros oficiais. Já calculado: MT 141,3*theta + 452,0 "
         "; CN 87,1*theta + 719,8. Preenche a banda (distribuição Primi real, ~87% dentro de "
         "[tri_min, tri_max] em MT), mas reescreve todas as notas; ~12% caem abaixo do tri_min "
         "(penalidade diferencial - feature Primi) e ~3 batem no teto 1000."),
        ("Opção B — prior centrado no theta médio da coorte",
         "Substituir o prior N(0,1) por uma normal centrada na média theta da própria coorte "
         "(~+2,2 em MT). Cancela a shrinkage do 'viés da amostra' sem trocar a fórmula final: "
         "mantém método C, só recalibra a EAP. As notas sobem em direção à mediana das bandas "
         "2017. Defensável quando se sabe que a coorte não é representativa da população 2009."),
        ("Preservado em ambos os caminhos",
         "V4 (parâmetros oficiais INEP 2017, prova AZUL aplicação regular - co_prova CN=391, "
         "MT=403), escala ENEM 2017, ancoragem na tabela oficial 2017. Muda só a etapa final "
         "theta -> nota."),
        ("", ""),
        ("-- Premissa metodológica em vigor --", ""),
        ("Itens do simulado",
         "Assume-se que o Simulado 2 reaplica os itens do ENEM 2017 AZUL nas mesmas posições "
         "(CN Q91-135, MT Q136-180). Q148 anulada pelo cursinho - excluída do scoring (MT com "
         "44 itens válidos)."),
        ("", ""),
        ("Documento gerado em", time.strftime("%Y-%m-%d")),
    ]
    return pd.DataFrame(rows, columns=["Item", "Conteúdo"])


def df_resumo(cn: Dict, mt: Dict, elapsed: float) -> pd.DataFrame:
    rows = [
        ("Simulado", "Fredao - Simulado 2"),
        ("Escala", f"ENEM {ANO} - parametros oficiais INEP (AZUL aplicacao regular)"),
        ("Motor", "V4 - 3PL oficial, EAP prior N(0,1), metodo C (TCC<->tabela)"),
        ("Provas usadas", f"CN co_prova={cn['co_prova']}, MT co_prova={mt['co_prova']}"),
        ("Data", time.strftime("%Y-%m-%d %H:%M")),
        ("Tempo (s)", round(elapsed, 1)),
        ("Alunos corrigidos", len(cn["ids"])),
        ("", ""),
        ("-- CN (Ciencias da Natureza) --", ""),
        ("Itens scorados", cn["n_itens"]),
        ("Acertos medio", f"{cn['acertos'].mean():.1f} / {cn['n_itens']}"),
        ("theta medio +/- dp", f"{cn['thetas'].mean():+.2f} +/- {cn['thetas'].std():.2f}"),
        ("Nota media", f"{cn['notas'].mean():.1f}"),
        ("Nota min / max", f"{cn['notas'].min():.0f} / {cn['notas'].max():.0f}"),
        ("Notas distintas", f"{len(set(round(x,1) for x in cn['notas']))} / {len(cn['notas'])}"),
        ("", ""),
        ("-- MT (Matematica) --", ""),
        ("Itens scorados", mt["n_itens"]),
        ("Q148", "ANULADA (excluida do scoring)"),
        ("Acertos medio", f"{mt['acertos'].mean():.1f} / {mt['n_itens']}"),
        ("theta medio +/- dp", f"{mt['thetas'].mean():+.2f} +/- {mt['thetas'].std():.2f}"),
        ("Nota media", f"{mt['notas'].mean():.1f}"),
        ("Nota min / max", f"{mt['notas'].min():.0f} / {mt['notas'].max():.0f}"),
        ("Notas distintas", f"{len(set(round(x,1) for x in mt['notas']))} / {len(mt['notas'])}"),
    ]
    return pd.DataFrame(rows, columns=["Metrica", "Valor"])


# ── Graficos ────────────────────────────────────────────────────────────────

def _banda(ax, area, n, tabela):
    ax_a = np.arange(0, n + 1)
    tmin = L._suavizar_banda(np.array([tabela.obter(area, a)["tri_min"] for a in ax_a]))
    tmed = L._suavizar_banda(np.array([tabela.obter(area, a)["tri_med"] for a in ax_a]))
    tmax = L._suavizar_banda(np.array([tabela.obter(area, a)["tri_max"] for a in ax_a]))
    ax.fill_between(ax_a, tmin, tmax, alpha=0.20, color="#888",
                    label=f"Banda INEP {ANO} (min..max)", zorder=1)
    ax.plot(ax_a, tmed, "--", color="#333", linewidth=1.6,
            label=f"Mediana INEP {ANO}", zorder=2)


def chart_dispersao(cn: Dict, mt: Dict, tabela, png_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(22, 8), dpi=140)
    for ax, d in zip(axes, [cn, mt]):
        cores = [L.NIVEIS_CORES[n] for n in d["niveis"]]
        _banda(ax, d["area"], d["n_itens"], tabela)
        L._zonas(ax, d["n_itens"], eixo="y")
        ax.scatter(d["acertos"], d["notas"], s=24, c=cores, edgecolor="white",
                   linewidth=0.5, alpha=0.85, zorder=3)
        ax.set_xlim(-0.5, d["n_itens"] + 0.5); ax.set_ylim(200, 1000)
        ax.set_xlabel(f"Acertos brutos {d['area']} (0 a {d['n_itens']})")
        ax.set_ylabel(f"Nota TRI {ANO}")
        ax.set_title(f"{d['area']} - {d['nome']} (V4 oficial prova {d['co_prova']})",
                     fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9); ax.grid(True, alpha=0.25)
    plt.suptitle("DISPERSAO - acertos x nota com banda INEP 2017",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.savefig(png_path, dpi=160, bbox_inches="tight"); plt.close()


def chart_primi(cn: Dict, mt: Dict, tabela, png_path: Path) -> None:
    """4 paineis: CN (raw, esperados) na linha 1; MT (raw, esperados) na linha 2."""
    fig, axes = plt.subplots(2, 2, figsize=(22, 14), dpi=140)
    for row, d in enumerate([cn, mt]):
        cores = [L.NIVEIS_CORES[n] for n in d["niveis"]]
        for col, (xvals, xlab, sub) in enumerate([
            (d["acertos"], f"Acertos BRUTOS {d['area']}", "vs acertos brutos"),
            (d["Eac"], f"Acertos ESPERADOS E[acertos|theta] (TCC) {d['area']}",
             "vs estimativas (escala θ̂-consistente)"),
        ]):
            ax = axes[row, col]
            _banda(ax, d["area"], d["n_itens"], tabela)
            L._zonas(ax, d["n_itens"], eixo="y")
            ax.scatter(xvals, d["notas"], s=24, c=cores, edgecolor="white",
                       linewidth=0.5, alpha=0.85, zorder=3)
            ax.set_xlim(-0.5, d["n_itens"] + 0.5); ax.set_ylim(200, 1000)
            ax.set_xlabel(xlab); ax.set_ylabel(f"Nota TRI {ANO}")
            ax.set_title(f"{d['area']} - {sub}", fontsize=12, fontweight="bold")
            ax.legend(loc="upper left", fontsize=9, framealpha=0.9); ax.grid(True, alpha=0.25)
    plt.suptitle("PRIMI - acertos brutos x acertos esperados (mesma nota, dois eixos)",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout(); plt.savefig(png_path, dpi=160, bbox_inches="tight"); plt.close()


def chart_histograma(cn: Dict, mt: Dict, png_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(22, 8), dpi=140)
    for ax, d in zip(axes, [cn, mt]):
        bins = np.linspace(200, 1000, 41)
        data, labels, cols = [], [], []
        for nv in L.NIVEIS_ORDEM:
            sub = d["notas"][np.array(d["niveis"]) == nv]
            if len(sub) > 0:
                data.append(sub); labels.append(f"{nv} ({len(sub)})")
                cols.append(L.NIVEIS_CORES[nv])
        n_vals, _, _ = ax.hist(data, bins=bins, color=cols, stacked=True,
                               edgecolor="white", linewidth=0.5, label=labels, alpha=0.95)
        ymax = float(np.max([np.max(h) for h in n_vals])) * 1.18 if len(n_vals) else 10.0
        L._zonas(ax, 0, eixo="x", labels=True, y_for_labels=ymax * 0.97)
        media = float(d["notas"].mean()); mediana = float(np.median(d["notas"]))
        ax.axvline(media, color="black", linewidth=1.4, alpha=0.75, label=f"mu = {media:.1f}")
        ax.axvline(mediana, color="black", linestyle="--", linewidth=1.2, alpha=0.65,
                   label=f"mediana = {mediana:.1f}")
        ax.set_xlim(200, 1020); ax.set_ylim(0, ymax)
        ax.set_xlabel(f"Nota TRI {d['area']} ({ANO})"); ax.set_ylabel("N alunos")
        ax.set_title(f"{d['area']} - {d['nome']} | "
                     f"sd={float(d['notas'].std()):.1f}",
                     fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9); ax.grid(True, alpha=0.25, axis="y")
    plt.suptitle("HISTOGRAMA - distribuicao da nota TRI por nivel (estimativa)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.savefig(png_path, dpi=160, bbox_inches="tight"); plt.close()


# ── Workbook builder ────────────────────────────────────────────────────────

def montar_planilha(cn: Dict, mt: Dict, tabela) -> Path:
    df_res = df_resumo(cn, mt, 0.0)
    df_al = df_alunos_consolidado(cn, mt)

    # 3 PNGs temporarios
    png_disp = PASTA / "_rf_dispersao.png"
    png_primi = PASTA / "_rf_primi.png"
    png_hist = PASTA / "_rf_histograma.png"
    chart_dispersao(cn, mt, tabela, png_disp)
    chart_primi(cn, mt, tabela, png_primi)
    chart_histograma(cn, mt, png_hist)

    df_rf = df_resumo_final(cn, mt)
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        df_rf.to_excel(writer, sheet_name="Resumo Final", index=False)
        df_res.to_excel(writer, sheet_name="Resumo", index=False)
        df_al.to_excel(writer, sheet_name="Alunos", index=False)

    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.drawing.image import Image as XLImage
    wb = openpyxl.load_workbook(OUT)

    # Estilo cabecalhos + larguras
    bold = Font(bold=True); centro = Alignment(horizontal="center")
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        for cell in ws[1]:
            cell.font = bold; cell.alignment = centro
        if sheet == "Resumo Final":
            # Coluna Conteudo: larga, com quebra de linha; rotulos em A em negrito
            ws.column_dimensions["A"].width = 48
            ws.column_dimensions["B"].width = 110
            wrap_top = Alignment(wrap_text=True, vertical="top")
            for r in range(2, ws.max_row + 1):
                ws.cell(r, 1).font = Font(bold=True)
                ws.cell(r, 1).alignment = wrap_top
                ws.cell(r, 2).alignment = wrap_top
        else:
            for col in ws.columns:
                mlen = max((len(str(c.value)) if c.value is not None else 0) for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(mlen + 2, 28)
        ws.freeze_panes = "A2"

    # 3 abas de grafico
    for nome_aba, png in [("Dispersao", png_disp),
                          ("PRIMI (acertos e estimativas)", png_primi),
                          ("Histograma por estimativa", png_hist)]:
        ws_g = wb.create_sheet(nome_aba)
        img = XLImage(str(png))
        img.width = int(img.width * 0.58); img.height = int(img.height * 0.58)
        ws_g.add_image(img, "A1")
    wb.save(OUT)

    for p in (png_disp, png_primi, png_hist):
        p.unlink(missing_ok=True)
    return OUT


def main():
    t0 = time.time()
    print("FREDAO Sim 2 - Resultado Final (V4 oficial + metodo C)")
    print("=" * 70)
    tabela = TabelaReferenciaTRI.carregar_multiano()[ANO]
    cn = carregar_e_scorear("CN", tabela)
    mt = carregar_e_scorear("MT", tabela)
    print(f"CN: {len(cn['ids'])} alunos | nota mu={cn['notas'].mean():.1f} | "
          f"min={cn['notas'].min():.0f} max={cn['notas'].max():.0f}")
    print(f"MT: {len(mt['ids'])} alunos | nota mu={mt['notas'].mean():.1f} | "
          f"min={mt['notas'].min():.0f} max={mt['notas'].max():.0f}")
    saida = montar_planilha(cn, mt, tabela)
    print(f"\nTOTAL: {time.time()-t0:.1f}s\nPlanilha gerada: {saida}")


if __name__ == "__main__":
    main()
