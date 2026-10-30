"""
Correção TRI de um simulado ENEM completo enviado como JSON binarizado.

Responsabilidades:
- Valida a estrutura do simulado (faixas de questões por área, respostas 0/1).
- Calibra cada área separadamente com o motor científico (`pipeline_cientifico`):
  3PL com c fixo em 0.20; com ≥ 500 alunos informativos na área, c é
  estimado por item sob prior Beta(10, 40).
- Ancora as notas na escala do ano de referência pela posição condicional
  aos acertos e devolve, por aluno e área, acertos, theta, erro padrão,
  posição no grupo, nota e nível.

Áreas com menos de `N_MIN_CALIBRACAO` alunos ficam sem nota e geram aviso.

Ausência por dia de prova (LC e CH no dia 1, CN e MT no dia 2): o aluno é
AUSENTE no dia quando ``presenca`` marca o dia como falso ou quando todas as
respostas das áreas do dia vêm ``null`` (nenhuma marcação). Ausente fica fora
da calibração e dos resumos, sai com nota ``null`` e nível ``AUSENTE``. Dentro
de um dia presente, ``null`` é questão em branco e conta como erro; zero
acertos com marcações continua sendo presente.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .irt.conditional import posicao_relativa
from .motores.cientifico import pipeline_cientifico
from .tabela_referencia import TRI_MINIMA_OFICIAL, TabelaReferenciaTRI

AREAS_VALIDAS = ("LC", "CH", "CN", "MT")
DIA_DA_AREA = {"LC": "dia1", "CH": "dia1", "CN": "dia2", "MT": "dia2"}
NIVEL_AUSENTE = "AUSENTE"
C_PADRAO = 0.20
N_MIN_C_LIVRE = 500  # regra do projeto: c fixo em 0.20 para amostras < 500
N_MIN_CALIBRACAO = 10
A_TETO = 4.0  # limite superior de `a` no L-BFGS-B do motor


class SimuladoInvalidoError(ValueError):
    """Entrada estruturalmente inválida (vira HTTP 400 na API)."""


def _validar(dados: Mapping[str, Any]) -> tuple[dict[str, tuple[int, int]], list[Mapping[str, Any]]]:
    areas = dados.get("areas") or {}
    alunos = dados.get("alunos") or []
    if not areas:
        raise SimuladoInvalidoError("Informe 'areas' com as faixas de questões, ex.: {\"LC\": [1, 45]}")
    if not alunos:
        raise SimuladoInvalidoError("Nenhum aluno em 'alunos'.")

    faixas: dict[str, tuple[int, int]] = {}
    for area, faixa in areas.items():
        if area not in AREAS_VALIDAS:
            raise SimuladoInvalidoError(f"Área inválida: {area}. Esperado: {', '.join(AREAS_VALIDAS)}")
        ini, fim = int(faixa[0]), int(faixa[1])
        if ini < 1 or fim < ini:
            raise SimuladoInvalidoError(f"Faixa inválida para {area}: {list(faixa)}")
        faixas[area] = (ini, fim)

    maior = max(fim for _, fim in faixas.values())
    for pos, aluno in enumerate(alunos):
        rotulo = aluno.get("id") or aluno.get("matricula") or f"#{pos}"
        respostas = aluno.get("respostas") or []
        if len(respostas) < maior:
            raise SimuladoInvalidoError(
                f"Aluno {rotulo}: {len(respostas)} respostas, mas as áreas vão até a questão {maior}."
            )
        fez = aluno.get("fez_area") or {}
        for area, (ini, fim) in faixas.items():
            if not fez.get(area, True):
                continue
            bloco = respostas[ini - 1:fim]
            if any(v not in (0, 1, None) for v in bloco):
                raise SimuladoInvalidoError(
                    f"Aluno {rotulo}, área {area}: respostas devem ser 0 ou 1, ou null para questão "
                    "em branco (marque fez_area=false para quem não fez a área)."
                )
    return faixas, list(alunos)


def _dias_ausentes(aluno: Mapping[str, Any], faixas: Mapping[str, tuple[int, int]]) -> set[str]:
    """Dias em que o aluno faltou: presença falsa ou nenhuma marcação no dia."""

    presenca = aluno.get("presenca") or {}
    respostas = aluno.get("respostas") or []
    ausentes: set[str] = set()
    for dia in sorted({DIA_DA_AREA[a] for a in faixas}):
        if presenca.get(dia) is False:
            ausentes.add(dia)
            continue
        if presenca.get(dia) is True:
            continue
        areas_dia = [a for a in faixas if DIA_DA_AREA[a] == dia]
        marcacoes = [
            v
            for a in areas_dia
            for v in respostas[faixas[a][0] - 1:faixas[a][1]]
        ]
        if marcacoes and all(v is None for v in marcacoes):
            ausentes.add(dia)
    return ausentes


def _nota_ausente() -> dict[str, Any]:
    return {
        "acertos": None,
        "theta": None,
        "se_theta": None,
        "posicao_no_grupo": None,
        "nota_tri": None,
        "nivel": NIVEL_AUSENTE,
        "ausente": True,
    }


def _resumo(notas: np.ndarray, acertos: np.ndarray, thetas: np.ndarray) -> dict[str, float]:
    return {
        "acertos_medio": round(float(acertos.mean()), 2),
        "theta_medio": round(float(thetas.mean()), 4),
        "nota_media": round(float(notas.mean()), 1),
        "nota_dp": round(float(notas.std(ddof=1)), 1) if len(notas) > 1 else 0.0,
        "nota_min": float(notas.min()),
        "nota_mediana": float(np.median(notas)),
        "nota_max": float(notas.max()),
    }


def _correlacao_item_resto(resp: np.ndarray, j: int) -> Optional[float]:
    resto = resp.sum(axis=1) - resp[:, j]
    if resp[:, j].std() == 0 or resto.std() == 0:
        return None
    return round(float(np.corrcoef(resp[:, j], resto)[0, 1]), 4)


def _processar_area(
    area: str,
    ini: int,
    fim: int,
    alunos: Sequence[Mapping[str, Any]],
    tabela: TabelaReferenciaTRI,
    ausentes: Sequence[set[str]],
    ano_referencia: Optional[int] = None,
) -> tuple[Optional[dict[str, Any]], dict[int, dict[str, Any]], list[str]]:
    dia = DIA_DA_AREA[area]
    sel = [
        i for i, a in enumerate(alunos)
        if (a.get("fez_area") or {}).get(area, True) and dia not in ausentes[i]
    ]
    if len(sel) < N_MIN_CALIBRACAO:
        aviso = (f"{area}: {len(sel)} aluno(s) fizeram a área; mínimo de {N_MIN_CALIBRACAO} "
                 "para calibrar. Área sem nota.")
        return None, {}, [aviso]

    # Presente no dia: questão em branco (null) conta como erro.
    resp = np.array(
        [[0 if v is None else v for v in alunos[i]["respostas"][ini - 1:fim]] for i in sel],
        dtype=int,
    )
    c_livre = len(sel) >= N_MIN_C_LIVRE
    resultado = pipeline_cientifico(
        alunos=[{"aluno_id": str(i), "nome": "", "respostas": linha.tolist()} for i, linha in zip(sel, resp)],
        area=area,
        tabela=tabela,
        c_fixo=C_PADRAO,
        c_strategy="bayes" if c_livre else "fixed",
        # A regra é pelo tamanho da coorte da área, decidida acima; o motor
        # não reaplica o próprio limiar de alunos informativos.
        bayes_min_n=0,
        ano_referencia=ano_referencia,
    )
    params = resultado["parametros_itens"]
    curva = resultado["curva_condicional"]

    por_aluno: dict[int, dict[str, Any]] = {}
    for r in resultado["resultados_alunos"]:
        u = posicao_relativa(float(r["theta"]), int(r["acertos"]), curva)
        por_aluno[int(r["aluno_id"])] = {
            "acertos": int(r["acertos"]),
            "theta": round(float(r["theta"]), 4),
            "se_theta": round(float(r["se"]), 4),
            "posicao_no_grupo": round(u, 4),
            "nota_tri": float(r["nota_ancorada"]),
            "nivel": r["estado_tri"],
            "ausente": False,
        }

    avisos = []
    bloco = {
        "questoes": [ini, fim],
        "n_alunos": len(sel),
        "n_ausentes": sum(1 for a in ausentes if dia in a),
        "execucao": resultado["execucao"],
        "modelo": "3PL c estimado (prior Beta 10,40)" if c_livre else f"3PL c fixo {C_PADRAO}",
        "resumo": _resumo(
            np.array([v["nota_tri"] for v in por_aluno.values()]),
            resp.sum(axis=1),
            np.array([v["theta"] for v in por_aluno.values()]),
        ),
        "curva_condicional": curva,
        "itens": [
            {
                "questao": ini + j,
                "a": float(p["a"]),
                "b": float(p["b"]),
                "c": float(p["c"]),
                "taxa_acerto": round(float(resp[:, j].mean()), 4),
                "r_item_resto": _correlacao_item_resto(resp, j),
                "saturado": bool(p.get("saturado", False)),
            }
            for j, p in enumerate(params)
        ],
    }
    suspeitos = [it["questao"] for it in bloco["itens"]
                 if it["r_item_resto"] is not None and it["r_item_resto"] < 0.10]
    if suspeitos:
        avisos.append(f"{area}: questões com correlação item-resto < 0,10 (conferir gabarito): "
                      + ", ".join(map(str, suspeitos)))
    no_teto = [it["questao"] for it in bloco["itens"] if it["a"] >= A_TETO - 0.01]
    if no_teto:
        avisos.append(f"{area}: questões com discriminação no limite do otimizador (a = {A_TETO}); "
                      "parâmetros pouco identificados nesta coorte: " + ", ".join(map(str, no_teto)))
    return bloco, por_aluno, avisos


def processar_simulado_binarizado(
    dados: Mapping[str, Any],
    tabela: TabelaReferenciaTRI,
    ano_referencia: int,
) -> dict[str, Any]:
    """Corrige o simulado inteiro, uma calibração independente por área."""

    faixas, alunos = _validar(dados)
    ausentes = [_dias_ausentes(a, faixas) for a in alunos]
    saida_alunos = [
        {
            "id": a.get("id"),
            "matricula": a.get("matricula"),
            "nome": a.get("nome"),
            "turma": a.get("turma"),
            "fez_area": {area: bool((a.get("fez_area") or {}).get(area, True)) for area in faixas},
            "ausente_dia": {dia: dia in ausentes[i] for dia in sorted({DIA_DA_AREA[x] for x in faixas})},
            "areas": {},
        }
        for i, a in enumerate(alunos)
    ]
    blocos: dict[str, Any] = {}
    avisos: list[str] = []
    for area, (ini, fim) in faixas.items():
        bloco, por_aluno, avisos_area = _processar_area(
            area, ini, fim, alunos, tabela, ausentes, ano_referencia,
        )
        avisos.extend(avisos_area)
        dia = DIA_DA_AREA[area]
        for i, a in enumerate(saida_alunos):
            if dia in ausentes[i] and a["fez_area"][area]:
                a["areas"][area] = _nota_ausente()
        if bloco is None:
            continue
        blocos[area] = bloco
        for i, r in por_aluno.items():
            saida_alunos[i]["areas"][area] = r

    for a in saida_alunos:
        notas = [
            a["areas"][ar]["nota_tri"] for ar in faixas
            if ar in a["areas"] and a["areas"][ar]["nota_tri"] is not None
        ]
        a["media_tri"] = round(float(np.mean(notas)), 1) if notas and len(notas) == len(faixas) else None

    return {
        "status": "sucesso",
        "ano_referencia": ano_referencia,
        "metodo": {
            "modelo": "TRI 3PL unidimensional por área (MML/EM, 61 pontos de quadratura)",
            "parametro_c": {area: b["modelo"] for area, b in blocos.items()},
            "regra_c": (f"c fixo em {C_PADRAO}; com ≥ {N_MIN_C_LIVRE} alunos na área, c estimado "
                        "por item com prior Beta(10, 40)"),
            "theta": "EAP com prior N(0,1)",
            "ausencia": ("AUSENTE no dia (LC e CH no dia 1, CN e MT no dia 2) quando presenca[dia] é false "
                         "ou não há nenhuma marcação no dia; ausente fica fora da calibração e das médias. "
                         "Em dia presente, null é questão em branco e conta como erro"),
            "ancoragem": ("posição condicional aos acertos u = Φ(w·(θ − m(k))/s(k)), w = min(0,9; n/(n+25)); "
                          f"nota = quantil u das notas oficiais do ENEM {ano_referencia} com k acertos"),
            "escala_referencia": f"ENEM {ano_referencia}",
            "piso_por_area": {area: TRI_MINIMA_OFICIAL[area] for area in faixas},
            "observacao": f"Projeção pedagógica na escala do ENEM {ano_referencia}; não é nota oficial do INEP.",
            "gerado_em": date.today().isoformat(),
        },
        "total_alunos": len(alunos),
        "total_questoes": dados.get("total_questoes") or max(fim for _, fim in faixas.values()),
        "areas": blocos,
        "alunos": saida_alunos,
        "avisos": avisos,
    }
