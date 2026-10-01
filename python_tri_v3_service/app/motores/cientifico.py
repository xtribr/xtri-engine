"""
Motor Científico TRI — Baseado no calibrar_tri.py (MIRT/XTRI Skill).

Implementa TRI logística com c fixo (formalmente um 3PL de guessing fixo, NÃO
um 2PL: a ICC é P = c + (1-c)·logística com c=0.20, cujo piso é 0.20 e não 0).
- Calibração MML (Marginal Maximum Likelihood) via L-BFGS-B (estima a, b; c fixo)
- Estimação EAP (Expected A Posteriori) com prior N(0,1)
- Ancoragem percentil na tabela de referência oficial do ENEM

Regras invioláveis:
1. Calibração independente por simulado × área (NUNCA misturar)
2. Parâmetro c fixo em 0.20 (5 alternativas) para N < 500 — isto torna o modelo
   um 3PL de c fixo; descrever como "2PL" nos rótulos/relatórios é incorreto.
3. Ancoragem condicional aos acertos: a posição na faixa de referência de K
   acertos é relativa a quem fez K acertos, u = Φ(w·(θ − m(K))/s(K)), com
   m(K) e s(K) derivados dos parâmetros dos itens (ver app/irt/conditional.py).
"""

import numpy as np
from typing import Any, Dict, List, Optional

from ..irt.conditional import curva_teorica
from ..irt.contracts import ItemParameter, ThetaEstimate
from ..irt.scoring import P, eap_theta, estado_tri
from ..irt.transform import (
    ancorar_nota_individual,
    ancorar_notas,
    calcular_transformacao_theta_nota,
)
from ..irt.unidim_em import calibrar_itens_mml_em
from ..tabela_referencia import TabelaReferenciaTRI


def calibrar_itens(
    resp_matrix: np.ndarray,
    c: float = 0.20,
    n_quad: int = 61,
    c_strategy: str = "fixed",
    bayes_min_n: Optional[int] = None,
) -> List[ItemParameter]:
    """
    Calibra parâmetros a, b de cada item via MML com quadratura gaussiana.

    Args:
        resp_matrix: ndarray (N_alunos × N_itens), valores 0/1
        c: Parâmetro de acerto casual (scalar; usado integralmente quando
            ``c_strategy='fixed'`` e como valor inicial quando ``'bayes'``).
            Padrão 0.20 = 1/5 para itens ENEM com 5 alternativas.
        n_quad: Número de pontos de quadratura (padrão 61 em [-5, 5])
        c_strategy: ``'fixed'`` (padrão) mantém ``c`` constante em todos os
            itens. ``'bayes'`` estima ``c`` por item com prior Beta(10, 40)
            quando o cohort tem alunos suficientes (>= 1000 informativos);
            abaixo disso cai para o fixo graciosamente. Expectativa do audit
            April-2026: 5-10 pontos de MAE contra INEP quando habilitado.
        bayes_min_n: limiar de alunos informativos para estimar ``c`` no modo
            ``'bayes'`` (``None`` = padrão do motor, 1000).

    Returns:
        Lista de dicts {a, b, c, taxa_acerto, saturado}
    """
    return calibrar_itens_mml_em(
        resp_matrix, c=c, n_quad=n_quad, c_strategy=c_strategy,
        bayes_min_n=bayes_min_n,
    )


# ════════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO: CALIBRAR + ESTIMAR + ANCORAR
# ════════════════════════════════════════════════════════════════════════════════

def pipeline_cientifico(
    alunos: List[Dict[str, Any]],
    area: str,
    tabela: TabelaReferenciaTRI,
    c_fixo: float = 0.20,
    c_strategy: str = "fixed",
    bayes_min_n: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executa o pipeline completo de calibração científica para UMA área.

    Regra inviolável: cada combinação simulado × área é um modelo separado.

    Args:
        alunos: Lista de dicts com 'aluno_id', 'nome', 'respostas' (lista de 0/1)
        area: Área (LC, CH, CN, MT)
        tabela: Tabela de referência oficial
        c_fixo: Parâmetro c fixo (0.20 para 5 alternativas)
        c_strategy: ``'fixed'`` (padrão) ou ``'bayes'`` — ver ``calibrar_itens``
        bayes_min_n: limiar de alunos informativos para estimar ``c``

    Returns:
        Dict com 'parametros_itens', 'resultados_alunos', 'diagnostico'
    """
    # Montar matriz de respostas (N_alunos × N_itens)
    resp_matrix = np.array([a['respostas'] for a in alunos], dtype=int)
    n_alunos, n_itens = resp_matrix.shape

    # 1. Calibrar itens
    item_params = calibrar_itens(
        resp_matrix, c=c_fixo, c_strategy=c_strategy, bayes_min_n=bayes_min_n,
    )

    # Estatísticas dos parâmetros
    b_vals = [p['b'] for p in item_params]
    n_saturados = sum(1 for b in b_vals if b >= 4.8)

    # 2. Classify items by difficulty subgroups (Primi & Cicchetto methodology)
    # Hard items: b > 1.5 | Medium: -1.5 <= b <= 1.5 | Easy: b < -1.5
    # Saturated items are excluded from ALL three buckets — their b estimate
    # was clamped at the L-BFGS-B bound and is not a trustworthy difficulty
    # signal. Including them in ``idx_dificeis`` would inflate a student's
    # hard-items-correct ratio (acert_dif) whenever they happened to respond
    # 1 on an untrustworthy item, which then biases the pedagogical recs.
    idx_faceis = [j for j, p in enumerate(item_params)
                  if p['b'] < -1.5 and not p.get('saturado')]
    idx_medios = [j for j, p in enumerate(item_params)
                  if -1.5 <= p['b'] <= 1.5 and not p.get('saturado')]
    idx_dificeis = [j for j, p in enumerate(item_params)
                    if p['b'] > 1.5 and not p.get('saturado')]

    n_faceis = len(idx_faceis)
    n_medios = len(idx_medios)
    n_dificeis = len(idx_dificeis)

    # 3. Estimar θ EAP para cada aluno + Primi metrics
    thetas: List[ThetaEstimate] = []
    for i, aluno in enumerate(alunos):
        t, se = eap_theta(resp_matrix[i], item_params)
        acertos = int(resp_matrix[i].sum())

        # Primi metrics: sub-scores by difficulty group
        acertos_faceis = int(resp_matrix[i, idx_faceis].sum()) if idx_faceis else 0
        acertos_medios = int(resp_matrix[i, idx_medios].sum()) if idx_medios else 0
        acertos_dificeis = int(resp_matrix[i, idx_dificeis].sum()) if idx_dificeis else 0
        acert_dif = round(acertos_dificeis / n_dificeis, 4) if n_dificeis > 0 else 0.0

        thetas.append({
            'aluno_id': aluno['aluno_id'],
            'nome': aluno.get('nome', ''),
            'acertos': acertos,
            'theta': t,
            'se': se,
            'acertos_faceis': acertos_faceis,
            'acertos_medios': acertos_medios,
            'acertos_dificeis': acertos_dificeis,
            'acert_dif': acert_dif,
        })

    # 4. Ancorar notas na tabela de referência, pela posição relativa a quem
    #    teve o mesmo número de acertos.
    curva = curva_teorica(item_params, n_calibracao=n_alunos)
    resultados = ancorar_notas(thetas, tabela, area, curva=curva)

    # 5. Classificar estados TRI
    for r in resultados:
        r['estado_tri'] = estado_tri(r['nota_ancorada'])

    transformacao = calcular_transformacao_theta_nota(
        [r["theta"] for r in resultados],
        [r["nota_ancorada"] for r in resultados],
    )
    # Tamanho da calibração: a escoragem individual usa para o peso da posição.
    transformacao["n_calibracao"] = float(n_alunos)

    # 6. Diagnóstico
    notas = [r['nota_ancorada'] for r in resultados]
    acert_difs = [r['acert_dif'] for r in resultados]
    diagnostico = {
        'n_alunos': n_alunos,
        'n_itens': n_itens,
        'b_medio': round(float(np.mean(b_vals)), 3),
        'b_desvio': round(float(np.std(b_vals)), 3),
        'itens_saturados': n_saturados,
        # n_saturados items are excluded from EAP scoring AND from the Primi
        # difficulty buckets (b estimate is untrustworthy). Documented here so
        # consumers of diagnostico understand what the flag implies.
        'itens_excluidos_eap': n_saturados,
        'nota_media': round(float(np.mean(notas)), 1),
        'nota_min': round(float(np.min(notas)), 1),
        'nota_max': round(float(np.max(notas)), 1),
        'nota_desvio': round(float(np.std(notas)), 1),
        # Primi metrics
        'itens_faceis': n_faceis,
        'itens_medios': n_medios,
        'itens_dificeis': n_dificeis,
        'acert_dif_medio': round(float(np.mean(acert_difs)), 3),
        'transformacao_metodo': transformacao["metodo"],
        'ancoragem_metodo': 'condicional_acertos',
        'ancoragem_peso': curva["peso"],
    }

    # Verificar % dentro do intervalo da tabela
    dentro = 0
    for r in resultados:
        ref = tabela.obter(area, r['acertos'])
        if ref['tri_min'] <= r['nota_ancorada'] <= ref['tri_max']:
            dentro += 1
    diagnostico['pct_dentro_intervalo'] = round(dentro / len(resultados) * 100, 1) if resultados else 0

    return {
        'parametros_itens': item_params,
        'resultados_alunos': resultados,
        'transformacao': transformacao,
        'curva_condicional': curva,
        'diagnostico': diagnostico,
    }


def estimar_theta_individual(
    respostas: List[int],
    parametros_itens: List[Dict[str, float]],
    area: str,
    tabela: TabelaReferenciaTRI,
    transformacao: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Estima nota de UM aluno usando parâmetros já calibrados.
    Ideal para alunos retardatários ou processamento assíncrono via OMR.

    Args:
        respostas: Vetor binário (0/1) de respostas
        parametros_itens: Lista de dicts com 'a', 'b', 'c'
        area: Área (LC, CH, CN, MT)
        tabela: Tabela de referência oficial

    Returns:
        Dict com acertos, theta, se, nota_ancorada, estado_tri
    """
    resp_array = np.array(respostas, dtype=int)
    acertos = int(resp_array.sum())

    t, se = eap_theta(resp_array, parametros_itens)
    n_calibracao = (transformacao or {}).get("n_calibracao")
    nota = ancorar_nota_individual(
        theta=t,
        area=area,
        acertos=acertos,
        tabela=tabela,
        transformacao=transformacao,
        curva=curva_teorica(parametros_itens, n_calibracao=n_calibracao),
    )

    return {
        'acertos': acertos,
        'theta': t,
        'se': se,
        'nota_ancorada': round(nota, 1),
        'estado_tri': estado_tri(nota),
    }
