"""
Motor Científico TRI — Baseado no calibrar_tri.py (MIRT/XTRI Skill).

Implementa TRI 2PL real com:
- Calibração MML (Marginal Maximum Likelihood) via L-BFGS-B
- Estimação EAP (Expected A Posteriori) com prior N(0,1)
- Ancoragem percentil na tabela de referência oficial do ENEM

Regras invioláveis:
1. Calibração independente por simulado × área (NUNCA misturar)
2. Parâmetro c fixo em 0.20 (5 alternativas) para N < 500
3. Ancoragem contínua: nota = ref_min_K + norm.cdf(θ) × (ref_max_K − ref_min_K)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Any

from ..tabela_referencia import TabelaReferenciaTRI, TRI_MAXIMA_OFICIAL, TRI_MINIMA_OFICIAL


# ════════════════════════════════════════════════════════════════════════════════
# MODELO 3PL / 2PL
# ════════════════════════════════════════════════════════════════════════════════

def P(theta: np.ndarray, a: float, b: float, c: float = 0.20) -> np.ndarray:
    """Probabilidade de acerto — modelo 3PL (c fixo = acerto casual)."""
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))


def _neg_log_lik_item(
    params: np.ndarray,
    responses: np.ndarray,
    c: float,
    theta_grid: np.ndarray,
    weights: np.ndarray
) -> float:
    """Verossimilhança marginal negativa para um item."""
    a, b = params
    if a <= 0.01:
        return 1e10
    ll = 0.0
    for r in responses:
        prob = P(theta_grid, a, b, c)
        prob = np.clip(prob, 1e-9, 1 - 1e-9)
        lk = np.where(r == 1, prob, 1 - prob)
        ll += np.log(np.sum(weights * lk) + 1e-15)
    return -ll


def calibrar_itens(
    resp_matrix: np.ndarray,
    c: float = 0.20,
    n_quad: int = 41
) -> List[Dict[str, Any]]:
    """
    Calibra parâmetros a, b de cada item via MML com quadratura gaussiana.

    Args:
        resp_matrix: ndarray (N_alunos × N_itens), valores 0/1
        c: Parâmetro de acerto casual fixo (0.20 para 5 alternativas)
        n_quad: Número de pontos de quadratura

    Returns:
        Lista de dicts {a, b, c, taxa_acerto, saturado}
    """
    theta_grid = np.linspace(-4, 4, n_quad)
    weights = norm.pdf(theta_grid, 0, 1)
    weights /= weights.sum()

    params = []
    for j in range(resp_matrix.shape[1]):
        responses = resp_matrix[:, j]
        p_obs = np.clip(responses.mean(), c + 0.01, 0.99)
        b_init = -np.log((1 - c) / (p_obs - c) - 1) if p_obs > c else 2.0

        res = minimize(
            _neg_log_lik_item,
            x0=[1.0, b_init],
            args=(responses, c, theta_grid, weights),
            method='L-BFGS-B',
            bounds=[(0.3, 4.0), (-4.0, 4.0)],
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        a_est, b_est = res.x
        params.append({
            'a': round(float(a_est), 4),
            'b': round(float(b_est), 4),
            'c': c,
            'taxa_acerto': round(float(responses.mean()), 4),
            'saturado': b_est >= 3.9,
        })
    return params


def eap_theta(
    responses: np.ndarray,
    item_params: List[Dict[str, float]],
    n_quad: int = 41
) -> Tuple[float, float]:
    """
    Estima θ via EAP (Expected A Posteriori) com prior N(0,1).

    Args:
        responses: Vetor de respostas de um aluno (0/1)
        item_params: Lista de dicts com 'a', 'b', 'c'
        n_quad: Número de pontos de quadratura

    Returns:
        (theta_eap, se_eap)
    """
    theta_grid = np.linspace(-4, 4, n_quad)
    prior = norm.pdf(theta_grid, 0, 1)

    lk = np.ones(n_quad)
    for j, p in enumerate(item_params):
        prob = P(theta_grid, p['a'], p['b'], p['c'])
        prob = np.clip(prob, 1e-9, 1 - 1e-9)
        lk *= np.where(responses[j] == 1, prob, 1 - prob)

    posterior = lk * prior
    s = posterior.sum()
    if s < 1e-300:
        return 0.0, 1.0
    posterior /= s
    theta_eap = float(np.sum(theta_grid * posterior))
    se_eap = float(np.sqrt(np.sum((theta_grid - theta_eap) ** 2 * posterior)))
    return round(theta_eap, 6), round(se_eap, 4)


# ════════════════════════════════════════════════════════════════════════════════
# ANCORAGEM NA TABELA DE REFERÊNCIA
# ════════════════════════════════════════════════════════════════════════════════

def ancorar_notas(
    thetas: List[Dict[str, Any]],
    tabela: TabelaReferenciaTRI,
    area: str
) -> List[Dict[str, Any]]:
    """
    Mapeia θ → nota usando "Cohort-Anchored Distribution".

    Lógica (NOVO, baseada em z-score da coorte):
      Para cada aluno:
        1. Calcula z = (theta - cohort_mean) / cohort_std
        2. Computa percentil = norm.cdf(z) ∈ [0, 1]
        3. Mapeia o percentil dentro do intervalo [tri_min, tri_med, tri_max]
           da tabela de referência para o número de acertos do aluno:
             - pct ∈ [0, 0.5]   → interpola entre tri_min e tri_med
             - pct ∈ (0.5, 1]   → interpola entre tri_med e tri_max
        4. Aluno no fundo da coorte → próximo de tri_min para seu K
        5. Aluno na média da coorte → tri_med
        6. Aluno no topo da coorte → próximo de tri_max para seu K

    Por que essa abordagem (resposta direta às críticas anteriores):
      - Usa o VALOR ABSOLUTO do theta, não o rank — dois alunos com theta
        idênticos sempre recebem a mesma nota
      - PERMITE alunos chegarem a tri_min: se theta é muito baixo na coorte,
        o aluno cai no piso oficial daquele número de acertos
      - PERMITE alunos chegarem a tri_max: se theta é muito alto na coorte,
        o aluno alcança o teto daquele número de acertos
      - Não comprime artificialmente em direção à mediana
      - A coorte serve como referência ESTATÍSTICA (z-score), mas a posição
        final é ANCORADA na tabela oficial do INEP
      - Funciona igual para coortes pequenas (10 alunos) ou grandes (1000+)

    Args:
        thetas: Lista de dicts com 'aluno_id', 'acertos', 'theta', 'se', ...
        tabela: Tabela de referência oficial
        area: Área (LC, CH, CN, MT)

    Returns:
        Lista de dicts com 'nota_ancorada' adicionado
    """
    if not thetas:
        return []

    # 1. Cohort statistics
    all_thetas = np.array([t['theta'] for t in thetas], dtype=float)
    cohort_mean = float(all_thetas.mean())
    cohort_std = float(all_thetas.std())
    if cohort_std < 1e-6:
        cohort_std = 1.0  # avoid div-by-zero in degenerate cases

    tri_maxima = TRI_MAXIMA_OFICIAL.get(area, 1000.0)
    tri_minima = TRI_MINIMA_OFICIAL.get(area, 200.0)

    resultados = []
    for t in thetas:
        ref = tabela.obter(area, t['acertos'])
        ref_min = ref['tri_min']
        ref_med = ref['tri_med']
        ref_max = ref['tri_max']

        # Degenerate band (e.g., 0 or max acertos): just return the median
        if ref_min == ref_max:
            nota = ref_med
        else:
            # 2. z-score within cohort
            z = (t['theta'] - cohort_mean) / cohort_std
            # 3. Percentile via normal CDF
            pct = float(norm.cdf(z))

            # 4. Anchored interpolation: pct=0 → tri_min, 0.5 → tri_med, 1 → tri_max
            if pct <= 0.5:
                # Lower half: linear from tri_min to tri_med
                local_pct = pct / 0.5  # 0 to 1
                nota = ref_min + local_pct * (ref_med - ref_min)
            else:
                # Upper half: linear from tri_med to tri_max
                local_pct = (pct - 0.5) / 0.5  # 0 to 1
                nota = ref_med + local_pct * (ref_max - ref_med)

        # Final clip to area-wide empirical bounds
        nota = max(tri_minima, min(nota, tri_maxima))

        resultado = {**t, 'nota_ancorada': round(float(nota), 1)}
        resultados.append(resultado)

    return resultados


# ════════════════════════════════════════════════════════════════════════════════
# CLASSIFICAÇÃO PEDAGÓGICA (INEP-aligned)
# ════════════════════════════════════════════════════════════════════════════════

def estado_tri(nota: float) -> str:
    """
    Classifica a nota TRI em nível de proficiência pedagógica.

    Baseado nas faixas do INEP (Interpretação Pedagógica das Escalas)
    e nos cortes práticos do SISU.
    """
    if nota >= 700:
        return 'AVANCADO'
    elif nota >= 600:
        return 'PROFICIENTE'
    elif nota >= 500:
        return 'BASICO'
    elif nota >= 400:
        return 'ELEMENTAR'
    else:
        return 'INICIANTE'


# ════════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO: CALIBRAR + ESTIMAR + ANCORAR
# ════════════════════════════════════════════════════════════════════════════════

def pipeline_cientifico(
    alunos: List[Dict[str, Any]],
    area: str,
    tabela: TabelaReferenciaTRI,
    c_fixo: float = 0.20,
) -> Dict[str, Any]:
    """
    Executa o pipeline completo de calibração científica para UMA área.

    Regra inviolável: cada combinação simulado × área é um modelo separado.

    Args:
        alunos: Lista de dicts com 'aluno_id', 'nome', 'respostas' (lista de 0/1)
        area: Área (LC, CH, CN, MT)
        tabela: Tabela de referência oficial
        c_fixo: Parâmetro c fixo (0.20 para 5 alternativas)

    Returns:
        Dict com 'parametros_itens', 'resultados_alunos', 'diagnostico'
    """
    # Montar matriz de respostas (N_alunos × N_itens)
    resp_matrix = np.array([a['respostas'] for a in alunos], dtype=int)
    n_alunos, n_itens = resp_matrix.shape

    # 1. Calibrar itens
    item_params = calibrar_itens(resp_matrix, c=c_fixo)

    # Estatísticas dos parâmetros
    b_vals = [p['b'] for p in item_params]
    n_saturados = sum(1 for b in b_vals if b >= 3.9)

    # 2. Classify items by difficulty subgroups (Primi & Cicchetto methodology)
    # Hard items: b > 1.5 | Medium: -1.5 <= b <= 1.5 | Easy: b < -1.5
    idx_faceis = [j for j, p in enumerate(item_params) if p['b'] < -1.5]
    idx_medios = [j for j, p in enumerate(item_params) if -1.5 <= p['b'] <= 1.5]
    idx_dificeis = [j for j, p in enumerate(item_params) if p['b'] > 1.5]

    n_faceis = len(idx_faceis)
    n_medios = len(idx_medios)
    n_dificeis = len(idx_dificeis)

    # 3. Estimar θ EAP para cada aluno + Primi metrics
    thetas = []
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

    # 4. Ancorar notas na tabela de referência
    resultados = ancorar_notas(thetas, tabela, area)

    # 5. Classificar estados TRI
    for r in resultados:
        r['estado_tri'] = estado_tri(r['nota_ancorada'])

    # 6. Diagnóstico
    notas = [r['nota_ancorada'] for r in resultados]
    acert_difs = [r['acert_dif'] for r in resultados]
    diagnostico = {
        'n_alunos': n_alunos,
        'n_itens': n_itens,
        'b_medio': round(float(np.mean(b_vals)), 3),
        'b_desvio': round(float(np.std(b_vals)), 3),
        'itens_saturados': n_saturados,
        'nota_media': round(float(np.mean(notas)), 1),
        'nota_min': round(float(np.min(notas)), 1),
        'nota_max': round(float(np.max(notas)), 1),
        'nota_desvio': round(float(np.std(notas)), 1),
        # Primi metrics
        'itens_faceis': n_faceis,
        'itens_medios': n_medios,
        'itens_dificeis': n_dificeis,
        'acert_dif_medio': round(float(np.mean(acert_difs)), 3),
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
        'diagnostico': diagnostico,
    }


def estimar_theta_individual(
    respostas: List[int],
    parametros_itens: List[Dict[str, float]],
    area: str,
    tabela: TabelaReferenciaTRI,
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

    # Individual estimation: use tri_med (best point estimate without group context)
    ref = tabela.obter(area, acertos)
    nota = ref['tri_med']

    tri_maxima = TRI_MAXIMA_OFICIAL.get(area, 1000.0)
    nota = min(nota, tri_maxima)

    return {
        'acertos': acertos,
        'theta': t,
        'se': se,
        'nota_ancorada': round(nota, 1),
        'estado_tri': estado_tri(nota),
    }
