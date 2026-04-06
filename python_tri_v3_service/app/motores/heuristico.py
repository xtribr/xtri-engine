"""
Motor Heurístico TRI — Portado do tri_v2_producao.py (Gabaritaí V2).

Usa análise de coerência pedagógica (TCT + ajustes) para estimar notas TRI.
É o motor rápido, ideal para feedback em tempo real.

Fluxo:
1. Calcula % de acerto de cada questão na turma
2. Classifica questões em 5 faixas de dificuldade
3. Para cada aluno, analisa coerência (acertou fáceis? errou difíceis?)
4. Ancora nota na tabela oficial com bônus/penalidade por coerência
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

from ..tabela_referencia import TabelaReferenciaTRI, TRI_MAXIMA_OFICIAL


# ════════════════════════════════════════════════════════════════════════════════
# ANÁLISE DE COERÊNCIA
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class AnaliseCoerencia:
    """Resultado da análise de coerência do aluno."""
    coerencia: float  # 0.0 a 1.0
    padrao_resposta: str
    taxa_muito_facil: float
    taxa_facil: float
    taxa_media: float
    taxa_dificil: float
    taxa_muito_dificil: float


def analisar_coerencia(respostas_por_dificuldade: Dict[str, int]) -> AnaliseCoerencia:
    """
    Analisa coerência das respostas de um aluno em uma área.

    Coerência esperada: acerta mais fáceis, menos difíceis.
    Padrão ideal: MF > F > M > D > MD

    Args:
        respostas_por_dificuldade: {
            'muito_facil': N_acertos, 'facil': N, 'media': N,
            'dificil': N, 'muito_dificil': N,
            '_peso_dificuldade': float  # peso baseado na dificuldade real
        }
    """
    total = sum(v for k, v in respostas_por_dificuldade.items()
                if not k.startswith('_'))

    if total == 0:
        return AnaliseCoerencia(
            coerencia=0.0, padrao_resposta='Aluno não respondeu',
            taxa_muito_facil=0.0, taxa_facil=0.0, taxa_media=0.0,
            taxa_dificil=0.0, taxa_muito_dificil=0.0
        )

    # Calcular taxas
    taxa_mf = respostas_por_dificuldade.get('muito_facil', 0) / total
    taxa_f = respostas_por_dificuldade.get('facil', 0) / total
    taxa_m = respostas_por_dificuldade.get('media', 0) / total
    taxa_d = respostas_por_dificuldade.get('dificil', 0) / total
    taxa_md = respostas_por_dificuldade.get('muito_dificil', 0) / total

    # Verificar coerência (padrão esperado: MF >= F >= M >= D >= MD)
    comparacoes = [
        taxa_mf >= taxa_f,
        taxa_f >= taxa_m,
        taxa_m >= taxa_d,
        taxa_d >= taxa_md
    ]
    coerencia_base = sum(comparacoes) / len(comparacoes)

    # Peso dos acertos por dificuldade
    peso_acertos = (
        respostas_por_dificuldade.get('muito_facil', 0) * 1.0 +
        respostas_por_dificuldade.get('facil', 0) * 0.8 +
        respostas_por_dificuldade.get('media', 0) * 0.5 +
        respostas_por_dificuldade.get('dificil', 0) * 0.3 +
        respostas_por_dificuldade.get('muito_dificil', 0) * 0.1
    )
    peso_normalizado = peso_acertos / (total * 1.0) if total > 0 else 0.5

    # Peso baseado na dificuldade REAL das questões acertadas
    peso_dificuldade = respostas_por_dificuldade.get('_peso_dificuldade', 0.5)

    # Coerência final: combina padrão + peso por classificação + peso dificuldade real
    coerencia = coerencia_base * 0.3 + peso_normalizado * 0.3 + peso_dificuldade * 0.4

    # Classificar padrão
    if coerencia >= 0.85:
        padrao = 'Padrão coerente (acertou mais fáceis)'
    elif coerencia >= 0.65:
        padrao = 'Padrão consistente'
    elif coerencia >= 0.45:
        padrao = 'Padrão parcialmente coerente'
    else:
        padrao = 'Padrão incoerente (acertou mais difíceis)'

    return AnaliseCoerencia(
        coerencia=coerencia, padrao_resposta=padrao,
        taxa_muito_facil=taxa_mf, taxa_facil=taxa_f, taxa_media=taxa_m,
        taxa_dificil=taxa_d, taxa_muito_dificil=taxa_md
    )


# ════════════════════════════════════════════════════════════════════════════════
# CLASSIFICAÇÃO DE DIFICULDADE
# ════════════════════════════════════════════════════════════════════════════════

def classificar_dificuldade(pct: float) -> str:
    """Classifica uma questão por dificuldade baseado no % de acerto da turma."""
    if pct >= 0.80:
        return 'muito_facil'
    elif pct >= 0.60:
        return 'facil'
    elif pct >= 0.40:
        return 'media'
    elif pct >= 0.20:
        return 'dificil'
    else:
        return 'muito_dificil'


# ════════════════════════════════════════════════════════════════════════════════
# CÁLCULO TRI HEURÍSTICO
# ════════════════════════════════════════════════════════════════════════════════

def calcular_tri_area(
    area: str,
    acertos: int,
    tabela: TabelaReferenciaTRI,
    analise_coerencia: Optional[AnaliseCoerencia] = None,
    relacao_com_outras_areas: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Calcula TRI para uma área com base em acertos e padrão de resposta.

    Args:
        area: 'CH', 'CN', 'LC', 'MT'
        acertos: Número de acertos (0-45)
        tabela: Tabela de referência oficial
        analise_coerencia: Análise de coerência das respostas
        relacao_com_outras_areas: TRI de outras áreas para ajuste relativo

    Returns:
        Dict com tri_baseline, ajustes, tri_ajustado, motivo
    """
    # Zero acertos → TRI mediana oficial sem ajustes
    if acertos == 0:
        baseline = tabela.obter(area, 0)
        tri_med = baseline['tri_med']
        return {
            'area': area,
            'acertos': acertos,
            'tri_baseline': tri_med,
            'ajuste_coerencia': 0.0,
            'ajuste_relacao': 0.0,
            'penalidade': 0.0,
            'tri_ajustado': tri_med,
            'motivo': f'Zero acertos: TRI oficial ({tri_med:.1f}) sem ajustes'
        }

    # Buscar valores baseline
    baseline = tabela.obter(area, acertos)
    tri_med = baseline['tri_med']
    tri_min = baseline['tri_min']
    tri_max = baseline['tri_max']

    ajuste_coerencia = 0.0
    ajuste_relacao = 0.0
    penalidade = 0.0
    motivo = f'{area}: {acertos} acertos'

    # Range disponível para ajustes
    range_disponivel = tri_max - tri_min

    # [AJUSTE 1] Coerência Pedagógica
    if analise_coerencia:
        coer = analise_coerencia.coerencia

        if coer >= 0.5:
            # Padrão coerente → bônus proporcional
            fator_bonus = (coer - 0.5) * 2  # 0 a 1
            ajuste_coerencia = fator_bonus * (range_disponivel * 0.5)
            if ajuste_coerencia > 0.5:
                motivo += f' | Coerência {coer:.2f}: +{ajuste_coerencia:.1f}'
        else:
            # Padrão incoerente → penalidade proporcional
            fator_penalidade = (0.5 - coer) * 2  # 0 a 1
            penalidade = fator_penalidade * (range_disponivel * 0.5)
            if penalidade > 0.5:
                motivo += f' | Incoerência {coer:.2f}: -{penalidade:.1f}'

        # Bônus por acertar questões difíceis
        if analise_coerencia.taxa_muito_dificil > 0.3:
            bonus_dificil = analise_coerencia.taxa_muito_dificil * 20.0
            ajuste_coerencia += bonus_dificil
            motivo += f' | Bônus difíceis: +{bonus_dificil:.1f}'
        elif analise_coerencia.taxa_dificil > 0.3:
            bonus_dificil = analise_coerencia.taxa_dificil * 10.0
            ajuste_coerencia += bonus_dificil
            motivo += f' | Bônus difíceis: +{bonus_dificil:.1f}'

    # [AJUSTE 2] Relação com outras áreas (contexto)
    if relacao_com_outras_areas and len(relacao_com_outras_areas) > 0:
        media_outras = np.mean(list(relacao_com_outras_areas.values()))
        diferenca = tri_med - media_outras

        if diferenca > 50:
            ajuste_relacao = -5.0
        elif diferenca < -50:
            ajuste_relacao = 5.0

    # Aplicar ajustes
    tri_ajustado = tri_med + ajuste_coerencia + ajuste_relacao - penalidade

    # Garantir que fica dentro do range [tri_min, tri_max] da tabela oficial
    tri_ajustado = max(tri_min, min(tri_max, tri_ajustado))

    # Aplicar limite máximo oficial do ENEM (teto absoluto)
    tri_maxima_oficial = TRI_MAXIMA_OFICIAL.get(area, 1000.0)
    if tri_ajustado > tri_maxima_oficial:
        motivo += f' | LIMITADO ao máximo oficial {area}: {tri_maxima_oficial}'
        tri_ajustado = tri_maxima_oficial

    return {
        'area': area,
        'acertos': acertos,
        'tri_baseline': round(tri_med, 1),
        'ajuste_coerencia': round(ajuste_coerencia, 1),
        'ajuste_relacao': round(ajuste_relacao, 1),
        'penalidade': round(penalidade, 1),
        'tri_ajustado': round(tri_ajustado, 1),
        'motivo': motivo
    }


# ════════════════════════════════════════════════════════════════════════════════
# PROCESSADOR DE TURMA COMPLETO
# ════════════════════════════════════════════════════════════════════════════════

def processar_turma(
    alunos: List[Dict[str, Any]],
    gabarito: Dict[str, str],
    areas_config: Dict[str, Tuple[int, int]],
    tabela: TabelaReferenciaTRI
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Processa uma turma completa com coerência pedagógica.

    Fluxo:
    1. Calcula % de acerto de cada questão (turma inteira)
    2. Classifica questões por dificuldade
    3. Para cada aluno, analisa coerência e calcula TRI por área

    Args:
        alunos: Lista de dicts com campos q1, q2, ..., nome, id
        gabarito: Dict {str(num): 'letra'} ex: {'1': 'A', '2': 'B'}
        areas_config: Dict {area: (inicio, fim)} ex: {'LC': (1, 45)}
        tabela: Tabela de referência oficial

    Returns:
        Tuple (prova_analysis, resultados)
    """
    # Mapeamento de nomes de áreas para códigos padrão
    area_mapping = {
        'LC': 'LC', 'Linguagens e Códigos': 'LC', 'Linguagens': 'LC',
        'CH': 'CH', 'Ciências Humanas': 'CH',
        'CN': 'CN', 'Ciências da Natureza': 'CN',
        'MT': 'MT', 'Matemática': 'MT',
    }

    normalized_areas = {}
    for area_name, range_config in areas_config.items():
        area_code = area_mapping.get(area_name, area_name)
        if area_code in ['LC', 'CH', 'CN', 'MT']:
            normalized_areas[area_code] = tuple(range_config)

    if not normalized_areas:
        raise ValueError(
            f"areas_config inválido ou vazio. Recebido: {areas_config}. "
            "Esperado: {'LC': [1, 45], 'CH': [46, 90], ...}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # PASSO 1: CALCULAR DIFICULDADE DE CADA QUESTÃO
    # ═══════════════════════════════════════════════════════════════════════
    questoes_stats = {}
    for q_num_str in gabarito.keys():
        q_num = int(q_num_str)
        acertos = 0
        total = len(alunos)

        for aluno in alunos:
            q_key = f'q{q_num}'
            resposta = aluno.get(q_key, '')
            if resposta and resposta != 'X' and resposta == gabarito[q_num_str]:
                acertos += 1

        pct = acertos / total if total > 0 else 0.0
        questoes_stats[q_num] = {
            'acertos': acertos,
            'total': total,
            'pct': pct,
            'dificuldade': classificar_dificuldade(pct)
        }

    # Distribuição de dificuldade
    dif_counts = {
        'muito_facil': 0, 'facil': 0, 'media': 0,
        'dificil': 0, 'muito_dificil': 0
    }
    for stats in questoes_stats.values():
        dif_counts[stats['dificuldade']] += 1

    # ═══════════════════════════════════════════════════════════════════════
    # PASSO 2: PROCESSAR CADA ALUNO COM COERÊNCIA
    # ═══════════════════════════════════════════════════════════════════════
    resultados = []

    for aluno in alunos:
        nome = aluno.get('nome', aluno.get('id', ''))

        # Contar acertos por área e por dificuldade
        acertos_por_area = {area: 0 for area in normalized_areas}
        coerencia_por_area = {
            area: {dif: {'acertos': 0, 'total': 0}
                   for dif in ['muito_facil', 'facil', 'media', 'dificil', 'muito_dificil']}
            for area in normalized_areas
        }
        questoes_acertadas_por_area = {area: [] for area in normalized_areas}

        for area_code, (start, end) in normalized_areas.items():
            for i in range(start, end + 1):
                q_key = f'q{i}'
                if i in questoes_stats and q_key in aluno:
                    dif = questoes_stats[i]['dificuldade']
                    coerencia_por_area[area_code][dif]['total'] += 1

                    if aluno.get(q_key) == gabarito.get(str(i)):
                        acertos_por_area[area_code] += 1
                        coerencia_por_area[area_code][dif]['acertos'] += 1
                        questoes_acertadas_por_area[area_code].append(i)

        # Converter coerência para formato esperado
        respostas_por_dificuldade = {}
        for area_code in normalized_areas:
            rpd = {}
            for dif in ['muito_facil', 'facil', 'media', 'dificil', 'muito_dificil']:
                rpd[dif] = coerencia_por_area[area_code][dif]['acertos']

            # Peso baseado na dificuldade REAL das questões acertadas
            questoes = questoes_acertadas_por_area.get(area_code, [])
            if questoes:
                dificuldade_media = sum(
                    (1.0 - questoes_stats[q]['pct'])
                    for q in questoes if q in questoes_stats
                ) / len(questoes)
                rpd['_peso_dificuldade'] = dificuldade_media
            else:
                rpd['_peso_dificuldade'] = 0.5

            respostas_por_dificuldade[area_code] = rpd

        # Calcular TRI para cada área
        tris = {}
        detalhes = {}
        for area_code in normalized_areas:
            analise = analisar_coerencia(respostas_por_dificuldade[area_code])

            # Relação com outras áreas
            outras = {
                k: tabela.obter(k, v)['tri_med']
                for k, v in acertos_por_area.items() if k != area_code
            }

            resultado_area = calcular_tri_area(
                area=area_code,
                acertos=acertos_por_area[area_code],
                tabela=tabela,
                analise_coerencia=analise,
                relacao_com_outras_areas=outras
            )

            tri_valor = min(
                resultado_area['tri_ajustado'],
                TRI_MAXIMA_OFICIAL.get(area_code, 1000.0)
            )
            tris[area_code] = tri_valor
            detalhes[area_code] = {
                'acertos': resultado_area['acertos'],
                'baseline': resultado_area['tri_baseline'],
                'ajustes': {
                    'coerencia': resultado_area['ajuste_coerencia'],
                    'relacao': resultado_area['ajuste_relacao'],
                    'penalidade': resultado_area['penalidade'],
                },
                'tri_ajustado': resultado_area['tri_ajustado'],
                'motivo': resultado_area['motivo'],
            }

        # Calcular TRI geral (média das áreas)
        tri_geral = float(np.mean(list(tris.values()))) if tris else 0.0

        # TCT (nota bruta 0-4)
        total_acertos = sum(acertos_por_area.values())
        total_questoes = sum(end - start + 1 for start, end in normalized_areas.values())
        tct = (total_acertos / total_questoes) * 4.0 if total_questoes > 0 else 0.0

        resultado_aluno = {
            'id': aluno.get('id', ''),
            'nome': nome,
            'tct': round(tct, 2),
            'tri_geral': round(tri_geral, 1),
            'detalhes': detalhes,
        }

        # Adicionar campos por área (formato V2 compatível)
        for area_code in normalized_areas:
            resultado_aluno[f'tri_{area_code.lower()}'] = round(tris.get(area_code, 0), 1)
            resultado_aluno[f'{area_code.lower()}_acertos'] = acertos_por_area.get(area_code, 0)

        resultados.append(resultado_aluno)

    # Análise da prova
    if resultados:
        tris_gerais = [r['tri_geral'] for r in resultados]
        prova_analysis = {
            'total_alunos': len(resultados),
            'tri_medio': round(float(np.mean(tris_gerais)), 1),
            'tri_min': round(float(np.min(tris_gerais)), 1),
            'tri_max': round(float(np.max(tris_gerais)), 1),
            'tct_medio': round(float(np.mean([r['tct'] for r in resultados])), 2),
            'questoes_stats': dif_counts,
        }
    else:
        prova_analysis = {
            'total_alunos': 0, 'tri_medio': 0, 'tri_min': 0,
            'tri_max': 0, 'tct_medio': 0,
        }

    return prova_analysis, resultados
