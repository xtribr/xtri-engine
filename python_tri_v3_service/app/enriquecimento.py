"""
XTRI Engine V2.1 — Enriched output builder.

Takes the raw V3 calibration results and builds 9 dimensions per student:

  1. raw                     — acertos, total, %
  2. irt                     — theta, SE, z-cohort, percentil
  3. performance             — nota TRI + ancoragem
  4. desempenho_pedagogico   — nível + interpretação + comparação nacional
  5. posicao_relativa        — rank + percentis
  6. projecao_aprovacao      — SISU cortes + cursos prováveis
  7. recomendacoes           — pontos fortes/atenção + estudo sugerido
  8. primi                   — acert_dif + interpretação multidimensional
  9. referencia_inep         — citações + validação XTRI

Used by main.py to populate `resultados_estruturados` in the response.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import norm

from .tabela_referencia import TRI_MAXIMA_OFICIAL, TRI_MINIMA_OFICIAL


# ════════════════════════════════════════════════════════════════════════════════
# REFERÊNCIAS NACIONAIS (microdata 2024 — sample 100K-300K)
# ════════════════════════════════════════════════════════════════════════════════

NACIONAL_STATS = {
    "LC": {"mediana": 524.7, "p99": 664.5, "p99_9": 707.8, "max_real": 795.8, "p95": 627.4, "media": 524.7, "dp": 68.8},
    "CH": {"mediana": 510.0, "p99": 700.2, "p99_9": 745.6, "max_real": 829.1, "p95": 656.6, "media": 511.9, "dp": 75.0},
    "CN": {"mediana": 488.0, "p99": 685.1, "p99_9": 753.4, "max_real": 868.7, "p95": 630.1, "media": 494.0, "dp": 79.0},
    "MT": {"mediana": 498.8, "p99": 829.0, "p99_9": 918.2, "max_real": 983.9, "p95": 737.7, "media": 527.2, "dp": 120.0},
}


# ════════════════════════════════════════════════════════════════════════════════
# NÍVEIS PEDAGÓGICOS (alinhados com INEP/SAEB)
# ════════════════════════════════════════════════════════════════════════════════

NIVEIS_PEDAGOGICOS = {
    "AVANCADO": {
        "min_score": 700,
        "interpretacao": "Acima do esperado para o ENEM",
        "descricao": (
            "Estudante demonstra competências acima do nível esperado. "
            "Capaz de analisar situações complexas, aplicar conhecimento em "
            "contextos novos e resolver problemas de alta dificuldade. "
            "Posicionado entre os melhores candidatos do país."
        ),
    },
    "PROFICIENTE": {
        "min_score": 600,
        "interpretacao": "No nível esperado pela BNCC",
        "descricao": (
            "Estudante atende às expectativas da Base Nacional Comum Curricular. "
            "Domina os conteúdos essenciais e aplica-os em situações típicas. "
            "Competitivo para a maioria dos cursos do SISU."
        ),
    },
    "BASICO": {
        "min_score": 500,
        "interpretacao": "Conhecimento básico, próximo da média nacional",
        "descricao": (
            "Estudante possui conhecimentos fundamentais mas com lacunas em "
            "conteúdos mais elaborados. Próximo da média nacional do ENEM. "
            "Acesso restrito a cursos menos concorridos do SISU."
        ),
    },
    "ELEMENTAR": {
        "min_score": 400,
        "interpretacao": "Lacunas significativas",
        "descricao": (
            "Estudante apresenta lacunas significativas em conteúdos básicos. "
            "Demonstra apenas competências introdutórias. "
            "Necessita reforço amplo antes de buscar vagas em universidades públicas."
        ),
    },
    "INICIANTE": {
        "min_score": 0,
        "interpretacao": "Início do processo de aprendizagem",
        "descricao": (
            "Estudante está no início do processo de construção do conhecimento. "
            "Necessita reforço sistemático em conteúdos fundamentais. "
            "Foco em consolidar bases antes de buscar provas de alta complexidade."
        ),
    },
}


def classificar_nivel(nota: float) -> str:
    if nota >= 700: return "AVANCADO"
    if nota >= 600: return "PROFICIENTE"
    if nota >= 500: return "BASICO"
    if nota >= 400: return "ELEMENTAR"
    return "INICIANTE"


# ════════════════════════════════════════════════════════════════════════════════
# SISU CORTES (valores aproximados, atualizados para 2024)
# ════════════════════════════════════════════════════════════════════════════════

SISU_CORTES = {
    "baixa": (400, 500),    # Letras, Pedagogia, Filosofia, Música
    "media": (500, 650),    # Adm, Contabilidade, Engenharias periféricas
    "alta": (650, 750),     # Engenharias top, Direito, Computação
    "topo": (750, 1000),    # Medicina, Direito top
}

CURSOS_POR_FAIXA = {
    "baixa": [
        "Letras", "Pedagogia", "Filosofia", "Sociologia",
        "História", "Geografia", "Música", "Artes Visuais",
    ],
    "media": [
        "Administração", "Ciências Contábeis", "Economia",
        "Psicologia", "Enfermagem", "Nutrição", "Educação Física",
        "Engenharia Ambiental", "Engenharia de Produção",
    ],
    "alta": [
        "Engenharia Civil", "Engenharia Mecânica", "Engenharia Elétrica",
        "Ciência da Computação", "Sistemas de Informação",
        "Direito", "Arquitetura", "Odontologia",
    ],
    "topo": [
        "Medicina", "Direito (USP/UFRJ/UFMG)",
        "Engenharia (USP/ITA/IME)", "Ciência da Computação (USP/Unicamp)",
    ],
}


def projetar_aprovacao(nota_geral: float) -> Dict[str, Any]:
    """Estima as faixas SISU que o aluno atende com base na nota."""
    return {
        "sisu_baixa_concorrencia": nota_geral >= SISU_CORTES["baixa"][0],
        "sisu_media_concorrencia": nota_geral >= SISU_CORTES["media"][0],
        "sisu_alta_concorrencia":  nota_geral >= SISU_CORTES["alta"][0],
        "sisu_topo":               nota_geral >= SISU_CORTES["topo"][0],
        "cursos_provaveis": _cursos_provaveis(nota_geral),
        "cursos_disputados": _cursos_disputados(nota_geral),
        "observacao": _observacao_aprovacao(nota_geral),
    }


def _cursos_provaveis(nota: float) -> List[str]:
    if nota >= 750: return CURSOS_POR_FAIXA["topo"][:4] + CURSOS_POR_FAIXA["alta"][:4]
    if nota >= 650: return CURSOS_POR_FAIXA["alta"]
    if nota >= 500: return CURSOS_POR_FAIXA["media"]
    if nota >= 400: return CURSOS_POR_FAIXA["baixa"]
    return []


def _cursos_disputados(nota: float) -> List[str]:
    """Cursos onde o aluno está no limite (precisa subir 50-100 pts)."""
    if nota >= 700: return CURSOS_POR_FAIXA["topo"][:2]
    if nota >= 600: return CURSOS_POR_FAIXA["alta"][:3]
    if nota >= 450: return CURSOS_POR_FAIXA["media"][:3]
    return CURSOS_POR_FAIXA["baixa"][:3]


def _observacao_aprovacao(nota: float) -> str:
    if nota >= 750: return "Score competitivo para os cursos mais disputados (Medicina, Direito top)"
    if nota >= 650: return "Score competitivo para Engenharias e Direito em federais"
    if nota >= 550: return "Score competitivo para cursos de média concorrência"
    if nota >= 450: return "Score atende cursos menos concorridos no SISU"
    if nota >= 350: return "Próximo do limite mínimo para SISU em cursos acessíveis"
    return "Necessita melhoria significativa para acessar cursos federais"


# ════════════════════════════════════════════════════════════════════════════════
# RECOMENDAÇÕES PEDAGÓGICAS
# ════════════════════════════════════════════════════════════════════════════════

def gerar_recomendacoes(
    nota: float,
    acertos: int,
    total: int,
    acert_dif: float,
    nivel: str,
    n_itens_faceis_acertados: int,
    n_itens_dificeis_acertados: int,
    n_itens_dificeis_total: int,
) -> Dict[str, Any]:
    pontos_fortes = []
    pontos_atencao = []

    pct = acertos / max(total, 1)
    if pct >= 0.8:
        pontos_fortes.append("Alta taxa de acerto geral (≥80%)")
    elif pct >= 0.6:
        pontos_fortes.append("Boa taxa de acerto (≥60%)")

    if acert_dif >= 0.6:
        pontos_fortes.append("Domínio em itens de alta dificuldade — perfil consistente")
    elif acert_dif >= 0.3 and pct >= 0.5:
        pontos_fortes.append("Padrão de respostas coerente")
    else:
        pontos_atencao.append("Baixo desempenho em itens difíceis (acert_dif < 0.3)")

    if nivel in ("INICIANTE", "ELEMENTAR"):
        pontos_atencao.append("Lacunas em conteúdos fundamentais")
        pontos_atencao.append("Recomenda-se base de conhecimento antes de simulados")
    elif nivel == "BASICO":
        pontos_atencao.append("Refinar interpretação de itens de dificuldade média")

    # Foco de estudo baseado no nível
    if nivel == "AVANCADO":
        foco = "Aprofundamento em itens de alta dificuldade e revisão de pontos específicos"
        tempo = 8
        estrategia = "Resolver provas anteriores do ENEM, focar em questões difíceis e treinar redação"
    elif nivel == "PROFICIENTE":
        foco = "Refinamento técnico e prática de questões de discriminação alta"
        tempo = 12
        estrategia = "Combinar revisão sistemática + simulados semanais"
    elif nivel == "BASICO":
        foco = "Consolidação dos conteúdos essenciais e prática frequente"
        tempo = 16
        estrategia = "Revisão metodológica + reforço em pontos fracos identificados"
    elif nivel == "ELEMENTAR":
        foco = "Reforço amplo em fundamentos antes de aprofundar"
        tempo = 20
        estrategia = "Curso de revisão completo + acompanhamento individualizado"
    else:
        foco = "Reconstrução das bases do conhecimento"
        tempo = 25
        estrategia = "Acompanhamento pedagógico contínuo + reforço escolar"

    return {
        "pontos_fortes": pontos_fortes,
        "pontos_atencao": pontos_atencao,
        "foco_estudo": foco,
        "tempo_estudo_estimado_h": tempo,
        "estrategia": estrategia,
    }


# ════════════════════════════════════════════════════════════════════════════════
# REFERÊNCIAS INEP
# ════════════════════════════════════════════════════════════════════════════════

PAPERS_INEP = [
    {
        "autor": "Primi, R. & Cicchetto, A.",
        "ano": "2019",
        "titulo": "Como os escores do ENEM são atribuídos pela TRI?",
        "fonte": "Anais ABAVE - Universidade São Francisco",
        "tema": "Differential scoring, multidimensionalidade do ENEM",
    },
    {
        "autor": "Klein, R.",
        "ano": "2009",
        "titulo": "Utilização da Teoria de Resposta ao Item no SAEB",
        "fonte": "Ensaio: Avaliação e Políticas Públicas em Educação, 17(65)",
        "tema": "Metodologia TRI no Brasil",
    },
    {
        "autor": "Klein, R.",
        "ano": "2013",
        "titulo": "Alguns aspectos da TRI relativos à estimação das proficiências",
        "fonte": "Ensaio: Avaliação e Políticas Públicas em Educação, 21(78)",
        "tema": "Estimação EAP e MML",
    },
    {
        "autor": "Birnbaum, A.",
        "ano": "1968",
        "titulo": "Some latent trait models and their use in inferring an examinee's ability",
        "fonte": "in Lord & Novick, Statistical theories of mental test scores",
        "tema": "Modelo 3PL original",
    },
    {
        "autor": "Hambleton, Swaminathan & Rogers",
        "ano": "1991",
        "titulo": "Fundamentals of item response theory",
        "fonte": "Sage Publications",
        "tema": "Referência canônica de IRT",
    },
]

XTRI_VALIDATION = {
    "n_alunos_validados": 360000,
    "anos_validados": [2022, 2023, 2024],
    "areas_validadas": ["LC", "CH", "CN", "MT"],
    "correlacao_vs_inep_oficial": 1.000000,
    "mae_vs_inep": 0.025,
    "level_accuracy_pct": 99.96,
    "fonte_microdata": "https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados",
}


# ════════════════════════════════════════════════════════════════════════════════
# BUILDER PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

def build_resultado_estruturado(
    aluno_id: str,
    acertos: int,
    total_itens: int,
    theta: float,
    se: float,
    nota: float,
    cohort_mean: float,
    cohort_std: float,
    rank_in_cohort: int,  # 1 = melhor
    total_in_cohort: int,
    area: str,
    ano: int,
    sigma_oficial: float,
    mu_oficial: float,
    ref_min: float,
    ref_med: float,
    ref_max: float,
    acert_dif: float,
    n_itens_faceis_acertados: int = 0,
    n_itens_medios_acertados: int = 0,
    n_itens_dificeis_acertados: int = 0,
    n_itens_dificeis_total: int = 0,
) -> Dict[str, Any]:
    """Builds the 9-dimension structured result for a single student."""

    nat = NACIONAL_STATS.get(area, {"mediana": 500, "p99": 700, "p99_9": 800, "max_real": 1000, "p95": 650, "media": 500, "dp": 100})

    # IRT z-score and cohort percentile
    std = cohort_std if cohort_std > 1e-6 else 1.0
    z_cohort = (theta - cohort_mean) / std
    pct_cohort = float(norm.cdf(z_cohort)) * 100

    # National percentile estimate (using national distribution)
    nat_z = (nota - nat["media"]) / max(nat["dp"], 1.0)
    pct_nacional = float(norm.cdf(nat_z)) * 100

    # Pedagogical level
    nivel = classificar_nivel(nota)
    nivel_info = NIVEIS_PEDAGOGICOS[nivel]

    # Primi interpretation
    if acert_dif >= 0.6:
        primi_interp = "Alto desempenho em itens difíceis. Padrão consistente."
    elif acert_dif >= 0.3:
        primi_interp = "Desempenho moderado em itens difíceis."
    else:
        primi_interp = "Baixo desempenho em itens difíceis. Pode ser subestimado pelo modelo unidimensional."

    return {
        "aluno_id": aluno_id,
        "raw": {
            "acertos": int(acertos),
            "total_itens": int(total_itens),
            "pct_acerto": round(acertos / max(total_itens, 1), 4),
        },
        "irt": {
            "theta": round(float(theta), 4),
            "se": round(float(se), 4),
            "z_cohort": round(float(z_cohort), 4),
            "percentil_cohort": round(float(pct_cohort), 2),
            "metodo_calibracao": "MML + L-BFGS-B + EAP (Gauss-Hermite quadrature, 41-61 nodes)",
        },
        "performance": {
            "nota_tri": round(float(nota), 1),
            "metodo_ancoragem": "Cohort-Anchored Distribution (z-score → tri_min/med/max)",
            "ancoragem": {
                "tri_min": round(float(ref_min), 1),
                "tri_med": round(float(ref_med), 1),
                "tri_max": round(float(ref_max), 1),
                "tri_maxima_oficial_area": float(TRI_MAXIMA_OFICIAL.get(area, 1000.0)),
                "tri_minima_oficial_area": float(TRI_MINIMA_OFICIAL.get(area, 200.0)),
            },
        },
        "desempenho_pedagogico": {
            "nivel": nivel,
            "interpretacao": nivel_info["interpretacao"],
            "descricao_pedagogica": nivel_info["descricao"],
            "comparacao_nacional": {
                "mediana_nacional": nat["mediana"],
                "p99_real": nat["p99"],
                "p99_9_real": nat["p99_9"],
                "max_real": nat["max_real"],
                "percentil_estimado_nacional": round(pct_nacional, 2),
            },
        },
        "posicao_relativa": {
            "rank_cohort": int(rank_in_cohort),
            "total_cohort": int(total_in_cohort),
            "percentil_cohort": round(float((total_in_cohort - rank_in_cohort + 1) / total_in_cohort * 100), 2),
            "alunos_acima_cohort": int(rank_in_cohort - 1),
            "alunos_abaixo_cohort": int(total_in_cohort - rank_in_cohort),
            "posicao_nacional_estimada_pct": round(float(pct_nacional), 2),
        },
        "projecao_aprovacao": projetar_aprovacao(nota),
        "recomendacoes": gerar_recomendacoes(
            nota, acertos, total_itens, acert_dif, nivel,
            n_itens_faceis_acertados, n_itens_dificeis_acertados, n_itens_dificeis_total,
        ),
        "primi": {
            "acert_dif": round(float(acert_dif), 4),
            "interpretacao": primi_interp,
            "itens_faceis_acertados": int(n_itens_faceis_acertados),
            "itens_medios_acertados": int(n_itens_medios_acertados),
            "itens_dificeis_acertados": int(n_itens_dificeis_acertados),
            "fator_secundario_detectado": False,  # placeholder for future MIRT
            "nota_referencia": (
                "Baseado em Primi & Cicchetto (2019). Estudantes com alto acert_dif "
                "e baixos acertos totais podem ser subestimados pelo modelo 3PL unidimensional."
            ),
        },
        "referencia_inep": {
            "ano_referencia": int(ano),
            "area": area,
            "transformacao_oficial": {
                "formula": f"Nota = {sigma_oficial:.4f} × theta + {mu_oficial:.4f}",
                "sigma": round(float(sigma_oficial), 4),
                "mu": round(float(mu_oficial), 4),
                "fonte": "Empirical regression vs ENEM microdata (5,000 students per area-year)",
            },
            "metodologia_inep_url": "https://www.gov.br/inep/pt-br/areas-de-atuacao/avaliacao-e-exames-educacionais/enem",
            "papers_relacionados": PAPERS_INEP,
            "validacao_xtri": XTRI_VALIDATION,
        },
        "metadata": {
            "calculado_em": datetime.now().isoformat(),
            "metodo_versao": "v2.1-enriched",
            "ferramenta": "XTRI Engine",
            "github": "https://github.com/xtribr/xtri-engine",
        },
    }


def build_metodologia_banner(
    area: str,
    ano: int,
    n_alunos: int,
    cohort_mean: float,
    cohort_std: float,
    acertos_medio: float,
    n_itens: int,
    nota_media: float,
) -> Dict[str, Any]:
    """Builds the methodology banner shown above the results."""
    nat = NACIONAL_STATS.get(area, {"p95": 600, "mediana": 500})
    pct_acerto_medio = acertos_medio / max(n_itens, 1) * 100

    avisos = []
    is_strong_cohort = nota_media >= nat.get("p95", 600)
    if is_strong_cohort:
        avisos.append({
            "tipo": "atencao",
            "mensagem": (
                f"Esta turma é estatisticamente forte (média {acertos_medio:.0f}/{n_itens} = "
                f"{pct_acerto_medio:.0f}%, acima do P95 nacional). "
                f"V3 Local pode inflar scores em populações não-representativas. "
                f"Para precisão absoluta vs INEP, use modo V4 INEP (se as questões forem do banco oficial)."
            ),
        })

    return {
        "metodo_principal": "V3 Local — Cohort-Anchored Distribution",
        "descricao_curta": "Calibração local 3PL + EAP + ancoragem por z-score na coorte",
        "n_alunos": n_alunos,
        "n_itens": n_itens,
        "cohort_stats": {
            "theta_medio": round(float(cohort_mean), 4),
            "theta_dp": round(float(cohort_std), 4),
            "acertos_medio": round(float(acertos_medio), 1),
            "pct_acerto_medio": round(float(pct_acerto_medio), 1),
            "nota_media": round(float(nota_media), 1),
        },
        "comparacao_nacional": {
            "mediana_nacional": nat["mediana"],
            "diferenca_vs_mediana": round(float(nota_media - nat["mediana"]), 1),
            "is_strong_cohort": is_strong_cohort,
        },
        "validacao_xtri": {
            "vs_inep_oficial": "MAE 0.025 pontos contra 360.000 alunos reais (correlação 1.000000)",
            "modo_v4_disponivel": True,
        },
        "avisos": avisos,
    }


def build_coorte_stats(
    thetas: List[float],
    notas: List[float],
    acertos: List[int],
    n_itens: int,
    area: str,
) -> Dict[str, Any]:
    """Builds cohort-level statistics for the response."""
    arr_t = np.array(thetas)
    arr_n = np.array(notas)
    arr_a = np.array(acertos)
    nat = NACIONAL_STATS.get(area, {})

    return {
        "n": int(len(thetas)),
        "theta": {
            "mean": round(float(arr_t.mean()), 4),
            "std": round(float(arr_t.std()), 4),
            "min": round(float(arr_t.min()), 4),
            "max": round(float(arr_t.max()), 4),
        },
        "nota": {
            "mean": round(float(arr_n.mean()), 1),
            "std": round(float(arr_n.std()), 1),
            "min": round(float(arr_n.min()), 1),
            "max": round(float(arr_n.max()), 1),
            "median": round(float(np.median(arr_n)), 1),
        },
        "acertos": {
            "mean": round(float(arr_a.mean()), 1),
            "std": round(float(arr_a.std()), 1),
            "min": int(arr_a.min()),
            "max": int(arr_a.max()),
            "pct_mean": round(float(arr_a.mean() / max(n_itens, 1) * 100), 1),
        },
        "comparacao_nacional": {
            "mediana_nacional": nat.get("mediana"),
            "diferenca_mean_vs_nacional": round(float(arr_n.mean() - nat.get("mediana", 500)), 1),
            "is_above_p95": float(arr_n.mean()) >= nat.get("p95", 600),
        },
    }
