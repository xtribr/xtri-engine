"""Ancoragem TRI para provas reduzidas.

Este módulo projeta provas com 10-45 itens para a escala ENEM de 45 itens.
Não produz nota oficial INEP; produz uma projeção pedagógica auditável.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import norm

from ..tabela_referencia import TRI_MINIMA_OFICIAL, TabelaReferenciaTRI
from .transform import _interpolar_faixa_referencia

N_ITENS_REFERENCIA_ENEM = 45
MODO_SCORING_REDUZIDO = "enem_projected_reduced"
METODO_ANCORAGEM_REDUZIDA = "proporcao_acertos_45_interpolada + theta_local"
AVISO_METODOLOGICO_REDUZIDO = (
    "Projeção de nota ENEM a partir de prova reduzida. Não é nota oficial "
    "INEP; a incerteza aumenta conforme diminui o número de itens válidos."
)


def precisao_por_tamanho(n_itens_validos: int) -> str:
    """Classifica a precisão esperada pela quantidade de itens válidos."""

    if n_itens_validos >= 30:
        return "ALTA"
    if n_itens_validos >= 20:
        return "MEDIA"
    return "BAIXA"


def obter_faixa_referencia_continua(
    *,
    area: str,
    acertos_equivalentes_45: float,
    tabela: TabelaReferenciaTRI,
) -> dict[str, float | int]:
    """Interpola a faixa oficial entre pontos inteiros de acertos."""

    area = area.upper()
    pontos = sorted(tabela.lookup.get(area, {}).keys())
    if not pontos:
        raise ValueError(f"Área inválida: {area}")

    x = float(np.clip(acertos_equivalentes_45, min(pontos), max(pontos)))
    baixo = int(math.floor(x))
    alto = int(math.ceil(x))
    baixo = max(min(pontos), min(baixo, max(pontos)))
    alto = max(min(pontos), min(alto, max(pontos)))
    if baixo not in tabela.lookup[area]:
        baixo = max(p for p in pontos if p <= x)
    if alto not in tabela.lookup[area]:
        alto = min(p for p in pontos if p >= x)

    ref_baixo = tabela.obter(area, baixo)
    ref_alto = tabela.obter(area, alto)
    peso = 0.0 if alto == baixo else (x - baixo) / (alto - baixo)

    def interp(campo: str) -> float:
        return round(float(ref_baixo[campo] + peso * (ref_alto[campo] - ref_baixo[campo])), 4)

    return {
        "acertos_equivalentes_45": round(x, 4),
        "acertos_base_inferior": baixo,
        "acertos_base_superior": alto,
        "tri_min": interp("tri_min"),
        "tri_med": interp("tri_med"),
        "tri_max": interp("tri_max"),
    }


def _intervalo_95(
    *,
    nota: float,
    theta: float,
    se_theta: float,
    n_itens_validos: int,
    area: str,
    ref_min: float,
    ref_med: float,
    ref_max: float,
) -> dict[str, float]:
    """Calcula intervalo aproximado em pontos de nota para prova reduzida."""

    fator_reducao = math.sqrt(N_ITENS_REFERENCIA_ENEM / max(n_itens_validos, 1))
    se_ajustado = max(float(se_theta), 0.0) * fator_reducao
    pct_low = float(norm.cdf(theta - 1.96 * se_ajustado))
    pct_high = float(norm.cdf(theta + 1.96 * se_ajustado))
    low = _interpolar_faixa_referencia(
        ref_min=ref_min,
        ref_med=ref_med,
        ref_max=ref_max,
        pct=pct_low,
    )
    high = _interpolar_faixa_referencia(
        ref_min=ref_min,
        ref_med=ref_med,
        ref_max=ref_max,
        pct=pct_high,
    )
    tri_minima = TRI_MINIMA_OFICIAL.get(area, 200.0)
    low = max(tri_minima, min(low, ref_max))
    high = max(tri_minima, min(high, ref_max))
    return {
        "limite_inferior": round(float(min(low, nota)), 1),
        "limite_superior": round(float(max(high, nota)), 1),
        "se_theta_ajustado": round(float(se_ajustado), 4),
        "fator_reducao": round(float(fator_reducao), 4),
    }


def ancorar_nota_reduzida(
    *,
    theta: float,
    se_theta: float,
    area: str,
    acertos_observados: int,
    n_itens_validos: int,
    tabela: TabelaReferenciaTRI,
) -> dict[str, Any]:
    """Projeta uma resposta reduzida na escala ENEM de 45 itens."""

    if n_itens_validos < 10:
        raise ValueError("Escala reduzida exige mínimo 10 itens válidos.")
    if acertos_observados < 0 or acertos_observados > n_itens_validos:
        raise ValueError("Acertos observados fora do intervalo válido.")

    pct_acerto = acertos_observados / n_itens_validos
    acertos_equivalentes = pct_acerto * N_ITENS_REFERENCIA_ENEM
    ref = obter_faixa_referencia_continua(
        area=area,
        acertos_equivalentes_45=acertos_equivalentes,
        tabela=tabela,
    )
    ref_min = float(ref["tri_min"])
    ref_med = float(ref["tri_med"])
    ref_max = float(ref["tri_max"])
    pct_theta = float(norm.cdf(theta))
    nota = _interpolar_faixa_referencia(
        ref_min=ref_min,
        ref_med=ref_med,
        ref_max=ref_max,
        pct=pct_theta,
    )
    nota = max(TRI_MINIMA_OFICIAL.get(area, 200.0), min(float(nota), ref_max))
    return {
        "acertos_observados": int(acertos_observados),
        "n_itens_validos": int(n_itens_validos),
        "pct_acerto": round(float(pct_acerto), 4),
        "acertos_equivalentes_45": round(float(acertos_equivalentes), 4),
        "theta": round(float(theta), 6),
        "se_theta": round(float(se_theta), 4),
        "nota_enem_projetada": round(float(nota), 1),
        "intervalo_95": _intervalo_95(
            nota=nota,
            theta=float(theta),
            se_theta=float(se_theta),
            n_itens_validos=n_itens_validos,
            area=area,
            ref_min=ref_min,
            ref_med=ref_med,
            ref_max=ref_max,
        ),
        "precisao": precisao_por_tamanho(n_itens_validos),
        "faixa_referencia": ref,
        "aviso_metodologico": AVISO_METODOLOGICO_REDUZIDO,
    }
