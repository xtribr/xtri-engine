"""
Transformações do núcleo IRT.

O motor sempre ancora notas dentro da faixa oficial por acertos
(`tri_min..tri_med..tri_max`). Quando há curva condicional (derivada dos
parâmetros dos itens, ver `conditional.py`), a posição na faixa é relativa a
quem teve o mesmo número de acertos. Sem curva, vale a regra legada: o corte
entre `MIN-MED` e `MED-MAX` vem do theta absoluto do modelo.
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np
from scipy.stats import norm

from ..tabela_referencia import (
    TRI_MAXIMA_OFICIAL,
    TRI_MINIMA_OFICIAL,
    TabelaReferenciaTRI,
)
from .conditional import CurvaCondicional, nota_por_posicao, posicao_relativa
from .contracts import ThetaEstimate

DEFAULT_REFERENCE_RANGE_TOLERANCE = 0.03


def _interpolar_faixa_referencia(
    *,
    ref_min: float,
    ref_med: float,
    ref_max: float,
    pct: float,
) -> float:
    """Interpola a nota dentro da faixa `min..med..max` a partir de um percentil."""

    pct = float(np.clip(pct, 0.0, 1.0))
    if ref_min == ref_max:
        return float(ref_med)
    if pct <= 0.5:
        local_pct = pct / 0.5
        return float(ref_min + local_pct * (ref_med - ref_min))
    local_pct = (pct - 0.5) / 0.5
    return float(ref_med + local_pct * (ref_max - ref_med))


def _clip_nota_na_faixa_referencia(
    *,
    nota: float,
    area: str,
    ref_min: float,
    ref_max: float,
    tolerancia_pct: float,
) -> float:
    """Aplica clip nos limites da faixa por acertos com pequena tolerância opcional.

    ``ref_min`` e ``ref_max`` já vêm da tabela de referência do **ano escolhido**
    (via ``TabelaReferenciaTRI.obter``), então eles são o teto/piso empírico
    corretos para aquele ano naquele número de acertos. O clip global
    ``TRI_MAXIMA_OFICIAL`` era derivado só do MT 2024 (961.9 → 960) e, quando
    aplicado em cima desses ``ref_max``, estrangulava as notas dos anos em que
    o teto real é maior — por exemplo MT 2017/2018/2022 com ``tri_max`` de
    983-988 em 41 acertos eram todos limitados em 960, fazendo alunos com
    θ alto caírem em uma nota idêntica independente do ano escolhido.
    Usamos o piso ``TRI_MINIMA_OFICIAL`` para garantir que nenhuma área vá
    abaixo do mínimo oficial, e um teto absoluto de 1000 como rede de
    segurança contra pathologias de cálculo — o ``ref_max`` por ano é quem
    manda dentro desse envelope.
    """

    span = max(ref_max - ref_min, 0.0)
    margem = span * max(tolerancia_pct, 0.0)
    tri_minima = TRI_MINIMA_OFICIAL.get(area, 200.0)
    lower = max(tri_minima, ref_min - margem)
    upper = min(1000.0, ref_max + margem)
    return max(lower, min(float(nota), upper))


def _theta_referencia_fixo(
    *,
    theta: float,
    transformacao: Optional[Dict[str, Union[float, str]]] = None,
) -> float:
    """
    Resolve o theta em escala de referência fixa.

    Por padrão, o próprio theta EAP já está na escala latente N(0,1) do modelo
    e o corte oficial entre `MIN-MED` e `MED-MAX` é feito em 0. Caso exista um
    artefato futuro com centro/escala explícitos, ele pode sobrescrever isso.
    """

    theta_ref = float(theta)
    if transformacao is None:
        return theta_ref

    centro = float(transformacao.get("theta_ref_centro", 0.0))
    escala = abs(float(transformacao.get("theta_ref_escala", 1.0)))
    if escala < 1e-6:
        escala = 1.0
    return (theta_ref - centro) / escala


def ancorar_nota_individual(
    *,
    theta: float,
    area: str,
    acertos: int,
    tabela: TabelaReferenciaTRI,
    transformacao: Optional[Dict[str, Union[float, str]]] = None,
    tolerancia_pct: float = DEFAULT_REFERENCE_RANGE_TOLERANCE,
    curva: Optional[CurvaCondicional] = None,
) -> float:
    """
    Ancora um único aluno na faixa oficial por acertos.

    Com `curva`, usa a posição relativa ao mesmo número de acertos (regra
    condicional). Sem ela, mantém a regra legada, na ordem de preferência:
    1. theta em escala de referência fixa explícita no artefato
    2. theta bruto do modelo, que já vive na escala latente N(0,1)
    """

    ref = tabela.obter(area, acertos)
    ref_min = float(ref["tri_min"])
    ref_med = float(ref["tri_med"])
    ref_max = float(ref["tri_max"])

    if curva is not None:
        obter_quantis = getattr(tabela, "obter_quantis", None)
        nota = nota_por_posicao(
            posicao_relativa(theta, acertos, curva),
            ref_min=ref_min,
            ref_med=ref_med,
            ref_max=ref_max,
            quantis=obter_quantis(area, acertos) if obter_quantis else None,
        )
        tri_minima = TRI_MINIMA_OFICIAL.get(area, 200.0)
        return max(tri_minima, min(float(nota), 1000.0))

    if ref_min == ref_max:
        return _clip_nota_na_faixa_referencia(
            nota=ref_med,
            area=area,
            ref_min=ref_min,
            ref_max=ref_max,
            tolerancia_pct=0.0,
        )

    theta_ref = _theta_referencia_fixo(theta=theta, transformacao=transformacao)
    pct = float(norm.cdf(theta_ref))
    nota = _interpolar_faixa_referencia(
        ref_min=ref_min,
        ref_med=ref_med,
        ref_max=ref_max,
        pct=pct,
    )
    return _clip_nota_na_faixa_referencia(
        nota=nota,
        area=area,
        ref_min=ref_min,
        ref_max=ref_max,
        tolerancia_pct=tolerancia_pct,
    )


def ancorar_notas(
    thetas: Sequence[ThetaEstimate],
    tabela: TabelaReferenciaTRI,
    area: str,
    curva: Optional[CurvaCondicional] = None,
) -> list[ThetaEstimate]:
    """Mapeia θ -> nota; com `curva`, pela posição relativa ao mesmo acerto."""

    if not thetas:
        return []

    resultados: list[ThetaEstimate] = []
    for t in thetas:
        nota = ancorar_nota_individual(
            theta=float(t["theta"]),
            area=area,
            acertos=int(t["acertos"]),
            tabela=tabela,
            tolerancia_pct=0.0,
            curva=curva,
        )
        resultados.append({**t, "nota_ancorada": round(float(nota), 1)})

    return resultados


def calcular_transformacao_theta_nota(
    thetas: Sequence[float],
    notas: Sequence[float],
) -> Dict[str, Union[float, str]]:
    """
    Ajusta uma transformação linear `nota = mu + sigma * theta`.

    Neste PR, a transformação é derivada dos resultados do próprio lote para
    que o scoring individual consiga reutilizar a mesma escala sem cair no
    ponto médio por acertos.
    """

    if len(thetas) != len(notas) or not thetas:
        return {"mu": 500.0, "sigma": 100.0, "metodo": "fallback_padrao"}

    theta_arr = np.array(thetas, dtype=float)
    nota_arr = np.array(notas, dtype=float)
    theta_var = float(np.var(theta_arr))

    if theta_var < 1e-9:
        return {
            "mu": round(float(np.mean(nota_arr)), 4),
            "sigma": 100.0,
            "theta_ref_centro": 0.0,
            "theta_ref_escala": 1.0,
            "theta_ref_origem": "escala_irt_fixa",
            "metodo": "fallback_theta_constante",
        }

    sigma = float(np.cov(theta_arr, nota_arr, bias=True)[0, 1] / theta_var)
    if sigma <= 0:
        sigma = 100.0
        mu = float(np.mean(nota_arr))
        metodo = "fallback_sigma_nao_positivo"
    else:
        mu = float(np.mean(nota_arr) - sigma * np.mean(theta_arr))
        metodo = "regressao_linear_lote"

    return {
        "mu": round(mu, 4),
        "sigma": round(sigma, 4),
        "theta_ref_centro": 0.0,
        "theta_ref_escala": 1.0,
        "theta_ref_origem": "escala_irt_fixa",
        "metodo": metodo,
    }


def aplicar_transformacao_theta(
    theta: float,
    area: str,
    mu: float,
    sigma: float,
) -> float:
    """Aplica `nota = mu + sigma * theta` com clip nos limites oficiais."""

    nota = mu + sigma * theta
    tri_maxima = TRI_MAXIMA_OFICIAL.get(area, 1000.0)
    tri_minima = TRI_MINIMA_OFICIAL.get(area, 200.0)
    return max(tri_minima, min(float(nota), tri_maxima))


def nota_individual_pelo_ponto_medio(
    area: str,
    acertos: int,
    tabela: TabelaReferenciaTRI,
) -> float:
    """
    Retorna o ponto médio oficial da faixa de acertos.

    Este é o fallback atual do scoring individual. Ele foi isolado aqui porque
    será substituído por uma transformação estável baseada no artefato da
    calibração unidimensional.
    """

    ref = tabela.obter(area, acertos)
    nota = ref["tri_med"]
    tri_maxima = TRI_MAXIMA_OFICIAL.get(area, 1000.0)
    return min(nota, tri_maxima)
