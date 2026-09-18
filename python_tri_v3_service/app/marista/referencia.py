"""Envelope de referência por área para as provas Marista.

Reproduz o método documentado na aba Premissas da planilha
``Tabela_Referencia_TRI_Fundamental_II_Provisoria.xlsx``:

    Mediana: curva suave entre 200 e 800 baseada na proporção de acertos p
    (3p² − 2p³), acrescida do deslocamento vertical do ano, que é máximo no
    centro e nulo nos extremos. Envelope: dispersão-base × ajuste da área ×
    √(10/nº de itens) × 4p(1−p).

A forma do deslocamento e do envelope é 4p(1−p); mediana, mínimo e máximo são
limitados a [200, 800]. A implementação foi conferida contra as 440 linhas da
aba "Base longa": 1734 dos 1760 valores batem exatamente, e os 26 restantes
diferem 0,1 por arredondamento de fronteira .x5 no Excel.

ATENÇÃO: isto é HIPÓTESE PRÉ-CALIBRAÇÃO. Não é nota TRI oficial nem resultado
empírico. Depois de aplicações reais, estes limites devem ser substituídos por
estimativas calibradas com as respostas dos estudantes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_HALF_UP

LIMITE_INFERIOR_PADRAO = 200.0
LIMITE_SUPERIOR_PADRAO = 800.0
DISPERSAO_BASE_PADRAO = 85.0


def _arredondar(valor: float) -> float:
    """Arredonda a uma casa decimal com desempate para cima, como o Excel."""
    return float(Decimal(repr(valor)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class FaixaReferencia:
    """Envelope de um número de acertos numa área."""

    acertos: int
    n_itens: int
    proporcao: float
    minimo: float
    mediana: float
    maximo: float
    amplitude: float

    def como_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ParametrosEnvelope:
    """Parâmetros editáveis da hipótese."""

    limite_inferior: float = LIMITE_INFERIOR_PADRAO
    limite_superior: float = LIMITE_SUPERIOR_PADRAO
    dispersao_base: float = DISPERSAO_BASE_PADRAO
    deslocamento: float = 0.0
    ajuste_area: float = 1.0

    def validar(self) -> None:
        if self.limite_superior <= self.limite_inferior:
            raise ValueError("limite_superior precisa ser maior que limite_inferior")
        if self.dispersao_base < 0:
            raise ValueError("dispersao_base não pode ser negativa")
        if self.ajuste_area <= 0:
            raise ValueError("ajuste_area precisa ser positivo")


def _forma(p: float) -> float:
    """Forma comum ao deslocamento e ao envelope: máxima no centro, nula nos extremos."""
    return 4.0 * p * (1.0 - p)


def faixa(acertos: int, n_itens: int, params: ParametrosEnvelope) -> FaixaReferencia:
    """Devolve o envelope de um número de acertos numa prova de ``n_itens``."""
    if n_itens <= 0:
        raise ValueError("n_itens precisa ser positivo")
    if not 0 <= acertos <= n_itens:
        raise ValueError(f"acertos {acertos} fora de 0..{n_itens}")
    params.validar()

    lo, hi = params.limite_inferior, params.limite_superior
    p = acertos / n_itens
    forma = _forma(p)

    bruto = lo + (hi - lo) * (3.0 * p * p - 2.0 * p**3) + params.deslocamento * forma
    mediana = _arredondar(min(max(bruto, lo), hi))
    amplitude = _arredondar(
        params.dispersao_base * params.ajuste_area * ((10.0 / n_itens) ** 0.5) * forma
    )
    minimo = _arredondar(min(max(mediana - amplitude, lo), hi))
    maximo = _arredondar(min(max(mediana + amplitude, lo), hi))

    return FaixaReferencia(
        acertos=acertos,
        n_itens=n_itens,
        proporcao=round(p, 4),
        minimo=minimo,
        mediana=mediana,
        maximo=maximo,
        amplitude=amplitude,
    )


def tabela(n_itens: int, params: ParametrosEnvelope | None = None) -> list[FaixaReferencia]:
    """Tabela completa de 0 a ``n_itens`` acertos."""
    p = params or ParametrosEnvelope()
    return [faixa(a, n_itens, p) for a in range(n_itens + 1)]


def tabela_por_area(
    composicao: dict[str, int],
    params: ParametrosEnvelope | None = None,
    incluir_geral: bool = True,
) -> dict[str, list[FaixaReferencia]]:
    """Tabelas de todas as áreas de uma prova.

    ``composicao`` mapeia código de área para número de itens, por exemplo
    ``{"LC": 15, "MT": 15, "CH": 12, "CN": 18}``. Quando ``incluir_geral`` é
    verdadeiro, acrescenta a chave ``GERAL`` com o total bruto — aproximação
    pelo total, sempre inferior à leitura por área para fins pedagógicos.
    """
    if not composicao:
        raise ValueError("composicao vazia")
    p = params or ParametrosEnvelope()
    saida = {area: tabela(n, p) for area, n in composicao.items()}
    if incluir_geral:
        saida["GERAL"] = tabela(sum(composicao.values()), p)
    return saida
