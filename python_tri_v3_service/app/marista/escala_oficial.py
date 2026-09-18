"""Escala de referência ancorada nas tabelas oficiais do ENEM.

Em vez de inventar uma curva, a nota de uma prova curta é lida na curva real do
ENEM por **proporção de acertos**: um estudante que acerta 8 de 15 itens de
Linguagens é posicionado onde o ENEM posiciona quem acerta a mesma proporção
numa prova de 45 itens.

    p = acertos / nº de itens da prova
    x = p × nº de itens da prova do ENEM naquele ano e área
    nota = interpolação linear da tabela oficial em x

Consequências, todas intencionais:

- Os extremos da área são preservados exatamente: zero acerto cai na nota de
  zero acerto do ENEM, e a prova inteira cai na nota de prova inteira.
- O envelope [mínimo, máximo] é o envelope oficial, alargado por √(N/n) porque
  uma prova mais curta mede a mesma proficiência com menos informação. O
  alargamento é limitado pelos extremos da própria escala da área.
- Nada aqui é nota oficial do ENEM. É a escala pedagógica do ENEM aplicada a
  outra prova, com outro número de itens e sem calibração dos itens novos.

Fonte: ``app/referencias/tri_tabela_referencia_multiano.json``, a mesma tabela
que o fluxo de CSV do frontend usa.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.marista.referencia import FaixaReferencia, _arredondar

CAMINHO_MULTIANO = Path(__file__).resolve().parent.parent / "referencias" / "tri_tabela_referencia_multiano.json"

ANO_PADRAO = 2024  # mais recente disponível e sem medianas não-monotônicas


@dataclass(frozen=True)
class PontoOficial:
    acertos: int
    minimo: float
    mediana: float
    maximo: float


@dataclass(frozen=True)
class EscalaOficial:
    """Curva oficial de uma área num ano do ENEM."""

    ano: int
    area: str
    pontos: tuple[PontoOficial, ...]

    @property
    def n_itens(self) -> int:
        return self.pontos[-1].acertos

    @property
    def piso(self) -> float:
        """Menor valor da coluna mínima em toda a curva.

        Não é o mínimo em zero acerto: em 2024, CN tem 2 pontos e MT tem 7
        pontos cuja coluna mínima fica ABAIXO do mínimo de zero acerto (MT:
        342,8 com 6 acertos contra 369,4 com zero). É dado oficial, não erro —
        na TRI um padrão incoerente com mais acertos pode pontuar menos.
        """
        return min(p.minimo for p in self.pontos)

    @property
    def teto(self) -> float:
        """Maior valor da coluna máxima em toda a curva."""
        return max(p.maximo for p in self.pontos)

    def interpolar(self, x: float) -> PontoOficial:
        """Interpola a curva oficial numa posição contínua de acertos."""
        if not 0 <= x <= self.n_itens:
            raise ValueError(f"posição {x} fora de 0..{self.n_itens}")
        acertos = [p.acertos for p in self.pontos]
        i = bisect_right(acertos, x) - 1
        if i >= len(self.pontos) - 1:
            return self.pontos[-1]
        a, b = self.pontos[i], self.pontos[i + 1]
        if b.acertos == a.acertos:
            return a
        t = (x - a.acertos) / (b.acertos - a.acertos)
        return PontoOficial(
            acertos=int(x),
            minimo=a.minimo + t * (b.minimo - a.minimo),
            mediana=a.mediana + t * (b.mediana - a.mediana),
            maximo=a.maximo + t * (b.maximo - a.maximo),
        )


@lru_cache(maxsize=8)
def _tabela_bruta(caminho: str | None = None) -> dict:
    origem = Path(caminho) if caminho else CAMINHO_MULTIANO
    if not origem.exists():
        raise FileNotFoundError(f"tabela multiano não encontrada: {origem}")
    with origem.open(encoding="utf-8") as fh:
        return json.load(fh)


def anos_disponiveis(caminho: str | None = None) -> list[int]:
    return sorted(int(a) for a in _tabela_bruta(caminho))


def carregar_escala(area: str, ano: int = ANO_PADRAO, caminho: str | None = None) -> EscalaOficial:
    """Curva oficial de uma área. Erro explícito quando ano ou área não existem."""
    bruta = _tabela_bruta(caminho)
    chave = str(ano)
    if chave not in bruta:
        raise KeyError(f"ano {ano} não está na tabela. Disponíveis: {anos_disponiveis(caminho)}")
    if area not in bruta[chave]:
        raise KeyError(f"área {area!r} não está no ano {ano}. Disponíveis: {sorted(bruta[chave])}")

    dados = bruta[chave][area]
    pontos = tuple(
        PontoOficial(
            acertos=int(k),
            minimo=float(dados[k]["tri_min"]),
            mediana=float(dados[k]["tri_med"]),
            maximo=float(dados[k]["tri_max"]),
        )
        for k in sorted(dados, key=int)
    )
    return EscalaOficial(ano=ano, area=area, pontos=pontos)


def adaptar(
    escala: EscalaOficial,
    n_itens: int,
    alargar_envelope: bool = True,
) -> list[FaixaReferencia]:
    """Adapta a curva oficial a uma prova de ``n_itens`` na mesma área."""
    if n_itens <= 0:
        raise ValueError("n_itens precisa ser positivo")

    fator = (escala.n_itens / n_itens) ** 0.5 if alargar_envelope else 1.0
    faixas: list[FaixaReferencia] = []
    for acertos in range(n_itens + 1):
        p = acertos / n_itens
        ponto = escala.interpolar(p * escala.n_itens)
        med = ponto.mediana
        abaixo = (med - ponto.minimo) * fator
        acima = (ponto.maximo - med) * fator
        minimo = max(escala.piso, med - abaixo)
        maximo = min(escala.teto, med + acima)
        faixas.append(
            FaixaReferencia(
                acertos=acertos,
                n_itens=n_itens,
                proporcao=round(p, 4),
                minimo=_arredondar(minimo),
                mediana=_arredondar(med),
                maximo=_arredondar(maximo),
                amplitude=_arredondar((maximo - minimo) / 2),
            )
        )
    return faixas


def tabelas_da_prova(
    composicao: dict[str, int],
    ano: int = ANO_PADRAO,
    alargar_envelope: bool = True,
    caminho: str | None = None,
) -> dict[str, list[FaixaReferencia]]:
    """Tabelas adaptadas de todas as áreas de uma prova, mais o total bruto.

    O quadro GERAL soma as quatro áreas ponderando cada uma pelo seu número de
    itens na prova — é aproximação por total bruto, e a leitura pedagógica
    correta é por área.
    """
    if not composicao:
        raise ValueError("composicao vazia")

    saida = {
        area: adaptar(carregar_escala(area, ano, caminho), n, alargar_envelope)
        for area, n in composicao.items()
    }

    total = sum(composicao.values())
    geral: list[FaixaReferencia] = []
    for acertos in range(total + 1):
        p = acertos / total
        med = sum(saida[a][round(p * n)].mediana * n for a, n in composicao.items()) / total
        mn = sum(saida[a][round(p * n)].minimo * n for a, n in composicao.items()) / total
        mx = sum(saida[a][round(p * n)].maximo * n for a, n in composicao.items()) / total
        geral.append(
            FaixaReferencia(
                acertos=acertos,
                n_itens=total,
                proporcao=round(p, 4),
                minimo=_arredondar(mn),
                mediana=_arredondar(med),
                maximo=_arredondar(mx),
                amplitude=_arredondar((mx - mn) / 2),
            )
        )
    saida["GERAL"] = geral
    return saida
