"""Estrutura das provas Marista: itens, áreas, gabarito e nº de alternativas.

A fonte é ``app/referencias/marista_mapa_itens.csv``, com uma linha por item:
bloco, nível, questão, componente, área, alternativas, gabarito, escopo e
fonte. O mapa foi extraído dos cadernos e dos gabaritos oficiais — não é
inferido em tempo de execução.

Regra que o mapa resolve: no Fundamental II as áreas NÃO ocupam faixas
contíguas (Ciências Humanas está partida entre História/Geografia e Filosofia,
esta última depois de Linguagens). Contar acertos por faixa contínua produz
área errada; a contagem tem de seguir o item.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CAMINHO_PADRAO = Path(__file__).resolve().parent.parent / "referencias" / "marista_mapa_itens.csv"

AREAS_VALIDAS = ("LC", "MT", "CH", "CN")


@dataclass(frozen=True)
class Item:
    questao: int
    componente: str
    area: str
    gabarito: str


@dataclass(frozen=True)
class Prova:
    """Uma prova de um nível (ex.: bloco EF, nível '6o ano')."""

    bloco: str
    nivel: str
    alternativas: int
    itens: tuple[Item, ...]
    escopo_correcao: bool
    fonte: str

    @property
    def n_itens(self) -> int:
        return len(self.itens)

    @property
    def composicao(self) -> dict[str, int]:
        """Número de itens por área, na ordem canônica."""
        contagem = Counter(i.area for i in self.itens)
        return {a: contagem[a] for a in AREAS_VALIDAS if contagem[a]}

    @property
    def gabarito(self) -> dict[int, str]:
        return {i.questao: i.gabarito for i in self.itens}

    @property
    def area_por_questao(self) -> dict[int, str]:
        return {i.questao: i.area for i in self.itens}

    @property
    def letras_validas(self) -> tuple[str, ...]:
        return tuple("ABCDE"[: self.alternativas])

    @property
    def chute_c(self) -> float:
        """Acerto casual: 1/alternativas. 0.25 com 4 alternativas, 0.20 com 5."""
        return 1.0 / self.alternativas

    def validar(self) -> None:
        """Falha alto em qualquer inconsistência do mapa."""
        questoes = [i.questao for i in self.itens]
        esperado = list(range(1, len(questoes) + 1))
        if sorted(questoes) != esperado:
            faltando = sorted(set(esperado) - set(questoes))
            repetidas = [q for q, n in Counter(questoes).items() if n > 1]
            raise ValueError(
                f"{self.bloco}/{self.nivel}: numeração inconsistente "
                f"(faltando={faltando}, repetidas={repetidas})"
            )
        for item in self.itens:
            if item.area not in AREAS_VALIDAS:
                raise ValueError(f"{self.bloco}/{self.nivel} q{item.questao}: área {item.area!r} inválida")
            if item.gabarito not in self.letras_validas:
                raise ValueError(
                    f"{self.bloco}/{self.nivel} q{item.questao}: gabarito {item.gabarito!r} "
                    f"fora de {self.letras_validas}"
                )


@lru_cache(maxsize=4)
def carregar_provas(caminho: str | None = None) -> dict[tuple[str, str], Prova]:
    """Carrega todas as provas do mapa, indexadas por (bloco, nível)."""
    origem = Path(caminho) if caminho else CAMINHO_PADRAO
    if not origem.exists():
        raise FileNotFoundError(f"mapa de itens não encontrado: {origem}")

    agrupado: dict[tuple[str, str], list[dict]] = {}
    with origem.open(encoding="utf-8") as fh:
        for linha in csv.DictReader(fh):
            agrupado.setdefault((linha["bloco"], linha["nivel"]), []).append(linha)

    provas: dict[tuple[str, str], Prova] = {}
    for chave, linhas in agrupado.items():
        linhas.sort(key=lambda r: int(r["questao"]))
        alternativas = {int(r["alternativas"]) for r in linhas}
        if len(alternativas) != 1:
            raise ValueError(f"{chave}: nº de alternativas inconsistente no mapa: {alternativas}")
        prova = Prova(
            bloco=chave[0],
            nivel=chave[1],
            alternativas=alternativas.pop(),
            itens=tuple(
                Item(
                    questao=int(r["questao"]),
                    componente=r["componente"],
                    area=r["area"],
                    gabarito=r["gabarito"].strip().upper(),
                )
                for r in linhas
            ),
            escopo_correcao=linhas[0]["escopo_correcao"].strip().lower() == "sim",
            fonte=linhas[0]["fonte"],
        )
        prova.validar()
        provas[chave] = prova
    return provas


def obter_prova(bloco: str, nivel: str, caminho: str | None = None) -> Prova:
    """Prova de um nível. Erro explícito quando o nível não existe no mapa."""
    provas = carregar_provas(caminho)
    chave = (bloco.strip().upper(), nivel.strip())
    if chave not in provas:
        disponiveis = ", ".join(f"{b}/{n}" for b, n in sorted(provas))
        raise KeyError(f"nível {chave} não está no mapa. Disponíveis: {disponiveis}")
    return provas[chave]
