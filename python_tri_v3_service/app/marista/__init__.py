"""Motor de correção das provas Marista (Fundamental II 6º–8º e Ensino Médio 1ª–2ª série).

Acertos por área ancorados no envelope de referência, com o método documentado
na planilha de Fundamental II. Nota provisória enquanto o envelope for
hipótese pré-calibração.
"""

from __future__ import annotations

from app.marista.correcao import (
    EstatisticaItem,
    ResultadoAluno,
    ResultadoArea,
    ResultadoTurma,
    corrigir,
    corrigir_aluno,
)
from app.marista.estrutura import Item, Prova, carregar_provas, obter_prova
from app.marista.referencia import (
    FaixaReferencia,
    ParametrosEnvelope,
    faixa,
    tabela,
    tabela_por_area,
)
from app.marista.respostas import RespostaAluno, ler_respostas

__all__ = [
    "EstatisticaItem",
    "FaixaReferencia",
    "Item",
    "ParametrosEnvelope",
    "Prova",
    "RespostaAluno",
    "ResultadoAluno",
    "ResultadoArea",
    "ResultadoTurma",
    "carregar_provas",
    "corrigir",
    "corrigir_aluno",
    "faixa",
    "ler_respostas",
    "obter_prova",
    "tabela",
    "tabela_por_area",
]
