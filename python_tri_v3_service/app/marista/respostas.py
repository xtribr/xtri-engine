"""Leitura das marcações dos estudantes.

Aceita CSV/TSV com uma linha por aluno: colunas de identificação
(``aluno_id``, ``nome``, ``turma``, ``matricula`` — todas opcionais exceto uma
forma de identificar) e uma coluna por questão. O cabeçalho das questões pode
ser ``1``, ``q1``, ``Q1`` ou ``questao_1``.

Regras que este módulo NÃO relaxa:

- Em branco e marcação múltipla são estados próprios, nunca viram erro
  silenciosamente. Ficam registrados em ``em_branco`` e ``multipla``.
- Letra fora do conjunto válido da prova (ex.: "E" numa prova de 4
  alternativas) é registrada como inválida, não descartada.
- Número de colunas de questão diferente do número de itens da prova
  interrompe a leitura: prova errada ou arquivo truncado.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.marista.estrutura import Prova

_COLUNAS_ID = ("aluno_id", "id", "matricula", "matrícula", "nome", "turma")
_PADRAO_QUESTAO = re.compile(r"^(?:q|quest[aã]o)?[_\s-]*(\d{1,3})$", re.IGNORECASE)
_MARCAS_MULTIPLAS = {"*", "#", "X", "XX", "MULTIPLA", "MÚLTIPLA", "ANULADA"}


@dataclass
class RespostaAluno:
    """Marcações de um estudante, já normalizadas."""

    identificacao: dict[str, str]
    marcacoes: dict[int, str | None]
    em_branco: list[int] = field(default_factory=list)
    multipla: list[int] = field(default_factory=list)
    invalidas: list[tuple[int, str]] = field(default_factory=list)

    @property
    def rotulo(self) -> str:
        for chave in ("aluno_id", "id", "matricula", "matrícula", "nome"):
            if self.identificacao.get(chave):
                return str(self.identificacao[chave])
        return "(sem identificação)"

    @property
    def n_respondidas(self) -> int:
        return sum(1 for v in self.marcacoes.values() if v is not None)

    @property
    def tem_pendencia(self) -> bool:
        return bool(self.em_branco or self.multipla or self.invalidas)


def _mapear_colunas(cabecalho: list[str]) -> tuple[list[str], dict[str, int]]:
    ids: list[str] = []
    questoes: dict[str, int] = {}
    for col in cabecalho:
        limpo = (col or "").strip()
        if not limpo:
            continue
        if limpo.lower() in _COLUNAS_ID:
            ids.append(col)
            continue
        m = _PADRAO_QUESTAO.match(limpo)
        if m:
            questoes[col] = int(m.group(1))
        else:
            ids.append(col)
    return ids, questoes


def _normalizar(valor: str | None, prova: Prova, questao: int, resp: RespostaAluno) -> str | None:
    bruto = (valor or "").strip().upper()
    if not bruto or bruto in {"-", "--", "."}:
        resp.em_branco.append(questao)
        return None
    if bruto in _MARCAS_MULTIPLAS or len(bruto) > 1:
        resp.multipla.append(questao)
        return None
    if bruto not in prova.letras_validas:
        resp.invalidas.append((questao, bruto))
        return None
    return bruto


def ler_respostas(caminho: str | Path, prova: Prova) -> list[RespostaAluno]:
    """Lê o arquivo de marcações e valida contra a estrutura da prova."""
    origem = Path(caminho)
    if not origem.exists():
        raise FileNotFoundError(f"arquivo de respostas não encontrado: {origem}")

    with origem.open(encoding="utf-8-sig", newline="") as fh:
        amostra = fh.read(4096)
        fh.seek(0)
        try:
            dialeto = csv.Sniffer().sniff(amostra, delimiters=",;\t")
        except csv.Error:
            dialeto = csv.excel
        leitor = csv.DictReader(fh, dialect=dialeto)
        if not leitor.fieldnames:
            raise ValueError(f"{origem.name}: arquivo sem cabeçalho")
        ids, questoes = _mapear_colunas(list(leitor.fieldnames))

        if len(questoes) != prova.n_itens:
            raise ValueError(
                f"{origem.name}: {len(questoes)} colunas de questão, mas "
                f"{prova.bloco}/{prova.nivel} tem {prova.n_itens} itens. "
                "Confira se o arquivo é do nível certo."
            )
        numeros = sorted(questoes.values())
        if numeros != list(range(1, prova.n_itens + 1)):
            raise ValueError(f"{origem.name}: numeração das questões fora de 1..{prova.n_itens}")

        alunos: list[RespostaAluno] = []
        for linha in leitor:
            if not any((linha.get(c) or "").strip() for c in linha):
                continue
            resp = RespostaAluno(
                identificacao={c.strip().lower(): (linha.get(c) or "").strip() for c in ids},
                marcacoes={},
            )
            for col, numero in questoes.items():
                resp.marcacoes[numero] = _normalizar(linha.get(col), prova, numero, resp)
            for lista in (resp.em_branco, resp.multipla):
                lista.sort()
            alunos.append(resp)

    if not alunos:
        raise ValueError(f"{origem.name}: nenhuma linha de aluno encontrada")
    return alunos
