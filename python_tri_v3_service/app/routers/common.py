"""Helpers compartilhados entre routers."""

import re
from typing import Any


def _parse_itens_anulados(raw: str) -> set[str]:
    """Normaliza a lista de itens anulados enviada no upload.

    Aceita formatos como ``141``, ``141, 142`` ou ``Questao 141``. A comparação
    é feita por número textual para funcionar tanto em headers ``141`` quanto
    ``Questao 141``.
    """

    if not raw:
        return set()
    return set(re.findall(r"\d+", str(raw)))


def _remover_itens_anulados(parsed: dict[str, Any], itens_anulados: set[str]) -> list[str]:
    """Remove colunas anuladas do payload já parseado.

    Item anulado não deve participar da calibração, do EAP nem do denominador
    bruto exibido ao professor. Os acertos dos demais itens permanecem reais.
    """

    question_ids = [str(q) for q in parsed.get("question_ids", [])]
    if not question_ids:
        return []

    keep_idx: list[int] = []
    removidos: list[str] = []
    for idx, qid in enumerate(question_ids):
        numeros = re.findall(r"\d+", qid)
        numero = numeros[-1] if numeros else qid
        if numero in itens_anulados:
            removidos.append(qid)
        else:
            keep_idx.append(idx)

    if not removidos:
        return []

    for aluno in parsed.get("alunos", []):
        respostas = aluno.get("respostas", [])
        aluno["respostas"] = [respostas[i] for i in keep_idx if i < len(respostas)]
    parsed["question_ids"] = [question_ids[i] for i in keep_idx]
    parsed["n_itens"] = len(keep_idx)
    return removidos
