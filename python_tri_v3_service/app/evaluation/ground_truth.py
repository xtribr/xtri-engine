# app/evaluation/ground_truth.py
"""Carrega params oficiais INEP e alinha aos itens calibrados por gabarito."""
from __future__ import annotations

import json
from pathlib import Path

_ANULADO = set("X.*")

OFFICIAL_PARAMS_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts_local"
    / "data"
    / "params_oficiais.json"
)


class OfficialParamsNotFoundError(FileNotFoundError):
    """Parâmetros oficiais INEP ausentes no ambiente local."""


def load_official_params(path: Path | str = OFFICIAL_PARAMS_PATH) -> dict:
    path = Path(path)
    if not path.is_file():
        raise OfficialParamsNotFoundError(
            "Parâmetros oficiais INEP não encontrados em "
            f"{path}. Copie params_oficiais.json para esse caminho para rodar "
            "recuperação de parâmetro/atribuição; o k-fold ponta a ponta não "
            "depende desse arquivo."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def official_coprova(params: dict, year: int, area: str) -> int:
    return int(params["_meta"]["co_provas"][f"{year}_{area}"])


def official_items(params: dict, year: int, area: str) -> list[dict]:
    """Lista de 45 itens: [{posicao, a, b, c, gabarito}, ...] em ordem canônica."""
    raw = params[str(year)][area]  # [[posicao, a, b, c, gab], ...]
    return [
        {
            "posicao": int(row[0]),
            "a": float(row[1]),
            "b": float(row[2]),
            "c": float(row[3]),
            "gabarito": str(row[4]),
        }
        for row in raw
    ]


def align_official_to_gabarito(
    official: list[dict], group_gabarito: str
) -> tuple[list[int], list[dict]]:
    """Valida que o gabarito oficial casa com o do grupo, posição a posição.

    Retorna (colunas_válidas, params_oficiais_válidos), pulando itens anulados.
    Levanta ValueError se algum item não-anulado tiver gabarito divergente —
    sinal de cor/ordem trocada (aborta em vez de produzir números errados).
    """
    if len(official) != len(group_gabarito):
        raise ValueError(
            f"tamanho do gabarito difere: oficial={len(official)} "
            f"grupo={len(group_gabarito)}"
        )
    valid_cols: list[int] = []
    aligned: list[dict] = []
    for j, item in enumerate(official):
        gab_grupo = group_gabarito[j]
        if gab_grupo in _ANULADO or item["gabarito"] in _ANULADO:
            continue
        if item["gabarito"] != gab_grupo:
            raise ValueError(
                f"gabarito divergente na posição {j}: "
                f"oficial={item['gabarito']!r} grupo={gab_grupo!r}"
            )
        valid_cols.append(j)
        aligned.append(
            {
                "a": item["a"],
                "b": item["b"],
                "c": item["c"],
                "posicao": item["posicao"],
                "saturado": False,
            }
        )
    return valid_cols, aligned
