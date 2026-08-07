"""
Helpers para construção e validação de Q-matrix.

Este módulo evita embutir suposições pedagógicas inventadas no código. O único
builder automático usa metadados reais do item bank (`numero_habilidade`).
Quando esses metadados não existem, a saída suportada é um template vazio.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Optional

from .contracts import (
    MIRTFactorSpec,
    MIRTModelSpec,
    MIRTQMatrix,
    MIRTQMatrixRow,
)

DEFAULT_QMATRIX_VERSION = "1.0.0"
DEFAULT_GENERAL_FACTOR_ID = "G"


def _normalize_area(area: str) -> str:
    return area.strip().upper()


def _factor_id_from_habilidade(area: str, numero_habilidade: int) -> str:
    """Cria um ID de fator estável a partir da habilidade real do item bank."""

    return f"{area}_H{int(numero_habilidade):02d}"


def validate_qmatrix(qmatrix: MIRTQMatrix, *, allow_template: bool = True) -> MIRTQMatrix:
    """
    Valida consistência estrutural da Q-matrix.

    Regras:
    - todos os itens devem ser únicos
    - toda carga deve apontar para fator declarado
    - fora de template, cada item precisa carregar ao menos um fator
    - fora de template, o fator geral precisa existir para modelos bifatoriais
    """

    factor_ids = [factor.factor_id for factor in qmatrix.factors]
    factor_id_set = set(factor_ids)

    if len(factor_ids) != len(factor_id_set):
        raise ValueError("Há fatores duplicados na Q-matrix.")
    for factor in qmatrix.factors:
        if factor.area is not None and factor.area != qmatrix.area:
            raise ValueError(
                f"Fator {factor.factor_id} pertence à área {factor.area}, não {qmatrix.area}."
            )

    row_ids: set[str] = set()
    used_factor_ids: set[str] = set()
    for row in qmatrix.rows:
        if row.area != qmatrix.area:
            raise ValueError(
                f"Item {row.item_id} pertence à área {row.area}, não {qmatrix.area}."
            )
        if row.item_id in row_ids:
            raise ValueError(f"Item duplicado na Q-matrix: {row.item_id}")
        row_ids.add(row.item_id)

        unknown = sorted(fid for fid in row.cargas if fid not in factor_id_set)
        if unknown:
            raise ValueError(
                f"Item {row.item_id} referencia fatores não declarados: {', '.join(unknown)}"
            )

        active_loads = [fid for fid, carga in row.cargas.items() if int(carga) == 1]
        used_factor_ids.update(active_loads)
        if not qmatrix.is_template and not active_loads:
            raise ValueError(
                f"Item {row.item_id} não carrega nenhum fator na Q-matrix."
            )

    if qmatrix.is_template and allow_template:
        return qmatrix

    if not factor_ids:
        raise ValueError("Q-matrix real precisa declarar ao menos um fator.")
    unused_factor_ids = sorted(factor_id_set - used_factor_ids)
    if unused_factor_ids:
        raise ValueError(
            "Fatores declarados sem nenhum item associado: "
            + ", ".join(unused_factor_ids)
        )

    return qmatrix


def build_empty_qmatrix_template(
    *,
    area: str,
    item_ids: Iterable[str],
    version: str = DEFAULT_QMATRIX_VERSION,
    source_label: str = "template_vazio",
) -> MIRTQMatrix:
    """
    Gera um template vazio para preenchimento manual.

    Importante: isso NÃO é uma Q-matrix pedagógica real. Serve apenas para
    congelar o formato quando a matriz ainda não foi fornecida.
    """

    area = _normalize_area(area)
    rows = [
        MIRTQMatrixRow(
            item_id=str(item_id),
            area=area,
            cargas={},
            metadata={"template": True},
        )
        for item_id in item_ids
    ]
    return MIRTQMatrix(
        version=version,
        area=area,
        source="template_vazio",
        is_template=True,
        factors=[],
        rows=rows,
        metadata={"source_label": source_label},
    )


def build_habilidade_qmatrix(
    *,
    area: str,
    item_rows: Iterable[Dict[str, Any]],
    version: str = DEFAULT_QMATRIX_VERSION,
    include_general_factor: bool = True,
) -> MIRTQMatrix:
    """
    Constrói uma Q-matrix bifatorial a partir do item bank real.

    Cada item recebe:
    - fator geral `G`, se solicitado
    - um fator específico por `numero_habilidade`

    O método depende de metadados reais vindos do item bank. Se faltarem campos,
    a função falha em vez de inventar uma alocação pedagógica.
    """

    area = _normalize_area(area)
    rows_by_item: Dict[str, MIRTQMatrixRow] = {}
    factor_specs: Dict[str, MIRTFactorSpec] = {}

    if include_general_factor:
        factor_specs[DEFAULT_GENERAL_FACTOR_ID] = MIRTFactorSpec(
            factor_id=DEFAULT_GENERAL_FACTOR_ID,
            label="Fator Geral",
            kind="geral",
            source="item_bank_habilidade",
            area=area,
        )

    for raw in item_rows:
        raw_area = _normalize_area(str(raw.get("area") or area))
        if raw_area != area:
            continue

        numero_habilidade = raw.get("numero_habilidade")
        if numero_habilidade is None:
            raise ValueError("Item bank sem `numero_habilidade`; não é possível montar a Q-matrix.")

        co_item = raw.get("co_item")
        item_id = str(raw.get("item_id") or co_item or "").strip()
        if not item_id:
            raise ValueError("Cada item precisa de `item_id` ou `co_item` para a Q-matrix.")

        factor_id = _factor_id_from_habilidade(area, int(numero_habilidade))
        factor_specs.setdefault(
            factor_id,
            MIRTFactorSpec(
                factor_id=factor_id,
                label=f"Habilidade {int(numero_habilidade):02d}",
                kind="especifico",
                source="item_bank_habilidade",
                area=area,
                metadata={"numero_habilidade": int(numero_habilidade)},
            ),
        )

        cargas = {factor_id: 1}
        if include_general_factor:
            cargas[DEFAULT_GENERAL_FACTOR_ID] = 1

        existing = rows_by_item.get(item_id)
        if existing is not None:
            existing_habs = {
                fid
                for fid, carga in existing.cargas.items()
                if carga and fid != DEFAULT_GENERAL_FACTOR_ID
            }
            if factor_id not in existing_habs:
                raise ValueError(
                    f"Item {item_id} aparece com habilidades conflitantes no item bank."
                )
            continue

        rows_by_item[item_id] = MIRTQMatrixRow(
            item_id=item_id,
            area=area,
            cargas=cargas,
            co_item=int(co_item) if co_item is not None else None,
            co_prova=int(raw["co_prova"]) if raw.get("co_prova") is not None else None,
            posicao=int(raw["posicao"]) if raw.get("posicao") is not None else None,
            metadata={
                "numero_habilidade": int(numero_habilidade),
                "tipo_lingua": raw.get("tipo_lingua"),
                "origem": "item_bank_habilidade",
            },
        )

    qmatrix = MIRTQMatrix(
        version=version,
        area=area,
        source="item_bank_habilidade",
        is_template=False,
        factors=sorted(
            factor_specs.values(),
            key=lambda factor: (factor.kind != "geral", factor.factor_id),
        ),
        rows=sorted(
            rows_by_item.values(),
            key=lambda row: (
                row.posicao if row.posicao is not None else 10**9,
                row.co_item if row.co_item is not None else 10**9,
                row.item_id,
            ),
        ),
        metadata={"include_general_factor": include_general_factor},
    )
    return validate_qmatrix(qmatrix, allow_template=False)


def build_model_spec_from_qmatrix(
    qmatrix: MIRTQMatrix,
    *,
    family: Literal["bifactor", "compensatorio"] = "bifactor",
    model_id: Optional[str] = None,
    scoring_factor_id: str = DEFAULT_GENERAL_FACTOR_ID,
) -> MIRTModelSpec:
    """
    Deriva uma especificação de modelo a partir da Q-matrix já validada.

    Para o `PR-010`, a estrutura automática suportada é:
    - `bifactor_habilidade` quando a origem é `item_bank_habilidade`
    - `qmatrix_manual` nos demais casos
    """

    validate_qmatrix(qmatrix, allow_template=qmatrix.is_template)

    declared_factor_ids = {factor.factor_id for factor in qmatrix.factors}
    if not qmatrix.is_template and scoring_factor_id not in declared_factor_ids:
        raise ValueError("`scoring_factor_id` precisa apontar para um fator declarado.")

    if family == "bifactor":
        general_factors = [factor for factor in qmatrix.factors if factor.kind == "geral"]
        if not qmatrix.is_template and len(general_factors) != 1:
            raise ValueError("Modelo bifactor exige exatamente um fator geral declarado.")
        if not qmatrix.is_template and scoring_factor_id not in {
            factor.factor_id for factor in general_factors
        }:
            raise ValueError("`scoring_factor_id` precisa apontar para um fator geral no bifactor.")

    initial_structure = (
        "bifactor_habilidade"
        if qmatrix.source == "item_bank_habilidade"
        else "qmatrix_manual"
    )
    model_id = model_id or f"mirt_{family}_{qmatrix.area.lower()}_{qmatrix.version}"

    return MIRTModelSpec(
        model_id=model_id,
        area=qmatrix.area,
        family=family,
        initial_structure=initial_structure,
        qmatrix_version=qmatrix.version,
        scoring_factor_id=scoring_factor_id,
        factors=qmatrix.factors,
        metadata={
            "qmatrix_source": qmatrix.source,
            "is_template": qmatrix.is_template,
            "n_fatores": len(qmatrix.factors),
            "n_itens": len(qmatrix.rows),
        },
    )


def qmatrix_to_binary_rows(qmatrix: MIRTQMatrix) -> List[Dict[str, Any]]:
    """
    Expande a Q-matrix para um formato tabular binário amigável ao worker R.

    Cada linha contém:
    - identificadores do item
    - uma coluna binária por fator
    """

    validate_qmatrix(qmatrix, allow_template=qmatrix.is_template)
    factor_ids = [factor.factor_id for factor in qmatrix.factors]
    table: List[Dict[str, Any]] = []

    for row in qmatrix.rows:
        payload: Dict[str, Any] = {
            "item_id": row.item_id,
            "area": row.area,
            "co_item": row.co_item,
            "co_prova": row.co_prova,
            "posicao": row.posicao,
        }
        for factor_id in factor_ids:
            payload[factor_id] = int(row.cargas.get(factor_id, 0))
        table.append(payload)

    return table
