"""
Resolução de artefatos MIRT no runtime local.

Centraliza a busca e a validação dos artefatos produzidos pelo worker em R.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..mirt.contracts import MIRTQMatrix, MIRTWorkerOutput
from ..mirt.runner import MIRT_MODELS_ROOT
from ..storage.model_registry import (
    list_active_models_by_area,
    list_active_models_by_scope,
)


class RuntimeMIRTArtifactError(RuntimeError):
    """Erro base de resolução de artefatos MIRT."""


class RuntimeMIRTArtifactNotFoundError(RuntimeMIRTArtifactError):
    """Nenhum artefato MIRT compatível foi encontrado."""


class RuntimeMIRTArtifactAmbiguousError(RuntimeMIRTArtifactError):
    """Mais de um artefato MIRT compatível foi encontrado."""


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _artifact_context_from_dir(artifact_dir: Path) -> Dict[str, Any]:
    """Deriva metadados básicos do diretório versionado do artefato MIRT."""

    try:
        modelo = artifact_dir.parents[0].name
        area = artifact_dir.parents[1].name
        ano_referencia = int(artifact_dir.parents[2].name)
    except (IndexError, ValueError) as exc:
        raise RuntimeMIRTArtifactError(
            f"Estrutura de diretório MIRT inválida: {artifact_dir}"
        ) from exc

    return {
        "artifact_id": artifact_dir.name,
        "artifact_dir": str(artifact_dir),
        "modelo": modelo,
        "area": area,
        "ano_referencia": ano_referencia,
    }


def load_mirt_artifact(
    artifact_dir: Path | str,
    *,
    registry_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Carrega e valida um artefato MIRT salvo em disco."""

    artifact_dir = Path(artifact_dir)
    worker_output_path = artifact_dir / "worker_output.json"
    if not worker_output_path.is_file():
        raise RuntimeMIRTArtifactNotFoundError(
            f"worker_output.json não encontrado em {artifact_dir}."
        )

    worker_output = MIRTWorkerOutput.model_validate_json(
        worker_output_path.read_text(encoding="utf-8")
    )
    artifact_files = worker_output.artifact_files
    if artifact_files is None:
        raise RuntimeMIRTArtifactError(
            f"O artefato MIRT {artifact_dir.name} não possui `artifact_files`."
        )

    model_rds_path = Path(artifact_files.model_rds_path)
    if not model_rds_path.is_file():
        raise RuntimeMIRTArtifactNotFoundError(
            f"Arquivo model.rds não encontrado para o artefato {artifact_dir.name}."
        )

    qmatrix_snapshot_path = Path(artifact_files.qmatrix_snapshot_path)
    fit_summary_path = Path(artifact_files.fit_summary_path)
    diagnostics_path = Path(artifact_files.diagnostics_path)
    for path in (qmatrix_snapshot_path, fit_summary_path, diagnostics_path):
        if not path.is_file():
            raise RuntimeMIRTArtifactNotFoundError(
                f"Arquivo obrigatório ausente no artefato MIRT: {path}"
            )

    qmatrix = MIRTQMatrix.model_validate_json(
        qmatrix_snapshot_path.read_text(encoding="utf-8")
    )
    context = _artifact_context_from_dir(artifact_dir)
    if registry_entry is not None:
        context.update(
            {
                "scope_id": registry_entry.get("scope_id"),
                "simulado_id": registry_entry.get("simulado_id"),
                "saved_at_utc": registry_entry.get("saved_at_utc"),
            }
        )

    return {
        "manifest": context,
        "worker_output": worker_output.model_dump(mode="json"),
        "qmatrix": qmatrix.model_dump(mode="json"),
        "fit_summary": _read_json(fit_summary_path),
        "diagnostics": _read_json(diagnostics_path),
        "model_rds_path": str(model_rds_path),
        "artifact_files": artifact_files.model_dump(mode="json"),
    }


def load_mirt_artifact_by_id(
    *,
    ano_referencia: int,
    area: str,
    artifact_id: str,
) -> Optional[Dict[str, Any]]:
    """Localiza e carrega um artefato MIRT por `artifact_id` dentro de `ano/area`."""

    base_dir = MIRT_MODELS_ROOT / str(ano_referencia) / area.upper()
    if not base_dir.is_dir():
        return None

    matches = list(base_dir.glob(f"*/{artifact_id}/worker_output.json"))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeMIRTArtifactAmbiguousError(
            f"Há múltiplos artefatos MIRT com ID {artifact_id} em {area}/{ano_referencia}."
        )

    return load_mirt_artifact(matches[0].parent)


def resolve_mirt_artifact(
    *,
    ano_referencia: int,
    area: str,
    artifact_id: Optional[str] = None,
    scope_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve um artefato MIRT do runtime local.

    Ordem:
    1. `artifact_id` explícito
    2. `scope_id`/simulado explícito
    3. fallback para o único modelo ativo da combinação `ano/área`
    """

    area = area.upper()

    if artifact_id:
        artifact = load_mirt_artifact_by_id(
            ano_referencia=ano_referencia,
            area=area,
            artifact_id=artifact_id,
        )
        if artifact is None:
            raise RuntimeMIRTArtifactNotFoundError(
                f"Nenhum artefato MIRT {artifact_id} encontrado para {area}/{ano_referencia}."
            )
        return artifact

    if scope_id:
        candidates = list_active_models_by_scope(
            ano_referencia=ano_referencia,
            area=area,
            scope_id=scope_id,
            family="mirt",
        )
        if not candidates:
            raise RuntimeMIRTArtifactNotFoundError(
                f"Nenhum modelo MIRT ativo encontrado para {area}/{ano_referencia}/{scope_id}."
            )
        if len(candidates) > 1:
            raise RuntimeMIRTArtifactAmbiguousError(
                f"Há {len(candidates)} modelos MIRT ativos para {area}/{ano_referencia}/{scope_id}. "
                "Informe `artefato_id`."
            )
        return load_mirt_artifact(candidates[0]["artifact_dir"], registry_entry=candidates[0])

    candidates = list_active_models_by_area(
        ano_referencia=ano_referencia,
        area=area,
        modelo=None,
        family="mirt",
    )
    if not candidates:
        raise RuntimeMIRTArtifactNotFoundError(
            f"Nenhum modelo MIRT ativo encontrado para {area}/{ano_referencia}."
        )
    if len(candidates) > 1:
        raise RuntimeMIRTArtifactAmbiguousError(
            f"Há {len(candidates)} modelos MIRT ativos para {area}/{ano_referencia}. "
            "Informe `simulado_id` ou `artefato_id`."
        )
    return load_mirt_artifact(candidates[0]["artifact_dir"], registry_entry=candidates[0])
