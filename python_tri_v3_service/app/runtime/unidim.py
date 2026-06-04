"""
Resolução de artefatos 2PL para o runtime local.

Centraliza a escolha do modelo ativo para que o endpoint de scoring não precise
conhecer os detalhes do registry ou do layout em disco.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..storage.artifacts import DEFAULT_UNIDIM_MODEL, load_unidim_artifact_by_id
from ..storage.model_registry import get_active_model, list_active_models_by_area


class RuntimeArtifactError(RuntimeError):
    """Erro base de resolução de artefatos do runtime local."""


class RuntimeArtifactNotFoundError(RuntimeArtifactError):
    """Nenhum artefato compatível foi encontrado para a consulta."""


class RuntimeArtifactAmbiguousError(RuntimeArtifactError):
    """Mais de um artefato compatível foi encontrado para a consulta."""


def resolve_unidim_artifact(
    *,
    ano_referencia: int,
    area: str,
    artifact_id: Optional[str] = None,
    scope_id: Optional[str] = None,
    modelo: str = DEFAULT_UNIDIM_MODEL,
) -> Dict[str, Any]:
    """
    Resolve um artefato 2PL do runtime local.

    Ordem:
    1. `artifact_id` explícito
    2. `scope_id`/simulado explícito
    3. fallback para o único modelo ativo da combinação `ano/área`
    """

    area = area.upper()

    if artifact_id:
        artifact = load_unidim_artifact_by_id(
            ano_referencia=ano_referencia,
            area=area,
            artifact_id=artifact_id,
            modelo=modelo,
        )
        if artifact is None:
            raise RuntimeArtifactNotFoundError(
                f"Nenhum artefato {artifact_id} encontrado para {area}/{ano_referencia}."
            )
        return artifact

    if scope_id:
        active = get_active_model(
            ano_referencia=ano_referencia,
            area=area,
            scope_id=scope_id,
            modelo=modelo,
        )
        if active is None:
            raise RuntimeArtifactNotFoundError(
                f"Nenhum modelo ativo encontrado para {area}/{ano_referencia}/{scope_id}."
            )

        artifact = load_unidim_artifact_by_id(
            ano_referencia=ano_referencia,
            area=area,
            artifact_id=active["artifact_id"],
            modelo=modelo,
        )
        if artifact is None:
            raise RuntimeArtifactNotFoundError(
                f"O artefato ativo {active['artifact_id']} de {scope_id} não está disponível no disco."
            )
        return artifact

    candidates = list_active_models_by_area(
        ano_referencia=ano_referencia,
        area=area,
        modelo=modelo,
    )
    if not candidates:
        raise RuntimeArtifactNotFoundError(
            f"Nenhum modelo ativo encontrado para {area}/{ano_referencia}."
        )
    if len(candidates) > 1:
        raise RuntimeArtifactAmbiguousError(
            f"Há {len(candidates)} modelos ativos para {area}/{ano_referencia}. "
            "Informe `simulado_id` ou `artefato_id`."
        )

    artifact = load_unidim_artifact_by_id(
        ano_referencia=ano_referencia,
        area=area,
        artifact_id=candidates[0]["artifact_id"],
        modelo=modelo,
    )
    if artifact is None:
        raise RuntimeArtifactNotFoundError(
            f"O artefato ativo {candidates[0]['artifact_id']} não está disponível no disco."
        )
    return artifact
