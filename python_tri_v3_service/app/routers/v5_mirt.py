"""Endpoints V5 — Scoring multidimensional via artefatos MIRT locais (worker R)."""

import csv
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException

# Indireção proposital: os testes fazem mock.patch em "app.main.mirt_runtime_available"
# e "app.main.run_mirt_score"; resolver via módulo app.main preserva esse contrato.
from .. import main as _main
from .. import state
from ..db.supabase_client import get_supabase, is_supabase_configured
from ..irt.transform import ancorar_nota_individual
from ..models import RequestEstimarThetaMIRT, ResponseEstimarThetaMIRT
from ..motores.cientifico import estado_tri, estimar_theta_individual
from ..runtime.mirt import (
    RuntimeMIRTArtifactAmbiguousError,
    RuntimeMIRTArtifactError,
    RuntimeMIRTArtifactNotFoundError,
    resolve_mirt_artifact,
)
from ..runtime.unidim import (
    RuntimeArtifactAmbiguousError,
    RuntimeArtifactNotFoundError,
    resolve_unidim_artifact,
)
from ..mirt.runner import MIRTWorkerExecutionError
from ..storage.model_registry import list_active_models_by_family

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_transformacao_supabase(
    *,
    ano_referencia: int,
    area: str,
) -> Optional[Dict[str, float]]:
    """Busca a transformação `theta -> nota` no Supabase quando disponível."""

    if not is_supabase_configured():
        return None

    try:
        sb = get_supabase()
        tr = (
            sb.table("enem_transformacoes")
            .select("sigma,mu")
            .eq("ano", ano_referencia)
            .eq("area", area)
            .limit(1)
            .execute()
        )
        if tr.data:
            return {
                "sigma": float(tr.data[0]["sigma"]),
                "mu": float(tr.data[0]["mu"]),
            }
    except Exception:
        return None

    return None


def _build_mirt_response(
    *,
    respostas: list[int],
    area: str,
    ano_referencia: int,
    artefato_id: str,
    scoring_factor_id: str,
    scores_mirt: Dict[str, float],
    se_mirt: Dict[str, float],
    transformacao: Optional[Dict[str, float]],
) -> ResponseEstimarThetaMIRT:
    """Normaliza o output multidimensional para o contrato HTTP do backend."""

    if scoring_factor_id not in scores_mirt:
        raise RuntimeMIRTArtifactError(
            f"O fator principal {scoring_factor_id} não foi retornado pelo scoring MIRT."
        )
    if scoring_factor_id not in se_mirt:
        raise RuntimeMIRTArtifactError(
            f"O erro padrão do fator principal {scoring_factor_id} não foi retornado pelo scoring MIRT."
        )

    acertos = int(sum(respostas))
    theta = float(scores_mirt[scoring_factor_id])
    se = float(se_mirt[scoring_factor_id])

    nota = ancorar_nota_individual(
        theta=theta,
        area=area,
        acertos=acertos,
        tabela=state.get_tabela(),
        transformacao=transformacao,
    )

    return ResponseEstimarThetaMIRT(
        acertos=acertos,
        theta=round(theta, 6),
        se=round(se, 4),
        nota_ancorada=round(float(nota), 1),
        estado_tri=estado_tri(nota),
        artefato_id=artefato_id,
        modelo_usado="mirt",
        fallback_modelo=None,
        scores_mirt={factor_id: round(float(value), 6) for factor_id, value in scores_mirt.items()},
        se_mirt={factor_id: round(float(value), 4) for factor_id, value in se_mirt.items()},
    )


def _resolve_unidim_fallback_response(
    *,
    payload: RequestEstimarThetaMIRT,
    area: str,
    fallback_from: str,
) -> ResponseEstimarThetaMIRT:
    """Reaproveita o fluxo unidimensional atual para o fallback do MIRT."""

    artefato = resolve_unidim_artifact(
        ano_referencia=payload.ano_referencia,
        area=area,
        scope_id=payload.simulado_id,
    )
    params = [
        {"a": p["a"], "b": p["b"], "c": p.get("c", 0.20)}
        for p in artefato["parametros_itens"]
    ]
    if len(payload.respostas) != len(params):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Número de respostas ({len(payload.respostas)}) != número de itens "
                f"do fallback unidimensional ({len(params)})"
            ),
        )
    transformacao = payload.transformacao
    if transformacao is None:
        transformacao = artefato.get("transformacao")
    if transformacao is None:
        transformacao = _resolve_transformacao_supabase(
            ano_referencia=payload.ano_referencia,
            area=area,
        )

    resultado = estimar_theta_individual(
        respostas=payload.respostas,
        parametros_itens=params,
        area=area,
        tabela=state.get_tabela(),
        transformacao=transformacao,
    )
    return ResponseEstimarThetaMIRT(
        **resultado,
        artefato_id=artefato["manifest"]["artifact_id"],
        modelo_usado="unidim",
        fallback_modelo=fallback_from,
        scores_mirt=None,
        se_mirt=None,
    )


def _try_mirt_score(
    *,
    payload: RequestEstimarThetaMIRT,
    area: str,
) -> ResponseEstimarThetaMIRT:
    """Executa o fluxo de score MIRT a partir de um artefato já calibrado."""

    artefato = resolve_mirt_artifact(
        ano_referencia=payload.ano_referencia,
        area=area,
        artifact_id=payload.artefato_id,
        scope_id=payload.simulado_id,
    )
    qmatrix_rows = artefato["qmatrix"]["rows"]
    item_ids = [row["item_id"] for row in qmatrix_rows]
    if len(payload.respostas) != len(item_ids):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Número de respostas ({len(payload.respostas)}) != número de itens "
                f"na Q-matrix ({len(item_ids)})"
            ),
        )

    if not _main.mirt_runtime_available():
        raise MIRTWorkerExecutionError("`Rscript` não está disponível no runtime local.")

    with tempfile.TemporaryDirectory(prefix="xtri_mirt_score_") as tmpdir:
        input_path = Path(tmpdir) / "single_student.csv"
        output_path = Path(tmpdir) / "scores.csv"

        with open(input_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["student_id", *item_ids])
            writer.writerow(["student_1", *payload.respostas])

        _main.run_mirt_score(
            model_rds_path=artefato["model_rds_path"],
            response_matrix_path=input_path,
            output_path=output_path,
        )

        with open(output_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
        if row is None:
            raise RuntimeMIRTArtifactError("O scoring MIRT não retornou nenhuma linha.")

    scores_mirt: Dict[str, float] = {}
    se_mirt: Dict[str, float] = {}
    for column, raw_value in row.items():
        if column == "student_id" or raw_value in (None, ""):
            continue
        value = float(raw_value)
        if column.startswith("SE_"):
            se_mirt[column[3:]] = value
        else:
            scores_mirt[column] = value

    transformacao = payload.transformacao or _resolve_transformacao_supabase(
        ano_referencia=payload.ano_referencia,
        area=area,
    )
    scoring_factor_id = artefato["worker_output"]["model_spec"]["scoring_factor_id"]
    return _build_mirt_response(
        respostas=payload.respostas,
        area=area,
        ano_referencia=payload.ano_referencia,
        artefato_id=artefato["manifest"]["artifact_id"],
        scoring_factor_id=scoring_factor_id,
        scores_mirt=scores_mirt,
        se_mirt=se_mirt,
        transformacao=transformacao,
    )


@router.post("/api/v5/mirt/estimar-theta", response_model=ResponseEstimarThetaMIRT)
async def estimar_theta_mirt(payload: RequestEstimarThetaMIRT):
    """Estima score multidimensional a partir de artefatos MIRT locais."""

    if state.get_tabela() is None:
        raise HTTPException(
            status_code=500,
            detail="Tabela de referência TRI não carregada",
        )

    area = payload.area.upper()
    if area not in ["LC", "CH", "CN", "MT"]:
        raise HTTPException(
            status_code=400,
            detail=f"Área inválida: {area}",
        )

    try:
        return _try_mirt_score(payload=payload, area=area)
    except HTTPException:
        raise
    except (
        RuntimeMIRTArtifactNotFoundError,
        RuntimeMIRTArtifactAmbiguousError,
        RuntimeMIRTArtifactError,
        MIRTWorkerExecutionError,
    ) as exc:
        if not payload.permitir_fallback_unidim:
            raise HTTPException(status_code=400, detail=str(exc))
        try:
            return _resolve_unidim_fallback_response(
                payload=payload,
                area=area,
                fallback_from="mirt",
            )
        except (RuntimeArtifactNotFoundError, RuntimeArtifactAmbiguousError) as fallback_exc:
            raise HTTPException(
                status_code=400,
                detail=f"{exc} Nenhum fallback unidimensional disponível: {fallback_exc}",
            )


@router.get("/api/v5/mirt/availability")
async def mirt_availability(ano_referencia: int = 2024):
    """Per-area availability of MIRT artifacts for a given reference year.

    Frontend calls this on page load to render the 'MIRT Avançado' status
    card. Returns a boolean per LC/CH/CN/MT plus the total count. Default
    scoring flow is unaffected — this endpoint only reports what's there.
    """
    artefatos = list_active_models_by_family(
        family="mirt", ano_referencia=ano_referencia,
    )
    disponibilidade = {"LC": False, "CH": False, "CN": False, "MT": False}
    for a in artefatos:
        area = str(a.get("area", "")).upper()
        if area in disponibilidade:
            disponibilidade[area] = True
    return {
        "ano_referencia": ano_referencia,
        "disponibilidade": disponibilidade,
        "total_artefatos": len(artefatos),
    }
