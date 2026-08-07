"""Endpoints V5 — Scoring multidimensional via artefatos MIRT locais (worker R)."""

import csv
from datetime import datetime, timezone
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

# Indireção proposital: os testes fazem mock.patch em "app.main.mirt_runtime_available"
# e "app.main.run_mirt_score"; resolver via módulo app.main preserva esse contrato.
from .. import main as _main
from .. import state
from ..api_auth import require_api_key
from ..irt.transform import ancorar_nota_individual
from ..mirt.jobs import (
    MIRTActiveJobError,
    MIRTQueueFullError,
    job_manager,
    submit_mirt_job,
)
from ..mirt.qmatrix import build_model_spec_from_qmatrix, validate_qmatrix
from ..mirt.runner import (
    MIRTWorkerError,
    MIRTWorkerBusyError,
    MIRTWorkerExecutionError,
    prepare_mirt_fit_input,
    run_mirt_fit,
)
from ..models import (
    RequestCalibrarMIRT,
    RequestEstimarLoteMIRT,
    RequestEstimarThetaMIRT,
    ResponseCalibrarMIRT,
    ResponseEstimarLoteMIRT,
    ResponseEstimarThetaMIRT,
    ResponseJobMIRT,
    ResultadoAlunoMIRTLote,
)
from ..motores.cientifico import estado_tri
from ..runtime.mirt import (
    RuntimeMIRTArtifactAmbiguousError,
    RuntimeMIRTArtifactError,
    RuntimeMIRTArtifactNotFoundError,
    load_mirt_artifact,
    resolve_mirt_artifact,
)
from ..storage.model_registry import list_active_models_by_family, set_active_model

logger = logging.getLogger(__name__)

router = APIRouter()


def _reference_table_for_year(ano_referencia: int):
    """Resolve a tabela do ano declarado; nunca ancora em outro ano silenciosamente."""

    table = state.get_tabelas_multiano().get(ano_referencia)
    if table is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Ano {ano_referencia} não disponível para ancoragem. "
                f"Opções: {sorted(state.get_tabelas_multiano())}"
            ),
        )
    return table


def _run_calibration_job(job_id: str, payload: RequestCalibrarMIRT) -> None:
    """Executa uma calibração fora da thread HTTP e ativa o artefato."""

    worker_input = None
    worker_output = None
    try:
        job_manager.mark_running(job_id)
        qmatrix = validate_qmatrix(payload.qmatrix, allow_template=False)
        model_spec = build_model_spec_from_qmatrix(
            qmatrix,
            family=payload.familia,
            scoring_factor_id=payload.scoring_factor_id,
        )
        item_ids = [row.item_id for row in qmatrix.rows]
        rows: list[dict[str, Any]] = []
        for aluno in payload.alunos:
            row: dict[str, Any] = {"student_id": aluno.aluno_id}
            row.update(dict(zip(item_ids, aluno.respostas)))
            rows.append(row)

        response_df = pd.DataFrame(rows, columns=["student_id", *item_ids])
        worker_input = prepare_mirt_fit_input(
            response_df=response_df,
            qmatrix=qmatrix,
            model_spec=model_spec,
            year=payload.ano_referencia,
            area=payload.area,
            job_id=job_id,
            sample_size=len(payload.alunos),
        )
        job_manager.update_progress(job_id, progress=0.25, stage="fitting_model")
        worker_output = run_mirt_fit(worker_input)
        if worker_output.status != "done":
            raise MIRTWorkerExecutionError(
                worker_output.error or "Worker MIRT não concluiu a calibração"
            )
        expected_dir = Path(worker_input.output_dir).resolve()
        if Path(worker_output.artifact_dir).resolve() != expected_dir:
            raise MIRTWorkerExecutionError(
                "Worker MIRT retornou artifact_dir divergente do job"
            )
        if worker_output.n_students != len(payload.alunos):
            raise MIRTWorkerExecutionError(
                "Worker MIRT retornou quantidade divergente de alunos"
            )
        if worker_output.n_items != len(qmatrix.rows):
            raise MIRTWorkerExecutionError(
                "Worker MIRT retornou quantidade divergente de itens"
            )
        if worker_output.model_spec != model_spec:
            raise MIRTWorkerExecutionError(
                "Worker MIRT retornou especificação de modelo divergente"
            )

        job_manager.update_progress(job_id, progress=0.9, stage="validating_artifact")
        artifact = load_mirt_artifact(worker_output.artifact_dir)
        if artifact["qmatrix"] != qmatrix.model_dump(mode="json"):
            raise MIRTWorkerExecutionError(
                "Snapshot da Q-matrix diverge do input validado"
            )
        saved_at = datetime.now(timezone.utc).isoformat()
        manifest = {
            "artifact_id": job_id,
            "artifact_dir": worker_output.artifact_dir,
            "simulado_id": payload.simulado_id,
            "scope_type": "simulado_id",
            "scope_id": payload.simulado_id,
            "ano_referencia": payload.ano_referencia,
            "area": payload.area,
            "modelo": model_spec.model_id,
            "saved_at_utc": saved_at,
        }
        set_active_model(
            manifest,
            modelo=model_spec.model_id,
            exclusive_family=True,
        )
        job_manager.mark_done(
            job_id,
            model_id=model_spec.model_id,
            artifact_id=job_id,
            fit_summary=artifact["fit_summary"],
            warnings=list(worker_output.warnings),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Calibração MIRT %s falhou", job_id)
        job_manager.mark_error(job_id, str(exc))
    finally:
        retain = os.getenv("XTRI_MIRT_RETAIN_RESPONSES", "").strip().lower()
        if worker_input is not None and retain not in {"1", "true", "yes"}:
            sensitive_paths = [Path(worker_input.response_matrix_path)]
            if worker_output is not None and worker_output.artifact_files is not None:
                sensitive_paths.append(Path(worker_output.artifact_files.factor_scores_path))
            for sensitive_path in sensitive_paths:
                try:
                    sensitive_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Não foi possível apagar dado sensível do job %s: %s",
                        job_id,
                        sensitive_path,
                    )


def _parse_worker_scores(
    row: dict[str, Optional[str]],
    *,
    expected_factor_ids: list[str],
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Separa scores e erros padrão, rejeitando saída não finita."""

    scores_mirt: Dict[str, float] = {}
    se_mirt: Dict[str, float] = {}
    for column, raw_value in row.items():
        if column is None:
            raise RuntimeMIRTArtifactError("Worker retornou colunas excedentes")
        if column == "student_id" or raw_value in (None, ""):
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeMIRTArtifactError(
                f"Valor inválido retornado pelo worker na coluna {column}"
            ) from exc
        if not math.isfinite(value):
            raise RuntimeMIRTArtifactError(
                f"Valor não finito retornado pelo worker na coluna {column}"
            )
        if column.startswith("SE_"):
            if value < 0:
                raise RuntimeMIRTArtifactError(
                    f"Erro padrão negativo retornado na coluna {column}"
                )
            se_mirt[column[3:]] = value
        else:
            scores_mirt[column] = value
    expected = set(expected_factor_ids)
    if set(scores_mirt) != expected or set(se_mirt) != expected:
        raise RuntimeMIRTArtifactError(
            "Colunas de fatores/SE retornadas pelo worker divergem da Q-matrix"
        )
    return scores_mirt, se_mirt


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
        tabela=_reference_table_for_year(ano_referencia),
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
    if payload.item_ids != item_ids:
        raise HTTPException(
            status_code=422,
            detail="item_ids deve coincidir exatamente, inclusive na ordem, com a Q-matrix",
        )
    if len(payload.respostas) != len(item_ids):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Número de respostas ({len(payload.respostas)}) != número de itens "
                f"na Q-matrix ({len(item_ids)})"
            ),
        )
    if len(item_ids) != 45:
        raise HTTPException(
            status_code=400,
            detail=(
                "O endpoint individual ancora apenas provas completas de 45 itens; "
                "use /api/v5/mirt/estimar-theta-lote para obter theta em escala reduzida"
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
            qmatrix_path=artefato["artifact_files"]["qmatrix_snapshot_path"],
        )

        with open(output_path, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1 or rows[0].get("student_id") != "student_1":
            raise RuntimeMIRTArtifactError(
                "O scoring individual deve retornar exatamente o student_id enviado"
            )
        row = rows[0]

    expected_factor_ids = [
        factor["factor_id"] for factor in artefato["qmatrix"]["factors"]
    ]
    scores_mirt, se_mirt = _parse_worker_scores(
        row,
        expected_factor_ids=expected_factor_ids,
    )

    transformacao = payload.transformacao
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


@router.post(
    "/api/v5/mirt/estimar-theta",
    response_model=ResponseEstimarThetaMIRT,
    dependencies=[Depends(require_api_key)],
)
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
        return await run_in_threadpool(_try_mirt_score, payload=payload, area=area)
    except HTTPException:
        raise
    except MIRTWorkerBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except RuntimeMIRTArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeMIRTArtifactAmbiguousError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except (RuntimeMIRTArtifactError, MIRTWorkerExecutionError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post(
    "/api/v5/mirt/calibracoes",
    response_model=ResponseCalibrarMIRT,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_api_key)],
)
async def criar_calibracao_mirt(payload: RequestCalibrarMIRT):
    """Enfileira calibração multidimensional para um simulado × área."""

    if not _main.mirt_runtime_available():
        raise HTTPException(
            status_code=503,
            detail="Runtime MIRT indisponível: instale Rscript, jsonlite e mirt",
        )

    _reference_table_for_year(payload.ano_referencia)

    try:
        job = job_manager.create(
            simulado_id=payload.simulado_id,
            area=payload.area,
            ano_referencia=payload.ano_referencia,
            n_alunos=len(payload.alunos),
            n_itens=len(payload.qmatrix.rows),
            require_scope_available=True,
        )
    except MIRTActiveJobError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except MIRTQueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    try:
        submit_mirt_job(job.job_id, lambda job_id: _run_calibration_job(job_id, payload))
    except RuntimeError as exc:
        job_manager.mark_error(job.job_id, f"Falha ao enfileirar calibração: {exc}")
        raise HTTPException(status_code=503, detail="Fila MIRT indisponível")
    return ResponseCalibrarMIRT(
        job_id=job.job_id,
        status="queued",
        status_url=f"/api/v5/mirt/calibracoes/{job.job_id}",
    )


@router.get(
    "/api/v5/mirt/calibracoes/{job_id}",
    response_model=ResponseJobMIRT,
    dependencies=[Depends(require_api_key)],
)
async def status_calibracao_mirt(job_id: str):
    """Consulta o sidecar persistido de uma calibração MIRT."""

    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job MIRT não encontrado")
    return ResponseJobMIRT(**job.to_api_payload())


def _score_mirt_batch(payload: RequestEstimarLoteMIRT) -> ResponseEstimarLoteMIRT:
    """Executa uma única chamada R para todas as linhas do lote."""

    artefato = resolve_mirt_artifact(
        ano_referencia=payload.ano_referencia,
        area=payload.area,
        artifact_id=payload.artefato_id,
        scope_id=payload.simulado_id,
    )
    item_ids = [row["item_id"] for row in artefato["qmatrix"]["rows"]]
    if payload.item_ids != item_ids:
        raise HTTPException(
            status_code=422,
            detail="item_ids deve coincidir exatamente, inclusive na ordem, com a Q-matrix",
        )
    for aluno in payload.alunos:
        if len(aluno.respostas) != len(item_ids):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Aluno {aluno.aluno_id} tem {len(aluno.respostas)} respostas; "
                    f"a Q-matrix exige {len(item_ids)}"
                ),
            )

    if not _main.mirt_runtime_available():
        raise HTTPException(status_code=503, detail="Runtime MIRT indisponível")

    with tempfile.TemporaryDirectory(prefix="xtri_mirt_batch_") as tmpdir:
        input_path = Path(tmpdir) / "batch.csv"
        output_path = Path(tmpdir) / "scores.csv"
        with open(input_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["student_id", *item_ids])
            for aluno in payload.alunos:
                writer.writerow([aluno.aluno_id, *aluno.respostas])

        _main.run_mirt_score(
            model_rds_path=artefato["model_rds_path"],
            response_matrix_path=input_path,
            output_path=output_path,
            qmatrix_path=artefato["artifact_files"]["qmatrix_snapshot_path"],
        )
        with open(output_path, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    rows_by_id: dict[str, dict[str, Optional[str]]] = {}
    for row in rows:
        student_id = str(row.get("student_id") or "")
        if not student_id:
            raise RuntimeMIRTArtifactError("Worker retornou linha sem student_id")
        if student_id in rows_by_id:
            raise RuntimeMIRTArtifactError(
                f"Worker retornou student_id duplicado: {student_id}"
            )
        rows_by_id[student_id] = row

    expected_ids = {aluno.aluno_id for aluno in payload.alunos}
    if set(rows_by_id) != expected_ids:
        missing = sorted(expected_ids - set(rows_by_id))
        extra = sorted(set(rows_by_id) - expected_ids)
        raise RuntimeMIRTArtifactError(
            f"Saída do worker diverge do lote; ausentes={missing}, extras={extra}"
        )

    scoring_factor_id = artefato["worker_output"]["model_spec"]["scoring_factor_id"]
    expected_factor_ids = [
        factor["factor_id"] for factor in artefato["qmatrix"]["factors"]
    ]
    reference_table = _reference_table_for_year(payload.ano_referencia)
    transformacao = payload.transformacao
    resultados: list[ResultadoAlunoMIRTLote] = []
    for aluno in payload.alunos:
        scores_mirt, se_mirt = _parse_worker_scores(
            rows_by_id[aluno.aluno_id],
            expected_factor_ids=expected_factor_ids,
        )
        if scoring_factor_id not in scores_mirt or scoring_factor_id not in se_mirt:
            raise RuntimeMIRTArtifactError(
                f"Fator principal {scoring_factor_id} ou seu SE não retornado para {aluno.aluno_id}"
            )

        itens_ausentes = sum(resposta is None for resposta in aluno.respostas)
        itens_respondidos = len(aluno.respostas) - itens_ausentes
        acertos = sum(resposta == 1 for resposta in aluno.respostas)
        theta = float(scores_mirt[scoring_factor_id])
        se = float(se_mirt[scoring_factor_id])
        completo = itens_ausentes == 0
        elegivel_nota = completo and len(item_ids) == 45
        nota: Optional[float] = None
        estado: Optional[str] = None
        avisos: list[str] = []
        if elegivel_nota:
            nota = ancorar_nota_individual(
                theta=theta,
                area=payload.area,
                acertos=acertos,
                tabela=reference_table,
                transformacao=transformacao,
            )
            estado = estado_tri(nota)
        elif not completo:
            avisos.append(
                "Nota ancorada omitida: respostas ausentes não são tratadas como erro"
            )
        else:
            avisos.append(
                "Nota ancorada omitida: a escala ENEM completa exige 45 itens na área"
            )

        resultados.append(
            ResultadoAlunoMIRTLote(
                aluno_id=aluno.aluno_id,
                nome=aluno.nome,
                acertos=int(acertos),
                itens_respondidos=itens_respondidos,
                itens_ausentes=itens_ausentes,
                completo=completo,
                elegivel_nota_ancorada=elegivel_nota,
                theta=round(theta, 6),
                se=round(se, 4),
                nota_ancorada=round(float(nota), 1) if nota is not None else None,
                estado_tri=estado,
                scores_mirt={key: round(value, 6) for key, value in scores_mirt.items()},
                se_mirt={key: round(value, 4) for key, value in se_mirt.items()},
                avisos=avisos,
            )
        )

    return ResponseEstimarLoteMIRT(
        status="sucesso",
        area=payload.area,
        ano_referencia=payload.ano_referencia,
        artefato_id=artefato["manifest"]["artifact_id"],
        qmatrix_version=artefato["qmatrix"]["version"],
        n_alunos=len(resultados),
        resultados=resultados,
    )


@router.post(
    "/api/v5/mirt/estimar-theta-lote",
    response_model=ResponseEstimarLoteMIRT,
    dependencies=[Depends(require_api_key)],
)
async def estimar_theta_mirt_lote(payload: RequestEstimarLoteMIRT):
    """Escora até 10 mil alunos reutilizando um único artefato MIRT."""

    try:
        return await run_in_threadpool(_score_mirt_batch, payload)
    except HTTPException:
        raise
    except RuntimeMIRTArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeMIRTArtifactAmbiguousError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except MIRTWorkerBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except (RuntimeMIRTArtifactError, MIRTWorkerError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))


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
