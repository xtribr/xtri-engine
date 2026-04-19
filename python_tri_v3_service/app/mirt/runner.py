"""
Runner do worker MIRT em R.

Responsabilidades deste módulo:
- preparar a entrada do worker em disco
- versionar a pasta de artefatos por ano/área/modelo/job
- disparar `Rscript` para calibração offline
- ler e validar a saída normalizada do worker
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ..runtime.local_paths import MIRT_MODELS_ROOT
from .contracts import MIRTModelSpec, MIRTQMatrix, MIRTWorkerInput, MIRTWorkerOutput
from .qmatrix import qmatrix_to_binary_rows, validate_qmatrix

DEFAULT_RSCRIPT_BIN = "Rscript"
DEFAULT_FIT_TIMEOUT_S = 60 * 60
MIRT_FIT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mirt_fit.R"
MIRT_SCORE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mirt_score.R"
WORKER_INPUT_FILENAME = "worker_input.json"
WORKER_OUTPUT_FILENAME = "worker_output.json"
QMATRIX_JSON_FILENAME = "qmatrix.json"
QMATRIX_TABLE_FILENAME = "qmatrix_table.csv"
RESPONSE_MATRIX_CSV_FILENAME = "response_matrix.csv"


class MIRTWorkerError(RuntimeError):
    """Erro base do runner MIRT."""


class MIRTWorkerExecutionError(MIRTWorkerError):
    """Falha ao executar o worker R."""


def mirt_runtime_available(*, rscript_bin: str = DEFAULT_RSCRIPT_BIN) -> bool:
    """Indica se há `Rscript` disponível no ambiente atual."""

    return shutil.which(rscript_bin) is not None


def ensure_mirt_scripts() -> None:
    """Valida a existência dos scripts R esperados pelo runner."""

    for script in (MIRT_FIT_SCRIPT, MIRT_SCORE_SCRIPT):
        if not script.is_file():
            raise MIRTWorkerError(f"Script MIRT não encontrado: {script}")


def build_mirt_output_dir(
    *,
    year: int,
    area: str,
    model_id: str,
    job_id: str,
    base_dir: Optional[Union[Path, str]] = None,
) -> Path:
    """Resolve o diretório versionado do job MIRT."""

    root = Path(base_dir) if base_dir is not None else MIRT_MODELS_ROOT
    output_dir = root / str(year) / area.upper() / model_id / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def worker_input_path(worker_input: MIRTWorkerInput) -> Path:
    """Caminho canônico do `worker_input.json`."""

    return Path(worker_input.output_dir) / WORKER_INPUT_FILENAME


def worker_output_path(worker_input: MIRTWorkerInput) -> Path:
    """Caminho canônico do `worker_output.json`."""

    return Path(worker_input.output_dir) / WORKER_OUTPUT_FILENAME


def _validate_response_matrix(
    response_df: pd.DataFrame,
    *,
    qmatrix: MIRTQMatrix,
    student_id_column: str,
) -> tuple[list[str], pd.DataFrame]:
    """
    Garante que a matriz de resposta e a Q-matrix descrevem o mesmo conjunto de itens.
    """

    if student_id_column not in response_df.columns:
        normalized = response_df.copy()
        normalized.insert(0, student_id_column, [f"student_{idx + 1}" for idx in range(len(response_df))])
    else:
        normalized = response_df.copy()

    item_ids = [row.item_id for row in qmatrix.rows]
    response_item_ids = [col for col in normalized.columns if col != student_id_column]

    missing_in_qmatrix = sorted(item for item in response_item_ids if item not in item_ids)
    missing_in_response = sorted(item for item in item_ids if item not in response_item_ids)
    if missing_in_qmatrix or missing_in_response:
        raise MIRTWorkerError(
            "Matriz de resposta e Q-matrix não descrevem o mesmo conjunto de itens. "
            f"Faltando na Q-matrix: {missing_in_qmatrix or 'nenhum'}. "
            f"Faltando na resposta: {missing_in_response or 'nenhum'}."
        )

    ordered = normalized[[student_id_column] + item_ids].copy()
    for item_id in item_ids:
        series = ordered[item_id]
        invalid = series[~series.isin([0, 1]) & series.notna()]
        if not invalid.empty:
            raise MIRTWorkerError(
                f"A coluna {item_id} contém valores fora do domínio binário 0/1."
            )
        ordered[item_id] = series.fillna(0).astype(int)

    ordered[student_id_column] = ordered[student_id_column].astype(str)
    return item_ids, ordered


def prepare_mirt_fit_input(
    *,
    response_df: pd.DataFrame,
    qmatrix: MIRTQMatrix,
    model_spec: MIRTModelSpec,
    year: int,
    area: str,
    job_id: str,
    sample_size: Optional[int] = None,
    base_dir: Optional[Union[Path, str]] = None,
    student_id_column: str = "student_id",
) -> MIRTWorkerInput:
    """
    Serializa todos os insumos esperados pelo worker R e devolve o contrato final.
    """

    ensure_mirt_scripts()
    qmatrix = validate_qmatrix(qmatrix, allow_template=qmatrix.is_template)

    normalized_area = area.upper()
    if qmatrix.area != normalized_area:
        raise MIRTWorkerError(
            f"Área da Q-matrix ({qmatrix.area}) difere da área solicitada ({normalized_area})."
        )
    if model_spec.area != normalized_area:
        raise MIRTWorkerError(
            f"Área do modelo ({model_spec.area}) difere da área solicitada ({normalized_area})."
        )

    item_ids, ordered_df = _validate_response_matrix(
        response_df,
        qmatrix=qmatrix,
        student_id_column=student_id_column,
    )

    output_dir = build_mirt_output_dir(
        year=year,
        area=normalized_area,
        model_id=model_spec.model_id,
        job_id=job_id,
        base_dir=base_dir,
    )

    response_path = output_dir / RESPONSE_MATRIX_CSV_FILENAME
    qmatrix_json_path = output_dir / QMATRIX_JSON_FILENAME
    qmatrix_table_path = output_dir / QMATRIX_TABLE_FILENAME

    ordered_df.to_csv(response_path, index=False)
    qmatrix_json_path.write_text(
        qmatrix.model_dump_json(indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(qmatrix_to_binary_rows(qmatrix)).to_csv(qmatrix_table_path, index=False)

    worker_input = MIRTWorkerInput(
        job_id=job_id,
        year=year,
        area=normalized_area,
        response_matrix_path=str(response_path),
        qmatrix_path=str(qmatrix_json_path),
        output_dir=str(output_dir),
        model_spec=model_spec,
        sample_size=sample_size if sample_size is not None else len(ordered_df),
        metadata={
            "student_id_column": student_id_column,
            "item_ids": item_ids,
            "qmatrix_table_path": str(qmatrix_table_path),
            "response_format": "csv",
        },
    )
    worker_input_path(worker_input).write_text(
        worker_input.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return worker_input


def _run_subprocess(
    *,
    command: list[str],
    timeout_s: int,
) -> subprocess.CompletedProcess[str]:
    """Executa um subprocesso com captura de saída textual."""

    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def run_mirt_fit(
    worker_input: MIRTWorkerInput,
    *,
    rscript_bin: str = DEFAULT_RSCRIPT_BIN,
    timeout_s: int = DEFAULT_FIT_TIMEOUT_S,
) -> MIRTWorkerOutput:
    """
    Executa o worker de calibração MIRT e valida o `worker_output.json`.
    """

    ensure_mirt_scripts()
    if not mirt_runtime_available(rscript_bin=rscript_bin):
        raise MIRTWorkerExecutionError(
            f"`{rscript_bin}` não está disponível no ambiente para o worker MIRT."
        )

    input_path = worker_input_path(worker_input)
    if not input_path.is_file():
        input_path.write_text(worker_input.model_dump_json(indent=2), encoding="utf-8")

    completed = _run_subprocess(
        command=[rscript_bin, str(MIRT_FIT_SCRIPT), str(input_path)],
        timeout_s=timeout_s,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or "worker R finalizou sem mensagem de erro."
        raise MIRTWorkerExecutionError(
            f"Falha ao executar o worker MIRT (exit={completed.returncode}): {detail}"
        )

    output_path = worker_output_path(worker_input)
    if not output_path.is_file():
        raise MIRTWorkerExecutionError(
            f"Worker MIRT finalizou, mas não produziu {output_path.name} em {output_path.parent}."
        )

    return MIRTWorkerOutput.model_validate_json(output_path.read_text(encoding="utf-8"))


def run_mirt_score(
    *,
    model_rds_path: Union[str, Path],
    response_matrix_path: Union[str, Path],
    output_path: Union[str, Path],
    rscript_bin: str = DEFAULT_RSCRIPT_BIN,
    timeout_s: int = DEFAULT_FIT_TIMEOUT_S,
) -> Path:
    """
    Executa o script de scoring MIRT sobre uma matriz de respostas já pronta em CSV.
    """

    ensure_mirt_scripts()
    if not mirt_runtime_available(rscript_bin=rscript_bin):
        raise MIRTWorkerExecutionError(
            f"`{rscript_bin}` não está disponível no ambiente para o worker MIRT."
        )

    model_rds_path = Path(model_rds_path)
    response_matrix_path = Path(response_matrix_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = _run_subprocess(
        command=[
            rscript_bin,
            str(MIRT_SCORE_SCRIPT),
            str(model_rds_path),
            str(response_matrix_path),
            str(output_path),
        ],
        timeout_s=timeout_s,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or "script de scoring finalizou sem mensagem de erro."
        raise MIRTWorkerExecutionError(
            f"Falha ao executar o scoring MIRT (exit={completed.returncode}): {detail}"
        )
    if not output_path.is_file():
        raise MIRTWorkerExecutionError(
            f"Scoring MIRT finalizou, mas não produziu o arquivo {output_path}."
        )
    return output_path
