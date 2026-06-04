"""
Pacote de contratos e utilitários do pipeline MIRT.

O worker em R entra no próximo PR. Aqui ficam apenas os tipos e helpers
necessários para congelar o formato da integração.
"""

from .contracts import (
    MIRTFactorSpec,
    MIRTModelSpec,
    MIRTQMatrix,
    MIRTQMatrixRow,
    MIRTWorkerArtifactFiles,
    MIRTWorkerInput,
    MIRTWorkerOutput,
)
from .qmatrix import (
    DEFAULT_GENERAL_FACTOR_ID,
    DEFAULT_QMATRIX_VERSION,
    build_empty_qmatrix_template,
    build_habilidade_qmatrix,
    build_model_spec_from_qmatrix,
    qmatrix_to_binary_rows,
    validate_qmatrix,
)
from .runner import (
    DEFAULT_FIT_TIMEOUT_S,
    DEFAULT_RSCRIPT_BIN,
    MIRTWorkerError,
    MIRTWorkerExecutionError,
    build_mirt_output_dir,
    mirt_runtime_available,
    prepare_mirt_fit_input,
    run_mirt_fit,
    run_mirt_score,
)

__all__ = [
    "DEFAULT_FIT_TIMEOUT_S",
    "DEFAULT_GENERAL_FACTOR_ID",
    "DEFAULT_QMATRIX_VERSION",
    "DEFAULT_RSCRIPT_BIN",
    "MIRTFactorSpec",
    "MIRTModelSpec",
    "MIRTQMatrix",
    "MIRTQMatrixRow",
    "MIRTWorkerError",
    "MIRTWorkerExecutionError",
    "MIRTWorkerArtifactFiles",
    "MIRTWorkerInput",
    "MIRTWorkerOutput",
    "build_mirt_output_dir",
    "build_empty_qmatrix_template",
    "build_habilidade_qmatrix",
    "build_model_spec_from_qmatrix",
    "mirt_runtime_available",
    "prepare_mirt_fit_input",
    "qmatrix_to_binary_rows",
    "run_mirt_fit",
    "run_mirt_score",
    "validate_qmatrix",
]
