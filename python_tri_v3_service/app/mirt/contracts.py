"""
Contratos do pipeline MIRT.

Este módulo congela o formato entre Python e o futuro worker em R.
Não executa calibração; apenas descreve de forma tipada:

- fatores do modelo
- Q-matrix
- payload de entrada do worker
- artefatos esperados na saída
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


VALID_AREAS = {"LC", "CH", "CN", "MT"}
VALID_MODEL_FAMILIES = {"bifactor", "compensatorio"}
VALID_INITIAL_STRUCTURES = {
    "bifactor_habilidade",
    "qmatrix_manual",
    "g_2_fatores",
    "g_3_fatores",
}


class MIRTFactorSpec(BaseModel):
    """Describe um fator latente do modelo MIRT."""

    factor_id: str = Field(..., description="Identificador estável do fator")
    label: str = Field(..., description="Nome curto exibível")
    kind: Literal["geral", "especifico"] = Field(
        ...,
        description="Fator geral ou fator específico",
    )
    source: Literal["item_bank_habilidade", "manual", "template"] = Field(
        ...,
        description="Origem da definição do fator",
    )
    area: Optional[str] = Field(
        default=None,
        description="Área ENEM à qual o fator pertence, quando aplicável",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Campos auxiliares para persistência/auditoria",
    )

    @field_validator("factor_id", "label")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Campo textual não pode ser vazio.")
        return value

    @field_validator("area")
    @classmethod
    def _normalize_area(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.upper()
        if value not in VALID_AREAS:
            raise ValueError(f"Área inválida para MIRT: {value}")
        return value


class MIRTQMatrixRow(BaseModel):
    """Linha da Q-matrix para um item."""

    item_id: str = Field(..., description="ID estável do item dentro do worker")
    area: str = Field(..., description="Área ENEM do item")
    cargas: Dict[str, int] = Field(
        default_factory=dict,
        description="Mapa binário `factor_id -> 0/1`",
    )
    co_item: Optional[int] = Field(default=None, description="CO_ITEM INEP")
    co_prova: Optional[int] = Field(default=None, description="CO_PROVA INEP")
    posicao: Optional[int] = Field(default=None, description="Posição na prova")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadados adicionais do item",
    )

    @field_validator("item_id")
    @classmethod
    def _validate_item_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("item_id não pode ser vazio.")
        return value

    @field_validator("area")
    @classmethod
    def _normalize_area(cls, value: str) -> str:
        value = value.upper()
        if value not in VALID_AREAS:
            raise ValueError(f"Área inválida para MIRT: {value}")
        return value

    @field_validator("cargas")
    @classmethod
    def _validate_cargas(cls, value: Dict[str, int]) -> Dict[str, int]:
        normalized: Dict[str, int] = {}
        for factor_id, carga in value.items():
            fid = factor_id.strip()
            if not fid:
                raise ValueError("factor_id em `cargas` não pode ser vazio.")
            normalized[fid] = 1 if int(carga) else 0
        return normalized


class MIRTQMatrix(BaseModel):
    """Definição completa da Q-matrix usada pelo worker MIRT."""

    version: str = Field(..., description="Versão estável da Q-matrix")
    area: str = Field(..., description="Área ENEM atendida pela matriz")
    source: Literal["item_bank_habilidade", "manual", "template_vazio"] = Field(
        ...,
        description="Origem da Q-matrix",
    )
    is_template: bool = Field(
        default=False,
        description="True quando a matriz é só um template estrutural, sem semântica final",
    )
    factors: List[MIRTFactorSpec] = Field(
        default_factory=list,
        description="Lista ordenada de fatores válidos na matriz",
    )
    rows: List[MIRTQMatrixRow] = Field(
        default_factory=list,
        description="Linhas item x fator",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contexto extra da matriz",
    )

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("A versão da Q-matrix não pode ser vazia.")
        return value

    @field_validator("area")
    @classmethod
    def _normalize_area(cls, value: str) -> str:
        value = value.upper()
        if value not in VALID_AREAS:
            raise ValueError(f"Área inválida para MIRT: {value}")
        return value


class MIRTModelSpec(BaseModel):
    """Especificação do modelo a ser ajustado pelo worker MIRT."""

    model_id: str = Field(..., description="Identificador estável do modelo")
    area: str = Field(..., description="Área ENEM")
    family: Literal["bifactor", "compensatorio"] = Field(
        ...,
        description="Família psicométrica do modelo",
    )
    initial_structure: Literal[
        "bifactor_habilidade",
        "qmatrix_manual",
        "g_2_fatores",
        "g_3_fatores",
    ] = Field(
        ...,
        description="Estrutura inicial declarada para a calibração",
    )
    qmatrix_version: str = Field(..., description="Versão da Q-matrix usada")
    scoring_factor_id: str = Field(
        ...,
        description="Fator principal usado para score/report padrão",
    )
    factors: List[MIRTFactorSpec] = Field(
        default_factory=list,
        description="Fatores disponíveis no modelo",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Campos auxiliares para auditoria do ajuste",
    )

    @field_validator("model_id", "qmatrix_version", "scoring_factor_id")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Campo textual obrigatório não pode ser vazio.")
        return value

    @field_validator("area")
    @classmethod
    def _normalize_area(cls, value: str) -> str:
        value = value.upper()
        if value not in VALID_AREAS:
            raise ValueError(f"Área inválida para MIRT: {value}")
        return value


class MIRTWorkerInput(BaseModel):
    """Payload persistível para o worker R."""

    job_id: str = Field(..., description="ID do job assíncrono")
    year: int = Field(..., description="Ano de referência")
    area: str = Field(..., description="Área ENEM")
    response_matrix_path: str = Field(..., description="CSV/Parquet com matriz N x J")
    qmatrix_path: str = Field(..., description="Arquivo serializado da Q-matrix")
    output_dir: str = Field(..., description="Diretório onde o worker escreverá artefatos")
    model_spec: MIRTModelSpec
    sample_size: Optional[int] = Field(
        default=None,
        description="N amostral usado na calibração, quando conhecido",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("job_id", "response_matrix_path", "qmatrix_path", "output_dir")
    @classmethod
    def _validate_path_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Campo de path/job não pode ser vazio.")
        return value

    @field_validator("area")
    @classmethod
    def _normalize_area(cls, value: str) -> str:
        value = value.upper()
        if value not in VALID_AREAS:
            raise ValueError(f"Área inválida para MIRT: {value}")
        return value


class MIRTWorkerArtifactFiles(BaseModel):
    """Arquivos que o worker MIRT precisa produzir para o runtime."""

    model_rds_path: str
    item_params_path: str
    factor_scores_path: str
    fit_summary_path: str
    diagnostics_path: str
    qmatrix_snapshot_path: str


class MIRTWorkerOutput(BaseModel):
    """Saída normalizada do worker MIRT."""

    status: Literal["done", "error"] = Field(..., description="Status final do job")
    artifact_dir: str = Field(..., description="Diretório raiz dos artefatos gerados")
    n_students: int = Field(default=0, description="Número de alunos calibrados")
    n_items: int = Field(default=0, description="Número de itens calibrados")
    model_spec: MIRTModelSpec
    artifact_files: Optional[MIRTWorkerArtifactFiles] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(
        default=None,
        description="Mensagem de erro serializável, quando status=error",
    )
