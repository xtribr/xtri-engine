"""
Modelos Pydantic para a API TRI V3.

Contém os schemas de entrada/saída para ambos os motores:
- Motor Heurístico (compatível com V2)
- Motor Científico (MML + EAP)
"""

import math

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Any, Union, Literal

from .mirt.contracts import (
    MIRTQMatrix,
    SAFE_ID_PATTERN,
    validate_mirt_factor_id,
    validate_mirt_item_id,
)
from .mirt.qmatrix import build_model_spec_from_qmatrix, validate_qmatrix


def _validar_id_publico(value: Optional[str], *, field_name: str) -> Optional[str]:
    """Normaliza IDs públicos usados em registry/glob e bloqueia curingas."""

    if value is None:
        return None
    value = value.strip()
    if not SAFE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} deve usar apenas letras, números, ponto, hífen ou underscore"
        )
    return value


def _validar_transformacao_latente(
    value: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    """Aceita somente linking explícito da escala theta; mu/sigma não são equivalentes."""

    if value is None:
        return None
    allowed = {"theta_ref_centro", "theta_ref_escala"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            "transformacao MIRT aceita apenas theta_ref_centro/theta_ref_escala; "
            f"campos inválidos: {', '.join(sorted(unknown))}"
        )
    centro = float(value.get("theta_ref_centro", 0.0))
    escala = float(value.get("theta_ref_escala", 1.0))
    if not math.isfinite(centro) or not math.isfinite(escala):
        raise ValueError("transformacao MIRT exige valores finitos")
    if escala <= 0:
        raise ValueError("theta_ref_escala deve ser positiva")
    return value


# ════════════════════════════════════════════════════════════════════════════════
# MODELOS DE ENTRADA — ROTA DE COMPATIBILIDADE (V2)
# ════════════════════════════════════════════════════════════════════════════════

class AlunoV2(BaseModel):
    """
    Aluno no formato V2 (compatível com o Gabaritaí).
    Respostas vêm como campos q1, q2, ..., q45 ou como lista em 'respostas'.
    """
    id: Optional[str] = ""
    nome: Optional[str] = ""
    respostas: Optional[List[str]] = None
    # Campos extras são aceitos (q1, q2, ..., turma, studentNumber, etc.)
    model_config = {"extra": "allow"}


class RequestCalcularTRI(BaseModel):
    """
    Payload de entrada para /api/calcular-tri (compatível com V2).
    Mesmo contrato que o Node.js envia para o Flask.
    """
    alunos: List[Dict[str, Any]]
    gabarito: Union[Dict[str, str], List[str]]
    areas_config: Optional[Dict[str, List[int]]] = None


class ResponseCalcularTRI(BaseModel):
    """
    Payload de saída para /api/calcular-tri (compatível com V2).
    Mesmo contrato que o Node.js espera receber.
    """
    status: str
    total_alunos: int
    prova_analysis: Optional[Dict[str, Any]] = None
    resultados: Optional[List[Dict[str, Any]]] = None
    mensagem: Optional[str] = None


# ════════════════════════════════════════════════════════════════════════════════
# MODELOS DE ENTRADA — MOTOR CIENTÍFICO (V3)
# ════════════════════════════════════════════════════════════════════════════════

class RespostaAlunoCientifico(BaseModel):
    """Aluno com vetor binário de respostas (0/1) para o motor científico."""
    aluno_id: str = Field(..., description="ID do aluno (UUID ou identificador)")
    nome: Optional[str] = ""
    respostas: List[int] = Field(
        ...,
        description="Vetor de respostas binárias: 1 = acerto, 0 = erro"
    )


class RequestCalibrarCientifico(BaseModel):
    """
    Payload de entrada para /api/v3/calibrar-cientifico.
    Recebe a matriz de respostas de UMA área de UM simulado.
    """
    simulado_id: str = Field(..., description="Identificador do simulado")
    area: str = Field(..., description="Área: LC, CH, CN ou MT")
    ano_referencia: int = Field(
        default=2024,
        description="Ano da tabela de ancoragem ENEM"
    )
    c_fixo: float = Field(
        default=0.20,
        description="Parâmetro c fixo (acerto casual). 0.20 = 5 alternativas"
    )
    alunos: List[RespostaAlunoCientifico]


class ItemParametros(BaseModel):
    """Parâmetros calibrados de um item."""
    item_index: int
    a: float = Field(..., description="Discriminação")
    b: float = Field(..., description="Dificuldade")
    c: float = Field(..., description="Acerto casual (fixo)")
    taxa_acerto: float
    saturado: bool = Field(
        default=False,
        description="True se b >= 3.9 (item com dificuldade saturada)"
    )


class ResultadoAlunoCientifico(BaseModel):
    """Resultado do motor científico para um aluno."""
    aluno_id: str
    nome: Optional[str] = ""
    acertos: int
    theta: float = Field(..., description="Traço latente estimado (EAP)")
    se: float = Field(..., description="Erro padrão do theta")
    nota_ancorada: float = Field(..., description="Nota na escala ENEM")
    estado_tri: str = Field(
        ...,
        description="AVANCADO, PROFICIENTE, BASICO, ELEMENTAR ou INICIANTE"
    )


class ResponseCalibrarCientifico(BaseModel):
    """Payload de saída para /api/v3/calibrar-cientifico."""
    status: str
    simulado_id: str
    area: str
    n_alunos: int
    n_itens: int
    parametros_itens: List[ItemParametros]
    resultados_alunos: List[ResultadoAlunoCientifico]
    transformacao: Optional[Dict[str, Any]] = None
    artefato_id: Optional[str] = Field(
        default=None,
        description="Artefato local salvo para o modelo unidimensional desta calibração",
    )
    diagnostico: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métricas de qualidade: % no intervalo, MAE, itens saturados"
    )


# ════════════════════════════════════════════════════════════════════════════════
# MODELOS — V6 ESCALA REDUZIDA
# ════════════════════════════════════════════════════════════════════════════════

class RequestTRIReduzida(BaseModel):
    """Payload de entrada para /api/v6/tri/reduzida."""
    simulado_id: str = Field(..., description="Identificador do simulado reduzido")
    area: str = Field(..., description="Área: LC, CH, CN ou MT")
    ano_referencia: int = Field(default=2024, description="Ano da escala ENEM")
    c_fixo: float = Field(default=0.20, description="Acerto casual fixo")
    itens_anulados: List[int] = Field(default_factory=list)
    alunos: List[RespostaAlunoCientifico]


class Intervalo95Reduzido(BaseModel):
    """Intervalo aproximado de 95% para a nota projetada."""
    limite_inferior: float
    limite_superior: float
    se_theta_ajustado: float
    fator_reducao: float


class FaixaReferenciaReduzida(BaseModel):
    """Faixa oficial interpolada a partir de acertos equivalentes em 45 itens."""
    acertos_equivalentes_45: float
    acertos_base_inferior: int
    acertos_base_superior: int
    tri_min: float
    tri_med: float
    tri_max: float


class ResultadoAlunoTRIReduzida(BaseModel):
    """Resultado por aluno para TRI com escala reduzida."""
    aluno_id: str
    nome: Optional[str] = ""
    acertos_observados: int
    n_itens_validos: int
    pct_acerto: float
    acertos_equivalentes_45: float
    theta: float
    se_theta: float
    nota_enem_projetada: float
    intervalo_95: Intervalo95Reduzido
    precisao: str
    faixa_referencia: FaixaReferenciaReduzida
    aviso_metodologico: str


class ResponseTRIReduzida(BaseModel):
    """Payload de saída para /api/v6/tri/reduzida."""
    status: str
    simulado_id: str
    area: str
    ano_referencia: int
    modo_scoring: str
    modo_escala: str
    n_itens_referencia: int
    n_itens_validos: int
    n_alunos: int
    metodo_ancoragem: str
    limites_uso: str
    resultados: List[ResultadoAlunoTRIReduzida]
    parametros_itens: List[ItemParametros]
    diagnostico: Dict[str, Any] = Field(default_factory=dict)
    avisos: List[str] = Field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════════════
# MODELOS — ESTIMAÇÃO RÁPIDA COM PARÂMETROS PRÉ-CALIBRADOS
# ════════════════════════════════════════════════════════════════════════════════

class ItemPreCalibrado(BaseModel):
    """Parâmetros de um item já calibrado previamente."""
    a: float
    b: float
    c: float = 0.20


class RequestEstimarTheta(BaseModel):
    """
    Payload para /api/v3/estimar-theta.
    Estima nota de UM aluno usando parâmetros já calibrados.
    """
    area: str
    ano_referencia: int = 2024
    simulado_id: Optional[str] = Field(
        default=None,
        description="Escopo do modelo ativo local quando `parametros_itens` não vier",
    )
    artefato_id: Optional[str] = Field(
        default=None,
        description="ID explícito do artefato local a ser reutilizado",
    )
    parametros_itens: Optional[List[ItemPreCalibrado]] = Field(
        default=None,
        description="Parâmetros calibrados enviados no payload; opcionais se houver artefato local",
    )
    transformacao: Optional[Dict[str, float]] = Field(
        default=None,
        description="Transformação opcional `theta -> nota` com campos `mu` e `sigma`"
    )
    respostas: List[int] = Field(
        ...,
        description="Vetor binário de respostas (0/1)"
    )


class ResponseEstimarTheta(BaseModel):
    """Payload de saída para /api/v3/estimar-theta."""
    acertos: int
    theta: float
    se: float
    nota_ancorada: float
    estado_tri: str
    artefato_id: Optional[str] = Field(
        default=None,
        description="Artefato local reaproveitado na estimação, quando aplicável",
    )


class RequestEstimarThetaMIRT(BaseModel):
    """
    Payload para /api/v5/mirt/estimar-theta.
    Scoreia uma prova completa via artefato MIRT local, com ordem explícita.
    """

    area: str
    ano_referencia: int = 2024
    simulado_id: Optional[str] = Field(
        default=None,
        description="Escopo lógico do modelo ativo local",
    )
    artefato_id: Optional[str] = Field(
        default=None,
        description="ID explícito do artefato MIRT a ser reutilizado",
    )
    transformacao: Optional[Dict[str, float]] = Field(
        default=None,
        description="Linking opcional com `theta_ref_centro` e `theta_ref_escala`",
    )
    permitir_fallback_unidim: Literal[False] = Field(
        default=False,
        description=(
            "Mantido no contrato apenas para explicitar que fallback posicional "
            "unidimensional não é permitido"
        ),
    )
    respostas: List[int] = Field(
        ...,
        description="Vetor binário de respostas (0/1)",
    )
    item_ids: List[str] = Field(
        ...,
        min_length=45,
        max_length=45,
        description="Ordem explícita dos 45 itens; deve coincidir com a Q-matrix",
    )

    @field_validator("respostas", mode="before")
    @classmethod
    def _validar_respostas_binarias(cls, value: Any) -> Any:
        if not isinstance(value, list) or not value:
            raise ValueError("respostas deve ser uma lista não vazia")
        if any(type(resposta) is not int or resposta not in (0, 1) for resposta in value):
            raise ValueError("respostas aceita somente 0 ou 1")
        return value

    @field_validator("simulado_id", "artefato_id")
    @classmethod
    def _validar_ids_publicos(cls, value: Optional[str], info: Any) -> Optional[str]:
        return _validar_id_publico(value, field_name=info.field_name)

    @field_validator("item_ids")
    @classmethod
    def _validar_item_ids(cls, value: List[str]) -> List[str]:
        normalized = [validate_mirt_item_id(item_id) for item_id in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("item_ids duplicados")
        return normalized

    @field_validator("transformacao")
    @classmethod
    def _validar_transformacao(
        cls, value: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        return _validar_transformacao_latente(value)

    @model_validator(mode="after")
    def _validar_seletor_de_artefato(self) -> "RequestEstimarThetaMIRT":
        if (self.simulado_id is None) == (self.artefato_id is None):
            raise ValueError("Informe exatamente um entre simulado_id e artefato_id")
        return self


class ResponseEstimarThetaMIRT(BaseModel):
    """Payload de saída para /api/v5/mirt/estimar-theta."""

    acertos: int
    theta: float
    se: float
    nota_ancorada: float
    estado_tri: str
    artefato_id: Optional[str] = Field(
        default=None,
        description="Artefato local efetivamente usado na estimação",
    )
    modelo_usado: str = Field(
        ...,
        description="`mirt` quando o score multidimensional foi usado; `unidim` no fallback",
    )
    fallback_modelo: Optional[str] = Field(
        default=None,
        description="Campo legado; a API pública MIRT não executa fallback posicional",
    )
    scores_mirt: Optional[Dict[str, float]] = Field(
        default=None,
        description="Scores por fator retornados pelo worker MIRT",
    )
    se_mirt: Optional[Dict[str, float]] = Field(
        default=None,
        description="Erros padrão por fator retornados pelo worker MIRT",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODELOS — API PÚBLICA MIRT (CALIBRAÇÃO E LOTE)
# ══════════════════════════════════════════════════════════════════════════════

RespostaBinariaOpcional = Optional[Literal[0, 1]]


class AlunoMIRT(BaseModel):
    """Linha de respostas de um aluno para calibração ou scoring MIRT."""

    aluno_id: str = Field(..., min_length=1, max_length=128)
    nome: Optional[str] = Field(default="", max_length=300)
    respostas: List[RespostaBinariaOpcional] = Field(
        ...,
        min_length=1,
        description="0=erro, 1=acerto, null=item ausente/não aplicado",
    )

    @field_validator("respostas", mode="before")
    @classmethod
    def _rejeitar_valores_nao_binarios(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("respostas deve ser uma lista")
        for resposta in value:
            if resposta is None:
                continue
            if type(resposta) is not int or resposta not in (0, 1):
                raise ValueError("respostas aceita somente 0, 1 ou null")
        return value

    @field_validator("aluno_id")
    @classmethod
    def _normalizar_aluno_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("aluno_id não pode ser vazio")
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("aluno_id não pode conter caracteres de controle")
        return value

    @field_validator("respostas")
    @classmethod
    def _validar_respostas_informativas(
        cls, value: List[RespostaBinariaOpcional]
    ) -> List[RespostaBinariaOpcional]:
        if all(resposta is None for resposta in value):
            raise ValueError("Aluno precisa ter ao menos uma resposta observada")
        return value


class RequestCalibrarMIRT(BaseModel):
    """Cria uma calibração MIRT assíncrona para um simulado × área."""

    simulado_id: str = Field(..., min_length=1, max_length=128)
    area: str
    ano_referencia: int = Field(default=2024, ge=2009, le=2100)
    familia: Literal["bifactor", "compensatorio"] = "bifactor"
    scoring_factor_id: str = Field(default="G", min_length=1, max_length=64)
    qmatrix: MIRTQMatrix
    alunos: List[AlunoMIRT] = Field(..., min_length=100, max_length=100_000)

    @field_validator("simulado_id")
    @classmethod
    def _normalizar_simulado_id(cls, value: str) -> str:
        validated = _validar_id_publico(value, field_name="simulado_id")
        assert validated is not None
        return validated

    @field_validator("scoring_factor_id")
    @classmethod
    def _validar_scoring_factor_id(cls, value: str) -> str:
        return validate_mirt_factor_id(value, field_name="scoring_factor_id")

    @field_validator("area")
    @classmethod
    def _normalizar_area(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in {"LC", "CH", "CN", "MT"}:
            raise ValueError("area deve ser LC, CH, CN ou MT")
        return value

    @model_validator(mode="after")
    def _validar_matriz(self) -> "RequestCalibrarMIRT":
        if self.qmatrix.area != self.area:
            raise ValueError("A área da Q-matrix difere da área da calibração")
        if self.qmatrix.is_template:
            raise ValueError("Q-matrix de template não pode ser usada em calibração")
        validate_qmatrix(self.qmatrix, allow_template=False)

        n_itens = len(self.qmatrix.rows)
        if n_itens < 2:
            raise ValueError("A calibração MIRT exige ao menos 2 itens")
        if n_itens > 45:
            raise ValueError("Uma calibração MIRT por área ENEM aceita no máximo 45 itens")
        ids = [aluno.aluno_id for aluno in self.alunos]
        if len(ids) != len(set(ids)):
            raise ValueError("aluno_id duplicado na calibração")
        for aluno in self.alunos:
            if len(aluno.respostas) != n_itens:
                raise ValueError(
                    f"Aluno {aluno.aluno_id} tem {len(aluno.respostas)} respostas; "
                    f"esperado: {n_itens}"
                )
        for idx, row in enumerate(zip(*(aluno.respostas for aluno in self.alunos))):
            observed = [resposta for resposta in row if resposta is not None]
            if not observed:
                raise ValueError(
                    f"Item {self.qmatrix.rows[idx].item_id} está ausente para toda a coorte"
                )
            n_errors = observed.count(0)
            n_correct = observed.count(1)
            if min(n_errors, n_correct) < 5:
                raise ValueError(
                    f"Item {self.qmatrix.rows[idx].item_id} exige ao menos 5 erros e 5 acertos observados"
                )

        if self.familia == "bifactor":
            general_ids = {
                factor.factor_id
                for factor in self.qmatrix.factors
                if factor.kind == "geral"
            }
            specific_ids = {
                factor.factor_id
                for factor in self.qmatrix.factors
                if factor.kind == "especifico"
            }
            if len(specific_ids) < 2:
                raise ValueError(
                    "Bifactor exige ao menos dois fatores específicos identificáveis"
                )
            if general_ids != {self.scoring_factor_id}:
                raise ValueError(
                    "Bifactor exige exatamente um fator geral, usado como scoring_factor_id"
                )
            indicator_counts = {factor_id: 0 for factor_id in specific_ids}
            for row in self.qmatrix.rows:
                if row.cargas.get(self.scoring_factor_id) != 1:
                    raise ValueError(
                        f"Item {row.item_id} precisa carregar o fator geral no bifactor"
                    )
                active_specifics = [
                    factor_id
                    for factor_id in specific_ids
                    if row.cargas.get(factor_id) == 1
                ]
                if len(active_specifics) != 1:
                    raise ValueError(
                        f"Item {row.item_id} precisa carregar exatamente um fator específico"
                    )
                indicator_counts[active_specifics[0]] += 1
            underidentified = [
                factor_id for factor_id, count in indicator_counts.items() if count < 3
            ]
            if underidentified:
                raise ValueError(
                    "Cada fator específico bifatorial exige ao menos 3 itens; insuficientes: "
                    + ", ".join(sorted(underidentified))
                )
        else:
            factor_order = [factor.factor_id for factor in self.qmatrix.factors]
            signatures = {
                factor_id: tuple(
                    int(row.cargas.get(factor_id, 0)) for row in self.qmatrix.rows
                )
                for factor_id in factor_order
            }
            if len(set(signatures.values())) != len(signatures):
                raise ValueError(
                    "Fatores compensatórios não podem ter vetores de carga idênticos"
                )
            indicator_counts = {
                factor.factor_id: sum(
                    row.cargas.get(factor.factor_id) == 1
                    for row in self.qmatrix.rows
                )
                for factor in self.qmatrix.factors
            }
            underidentified = [
                factor_id for factor_id, count in indicator_counts.items() if count < 3
            ]
            if underidentified:
                raise ValueError(
                    "Cada fator compensatório exige ao menos 3 itens; insuficientes: "
                    + ", ".join(sorted(underidentified))
                )
        build_model_spec_from_qmatrix(
            self.qmatrix,
            family=self.familia,
            scoring_factor_id=self.scoring_factor_id,
        )
        return self


class ResponseCalibrarMIRT(BaseModel):
    """Confirmação de enfileiramento da calibração MIRT."""

    job_id: str
    status: Literal["queued"]
    status_url: str


class ResponseJobMIRT(BaseModel):
    """Estado persistido de uma calibração MIRT."""

    job_id: str
    status: Literal["queued", "running", "done", "error"]
    progress: float
    stage: Optional[str] = None
    simulado_id: str
    area: str
    ano_referencia: int
    n_alunos: int
    n_itens: int
    model_id: Optional[str] = None
    artifact_id: Optional[str] = None
    fit_summary: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    elapsed_seconds: Optional[float] = None


class RequestEstimarLoteMIRT(BaseModel):
    """Escora vários alunos com uma única chamada ao worker MIRT."""

    area: str
    ano_referencia: int = Field(default=2024, ge=2009, le=2100)
    simulado_id: Optional[str] = Field(default=None, max_length=128)
    artefato_id: Optional[str] = Field(default=None, max_length=128)
    item_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=45,
        description="Ordem explícita dos itens; deve coincidir com a Q-matrix do artefato",
    )
    transformacao: Optional[Dict[str, float]] = None
    alunos: List[AlunoMIRT] = Field(..., min_length=1, max_length=10_000)

    @field_validator("area")
    @classmethod
    def _normalizar_area(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in {"LC", "CH", "CN", "MT"}:
            raise ValueError("area deve ser LC, CH, CN ou MT")
        return value

    @field_validator("simulado_id", "artefato_id")
    @classmethod
    def _validar_ids_publicos(cls, value: Optional[str], info: Any) -> Optional[str]:
        return _validar_id_publico(value, field_name=info.field_name)

    @field_validator("transformacao")
    @classmethod
    def _validar_transformacao(
        cls, value: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        return _validar_transformacao_latente(value)

    @field_validator("item_ids")
    @classmethod
    def _validar_item_ids(cls, value: List[str]) -> List[str]:
        normalized: List[str] = []
        for item_id in value:
            normalized.append(validate_mirt_item_id(item_id))
        if len(normalized) != len(set(normalized)):
            raise ValueError("item_ids duplicados no lote")
        return normalized

    @model_validator(mode="after")
    def _validar_ids(self) -> "RequestEstimarLoteMIRT":
        if (self.simulado_id is None) == (self.artefato_id is None):
            raise ValueError("Informe exatamente um entre simulado_id e artefato_id")
        ids = [aluno.aluno_id for aluno in self.alunos]
        if len(ids) != len(set(ids)):
            raise ValueError("aluno_id duplicado no lote")
        return self


class ResultadoAlunoMIRTLote(BaseModel):
    """Resultado multidimensional individual dentro de um lote."""

    aluno_id: str
    nome: Optional[str] = ""
    acertos: int
    itens_respondidos: int
    itens_ausentes: int
    completo: bool
    elegivel_nota_ancorada: bool
    theta: float
    se: float
    nota_ancorada: Optional[float] = None
    estado_tri: Optional[str] = None
    scores_mirt: Dict[str, float]
    se_mirt: Dict[str, float]
    avisos: List[str] = Field(default_factory=list)


class ResponseEstimarLoteMIRT(BaseModel):
    """Resposta do scoring MIRT em lote."""

    status: Literal["sucesso"]
    area: str
    ano_referencia: int
    artefato_id: str
    qmatrix_version: str
    modelo_usado: Literal["mirt"] = "mirt"
    n_alunos: int
    resultados: List[ResultadoAlunoMIRTLote]


# ════════════════════════════════════════════════════════════════════════════════
# MODELOS — FRONTEND (CSV Upload Workflow)
# ════════════════════════════════════════════════════════════════════════════════

class DiagnosticoArea(BaseModel):
    """Per-area diagnostic info returned with CSV processing results."""
    area: str
    n_itens: int
    n_alunos: int
    b_medio: float
    b_desvio: float = 0.0
    nota_media: float
    nota_min: float
    nota_max: float
    pct_dentro_intervalo: float
    # Primi metrics
    itens_faceis: int = 0
    itens_medios: int = 0
    itens_dificeis: int = 0
    acert_dif_medio: float = 0.0


class ResultadoAlunoFrontend(BaseModel):
    """Per-student result for the frontend table."""
    aluno_id: str
    acertos_lc: Optional[int] = None
    theta_lc: Optional[float] = None
    nota_lc: Optional[float] = None
    estado_lc: Optional[str] = None
    acert_dif_lc: Optional[float] = None
    acertos_ch: Optional[int] = None
    theta_ch: Optional[float] = None
    nota_ch: Optional[float] = None
    estado_ch: Optional[str] = None
    acert_dif_ch: Optional[float] = None
    acertos_cn: Optional[int] = None
    theta_cn: Optional[float] = None
    nota_cn: Optional[float] = None
    estado_cn: Optional[str] = None
    acert_dif_cn: Optional[float] = None
    acertos_mt: Optional[int] = None
    theta_mt: Optional[float] = None
    nota_mt: Optional[float] = None
    estado_mt: Optional[str] = None
    acert_dif_mt: Optional[float] = None


class PontoReferencia(BaseModel):
    """One point in the reference scale (for the anchor graph)."""
    acertos: int
    tri_min: float
    tri_med: float
    tri_max: float


class ResponseProcessarCSV(BaseModel):
    """Response for POST /api/v3/processar-csv."""
    status: str
    ano_referencia: int
    n_alunos: int
    areas_processadas: List[str]
    resultados: List[ResultadoAlunoFrontend]
    diagnosticos: List[DiagnosticoArea]
    referencia: List[PontoReferencia] = []
    avisos: List[str] = []
    tempo_processamento: float = 0.0
    # V2.1: enriched per-student data with 8 dimensions
    resultados_estruturados: List[Dict[str, Any]] = []
    metodologia: Optional[Dict[str, Any]] = None  # banner/diagnóstico data
    coorte_stats: Optional[Dict[str, Any]] = None  # cohort-level statistics


# ════════════════════════════════════════════════════════════════════════════════
# MODELOS — V2.1 ENRICHED OUTPUT (8 dimensões por aluno)
# ════════════════════════════════════════════════════════════════════════════════

class DimRaw(BaseModel):
    """Dimensão 1: Dados brutos do aluno."""
    acertos: int
    total_itens: int
    pct_acerto: float


class DimIRT(BaseModel):
    """Dimensão 2: Estimativa IRT (theta + estatísticas da coorte)."""
    theta: float
    se: float
    z_cohort: float  # z-score do theta dentro da coorte
    percentil_cohort: float  # 0-100, posição percentual na coorte
    metodo_calibracao: str = "MML + L-BFGS-B + EAP (Gauss-Hermite quadrature)"


class DimAncoragem(BaseModel):
    """Faixa de referência oficial INEP para esse número de acertos."""
    tri_min: float
    tri_med: float
    tri_max: float
    tri_maxima_oficial_area: float
    tri_minima_oficial_area: float


class DimPerformance(BaseModel):
    """Dimensão 3: Nota TRI calculada e como foi obtida."""
    nota_tri: float
    metodo_ancoragem: str
    ancoragem: DimAncoragem


class DimComparacaoNacional(BaseModel):
    """Comparação com a distribuição nacional do ENEM."""
    mediana_nacional: float
    p99_real: float
    p99_9_real: float
    max_real: float
    percentil_estimado_nacional: float  # 0-100, posição estimada nacionalmente


class DimDesempenhoPedagogico(BaseModel):
    """Dimensão 4: Interpretação pedagógica do score."""
    nivel: str  # AVANCADO, PROFICIENTE, BASICO, ELEMENTAR, INICIANTE
    interpretacao: str  # human-readable
    descricao_pedagogica: str  # what this level can do
    comparacao_nacional: DimComparacaoNacional


class DimPosicaoRelativa(BaseModel):
    """Dimensão 5: Posição do aluno relativa à coorte e ao Brasil."""
    rank_cohort: int  # 1-based, 1 = melhor da turma
    total_cohort: int
    percentil_cohort: float  # 0-100
    alunos_acima_cohort: int
    alunos_abaixo_cohort: int
    posicao_nacional_estimada_pct: float  # 0-100


class DimProjecaoAprovacao(BaseModel):
    """Dimensão 6: Projeção em vestibulares e cursos."""
    sisu_baixa_concorrencia: bool  # cursos com corte 400-500 (Letras, Pedagogia)
    sisu_media_concorrencia: bool  # cursos 500-650 (Engenharia média, Adm)
    sisu_alta_concorrencia: bool   # cursos 650-750 (Eng top, Direito)
    sisu_topo: bool                # Medicina (>750)
    cursos_provaveis: List[str] = []
    cursos_disputados: List[str] = []
    observacao: str


class DimRecomendacoes(BaseModel):
    """Dimensão 7: Recomendações pedagógicas."""
    pontos_fortes: List[str] = []
    pontos_atencao: List[str] = []
    foco_estudo: str
    tempo_estudo_estimado_h: int  # horas/semana sugeridas
    estrategia: str


class DimPrimi(BaseModel):
    """Dimensão 8: Métricas de Primi (acert_dif, multidimensionalidade)."""
    acert_dif: float  # proporção de itens difíceis acertados
    interpretacao: str
    itens_faceis_acertados: int
    itens_medios_acertados: int
    itens_dificeis_acertados: int
    fator_secundario_detectado: bool = False
    nota_referencia: str = (
        "Métrica baseada em Primi & Cicchetto (2019). "
        "Estudantes com alto acert_dif e baixos acertos totais podem ser "
        "subestimados pelo modelo 3PL unidimensional."
    )


class DimReferenciaINEP(BaseModel):
    """Dimensão 9: Conexão com referências oficiais INEP."""
    ano_referencia: int
    area: str
    transformacao_oficial: Dict[str, Any]
    metodologia_inep_url: str = (
        "https://www.gov.br/inep/pt-br/areas-de-atuacao/avaliacao-e-exames-educacionais/enem"
    )
    papers_relacionados: List[Dict[str, str]] = []
    validacao_xtri: Dict[str, Any]


class ResultadoAlunoEstruturado(BaseModel):
    """Aluno com 9 dimensões enriquecidas (V2.1)."""
    aluno_id: str
    raw: DimRaw
    irt: DimIRT
    performance: DimPerformance
    desempenho_pedagogico: DimDesempenhoPedagogico
    posicao_relativa: DimPosicaoRelativa
    projecao_aprovacao: DimProjecaoAprovacao
    recomendacoes: DimRecomendacoes
    primi: DimPrimi
    referencia_inep: DimReferenciaINEP
    metadata: Dict[str, Any]


# ════════════════════════════════════════════════════════════════════════════════
# MODELOS — V4 (Validação INEP)
# ════════════════════════════════════════════════════════════════════════════════

class TransformacaoENEM(BaseModel):
    """Transformação empírica (σ, μ) derivada dos microdados INEP."""
    ano: int
    area: str
    sigma: float
    mu: float
    n_alunos: int
    correlacao: float
    mae: float


class ItemINEP(BaseModel):
    """Item com parâmetros oficiais do INEP."""
    co_item: int
    posicao: int
    gabarito: Optional[str] = None
    a: float
    b: float
    c: float
    tp_lingua: Optional[int] = None
    abandonado: bool = False


class ProvaINEP(BaseModel):
    """Prova com seus itens oficiais."""
    ano: int
    area: str
    co_prova: int
    descricao: Optional[str] = None
    is_reaplicacao: bool = False
    itens: List[ItemINEP]


class ENEMInfo(BaseModel):
    """Sumário do que está disponível no item bank ENEM."""
    anos_disponiveis: List[int]
    areas: List[str]
    transformacoes: List[TransformacaoENEM]
    total_itens: int
    total_provas: int


class ResultadoAlunoINEP(BaseModel):
    """Resultado de um aluno calculado com parâmetros oficiais INEP."""
    aluno_id: str
    acertos: int
    theta: float
    nota_inep: float
    estado_tri: str


class RequestCalcularComINEP(BaseModel):
    """Payload para cálculo usando parâmetros oficiais do INEP."""
    ano: int
    area: str
    co_prova: int  # versão da prova (define quais itens usar)
    tp_lingua: Optional[int] = None  # 0=inglês, 1=espanhol (obrigatório se area=LC)
    alunos: List[RespostaAlunoCientifico]


class ResponseCalcularComINEP(BaseModel):
    """Response com resultados usando parâmetros oficiais INEP."""
    status: str
    ano: int
    area: str
    co_prova: int
    n_alunos: int
    n_itens: int
    transformacao_usada: TransformacaoENEM
    resultados: List[ResultadoAlunoINEP]
    diagnostico: Dict[str, Any]
    tempo_processamento: float


class ResultadoComparacao(BaseModel):
    """Resultado de um aluno comparando V3 (local) vs V4 (INEP)."""
    aluno_id: str
    acertos: int
    # V3: calibração local
    theta_v3: float
    nota_v3: float
    estado_v3: str
    # V4: parâmetros oficiais INEP
    theta_v4: float
    nota_v4: float
    estado_v4: str
    # Primi acert_dif computed from INEP's official item difficulties (b > 1.5).
    # Lets the frontend color V4 dots by hard-item mastery using ground-truth
    # difficulties instead of the local V3 calibration. Default 0.0 when the
    # prova has no items classified as "hard" by INEP.
    acert_dif_v4: float = 0.0
    # Diferença
    diff: float


class EstatisticasComparacao(BaseModel):
    """Estatísticas agregadas da comparação V3 vs V4."""
    n_alunos: int
    mae: float
    rmse: float
    correlacao: float
    mean_diff: float
    max_diff_abs: float
    pct_concordancia_nivel: float
    nota_media_v3: float
    nota_media_v4: float


class ResponseValidarSimulado(BaseModel):
    """Response do endpoint de validação V3 vs V4."""
    status: str
    ano: int
    area: str
    co_prova: int
    n_alunos: int
    n_itens: int
    transformacao_inep: TransformacaoENEM
    estatisticas: EstatisticasComparacao
    resultados: List[ResultadoComparacao]
    avisos: List[str] = []
    tempo_processamento: float


# ════════════════════════════════════════════════════════════════════════════════
# MODELOS — V5 (Microdata Analysis: Phase 8)
# ════════════════════════════════════════════════════════════════════════════════


class MicrodataYearStatus(BaseModel):
    """Status of a single microdata year — file present, cache state, etc."""
    year: int
    csv_path: str
    csv_exists: bool
    csv_size_mb: Optional[float] = None
    csv_mtime: Optional[float] = None
    cache_raw_present: bool
    cache_calibrated_present: bool
    n_sample: Optional[int] = None


class MicrodataAvailabilityResponse(BaseModel):
    """List of years that have a microdata CSV on disk."""
    anos: List[int]
    microdata_dir: str


class MicrodataPrepareRequest(BaseModel):
    """Request body for kicking off the load + calibrate pipeline."""
    year: int
    sample_size: int = 200_000
    rng_seed: int = 42
    force_rerun: bool = False  # if True, ignore existing cache and re-run


class MicrodataPrepareResponse(BaseModel):
    """Returned immediately by /api/v5/microdata/prepare."""
    job_id: str
    status: str  # queued | running | done | error
    message: Optional[str] = None


class MicrodataJobStatusResponse(BaseModel):
    """Polled by the frontend to track pipeline progress."""
    job_id: str
    status: str  # queued | running | done | error
    progress: float = 0.0  # 0.0..1.0
    stage: Optional[str] = None
    started_at: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    error: Optional[str] = None


class MicrodataDifficultyRow(BaseModel):
    """One row of the cohort-empirical difficulty table for a single area."""
    acertos: int
    freq: int
    tri_min: float
    tri_med: float
    tri_max: float


class MicrodataAreaSummary(BaseModel):
    """Per-area calibration summary (V3 Local vs official INEP)."""
    n: int
    correlation: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mean_v3: Optional[float] = None
    mean_ref: Optional[float] = None
    mean_shift: Optional[float] = None


class MicrodataAreaDashboardResponse(BaseModel):
    """Single-area dashboard: 1×3 PNG + summary + difficulty table."""
    year: int
    area: str
    n_students: int
    summary: MicrodataAreaSummary
    dashboard_png_base64: str
    difficulty_table: List[MicrodataDifficultyRow]


class MicrodataFullReportResponse(BaseModel):
    """4×3 full report PNG + per-area summaries."""
    year: int
    n_students: int
    summaries: Dict[str, MicrodataAreaSummary]
    full_report_png_base64: str


# ════════════════════════════════════════════════════════════════════════════════
# MODELOS — Runtime local / desktop
# ════════════════════════════════════════════════════════════════════════════════


class RuntimeDirectoryStatus(BaseModel):
    """Estado resumido de um diretório do runtime local."""
    path: str
    exists: bool
    writable: bool = False
    file_count: int = 0
    size_bytes: int = 0


class RuntimeModelStatus(BaseModel):
    """Resumo de um modelo ativo conhecido pelo registry local."""
    ano_referencia: int
    area: str
    scope_id: str
    modelo: str
    artifact_id: Optional[str] = None
    artifact_dir: Optional[str] = None
    saved_at_utc: Optional[str] = None
    status: str


class RuntimeJobStatus(BaseModel):
    """Resumo de um job local em memória, exposto ao desktop."""
    job_id: str
    year: int
    sample_size: int
    rng_seed: int
    status: str
    progress: float = 0.0
    stage: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    error: Optional[str] = None


class RuntimeDesktopStatusResponse(BaseModel):
    """Snapshot do runtime local para o app desktop."""
    app_support_dir: str
    directories: Dict[str, RuntimeDirectoryStatus]
    models: List[RuntimeModelStatus]
    job_counts: Dict[str, int]
    jobs: List[RuntimeJobStatus]
    ano_referencia: int
    simulado_id: Optional[str] = None
    status_areas: Dict[str, str] = Field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════════
# MODELOS — SIMULADO BINARIZADO (POST /api/v3/processar-json)
# ════════════════════════════════════════════════════════════════════════════════

class AlunoBinarizado(BaseModel):
    """Um aluno do simulado: respostas 0/1 na ordem das questões (1..N)."""
    id: Optional[str] = None
    matricula: Optional[str] = None
    nome: Optional[str] = None
    turma: Optional[str] = None
    fez_area: Dict[str, bool] = Field(
        default_factory=dict,
        description="Área ausente aqui conta como feita; false exclui o aluno da área",
    )
    presenca: Optional[Dict[Literal["dia1", "dia2"], bool]] = Field(
        None,
        description=(
            "Opcional. Presença por dia (dia1 = LC e CH, dia2 = CN e MT). false marca o aluno "
            "como AUSENTE no dia; sem o campo, o dia é ausente quando todas as respostas dele são null"
        ),
    )
    acertos: Optional[Dict[str, int]] = Field(None, description="Ignorado; recalculado das respostas")
    respostas: List[Optional[int]] = Field(..., min_length=1, max_length=400)


class RequestSimuladoBinarizado(BaseModel):
    """Simulado ENEM inteiro, no formato binarizado da plataforma."""
    ano_referencia: int = Field(2023, description="Ano da escala ENEM (ver /api/v3/anos-referencia)")
    total_questoes: Optional[int] = None
    total_alunos: Optional[int] = None
    areas: Dict[str, List[int]] = Field(
        default_factory=lambda: {"LC": [1, 45], "CH": [46, 90], "CN": [91, 135], "MT": [136, 180]},
        description="Faixa [início, fim] de questões de cada área (1-indexado, inclusivo)",
    )
    alunos: List[AlunoBinarizado] = Field(..., min_length=1, max_length=50_000)

    @field_validator("areas")
    @classmethod
    def _faixas_com_dois_limites(cls, v: Dict[str, List[int]]) -> Dict[str, List[int]]:
        for area, faixa in v.items():
            if len(faixa) != 2:
                raise ValueError(f"Faixa de {area} deve ser [início, fim]")
        return v


class ItemBinarizado(BaseModel):
    questao: int
    a: float
    b: float
    c: float
    taxa_acerto: float
    r_item_resto: Optional[float] = None
    saturado: bool = False


class ResumoAreaBinarizado(BaseModel):
    acertos_medio: float
    theta_medio: float
    nota_media: float
    nota_dp: float
    nota_min: float
    nota_mediana: float
    nota_max: float


class ExecucaoArea(BaseModel):
    """Como a nota da área foi calculada (para gravar junto das notas)."""
    motor_versao: str = Field(..., description="Hash do commit do motor, ou 'desconhecida'")
    ano_referencia: Optional[int] = None
    c_strategy: str
    em_iteracoes: int
    em_convergiu: bool
    processado_em: str = Field(..., description="ISO 8601 no fuso America/Fortaleza")


class AreaBinarizado(BaseModel):
    questoes: List[int]
    n_alunos: int
    n_ausentes: int = 0
    execucao: ExecucaoArea
    modelo: str
    resumo: ResumoAreaBinarizado
    curva_condicional: Dict[str, Any]
    itens: List[ItemBinarizado]


class NotaAreaBinarizado(BaseModel):
    acertos: Optional[int] = None
    theta: Optional[float] = None
    se_theta: Optional[float] = None
    posicao_no_grupo: Optional[float] = Field(None, description="u: posição entre quem teve os mesmos acertos (0,01–0,99)")
    nota_tri: Optional[float] = Field(None, description="null quando o aluno está AUSENTE no dia da área")
    nivel: str
    ausente: bool = False


class ResultadoAlunoBinarizado(BaseModel):
    id: Optional[str] = None
    matricula: Optional[str] = None
    nome: Optional[str] = None
    turma: Optional[str] = None
    fez_area: Dict[str, bool]
    ausente_dia: Dict[str, bool] = Field(default_factory=dict)
    areas: Dict[str, NotaAreaBinarizado]
    media_tri: Optional[float] = Field(None, description="Média das áreas; null se faltar alguma área")


class ResponseSimuladoBinarizado(BaseModel):
    status: Literal["sucesso"]
    ano_referencia: int
    metodo: Dict[str, Any]
    total_alunos: int
    total_questoes: int
    areas: Dict[str, AreaBinarizado]
    alunos: List[ResultadoAlunoBinarizado]
    avisos: List[str] = []
    tempo_processamento: float = 0.0
