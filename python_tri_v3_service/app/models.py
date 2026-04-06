"""
Modelos Pydantic para a API TRI V3.

Contém os schemas de entrada/saída para ambos os motores:
- Motor Heurístico (compatível com V2)
- Motor Científico (MML + EAP)
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union


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
    diagnostico: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métricas de qualidade: % no intervalo, MAE, itens saturados"
    )


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
    parametros_itens: List[ItemPreCalibrado]
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
