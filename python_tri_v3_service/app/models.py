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
    Tenta scorear via artefato MIRT local e, se necessário, cai para unidim.
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
        description="Transformação opcional `theta -> nota` com campos `mu` e `sigma`",
    )
    permitir_fallback_unidim: bool = Field(
        default=True,
        description="Se True, cai para o modelo unidimensional quando o MIRT não estiver disponível",
    )
    respostas: List[int] = Field(
        ...,
        description="Vetor binário de respostas (0/1)",
    )


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
        description="Modelo originalmente tentado antes do fallback, quando aplicável",
    )
    scores_mirt: Optional[Dict[str, float]] = Field(
        default=None,
        description="Scores por fator retornados pelo worker MIRT",
    )
    se_mirt: Optional[Dict[str, float]] = Field(
        default=None,
        description="Erros padrão por fator retornados pelo worker MIRT",
    )


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
