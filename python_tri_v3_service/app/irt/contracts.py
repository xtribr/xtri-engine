"""
Contratos tipados para o núcleo IRT.

Mantém a estrutura dos dicionários já usados no projeto, mas explicita os
campos esperados para facilitar a próxima etapa de persistência de artefatos.
"""

try:
    from typing import NotRequired, TypedDict
except ImportError:
    from typing import TypedDict
    from typing_extensions import NotRequired


class ItemParameter(TypedDict):
    """Parâmetros calibrados de um item no formato interno atual."""

    a: float
    b: float
    c: float
    taxa_acerto: float
    saturado: bool


class ThetaEstimate(TypedDict):
    """Estimativa por aluno antes ou depois da transformação para nota."""

    aluno_id: str
    nome: str
    acertos: int
    theta: float
    se: float
    acertos_faceis: NotRequired[int]
    acertos_medios: NotRequired[int]
    acertos_dificeis: NotRequired[int]
    acert_dif: NotRequired[float]
    nota_ancorada: NotRequired[float]
    estado_tri: NotRequired[str]
