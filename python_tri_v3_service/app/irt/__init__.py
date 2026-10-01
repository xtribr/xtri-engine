"""
Núcleo estatístico do XTRI Engine.

Este pacote concentra contratos e funções reutilizáveis de scoring e
transformação para desacoplar o motor científico das rotas HTTP e do fluxo
de microdados.
"""

from .conditional import curva_teorica, nota_por_posicao, posicao_relativa
from .contracts import ItemParameter, ThetaEstimate
from .reduced_scale import ancorar_nota_reduzida, obter_faixa_referencia_continua
from .scoring import P, eap_theta, estado_tri
from .transform import ancorar_nota_individual, ancorar_notas, nota_individual_pelo_ponto_medio
from .unidim_em import calibrar_itens_mml_em

__all__ = [
    "ItemParameter",
    "ThetaEstimate",
    "P",
    "ancorar_nota_reduzida",
    "curva_teorica",
    "nota_por_posicao",
    "posicao_relativa",
    "obter_faixa_referencia_continua",
    "eap_theta",
    "estado_tri",
    "ancorar_nota_individual",
    "ancorar_notas",
    "nota_individual_pelo_ponto_medio",
    "calibrar_itens_mml_em",
]
