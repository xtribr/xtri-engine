"""
Núcleo estatístico do XTRI Engine.

Este pacote concentra contratos e funções reutilizáveis de scoring e
transformação para desacoplar o motor científico das rotas HTTP e do fluxo
de microdados.
"""

from .contracts import ItemParameter, ThetaEstimate
from .scoring import P, eap_theta, estado_tri
from .transform import ancorar_nota_individual, ancorar_notas, nota_individual_pelo_ponto_medio
from .unidim_em import calibrar_itens_mml_em

__all__ = [
    "ItemParameter",
    "ThetaEstimate",
    "P",
    "eap_theta",
    "estado_tri",
    "ancorar_nota_individual",
    "ancorar_notas",
    "nota_individual_pelo_ponto_medio",
    "calibrar_itens_mml_em",
]
