"""
Estado compartilhado da aplicação, carregado no lifespan do FastAPI.

Centraliza as tabelas de referência TRI (oficial e multi-ano) para que os
routers não precisem acessar globais de ``app.main``. O lifespan chama
``init(...)`` e os routers usam os getters.
"""

from typing import Dict, Optional

from .tabela_referencia import TabelaReferenciaTRI

_tabela: Optional[TabelaReferenciaTRI] = None
_tabelas_multiano: Dict[int, TabelaReferenciaTRI] = {}


def init(
    tabela: Optional[TabelaReferenciaTRI],
    tabelas_multiano: Dict[int, TabelaReferenciaTRI],
) -> None:
    """Registra as tabelas carregadas no lifespan."""
    global _tabela, _tabelas_multiano
    _tabela = tabela
    _tabelas_multiano = tabelas_multiano


def get_tabela() -> Optional[TabelaReferenciaTRI]:
    """Tabela de referência oficial (ano único), ou None se não carregou."""
    return _tabela


def get_tabelas_multiano() -> Dict[int, TabelaReferenciaTRI]:
    """Tabelas de referência multi-ano (chave = ano de referência)."""
    return _tabelas_multiano
