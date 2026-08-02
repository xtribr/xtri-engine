"""Endpoints de saúde e debug (compatíveis com V2 e Docker healthcheck)."""

import os
import sys

from fastapi import APIRouter

from .. import state
from ..db.supabase_client import is_supabase_configured
from ..mirt.runner import mirt_runtime_available

router = APIRouter()


@router.get("/health")
async def health():
    """Endpoint de saúde agregado (compatível com V2 e Docker healthcheck)."""
    tabela = state.get_tabela()
    return {
        "status": "online",
        "service": "xtri_engine_api",
        "version": "3.0.0",
        "motores": ["heuristico_v2", "cientifico_mirt"],
        "tabela_carregada": tabela is not None,
        "supabase_configurado": is_supabase_configured(),
        "anthropic_api_key_presente": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "mirt_runtime_disponivel": mirt_runtime_available(),
        "tabelas_multiano_anos": len(state.get_tabelas_multiano()),
    }


@router.get("/api/debug")
async def debug():
    """Endpoint de debug para verificar configuração."""
    tabela = state.get_tabela()
    return {
        "service": "XTRI Engine API — TRI V3",
        "version": "3.0.0",
        "tabela_carregada": tabela is not None,
        "tabela_info": tabela.info() if tabela else None,
        "python_version": sys.version,
        "motores_disponiveis": ["heuristico_v2", "cientifico_mirt"],
    }
