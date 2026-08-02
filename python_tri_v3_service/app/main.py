"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    XTRI ENGINE API — TRI V3 (FastAPI)                        ║
║                                                                              ║
║  API híbrida com dois motores de estimação TRI:                              ║
║  • Motor Heurístico: Rápido, compatível com V2 (coerência pedagógica)       ║
║  • Motor Científico: MML + EAP + Ancoragem (calibração real)                ║
║                                                                              ║
║  Porta: 5003 (retrocompatível com V2)                                        ║
║                                                                              ║
╚════════════════════════════════════════════════════════════════════════════════╝

Entrypoint da aplicação: cria o app FastAPI, lifespan, CORS e inclui os
routers de ``app.routers`` (as rotas em si vivem nos módulos de router).

As re-exportações abaixo (mirt_runtime_available, run_mirt_score,
default_csv_path, submit_microdata_job, Path) existem para compatibilidade:
testes e consumidores fazem mock.patch/import desses nomes em ``app.main`` e
os routers resolvem os mesmos nomes via este módulo em runtime.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

# Load .env if available (from project root or current dir)
try:
    from dotenv import load_dotenv
    for candidate in (
        Path(__file__).resolve().parents[2] / ".env",  # /Volumes/KINGSTON/apps/MIRT/.env
        Path(__file__).resolve().parents[1] / ".env",  # python_tri_v3_service/.env
        Path(".env"),
    ):
        if candidate.exists():
            load_dotenv(candidate)
            break
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import state
from .tabela_referencia import TabelaReferenciaTRI
from .jobs.calibration_jobs import submit_microdata_job  # noqa: F401 (re-export)
from .microdata.jobs import job_manager
from .microdata.loader import default_csv_path  # noqa: F401 (re-export)
from .mirt.runner import mirt_runtime_available, run_mirt_score  # noqa: F401 (re-export)
from .runtime.local_paths import ensure_app_support_layout


# ════════════════════════════════════════════════════════════════════════════════
# STARTUP / SHUTDOWN
# ════════════════════════════════════════════════════════════════════════════════

# Mantidos como atributos de módulo por retrocompatibilidade; a fonte de
# verdade para os routers é ``app.state`` (ver state.init abaixo).
tabela: TabelaReferenciaTRI = None  # type: ignore
tabelas_multiano: Dict[int, TabelaReferenciaTRI] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega a tabela de referência na inicialização."""
    global tabela, tabelas_multiano
    ensure_app_support_layout()
    print("=" * 80)
    print("  XTRI ENGINE API — TRI V3 (FastAPI)")
    print("=" * 80)

    try:
        restored_jobs = job_manager.reload_from_disk()
        print(f"  Jobs restaurados do disco: {restored_jobs}")
    except Exception as e:
        print(f"  AVISO: jobs locais não restaurados: {e}")

    try:
        tabela = TabelaReferenciaTRI()
        tabela.validar()
        info = tabela.info()
        print(f"  Tabela de referência carregada: {info['areas']}")
        print(f"  Acertos por área: {info['acertos_por_area']}")
        print(f"  Limites oficiais: {info['limites_oficiais']}")
    except Exception as e:
        print(f"  ERRO ao carregar tabela: {e}")
        tabela = None

    try:
        tabelas_multiano = TabelaReferenciaTRI.carregar_multiano()
        anos = sorted(tabelas_multiano.keys())
        print(f"  Tabelas multi-ano carregadas: {anos}")
    except Exception as e:
        print(f"  AVISO: tabelas multi-ano não carregadas: {e}")
        tabelas_multiano = {}

    state.init(tabela, tabelas_multiano)

    print(f"  Python: {sys.version}")
    print(f"  Servidor: http://0.0.0.0:5003")
    print("=" * 80)

    yield  # A aplicação roda aqui

    print("Encerrando XTRI Engine API...")


# ════════════════════════════════════════════════════════════════════════════════
# APP
# ════════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

app = FastAPI(
    title="XTRI Engine API",
    description="API de estimação TRI para simulados ENEM — Motor Heurístico + Científico",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS aberto (mesmo comportamento do Flask V2)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════════════════════════
# ROUTERS
# ════════════════════════════════════════════════════════════════════════════════

from .routers import (  # noqa: E402
    exports,
    frontend,
    system,
    v2,
    v3,
    v4_inep,
    v5_microdata,
    v5_mirt,
    v6_reduzida,
)

app.include_router(system.router)
app.include_router(v2.router)
app.include_router(v3.router)
app.include_router(v5_mirt.router)
app.include_router(v4_inep.router)
app.include_router(v5_microdata.router)
app.include_router(v6_reduzida.router)
app.include_router(exports.router)
# Frontend por último: as páginas estáticas vêm depois de todas as rotas de API
app.include_router(frontend.router)


# ════════════════════════════════════════════════════════════════════════════════
# FRONTEND — Static Files (must be AFTER all API routes)
# ════════════════════════════════════════════════════════════════════════════════

_static_dir = os.path.join(os.path.dirname(__file__), "static")

if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")
