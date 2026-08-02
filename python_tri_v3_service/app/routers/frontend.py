"""Frontend — páginas estáticas (deve ser incluído DEPOIS das rotas de API)."""

import os

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")

_NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


@router.get("/", include_in_schema=False)
async def frontend():
    """Serve the frontend application."""
    return FileResponse(os.path.join(_static_dir, "index.html"), headers=_NO_CACHE_HEADERS)


@router.get("/validator", include_in_schema=False)
async def frontend_validator():
    """Serve the INEP validator page."""
    return FileResponse(os.path.join(_static_dir, "validator.html"), headers=_NO_CACHE_HEADERS)


@router.get("/microdata", include_in_schema=False)
async def frontend_microdata():
    """Serve the microdata analysis page (Phase 8)."""
    return FileResponse(os.path.join(_static_dir, "microdata.html"), headers=_NO_CACHE_HEADERS)
