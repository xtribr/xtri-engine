"""Autenticação por chave para endpoints MIRT de integração externa."""

from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import Header, HTTPException


async def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """Exige `X-API-Key`; o modo aberto demanda opt-out explícito de dev."""

    configured = os.getenv("XTRI_API_KEY", "").strip()
    if not configured:
        allow_open = os.getenv("XTRI_MIRT_ALLOW_UNAUTHENTICATED", "").strip().lower()
        if allow_open in {"1", "true", "yes"}:
            return
        raise HTTPException(
            status_code=503,
            detail=(
                "Autenticação MIRT não configurada; defina XTRI_API_KEY ou habilite "
                "XTRI_MIRT_ALLOW_UNAUTHENTICATED=true apenas em desenvolvimento"
            ),
        )
    if x_api_key is None or not hmac.compare_digest(x_api_key, configured):
        raise HTTPException(
            status_code=401,
            detail="Chave de API ausente ou inválida",
            headers={"WWW-Authenticate": "ApiKey"},
        )
