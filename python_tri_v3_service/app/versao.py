"""
Identificação da execução do motor TRI.

Responsabilidades:
- Informar a versão do motor (hash do commit) para registrar junto das notas.
  Ordem de busca: variável ``XTRI_MOTOR_VERSAO`` (definida no build da imagem
  a partir do ``GIT_COMMIT``), depois ``git rev-parse`` no checkout local;
  sem nenhum dos dois, ``"desconhecida"``.
- Carimbar o horário da execução no fuso America/Fortaleza.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

FUSO_PADRAO = "America/Fortaleza"


@lru_cache(maxsize=1)
def motor_versao() -> str:
    """Hash do commit do motor, ou ``"desconhecida"``."""

    for var in ("XTRI_MOTOR_VERSAO", "GIT_COMMIT"):
        valor = os.getenv(var, "").strip()
        if valor:
            return valor
    try:
        saida = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        valor = saida.stdout.strip()
        if valor:
            return valor
    except (OSError, subprocess.SubprocessError):
        pass
    return "desconhecida"


def _fuso_fortaleza():
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(FUSO_PADRAO)
    except Exception:
        # Imagens slim podem não ter tzdata; Fortaleza não tem horário de verão.
        return timezone(timedelta(hours=-3), FUSO_PADRAO)


def agora_fortaleza() -> str:
    """Data e hora atuais em ISO 8601 com o fuso de Fortaleza (UTC-3)."""

    return datetime.now(_fuso_fortaleza()).isoformat(timespec="seconds")
