"""
Diretórios locais usados pelo runtime desktop.

Separa semanticamente cache, modelos, jobs e logs dentro de
`~/Library/Application Support/XTRI Engine/`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


APP_SUPPORT_ROOT = Path(
    os.getenv(
        "XTRI_APP_SUPPORT_DIR",
        Path.home() / "Library" / "Application Support" / "XTRI Engine",
    )
)
MODELS_ROOT = APP_SUPPORT_ROOT / "models"
UNIDIM_MODELS_ROOT = MODELS_ROOT / "unidim"
MIRT_MODELS_ROOT = MODELS_ROOT / "mirt"
MICRODATA_CACHE_ROOT = APP_SUPPORT_ROOT / "cache" / "microdata"
JOBS_ROOT = APP_SUPPORT_ROOT / "jobs"
LOGS_ROOT = APP_SUPPORT_ROOT / "logs"
LEGACY_MICRODATA_CACHE_ROOT = APP_SUPPORT_ROOT / "microdata_cache"


def runtime_dirs() -> Dict[str, Path]:
    """Mapa canônico dos diretórios do runtime local."""

    return {
        "app_support": APP_SUPPORT_ROOT,
        "models": MODELS_ROOT,
        "models_unidim": UNIDIM_MODELS_ROOT,
        "models_mirt": MIRT_MODELS_ROOT,
        "cache_microdata": MICRODATA_CACHE_ROOT,
        "jobs": JOBS_ROOT,
        "logs": LOGS_ROOT,
    }


def ensure_app_support_layout() -> Dict[str, Path]:
    """Cria a árvore mínima do runtime local, se ainda não existir."""

    paths = runtime_dirs()
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _count_files(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def _size_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0

    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def _is_writable(path: Path) -> bool:
    if not path.exists():
        return False
    return os.access(path, os.W_OK)


def describe_runtime_layout() -> Dict[str, Dict[str, object]]:
    """Retorna um snapshot leve do layout local para diagnóstico do desktop."""

    ensure_app_support_layout()
    snapshot: Dict[str, Dict[str, object]] = {}
    for name, path in runtime_dirs().items():
        snapshot[name] = {
            "path": str(path),
            "exists": path.exists(),
            "writable": _is_writable(path),
            "file_count": _count_files(path),
            "size_bytes": _size_bytes(path),
        }

    snapshot["cache_microdata_legacy"] = {
        "path": str(LEGACY_MICRODATA_CACHE_ROOT),
        "exists": LEGACY_MICRODATA_CACHE_ROOT.exists(),
        "writable": _is_writable(LEGACY_MICRODATA_CACHE_ROOT),
        "file_count": _count_files(LEGACY_MICRODATA_CACHE_ROOT),
        "size_bytes": _size_bytes(LEGACY_MICRODATA_CACHE_ROOT),
    }
    return snapshot
