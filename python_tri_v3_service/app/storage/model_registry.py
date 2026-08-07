"""
Registry local de modelos ativos.

Mantém um ponteiro simples para o artefato ativo por combinação
`ano/area/scope/modelo`, sem depender de banco externo.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Dict, Optional

from .artifacts import APP_SUPPORT_ROOT, DEFAULT_UNIDIM_MODEL

MODEL_REGISTRY_PATH = APP_SUPPORT_ROOT / "models" / "active_models.json"
_REGISTRY_LOCK = threading.RLock()


def infer_model_family(modelo: str) -> str:
    """Infere a família lógica do modelo salvo no registry."""

    normalized = str(modelo).strip().lower()
    if normalized.startswith("mirt_"):
        return "mirt"
    if normalized.startswith("unidim_"):
        return "unidim"
    return "unknown"


def _read_registry() -> Dict[str, Any]:
    if not MODEL_REGISTRY_PATH.is_file():
        return {}
    with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_registry(payload: Dict[str, Any]) -> None:
    MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_REGISTRY_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    tmp_path.replace(MODEL_REGISTRY_PATH)


def registry_key(
    *,
    ano_referencia: int,
    area: str,
    scope_id: str,
    modelo: str = DEFAULT_UNIDIM_MODEL,
) -> str:
    return f"{ano_referencia}:{area.upper()}:{scope_id}:{modelo}"


def set_active_model(
    manifest: Dict[str, Any],
    *,
    modelo: Optional[str] = None,
    exclusive_family: bool = False,
) -> Dict[str, Any]:
    """Registra o manifesto informado como modelo ativo do seu escopo."""

    model_name = modelo or str(manifest["modelo"])
    key = registry_key(
        ano_referencia=int(manifest["ano_referencia"]),
        area=str(manifest["area"]),
        scope_id=str(manifest["scope_id"]),
        modelo=model_name,
    )
    entry = {
        "artifact_id": manifest["artifact_id"],
        "artifact_dir": manifest["artifact_dir"],
        "simulado_id": manifest["simulado_id"],
        "scope_id": manifest["scope_id"],
        "scope_type": manifest.get("scope_type", "simulado_id"),
        "ano_referencia": manifest["ano_referencia"],
        "area": manifest["area"],
        "modelo": model_name,
        "saved_at_utc": manifest["saved_at_utc"],
    }
    with _REGISTRY_LOCK:
        current = _read_registry()
        if exclusive_family:
            family = infer_model_family(model_name)
            for existing_key, existing in list(current.items()):
                same_scope = (
                    int(existing.get("ano_referencia", -1))
                    == int(manifest["ano_referencia"])
                    and str(existing.get("area", "")).upper()
                    == str(manifest["area"]).upper()
                    and str(existing.get("scope_id", "")) == str(manifest["scope_id"])
                )
                if same_scope and infer_model_family(str(existing.get("modelo", ""))) == family:
                    current.pop(existing_key, None)
        current[key] = entry
        _write_registry(current)
    return entry


def get_active_model(
    *,
    ano_referencia: int,
    area: str,
    scope_id: str,
    modelo: str = DEFAULT_UNIDIM_MODEL,
) -> Optional[Dict[str, Any]]:
    """Retorna o modelo ativo de um escopo ou `None` se não existir."""

    current = _read_registry()
    return current.get(
        registry_key(
            ano_referencia=ano_referencia,
            area=area,
            scope_id=scope_id,
            modelo=modelo,
        )
    )


def list_active_models() -> Dict[str, Any]:
    """Retorna o registry bruto para inspeção/diagnóstico local."""

    return _read_registry()


def _filter_active_models(
    *,
    ano_referencia: Optional[int] = None,
    area: Optional[str] = None,
    scope_id: Optional[str] = None,
    modelo: Optional[str] = None,
    family: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """Filtro interno compartilhado pelos helpers públicos do registry."""

    current = _read_registry()
    entries: list[Dict[str, Any]] = []
    for entry in current.values():
        entry_model = str(entry.get("modelo", ""))
        if modelo is not None and entry_model != modelo:
            continue
        if family is not None and infer_model_family(entry_model) != family:
            continue
        if ano_referencia is not None and int(entry.get("ano_referencia", -1)) != ano_referencia:
            continue
        if area is not None and str(entry.get("area", "")).upper() != area.upper():
            continue
        if scope_id is not None and str(entry.get("scope_id", "")) != scope_id:
            continue
        entries.append(entry)

    return sorted(
        entries,
        key=lambda item: (
            int(item.get("ano_referencia", 0)),
            str(item.get("area", "")),
            str(item.get("saved_at_utc", "")),
            str(item.get("scope_id", "")),
        ),
        reverse=True,
    )


def list_active_models_by_area(
    *,
    ano_referencia: Optional[int] = None,
    area: Optional[str] = None,
    modelo: Optional[str] = DEFAULT_UNIDIM_MODEL,
    family: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """Lista modelos ativos filtrando por ano/área/modelo."""

    return _filter_active_models(
        ano_referencia=ano_referencia,
        area=area,
        modelo=modelo,
        family=family,
    )


def list_active_models_by_scope(
    *,
    ano_referencia: Optional[int] = None,
    area: Optional[str] = None,
    scope_id: Optional[str] = None,
    modelo: Optional[str] = None,
    family: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """Lista modelos ativos filtrando por escopo lógico."""

    return _filter_active_models(
        ano_referencia=ano_referencia,
        area=area,
        scope_id=scope_id,
        modelo=modelo,
        family=family,
    )


def list_active_models_by_family(
    *,
    family: str,
    ano_referencia: Optional[int] = None,
    area: Optional[str] = None,
    scope_id: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """Lista modelos ativos apenas pela família lógica do modelo."""

    return _filter_active_models(
        ano_referencia=ano_referencia,
        area=area,
        scope_id=scope_id,
        modelo=None,
        family=family,
    )
