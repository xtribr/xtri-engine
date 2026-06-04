"""
Persistência local de artefatos calibrados do motor unidimensional.

Os artefatos ficam fora do bundle do app e são versionados por
`ano/area/scope/modelo/artifact_id` em Application Support do macOS.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..runtime.local_paths import APP_SUPPORT_ROOT, MODELS_ROOT, UNIDIM_MODELS_ROOT

DEFAULT_UNIDIM_MODEL = "unidim_mml_eap_v1"


def _slug_scope(scope_id: str) -> str:
    """Normaliza o identificador do escopo para uso seguro no filesystem."""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", scope_id.strip())
    return normalized.strip("-") or "sem-escopo"


def _write_json(path: Path, payload: Any) -> None:
    """Escreve JSON de forma atômica para evitar arquivo truncado."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def artifact_dir_for(
    *,
    ano_referencia: int,
    area: str,
    scope_id: str,
    modelo: str = DEFAULT_UNIDIM_MODEL,
    artifact_id: Optional[str] = None,
) -> Path:
    """Resolve o diretório do artefato unidimensional."""

    if artifact_id is None:
        artifact_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid.uuid4().hex[:8]
        )
    return (
        UNIDIM_MODELS_ROOT
        / str(ano_referencia)
        / area.upper()
        / _slug_scope(scope_id)
        / modelo
        / artifact_id
    )


def save_unidim_artifact(
    *,
    simulado_id: str,
    area: str,
    ano_referencia: int,
    c_fixo: float,
    parametros_itens: list[dict[str, Any]],
    transformacao: Dict[str, Any],
    diagnostico: Dict[str, Any],
    n_alunos: int,
    n_itens: int,
    modelo: str = DEFAULT_UNIDIM_MODEL,
) -> Dict[str, Any]:
    """Salva o artefato do modelo unidimensional e retorna o manifesto."""

    artifact_dir = artifact_dir_for(
        ano_referencia=ano_referencia,
        area=area,
        scope_id=simulado_id,
        modelo=modelo,
    )
    artifact_id = artifact_dir.name
    saved_at = datetime.now(timezone.utc).isoformat()

    manifest = {
        "artifact_id": artifact_id,
        "artifact_dir": str(artifact_dir),
        "simulado_id": simulado_id,
        "scope_type": "simulado_id",
        "scope_id": simulado_id,
        "ano_referencia": ano_referencia,
        "area": area.upper(),
        "modelo": modelo,
        "c_fixo": c_fixo,
        "n_alunos": n_alunos,
        "n_itens": n_itens,
        "saved_at_utc": saved_at,
        "arquivos": {
            "manifesto": "manifest.json",
            "parametros_itens": "item_params.json",
            "transformacao": "transformacao.json",
            "diagnostico": "diagnostico.json",
        },
    }

    _write_json(artifact_dir / "item_params.json", parametros_itens)
    _write_json(artifact_dir / "transformacao.json", transformacao)
    _write_json(artifact_dir / "diagnostico.json", diagnostico)
    _write_json(artifact_dir / "manifest.json", manifest)

    return manifest


def load_unidim_artifact(artifact_dir: Path | str) -> Dict[str, Any]:
    """Carrega um artefato salvo a partir do diretório do manifesto."""

    artifact_dir = Path(artifact_dir)
    manifest = _read_json(artifact_dir / "manifest.json")
    return {
        "manifest": manifest,
        "parametros_itens": _read_json(artifact_dir / "item_params.json"),
        "transformacao": _read_json(artifact_dir / "transformacao.json"),
        "diagnostico": _read_json(artifact_dir / "diagnostico.json"),
    }


def load_unidim_artifact_by_id(
    *,
    ano_referencia: int,
    area: str,
    artifact_id: str,
    modelo: str = DEFAULT_UNIDIM_MODEL,
) -> Optional[Dict[str, Any]]:
    """Localiza e carrega um artefato por `artifact_id` dentro de `ano/area`."""

    base_dir = UNIDIM_MODELS_ROOT / str(ano_referencia) / area.upper()
    if not base_dir.is_dir():
        return None

    manifests = list(base_dir.glob(f"*/{modelo}/{artifact_id}/manifest.json"))
    if not manifests:
        return None

    return load_unidim_artifact(manifests[0].parent)
