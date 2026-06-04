"""Tests do resolver de artefatos MIRT no runtime local."""

from __future__ import annotations

import json

import pytest

from app.mirt.qmatrix import build_habilidade_qmatrix, build_model_spec_from_qmatrix
from app.mirt.runner import build_mirt_output_dir
from app.runtime.mirt import (
    RuntimeMIRTArtifactAmbiguousError,
    RuntimeMIRTArtifactNotFoundError,
    resolve_mirt_artifact,
)
from app.storage.model_registry import set_active_model


def _save_active_mirt_artifact(
    *,
    simulado_id: str,
    area: str,
    ano_referencia: int = 2024,
    artifact_id: str | None = None,
    register_active: bool = True,
):
    item_rows = [
        {"item_id": f"{area.lower()}_001", "co_item": 1001, "numero_habilidade": 1, "posicao": 1},
        {"item_id": f"{area.lower()}_002", "co_item": 1002, "numero_habilidade": 2, "posicao": 2},
    ]
    qmatrix = build_habilidade_qmatrix(area=area, item_rows=item_rows)
    model = build_model_spec_from_qmatrix(qmatrix)
    artifact_id = artifact_id or f"job_{simulado_id}"
    artifact_dir = build_mirt_output_dir(
        year=ano_referencia,
        area=area,
        model_id=model.model_id,
        job_id=artifact_id,
    )

    (artifact_dir / "model.rds").write_text("placeholder", encoding="utf-8")
    (artifact_dir / "item_params.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "factor_scores.csv").write_text("student_id,G,SE_G\ns1,0.2,0.1\n", encoding="utf-8")
    (artifact_dir / "fit_summary.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "diagnostics.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "qmatrix_snapshot.json").write_text(
        qmatrix.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "worker_output.json").write_text(
        json.dumps(
            {
                "status": "done",
                "artifact_dir": str(artifact_dir),
                "n_students": 10,
                "n_items": 2,
                "model_spec": model.model_dump(mode="json"),
                "artifact_files": {
                    "model_rds_path": str(artifact_dir / "model.rds"),
                    "item_params_path": str(artifact_dir / "item_params.json"),
                    "factor_scores_path": str(artifact_dir / "factor_scores.csv"),
                    "fit_summary_path": str(artifact_dir / "fit_summary.json"),
                    "diagnostics_path": str(artifact_dir / "diagnostics.json"),
                    "qmatrix_snapshot_path": str(artifact_dir / "qmatrix_snapshot.json"),
                },
                "warnings": [],
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    if register_active:
        set_active_model(
            {
                "artifact_id": artifact_id,
                "artifact_dir": str(artifact_dir),
                "simulado_id": simulado_id,
                "scope_type": "simulado_id",
                "scope_id": simulado_id,
                "ano_referencia": ano_referencia,
                "area": area,
                "modelo": model.model_id,
                "saved_at_utc": "2026-04-08T12:00:00+00:00",
            },
            modelo=model.model_id,
        )

    return {
        "artifact_id": artifact_id,
        "artifact_dir": str(artifact_dir),
        "model_id": model.model_id,
    }


class TestResolveMIRTArtifact:
    @pytest.mark.unit
    def test_resolve_by_artifact_id(self, temp_model_storage):
        artifact = _save_active_mirt_artifact(simulado_id="sim_mirt_a", area="MT", register_active=False)

        loaded = resolve_mirt_artifact(
            ano_referencia=2024,
            area="MT",
            artifact_id=artifact["artifact_id"],
        )

        assert loaded["manifest"]["artifact_id"] == artifact["artifact_id"]

    @pytest.mark.unit
    def test_resolve_by_scope_id(self, temp_model_storage):
        artifact = _save_active_mirt_artifact(simulado_id="sim_mirt_b", area="CH")

        loaded = resolve_mirt_artifact(
            ano_referencia=2024,
            area="CH",
            scope_id="sim_mirt_b",
        )

        assert loaded["manifest"]["artifact_id"] == artifact["artifact_id"]
        assert loaded["manifest"]["scope_id"] == "sim_mirt_b"

    @pytest.mark.unit
    def test_resolve_unique_model_without_scope(self, temp_model_storage):
        artifact = _save_active_mirt_artifact(simulado_id="sim_mirt_c", area="CN")

        loaded = resolve_mirt_artifact(
            ano_referencia=2024,
            area="CN",
        )

        assert loaded["manifest"]["artifact_id"] == artifact["artifact_id"]

    @pytest.mark.unit
    def test_raise_when_multiple_models_active(self, temp_model_storage):
        _save_active_mirt_artifact(simulado_id="sim_mirt_1", area="LC", artifact_id="job_lc_1")
        _save_active_mirt_artifact(simulado_id="sim_mirt_2", area="LC", artifact_id="job_lc_2")

        with pytest.raises(RuntimeMIRTArtifactAmbiguousError):
            resolve_mirt_artifact(
                ano_referencia=2024,
                area="LC",
            )

    @pytest.mark.unit
    def test_raise_when_no_model_exists(self, temp_model_storage):
        with pytest.raises(RuntimeMIRTArtifactNotFoundError):
            resolve_mirt_artifact(
                ano_referencia=2099,
                area="MT",
            )
