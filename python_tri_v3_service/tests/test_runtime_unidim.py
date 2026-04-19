"""Tests for the unidim runtime artifact resolver."""

from __future__ import annotations

import pytest

from app.runtime.unidim import (
    RuntimeArtifactAmbiguousError,
    RuntimeArtifactNotFoundError,
    resolve_unidim_artifact,
)
from app.storage.artifacts import save_unidim_artifact
from app.storage.model_registry import set_active_model


def _save_active_artifact(*, simulado_id: str, area: str, ano_referencia: int = 2024):
    manifest = save_unidim_artifact(
        simulado_id=simulado_id,
        area=area,
        ano_referencia=ano_referencia,
        c_fixo=0.20,
        parametros_itens=[{"a": 1.1, "b": 0.2, "c": 0.2, "taxa_acerto": 0.54, "saturado": False}],
        transformacao={"mu": 500.0, "sigma": 100.0, "metodo": "regressao_linear_lote"},
        diagnostico={"n_alunos": 50, "n_itens": 45},
        n_alunos=50,
        n_itens=45,
    )
    set_active_model(manifest)
    return manifest


class TestResolveUnidimArtifact:
    @pytest.mark.unit
    def test_resolve_by_artifact_id(self, temp_model_storage):
        manifest = _save_active_artifact(simulado_id="sim_a", area="MT")

        artifact = resolve_unidim_artifact(
            ano_referencia=2024,
            area="MT",
            artifact_id=manifest["artifact_id"],
        )

        assert artifact["manifest"]["artifact_id"] == manifest["artifact_id"]

    @pytest.mark.unit
    def test_resolve_by_scope_id(self, temp_model_storage):
        manifest = _save_active_artifact(simulado_id="sim_b", area="CH")

        artifact = resolve_unidim_artifact(
            ano_referencia=2024,
            area="CH",
            scope_id="sim_b",
        )

        assert artifact["manifest"]["artifact_id"] == manifest["artifact_id"]

    @pytest.mark.unit
    def test_resolve_unique_model_without_scope(self, temp_model_storage):
        manifest = _save_active_artifact(simulado_id="sim_c", area="CN")

        artifact = resolve_unidim_artifact(
            ano_referencia=2024,
            area="CN",
        )

        assert artifact["manifest"]["artifact_id"] == manifest["artifact_id"]

    @pytest.mark.unit
    def test_raise_when_multiple_models_active(self, temp_model_storage):
        _save_active_artifact(simulado_id="sim_1", area="LC")
        _save_active_artifact(simulado_id="sim_2", area="LC")

        with pytest.raises(RuntimeArtifactAmbiguousError):
            resolve_unidim_artifact(
                ano_referencia=2024,
                area="LC",
            )

    @pytest.mark.unit
    def test_raise_when_no_model_exists(self, temp_model_storage):
        with pytest.raises(RuntimeArtifactNotFoundError):
            resolve_unidim_artifact(
                ano_referencia=2099,
                area="MT",
            )
