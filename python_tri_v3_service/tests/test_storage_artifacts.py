"""Tests for local artifact persistence and active-model registry."""

from pathlib import Path
from unittest import mock

import pytest

from app.runtime.local_paths import describe_runtime_layout, ensure_app_support_layout
from app.storage.artifacts import (
    artifact_dir_for,
    load_unidim_artifact,
    load_unidim_artifact_by_id,
    save_unidim_artifact,
)
from app.storage.model_registry import get_active_model, list_active_models, set_active_model
from app.storage.model_registry import (
    infer_model_family,
    list_active_models_by_family,
    list_active_models_by_scope,
)


class TestUnidimArtifacts:
    @pytest.fixture
    def temp_storage_root(self, tmp_path):
        temp_root = tmp_path / "app_support"
        with mock.patch("app.runtime.local_paths.APP_SUPPORT_ROOT", temp_root), mock.patch(
            "app.runtime.local_paths.MODELS_ROOT", temp_root / "models"
        ), mock.patch(
            "app.runtime.local_paths.UNIDIM_MODELS_ROOT", temp_root / "models" / "unidim"
        ), mock.patch(
            "app.runtime.local_paths.MICRODATA_CACHE_ROOT", temp_root / "cache" / "microdata"
        ), mock.patch(
            "app.runtime.local_paths.JOBS_ROOT", temp_root / "jobs"
        ), mock.patch(
            "app.runtime.local_paths.LOGS_ROOT", temp_root / "logs"
        ), mock.patch(
            "app.runtime.local_paths.LEGACY_MICRODATA_CACHE_ROOT", temp_root / "microdata_cache"
        ), mock.patch("app.storage.artifacts.APP_SUPPORT_ROOT", temp_root), mock.patch(
            "app.storage.artifacts.MODELS_ROOT", temp_root / "models"
        ), mock.patch(
            "app.storage.artifacts.UNIDIM_MODELS_ROOT", temp_root / "models" / "unidim"
        ), mock.patch(
            "app.storage.model_registry.APP_SUPPORT_ROOT", temp_root
        ), mock.patch(
            "app.storage.model_registry.MODEL_REGISTRY_PATH",
            temp_root / "models" / "active_models.json",
        ):
            yield temp_root

    @pytest.mark.unit
    def test_save_and_load_round_trip(self, temp_storage_root):
        manifest = save_unidim_artifact(
            simulado_id="simulado_teste",
            area="MT",
            ano_referencia=2024,
            c_fixo=0.20,
            parametros_itens=[{"a": 1.2, "b": 0.1, "c": 0.2, "taxa_acerto": 0.55, "saturado": False}],
            transformacao={"mu": 500.0, "sigma": 100.0, "metodo": "regressao_linear_lote"},
            diagnostico={"n_alunos": 50, "n_itens": 45},
            n_alunos=50,
            n_itens=45,
        )

        artifact_dir = Path(manifest["artifact_dir"])
        assert artifact_dir.is_dir()
        assert (artifact_dir / "manifest.json").is_file()
        assert (artifact_dir / "item_params.json").is_file()
        assert (artifact_dir / "transformacao.json").is_file()
        assert (artifact_dir / "diagnostico.json").is_file()

        loaded = load_unidim_artifact(artifact_dir)
        assert loaded["manifest"]["simulado_id"] == "simulado_teste"
        assert loaded["manifest"]["area"] == "MT"
        assert loaded["transformacao"]["sigma"] == 100.0
        assert loaded["parametros_itens"][0]["a"] == 1.2

    @pytest.mark.unit
    def test_registry_tracks_active_model(self, temp_storage_root):
        manifest = save_unidim_artifact(
            simulado_id="simulado_teste",
            area="CN",
            ano_referencia=2024,
            c_fixo=0.20,
            parametros_itens=[{"a": 1.0, "b": -0.2, "c": 0.2, "taxa_acerto": 0.48, "saturado": False}],
            transformacao={"mu": 510.0, "sigma": 97.5, "metodo": "regressao_linear_lote"},
            diagnostico={"n_alunos": 80, "n_itens": 45},
            n_alunos=80,
            n_itens=45,
        )

        active = set_active_model(manifest)

        assert active["artifact_id"] == manifest["artifact_id"]
        assert active["area"] == "CN"
        assert active["simulado_id"] == "simulado_teste"

        lookup = get_active_model(
            ano_referencia=2024,
            area="CN",
            scope_id="simulado_teste",
        )
        assert lookup is not None
        assert lookup["artifact_dir"] == manifest["artifact_dir"]

        listed = list_active_models()
        assert len(listed) == 1

    @pytest.mark.unit
    def test_load_by_artifact_id(self, temp_storage_root):
        manifest = save_unidim_artifact(
            simulado_id="simulado_teste",
            area="CH",
            ano_referencia=2024,
            c_fixo=0.20,
            parametros_itens=[{"a": 1.1, "b": 0.3, "c": 0.2, "taxa_acerto": 0.51, "saturado": False}],
            transformacao={"mu": 505.0, "sigma": 99.0, "metodo": "regressao_linear_lote"},
            diagnostico={"n_alunos": 64, "n_itens": 45},
            n_alunos=64,
            n_itens=45,
        )

        loaded = load_unidim_artifact_by_id(
            ano_referencia=2024,
            area="CH",
            artifact_id=manifest["artifact_id"],
        )

        assert loaded is not None
        assert loaded["manifest"]["artifact_id"] == manifest["artifact_id"]
        assert loaded["transformacao"]["mu"] == 505.0

    @pytest.mark.unit
    def test_registry_infers_model_family(self, temp_storage_root):
        assert infer_model_family("unidim_mml_eap_v1") == "unidim"
        assert infer_model_family("mirt_bifactor_mt_v1") == "mirt"
        assert infer_model_family("outro_modelo") == "unknown"

    @pytest.mark.unit
    def test_registry_lists_models_by_family_and_scope(self, temp_storage_root):
        unidim_manifest = save_unidim_artifact(
            simulado_id="simulado_teste",
            area="MT",
            ano_referencia=2024,
            c_fixo=0.20,
            parametros_itens=[{"a": 1.0, "b": -0.1, "c": 0.2, "taxa_acerto": 0.5, "saturado": False}],
            transformacao={"mu": 500.0, "sigma": 100.0, "metodo": "regressao_linear_lote"},
            diagnostico={"n_alunos": 60, "n_itens": 45},
            n_alunos=60,
            n_itens=45,
        )
        set_active_model(unidim_manifest)
        set_active_model(
            {
                "artifact_id": "job_mirt_001",
                "artifact_dir": str(temp_storage_root / "models" / "mirt" / "2024" / "MT" / "mirt_bifactor_mt_v1" / "job_mirt_001"),
                "simulado_id": "simulado_teste",
                "scope_type": "simulado_id",
                "scope_id": "simulado_teste",
                "ano_referencia": 2024,
                "area": "MT",
                "modelo": "mirt_bifactor_mt_v1",
                "saved_at_utc": "2026-04-08T12:00:00+00:00",
            },
            modelo="mirt_bifactor_mt_v1",
        )

        family_models = list_active_models_by_family(
            family="mirt",
            ano_referencia=2024,
            area="MT",
        )
        scope_models = list_active_models_by_scope(
            ano_referencia=2024,
            area="MT",
            scope_id="simulado_teste",
            family="mirt",
        )

        assert len(family_models) == 1
        assert family_models[0]["modelo"] == "mirt_bifactor_mt_v1"
        assert len(scope_models) == 1
        assert scope_models[0]["artifact_id"] == "job_mirt_001"

    @pytest.mark.unit
    def test_artifact_dir_is_versioned_by_scope(self, temp_storage_root):
        artifact_dir = artifact_dir_for(
            ano_referencia=2024,
            area="LC",
            scope_id="simulado 1/a",
        )
        path_str = str(artifact_dir)
        assert "/2024/" in path_str
        assert "/LC/" in path_str
        assert "/simulado-1-a/" in path_str

    @pytest.mark.unit
    def test_runtime_layout_cria_arvore_canonica(self, temp_storage_root):
        paths = ensure_app_support_layout()

        assert paths["app_support"].is_dir()
        assert paths["models"].is_dir()
        assert paths["models_unidim"].is_dir()
        assert paths["cache_microdata"].is_dir()
        assert paths["jobs"].is_dir()
        assert paths["logs"].is_dir()

    @pytest.mark.unit
    def test_runtime_layout_descreve_paths(self, temp_storage_root):
        snapshot = describe_runtime_layout()

        assert snapshot["app_support"]["exists"] is True
        assert snapshot["models"]["exists"] is True
        assert snapshot["cache_microdata"]["exists"] is True
        assert "writable" in snapshot["models"]
        assert snapshot["cache_microdata_legacy"]["path"].endswith("microdata_cache")
