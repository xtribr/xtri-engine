"""API endpoint tests (regression baseline for V2/V3 routes + new frontend endpoints)."""

import io
import json
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from app.mirt.qmatrix import build_habilidade_qmatrix, build_model_spec_from_qmatrix
from app.mirt.runner import MIRTWorkerExecutionError, build_mirt_output_dir
from app.microdata.jobs import job_manager
from app.storage.model_registry import set_active_model
from app.tabela_referencia import TabelaReferenciaTRI


def _save_active_mirt_artifact(
    *,
    simulado_id: str,
    area: str,
    ano_referencia: int,
    n_items: int,
):
    item_rows = [
        {
            "item_id": f"{area.lower()}_{idx:03d}",
            "co_item": 10_000 + idx,
            "numero_habilidade": ((idx - 1) % 3) + 1,
            "posicao": idx,
        }
        for idx in range(1, n_items + 1)
    ]
    qmatrix = build_habilidade_qmatrix(area=area, item_rows=item_rows)
    model = build_model_spec_from_qmatrix(qmatrix)
    artifact_id = f"job_{simulado_id}_{ano_referencia}"
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
                "n_students": 50,
                "n_items": n_items,
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
        "factor_ids": [factor.factor_id for factor in model.factors],
        "model_id": model.model_id,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Health & Debug
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoints:

    @pytest.mark.api
    def test_health(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "online"
        assert "heuristico_v2" in body["motores"]
        assert "cientifico_mirt" in body["motores"]

    @pytest.mark.api
    def test_debug(self, client: TestClient):
        resp = client.get("/api/debug")
        assert resp.status_code == 200
        body = resp.json()
        assert body["version"] == "3.0.0"


# ═══════════════════════════════════════════════════════════════════════════════
# V2 Heuristic Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalcularTRI:

    @pytest.mark.api
    def test_payload_valido(self, client: TestClient, alunos_v2_30, gabarito_45):
        resp = client.post("/api/calcular-tri", json={
            "alunos": alunos_v2_30,
            "gabarito": gabarito_45,
            "areas_config": {"MT": [1, 45]},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "sucesso"
        assert body["total_alunos"] == 30
        assert len(body["resultados"]) == 30

    @pytest.mark.api
    def test_payload_sem_alunos(self, client: TestClient):
        resp = client.post("/api/calcular-tri", json={
            "gabarito": {"1": "A"},
        })
        assert resp.status_code == 400

    @pytest.mark.api
    def test_payload_vazio(self, client: TestClient):
        resp = client.post("/api/calcular-tri", json={})
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════════
# V3 Scientific Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibrarCientifico:

    @pytest.mark.api
    def test_payload_valido(self, client: TestClient, respostas_binarias_50x45):
        alunos = [
            {"aluno_id": str(i), "nome": f"Aluno {i}", "respostas": r}
            for i, r in enumerate(respostas_binarias_50x45)
        ]
        resp = client.post("/api/v3/calibrar-cientifico", json={
            "simulado_id": "test_sim",
            "area": "MT",
            "alunos": alunos,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "sucesso"
        assert body["n_alunos"] == 50
        assert body["n_itens"] == 45
        assert body["transformacao"] is not None
        assert body["artefato_id"] is not None

    @pytest.mark.api
    def test_persiste_artefato_local_no_diagnostico(
        self,
        client: TestClient,
        respostas_binarias_50x45,
        tmp_path,
    ):
        alunos = [
            {"aluno_id": str(i), "nome": f"Aluno {i}", "respostas": r}
            for i, r in enumerate(respostas_binarias_50x45)
        ]
        app_support = tmp_path / "app_support"
        with mock.patch("app.storage.artifacts.APP_SUPPORT_ROOT", app_support), mock.patch(
            "app.storage.artifacts.MODELS_ROOT", app_support / "models"
        ), mock.patch(
            "app.storage.artifacts.UNIDIM_MODELS_ROOT", app_support / "models" / "unidim"
        ), mock.patch(
            "app.storage.model_registry.APP_SUPPORT_ROOT", app_support
        ), mock.patch(
            "app.storage.model_registry.MODEL_REGISTRY_PATH",
            app_support / "models" / "active_models.json",
        ):
            resp = client.post("/api/v3/calibrar-cientifico", json={
                "simulado_id": "test_sim",
                "area": "MT",
                "alunos": alunos,
            })

        assert resp.status_code == 200
        body = resp.json()
        assert body["diagnostico"]["artefato_persistido"] is True
        assert body["diagnostico"]["artefato_id"]
        assert body["diagnostico"]["artefato_dir"]
        assert (app_support / "models" / "active_models.json").is_file()

    @pytest.mark.api
    def test_area_invalida(self, client: TestClient, respostas_binarias_50x45):
        alunos = [
            {"aluno_id": str(i), "respostas": r}
            for i, r in enumerate(respostas_binarias_50x45)
        ]
        resp = client.post("/api/v3/calibrar-cientifico", json={
            "simulado_id": "test_sim",
            "area": "XX",
            "alunos": alunos,
        })
        assert resp.status_code == 400

    @pytest.mark.api
    def test_menos_de_10_alunos(self, client: TestClient):
        alunos = [
            {"aluno_id": str(i), "respostas": [1, 0, 1] * 5}
            for i in range(5)
        ]
        resp = client.post("/api/v3/calibrar-cientifico", json={
            "simulado_id": "test_sim",
            "area": "MT",
            "alunos": alunos,
        })
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════════
# V3 Individual Theta Estimation
# ═══════════════════════════════════════════════════════════════════════════════

class TestEstimarTheta:

    @pytest.mark.api
    def test_payload_valido(self, client: TestClient):
        resp = client.post("/api/v3/estimar-theta", json={
            "area": "MT",
            "parametros_itens": [{"a": 1.0, "b": 0.0, "c": 0.20}] * 10,
            "respostas": [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "theta" in body
        assert "nota_ancorada" in body
        assert "estado_tri" in body

    @pytest.mark.api
    def test_respostas_e_itens_tamanho_diferente(self, client: TestClient):
        resp = client.post("/api/v3/estimar-theta", json={
            "area": "MT",
            "parametros_itens": [{"a": 1.0, "b": 0.0, "c": 0.20}] * 10,
            "respostas": [1, 0, 1],  # 3 != 10
        })
        assert resp.status_code == 400

    @pytest.mark.api
    def test_transformacao_opcional_afeta_nota(self, client: TestClient):
        params = (
            [{"a": 1.2, "b": -2.0, "c": 0.20}] * 2
            + [{"a": 1.2, "b": -1.0, "c": 0.20}] * 2
            + [{"a": 1.2, "b": 0.5, "c": 0.20}] * 2
            + [{"a": 1.2, "b": 1.5, "c": 0.20}] * 2
        )
        payload_base = {
            "area": "MT",
            "parametros_itens": params,
            "transformacao": {"mu": 500.0, "sigma": 100.0},
        }

        resp_faceis = client.post(
            "/api/v3/estimar-theta",
            json={**payload_base, "respostas": [1, 1, 1, 1, 0, 0, 0, 0]},
        )
        resp_dificeis = client.post(
            "/api/v3/estimar-theta",
            json={**payload_base, "respostas": [0, 0, 0, 0, 1, 1, 1, 1]},
        )

        assert resp_faceis.status_code == 200
        assert resp_dificeis.status_code == 200
        nota_1 = resp_faceis.json()["nota_ancorada"]
        nota_2 = resp_dificeis.json()["nota_ancorada"]
        theta_1 = resp_faceis.json()["theta"]
        theta_2 = resp_dificeis.json()["theta"]

        assert nota_1 != nota_2
        assert (theta_1 > theta_2) == (nota_1 > nota_2)

    @pytest.mark.api
    def test_sem_transformacao_nao_fixa_nota_na_mediana(self, client: TestClient):
        params = (
            [{"a": 1.2, "b": -2.0, "c": 0.20}] * 2
            + [{"a": 1.2, "b": -1.0, "c": 0.20}] * 2
            + [{"a": 1.2, "b": 0.5, "c": 0.20}] * 2
            + [{"a": 1.2, "b": 1.5, "c": 0.20}] * 2
        )
        resp_faceis = client.post(
            "/api/v3/estimar-theta",
            json={
                "area": "MT",
                "parametros_itens": params,
                "respostas": [1, 1, 1, 1, 0, 0, 0, 0],
            },
        )
        resp_dificeis = client.post(
            "/api/v3/estimar-theta",
            json={
                "area": "MT",
                "parametros_itens": params,
                "respostas": [0, 0, 0, 0, 1, 1, 1, 1],
            },
        )

        assert resp_faceis.status_code == 200
        assert resp_dificeis.status_code == 200

        body_1 = resp_faceis.json()
        body_2 = resp_dificeis.json()
        ref = TabelaReferenciaTRI().obter("MT", 4)
        margem = (ref["tri_max"] - ref["tri_min"]) * 0.03 + 0.1

        assert body_1["acertos"] == body_2["acertos"] == 4
        assert body_1["theta"] != body_2["theta"]
        assert body_1["nota_ancorada"] != body_2["nota_ancorada"]
        assert (body_1["theta"] > body_2["theta"]) == (body_1["nota_ancorada"] > body_2["nota_ancorada"])
        for body in (body_1, body_2):
            assert ref["tri_min"] - margem <= body["nota_ancorada"] <= ref["tri_max"] + margem

    @pytest.mark.api
    def test_reutiliza_modelo_local_por_simulado(
        self,
        client: TestClient,
        respostas_binarias_50x45,
    ):
        alunos = [
            {"aluno_id": str(i), "nome": f"Aluno {i}", "respostas": r}
            for i, r in enumerate(respostas_binarias_50x45)
        ]
        calib = client.post("/api/v3/calibrar-cientifico", json={
            "simulado_id": "simulado_reuso",
            "area": "MT",
            "alunos": alunos,
        })
        assert calib.status_code == 200

        resp = client.post("/api/v3/estimar-theta", json={
            "area": "MT",
            "ano_referencia": 2024,
            "simulado_id": "simulado_reuso",
            "respostas": respostas_binarias_50x45[0],
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["artefato_id"] == calib.json()["artefato_id"]
        assert "theta" in body
        assert "nota_ancorada" in body

    @pytest.mark.api
    def test_reutiliza_modelo_local_por_artefato_id(
        self,
        client: TestClient,
        respostas_binarias_50x45,
    ):
        alunos = [
            {"aluno_id": str(i), "nome": f"Aluno {i}", "respostas": r}
            for i, r in enumerate(respostas_binarias_50x45)
        ]
        calib = client.post("/api/v3/calibrar-cientifico", json={
            "simulado_id": "simulado_reuso_id",
            "area": "MT",
            "alunos": alunos,
        })
        assert calib.status_code == 200
        artefato_id = calib.json()["artefato_id"]

        resp = client.post("/api/v3/estimar-theta", json={
            "area": "MT",
            "ano_referencia": 2024,
            "artefato_id": artefato_id,
            "respostas": respostas_binarias_50x45[1],
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["artefato_id"] == artefato_id
        assert "theta" in body

    @pytest.mark.api
    def test_reutiliza_unico_modelo_local_sem_simulado(
        self,
        client: TestClient,
        respostas_binarias_50x45,
    ):
        alunos = [
            {"aluno_id": str(i), "nome": f"Aluno {i}", "respostas": r}
            for i, r in enumerate(respostas_binarias_50x45)
        ]
        calib = client.post("/api/v3/calibrar-cientifico", json={
            "simulado_id": "simulado_unico_area",
            "area": "CN",
            "ano_referencia": 2098,
            "alunos": alunos,
        })
        assert calib.status_code == 200
        artefato_id = calib.json()["artefato_id"]

        resp = client.post("/api/v3/estimar-theta", json={
            "area": "CN",
            "ano_referencia": 2098,
            "respostas": respostas_binarias_50x45[2],
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["artefato_id"] == artefato_id
        assert "theta" in body

    @pytest.mark.api
    def test_falha_quando_area_tem_modelos_ativos_ambiguous(
        self,
        client: TestClient,
        respostas_binarias_50x45,
    ):
        alunos = [
            {"aluno_id": str(i), "nome": f"Aluno {i}", "respostas": r}
            for i, r in enumerate(respostas_binarias_50x45)
        ]
        for simulado_id in ("simulado_lc_a", "simulado_lc_b"):
            calib = client.post("/api/v3/calibrar-cientifico", json={
                "simulado_id": simulado_id,
                "area": "LC",
                "alunos": alunos,
            })
            assert calib.status_code == 200

        resp = client.post("/api/v3/estimar-theta", json={
            "area": "LC",
            "ano_referencia": 2024,
            "respostas": respostas_binarias_50x45[0],
        })

        assert resp.status_code == 400
        assert "Informe `simulado_id` ou `artefato_id`" in resp.json()["detail"]

    @pytest.mark.api
    def test_exige_parametros_ou_modelo_local(self, client: TestClient):
        resp = client.post("/api/v3/estimar-theta", json={
            "area": "MT",
            "ano_referencia": 2100,
            "respostas": [1, 0, 1, 0],
        })

        assert resp.status_code == 400
        assert "parametros_itens" in resp.json()["detail"]


class TestEstimarThetaMIRT:

    @pytest.mark.api
    def test_usa_artefato_mirt_e_retorna_scores(
        self,
        client: TestClient,
        respostas_binarias_50x45,
    ):
        simulado_id = "simulado_mirt_ok"
        area = "MT"
        ano = 2091
        artifact = _save_active_mirt_artifact(
            simulado_id=simulado_id,
            area=area,
            ano_referencia=ano,
            n_items=len(respostas_binarias_50x45[0]),
        )

        def _fake_run_mirt_score(*, output_path, **kwargs):
            Path(output_path).write_text(
                (
                    "student_id,G,MT_H01,MT_H02,MT_H03,SE_G,SE_MT_H01,SE_MT_H02,SE_MT_H03\n"
                    "student_1,0.82,0.55,0.44,0.31,0.11,0.20,0.21,0.22\n"
                ),
                encoding="utf-8",
            )
            return output_path

        with mock.patch("app.main.mirt_runtime_available", return_value=True), mock.patch(
            "app.main.run_mirt_score",
            side_effect=_fake_run_mirt_score,
        ):
            resp = client.post("/api/v5/mirt/estimar-theta", json={
                "area": area,
                "ano_referencia": ano,
                "simulado_id": simulado_id,
                "transformacao": {"mu": 500.0, "sigma": 100.0},
                "respostas": respostas_binarias_50x45[0],
            })

        assert resp.status_code == 200
        body = resp.json()
        assert body["modelo_usado"] == "mirt"
        assert body["fallback_modelo"] is None
        assert body["artefato_id"] == artifact["artifact_id"]
        assert body["theta"] == 0.82
        assert body["se"] == 0.11
        assert body["scores_mirt"]["G"] == 0.82
        assert body["se_mirt"]["G"] == 0.11

    @pytest.mark.api
    def test_cai_para_unidim_quando_nao_ha_artefato_mirt(
        self,
        client: TestClient,
        respostas_binarias_50x45,
    ):
        simulado_id = "simulado_sem_mirt"
        ano = 2092
        alunos = [
            {"aluno_id": str(i), "nome": f"Aluno {i}", "respostas": r}
            for i, r in enumerate(respostas_binarias_50x45)
        ]
        calib = client.post("/api/v3/calibrar-cientifico", json={
            "simulado_id": simulado_id,
            "area": "MT",
            "ano_referencia": ano,
            "alunos": alunos,
        })
        assert calib.status_code == 200

        resp = client.post("/api/v5/mirt/estimar-theta", json={
            "area": "MT",
            "ano_referencia": ano,
            "simulado_id": simulado_id,
            "respostas": respostas_binarias_50x45[0],
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["modelo_usado"] == "unidim"
        assert body["fallback_modelo"] == "mirt"
        assert body["scores_mirt"] is None

    @pytest.mark.api
    def test_cai_para_unidim_quando_rscript_indisponivel(
        self,
        client: TestClient,
        respostas_binarias_50x45,
    ):
        simulado_id = "simulado_mirt_sem_rscript"
        ano = 2093
        alunos = [
            {"aluno_id": str(i), "nome": f"Aluno {i}", "respostas": r}
            for i, r in enumerate(respostas_binarias_50x45)
        ]
        calib = client.post("/api/v3/calibrar-cientifico", json={
            "simulado_id": simulado_id,
            "area": "MT",
            "ano_referencia": ano,
            "alunos": alunos,
        })
        assert calib.status_code == 200
        _save_active_mirt_artifact(
            simulado_id=simulado_id,
            area="MT",
            ano_referencia=ano,
            n_items=len(respostas_binarias_50x45[0]),
        )

        with mock.patch("app.main.mirt_runtime_available", return_value=False):
            resp = client.post("/api/v5/mirt/estimar-theta", json={
                "area": "MT",
                "ano_referencia": ano,
                "simulado_id": simulado_id,
                "respostas": respostas_binarias_50x45[1],
            })

        assert resp.status_code == 200
        body = resp.json()
        assert body["modelo_usado"] == "unidim"
        assert body["fallback_modelo"] == "mirt"

    @pytest.mark.api
    def test_cai_para_unidim_quando_scoring_mirt_falha(
        self,
        client: TestClient,
        respostas_binarias_50x45,
    ):
        simulado_id = "simulado_mirt_falha"
        ano = 2094
        alunos = [
            {"aluno_id": str(i), "nome": f"Aluno {i}", "respostas": r}
            for i, r in enumerate(respostas_binarias_50x45)
        ]
        calib = client.post("/api/v3/calibrar-cientifico", json={
            "simulado_id": simulado_id,
            "area": "MT",
            "ano_referencia": ano,
            "alunos": alunos,
        })
        assert calib.status_code == 200
        _save_active_mirt_artifact(
            simulado_id=simulado_id,
            area="MT",
            ano_referencia=ano,
            n_items=len(respostas_binarias_50x45[0]),
        )

        with mock.patch("app.main.mirt_runtime_available", return_value=True), mock.patch(
            "app.main.run_mirt_score",
            side_effect=MIRTWorkerExecutionError("falha no score"),
        ):
            resp = client.post("/api/v5/mirt/estimar-theta", json={
                "area": "MT",
                "ano_referencia": ano,
                "simulado_id": simulado_id,
                "respostas": respostas_binarias_50x45[2],
            })

        assert resp.status_code == 200
        body = resp.json()
        assert body["modelo_usado"] == "unidim"
        assert body["fallback_modelo"] == "mirt"

    @pytest.mark.api
    def test_retorna_erro_quando_fallback_esta_desligado(
        self,
        client: TestClient,
        respostas_binarias_50x45,
    ):
        resp = client.post("/api/v5/mirt/estimar-theta", json={
            "area": "MT",
            "ano_referencia": 2095,
            "simulado_id": "simulado_sem_fallback",
            "permitir_fallback_unidim": False,
            "respostas": respostas_binarias_50x45[0],
        })

        assert resp.status_code == 400
        assert "Nenhum modelo MIRT ativo" in resp.json()["detail"]

    @pytest.mark.api
    def test_retorna_erro_quando_respostas_divergem_da_qmatrix(
        self,
        client: TestClient,
    ):
        _save_active_mirt_artifact(
            simulado_id="simulado_mirt_tamanho",
            area="MT",
            ano_referencia=2096,
            n_items=45,
        )

        resp = client.post("/api/v5/mirt/estimar-theta", json={
            "area": "MT",
            "ano_referencia": 2096,
            "simulado_id": "simulado_mirt_tamanho",
            "respostas": [1, 0, 1],
        })

        assert resp.status_code == 400
        assert "Q-matrix" in resp.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════════════
# V3 Reference Table Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestReferencias:

    @pytest.mark.api
    def test_area_valida(self, client: TestClient):
        resp = client.get("/api/v3/referencias/MT")
        assert resp.status_code == 200
        body = resp.json()
        assert body["area"] == "MT"
        assert body["total_registros"] == 46
        assert len(body["dados"]) == 46

    @pytest.mark.api
    def test_area_invalida(self, client: TestClient):
        resp = client.get("/api/v3/referencias/XX")
        assert resp.status_code == 404

    @pytest.mark.api
    def test_area_case_insensitive(self, client: TestClient):
        resp = client.get("/api/v3/referencias/mt")
        assert resp.status_code == 200
        assert resp.json()["area"] == "MT"


# ═══════════════════════════════════════════════════════════════════════════════
# V5 Runtime / Desktop Status
# ═══════════════════════════════════════════════════════════════════════════════

class TestRuntimeDesktopStatus:

    @pytest.mark.api
    def test_runtime_status_retorna_layout(self, client: TestClient, temp_model_storage):
        resp = client.get("/api/v5/runtime/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["app_support_dir"] == str(temp_model_storage)
        assert "app_support" in body["directories"]
        assert body["directories"]["models"]["exists"] is True
        assert "writable" in body["directories"]["models"]

    @pytest.mark.api
    def test_runtime_status_lista_modelo_pronto(
        self,
        client: TestClient,
        respostas_binarias_50x45,
    ):
        alunos = [
            {"aluno_id": str(i), "nome": f"Aluno {i}", "respostas": r}
            for i, r in enumerate(respostas_binarias_50x45)
        ]
        simulado_id = "simulado_runtime_status"
        calib = client.post("/api/v3/calibrar-cientifico", json={
            "simulado_id": simulado_id,
            "area": "MT",
            "alunos": alunos,
        })
        assert calib.status_code == 200

        resp = client.get(
            "/api/v5/runtime/status",
            params={"ano_referencia": 2024, "simulado_id": simulado_id},
        )

        assert resp.status_code == 200
        body = resp.json()
        model = next((m for m in body["models"] if m["scope_id"] == simulado_id and m["area"] == "MT"), None)
        assert model is not None
        assert model["status"] == "pronto"
        assert body["status_areas"]["MT"] == "pronto"

    @pytest.mark.api
    def test_runtime_status_cai_para_fallback_sem_modelo(self, client: TestClient):
        resp = client.get(
            "/api/v5/runtime/status",
            params={"ano_referencia": 2024, "simulado_id": "simulado_inexistente"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status_areas"]["MT"] == "fallback_heuristico"
        assert body["status_areas"]["LC"] == "fallback_heuristico"

    @pytest.mark.api
    def test_runtime_status_lista_jobs(self, client: TestClient):
        job_manager.clear()
        job = job_manager.create(year=2024, sample_size=200_000, rng_seed=42)
        job_manager.mark_running(job.job_id)

        resp = client.get("/api/v5/runtime/status")

        job_manager.clear()

        assert resp.status_code == 200
        body = resp.json()
        runtime_job = next((j for j in body["jobs"] if j["job_id"] == job.job_id), None)
        assert runtime_job is not None
        assert runtime_job["status"] == "running"
        assert body["job_counts"]["running"] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Frontend Endpoints — Years + CSV Processing
# ═══════════════════════════════════════════════════════════════════════════════

def _make_single_area_csv(n: int = 30, n_itens: int = 45) -> str:
    """Synthetic single-area CSV with binary responses."""
    import numpy as np
    rng = np.random.default_rng(42)
    headers = ["ID"] + [str(i) for i in range(1, n_itens + 1)]
    lines = [",".join(headers)]
    for i in range(n):
        row = [f"S{i:03d}"] + [str(int(rng.random() > 0.5)) for _ in range(n_itens)]
        lines.append(",".join(row))
    return "\n".join(lines)


def _make_area_file(
    n: int = 15,
    *,
    n_itens: int = 45,
    delimiter: str = ",",
    with_header: bool = True,
    with_id: bool = True,
    header_style: str = "numeric",  # "numeric" → "1,2,..."; "questao" → "Questao 1,..."
    as_xlsx: bool = False,
    seed: int = 42,
):
    """Build a synthetic response file in any of the supported shapes.

    Returns bytes (for XLSX) or str (for delimited text). The parser
    is expected to handle all (delimiter × header × id × format) combos,
    so this helper drives the format matrix.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    data_rows: list[list[str]] = []
    for i in range(n):
        respostas = [str(int(rng.random() > 0.5)) for _ in range(n_itens)]
        row = ([f"S{i:03d}"] if with_id else []) + respostas
        data_rows.append(row)

    if as_xlsx:
        import openpyxl
        from io import BytesIO
        wb = openpyxl.Workbook()
        ws = wb.active
        if with_header:
            header_cells: list = []
            if with_id:
                header_cells.append("ID")
            if header_style == "questao":
                header_cells.extend([f"Questao {i}" for i in range(1, n_itens + 1)])
            else:
                header_cells.extend(list(range(1, n_itens + 1)))
            ws.append(header_cells)
        for row in data_rows:
            ws.append([int(c) if c.isdigit() and (not with_id or idx > 0) else c
                       for idx, c in enumerate(row)])
        buf = BytesIO()
        wb.save(buf)
        return buf.getvalue()

    lines: list[str] = []
    if with_header:
        header_row: list[str] = []
        if with_id:
            header_row.append("ID")
        if header_style == "questao":
            header_row.extend([f"Questao {i}" for i in range(1, n_itens + 1)])
        else:
            header_row.extend([str(i) for i in range(1, n_itens + 1)])
        lines.append(delimiter.join(header_row))
    for row in data_rows:
        lines.append(delimiter.join(row))
    return "\n".join(lines)


class TestAnosReferencia:

    @pytest.mark.api
    def test_retorna_lista_de_anos(self, client: TestClient):
        resp = client.get("/api/v3/anos-referencia")
        assert resp.status_code == 200
        anos = resp.json()
        assert isinstance(anos, list)
        assert len(anos) >= 1

    @pytest.mark.api
    def test_anos_sao_inteiros_ordenados(self, client: TestClient):
        resp = client.get("/api/v3/anos-referencia")
        anos = resp.json()
        assert anos == sorted(anos)
        assert all(isinstance(a, int) for a in anos)


class TestProcessarCSV:

    @pytest.mark.api
    def test_upload_area_mt(self, client: TestClient):
        csv_content = _make_single_area_csv(30)
        anos = client.get("/api/v3/anos-referencia").json()
        ano = anos[-1]

        resp = client.post(
            "/api/v3/processar-csv",
            data={"area": "MT", "ano_referencia": str(ano)},
            files={"arquivo": ("mt.csv", csv_content, "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "sucesso"
        assert body["areas_processadas"] == ["MT"]
        assert body["n_alunos"] == 30
        r0 = body["resultados"][0]
        assert r0["nota_mt"] is not None
        assert r0["estado_mt"] is not None
        # Other areas should be null
        assert r0["nota_lc"] is None

    @pytest.mark.api
    def test_area_invalida_retorna_400(self, client: TestClient):
        csv_content = _make_single_area_csv(15)
        resp = client.post(
            "/api/v3/processar-csv",
            data={"area": "XX", "ano_referencia": "2022"},
            files={"arquivo": ("test.csv", csv_content, "text/csv")},
        )
        assert resp.status_code == 400

    @pytest.mark.api
    def test_ano_invalido_retorna_400(self, client: TestClient):
        csv_content = _make_single_area_csv(15)
        resp = client.post(
            "/api/v3/processar-csv",
            data={"area": "MT", "ano_referencia": "1999"},
            files={"arquivo": ("test.csv", csv_content, "text/csv")},
        )
        assert resp.status_code == 400

    @pytest.mark.api
    def test_csv_malformado_retorna_400(self, client: TestClient):
        csv_bad = "ID,Nome,Turma\nA,Joao,3A"
        resp = client.post(
            "/api/v3/processar-csv",
            data={"area": "MT", "ano_referencia": "2022"},
            files={"arquivo": ("bad.csv", csv_bad, "text/csv")},
        )
        assert resp.status_code == 400

    @pytest.mark.api
    def test_menos_de_10_alunos_retorna_400(self, client: TestClient):
        csv_content = _make_single_area_csv(5)
        resp = client.post(
            "/api/v3/processar-csv",
            data={"area": "MT", "ano_referencia": "2022"},
            files={"arquivo": ("small.csv", csv_content, "text/csv")},
        )
        assert resp.status_code == 400


class TestProcessarCSVFormatMatrix:
    """End-to-end HTTP coverage of the (delimiter × header × id × format) grid.

    The parser unit tests (``test_csv_parser.py``) cover individual parsing
    paths; this class confirms that each shape actually flows through the
    ``/api/v3/processar-csv`` handler and produces a calibrated response.
    Drives the brittleness concern from task_plan.md Phase 1-3 that was
    never closed out after Phase 4/5 superseded the original test plan.
    """

    @pytest.fixture(scope="class")
    def ano(self, client: TestClient) -> int:
        return client.get("/api/v3/anos-referencia").json()[-1]

    @pytest.mark.api
    @pytest.mark.parametrize(
        "delimiter,label",
        [(",", "csv-comma"), ("\t", "tsv-tab"), (";", "scsv-semicolon")],
    )
    def test_delimiter_variants(
        self, client: TestClient, ano: int, delimiter: str, label: str,
    ):
        """Comma, tab, and semicolon delimiters all reach a successful response."""
        content = _make_area_file(n=20, delimiter=delimiter, with_header=True, with_id=True)
        resp = client.post(
            "/api/v3/processar-csv",
            data={"area": "MT", "ano_referencia": str(ano)},
            files={"arquivo": (f"{label}.csv", content, "text/csv")},
        )
        assert resp.status_code == 200, f"{label}: {resp.text[:200]}"
        body = resp.json()
        assert body["n_alunos"] >= 18, f"{label}: only {body['n_alunos']}/20"
        assert body["resultados"][0]["aluno_id"] == "S000"

    @pytest.mark.api
    @pytest.mark.parametrize("with_header", [True, False])
    @pytest.mark.parametrize("with_id", [True, False])
    def test_header_and_id_combinations(
        self, client: TestClient, ano: int, with_header: bool, with_id: bool,
    ):
        """Headered/headerless × with-id/without-id all process successfully.

        Headerless detection reads the first row; without an ID column, the
        handler should synthesize ``aluno_N`` IDs.
        """
        content = _make_area_file(
            n=20, delimiter=",", with_header=with_header, with_id=with_id,
        )
        resp = client.post(
            "/api/v3/processar-csv",
            data={"area": "MT", "ano_referencia": str(ano)},
            files={"arquivo": ("test.csv", content, "text/csv")},
        )
        assert resp.status_code == 200, (
            f"header={with_header} id={with_id}: {resp.text[:200]}"
        )
        body = resp.json()
        # Calibration can drop students with degenerate response patterns
        # (all-0/all-1 or unresolved theta); 90% survival is the floor.
        assert body["n_alunos"] >= 18, (
            f"header={with_header} id={with_id}: only {body['n_alunos']}/20"
        )
        first_id = body["resultados"][0]["aluno_id"]
        if with_id:
            assert first_id.startswith("S")
        else:
            assert first_id.startswith("aluno_")

    @pytest.mark.api
    def test_questao_style_header(self, client: TestClient, ano: int):
        """'Questao 1', 'Questao 2', ... headers parse as question columns."""
        content = _make_area_file(
            n=20, delimiter=",", with_header=True, with_id=True, header_style="questao",
        )
        resp = client.post(
            "/api/v3/processar-csv",
            data={"area": "MT", "ano_referencia": str(ano)},
            files={"arquivo": ("questao.csv", content, "text/csv")},
        )
        assert resp.status_code == 200, resp.text[:200]
        assert resp.json()["n_alunos"] >= 18

    @pytest.mark.api
    def test_xlsx_round_trip(self, client: TestClient, ano: int):
        """XLSX uploads go through openpyxl and produce a calibrated response."""
        content = _make_area_file(n=20, with_header=True, with_id=True, as_xlsx=True)
        resp = client.post(
            "/api/v3/processar-csv",
            data={"area": "MT", "ano_referencia": str(ano)},
            files={
                "arquivo": (
                    "test.xlsx",
                    content,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ),
            },
        )
        assert resp.status_code == 200, resp.text[:200]
        assert resp.json()["n_alunos"] >= 18

    @pytest.mark.api
    def test_utf8_bom_is_tolerated(self, client: TestClient, ano: int):
        """UTF-8 with BOM (common Excel export) decodes correctly."""
        content = _make_area_file(n=20, delimiter=",", with_header=True, with_id=True)
        bom_bytes = b"\xef\xbb\xbf" + content.encode("utf-8")
        resp = client.post(
            "/api/v3/processar-csv",
            data={"area": "MT", "ano_referencia": str(ano)},
            files={"arquivo": ("bom.csv", bom_bytes, "text/csv")},
        )
        assert resp.status_code == 200, resp.text[:200]
        assert resp.json()["n_alunos"] >= 18

    @pytest.mark.api
    def test_latin1_encoding_is_tolerated(self, client: TestClient, ano: int):
        """Latin-1 fallback for files that aren't valid UTF-8."""
        content = _make_area_file(n=20, delimiter=",", with_header=True, with_id=True)
        # Replace a student id with a non-UTF8 byte sequence (Latin-1 only)
        mangled = content.replace("S003", "A\xe7\xe3o").encode("latin-1")
        resp = client.post(
            "/api/v3/processar-csv",
            data={"area": "MT", "ano_referencia": str(ano)},
            files={"arquivo": ("latin1.csv", mangled, "text/csv")},
        )
        assert resp.status_code == 200, resp.text[:200]
        assert resp.json()["n_alunos"] >= 18


class TestFrontendStatic:

    @pytest.mark.api
    def test_root_retorna_html(self, client: TestClient):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "XTRI Engine" in resp.text


# ═══════════════════════════════════════════════════════════════════════════════
# V4 endpoints — INEP validation (requires Supabase)
# ═══════════════════════════════════════════════════════════════════════════════

import os


def _supabase_available() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"))


class TestV4INEPInfo:

    @pytest.mark.api
    def test_info_retorna_estrutura_quando_supabase_configurado(self, client: TestClient):
        if not _supabase_available():
            pytest.skip("Supabase não configurado")
        resp = client.get("/api/v4/enem/info")
        if resp.status_code == 503:
            pytest.skip("Supabase indisponível")
        assert resp.status_code == 200
        body = resp.json()
        assert "anos_disponiveis" in body
        assert "areas" in body
        assert "transformacoes" in body
        assert "total_itens" in body
        assert "total_provas" in body
        assert isinstance(body["anos_disponiveis"], list)

    @pytest.mark.api
    def test_info_sem_supabase_retorna_503(self, client: TestClient, monkeypatch):
        # Remove env vars temporarily
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_KEY", raising=False)
        # Clear the cached client
        from app.db.supabase_client import get_supabase
        get_supabase.cache_clear()
        resp = client.get("/api/v4/enem/info")
        assert resp.status_code == 503
        # Restore for other tests
        get_supabase.cache_clear()


class TestV4Transformacao:

    @pytest.mark.api
    def test_transformacao_valida(self, client: TestClient):
        if not _supabase_available():
            pytest.skip("Supabase não configurado")
        resp = client.get("/api/v4/enem/transformacao/2024/MT")
        if resp.status_code == 503:
            pytest.skip("Supabase indisponível")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ano"] == 2024
        assert body["area"] == "MT"
        assert "sigma" in body
        assert "mu" in body
        # Sanity: correlação deve ser ~1.0
        assert body["correlacao"] > 0.99

    @pytest.mark.api
    def test_transformacao_ano_inexistente(self, client: TestClient):
        if not _supabase_available():
            pytest.skip("Supabase não configurado")
        resp = client.get("/api/v4/enem/transformacao/1999/MT")
        if resp.status_code == 503:
            pytest.skip("Supabase indisponível")
        assert resp.status_code == 404


class TestV4ItensProva:

    @pytest.mark.api
    def test_itens_prova_valida(self, client: TestClient):
        if not _supabase_available():
            pytest.skip("Supabase não configurado")
        resp = client.get("/api/v4/enem/itens/2024/MT/1408")
        if resp.status_code == 503:
            pytest.skip("Supabase indisponível")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ano"] == 2024
        assert body["area"] == "MT"
        assert body["co_prova"] == 1408
        assert len(body["itens"]) >= 40
        # Each item should have a, b, c parameters
        for it in body["itens"][:3]:
            assert "a" in it
            assert "b" in it
            assert "c" in it
            assert "posicao" in it


class TestV4CalcularComINEP:

    @pytest.mark.api
    def test_calcular_com_inep_retorna_resultados(self, client: TestClient):
        if not _supabase_available():
            pytest.skip("Supabase não configurado")
        payload = {
            "ano": 2024,
            "area": "MT",
            "co_prova": 1408,
            "alunos": [
                {"aluno_id": "A1", "respostas": [1] * 40 + [0] * 5},
                {"aluno_id": "A2", "respostas": [1] * 20 + [0] * 25},
            ],
        }
        resp = client.post("/api/v4/enem/calcular-com-inep", json=payload)
        if resp.status_code == 503:
            pytest.skip("Supabase indisponível")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "sucesso"
        assert body["n_alunos"] == 2
        assert len(body["resultados"]) == 2
        assert body["transformacao_usada"]["sigma"] > 100
        for r in body["resultados"]:
            assert "nota_inep" in r
            assert "theta" in r
            assert "estado_tri" in r

    @pytest.mark.api
    def test_calcular_com_inep_area_invalida(self, client: TestClient):
        if not _supabase_available():
            pytest.skip("Supabase não configurado")
        payload = {
            "ano": 2024, "area": "XX", "co_prova": 1408,
            "alunos": [{"aluno_id": "A1", "respostas": [1] * 45}],
        }
        resp = client.post("/api/v4/enem/calcular-com-inep", json=payload)
        if resp.status_code == 503:
            pytest.skip()
        assert resp.status_code == 400

    @pytest.mark.api
    def test_calcular_com_inep_lc_sem_lingua(self, client: TestClient):
        if not _supabase_available():
            pytest.skip("Supabase não configurado")
        # LC requires tp_lingua
        payload = {
            "ano": 2024, "area": "LC", "co_prova": 1395,
            "alunos": [{"aluno_id": "A1", "respostas": [1] * 45}],
        }
        resp = client.post("/api/v4/enem/calcular-com-inep", json=payload)
        if resp.status_code == 503:
            pytest.skip()
        assert resp.status_code == 400

    @pytest.mark.api
    def test_calcular_com_inep_prova_inexistente(self, client: TestClient):
        if not _supabase_available():
            pytest.skip("Supabase não configurado")
        payload = {
            "ano": 2024, "area": "MT", "co_prova": 999999,
            "alunos": [{"aluno_id": "A1", "respostas": [1] * 45}],
        }
        resp = client.post("/api/v4/enem/calcular-com-inep", json=payload)
        if resp.status_code == 503:
            pytest.skip()
        assert resp.status_code == 404


class TestV4ValidarSimulado:

    @pytest.mark.api
    def test_validar_simulado_basico(self, client: TestClient):
        if not _supabase_available():
            pytest.skip("Supabase não configurado")
        # Build a synthetic CSV for MT with 15 students, 45 items
        import numpy as np
        rng = np.random.default_rng(42)
        headers = ["ID"] + [str(i + 136) for i in range(45)]
        rows = [",".join(headers)]
        for i in range(15):
            row = [f"S{i:03d}"] + [str(int(rng.random() > 0.5)) for _ in range(45)]
            rows.append(",".join(row))
        csv_content = "\n".join(rows)

        resp = client.post(
            "/api/v4/enem/validar-simulado",
            data={"ano": "2024", "area": "MT", "co_prova": "1408"},
            files={"arquivo": ("test.csv", csv_content, "text/csv")},
        )
        if resp.status_code == 503:
            pytest.skip("Supabase indisponível")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "sucesso"
        assert body["n_alunos"] == 15
        assert "estatisticas" in body
        assert "transformacao_inep" in body
        assert "resultados" in body
        s = body["estatisticas"]
        assert "correlacao" in s
        assert "mae" in s
        assert "rmse" in s

    @pytest.mark.api
    def test_validar_simulado_menos_10_alunos(self, client: TestClient):
        if not _supabase_available():
            pytest.skip("Supabase não configurado")
        headers = ["ID"] + [str(i + 136) for i in range(45)]
        rows = [",".join(headers)]
        for i in range(5):
            rows.append(",".join([f"S{i}"] + ["1"] * 45))
        resp = client.post(
            "/api/v4/enem/validar-simulado",
            data={"ano": "2024", "area": "MT", "co_prova": "1408"},
            files={"arquivo": ("test.csv", "\n".join(rows), "text/csv")},
        )
        if resp.status_code == 503:
            pytest.skip()
        assert resp.status_code == 400


class TestValidatorPage:

    @pytest.mark.api
    def test_validator_html_served(self, client: TestClient):
        resp = client.get("/validator")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "XTRI Validator" in resp.text
        assert "btnExportCsv" in resp.text
        assert "btnExportExcel" in resp.text


class TestExportarExcel:

    @pytest.mark.api
    def test_exportar_csv_retorna_csv(self, client: TestClient):
        payload = {
            "area": "MT",
            "ano_referencia": 2024,
            "resultados": [
                {
                    "aluno_id": "A1",
                    "acertos_mt": 20,
                    "theta_mt": 0.15,
                    "nota_mt": 612.4,
                    "acert_dif_mt": 0.42,
                    "estado_mt": "PROFICIENTE",
                }
            ],
        }

        resp = client.post("/api/v3/exportar-csv", json=payload)
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert "attachment" in resp.headers["content-disposition"]
        assert "A1" in resp.content.decode("utf-8")

    @pytest.mark.api
    def test_exportar_excel_retorna_xlsx(self, client: TestClient):
        # First process a CSV to get real data
        csv_content = _make_single_area_csv(30)
        anos = client.get("/api/v3/anos-referencia").json()
        ano = anos[-1]
        proc = client.post(
            "/api/v3/processar-csv",
            data={"area": "MT", "ano_referencia": str(ano)},
            files={"arquivo": ("mt.csv", csv_content, "text/csv")},
        )
        data = proc.json()

        payload = {
            "area": "MT",
            "ano_referencia": ano,
            "metodo": "Rank Intragrupo (Primi)",
            "n_alunos": data["n_alunos"],
            "resultados": data["resultados"],
            "referencia": data["referencia"],
            "diagnosticos": data["diagnosticos"],
        }

        resp = client.post("/api/v3/exportar-excel", json=payload)
        assert resp.status_code == 200
        assert "spreadsheetml" in resp.headers["content-type"]
        assert "attachment" in resp.headers["content-disposition"]
        assert len(resp.content) > 1000  # non-trivial xlsx file

    @pytest.mark.api
    def test_exportar_excel_com_grafico_base64(self, client: TestClient):
        # Minimal valid 1x1 PNG in base64
        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkAAIAAAoAAv/lxKUAAAAASUVORK5CYII="
        payload = {
            "area": "MT",
            "ano_referencia": 2022,
            "metodo": "Test",
            "n_alunos": 2,
            "resultados": [
                {"aluno_id": "A1", "acertos_mt": 20, "theta_mt": 0.5, "nota_mt": 600.0, "acert_dif_mt": 0.3, "estado_mt": "PROFICIENTE"},
                {"aluno_id": "A2", "acertos_mt": 30, "theta_mt": 1.5, "nota_mt": 750.0, "acert_dif_mt": 0.7, "estado_mt": "AVANCADO"},
            ],
            "referencia": [
                {"acertos": 20, "tri_min": 500.0, "tri_med": 600.0, "tri_max": 700.0},
                {"acertos": 30, "tri_min": 650.0, "tri_med": 750.0, "tri_max": 850.0},
            ],
            "diagnosticos": [{
                "area": "MT", "n_itens": 45, "n_alunos": 2, "b_medio": 0.0,
                "b_desvio": 1.0, "nota_media": 675.0, "nota_min": 600.0,
                "nota_max": 750.0, "pct_dentro_intervalo": 100.0,
                "itens_faceis": 15, "itens_medios": 20, "itens_dificeis": 10,
                "acert_dif_medio": 0.5,
            }],
            "chart_png_base64": png_b64,
        }

        resp = client.post("/api/v3/exportar-excel", json=payload)
        assert resp.status_code == 200
        assert len(resp.content) > 1000

    @pytest.mark.api
    def test_exportacoes_locais_salvam_em_downloads(self, client: TestClient, tmp_path: Path):
        payload = {
            "area": "MT",
            "ano_referencia": 2024,
            "metodo": "Theta Ref Fixo",
            "n_alunos": 2,
            "resultados": [
                {
                    "aluno_id": "A1",
                    "acertos_mt": 20,
                    "theta_mt": 0.5,
                    "nota_mt": 600.0,
                    "acert_dif_mt": 0.3,
                    "estado_mt": "PROFICIENTE",
                },
                {
                    "aluno_id": "A2",
                    "acertos_mt": 30,
                    "theta_mt": 1.5,
                    "nota_mt": 750.0,
                    "acert_dif_mt": 0.7,
                    "estado_mt": "AVANCADO",
                },
            ],
            "referencia": [
                {"acertos": 20, "tri_min": 500.0, "tri_med": 600.0, "tri_max": 700.0},
                {"acertos": 30, "tri_min": 650.0, "tri_med": 750.0, "tri_max": 850.0},
            ],
            "diagnosticos": [
                {
                    "area": "MT",
                    "n_itens": 45,
                    "n_alunos": 2,
                    "b_medio": 0.0,
                    "b_desvio": 1.0,
                    "nota_media": 675.0,
                    "nota_min": 600.0,
                    "nota_max": 750.0,
                    "pct_dentro_intervalo": 100.0,
                    "itens_faceis": 15,
                    "itens_medios": 20,
                    "itens_dificeis": 10,
                    "acert_dif_medio": 0.5,
                }
            ],
        }
        validator_payload = {
            "ano": 2024,
            "area": "MT",
            "co_prova": "1408",
            "n_itens": 45,
            "tempo_processamento": 3.2,
            "avisos": [],
            "estatisticas": {
                "n_alunos": 2,
                "mae": 12.3,
                "rmse": 15.1,
                "correlacao": 0.98,
                "mean_diff": 1.2,
                "max_diff_abs": 22.0,
                "pct_concordancia_nivel": 100.0,
                "nota_media_v3": 675.0,
                "nota_media_v4": 671.0,
            },
            "transformacao_inep": {
                "ano": 2024,
                "area": "MT",
                "sigma": 100.0,
                "mu": 500.0,
                "n_alunos": 1000,
                "correlacao": 0.99,
                "mae": 11.2,
            },
            "resultados": [
                {
                    "aluno_id": "A1",
                    "acertos": 20,
                    "theta_v3": 0.5,
                    "theta_v4": 0.4,
                    "nota_v3": 600.0,
                    "nota_v4": 590.0,
                    "estado_v3": "PROFICIENTE",
                    "estado_v4": "PROFICIENTE",
                    "diff": 10.0,
                }
            ],
        }

        with mock.patch("app.main.Path.home", return_value=tmp_path):
            resp_csv = client.post("/api/v3/exportar-csv?salvar_localmente=true", json=payload)
            resp_xlsx = client.post("/api/v3/exportar-excel?salvar_localmente=true", json=payload)
            resp_validator_csv = client.post(
                "/api/v4/enem/exportar-validacao-csv?salvar_localmente=true",
                json=validator_payload,
            )
            resp_validator_xlsx = client.post(
                "/api/v4/enem/exportar-validacao-excel?salvar_localmente=true",
                json=validator_payload,
            )

        for resp in (resp_csv, resp_xlsx, resp_validator_csv, resp_validator_xlsx):
            assert resp.status_code == 200
            body = resp.json()
            saved = Path(body["caminho"])
            assert saved.exists()
            assert tmp_path / "Downloads" / "XTRI Engine" in saved.parents
            assert saved.stat().st_size > 10
