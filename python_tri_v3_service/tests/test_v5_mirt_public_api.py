"""Contrato público da API MIRT: calibração, jobs, lote e autenticação."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.mirt.contracts import MIRTFactorSpec, MIRTQMatrixRow, MIRTWorkerOutput
from app.mirt.jobs import job_manager
from app.mirt.qmatrix import build_habilidade_qmatrix, build_model_spec_from_qmatrix
from app.mirt.runner import build_mirt_output_dir
from app.models import RequestEstimarLoteMIRT, RequestEstimarThetaMIRT
from app.storage.model_registry import set_active_model


def _qmatrix(area: str = "MT", n_items: int = 2):
    return build_habilidade_qmatrix(
        area=area,
        item_rows=[
            {
                "item_id": f"{area.lower()}_{idx:03d}",
                "co_item": 20_000 + idx,
                "numero_habilidade": ((idx - 1) % 2) + 1,
                "posicao": idx,
            }
            for idx in range(1, n_items + 1)
        ],
    )


def _calibration_payload(*, n_students: int = 100, n_items: int = 6) -> dict:
    qmatrix = _qmatrix(n_items=n_items)
    return {
        "simulado_id": "simulado_api_mirt",
        "area": "MT",
        "ano_referencia": 2024,
        "familia": "bifactor",
        "scoring_factor_id": "G",
        "qmatrix": qmatrix.model_dump(mode="json"),
        "alunos": [
            {
                "aluno_id": f"a{idx:03d}",
                "nome": f"Aluno {idx}",
                "respostas": [(idx + item_idx) % 2 for item_idx in range(n_items)],
            }
            for idx in range(n_students)
        ],
    }


def _write_fake_worker_artifact(worker_input) -> MIRTWorkerOutput:
    output_dir = Path(worker_input.output_dir)
    paths = {
        "model_rds_path": output_dir / "model.rds",
        "item_params_path": output_dir / "item_params.json",
        "factor_scores_path": output_dir / "factor_scores.csv",
        "fit_summary_path": output_dir / "fit_summary.json",
        "diagnostics_path": output_dir / "diagnostics.json",
        "qmatrix_snapshot_path": output_dir / "qmatrix_snapshot.json",
    }
    paths["model_rds_path"].write_text("fake-rds", encoding="utf-8")
    paths["item_params_path"].write_text("[]", encoding="utf-8")
    paths["factor_scores_path"].write_text(
        "student_id,G,SE_G\na000,0.1,0.2\n", encoding="utf-8"
    )
    paths["fit_summary_path"].write_text(
        json.dumps(
            {
                "model_id": worker_input.model_spec.model_id,
                "family": worker_input.model_spec.family,
                "qmatrix_version": worker_input.model_spec.qmatrix_version,
                "AIC": 10.0,
                "BIC": 20.0,
                "logLik": -3.0,
            }
        ),
        encoding="utf-8",
    )
    paths["diagnostics_path"].write_text("{}", encoding="utf-8")
    paths["qmatrix_snapshot_path"].write_text(
        Path(worker_input.qmatrix_path).read_text(encoding="utf-8"), encoding="utf-8"
    )
    payload = {
        "status": "done",
        "artifact_dir": str(output_dir),
        "n_students": worker_input.sample_size,
        "n_items": len(worker_input.metadata["item_ids"]),
        "model_spec": worker_input.model_spec.model_dump(mode="json"),
        "artifact_files": {key: str(path) for key, path in paths.items()},
        "warnings": ["worker-fake"],
        "error": None,
    }
    (output_dir / "worker_output.json").write_text(json.dumps(payload), encoding="utf-8")
    return MIRTWorkerOutput.model_validate(payload)


def _save_active_artifact(*, simulado_id: str, year: int = 2024, n_items: int = 2):
    qmatrix = _qmatrix(n_items=n_items)
    model = build_model_spec_from_qmatrix(qmatrix)
    artifact_id = "artifact_batch_001"
    output_dir = build_mirt_output_dir(
        year=year,
        area="MT",
        model_id=model.model_id,
        job_id=artifact_id,
    )
    worker_input = type(
        "WorkerInputStub",
        (),
        {
            "output_dir": str(output_dir),
            "sample_size": 500,
            "metadata": {"item_ids": [row.item_id for row in qmatrix.rows]},
            "model_spec": model,
            "qmatrix_path": str(output_dir / "qmatrix.json"),
        },
    )()
    (output_dir / "qmatrix.json").write_text(qmatrix.model_dump_json(), encoding="utf-8")
    _write_fake_worker_artifact(worker_input)
    set_active_model(
        {
            "artifact_id": artifact_id,
            "artifact_dir": str(output_dir),
            "simulado_id": simulado_id,
            "scope_type": "simulado_id",
            "scope_id": simulado_id,
            "ano_referencia": year,
            "area": "MT",
            "modelo": model.model_id,
            "saved_at_utc": "2026-08-07T12:00:00+00:00",
        },
        modelo=model.model_id,
    )
    return artifact_id


@pytest.fixture(autouse=True)
def _clear_mirt_jobs():
    job_manager.clear()
    yield
    job_manager.clear()


@pytest.mark.api
def test_calibracao_assincrona_registra_artefato(client: TestClient):
    payload = _calibration_payload()
    with mock.patch("app.main.mirt_runtime_available", return_value=True), mock.patch(
        "app.routers.v5_mirt.run_mirt_fit", side_effect=_write_fake_worker_artifact
    ):
        response = client.post("/api/v5/mirt/calibracoes", json=payload)
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        body = None
        for _ in range(100):
            status_response = client.get(f"/api/v5/mirt/calibracoes/{job_id}")
            assert status_response.status_code == 200
            body = status_response.json()
            if body["status"] in {"done", "error"}:
                break
            time.sleep(0.01)

    assert body is not None
    assert body["status"] == "done", body
    assert body["artifact_id"] == job_id
    assert body["fit_summary"]["BIC"] == 20.0
    assert body["warnings"] == ["worker-fake"]


@pytest.mark.api
def test_calibracao_rejeita_amostra_pequena(client: TestClient):
    response = client.post(
        "/api/v5/mirt/calibracoes", json=_calibration_payload(n_students=99)
    )
    assert response.status_code == 422


@pytest.mark.api
def test_calibracao_bifactor_rejeita_um_unico_fator_especifico(client: TestClient):
    payload = _calibration_payload()
    factors = payload["qmatrix"]["factors"]
    specific_ids = [factor["factor_id"] for factor in factors if factor["kind"] == "especifico"]
    kept_specific = specific_ids[0]
    payload["qmatrix"]["factors"] = [
        factor
        for factor in factors
        if factor["kind"] == "geral" or factor["factor_id"] == kept_specific
    ]
    for row in payload["qmatrix"]["rows"]:
        row["cargas"] = {"G": 1, kept_specific: 1}

    response = client.post("/api/v5/mirt/calibracoes", json=payload)

    assert response.status_code == 422
    assert "dois fatores específicos" in response.text


@pytest.mark.api
def test_calibracao_compensatoria_rejeita_cargas_fatoriais_identicas(
    client: TestClient,
):
    payload = _calibration_payload()
    payload["familia"] = "compensatorio"
    factor_ids = [factor["factor_id"] for factor in payload["qmatrix"]["factors"]]
    for row in payload["qmatrix"]["rows"]:
        row["cargas"] = {factor_id: 1 for factor_id in factor_ids}

    response = client.post("/api/v5/mirt/calibracoes", json=payload)

    assert response.status_code == 422
    assert "vetores de carga idênticos" in response.text


@pytest.mark.api
def test_scoring_lote_usa_um_worker_e_preserva_ausencia(client: TestClient):
    artifact_id = _save_active_artifact(simulado_id="sim_batch")
    calls = []

    def _fake_score(*, response_matrix_path, output_path, qmatrix_path, **kwargs):
        calls.append((response_matrix_path, qmatrix_path))
        with open(response_matrix_path, "r", encoding="utf-8", newline="") as handle:
            input_rows = list(csv.DictReader(handle))
        assert input_rows[1]["mt_002"] == ""
        Path(output_path).write_text(
            (
                "student_id,G,MT_H01,MT_H02,SE_G,SE_MT_H01,SE_MT_H02\n"
                "a1,0.8,0.4,0.2,0.1,0.2,0.3\n"
                "a2,0.3,0.1,0.2,0.2,0.3,0.4\n"
            ),
            encoding="utf-8",
        )
        return output_path

    with mock.patch("app.main.mirt_runtime_available", return_value=True), mock.patch(
        "app.main.run_mirt_score", side_effect=_fake_score
    ):
        response = client.post(
            "/api/v5/mirt/estimar-theta-lote",
            json={
                "area": "MT",
                "ano_referencia": 2024,
                "simulado_id": "sim_batch",
                "item_ids": ["mt_001", "mt_002"],
                "alunos": [
                    {"aluno_id": "a1", "respostas": [1, 0]},
                    {"aluno_id": "a2", "respostas": [1, None]},
                ],
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["artefato_id"] == artifact_id
    assert len(calls) == 1
    assert body["resultados"][0]["nota_ancorada"] is None
    assert body["resultados"][0]["elegivel_nota_ancorada"] is False
    assert body["resultados"][0]["avisos"]
    assert body["resultados"][1]["itens_ausentes"] == 1
    assert body["resultados"][1]["nota_ancorada"] is None
    assert body["resultados"][1]["avisos"]


@pytest.mark.api
def test_api_key_protege_rotas_mirt(client: TestClient, monkeypatch):
    monkeypatch.setenv("XTRI_API_KEY", "segredo-forte")
    with mock.patch("app.main.mirt_runtime_available", return_value=True), mock.patch(
        "app.routers.v5_mirt.submit_mirt_job"
    ):
        denied = client.post("/api/v5/mirt/calibracoes", json=_calibration_payload())
        allowed = client.post(
            "/api/v5/mirt/calibracoes",
            json=_calibration_payload(),
            headers={"X-API-Key": "segredo-forte"},
        )

    assert denied.status_code == 401
    assert allowed.status_code == 202


@pytest.mark.api
def test_rotas_mirt_falham_fechadas_sem_configuracao_de_auth(
    client: TestClient, monkeypatch
):
    monkeypatch.delenv("XTRI_API_KEY", raising=False)
    monkeypatch.delenv("XTRI_MIRT_ALLOW_UNAUTHENTICATED", raising=False)

    response = client.post("/api/v5/mirt/calibracoes", json=_calibration_payload())

    assert response.status_code == 503
    assert "Autenticação MIRT não configurada" in response.json()["detail"]


@pytest.mark.api
def test_calibracao_rejeita_conflito_e_saturacao_global(client: TestClient):
    first_payload = _calibration_payload()
    second_payload = _calibration_payload()
    second_payload["simulado_id"] = "outro_simulado"
    with mock.patch("app.main.mirt_runtime_available", return_value=True), mock.patch(
        "app.routers.v5_mirt.submit_mirt_job"
    ):
        first = client.post("/api/v5/mirt/calibracoes", json=first_payload)
        same_scope = client.post("/api/v5/mirt/calibracoes", json=first_payload)
        other_scope = client.post("/api/v5/mirt/calibracoes", json=second_payload)

    assert first.status_code == 202
    assert same_scope.status_code == 409
    assert other_scope.status_code == 429


@pytest.mark.api
def test_calibracao_nao_ativa_saida_inconsistente(client: TestClient):
    def _inconsistent_worker(worker_input):
        result = _write_fake_worker_artifact(worker_input)
        return result.model_copy(update={"n_items": result.n_items + 1})

    with mock.patch("app.main.mirt_runtime_available", return_value=True), mock.patch(
        "app.routers.v5_mirt.run_mirt_fit", side_effect=_inconsistent_worker
    ):
        response = client.post("/api/v5/mirt/calibracoes", json=_calibration_payload())
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        body = None
        for _ in range(100):
            body = client.get(f"/api/v5/mirt/calibracoes/{job_id}").json()
            if body["status"] in {"done", "error"}:
                break
            time.sleep(0.01)

    assert body is not None
    assert body["status"] == "error"
    assert "quantidade divergente de itens" in body["error"]


@pytest.mark.api
def test_scoring_individual_rejeita_valor_fora_de_zero_um(client: TestClient):
    response = client.post(
        "/api/v5/mirt/estimar-theta",
        json={
            "area": "MT",
            "simulado_id": "sim_binario",
            "item_ids": [f"mt_{idx:03d}" for idx in range(1, 46)],
            "respostas": [1, 2, 0],
        },
    )
    assert response.status_code == 422


@pytest.mark.api
def test_scoring_individual_rejeita_mu_sigma_ambiguo(client: TestClient):
    response = client.post(
        "/api/v5/mirt/estimar-theta",
        json={
            "area": "MT",
            "simulado_id": "sim_transformacao",
            "item_ids": [f"mt_{idx:03d}" for idx in range(1, 46)],
            "respostas": [1, 0, 1],
            "transformacao": {"mu": 500.0, "sigma": 100.0},
        },
    )
    assert response.status_code == 422


@pytest.mark.api
def test_scoring_lote_rejeita_se_negativo(client: TestClient):
    _save_active_artifact(simulado_id="sim_se_negativo")

    def _negative_se(*, output_path, **kwargs):
        Path(output_path).write_text(
            (
                "student_id,G,MT_H01,MT_H02,SE_G,SE_MT_H01,SE_MT_H02\n"
                "a1,0.2,0.1,0.1,-0.1,0.2,0.2\n"
            ),
            encoding="utf-8",
        )
        return output_path

    with mock.patch("app.main.mirt_runtime_available", return_value=True), mock.patch(
        "app.main.run_mirt_score", side_effect=_negative_se
    ):
        response = client.post(
            "/api/v5/mirt/estimar-theta-lote",
            json={
                "area": "MT",
                "ano_referencia": 2024,
                "simulado_id": "sim_se_negativo",
                "item_ids": ["mt_001", "mt_002"],
                "alunos": [{"aluno_id": "a1", "respostas": [1, 0]}],
            },
        )

    assert response.status_code == 502
    assert "negativo" in response.json()["detail"]


@pytest.mark.api
@pytest.mark.parametrize("artifact_id", ["*", "../fora"])
def test_scoring_lote_rejeita_artefato_id_inseguro(
    client: TestClient, artifact_id: str
):
    response = client.post(
        "/api/v5/mirt/estimar-theta-lote",
        json={
            "area": "MT",
            "ano_referencia": 2024,
            "artefato_id": artifact_id,
            "item_ids": ["mt_001"],
            "alunos": [{"aluno_id": "a1", "respostas": [1]}],
        },
    )
    assert response.status_code == 422


@pytest.mark.api
def test_scoring_lote_rejeita_ordem_de_itens_divergente(client: TestClient):
    _save_active_artifact(simulado_id="sim_item_order")
    response = client.post(
        "/api/v5/mirt/estimar-theta-lote",
        json={
            "area": "MT",
            "ano_referencia": 2024,
            "simulado_id": "sim_item_order",
            "item_ids": ["mt_002", "mt_001"],
            "alunos": [{"aluno_id": "a1", "respostas": [1, 0]}],
        },
    )
    assert response.status_code == 422


@pytest.mark.api
def test_scoring_individual_rejeita_ordem_de_itens_divergente(client: TestClient):
    _save_active_artifact(simulado_id="sim_item_order_individual", n_items=45)
    response = client.post(
        "/api/v5/mirt/estimar-theta",
        json={
            "area": "MT",
            "ano_referencia": 2024,
            "simulado_id": "sim_item_order_individual",
            "item_ids": [f"mt_{idx:03d}" for idx in range(45, 0, -1)],
            "respostas": [1, 0] * 22 + [1],
        },
    )

    assert response.status_code == 422
    assert "ordem" in response.json()["detail"]


@pytest.mark.api
@pytest.mark.parametrize("endpoint", ["individual", "lote"])
@pytest.mark.parametrize(
    "selectors",
    [
        {},
        {"simulado_id": "sim_xor", "artefato_id": "artifact_xor"},
    ],
)
def test_scoring_exige_exatamente_um_seletor_de_artefato(
    endpoint: str, selectors: dict[str, str]
):
    common = {
        "area": "MT",
        "item_ids": [f"mt_{idx:03d}" for idx in range(1, 46)],
        **selectors,
    }
    with pytest.raises(ValidationError, match="exatamente um"):
        if endpoint == "individual":
            RequestEstimarThetaMIRT.model_validate({**common, "respostas": [1] * 45})
        else:
            RequestEstimarLoteMIRT.model_validate(
                {**common, "alunos": [{"aluno_id": "a1", "respostas": [1] * 45}]}
            )


@pytest.mark.api
@pytest.mark.parametrize(
    "factor_id", ["student_id", "SE_G", "se_habilidade", "COV", "prior"]
)
def test_contrato_rejeita_fatores_que_colidem_com_colunas_do_worker(factor_id: str):
    with pytest.raises(ValidationError):
        MIRTFactorSpec(
            factor_id=factor_id,
            label="Fator reservado",
            kind="geral",
            source="manual",
            area="MT",
        )


@pytest.mark.api
def test_contrato_rejeita_item_student_id():
    with pytest.raises(ValidationError):
        MIRTQMatrixRow(item_id="student_id", area="MT", cargas={"G": 1})

    response_payload = {
        "area": "MT",
        "item_ids": ["student_id"],
        "alunos": [{"aluno_id": "a1", "respostas": [1]}],
    }
    with pytest.raises(ValidationError):
        RequestEstimarLoteMIRT.model_validate(response_payload)


@pytest.mark.api
@pytest.mark.parametrize(
    "transformacao",
    [
        {"theta_ref_centro": float("inf")},
        {"theta_ref_centro": float("nan")},
        {"theta_ref_escala": float("inf")},
        {"theta_ref_escala": float("nan")},
    ],
)
def test_contrato_rejeita_transformacao_nao_finita(transformacao: dict[str, float]):
    with pytest.raises(ValidationError):
        RequestEstimarLoteMIRT.model_validate(
            {
                "area": "MT",
                "item_ids": ["mt_001"],
                "transformacao": transformacao,
                "alunos": [{"aluno_id": "a1", "respostas": [1]}],
            }
        )


@pytest.mark.api
def test_fallback_unidimensional_nao_pode_ser_habilitado(client: TestClient):
    response = client.post(
        "/api/v5/mirt/estimar-theta",
        json={
            "area": "MT",
            "simulado_id": "sim_sem_fallback_posicional",
            "item_ids": [f"mt_{idx:03d}" for idx in range(1, 46)],
            "respostas": [1] * 45,
            "permitir_fallback_unidim": True,
        },
    )

    assert response.status_code == 422
