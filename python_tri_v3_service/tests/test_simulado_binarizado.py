"""Testes da correção de simulado binarizado (POST /api/v3/processar-json)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.simulado_binarizado import (
    N_MIN_C_LIVRE,
    SimuladoInvalidoError,
    processar_simulado_binarizado,
)
from app.tabela_referencia import TRI_MINIMA_OFICIAL, TabelaReferenciaTRI

AREAS = {"LC": [1, 45], "CH": [46, 90], "CN": [91, 135], "MT": [136, 180]}


def _simulado(n_alunos: int, seed: int = 7, areas=AREAS, n_questoes: int = 180) -> dict:
    """Respostas 3PL sintéticas com c = 0,20."""
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(n_alunos)
    a = rng.uniform(0.8, 2.0, n_questoes)
    b = rng.uniform(-1.5, 2.0, n_questoes)
    prob = 0.2 + 0.8 / (1 + np.exp(-a * (theta[:, None] - b)))
    resp = (rng.random(prob.shape) < prob).astype(int)
    return {
        "total_questoes": n_questoes,
        "areas": areas,
        "alunos": [
            {"id": f"a{i}", "matricula": str(1000 + i), "nome": f"Aluno {i}", "turma": "EM3A",
             "fez_area": {k: True for k in areas}, "respostas": resp[i].tolist()}
            for i in range(n_alunos)
        ],
    }


@pytest.fixture(scope="module")
def tabela_2023() -> TabelaReferenciaTRI:
    return TabelaReferenciaTRI.carregar_multiano()[2023]


@pytest.mark.unit
class TestProcessarSimulado:

    def test_turma_pequena_usa_c_fixo(self, tabela_2023):
        out = processar_simulado_binarizado(_simulado(60), tabela_2023, 2023)
        assert set(out["areas"]) == set(AREAS)
        for bloco in out["areas"].values():
            assert bloco["modelo"] == "3PL c fixo 0.2"
            assert all(it["c"] == 0.20 for it in bloco["itens"])
            assert len(bloco["itens"]) == 45
        aluno = out["alunos"][0]
        assert set(aluno["areas"]) == set(AREAS)
        assert aluno["media_tri"] is not None
        for area, r in aluno["areas"].items():
            assert r["nota_tri"] >= TRI_MINIMA_OFICIAL[area]
            assert 0.01 <= r["posicao_no_grupo"] <= 0.99

    def test_coorte_de_500_estima_c(self, tabela_2023):
        dados = _simulado(N_MIN_C_LIVRE, areas={"MT": [1, 20]}, n_questoes=20)
        out = processar_simulado_binarizado(dados, tabela_2023, 2023)
        bloco = out["areas"]["MT"]
        assert bloco["modelo"].startswith("3PL c estimado")
        assert len({it["c"] for it in bloco["itens"]}) > 1

    def test_aluno_que_nao_fez_area_fica_sem_nota(self, tabela_2023):
        dados = _simulado(40)
        dados["alunos"][0]["fez_area"]["CN"] = False
        dados["alunos"][0]["respostas"][90:135] = [None] * 45
        out = processar_simulado_binarizado(dados, tabela_2023, 2023)
        aluno = out["alunos"][0]
        assert "CN" not in aluno["areas"]
        assert aluno["media_tri"] is None
        assert out["areas"]["CN"]["n_alunos"] == 39

    def test_area_com_poucos_alunos_gera_aviso(self, tabela_2023):
        dados = _simulado(40)
        for aluno in dados["alunos"][5:]:
            aluno["fez_area"]["MT"] = False
        out = processar_simulado_binarizado(dados, tabela_2023, 2023)
        assert "MT" not in out["areas"]
        assert any(av.startswith("MT:") for av in out["avisos"])

    def test_resposta_fora_de_0_1_e_rejeitada(self, tabela_2023):
        dados = _simulado(20)
        dados["alunos"][3]["respostas"][10] = 2
        with pytest.raises(SimuladoInvalidoError, match="0 ou 1"):
            processar_simulado_binarizado(dados, tabela_2023, 2023)

    def test_respostas_curtas_sao_rejeitadas(self, tabela_2023):
        dados = _simulado(20)
        dados["alunos"][0]["respostas"] = dados["alunos"][0]["respostas"][:100]
        with pytest.raises(SimuladoInvalidoError, match="respostas"):
            processar_simulado_binarizado(dados, tabela_2023, 2023)


@pytest.mark.api
class TestEndpointProcessarJson:

    def test_sucesso(self, client: TestClient):
        resp = client.post("/api/v3/processar-json", json={**_simulado(40), "ano_referencia": 2023})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "sucesso"
        assert body["ano_referencia"] == 2023
        assert body["total_alunos"] == 40
        assert body["alunos"][0]["matricula"] == "1000"
        assert set(body["alunos"][0]["areas"]["MT"]) >= {"acertos", "theta", "se_theta", "nota_tri", "nivel"}

    def test_ano_invalido_retorna_400(self, client: TestClient):
        resp = client.post("/api/v3/processar-json", json={**_simulado(15), "ano_referencia": 1999})
        assert resp.status_code == 400

    def test_area_invalida_retorna_400(self, client: TestClient):
        dados = _simulado(15, areas={"XX": [1, 45]})
        resp = client.post("/api/v3/processar-json", json=dados)
        assert resp.status_code == 400

    def test_nenhuma_area_calibrada_retorna_400(self, client: TestClient):
        resp = client.post("/api/v3/processar-json", json=_simulado(5))
        assert resp.status_code == 400
        assert "mínimo" in resp.json()["detail"]
