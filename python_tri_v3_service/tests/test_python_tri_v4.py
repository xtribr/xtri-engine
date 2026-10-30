"""Testes das correções do python_tri_v4: chute inicial de b, convergência do EM,
bloco ``execucao`` e alunos ausentes por dia de prova."""

import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import versao
from app.irt.unidim_em import _init_item_params, calibrar_itens_mml_em
from app.motores.cientifico import pipeline_cientifico
from app.simulado_binarizado import NIVEL_AUSENTE, processar_simulado_binarizado
from app.tabela_referencia import TabelaReferenciaTRI

AREAS = {"LC": [1, 45], "CH": [46, 90], "CN": [91, 135], "MT": [136, 180]}
CAMPOS_EXECUCAO = {"motor_versao", "ano_referencia", "c_strategy", "em_iteracoes", "em_convergiu", "processado_em"}


def _simulado(n_alunos: int, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(n_alunos)
    a = rng.uniform(0.8, 2.0, 180)
    b = rng.uniform(-1.5, 2.0, 180)
    prob = 0.2 + 0.8 / (1 + np.exp(-a * (theta[:, None] - b)))
    resp = (rng.random(prob.shape) < prob).astype(int)
    return {
        "areas": AREAS,
        "alunos": [{"id": f"a{i}", "respostas": resp[i].tolist()} for i in range(n_alunos)],
    }


@pytest.fixture(scope="module")
def tabela_2023() -> TabelaReferenciaTRI:
    return TabelaReferenciaTRI.carregar_multiano()[2023]


@pytest.mark.unit
class TestChuteInicialEConvergencia:

    def test_sinal_do_b_inicial(self):
        # 30% de acerto com c = 0,20: b = ln(0,8/0,1 - 1) = ln 7 = +1,95 (item difícil)
        dificil = np.zeros((100, 1), dtype=int)
        dificil[:30, 0] = 1
        assert _init_item_params(dificil, 0.20)[0]["b"] == pytest.approx(np.log(7), abs=1e-3)
        # 90% de acerto: item fácil começa com b negativo
        facil = np.zeros((100, 1), dtype=int)
        facil[:90, 0] = 1
        assert _init_item_params(facil, 0.20)[0]["b"] < 0

    def test_retorno_padrao_continua_lista(self):
        rng = np.random.default_rng(1)
        matriz = (rng.random((80, 10)) < 0.6).astype(int)
        params = calibrar_itens_mml_em(matriz, c=0.20)
        assert isinstance(params, list) and len(params) == 10

    def test_recupera_parametros_e_converge(self):
        rng = np.random.default_rng(2026)
        n, j = 3000, 20
        a_true = rng.uniform(0.8, 2.0, j)
        b_true = np.linspace(-2.0, 2.0, j)
        theta = rng.standard_normal(n)
        prob = 0.2 + 0.8 / (1 + np.exp(-a_true * (theta[:, None] - b_true)))
        matriz = (rng.random(prob.shape) < prob).astype(int)

        params, info = calibrar_itens_mml_em(matriz, c=0.20, return_info=True)

        assert info["convergiu"] is True
        assert 1 <= info["iteracoes"] < 500
        b_est = np.array([p["b"] for p in params])
        a_est = np.array([p["a"] for p in params])
        assert np.corrcoef(b_est, b_true)[0, 1] > 0.98
        assert np.mean(np.abs(b_est - b_true)) < 0.25
        assert np.corrcoef(a_est, a_true)[0, 1] > 0.7

    def test_aviso_quando_nao_converge(self, caplog):
        rng = np.random.default_rng(3)
        matriz = (rng.random((200, 15)) < 0.5).astype(int)
        _, info = calibrar_itens_mml_em(matriz, c=0.20, max_iter=1, return_info=True)
        assert info == {"iteracoes": 1, "convergiu": False}
        assert any("sem convergir" in r.message for r in caplog.records)


@pytest.mark.unit
class TestExecucao:

    def test_pipeline_devolve_execucao_e_diagnostico(self, tabela_2023, monkeypatch):
        monkeypatch.setenv("XTRI_MOTOR_VERSAO", "abc123def456")
        versao.motor_versao.cache_clear()
        dados = _simulado(60)
        alunos = [{"aluno_id": str(i), "respostas": a["respostas"][135:180]} for i, a in enumerate(dados["alunos"])]
        out = pipeline_cientifico(alunos, "MT", tabela_2023, ano_referencia=2023)
        versao.motor_versao.cache_clear()

        ex = out["execucao"]
        assert set(ex) == CAMPOS_EXECUCAO
        assert ex["motor_versao"] == "abc123def456"
        assert ex["ano_referencia"] == 2023
        assert ex["c_strategy"] == "fixed"
        assert isinstance(ex["em_convergiu"], bool) and ex["em_iteracoes"] >= 1
        assert re.search(r"[+-]03:00$", ex["processado_em"])
        assert out["diagnostico"]["em_iteracoes"] == ex["em_iteracoes"]
        assert out["diagnostico"]["em_convergiu"] == ex["em_convergiu"]
        assert len(out["parametros_itens"]) == 45

    def test_versao_desconhecida_sem_env_nem_git(self, monkeypatch):
        monkeypatch.delenv("XTRI_MOTOR_VERSAO", raising=False)
        monkeypatch.delenv("GIT_COMMIT", raising=False)
        monkeypatch.setattr(versao.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
        versao.motor_versao.cache_clear()
        assert versao.motor_versao() == "desconhecida"
        versao.motor_versao.cache_clear()


@pytest.mark.unit
class TestAusentes:

    def test_ausente_sem_marcacao_fica_fora_e_presente_com_zero_tem_nota(self, tabela_2023):
        dados = _simulado(60)
        # aluno 0: nenhuma marcação no dia 2 (CN e MT) -> AUSENTE no dia 2
        dados["alunos"][0]["respostas"][90:180] = [None] * 90
        # aluno 1: presente no dia 2, marcou tudo errado (zero acertos em MT)
        dados["alunos"][1]["respostas"][135:180] = [0] * 45
        # aluno 2: presença falsa no dia 1, mesmo com respostas preenchidas
        dados["alunos"][2]["presenca"] = {"dia1": False}

        out = processar_simulado_binarizado(dados, tabela_2023, 2023)

        a0 = out["alunos"][0]
        assert a0["ausente_dia"] == {"dia1": False, "dia2": True}
        for area in ("CN", "MT"):
            assert a0["areas"][area]["nota_tri"] is None
            assert a0["areas"][area]["nivel"] == NIVEL_AUSENTE
        assert a0["areas"]["LC"]["nota_tri"] is not None
        assert a0["media_tri"] is None

        a2 = out["alunos"][2]
        assert a2["areas"]["LC"]["nivel"] == NIVEL_AUSENTE and a2["areas"]["CH"]["nota_tri"] is None
        assert a2["areas"]["MT"]["nota_tri"] is not None

        # fora da calibração e do resumo da área
        assert out["areas"]["MT"]["n_alunos"] == 59 and out["areas"]["MT"]["n_ausentes"] == 1
        assert out["areas"]["LC"]["n_alunos"] == 59
        notas_mt = [a["areas"]["MT"]["nota_tri"] for a in out["alunos"] if a["areas"]["MT"]["nota_tri"] is not None]
        assert len(notas_mt) == 59
        assert out["areas"]["MT"]["resumo"]["nota_min"] == min(notas_mt)

        a1 = out["alunos"][1]["areas"]["MT"]
        assert a1["acertos"] == 0 and a1["nivel"] != NIVEL_AUSENTE
        assert a1["nota_tri"] is not None and a1["nota_tri"] == min(notas_mt)

    def test_branco_em_dia_presente_conta_como_erro(self, tabela_2023):
        dados = _simulado(40)
        resp = dados["alunos"][5]["respostas"]
        resp[140] = None  # uma questão em branco em MT
        out = processar_simulado_binarizado(dados, tabela_2023, 2023)
        mt = out["alunos"][5]["areas"]["MT"]
        assert mt["nota_tri"] is not None
        assert mt["acertos"] == sum(v for v in resp[135:180] if v is not None)


@pytest.mark.api
class TestEndpointV4:

    def test_rota_devolve_execucao_e_ausentes(self, client: TestClient):
        dados = {**_simulado(40), "ano_referencia": 2023}
        dados["alunos"][0]["respostas"][90:180] = [None] * 90
        resp = client.post("/api/v3/processar-json", json=dados)
        assert resp.status_code == 200
        body = resp.json()
        for area, bloco in body["areas"].items():
            assert set(bloco["execucao"]) == CAMPOS_EXECUCAO
            assert bloco["execucao"]["ano_referencia"] == 2023
            assert len(bloco["itens"]) == 45
        mt = body["alunos"][0]["areas"]["MT"]
        assert mt["nota_tri"] is None and mt["nivel"] == "AUSENTE" and mt["ausente"] is True
        assert body["alunos"][0]["media_tri"] is None
