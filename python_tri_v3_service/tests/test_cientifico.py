"""Tests for the scientific IRT engine (unidimensional 2PL)."""

import numpy as np
import pytest

from app.motores.cientifico import (
    P,
    calibrar_itens,
    eap_theta,
    ancorar_notas,
    estado_tri,
    pipeline_cientifico,
    estimar_theta_individual,
)
from app.tabela_referencia import TabelaReferenciaTRI, TRI_MAXIMA_OFICIAL


# ═══════════════════════════════════════════════════════════════════════════════
# Item Response Function
# ═══════════════════════════════════════════════════════════════════════════════

class TestIRF:

    @pytest.mark.unit
    def test_P_at_b_equals_midpoint(self):
        """At theta == b, P should equal (1+c)/2."""
        c = 0.20
        theta = np.array([1.5])
        prob = P(theta, a=1.0, b=1.5, c=c)
        expected = (1 + c) / 2  # 0.60
        assert abs(prob[0] - expected) < 1e-6

    @pytest.mark.unit
    def test_P_bounds(self):
        """P must be in [c, 1] for all theta."""
        c = 0.20
        thetas = np.linspace(-4, 4, 100)
        probs = P(thetas, a=1.5, b=0.0, c=c)
        assert np.all(probs >= c - 1e-9)
        assert np.all(probs <= 1.0 + 1e-9)

    @pytest.mark.unit
    def test_P_monotonically_increasing(self):
        """P must increase with theta for positive a."""
        thetas = np.linspace(-4, 4, 100)
        probs = P(thetas, a=1.0, b=0.0, c=0.20)
        assert np.all(np.diff(probs) >= -1e-9)

    @pytest.mark.unit
    def test_P_higher_a_steeper(self):
        """Higher discrimination (a) produces steeper curve."""
        thetas = np.linspace(-1, 1, 50)
        probs_low = P(thetas, a=0.5, b=0.0, c=0.20)
        probs_high = P(thetas, a=2.0, b=0.0, c=0.20)
        # Range of probs should be wider for higher a
        assert (probs_high.max() - probs_high.min()) > (probs_low.max() - probs_low.min())


# ═══════════════════════════════════════════════════════════════════════════════
# Item Calibration
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibrarItens:

    @pytest.mark.integration
    def test_retorna_parametros_para_cada_item(self, respostas_binarias_50x45):
        matrix = np.array(respostas_binarias_50x45, dtype=int)
        params = calibrar_itens(matrix, c=0.20)
        assert len(params) == 45

    @pytest.mark.integration
    def test_parametros_tem_campos_esperados(self, respostas_binarias_50x45):
        matrix = np.array(respostas_binarias_50x45, dtype=int)
        params = calibrar_itens(matrix, c=0.20)
        for p in params:
            assert "a" in p
            assert "b" in p
            assert "c" in p
            assert "taxa_acerto" in p
            assert "saturado" in p

    @pytest.mark.integration
    def test_a_dentro_dos_limites(self, respostas_binarias_50x45):
        matrix = np.array(respostas_binarias_50x45, dtype=int)
        params = calibrar_itens(matrix, c=0.20)
        for p in params:
            assert 0.3 <= p["a"] <= 4.0

    @pytest.mark.integration
    def test_b_dentro_dos_limites(self, respostas_binarias_50x45):
        matrix = np.array(respostas_binarias_50x45, dtype=int)
        params = calibrar_itens(matrix, c=0.20)
        for p in params:
            assert -4.0 <= p["b"] <= 4.0

    @pytest.mark.integration
    def test_c_fixo(self, respostas_binarias_50x45):
        matrix = np.array(respostas_binarias_50x45, dtype=int)
        params = calibrar_itens(matrix, c=0.20)
        for p in params:
            assert p["c"] == 0.20

    @pytest.mark.integration
    def test_taxa_acerto_entre_0_e_1(self, respostas_binarias_50x45):
        matrix = np.array(respostas_binarias_50x45, dtype=int)
        params = calibrar_itens(matrix, c=0.20)
        for p in params:
            assert 0.0 <= p["taxa_acerto"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# EAP Theta Estimation
# ═══════════════════════════════════════════════════════════════════════════════

class TestEAPTheta:

    @pytest.mark.unit
    def test_retorna_tupla_theta_se(self):
        params = [{"a": 1.0, "b": 0.0, "c": 0.20}] * 10
        respostas = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        theta, se = eap_theta(respostas, params)
        assert isinstance(theta, float)
        assert isinstance(se, float)

    @pytest.mark.unit
    def test_theta_maior_para_mais_acertos(self):
        params = [{"a": 1.0, "b": 0.0, "c": 0.20}] * 10
        baixo = np.array([0] * 8 + [1] * 2)
        alto = np.array([1] * 8 + [0] * 2)
        theta_baixo, _ = eap_theta(baixo, params)
        theta_alto, _ = eap_theta(alto, params)
        assert theta_alto > theta_baixo

    @pytest.mark.unit
    def test_se_positivo(self):
        params = [{"a": 1.0, "b": 0.0, "c": 0.20}] * 10
        respostas = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        _, se = eap_theta(respostas, params)
        assert se > 0

    @pytest.mark.unit
    def test_todos_zero_retorna_theta_negativo(self):
        params = [{"a": 1.0, "b": 0.0, "c": 0.20}] * 10
        respostas = np.zeros(10)
        theta, _ = eap_theta(respostas, params)
        assert theta < 0

    @pytest.mark.unit
    def test_todos_um_retorna_theta_positivo(self):
        params = [{"a": 1.0, "b": 0.0, "c": 0.20}] * 10
        respostas = np.ones(10)
        theta, _ = eap_theta(respostas, params)
        assert theta > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Score Anchoring
# ═══════════════════════════════════════════════════════════════════════════════

class TestAncorarNotas:

    @pytest.mark.unit
    def test_nota_dentro_do_intervalo_da_tabela(self, tabela):
        thetas = [
            {"aluno_id": "1", "acertos": 20, "theta": 0.5, "se": 0.3},
            {"aluno_id": "2", "acertos": 10, "theta": -1.0, "se": 0.4},
        ]
        resultados = ancorar_notas(thetas, tabela, "MT")
        for r in resultados:
            ref = tabela.obter("MT", r["acertos"])
            assert ref["tri_min"] <= r["nota_ancorada"] <= TRI_MAXIMA_OFICIAL["MT"]

    @pytest.mark.unit
    def test_theta_maior_gera_nota_maior_mesmo_acertos(self, tabela):
        thetas = [
            {"aluno_id": "1", "acertos": 20, "theta": -1.0, "se": 0.3},
            {"aluno_id": "2", "acertos": 20, "theta": 1.0, "se": 0.3},
        ]
        resultados = ancorar_notas(thetas, tabela, "MT")
        assert resultados[1]["nota_ancorada"] >= resultados[0]["nota_ancorada"]

    @pytest.mark.unit
    def test_zero_acertos_nota_fixa(self, tabela):
        thetas = [{"aluno_id": "1", "acertos": 0, "theta": -2.0, "se": 1.0}]
        resultados = ancorar_notas(thetas, tabela, "MT")
        ref = tabela.obter("MT", 0)
        # 0 acertos: interval is collapsed (min == max), so nota == tri_med
        assert abs(resultados[0]["nota_ancorada"] - ref["tri_med"]) < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Estado TRI Classification
# ═══════════════════════════════════════════════════════════════════════════════

class TestEstadoTRI:

    @pytest.mark.unit
    @pytest.mark.parametrize("nota,esperado", [
        (750, "AVANCADO"),
        (700, "AVANCADO"),
        (650, "PROFICIENTE"),
        (600, "PROFICIENTE"),
        (550, "BASICO"),
        (500, "BASICO"),
        (450, "ELEMENTAR"),
        (400, "ELEMENTAR"),
        (350, "INICIANTE"),
        (0, "INICIANTE"),
    ])
    def test_classificacao(self, nota, esperado):
        assert estado_tri(nota) == esperado


# ═══════════════════════════════════════════════════════════════════════════════
# Full Scientific Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineCientifico:

    @pytest.mark.integration
    def test_pipeline_retorna_estrutura_completa(
        self, tabela, respostas_binarias_50x45
    ):
        alunos = [
            {"aluno_id": str(i), "nome": f"Aluno {i}", "respostas": r}
            for i, r in enumerate(respostas_binarias_50x45)
        ]
        resultado = pipeline_cientifico(alunos, "MT", tabela, c_fixo=0.20)

        assert "parametros_itens" in resultado
        assert "resultados_alunos" in resultado
        assert "diagnostico" in resultado
        assert len(resultado["parametros_itens"]) == 45
        assert len(resultado["resultados_alunos"]) == 50

    @pytest.mark.integration
    def test_notas_dentro_do_intervalo_oficial(
        self, tabela, respostas_binarias_50x45
    ):
        alunos = [
            {"aluno_id": str(i), "respostas": r}
            for i, r in enumerate(respostas_binarias_50x45)
        ]
        resultado = pipeline_cientifico(alunos, "MT", tabela)

        for r in resultado["resultados_alunos"]:
            assert r["nota_ancorada"] <= TRI_MAXIMA_OFICIAL["MT"] + 0.1
            assert r["nota_ancorada"] > 0

    @pytest.mark.integration
    def test_diagnostico_tem_campos_esperados(
        self, tabela, respostas_binarias_50x45
    ):
        alunos = [
            {"aluno_id": str(i), "respostas": r}
            for i, r in enumerate(respostas_binarias_50x45)
        ]
        resultado = pipeline_cientifico(alunos, "MT", tabela)
        diag = resultado["diagnostico"]

        assert "n_alunos" in diag
        assert "n_itens" in diag
        assert "b_medio" in diag
        assert "nota_media" in diag
        assert "pct_dentro_intervalo" in diag


# ═══════════════════════════════════════════════════════════════════════════════
# Individual Theta Estimation
# ═══════════════════════════════════════════════════════════════════════════════

class TestEstimarThetaIndividual:

    @pytest.mark.integration
    def test_retorna_campos_esperados(self, tabela):
        params = [{"a": 1.0, "b": 0.0, "c": 0.20}] * 20
        respostas = [1, 0, 1, 1, 0] * 4
        resultado = estimar_theta_individual(respostas, params, "MT", tabela)

        assert "acertos" in resultado
        assert "theta" in resultado
        assert "se" in resultado
        assert "nota_ancorada" in resultado
        assert "estado_tri" in resultado

    @pytest.mark.integration
    def test_nota_dentro_do_intervalo(self, tabela):
        params = [{"a": 1.0, "b": 0.0, "c": 0.20}] * 20
        respostas = [1, 0, 1, 1, 0] * 4
        resultado = estimar_theta_individual(respostas, params, "MT", tabela)
        assert resultado["nota_ancorada"] <= TRI_MAXIMA_OFICIAL["MT"]
        assert resultado["nota_ancorada"] > 0
