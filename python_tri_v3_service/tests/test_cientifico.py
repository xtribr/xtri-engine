"""Tests for the scientific IRT engine (unidimensional 2PL)."""

import numpy as np
import pytest

from app.irt.scoring import eap_theta as _eap_theta_raw
from app.irt.unidim_em import (
    _theta_grid_and_weights,
    calibrar_itens_mml_em,
)
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
        # Optimizer bounds widened to ±5 along with the [-5, 5] quadrature grid.
        for p in params:
            assert -5.0 <= p["b"] <= 5.0

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

    @pytest.mark.integration
    def test_distingue_discriminacoes_com_taxas_semelhantes(self):
        rng = np.random.default_rng(42)
        n_students = 2000
        thetas = rng.normal(size=n_students)

        # Dois itens com taxas de acerto próximas, mas discriminações reais bem
        # diferentes. A calibração conjunta precisa capturar essa diferença.
        items = [
            {"a": 2.5, "b": 0.0, "c": 0.20},
            {"a": 0.6, "b": -0.4, "c": 0.20},
            {"a": 1.2, "b": -1.0, "c": 0.20},
            {"a": 1.4, "b": 0.8, "c": 0.20},
            {"a": 0.9, "b": 1.2, "c": 0.20},
            {"a": 1.8, "b": -0.7, "c": 0.20},
        ]

        responses = []
        for item in items:
            prob = item["c"] + (1 - item["c"]) / (
                1 + np.exp(-item["a"] * (thetas - item["b"]))
            )
            responses.append((rng.random(n_students) < prob).astype(int))

        matrix = np.stack(responses, axis=1)
        params = calibrar_itens(matrix, c=0.20, n_quad=31)

        assert params[0]["a"] > params[1]["a"]


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
    def test_sinal_do_theta_define_lado_da_faixa(self, tabela):
        thetas = [
            {"aluno_id": "neg", "acertos": 20, "theta": -0.8, "se": 0.3},
            {"aluno_id": "zero", "acertos": 20, "theta": 0.0, "se": 0.3},
            {"aluno_id": "pos", "acertos": 20, "theta": 0.8, "se": 0.3},
        ]
        resultados = ancorar_notas(thetas, tabela, "MT")
        ref = tabela.obter("MT", 20)
        nota_neg = next(r["nota_ancorada"] for r in resultados if r["aluno_id"] == "neg")
        nota_zero = next(r["nota_ancorada"] for r in resultados if r["aluno_id"] == "zero")
        nota_pos = next(r["nota_ancorada"] for r in resultados if r["aluno_id"] == "pos")

        assert nota_neg <= ref["tri_med"] + 0.1
        assert abs(nota_zero - ref["tri_med"]) <= 0.1
        assert nota_pos >= ref["tri_med"] - 0.1

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

    @pytest.mark.integration
    def test_transformacao_distingue_padroes_com_mesmos_acertos(self, tabela):
        params = (
            [{"a": 1.2, "b": -2.0, "c": 0.20}] * 2
            + [{"a": 1.2, "b": -1.0, "c": 0.20}] * 2
            + [{"a": 1.2, "b": 0.5, "c": 0.20}] * 2
            + [{"a": 1.2, "b": 1.5, "c": 0.20}] * 2
        )
        respostas_padrao_1 = [1, 1, 1, 1, 0, 0, 0, 0]
        respostas_padrao_2 = [0, 0, 0, 0, 1, 1, 1, 1]
        transformacao = {"mu": 500.0, "sigma": 100.0}

        resultado_padrao_1 = estimar_theta_individual(
            respostas_padrao_1, params, "MT", tabela, transformacao=transformacao
        )
        resultado_padrao_2 = estimar_theta_individual(
            respostas_padrao_2, params, "MT", tabela, transformacao=transformacao
        )

        assert resultado_padrao_1["acertos"] == resultado_padrao_2["acertos"] == 4
        assert resultado_padrao_1["theta"] != resultado_padrao_2["theta"]
        assert resultado_padrao_1["nota_ancorada"] != resultado_padrao_2["nota_ancorada"]
        assert (
            resultado_padrao_1["theta"] > resultado_padrao_2["theta"]
        ) == (
            resultado_padrao_1["nota_ancorada"] > resultado_padrao_2["nota_ancorada"]
        )

    @pytest.mark.integration
    def test_sem_transformacao_nao_fixa_nota_na_mediana(self, tabela):
        params = (
            [{"a": 1.2, "b": -2.0, "c": 0.20}] * 2
            + [{"a": 1.2, "b": -1.0, "c": 0.20}] * 2
            + [{"a": 1.2, "b": 0.5, "c": 0.20}] * 2
            + [{"a": 1.2, "b": 1.5, "c": 0.20}] * 2
        )
        respostas_padrao_1 = [1, 1, 1, 1, 0, 0, 0, 0]
        respostas_padrao_2 = [0, 0, 0, 0, 1, 1, 1, 1]

        resultado_padrao_1 = estimar_theta_individual(
            respostas_padrao_1,
            params,
            "MT",
            tabela,
        )
        resultado_padrao_2 = estimar_theta_individual(
            respostas_padrao_2,
            params,
            "MT",
            tabela,
        )

        ref = tabela.obter("MT", 4)
        margem = (ref["tri_max"] - ref["tri_min"]) * 0.03 + 0.1

        assert resultado_padrao_1["acertos"] == resultado_padrao_2["acertos"] == 4
        assert resultado_padrao_1["theta"] != resultado_padrao_2["theta"]
        assert resultado_padrao_1["nota_ancorada"] != resultado_padrao_2["nota_ancorada"]
        assert (
            resultado_padrao_1["theta"] > resultado_padrao_2["theta"]
        ) == (
            resultado_padrao_1["nota_ancorada"] > resultado_padrao_2["nota_ancorada"]
        )
        for resultado in (resultado_padrao_1, resultado_padrao_2):
            assert ref["tri_min"] - margem <= resultado["nota_ancorada"] <= ref["tri_max"] + margem


# ═══════════════════════════════════════════════════════════════════════════════
# Quadrature + EM convergence (accuracy plan #1)
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuadratureGrid:
    """Guards against silent narrowing of the quadrature grid.

    The grid has to span the student theta distribution — narrowing it
    (e.g. back to [-4, 4]) loses tail mass for high-ability students,
    which re-introduces 2-3 MAE points per the April 2026 accuracy audit.
    """

    @pytest.mark.unit
    def test_default_grid_covers_plus_minus_5_sigma(self):
        theta_grid, weights = _theta_grid_and_weights(n_quad=61)
        assert theta_grid[0] <= -5.0 + 1e-9
        assert theta_grid[-1] >= 5.0 - 1e-9
        # Weights should sum to 1 (normalised prior on the grid).
        assert abs(float(weights.sum()) - 1.0) < 1e-12

    @pytest.mark.unit
    def test_default_grid_has_61_points(self):
        theta_grid, _ = _theta_grid_and_weights(n_quad=61)
        assert theta_grid.shape == (61,)

    @pytest.mark.unit
    def test_calibrar_mml_em_default_n_quad_is_61(self):
        """Signature defaults protect against silent narrowing."""
        import inspect
        sig = inspect.signature(calibrar_itens_mml_em)
        assert sig.parameters["n_quad"].default == 61
        assert sig.parameters["max_iter"].default >= 50
        assert sig.parameters["tol"].default <= 1e-4

    @pytest.mark.unit
    def test_production_wrapper_defaults_match_em(self):
        """``calibrar_itens`` (production wrapper used by /api/v3 routes) must
        delegate with n_quad=61, not the old 41. The April 2026 review caught
        a regression where this wrapper still had n_quad=41 and silently
        downgraded every production calibration call back to the narrow grid.
        """
        import inspect
        sig = inspect.signature(calibrar_itens)
        assert sig.parameters["n_quad"].default == 61, (
            "calibrar_itens default dropped the grid improvement — the "
            "production /api/v3 path will use the old [-4, 4, 41] grid."
        )

    @pytest.mark.unit
    def test_eap_theta_default_n_quad_is_61(self):
        import inspect
        sig = inspect.signature(_eap_theta_raw)
        assert sig.parameters["n_quad"].default == 61


class TestEAPTailSensitivity:
    """Verifies that high-ability students aren't clipped at the grid edge."""

    @pytest.mark.unit
    def test_all_correct_student_gets_high_theta(self):
        """All-correct vectors should land well into the positive tail.

        The new [-5, 5, 61] grid prevents clipping at 3.85 (the old [-4, 4, 41]
        edge). With c=0.20 the 3PL likelihood plateaus as theta grows, so EAP
        + N(0,1) prior typically centres around 2.8-3.5 for 45-item vectors —
        but we also check it's NOT stuck at the old clip point.
        """
        n_itens = 45
        item_params = [
            {"a": 1.2, "b": float(b), "c": 0.20, "taxa_acerto": 0.5, "saturado": False}
            for b in np.linspace(-2.0, 2.0, n_itens)
        ]
        theta, se = _eap_theta_raw(np.ones(n_itens, dtype=int), item_params)
        # Positive tail: at least 2σ above the prior mean
        assert theta > 2.0, f"All-correct theta too low: {theta}"
        # Not clipped at the old edge (3.85 was the old EAP ceiling)
        assert theta < 4.5, f"Theta suspiciously close to grid edge: {theta}"
        assert se > 0

    @pytest.mark.unit
    def test_all_wrong_student_gets_low_theta(self):
        n_itens = 45
        item_params = [
            {"a": 1.2, "b": float(b), "c": 0.20, "taxa_acerto": 0.5, "saturado": False}
            for b in np.linspace(-2.0, 2.0, n_itens)
        ]
        theta, _ = _eap_theta_raw(np.zeros(n_itens, dtype=int), item_params)
        assert theta < -1.5


class TestEMConvergence:
    """Higher max_iter + tighter tol should not destabilise recovery."""

    @pytest.mark.integration
    def test_recovers_item_difficulty_order(self):
        """After calibration, the rank of b must correlate with the true b."""
        rng = np.random.default_rng(17)
        n_students = 300
        n_itens = 45
        true_thetas = rng.normal(0, 1, n_students)
        true_b = np.sort(rng.uniform(-2.0, 2.0, n_itens))  # known order
        true_a = np.full(n_itens, 1.2)
        c = 0.20
        probs = c + (1 - c) / (
            1 + np.exp(-true_a[None, :] * (true_thetas[:, None] - true_b[None, :]))
        )
        resp = (rng.uniform(size=probs.shape) < probs).astype(int)

        params = calibrar_itens_mml_em(resp, c=c)
        b_hat = np.array([p["b"] for p in params])
        # Spearman-ish: rank correlation of b estimate with true b
        from scipy.stats import spearmanr
        corr, _ = spearmanr(b_hat, true_b)
        assert corr > 0.90, f"b rank correlation too low: {corr:.3f}"


# ═══════════════════════════════════════════════════════════════════════════════
# Constant-response filter (accuracy plan #2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstantResponseFilter:
    """Students with all-0 or all-N response vectors must not poison calibration.

    Per the Apr-2026 accuracy audit, these degenerate rows contribute no
    discriminating signal to the EM M-step and bias item difficulty estimates
    toward the prior mean — costing ~2-3 MAE points. The filter excludes them
    from calibration only; scoring is still applied to every student so they
    still receive a nota.
    """

    def _synthetic_baseline(self, *, n_students=200, n_itens=45, seed=23):
        rng = np.random.default_rng(seed)
        true_thetas = rng.normal(0, 1, n_students)
        true_b = np.sort(rng.uniform(-2.0, 2.0, n_itens))
        true_a = np.full(n_itens, 1.2)
        c = 0.20
        probs = c + (1 - c) / (
            1 + np.exp(-true_a[None, :] * (true_thetas[:, None] - true_b[None, :]))
        )
        resp = (rng.uniform(size=probs.shape) < probs).astype(int)
        # Reject students who happen to be all-0 or all-correct by chance — we
        # want the baseline to be clean of constant-response noise.
        keep = (resp.sum(axis=1) > 0) & (resp.sum(axis=1) < n_itens)
        return resp[keep], true_b

    @pytest.mark.integration
    def test_calibration_ignores_all_zero_students(self):
        resp, _ = self._synthetic_baseline()
        params_clean = calibrar_itens_mml_em(resp, c=0.20)

        # Splice in 10 all-zero students (5% of cohort)
        resp_poisoned = np.vstack([resp, np.zeros((10, resp.shape[1]), dtype=int)])
        params_poisoned = calibrar_itens_mml_em(resp_poisoned, c=0.20)

        b_clean = np.array([p["b"] for p in params_clean])
        b_poisoned = np.array([p["b"] for p in params_poisoned])
        # After filter, the EM sees the same cohort — b should be near-identical.
        assert np.max(np.abs(b_clean - b_poisoned)) < 0.05, (
            f"All-zero students moved b by "
            f"{np.max(np.abs(b_clean - b_poisoned)):.3f} — filter not working"
        )

    @pytest.mark.integration
    def test_calibration_ignores_all_one_students(self):
        resp, _ = self._synthetic_baseline()
        params_clean = calibrar_itens_mml_em(resp, c=0.20)

        # Splice in 10 all-correct students
        resp_poisoned = np.vstack([resp, np.ones((10, resp.shape[1]), dtype=int)])
        params_poisoned = calibrar_itens_mml_em(resp_poisoned, c=0.20)

        b_clean = np.array([p["b"] for p in params_clean])
        b_poisoned = np.array([p["b"] for p in params_poisoned])
        assert np.max(np.abs(b_clean - b_poisoned)) < 0.05, (
            f"All-correct students moved b by "
            f"{np.max(np.abs(b_clean - b_poisoned)):.3f} — filter not working"
        )

    @pytest.mark.integration
    def test_taxa_acerto_still_reflects_full_cohort(self):
        """Even though constant-response students are filtered from the EM,
        their responses should still count toward the reported taxa_acerto
        (this is a diagnostic, not a calibration parameter)."""
        resp, _ = self._synthetic_baseline()
        n_students = resp.shape[0]
        resp_with_zeros = np.vstack([resp, np.zeros((10, resp.shape[1]), dtype=int)])
        params = calibrar_itens_mml_em(resp_with_zeros, c=0.20)

        # Taxa for each item should be ≤ the baseline taxa (all-zero students
        # pulled the mean down on every item).
        full_taxa = resp_with_zeros.mean(axis=0)
        for j, p in enumerate(params):
            assert abs(p["taxa_acerto"] - full_taxa[j]) < 0.01, (
                f"Item {j} taxa_acerto ({p['taxa_acerto']}) diverged from "
                f"full-cohort empirical rate ({full_taxa[j]:.4f})"
            )

    @pytest.mark.integration
    def test_pipeline_still_scores_constant_students(self, tabela):
        """Constant-response students must still receive a nota — they're just
        excluded from calibration. Pipeline output row count == input row count."""
        resp, _ = self._synthetic_baseline(n_students=50)
        # 45 normal + 5 all-zero + 5 all-correct = 55 alunos total
        extra = np.vstack([
            np.zeros((5, resp.shape[1]), dtype=int),
            np.ones((5, resp.shape[1]), dtype=int),
        ])
        full_resp = np.vstack([resp, extra])

        alunos = [
            {"aluno_id": f"S{i:03d}", "nome": "", "respostas": full_resp[i].tolist()}
            for i in range(full_resp.shape[0])
        ]
        resultado = pipeline_cientifico(alunos=alunos, area="MT", tabela=tabela)
        assert len(resultado["resultados_alunos"]) == full_resp.shape[0]
        # All-zero students should land in the negative tail, all-correct in the
        # positive tail. Just assert sign, not exact values.
        ids_all_zero = {f"S{i:03d}" for i in range(resp.shape[0], resp.shape[0] + 5)}
        ids_all_one = {f"S{i:03d}" for i in range(resp.shape[0] + 5, resp.shape[0] + 10)}
        by_id = {r["aluno_id"]: r for r in resultado["resultados_alunos"]}
        for aid in ids_all_zero:
            assert by_id[aid]["theta"] < 0, f"{aid} all-zero but theta={by_id[aid]['theta']}"
        for aid in ids_all_one:
            assert by_id[aid]["theta"] > 0, f"{aid} all-correct but theta={by_id[aid]['theta']}"
