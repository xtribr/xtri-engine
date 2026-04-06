"""Tests for the heuristic engine (V2-compatible, pedagogical coherence)."""

import pytest

from app.motores.heuristico import (
    classificar_dificuldade,
    analisar_coerencia,
    calcular_tri_area,
    processar_turma,
    AnaliseCoerencia,
)
from app.tabela_referencia import TabelaReferenciaTRI, TRI_MAXIMA_OFICIAL


# ═══════════════════════════════════════════════════════════════════════════════
# Difficulty Classification
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassificarDificuldade:

    @pytest.mark.unit
    @pytest.mark.parametrize("pct,esperado", [
        (0.90, "muito_facil"),
        (0.80, "muito_facil"),
        (0.70, "facil"),
        (0.60, "facil"),
        (0.50, "media"),
        (0.40, "media"),
        (0.30, "dificil"),
        (0.20, "dificil"),
        (0.10, "muito_dificil"),
        (0.00, "muito_dificil"),
    ])
    def test_faixas(self, pct, esperado):
        assert classificar_dificuldade(pct) == esperado


# ═══════════════════════════════════════════════════════════════════════════════
# Coherence Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalisarCoerencia:

    @pytest.mark.unit
    def test_padrao_coerente(self):
        """Student who answers more easy questions correctly has high coherence."""
        rpd = {
            "muito_facil": 8,
            "facil": 6,
            "media": 4,
            "dificil": 2,
            "muito_dificil": 0,
            "_peso_dificuldade": 0.3,
        }
        analise = analisar_coerencia(rpd)
        assert analise.coerencia > 0.5
        assert "coerente" in analise.padrao_resposta.lower() or "consistente" in analise.padrao_resposta.lower()

    @pytest.mark.unit
    def test_padrao_incoerente(self):
        """Student who only answers hard questions has low coherence."""
        rpd = {
            "muito_facil": 0,
            "facil": 1,
            "media": 2,
            "dificil": 6,
            "muito_dificil": 8,
            "_peso_dificuldade": 0.8,
        }
        analise = analisar_coerencia(rpd)
        assert analise.coerencia < 0.7

    @pytest.mark.unit
    def test_zero_respostas(self):
        rpd = {
            "muito_facil": 0, "facil": 0, "media": 0,
            "dificil": 0, "muito_dificil": 0,
            "_peso_dificuldade": 0.5,
        }
        analise = analisar_coerencia(rpd)
        assert analise.coerencia == 0.0

    @pytest.mark.unit
    def test_retorna_taxas(self):
        rpd = {
            "muito_facil": 5, "facil": 3, "media": 2,
            "dificil": 0, "muito_dificil": 0,
            "_peso_dificuldade": 0.4,
        }
        analise = analisar_coerencia(rpd)
        assert analise.taxa_muito_facil == pytest.approx(0.5)
        assert analise.taxa_facil == pytest.approx(0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# Area TRI Calculation
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalcularTriArea:

    @pytest.mark.unit
    def test_zero_acertos_sem_ajuste(self, tabela):
        resultado = calcular_tri_area("MT", 0, tabela)
        ref = tabela.obter("MT", 0)
        assert resultado["tri_ajustado"] == ref["tri_med"]
        assert resultado["ajuste_coerencia"] == 0.0

    @pytest.mark.unit
    def test_resultado_dentro_do_intervalo(self, tabela):
        analise = AnaliseCoerencia(
            coerencia=0.7, padrao_resposta="consistente",
            taxa_muito_facil=0.4, taxa_facil=0.3, taxa_media=0.2,
            taxa_dificil=0.1, taxa_muito_dificil=0.0,
        )
        resultado = calcular_tri_area("MT", 20, tabela, analise_coerencia=analise)
        ref = tabela.obter("MT", 20)
        assert ref["tri_min"] <= resultado["tri_ajustado"] <= ref["tri_max"]

    @pytest.mark.unit
    def test_teto_oficial_respeitado(self, tabela):
        resultado = calcular_tri_area("MT", 45, tabela)
        assert resultado["tri_ajustado"] <= TRI_MAXIMA_OFICIAL["MT"]

    @pytest.mark.unit
    def test_todas_areas_validas(self, tabela):
        for area in ["LC", "CH", "CN", "MT"]:
            resultado = calcular_tri_area(area, 20, tabela)
            assert resultado["area"] == area
            assert resultado["tri_ajustado"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Full Turma Processing
# ═══════════════════════════════════════════════════════════════════════════════

class TestProcessarTurma:

    @pytest.mark.integration
    def test_retorna_prova_analysis_e_resultados(
        self, tabela, alunos_v2_30, gabarito_45, areas_config_simples
    ):
        prova, resultados = processar_turma(
            alunos_v2_30, gabarito_45, areas_config_simples, tabela
        )
        assert prova["total_alunos"] == 30
        assert len(resultados) == 30

    @pytest.mark.integration
    def test_cada_resultado_tem_tri_geral(
        self, tabela, alunos_v2_30, gabarito_45, areas_config_simples
    ):
        _, resultados = processar_turma(
            alunos_v2_30, gabarito_45, areas_config_simples, tabela
        )
        for r in resultados:
            assert "tri_geral" in r
            assert r["tri_geral"] > 0

    @pytest.mark.integration
    def test_notas_dentro_do_intervalo_oficial(
        self, tabela, alunos_v2_30, gabarito_45, areas_config_simples
    ):
        _, resultados = processar_turma(
            alunos_v2_30, gabarito_45, areas_config_simples, tabela
        )
        for r in resultados:
            assert r["tri_geral"] <= max(TRI_MAXIMA_OFICIAL.values())

    @pytest.mark.integration
    def test_areas_config_invalido_levanta_erro(self, tabela, alunos_v2_30, gabarito_45):
        with pytest.raises(ValueError, match="areas_config inválido"):
            processar_turma(alunos_v2_30, gabarito_45, {"XX": (1, 45)}, tabela)
