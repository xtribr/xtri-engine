"""Tests for the ENEM reference table loader."""

import pytest

from app.tabela_referencia import TabelaReferenciaTRI, TRI_MAXIMA_OFICIAL, TRI_MINIMA_OFICIAL


class TestTabelaReferenciaTRI:
    """Validates that the official reference table loads and behaves correctly."""

    @pytest.mark.unit
    def test_carrega_quatro_areas(self, tabela: TabelaReferenciaTRI):
        assert set(tabela.areas) == {"LC", "CH", "CN", "MT"}

    @pytest.mark.unit
    def test_cada_area_tem_range_completo(self, tabela: TabelaReferenciaTRI):
        # LC has 0-44 (45 items in its caderno), others have 0-45
        for area in ["CH", "CN", "MT"]:
            chaves = set(tabela.lookup[area].keys())
            assert set(range(46)) == chaves, f"{area} missing keys"
        assert set(range(45)) == set(tabela.lookup["LC"].keys()), "LC missing keys"

    @pytest.mark.unit
    def test_mediana_monotonicamente_crescente(self, tabela: TabelaReferenciaTRI):
        for area in tabela.areas:
            acertos_ordenados = sorted(tabela.lookup[area].keys())
            meds = [tabela.lookup[area][a]["tri_med"] for a in acertos_ordenados]
            assert meds == sorted(meds), f"{area}: mediana nao e monotonica"

    @pytest.mark.unit
    def test_min_menor_ou_igual_max(self, tabela: TabelaReferenciaTRI):
        for area in tabela.areas:
            for acertos in range(46):
                ref = tabela.obter(area, acertos)
                assert ref["tri_min"] <= ref["tri_med"] <= ref["tri_max"], (
                    f"{area}/{acertos}: min={ref['tri_min']} med={ref['tri_med']} max={ref['tri_max']}"
                )

    @pytest.mark.unit
    def test_obter_zero_acertos(self, tabela: TabelaReferenciaTRI):
        for area in tabela.areas:
            ref = tabela.obter(area, 0)
            assert ref["tri_min"] > 0
            assert ref["tri_med"] > 0

    @pytest.mark.unit
    def test_obter_45_acertos(self, tabela: TabelaReferenciaTRI):
        # Table values for max acertos must be positive and within
        # a sane historical range (the constant TRI_MAXIMA_OFICIAL acts
        # as a final clip in ancorar_notas; table values may slightly
        # exceed it for the very top of the historical band).
        for area in tabela.areas:
            ref = tabela.obter(area, 45)
            assert ref["tri_max"] > 0
            assert ref["tri_max"] < 1000.0  # nothing exceeds the absolute ENEM ceiling

    @pytest.mark.unit
    def test_obter_acertos_negativo_retorna_zero(self, tabela: TabelaReferenciaTRI):
        ref = tabela.obter("MT", -5)
        ref_zero = tabela.obter("MT", 0)
        assert ref == ref_zero

    @pytest.mark.unit
    def test_obter_acertos_acima_45_retorna_maximo(self, tabela: TabelaReferenciaTRI):
        ref = tabela.obter("MT", 999)
        ref_45 = tabela.obter("MT", 45)
        assert ref == ref_45

    @pytest.mark.unit
    def test_area_invalida_levanta_erro(self, tabela: TabelaReferenciaTRI):
        with pytest.raises(ValueError, match="Área inválida"):
            tabela.obter("XX", 10)

    @pytest.mark.unit
    def test_validar_retorna_true(self, tabela: TabelaReferenciaTRI):
        assert tabela.validar() is True

    @pytest.mark.unit
    def test_info_retorna_estrutura_correta(self, tabela: TabelaReferenciaTRI):
        info = tabela.info()
        assert "areas" in info
        assert "acertos_por_area" in info
        assert "limites_oficiais" in info
        assert info["acertos_por_area"]["LC"] == 45  # LC: 0-44
        for area in ["CH", "CN", "MT"]:
            assert info["acertos_por_area"][area] == 46

    @pytest.mark.unit
    def test_constantes_oficiais_coerentes(self):
        for area in ["LC", "CH", "CN", "MT"]:
            assert area in TRI_MAXIMA_OFICIAL
            assert area in TRI_MINIMA_OFICIAL
            assert TRI_MINIMA_OFICIAL[area] < TRI_MAXIMA_OFICIAL[area]


class TestTabelaReferenciaMultiano:
    """Tests for multi-year reference table loading."""

    @pytest.fixture(scope="class")
    def multiano(self) -> dict[int, TabelaReferenciaTRI]:
        return TabelaReferenciaTRI.carregar_multiano()

    @pytest.mark.unit
    def test_anos_esperados(self, multiano):
        anos = set(multiano.keys())
        assert {2009, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022} == anos

    @pytest.mark.unit
    def test_cada_ano_tem_quatro_areas(self, multiano):
        for ano, tab in multiano.items():
            assert set(tab.areas) == {"LC", "CH", "CN", "MT"}, f"Year {ano}"

    @pytest.mark.unit
    def test_min_menor_ou_igual_max_todos_anos(self, multiano):
        for ano, tab in multiano.items():
            for area in tab.areas:
                for acertos in tab.lookup[area]:
                    ref = tab.lookup[area][acertos]
                    assert ref["tri_min"] <= ref["tri_max"], (
                        f"Year {ano}, {area}/{acertos}"
                    )

    @pytest.mark.unit
    def test_obter_funciona_por_ano(self, multiano):
        ref = multiano[2022].obter("MT", 25)
        assert ref["tri_min"] > 0
        assert ref["tri_med"] > 0

    @pytest.mark.unit
    def test_ano_tem_atributo(self, multiano):
        assert multiano[2022].ano == 2022
        assert multiano[2009].ano == 2009

    @pytest.mark.unit
    def test_tabela_padrao_nao_afetada(self):
        """Original single-year constructor still works."""
        t = TabelaReferenciaTRI()
        t.validar()
        assert t.ano is None
        assert set(t.areas) == {"LC", "CH", "CN", "MT"}
