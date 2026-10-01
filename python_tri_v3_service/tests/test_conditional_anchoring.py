"""Testes da ancoragem condicional aos acertos (app/irt/conditional.py)."""

import numpy as np
import pytest

from app.irt.conditional import (
    PESO_MAXIMO,
    PESO_N_DESCONHECIDO,
    PROBS_QUANTIS,
    ajustar_curva,
    curva_teorica,
    nota_por_posicao,
    peso_por_tamanho,
    posicao_relativa,
)
from app.irt.transform import ancorar_nota_individual
from app.motores.cientifico import estimar_theta_individual, pipeline_cientifico
from app.tabela_referencia import TabelaReferenciaTRI


def _coorte_sintetica(n_alunos: int, n_itens: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n_alunos)
    a = rng.uniform(0.8, 2.2, n_itens)
    b = rng.uniform(-2.0, 2.0, n_itens)
    prob = 0.2 + 0.8 / (1 + np.exp(-a[None, :] * (theta[:, None] - b[None, :])))
    resp = (rng.uniform(size=prob.shape) < prob).astype(int)
    return [
        {"aluno_id": f"S{i:04d}", "nome": "", "respostas": resp[i].tolist()}
        for i in range(n_alunos)
    ]


class TestPeso:

    @pytest.mark.unit
    def test_cresce_com_a_coorte_e_tem_teto(self):
        pesos = [peso_por_tamanho(n) for n in (10, 40, 100, 300, 5000)]
        assert pesos == sorted(pesos)
        assert pesos[-1] == PESO_MAXIMO
        assert pesos[0] < 0.4

    @pytest.mark.unit
    @pytest.mark.parametrize("n", [None, 0, -5, float("nan")])
    def test_tamanho_desconhecido_usa_peso_conservador(self, n):
        assert peso_por_tamanho(n) == PESO_N_DESCONHECIDO


class TestCurva:

    @pytest.mark.unit
    def test_media_e_monotonica_nos_acertos(self):
        rng = np.random.default_rng(1)
        acertos = rng.integers(5, 40, size=2000)
        theta = (acertos - 22) / 8 + rng.normal(0, 0.2, size=2000)
        curva = ajustar_curva(acertos, theta, n_calibracao=2000)
        assert curva["theta_medio"] == sorted(curva["theta_medio"])
        assert min(curva["theta_dp"]) > 0

    @pytest.mark.unit
    def test_aluno_tipico_do_grupo_fica_no_meio_da_faixa(self):
        curva = ajustar_curva([10, 10, 20, 20, 30, 30], [-1.1, -0.9, -0.1, 0.1, 0.9, 1.1], n_calibracao=500)
        assert posicao_relativa(-1.0, 10, curva) == pytest.approx(0.5, abs=1e-6)
        assert posicao_relativa(1.0, 30, curva) == pytest.approx(0.5, abs=1e-6)
        assert posicao_relativa(1.5, 30, curva) > 0.5 > posicao_relativa(0.5, 30, curva)

    @pytest.mark.unit
    def test_curva_teorica_e_deterministica(self):
        params = [{"a": 1.2, "b": b, "c": 0.2} for b in np.linspace(-2, 2, 20)]
        primeira = curva_teorica(params, n_calibracao=300)
        segunda = curva_teorica(params, n_calibracao=300)
        assert primeira == segunda
        assert primeira["theta_medio"] == sorted(primeira["theta_medio"])


class TestNotaPorPosicao:

    @pytest.mark.unit
    def test_meio_da_faixa_devolve_a_mediana(self):
        assert nota_por_posicao(0.5, ref_min=400, ref_med=520, ref_max=600) == pytest.approx(520)

    @pytest.mark.unit
    @pytest.mark.parametrize("u", [0.01, 0.2, 0.5, 0.8, 0.99])
    def test_tres_pontos_fica_dentro_da_faixa(self, u):
        nota = nota_por_posicao(u, ref_min=400, ref_med=520, ref_max=600)
        assert 400 <= nota <= 600

    @pytest.mark.unit
    def test_com_quantis_segue_a_distribuicao_real(self):
        quantis = [400, 450, 480, 495, 510, 520, 531, 545, 552, 570, 600]
        assert nota_por_posicao(0.5, ref_min=400, ref_med=520, ref_max=600, quantis=quantis) == 520
        assert nota_por_posicao(0.25, ref_min=400, ref_med=520, ref_max=600, quantis=quantis) == 510
        assert len(quantis) == len(PROBS_QUANTIS)


class TestTabelaComQuantis:

    @pytest.mark.unit
    def test_2024_tem_quantis_e_anos_antigos_nao(self):
        multiano = TabelaReferenciaTRI.carregar_multiano()
        quantis = multiano[2024].obter_quantis("MT", 30)
        assert quantis is not None and len(quantis) == len(PROBS_QUANTIS)
        assert quantis == sorted(quantis)
        assert multiano[2017].obter_quantis("MT", 30) is None

    @pytest.mark.unit
    def test_2023_faixa_e_quantis_coerentes(self):
        """2023 vem dos microdados: a faixa é o mín, a mediana e o máx dos percentis."""
        ano = TabelaReferenciaTRI.carregar_multiano()[2023]
        for area in ("LC", "CH", "CN", "MT"):
            for acertos in range(46):
                ref = ano.obter(area, acertos)
                quantis = ano.obter_quantis(area, acertos)
                assert quantis is not None and len(quantis) == len(PROBS_QUANTIS)
                assert quantis == sorted(quantis)
                assert (ref["tri_min"], ref["tri_med"], ref["tri_max"]) == (
                    quantis[0], quantis[PROBS_QUANTIS.index(0.5)], quantis[-1])
        # Pisos oficiais do ENEM 2023 (nota de quem zera a área).
        assert [ano.obter(a, 0)["tri_med"] for a in ("LC", "CH", "CN", "MT")] == [287.2, 293.5, 323.3, 328.2]

    @pytest.mark.unit
    def test_tabela_de_ano_unico_nao_tem_quantis(self, tabela):
        assert tabela.obter_quantis("MT", 30) is None


class TestMotor:

    @pytest.mark.unit
    def test_sem_curva_mantem_regra_legada(self, tabela):
        ref = tabela.obter("MT", 20)
        nota = ancorar_nota_individual(theta=0.0, area="MT", acertos=20, tabela=tabela)
        assert nota == pytest.approx(ref["tri_med"], abs=0.1)

    @pytest.mark.integration
    def test_nota_tipica_acompanha_a_mediana_de_referencia(self, tabela):
        """Regressão do defeito original: notas coladas no mínimo ou no máximo da faixa."""
        resultado = pipeline_cientifico(
            alunos=_coorte_sintetica(1500, 45, seed=7), area="MT", tabela=tabela,
        )
        assert resultado["diagnostico"]["ancoragem_metodo"] == "condicional_acertos"
        por_acertos: dict[int, list[float]] = {}
        for r in resultado["resultados_alunos"]:
            por_acertos.setdefault(r["acertos"], []).append(r["nota_ancorada"])
        conferidos = 0
        for acertos, notas in por_acertos.items():
            if len(notas) < 20:
                continue
            ref = tabela.obter("MT", acertos)
            # O defeito original deslocava a nota típica em 100-200 pontos;
            # 15 pontos cobre o ruído amostral de grupos de 20+ alunos.
            assert abs(float(np.median(notas)) - ref["tri_med"]) <= 15.0, (
                f"{acertos} acertos: mediana {np.median(notas):.1f} longe de {ref['tri_med']}"
            )
            assert ref["tri_min"] <= min(notas) and max(notas) <= ref["tri_max"]
            conferidos += 1
        assert conferidos >= 15

    @pytest.mark.integration
    def test_escoragem_individual_coincide_com_o_lote(self, tabela):
        alunos = _coorte_sintetica(400, 45, seed=11)
        resultado = pipeline_cientifico(alunos=alunos, area="MT", tabela=tabela)
        for aluno, esperado in list(zip(alunos, resultado["resultados_alunos"]))[:25]:
            individual = estimar_theta_individual(
                respostas=aluno["respostas"],
                parametros_itens=resultado["parametros_itens"],
                area="MT",
                tabela=tabela,
                transformacao=resultado["transformacao"],
            )
            assert individual["nota_ancorada"] == pytest.approx(esperado["nota_ancorada"], abs=0.2)
