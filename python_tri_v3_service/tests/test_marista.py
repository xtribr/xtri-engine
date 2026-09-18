"""Testes do motor de correção Marista.

Os valores de referência usados em `test_referencia_*` são valores REAIS da
planilha ``Tabela_Referencia_TRI_Fundamental_II_Provisoria.xlsx`` (aba
"Base longa"), usados como golden values. As marcações de estudantes nos
demais testes são fabricadas para o teste e não representam aplicação real.
"""

from __future__ import annotations

import pytest

from app.marista.correcao import corrigir, corrigir_aluno
from app.marista.estrutura import obter_prova
from app.marista.escala_oficial import adaptar, carregar_escala, tabelas_da_prova
from app.marista.referencia import ParametrosEnvelope, faixa, tabela_por_area
from app.marista.respostas import RespostaAluno, ler_respostas

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Envelope de referência
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "acertos,n_itens,minimo,mediana,maximo",
    [
        # 6º ano, GERAL, 50 itens (deslocamento 0)
        (0, 50, 200.0, 200.0, 200.0),
        (1, 50, 200.0, 200.7, 203.7),
        (2, 50, 200.0, 202.8, 208.6),
        (5, 50, 203.1, 216.8, 230.5),
        (6, 50, 207.7, 223.8, 239.9),
        # 6º ano, LC, 20 itens
        (1, 20, 200.0, 204.4, 215.8),
        (2, 20, 200.0, 216.8, 238.4),
        (3, 20, 205.8, 236.5, 267.2),
    ],
)
def test_referencia_reproduz_planilha(acertos, n_itens, minimo, mediana, maximo):
    f = faixa(acertos, n_itens, ParametrosEnvelope())
    assert f.minimo == pytest.approx(minimo, abs=0.1)
    assert f.mediana == pytest.approx(mediana, abs=0.1)
    assert f.maximo == pytest.approx(maximo, abs=0.1)


def test_referencia_extremos_sem_amplitude():
    """Nos extremos o envelope colapsa: 4p(1-p) vale zero em p=0 e p=1."""
    for n in (12, 15, 18, 60):
        assert faixa(0, n, ParametrosEnvelope()).amplitude == 0.0
        assert faixa(n, n, ParametrosEnvelope()).amplitude == 0.0
        assert faixa(0, n, ParametrosEnvelope()).mediana == 200.0
        assert faixa(n, n, ParametrosEnvelope()).mediana == 800.0


def test_referencia_prova_curta_tem_envelope_mais_largo():
    """√(10/n): quanto menos itens, maior a incerteza no mesmo p."""
    curta = faixa(6, 12, ParametrosEnvelope()).amplitude
    longa = faixa(30, 60, ParametrosEnvelope()).amplitude
    assert curta > longa


def test_referencia_acertos_fora_da_faixa():
    with pytest.raises(ValueError):
        faixa(13, 12, ParametrosEnvelope())


def test_tabela_por_area_do_ensino_medio():
    tabs = tabela_por_area({"LC": 15, "MT": 15, "CH": 12, "CN": 18})
    assert {a: len(t) - 1 for a, t in tabs.items()} == {
        "LC": 15, "MT": 15, "CH": 12, "CN": 18, "GERAL": 60
    }


# --------------------------------------------------------------------------
# Estrutura das provas
# --------------------------------------------------------------------------

def test_composicao_ensino_medio():
    for nivel in ("1a serie", "2a serie"):
        prova = obter_prova("EM", nivel)
        assert prova.n_itens == 60
        assert prova.composicao == {"LC": 15, "MT": 15, "CH": 12, "CN": 18}
        assert prova.alternativas == 5
        assert prova.chute_c == pytest.approx(0.20)


def test_composicao_fundamental():
    for nivel in ("6o ano", "7o ano", "8o ano"):
        prova = obter_prova("EF", nivel)
        assert prova.n_itens == 50
        assert prova.composicao == {"LC": 20, "MT": 10, "CH": 15, "CN": 5}
        assert prova.alternativas == 4
        assert prova.chute_c == pytest.approx(0.25)


def test_areas_do_fundamental_nao_sao_contiguas():
    """Filosofia (46-50) é CH e vem depois de Linguagens: faixa contínua erraria a área."""
    prova = obter_prova("EF", "6o ano")
    area = prova.area_por_questao
    assert [area[q] for q in range(46, 51)] == ["CH"] * 5
    assert [area[q] for q in range(36, 46)] == ["LC"] * 10
    assert [area[q] for q in range(16, 26)] == ["CH"] * 10


def test_nivel_inexistente():
    with pytest.raises(KeyError):
        obter_prova("EM", "3a serie")


# --------------------------------------------------------------------------
# Leitura de marcações
# --------------------------------------------------------------------------

def _csv_respostas(tmp_path, n_itens, letras, nome="turma.csv"):
    caminho = tmp_path / nome
    cabecalho = ["aluno_id", "nome", "turma"] + [str(i) for i in range(1, n_itens + 1)]
    linha = ["A1", "Fulano de Teste", "1º A"] + list(letras)
    caminho.write_text(",".join(cabecalho) + "\n" + ",".join(linha) + "\n", encoding="utf-8")
    return caminho


def test_leitura_recusa_numero_errado_de_questoes(tmp_path):
    prova = obter_prova("EM", "1a serie")
    arquivo = _csv_respostas(tmp_path, 50, ["A"] * 50)
    with pytest.raises(ValueError, match="colunas de questão"):
        ler_respostas(arquivo, prova)


def test_leitura_classifica_branco_multipla_e_invalida(tmp_path):
    prova = obter_prova("EF", "6o ano")
    letras = ["A"] * 50
    letras[0] = ""       # em branco
    letras[1] = "*"      # marcação múltipla
    letras[2] = "E"      # inválida numa prova de 4 alternativas
    arquivo = _csv_respostas(tmp_path, 50, letras)
    (aluno,) = ler_respostas(arquivo, prova)
    assert aluno.em_branco == [1]
    assert aluno.multipla == [2]
    assert aluno.invalidas == [(3, "E")]
    assert aluno.n_respondidas == 47


# --------------------------------------------------------------------------
# Correção
# --------------------------------------------------------------------------

def _resposta(prova, marcacoes):
    return RespostaAluno(identificacao={"aluno_id": "T1"}, marcacoes=dict(marcacoes))


def test_aluno_gabarito_perfeito_recebe_teto_por_area():
    prova = obter_prova("EM", "1a serie")
    tabs = tabela_por_area(prova.composicao)
    resultado = corrigir_aluno(_resposta(prova, prova.gabarito), prova, tabs)
    for area, r in resultado.por_area.items():
        assert r.acertos == prova.composicao[area]
        assert r.nota_mediana == 800.0
    assert resultado.geral.acertos == 60
    assert resultado.confiavel


def test_aluno_sem_nenhum_acerto_recebe_piso():
    prova = obter_prova("EM", "1a serie")
    tabs = tabela_por_area(prova.composicao)
    erradas = {q: ("A" if g != "A" else "B") for q, g in prova.gabarito.items()}
    resultado = corrigir_aluno(_resposta(prova, erradas), prova, tabs)
    assert resultado.geral.acertos == 0
    for r in resultado.por_area.values():
        assert r.nota_mediana == 200.0
        assert r.erros == r.n_itens


def test_acertos_sao_contados_na_area_certa():
    """Acerta só Filosofia (46-50, CH) numa prova de 6º ano."""
    prova = obter_prova("EF", "6o ano")
    tabs = tabela_por_area(prova.composicao)
    gab = prova.gabarito
    marcacoes = {
        q: (gab[q] if 46 <= q <= 50 else ("A" if gab[q] != "A" else "B"))
        for q in gab
    }
    resultado = corrigir_aluno(_resposta(prova, marcacoes), prova, tabs)
    assert resultado.por_area["CH"].acertos == 5
    assert resultado.por_area["LC"].acertos == 0
    assert resultado.geral.acertos == 5


def test_branco_nao_conta_como_erro_e_rebaixa_confiabilidade():
    prova = obter_prova("EM", "1a serie")
    tabs = tabela_por_area(prova.composicao)
    marcacoes = dict(prova.gabarito)
    for q in range(1, 6):          # 5 em branco em LC (15 itens) = 33%
        marcacoes[q] = None
    resposta = RespostaAluno(
        identificacao={"aluno_id": "T2"}, marcacoes=marcacoes, em_branco=list(range(1, 6))
    )
    resultado = corrigir_aluno(resposta, prova, tabs)
    lc = resultado.por_area["LC"]
    assert lc.acertos == 10
    assert lc.em_branco == 5
    assert lc.erros == 0           # branco não virou erro
    assert not lc.confiavel
    assert "em branco" in lc.observacao


def test_nivel_fora_de_escopo_recusa_correcao():
    prova = obter_prova("EF", "9o ano")
    with pytest.raises(ValueError, match="fora do escopo"):
        corrigir([_resposta(prova, prova.gabarito)], prova)


def test_turma_pequena_avisa_e_nao_calcula_discriminacao():
    prova = obter_prova("EM", "2a serie")
    respostas = [_resposta(prova, prova.gabarito) for _ in range(3)]
    resultado = corrigir(respostas, prova)
    assert resultado.n_alunos == 3
    assert all(i.discriminacao is None for i in resultado.itens)
    assert any("amostra insuficiente" in a for a in resultado.avisos)
    assert any("não nota oficial do ENEM" in a for a in resultado.avisos)


def test_estatistica_de_item_cobre_todos_os_itens():
    prova = obter_prova("EM", "1a serie")
    resultado = corrigir([_resposta(prova, prova.gabarito)], prova)
    assert len(resultado.itens) == 60
    assert {i.area for i in resultado.itens} == {"LC", "MT", "CH", "CN"}
    assert all(i.taxa_acerto == 1.0 for i in resultado.itens)


# --------------------------------------------------------------------------
# Escala oficial do ENEM adaptada
# --------------------------------------------------------------------------

@pytest.mark.parametrize("area", ["LC", "CH", "CN", "MT"])
def test_adaptacao_reproduz_a_tabela_oficial(area):
    """Adaptar para o próprio nº de itens do ENEM devolve a tabela oficial."""
    escala = carregar_escala(area, 2024)
    faixas = adaptar(escala, escala.n_itens, alargar_envelope=False)
    for ponto in escala.pontos:
        f = faixas[ponto.acertos]
        assert f.mediana == pytest.approx(ponto.mediana, abs=0.05)
        assert f.minimo == pytest.approx(ponto.minimo, abs=0.05)
        assert f.maximo == pytest.approx(ponto.maximo, abs=0.05)


@pytest.mark.parametrize("n_itens", [5, 10, 12, 15, 18, 20])
def test_extremos_da_area_sao_preservados(n_itens):
    """Zero acerto e prova inteira caem nos extremos oficiais, qualquer que seja n."""
    escala = carregar_escala("LC", 2024)
    faixas = adaptar(escala, n_itens)
    assert faixas[0].mediana == pytest.approx(escala.pontos[0].mediana, abs=0.05)
    assert faixas[-1].mediana == pytest.approx(escala.pontos[-1].mediana, abs=0.05)


def test_prova_curta_tem_envelope_mais_largo():
    escala = carregar_escala("CN", 2024)
    curta = adaptar(escala, 5)
    longa = adaptar(escala, 18)
    assert curta[2].amplitude > longa[9].amplitude


def test_piso_da_area_e_o_menor_minimo_da_curva():
    """Em 2024, MT tem pontos com mínimo abaixo do mínimo de zero acerto."""
    mt = carregar_escala("MT", 2024)
    assert mt.piso < mt.pontos[0].minimo
    assert mt.piso == pytest.approx(min(p.minimo for p in mt.pontos))
    lc = carregar_escala("LC", 2024)
    assert lc.piso == pytest.approx(lc.pontos[0].minimo)


def test_envelope_nao_ultrapassa_os_limites_da_area():
    escala = carregar_escala("MT", 2024)
    for f in adaptar(escala, 10):
        assert f.minimo >= escala.piso - 0.05
        assert f.maximo <= escala.teto + 0.05


def test_ano_ou_area_inexistente():
    with pytest.raises(KeyError):
        carregar_escala("LC", 1998)
    with pytest.raises(KeyError):
        carregar_escala("XX", 2024)


def test_correcao_ancora_na_escala_oficial_por_padrao():
    prova = obter_prova("EM", "1a serie")
    escalas = {a: carregar_escala(a, 2024) for a in prova.composicao}
    resultado = corrigir([_resposta(prova, prova.gabarito)], prova)
    aluno = resultado.alunos[0]
    for area, escala in escalas.items():
        assert aluno.por_area[area].nota_mediana == pytest.approx(escala.pontos[-1].mediana, abs=0.05)
    assert any("ENEM 2024" in a for a in resultado.avisos)
    assert any("não nota oficial do ENEM" in a for a in resultado.avisos)


def test_tabelas_da_prova_cobre_areas_e_geral():
    tabs = tabelas_da_prova({"LC": 15, "MT": 15, "CH": 12, "CN": 18}, 2024)
    assert {a: len(t) - 1 for a, t in tabs.items()} == {
        "LC": 15, "MT": 15, "CH": 12, "CN": 18, "GERAL": 60
    }
