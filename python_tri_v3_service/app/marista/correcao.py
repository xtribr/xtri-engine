"""Correção das provas Marista: acertos por área ancorados no envelope.

O que este motor faz:

1. Compara marcação a marcação com o gabarito do nível, seguindo o mapa de
   itens (as áreas não são faixas contíguas no Fundamental II).
2. Conta acertos por área e no total bruto.
3. Ancora cada contagem no envelope de referência da área, devolvendo
   mínimo, mediana e máximo.

O que ele NÃO faz, de propósito:

- Não estima parâmetros de item nem theta. Com 12 a 38 estudantes por nível e
  áreas de 12 a 20 itens, calibração 2PL/3PL não é identificável de forma
  estável; o resultado honesto é acerto bruto ancorado.
- Não posiciona o aluno dentro do envelope. Posicionar exige calibração ou um
  critério externo de dificuldade; sem isso, a mediana é a estimativa central e
  [mínimo, máximo] é a incerteza declarada.
- Não converte item em branco em erro silenciosamente: em branco, marcação
  múltipla e letra inválida entram no relatório e rebaixam a confiabilidade.

A nota é PROVISÓRIA enquanto o envelope for hipótese pré-calibração.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev

from app.marista.escala_oficial import ANO_PADRAO, tabelas_da_prova
from app.marista.estrutura import Prova
from app.marista.referencia import FaixaReferencia, ParametrosEnvelope, tabela_por_area
from app.marista.respostas import RespostaAluno

LIMITE_BRANCOS_CONFIAVEL = 0.10


@dataclass(frozen=True)
class ResultadoArea:
    area: str
    n_itens: int
    acertos: int
    erros: int
    em_branco: int
    multipla: int
    invalidas: int
    nota_minima: float
    nota_mediana: float
    nota_maxima: float
    confiavel: bool
    observacao: str = ""

    @property
    def percentual_acerto(self) -> float:
        return round(self.acertos / self.n_itens, 4) if self.n_itens else 0.0


@dataclass
class ResultadoAluno:
    identificacao: dict[str, str]
    bloco: str
    nivel: str
    por_area: dict[str, ResultadoArea]
    geral: ResultadoArea
    itens_corretos: list[int] = field(default_factory=list)

    @property
    def rotulo(self) -> str:
        for chave in ("aluno_id", "id", "matricula", "nome"):
            if self.identificacao.get(chave):
                return str(self.identificacao[chave])
        return "(sem identificação)"

    @property
    def confiavel(self) -> bool:
        return self.geral.confiavel and all(r.confiavel for r in self.por_area.values())


@dataclass
class EstatisticaItem:
    questao: int
    area: str
    componente: str
    gabarito: str
    respondentes: int
    acertos: int
    taxa_acerto: float
    discriminacao: float | None
    observacao: str = ""


@dataclass
class ResultadoTurma:
    bloco: str
    nivel: str
    n_alunos: int
    parametros: ParametrosEnvelope
    alunos: list[ResultadoAluno]
    itens: list[EstatisticaItem]
    avisos: list[str] = field(default_factory=list)


def _classificar(
    area: str,
    n_itens: int,
    acertos: int,
    em_branco: int,
    multipla: int,
    invalidas: int,
    faixa: FaixaReferencia,
) -> ResultadoArea:
    pendencias = []
    confiavel = True
    if multipla:
        pendencias.append(f"{multipla} marcação(ões) múltipla(s)")
        confiavel = False
    if invalidas:
        pendencias.append(f"{invalidas} marcação(ões) inválida(s)")
        confiavel = False
    if em_branco:
        pendencias.append(f"{em_branco} em branco")
        if n_itens and em_branco / n_itens > LIMITE_BRANCOS_CONFIAVEL:
            confiavel = False
    return ResultadoArea(
        area=area,
        n_itens=n_itens,
        acertos=acertos,
        erros=n_itens - acertos - em_branco - multipla - invalidas,
        em_branco=em_branco,
        multipla=multipla,
        invalidas=invalidas,
        nota_minima=faixa.minimo,
        nota_mediana=faixa.mediana,
        nota_maxima=faixa.maximo,
        confiavel=confiavel,
        observacao="; ".join(pendencias),
    )


def corrigir_aluno(
    resposta: RespostaAluno,
    prova: Prova,
    tabelas: dict[str, list[FaixaReferencia]],
) -> ResultadoAluno:
    """Corrige um estudante contra o gabarito e ancora por área."""
    gabarito = prova.gabarito
    area_de = prova.area_por_questao
    composicao = prova.composicao

    acertos = {a: 0 for a in composicao}
    brancos = {a: 0 for a in composicao}
    multiplas = {a: 0 for a in composicao}
    invalidas = {a: 0 for a in composicao}
    corretos: list[int] = []

    branco_set = set(resposta.em_branco)
    multipla_set = set(resposta.multipla)
    invalida_set = {q for q, _ in resposta.invalidas}

    for questao, area in area_de.items():
        marcada = resposta.marcacoes.get(questao)
        if questao in multipla_set:
            multiplas[area] += 1
        elif questao in invalida_set:
            invalidas[area] += 1
        elif marcada is None or questao in branco_set:
            brancos[area] += 1
        elif marcada == gabarito[questao]:
            acertos[area] += 1
            corretos.append(questao)

    por_area = {
        area: _classificar(
            area,
            n_itens,
            acertos[area],
            brancos[area],
            multiplas[area],
            invalidas[area],
            tabelas[area][acertos[area]],
        )
        for area, n_itens in composicao.items()
    }
    total = sum(acertos.values())
    geral = _classificar(
        "GERAL",
        prova.n_itens,
        total,
        sum(brancos.values()),
        sum(multiplas.values()),
        sum(invalidas.values()),
        tabelas["GERAL"][total],
    )
    return ResultadoAluno(
        identificacao=resposta.identificacao,
        bloco=prova.bloco,
        nivel=prova.nivel,
        por_area=por_area,
        geral=geral,
        itens_corretos=sorted(corretos),
    )


def _ponto_bisserial(acertou: list[int], totais: list[int]) -> float | None:
    """Correlação ponto-bisserial item-total. None quando não é calculável."""
    if len(acertou) < 2 or len(set(acertou)) < 2:
        return None
    desvio = pstdev(totais)
    if desvio == 0:
        return None
    m1 = mean([t for a, t in zip(acertou, totais) if a == 1])
    m0 = mean([t for a, t in zip(acertou, totais) if a == 0])
    p = sum(acertou) / len(acertou)
    return round(((m1 - m0) / desvio) * ((p * (1 - p)) ** 0.5), 4)


def corrigir(
    respostas: list[RespostaAluno],
    prova: Prova,
    ano_referencia: int = ANO_PADRAO,
    parametros: ParametrosEnvelope | None = None,
    tabelas: dict[str, list[FaixaReferencia]] | None = None,
    minimo_para_discriminacao: int = 30,
) -> ResultadoTurma:
    """Corrige um conjunto de estudantes do mesmo nível.

    Por padrão ancora na escala oficial do ENEM do ``ano_referencia``, adaptada
    ao número de itens de cada área. Passar ``parametros`` troca para a hipótese
    pré-calibração de Fundamental II (curva sintética 200–800); ``tabelas``
    permite injetar um envelope já pronto.
    """
    if not prova.escopo_correcao:
        raise ValueError(
            f"{prova.bloco}/{prova.nivel} está marcado como fora do escopo de correção no mapa de itens"
        )
    if not respostas:
        raise ValueError("nenhuma resposta para corrigir")

    params = parametros
    if tabelas is None:
        if params is not None:
            tabelas = tabela_por_area(prova.composicao, params)
            origem = "hipótese pré-calibração (curva sintética 200–800)"
        else:
            tabelas = tabelas_da_prova(prova.composicao, ano_referencia)
            origem = f"escala oficial do ENEM {ano_referencia}, adaptada por proporção de acertos"
    else:
        origem = "envelope fornecido pelo chamador"
    alunos = [corrigir_aluno(r, prova, tabelas) for r in respostas]

    totais = [a.geral.acertos for a in alunos]
    itens: list[EstatisticaItem] = []
    for item in prova.itens:
        marcou = [
            r for r in respostas if r.marcacoes.get(item.questao) is not None
        ]
        acertou_flags = [1 if a.geral and item.questao in a.itens_corretos else 0 for a in alunos]
        n_resp = len(marcou)
        n_acertos = sum(acertou_flags)
        disc = (
            _ponto_bisserial(acertou_flags, totais)
            if len(alunos) >= minimo_para_discriminacao
            else None
        )
        itens.append(
            EstatisticaItem(
                questao=item.questao,
                area=item.area,
                componente=item.componente,
                gabarito=item.gabarito,
                respondentes=n_resp,
                acertos=n_acertos,
                taxa_acerto=round(n_acertos / len(alunos), 4) if alunos else 0.0,
                discriminacao=disc,
                observacao=(
                    ""
                    if disc is not None
                    else f"discriminação não calculada: menos de {minimo_para_discriminacao} estudantes"
                ),
            )
        )

    avisos: list[str] = [
        f"Ancoragem: {origem}.",
        "Nota PEDAGÓGICA, não nota oficial do ENEM: os itens desta prova não foram "
        "calibrados e a escala foi adaptada de outra quantidade de itens.",
    ]
    if len(alunos) < 100:
        avisos.append(
            f"{len(alunos)} estudantes: amostra insuficiente para calibrar parâmetros de item. "
            "O resultado é acerto bruto ancorado, não TRI calibrada."
        )
    nao_confiaveis = [a.rotulo for a in alunos if not a.confiavel]
    if nao_confiaveis:
        avisos.append(
            f"{len(nao_confiaveis)} estudante(s) com marcações em branco, múltiplas ou inválidas "
            f"acima do tolerado: {', '.join(nao_confiaveis[:10])}"
            + ("…" if len(nao_confiaveis) > 10 else "")
        )
    return ResultadoTurma(
        bloco=prova.bloco,
        nivel=prova.nivel,
        n_alunos=len(alunos),
        parametros=params or ParametrosEnvelope(),
        alunos=alunos,
        itens=itens,
        avisos=avisos,
    )
