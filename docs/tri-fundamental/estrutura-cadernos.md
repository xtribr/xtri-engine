# TRI adaptada — Cadernos Mentoria Marista (Fundamental II)

Documento vivo. Estrutura verificada dos cadernos de 6º a 9º ano, para as
correções que vão chegar. Atualizar a cada aplicação.

Última atualização: 2026-09-18
Fonte: os quatro arquivos `Mentoria Marista — Nº Ano — Caderno ENEM.docx`,
lidos item a item. Dados derivados em `mapa_itens.csv` / `mapa_itens.json`
(240 itens: ano, questão, componente, área, gabarito).

## 1. Composição por área — confirmada

| Ano | Total | LC | MT | CH | CN |
|---|---:|---:|---:|---:|---:|
| 6º | 50 | 20 | 10 | 15 | 5 |
| 7º | 50 | 20 | 10 | 15 | 5 |
| 8º | 50 | 20 | 10 | 15 | 5 |
| 9º | 60 | 20 | 10 | 15 | 15 |

Confere exatamente com a seção "COMPOSIÇÃO DAS PROVAS POR ÁREA" da
`outputs/reference_table_fundamental/Tabela_Referencia_TRI_Fundamental_II_Provisoria.xlsx`.
As tabelas de referência por área já existentes seguem válidas quanto ao
número de itens por área.

## 2. Mapa de posições — ordem real do caderno

As áreas **não** ocupam blocos contíguos. Ciências Humanas está partida em
dois trechos: História e Geografia no meio da prova, Filosofia no fim, depois
de Linguagens. Somar acertos por área exige a posição item a item; agrupar por
faixa contínua produz área errada.

### 6º, 7º e 8º ano (50 itens)

| Posições | Componente | Área |
|---|---|---|
| 1–5 | Ciências Naturais | CN |
| 6–15 | Matemática | MT |
| 16–20 | História | CH |
| 21–25 | Geografia | CH |
| 26–35 | Língua Portuguesa | LC |
| 36–40 | Inglês / Língua Inglesa | LC |
| 41–45 | Arte | LC |
| 46–50 | Filosofia | CH |

### 9º ano (60 itens)

| Posições | Componente | Área |
|---|---|---|
| 1–5 | Biologia | CN |
| 6–10 | Química | CN |
| 11–15 | Física | CN |
| 16–25 | Matemática | MT |
| 26–30 | História | CH |
| 31–35 | Geografia | CH |
| 36–45 | Língua Portuguesa | LC |
| 46–50 | Língua Inglesa | LC |
| 51–55 | Arte | LC |
| 56–60 | Filosofia | CH |

No 9º ano CN aparece desdobrada em Biologia, Química e Física — coerente com a
linha "Biologia + Química + Física | CN | 9º" da aba Premissas.

## 3. Alternativas e parâmetro c

**Todos os 240 itens têm 4 alternativas (A–D)**, verificado item a item nos
quatro cadernos. Nenhum item foge do padrão.

Consequência direta: o `c` de acerto casual é **0.25** (1/4), não o 0.20
(1/5) fixado em `app/motores/cientifico.py` para o ENEM. Rodar esses cadernos
com c = 0.20 subestima o chute e infla a nota na faixa baixa da escala. A
correção precisa receber o número de alternativas como parâmetro.

## 4. Ressalvas de leitura dos arquivos

- **7º ano — cabeçalho "História" ausente no .docx.** O arquivo vai direto de
  Matemática para as questões 16–20 sem título de componente. O gabarito
  comentado confirma que esses cinco itens são de História (procissão do
  Espírito Santo, registro arqueológico como fonte, patrimônio imaterial,
  exclusão das mulheres nos jogos da Grécia antiga, pirâmides egípcias como
  monumentos funerários). O mapa registra 16–20 = História; sem essa correção,
  Matemática apareceria com 15 itens e CH com 10.
- **Gabarito**: extraído da seção "Gabarito comentado" de cada caderno, que
  declara "as respostas foram conferidas pelo enunciado e por resolução
  independente". Os 240 itens têm alternativa registrada, nenhuma lacuna.
- O 6º ano rotula o componente como "Inglês"; 7º, 8º e 9º usam "Língua
  Inglesa". Nenhum caderno traz versão em Espanhol.

## 5. Escala e método

- 6º ano: escala própria, independente. 7º–9º: escala comum, com deslocamento
  vertical −50 / 0 / +50. Limites 200–800, dispersão-base 85.
- A tabela de referência de Fundamental II é **hipótese pré-calibração**,
  declarada como tal na própria planilha: mediana por curva `3p² − 2p³` sobre a
  proporção de acertos, não resultado empírico. Substituir por estimativa
  calibrada assim que houver respostas reais.
- CN no 6º, 7º e 8º ano tem 5 itens. Não existe calibração 2PL/3PL estável com
  5 itens — para essas áreas a saída honesta é escore bruto ancorado no
  envelope, não parâmetro estimado. MT com 10 itens é frágil pelo mesmo motivo.
  No 9º ano, CN com 15 itens já comporta tratamento melhor.

## 6. Pendências

- [ ] Quantidade de estudantes por aplicação (define calibração × ancoragem)
- [ ] Formato de chegada das respostas (CSV, XLSX, cartão-resposta OMR)
- [ ] Confirmar se todas as turmas fazem a mesma versão do caderno
- [ ] Parametrizar o `c` por número de alternativas no motor
- [ ] Decidir a saída: 4 notas por área e/ou nota geral pelo total bruto

## 7. Histórico de aplicações

Nenhuma aplicação registrada até 2026-09-18.

| Data | Ano | Escola/turma | Nº estudantes | Arquivo bruto | Método | Saída |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |
