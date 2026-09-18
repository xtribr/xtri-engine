# TRI adaptada — Projeto Marista

Documento vivo. Estrutura verificada das provas, para as correções que vão
chegar. Atualizar a cada aplicação.

Última atualização: 2026-09-18

## 0. Escopo

Dois blocos entram na correção:

| Bloco | Níveis | Situação |
|---|---|---|
| **Fundamental II** | 6º, 7º e 8º ano | corrigidos |
| | 9º ano | **fora da correção** (estrutura mantida abaixo só como referência) |
| **Ensino Médio** | 1ª e 2ª série | corrigidos |

Fora do escopo por decisão explícita: o caderno reorganizado de 9º ano com
45 questões (`GABARITO COMENTADO - 9o ANO - 45 QUESTOES.pdf`).

Os dados item a item estão em `mapa_itens.csv`: bloco, nível, questão,
componente, área, nº de alternativas, gabarito, escopo e fonte.

## 1. Bloco Fundamental II — 6º, 7º e 8º ano

Fonte: `Mentoria Marista — Nº Ano — Caderno ENEM.docx`, lidos item a item.

**50 itens, 4 alternativas (A–D).** Composição idêntica nos três anos:
LC 20, MT 10, CH 15, CN 5.

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

As áreas **não** são blocos contíguos: CH está partida entre História/Geografia
(16–25) e Filosofia (46–50), esta depois de Linguagens. Somar acertos por faixa
contínua produz área errada — o somatório tem que seguir o mapa item a item.

Ressalva: no 7º ano o cabeçalho "História" não existe no .docx; os itens 16–20
foram identificados como História pelo gabarito comentado. Sem essa correção,
Matemática apareceria com 15 itens e CH com 10.

## 2. Bloco Ensino Médio — 1ª e 2ª série

Fonte: `CADERNO DE PROVA - PRIMEIRA/SEGUNDA SÉRIE.pdf` (faixas por bloco) e
`GABARITO - 1a/2a SERIE - LETRAS.pdf` (chave conferida contra a versão final
do caderno).

**60 itens, 5 alternativas (A–E).** Mesma composição nas duas séries:
LC 15, MT 15, CN 18, CH 12.

| Posições | Bloco do caderno | Área |
|---|---|---|
| 01–15 | Língua Portuguesa | LC |
| 16–30 | Matemática | MT |
| 31–48 | Biologia, Física e Química | CN |
| 49–60 | História e Geografia | CH |

Aqui as áreas são contíguas. Os 60 gabaritos de cada série foram lidos sem
lacuna nem duplicidade.

## 3. Parâmetro c por bloco

| Bloco | Alternativas | c |
|---|---:|---:|
| Fundamental II (6º–8º) | 4 | **0.25** |
| Ensino Médio (1ª–2ª) | 5 | **0.20** |

O motor científico fixa c = 0.20 em `app/motores/cientifico.py`, valor do ENEM
(5 alternativas). Serve para o Ensino Médio; para o Fundamental II subestima o
chute e infla a nota na faixa baixa. O `c` precisa virar parâmetro por prova.

## 4. Tabelas de referência — incompatibilidade a resolver

As tabelas de referência seguem acertos por área, e **nenhum dos dois blocos
tem a composição para a qual as tabelas existentes foram construídas**:

| Composição | LC | MT | CH | CN |
|---|---:|---:|---:|---:|
| Tabela Fundamental II provisória (6º–8º) | 20 | 10 | 15 | 5 |
| Prova real 6º–8º | 20 | 10 | 15 | 5 | ✔ compatível |
| Tabela Fundamental II provisória (9º) | 20 | 10 | 15 | 15 |
| Prova real 1ª/2ª série | 15 | 15 | 12 | 18 | ✘ sem tabela correspondente |

O Fundamental II está coberto. Para o Ensino Médio é preciso construir o
envelope por área para 15/15/12/18 — as tabelas do ENEM (45 itens por área)
também não servem diretamente.

Lembrete que vale para os dois blocos: a tabela de Fundamental II é **hipótese
pré-calibração** (mediana por curva `3p² − 2p³` sobre a proporção de acertos),
declarada como tal na própria planilha. Vira estimativa só depois de aplicação
real.

## 5. Cartões-resposta — estado atual

Em `projeto marista/GABARITOS/` há 21 PDFs de cartão-resposta, um aluno por
página, com nome e turma impressos. **Estão em branco** — são os cartões a
serem aplicados, não as marcações dos alunos. Os cartões preenchidos chegam
depois.

| Bloco | Turmas | Alunos | Grade do cartão |
|---|---|---:|---|
| EF 6º, 7º e 8º ano | 3 | 12 | 50 questões, A–D |
| EF 5º ano (fora do escopo) | 1 | 6 | 50 questões, A–D |
| EF 9º ano (fora do escopo) | EF9AM–EF9GV (7) | 92 | 60 questões, A–E |
| EM 1ª série | 1ºA–1ºE (5) | 38 | 60 questões, A–E |
| EM 2ª série | 2ºA–2ºE (5) | 34 | 60 questões, A–E |

Nomes e matrículas dos estudantes ficam **fora** deste repositório.

Pontos a confirmar nos cartões:
- Os cartões de 1ª e 2ª série trazem o título "PROVA - 9º" no cabeçalho, mas a
  grade é de 60 itens e existe gabarito próprio de 60 para cada série. Parece
  resíduo de template — confirmar.
- Todo cartão tem campos "MODELO DE PROVA 2 3 4" e "LÍNGUA ESTRANGEIRA:
  INGLÊS / ESPANHOL". Se houver mais de um modelo de caderno em circulação,
  cada modelo precisa da sua própria chave, e a escolha de língua muda quais
  itens contam para o aluno.

## 6. Pendências antes da primeira correção

- [ ] Cartões preenchidos (digitalizados) — formato, resolução e se vêm por turma
- [ ] Gabarito oficial dos cadernos de 6º, 7º e 8º ano (hoje a chave vem do
      gabarito comentado dentro do .docx)
- [ ] Confirmar se há mais de um modelo de prova por nível
- [ ] Envelope por área para a composição 15/15/12/18 (Ensino Médio)
- [ ] Parametrizar o `c` por número de alternativas no motor
- [ ] Decidir a saída: nota por área e/ou nota geral pelo total bruto

## 7. Motor de correção

Código em `python_tri_v3_service/app/marista/`:

| Módulo | Responsabilidade |
|---|---|
| `estrutura.py` | Carrega o mapa de itens (`app/referencias/marista_mapa_itens.csv`), monta a prova de cada nível e valida numeração, área e gabarito |
| `referencia.py` | Gera o envelope por área pelo método documentado na planilha de Fundamental II |
| `respostas.py` | Lê as marcações dos estudantes e classifica em branco, marcação múltipla e letra inválida |
| `correcao.py` | Conta acertos por área, ancora no envelope e calcula estatística de item |
| `../scripts/corrigir_marista.py` | CLI |

```bash
cd python_tri_v3_service
python -m app.scripts.corrigir_marista \
    --bloco EM --nivel "1a serie" \
    --respostas marcacoes_1a.csv \
    --saida-alunos notas_1a.csv --saida-itens itens_1a.csv
```

O arquivo de marcações tem uma linha por estudante: colunas de identificação
(`aluno_id`, `nome`, `turma`, `matricula`) e uma coluna por questão, com
cabeçalho `1`…`60`, `q1`… ou `questao_1`…

### Ancoragem: escala oficial do ENEM adaptada

A nota não sai de curva inventada. Para cada área, a proporção de acertos na
prova curta é lida na **curva real do ENEM** daquela área:

    p    = acertos / nº de itens da prova
    x    = p × nº de itens da prova do ENEM na mesma área
    nota = interpolação linear da tabela oficial em x

Fonte: `app/referencias/tri_tabela_referencia_multiano.json`, a mesma tabela que
o fluxo de CSV do frontend usa. Ano padrão **2024** — o mais recente disponível
e o único, junto com 2009, 2015, 2017, 2019, 2021 e 2022, sem mediana
não-monotônica em nenhuma das quatro áreas.

Extremos de cada área em 2024, preservados exatamente na adaptação:

| Área | Itens no ENEM | Zero acerto | Prova inteira |
|---|---:|---:|---:|
| LC | 45 | 301,4 | 795,8 |
| CH | 45 | 283,8 | 819,7 |
| CN | 44 | 342,1 | 867,2 |
| MT | 45 | 371,0 | 961,9 |

O envelope [mínimo, máximo] é o envelope oficial **alargado por √(N/n)**: a
mesma proficiência medida com menos itens é medida com menos informação, e o
intervalo tem que refletir isso. CN do 6º ano, com 5 itens contra 44 do ENEM,
alarga por 2,97; LC do 6º ano, com 20 contra 45, alarga por 1,50. O alargamento
é limitado ao piso e ao teto da própria área.

Validação: adaptar uma área para o próprio número de itens do ENEM reproduz a
tabela oficial valor a valor (diferença máxima 0,000 nas quatro áreas).

Achado nos dados oficiais de 2024: a coluna mínima **não** é monotônica em CN
(2 pontos) e em MT (7 pontos) — em Matemática, o mínimo com 6 acertos é 342,8,
abaixo dos 369,4 de zero acerto. É dado oficial, não erro: na TRI um padrão de
respostas incoerente com mais acertos pode pontuar menos. Por isso o piso da
área é o menor mínimo de toda a curva, não o valor em zero acerto.

A tabela gerada para os dois blocos está em `tabela_referencia_marista.csv`
(565 linhas), com a coluna `ancoragem = ENEM_2024_ADAPTADO_POR_PROPORCAO`.

A hipótese pré-calibração de Fundamental II (curva sintética 200–800) continua
disponível em `referencia.py` e é usada passando `parametros=` para `corrigir`.

### Leitura pedagógica: o que essa escala significa

Aplicar a escala do ENEM a uma prova de 6º ano é uma decisão pedagógica, não
estatística. Um estudante de 6º ano que acerta 5 dos 10 itens de Matemática
aparece em 693,3 — que na escala do ENEM é desempenho alto de fim de Ensino
Médio. A nota diz "metade dos itens desta prova", não "equivale a um
concluinte do ENEM com 693". O relatório para as famílias precisa dizer isso
explicitamente, ou a escala será lida como comparação direta com o ENEM.

### O que o motor deliberadamente não faz

- **Não estima parâmetros de item nem theta.** Com 12 a 38 estudantes por nível
  e áreas de 12 a 20 itens, calibração 2PL/3PL não é identificável de forma
  estável. A saída honesta é acerto bruto ancorado.
- **Não posiciona o aluno dentro do envelope.** Isso exigiria calibração ou um
  critério externo de dificuldade. A mediana é a estimativa central e
  [mínimo, máximo] é a incerteza declarada, não enfeite.
- **Não transforma item em branco em erro.** Branco, marcação múltipla e letra
  inválida entram no relatório e rebaixam a confiabilidade do resultado daquela
  área (`confiavel = nao`), com a razão escrita na coluna de observação.
- **Não corrige nível fora de escopo.** Pedir a correção do 9º ano levanta erro.

A estatística de item (taxa de acerto observada e ponto-bisserial) sai junto;
a discriminação só é calculada com 30 estudantes ou mais, e abaixo disso a
coluna vem vazia com a razão declarada.

Cobertura: `tests/test_marista.py`, 25 testes — envelope conferido contra
valores reais da planilha, composição por área dos dois blocos, contagem na
área certa quando as faixas não são contíguas, tratamento de branco/múltipla/
inválida e recusa do nível fora de escopo.

## 8. Histórico de aplicações

Nenhuma correção executada até 2026-09-18.

| Data | Bloco/nível | Turma | Nº alunos | Arquivo bruto | Método | Saída |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |
