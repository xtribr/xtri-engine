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

## 7. Histórico de aplicações

Nenhuma correção executada até 2026-09-18.

| Data | Bloco/nível | Turma | Nº alunos | Arquivo bruto | Método | Saída |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |
