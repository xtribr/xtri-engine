# TRI adaptada — Fundamental II · 6º ano

Documento vivo. Registra a estrutura da prova para as correções que vão chegar.
Atualizar a cada aplicação; nunca preencher lacuna com estimativa.

Última atualização: 2026-09-18

## 1. Identificação da prova

| Campo | Valor | Origem |
|---|---|---|
| Ano escolar | 6º ano (Fundamental II) | informado |
| Total de itens | 50 | informado |
| Organização do caderno | agrupado por área, na ordem **Humanas → Natureza → Linguagens → Matemática** | informado |
| Escala | própria do 6º ano, **independente** da escala comum 7º–9º | `outputs/reference_table_fundamental/Tabela_Referencia_TRI_Fundamental_II_Provisoria.xlsx`, aba Premissas |
| Limites da escala | 200 a 800 | idem |

## 2. Composição por componente

Distribuição informada (imagem do documento da escola):

| Componente | Itens | Área |
|---|---:|---|
| História | 5 | CH |
| Geografia | 5 | CH |
| Filosofia | 5 | CH |
| Ciências | 5 | CN |
| Língua Portuguesa | 10 | LC |
| Inglês ou Espanhol | 5 | LC |
| Arte | 5 | LC |
| Matemática | 10 | MT |
| **Total** | **50** | |

## 3. Agregação por área

| Área | Código | Itens | Componentes |
|---|---|---:|---|
| Ciências Humanas | CH | 15 | História + Geografia + Filosofia |
| Ciências da Natureza | CN | 5 | Ciências |
| Linguagens e Códigos | LC | 20 | Língua Portuguesa + Inglês/Espanhol + Arte |
| Matemática | MT | 10 | Matemática |
| Geral | GERAL | 50 | total bruto |

Essa agregação **não é inferência nova**: é exatamente a unificação de componentes
já registrada na aba Premissas da tabela provisória de Fundamental II
("UNIFICAÇÃO DOS COMPONENTES") e a composição já prevista para o 6º ano
("COMPOSIÇÃO DAS PROVAS POR ÁREA": 50 itens = LC 20, MT 10, CH 15, CN 5).

## 4. Mapa de posições dos itens — A CONFIRMAR

Derivado da ordem informada das áreas somada às contagens acima:

| Posições | Área | Itens |
|---|---|---:|
| 1–15 | CH | 15 |
| 16–20 | CN | 5 |
| 21–40 | LC | 20 |
| 41–50 | MT | 10 |

**Status: inferido, não confirmado.** A ordem interna dos componentes dentro de cada
área (ex.: História antes ou depois de Geografia) não foi informada e não está
assumida aqui. Conferir contra o caderno antes da primeira correção — um
deslocamento de posição corrompe silenciosamente a calibração.

## 5. Método de correção — decisões em aberto

O envelope de referência disponível hoje é **hipótese pré-calibração**, declarada
na própria planilha: *"não representa nota TRI oficial nem resultado empírico
observado"*. Método de construção da hipótese: mediana por curva suave
`3p² − 2p³` entre 200 e 800 sobre a proporção de acertos, com deslocamento
vertical por ano e envelope de dispersão-base 85. Após aplicações reais, esses
limites devem ser substituídos por estimativas calibradas com respostas dos
estudantes.

Restrições psicométricas que condicionam a escolha do motor:

- **CN tem 5 itens e MT tem 10.** Calibração 2PL/3PL por área nesses tamanhos não
  se sustenta: os parâmetros não são identificáveis com estabilidade. A regra
  inviolável de "uma calibração por simulado × área" continua valendo, mas com
  5 itens a saída honesta é escore bruto ancorado, não parâmetro estimado.
- **Q-matrix por área no MIRT bifator** exige no mínimo 3 itens por fator
  específico. CN (5) passa no limite formal, mas com margem pequena.
- **Parâmetro c** está fixo em 0.20 no motor científico, valor que pressupõe
  5 alternativas. Se a prova do 6º ano tiver 4 alternativas, c = 0.25.
- **Rota `v6_reduzida`** (`POST /api/v6/tri/reduzida`) foi feita para provas de
  10 a 45 itens projetadas na escala ENEM de 45 itens, e é explicitamente
  pedagógica, não oficial. A prova aqui tem 50 itens e escala própria 200–800,
  então a rota não se aplica sem adaptação.

## 6. Pendências antes da primeira correção

- [ ] Número de alternativas por item (define o c)
- [ ] Quantidade de estudantes por aplicação (define se há calibração ou só ancoragem)
- [ ] Formato de chegada das respostas (CSV, XLSX, cartão-resposta OMR)
- [ ] Gabarito oficial e identificação do caderno/versão
- [ ] Confirmação do mapa de posições da seção 4
- [ ] Se há itens comuns entre anos ou entre aplicações (permitiria equalização)
- [ ] Decisão: escore por área (4 notas) e/ou nota geral pelo total bruto

## 7. Histórico de aplicações

Nenhuma aplicação registrada até 2026-09-18.

| Data | Escola/turma | Nº estudantes | Arquivo bruto | Método usado | Saída |
|---|---|---|---|---|---|
| — | — | — | — | — | — |
