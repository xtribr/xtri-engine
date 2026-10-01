# CLAUDE.md

Orienta o Claude Code (claude.ai/code) ao trabalhar neste repositório.
Documento irmão: `AGENTS.md` (mesmo conteúdo, para agentes de codificação em geral).

## Visão geral

**XTRI Engine API (TRI V3)** — serviço FastAPI que estima notas TRI (Teoria de
Resposta ao Item) para simulados do ENEM, com frontend web de página única para
upload de CSV e processamento em lote. Faz parte do ecossistema Gabaritai/XTRI:
um backend Node.js o consome via `PYTHON_TRI_URL`.

O código da aplicação fica em `python_tri_v3_service/`. A raiz do repositório
também guarda material de apoio:

| Caminho | O que é |
|---|---|
| `python_tri_v3_service/` | O serviço FastAPI (o projeto propriamente dito) |
| `desktop/xtri-engine/` | Shell desktop nativo (BeeWare Briefcase + Toga) que embute o serviço numa WebView |
| `microdata/` | CSVs oficiais de microdados do ENEM (ex.: `MICRODADOS_ENEM_2024.csv`) — ignorado pelo git |
| `microdados-2025/` | Staging local dos microdados INEP 2025 + censo escolar — ignorado pelo git, dados fictícios proibidos |
| `outputs/` | Artefatos gerados pelos pipelines (relatórios psicométricos, tabelas de referência provisórias) |
| `exports/` | Exportações CSV geradas (notas de corte SISU, cursos) — ignorado pelo git |
| `docs/`, `wordpress/` | Rascunhos de blog/paper e artefatos publicáveis |
| `Simulado Fredão/`, `tmp/` | Planilhas e dados de validação |
| `findings.md`, `progress.md`, `task_plan.md` | Notas de trabalho dos esforços de avaliação/refatoração |

## Stack

- **Python ≥ 3.11**, **FastAPI** + **Pydantic v2**, servido por **Uvicorn**
- **NumPy / SciPy / pandas** nos motores TRI e no pipeline de microdados
- **pyarrow** (cache Parquet), **openpyxl** (Excel), **matplotlib** (dashboards
  PNG no servidor — sempre `matplotlib.use('Agg')` antes do pyplot; não há
  backend de GUI na thread de trabalho)
- **supabase** (cliente externo opcional), **anthropic** (relatórios narrativos
  opcionais — o endpoint retorna 501 sem `ANTHROPIC_API_KEY`)
- **R + Rscript** com o pacote `mirt` (fixado em 1.46.1) para o worker
  multidimensional (`app/scripts/mirt_fit.R`, `mirt_score.R`); o serviço degrada
  graciosamente quando o R não está disponível
- **Docker** para execução containerizada; **BeeWare Briefcase + Toga** no desktop

## Build e execução

```bash
# Local (dev com hot-reload)
cd python_tri_v3_service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 5003 --reload

# Docker
cd python_tri_v3_service
docker compose up --build

# Frontend UI:              http://localhost:5003/
# Página do validador:      http://localhost:5003/validator
# Dashboards de microdados: http://localhost:5003/microdata
# Documentação interativa:  http://localhost:5003/docs
```

O serviço roda na **porta 5003** (retrocompatível com o serviço Flask V2).

Variáveis de ambiente (do `.env` na raiz ou no diretório do serviço):
`SUPABASE_URL`, `SUPABASE_KEY` (service_role; os endpoints V4/INEP precisam
delas), `MICRODATA_PATH` (caminho do host montado em `/data` no Docker),
`ANTHROPIC_API_KEY` e `XTRI_REPORT_MODEL` (relatórios opcionais),
`XTRI_APP_SUPPORT_DIR` (diretório de runtime, padrão
`~/Library/Application Support/XTRI Engine/`).

## Testes

```bash
cd python_tri_v3_service

# Suite completa com cobertura (Docker)
docker compose -f docker-compose.test.yml run --rm test

# Arquivo único / palavra-chave / marcador
docker compose -f docker-compose.test.yml run --rm test pytest tests/test_cientifico.py -v
docker compose -f docker-compose.test.yml run --rm test pytest -k "test_eap" -v
docker compose -f docker-compose.test.yml run --rm test pytest -m unit

# Ou diretamente no venv local
.venv/bin/python -m pytest -q
```

- Última execução registrada: **383 testes coletados, 371 passaram e 12 pulados**
  no `.venv` local, em **2026-08-02**. Marcadores `unit | integration | api`
  definidos no `pyproject.toml`. Não é preciso rebuild ao editar — código e
  testes são montados como volumes somente-leitura no serviço Docker de testes.
- Cobertura configurada no `pyproject.toml`: fonte `app/`, `fail_under = 80`;
  `app/scripts/*` e `app/db/*` omitidos.
- Os testes de API usam o TestClient do FastAPI e fazem `mock.patch` nos nomes
  re-exportados por `app.main` (`mirt_runtime_available`, `run_mirt_score`,
  `default_csv_path`, `submit_microdata_job`) — os routers resolvem esses nomes
  através do módulo `app.main` em tempo de chamada, então mantenha os
  re-exports intactos.
- Arnês de acurácia sobre microdados: `python -m app.evaluation --year 2024`
  (hold-out k-fold sobre alunos; requer o CSV real de microdados do ENEM).

## Arquitetura

Todo o código da aplicação fica em `python_tri_v3_service/app/`.

### Ponto de entrada e roteamento

`app/main.py` é **apenas o ponto de entrada**: constrói o app FastAPI, executa o
lifespan (carrega tabelas de referência, restaura jobs persistidos), adiciona
CORS aberto (mesmo comportamento do V2), inclui os routers e monta
`app/static/` por último. O estado compartilhado fica em `app/state.py`
(`state.init(...)` no lifespan; os routers chamam `state.get_tabela()` /
`state.get_tabelas_multiano()`).

Routers em `app/routers/`:

| Router | Endpoints | Motor | Finalidade |
|---|---|---|---|
| `system.py` | `GET /health`, `GET /api/debug` | — | Saúde/debug |
| `v2.py` | `POST /api/calcular-tri` | Heurístico | Substituto drop-in do V2, JSON bruto com campos `qN` |
| `v3.py` | `POST /api/v3/calibrar-cientifico`, `POST /api/v3/estimar-theta`, `GET /api/v3/referencias/{area}`, `GET /api/v3/anos-referencia`, `POST /api/v3/processar-csv`, `POST /api/v3/processar-json` | Científico | Calibração completa (mín. 10 alunos), theta individual, tabelas de referência, CSV em lote, simulado completo em JSON binarizado (`app/simulado_binarizado.py`) |
| `v4_inep.py` | `POST /api/v4/enem/calcular-com-inep`, `POST /api/v4/enem/validar-simulado`, `GET /api/v4/enem/info`, `GET /api/v4/enem/provas/{ano}/{area}`, `GET /api/v4/enem/itens/{ano}/{area}/{co_prova}`, `GET /api/v4/enem/transformacao/{ano}/{area}` | Parâmetros INEP | Validação contra parâmetros oficiais de itens vindos do Supabase |
| `v5_microdata.py` | `POST /api/v5/microdata/prepare`, `GET /api/v5/microdata/job/{job_id}`, `/status/{year}`, `/anos-disponiveis`, `/breakdown/{year}`, `/dashboard{,-data}/{year}`, `/report/{year}[/full]`, `/full-report/{year}`, `/export-csv/{year}`, `/export-excel/{year}`, `GET /api/v5/runtime/status` | Microdados | Pipeline longo: `prepare` → polling no job → dashboards/relatórios/exportação |
| `v5_mirt.py` | `POST /api/v5/mirt/calibracoes`, `GET /api/v5/mirt/calibracoes/{job_id}`, `POST /api/v5/mirt/estimar-theta`, `POST /api/v5/mirt/estimar-theta-lote`, `GET /api/v5/mirt/availability` | MIRT (R) | Calibração multidimensional assíncrona e escoragem em lote |
| `v6_reduzida.py` | `POST /api/v6/tri/reduzida`, `POST /api/v6/tri/reduzida/upload` | Escala reduzida | Projeção pedagógica auditável de provas de 10–45 itens na escala ENEM de 45 itens (não é nota oficial INEP) |
| `exports.py` | `/api/v3/exportar-csv`, `/api/v3/exportar-excel`, `/api/v4/enem/exportar-validacao-csv`, `/api/v4/enem/exportar-validacao-excel` | — | Exportações CSV/Excel |
| `frontend.py` | `GET /`, `GET /validator`, `GET /microdata` | — | Páginas HTML estáticas |

### Os dois motores de estimativa

1. **Motor Heurístico** (`app/motores/heuristico.py`) — rápido, compatível com
   V2, Teoria Clássica dos Testes com análise de coerência pedagógica.
   Classifica itens em 5 faixas de dificuldade pela taxa de acerto da turma e
   ajusta a nota de cada aluno pelos padrões de coerência. Feedback em tempo real.
2. **Motor Científico** (`app/motores/cientifico.py`) — TRI 2PL/3PL de verdade.
   Calibra `(a, b)` via MML com quadratura gaussiana + L-BFGS-B, estima theta
   por EAP com prior N(0,1) e ancora nas tabelas de referência do ENEM. Usado
   nos fechamentos oficiais de simulados e no fluxo de CSV do frontend.

### Pipeline de ancoragem de notas

`app/tabela_referencia.py` carrega dois conjuntos de referência de
`app/referencias/` na inicialização:

- **Ano único** (`tri_tabela_referencia_oficial.json`) — endpoints V2/V3
- **Multi-ano** (`tri_tabela_referencia_multiano.json`) — **11 anos**
  (2009, 2015–2024) usados pelo fluxo de CSV do frontend; 2023 foi gerado dos
  microdados oficiais por `scripts_local/gerar_referencia_ano.py`

Cada tabela mapeia `(area, acertos) -> {tri_min, tri_med, tri_max}`. A posição
do aluno dentro da faixa é **condicional aos acertos** (`app/irt/conditional.py`):
`u = Φ(w·(theta − m(k))/s(k))`, em que `m(k)` e `s(k)` são a média e o desvio do
theta entre quem faz `k` acertos, derivados dos parâmetros dos itens, e
`w = min(0,9; n/(n+25))` encolhe a posição em direção à mediana em calibrações
pequenas. O aluno típico de cada número de acertos recebe `tri_med`. Os anos 2023 e 2024
têm percentis completos por acertos (`tri_quantis_referencia.json`); os demais
usam mín/med/máx. A regra antiga, `norm.cdf(theta)` absoluto, travava a nota no
mínimo ou no máximo da faixa e só permanece como caminho legado quando não há
curva (rotas MIRT). Os limites de `TRI_MINIMA_OFICIAL` continuam valendo.

### Módulos de apoio

- `app/csv_parser.py` — parse de CSVs binários de respostas do ENEM
  (vírgula/ponto-e-vírgula, cabeçalhos numéricos ou "Questao N", XLSX, variantes
  TAB/sem cabeçalho). Dia 1: 1–90 = LC (1–45) + CH (46–90); Dia 2: 91–180 =
  CN (91–135) + MT (136–180). Ignora colunas de metadados (ID, Idioma, Acerto,
  TRI 1/2, CALI1/2).
- `app/irt/` — calibração unidimensional conjunta MML/EM (`unidim_em.py`,
  substitui a calibração item a item mantendo o contrato
  `{a, b, c, taxa_acerto, saturado}`), ancoragem em escala reduzida, escoragem,
  transformações.
- `app/mirt/` + `app/scripts/mirt_*.R` — worker de TRI multidimensional em R: o
  Python prepara a entrada em disco, versiona artefatos por ano/área/modelo/job,
  chama o `Rscript` via subprocesso e valida a saída normalizada.
- `app/runtime/` + `app/storage/` — registro local de modelos/artefatos em
  `~/Library/Application Support/XTRI Engine/` (modelos, cache, jobs, logs),
  fora do bundle do app.
- `app/microdata/` — pipeline dos microdados oficiais: loader em chunks com
  streaming (~1 GB de pico) → cache Parquet → calibração → dashboards
  matplotlib (PNG em base64) → exportação Excel/CSV → relatório narrativo
  opcional. `psychometrics_2025.py` + `app/scripts/gerar_relatorio_psicometrico_2025.py`
  geram o relatório psicométrico da base 2025.
- `app/evaluation/` — arnês de acurácia k-fold (`python -m app.evaluation`):
  calibra num split de treino e escora um hold-out; nunca mistura exames/áreas
  num mesmo modelo.
- `app/jobs/` — dispatcher ThreadPoolExecutor para calibrações pesadas em
  background, isolado do event loop.
- `app/api_auth.py` — dependência de API key para as rotas MIRT.
- `app/db/supabase_client.py` — cliente Supabase singleton (service_role).
- `app/enriquecimento.py` — saída enriquecida de 9 dimensões por aluno (bruta,
  TRI, desempenho, nível pedagógico, posição relativa, projeção SISU,
  recomendações, PRIMI, referências INEP).
- `app/models.py` — todos os schemas Pydantic de requisição/resposta.

### Fluxo MIRT para integração web

O fluxo público separa o ajuste pesado do scoring cotidiano:

1. `POST /api/v5/mirt/calibracoes` com uma coorte de no mínimo 100 alunos, uma
   única área e a Q-matrix real. No bifator, cada item carrega o fator geral e
   exatamente um específico; cada específico precisa de ao menos três itens.
2. Polling em `GET /api/v5/mirt/calibracoes/{job_id}` até `status=done`.
3. `POST /api/v5/mirt/estimar-theta-lote` com exatamente um entre
   `simulado_id` e `artefato_id`, e `item_ids` na ordem exata da Q-matrix
   (impede troca silenciosa de itens). Escora até 10 mil alunos por artefato.

Respostas usam `0=erro`, `1=acerto`, `null=ausente`; ausentes vão como `NA` ao
MIRT e nunca viram erro silenciosamente. Com itens ausentes, a API devolve os
scores latentes mas **omite a nota ENEM ancorada**; a nota também é omitida em
provas com menos de 45 itens. O endpoint individual exige a mesma ordem
explícita e não faz fallback posicional para artefatos unidimensionais — use
`/api/v3/estimar-theta` deliberadamente para o motor unidimensional.

Q-matrix automática por habilidade só é identificável quando cada fator tem
indicadores suficientes. Em provas com uma ou duas questões por habilidade,
forneça matriz manual baseada em agrupamentos pedagógicos reais (competência,
domínio ou testlet) — a API não inventa esses agrupamentos.

Configuração em produção:

```bash
XTRI_API_KEY='uma-chave-longa-e-aleatoria'
XTRI_CORS_ORIGINS='https://seu-site.com.br'
XTRI_APP_SUPPORT_DIR='/var/lib/xtri'
# Somente em desenvolvimento local sem chave:
# XTRI_MIRT_ALLOW_UNAUTHENTICATED='true'
XTRI_MIRT_RETAIN_RESPONSES='false'
XTRI_MIRT_MAX_ACTIVE_JOBS='1'
XTRI_MIRT_MAX_SCORING_PROCESSES='2'
XTRI_MIRT_SCORE_TIMEOUT_SECONDS='300'
```

A chave vai no header `X-API-Key`. Sem `XTRI_API_KEY` as rotas falham de forma
segura. `XTRI_API_KEY` é segredo **server-to-server**: mantenha no backend
(Node.js → FastAPI), nunca no JavaScript do navegador — CORS limita origens no
browser, mas não protege a chave nem substitui autenticação. O `model.rds` do
`mirt` contém padrões de resposta desidentificados; trate o volume
`xtri-runtime` como dado sensível (acesso restrito, criptografia no host,
política de retenção). O scoring rejeita artefatos criados por outra versão do
pacote `mirt`.

### App desktop (`desktop/xtri-engine/`)

Projeto BeeWare Briefcase (Toga) que depende do serviço por caminho relativo
(`../../python_tri_v3_service` no seu `pyproject.toml` — mantenha o layout do
repositório ou atualize o caminho e rode `briefcase update`). Em runtime inicia
o uvicorn **numa worker thread** (o binário GUI-stub do Briefcase não é um
interpretador Python genérico, então `python -m uvicorn` como subprocesso trava
num `.app` empacotado), faz polling em `/health` e abre uma WebView apontada
para o backend. Build macOS apenas arm64, mínimo 12.0 (restrição dos wheels de
scipy/pandas/numpy).

## Regras de domínio fundamentais

- **Áreas**: LC (Linguagens), CH (Ciências Humanas), CN (Ciências da Natureza),
  MT (Matemática) — sempre códigos de duas letras maiúsculas.
- **Regra inviolável**: cada calibração é escopada a uma combinação
  simulado × área. Nunca misture áreas ou simulados num mesmo modelo.
- **Parâmetro c** (chute) fixo em 0.20 (5 alternativas) para amostras < 500.
- **Níveis do aluno** por nota ancorada (`app/enriquecimento.py`):
  AVANCADO ≥ 700, PROFICIENTE ≥ 600, BASICO ≥ 500, ELEMENTAR ≥ 400,
  INICIANTE < 400.
- **Referência multi-ano**: LC vai de 0–44 acertos (não 0–45) em todos os anos;
  alguns anos têm medianas não-monotônicas em acertos extremos (artefato de
  amostra pequena nos dados reais do ENEM).
- **Acertos não são nota**: a quantidade de acertos nunca é convertida por
  regra fixa em nota TRI; a ancoragem é sempre pelo envelope de referência.

## Convenções

- Comentários, docstrings, strings visíveis ao usuário, níveis do aluno e nomes
  de campos da API em `app/` são em **português** (brasileiro). Mantenha código
  novo consistente com o idioma e o estilo ao redor.
- Siga o estilo de docstring de módulo existente (resumo curto em português das
  responsabilidades) e `from __future__ import annotations` nos módulos novos.
- Deploy: o `Dockerfile` de produção usa `python:3.11-slim`, instala do
  `requirements.txt`, roda como usuário não-root com 2 workers uvicorn e faz
  health-check em `/health`. A imagem instala R, `jsonlite` e `mirt`, usa fila
  de calibração por processo e persiste modelos/jobs no volume `xtri-runtime`.
  O Compose monta `app/` como somente-leitura para hot-reload e o diretório de
  microdados do host em `/data:ro`.
- Segurança: nunca commite microdados reais do INEP, `.env` ou chaves do
  Supabase — `microdata/` e `microdados-2025/` são áreas de staging ignoradas
  pelo git; o serviço lê segredos apenas de variáveis de ambiente.

## Porta

O serviço roda na **porta 5003** (retrocompatível com o serviço Flask V2).
