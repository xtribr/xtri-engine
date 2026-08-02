# AGENTS.md

Este arquivo orienta agentes de codificação de IA que trabalham neste repositório.
Ele não pressupõe nenhum conhecimento prévio do projeto.

## Visão Geral do Projeto

**XTRI Engine API (TRI V3)** — um serviço FastAPI que estima notas TRI (Teoria
de Resposta ao Item) para simulados do ENEM. Inclui um frontend web de página
única para upload de CSV + processamento em lote e faz parte do ecossistema
Gabaritai/XTRI: um backend Node.js o consome via a variável de ambiente
`PYTHON_TRI_URL`.

O código principal da aplicação fica em `python_tri_v3_service/`. A raiz do
repositório também contém material de apoio:

| Caminho | O que é |
|---|---|
| `python_tri_v3_service/` | O serviço FastAPI (o projeto propriamente dito) |
| `desktop/xtri-engine/` | Shell desktop nativo (BeeWare Briefcase + Toga) que embute o serviço numa WebView |
| `microdados-2025/` | Área de staging local para microdados reais do INEP ENEM 2025 (ignorada pelo git, dados fictícios proibidos — ver o README dela) |
| `microdata/` | CSVs oficiais de microdados do ENEM (ex.: `MICRODADOS_ENEM_2024.csv`) lidos pelo pipeline de análise |
| `exports/` | Exportações CSV geradas (notas de corte SISU, dados de cursos) |
| `docs/`, `wordpress/` | Rascunhos de blog/paper e artefatos publicáveis |
| `Simulado Fredão/`, `tmp/` | Planilhas e dados temporários para validação |
| `findings.md`, `progress.md`, `task_plan.md` | Notas de trabalho dos esforços de avaliação/refatoração em andamento |

## Stack Tecnológica

- **Python ≥ 3.11**, **FastAPI** + **Pydantic v2**, servido por **Uvicorn**
- **NumPy / SciPy / pandas** para os motores TRI e o pipeline de microdados
- **pyarrow** (cache Parquet), **openpyxl** (exportação Excel),
  **matplotlib** (dashboards PNG no servidor — sempre `matplotlib.use('Agg')`
  antes de importar o pyplot; não há backend de GUI na thread de trabalho)
- **supabase** (cliente externo opcional), **anthropic** (relatórios
  narrativos opcionais — o endpoint retorna 501 quando `ANTHROPIC_API_KEY`
  não está definida)
- **R + Rscript** com o pacote `mirt` para o worker multidimensional opcional
  (`app/scripts/mirt_fit.R`, `mirt_score.R`); o serviço degrada graciosamente
  quando o R não está disponível
- **Docker** para execução containerizada; **BeeWare Briefcase + Toga** para
  o app desktop
- Metadados de pacote/dependências: `python_tri_v3_service/pyproject.toml`
  (backend setuptools; espelhado em `requirements.txt` /
  `requirements-dev.txt`; `uv.lock` presente),
  `desktop/xtri-engine/pyproject.toml` (Briefcase)

## Comandos de Build e Execução

```bash
# Rodar localmente (dev com hot-reload)
cd python_tri_v3_service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 5003 --reload

# Docker (desenvolvimento)
cd python_tri_v3_service
docker compose up --build

# Frontend UI:             http://localhost:5003/
# Página do validador:     http://localhost:5003/validator
# Dashboards de microdados: http://localhost:5003/microdata
# Documentação interativa: http://localhost:5003/docs
```

O serviço roda na **porta 5003** (retrocompatível com o serviço Flask V2).

Variáveis de ambiente (carregadas do `.env` na raiz do repositório ou no
diretório do serviço): `SUPABASE_URL`, `SUPABASE_KEY` (chave service_role;
os endpoints de validação INEP V4 precisam delas), `MICRODATA_PATH` (caminho
do host montado em `/data` no Docker), `ANTHROPIC_API_KEY` e
`XTRI_REPORT_MODEL` (geração opcional de relatórios), `XTRI_APP_SUPPORT_DIR`
(sobrescreve o diretório de runtime do desktop, padrão
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

- 383 testes coletados (371 passaram / 12 pulados no `.venv` local,
  verificado em 2026-08-02; marcadores unit | integration | api definidos no
  `pyproject.toml`). Não é preciso rebuild ao editar — código e testes são
  montados como volumes somente-leitura no serviço Docker de testes.
- A cobertura é configurada no `pyproject.toml`: fonte `app/`,
  `fail_under = 80`; `app/scripts/*` e `app/db/*` são omitidos.
- Os testes de API usam o TestClient do FastAPI e fazem `mock.patch` nos
  nomes re-exportados por `app.main` (`mirt_runtime_available`,
  `run_mirt_score`, `default_csv_path`, `submit_microdata_job`) — os routers
  resolvem esses nomes através do módulo `app.main` em tempo de chamada,
  portanto mantenha os re-exports intactos.
- Há também um arnês de acurácia baseado em microdados:
  `python -m app.evaluation --year 2024` (hold-out k-fold sobre alunos,
  requer o CSV real de microdados do ENEM).

## Arquitetura

Todo o código da aplicação fica em `python_tri_v3_service/app/`.

### Ponto de entrada e roteamento

`app/main.py` é apenas o ponto de entrada: constrói o app FastAPI, executa o
lifespan (carrega as tabelas de referência, restaura jobs persistidos),
adiciona CORS aberto (mesmo comportamento do V2), inclui os routers e monta
`app/static/` por último. O estado compartilhado de inicialização fica em
`app/state.py` (`state.init(...)` no lifespan; os routers chamam
`state.get_tabela()` / `state.get_tabelas_multiano()`).

Routers em `app/routers/`:

| Router | Endpoints | Motor | Finalidade |
|---|---|---|---|
| `system.py` | `GET /health`, `GET /api/debug` | — | Saúde/debug |
| `v2.py` | `POST /api/calcular-tri` | Heurístico | Substituto drop-in do V2, JSON bruto com campos `qN` |
| `v3.py` | `POST /api/v3/calibrar-cientifico`, `POST /api/v3/estimar-theta`, `GET /api/v3/referencias/{area}`, `GET /api/v3/anos-referencia`, `POST /api/v3/processar-csv` | Científico | Calibração completa (mín. 10 alunos), theta individual, tabelas de referência, upload de CSV em lote |
| `v4_inep.py` | `/api/v4/enem/*` | Parâmetros INEP | Validação contra parâmetros oficiais de itens do INEP vindos do Supabase |
| `v5_microdata.py` | `/api/v5/microdata/*`, `/api/v5/runtime/status` | Microdados | Pipeline longo de microdados: `POST /prepare` → polling em `/job/{id}` → dashboards/relatórios/exportação Excel |
| `v5_mirt.py` | `POST /api/v5/mirt/estimar-theta`, `GET /api/v5/mirt/availability` | MIRT (R) | Escoragem multidimensional via artefatos R pré-ajustados |
| `v6_reduzida.py` | `POST /api/v6/tri/reduzida[/upload]` | Escala reduzida | Projeção pedagógica auditável de provas de 10–45 itens na escala ENEM de 45 itens (não é nota oficial INEP) |
| `exports.py` | `/api/v3/exportar-*`, `/api/v4/enem/exportar-*` | — | Exportações CSV/Excel |
| `frontend.py` | `GET /`, `/validator`, `/microdata` | — | Páginas HTML estáticas |

### Os dois motores de estimativa

1. **Motor Heurístico** (`app/motores/heuristico.py`) — motor rápido,
   compatível com V2, usando Teoria Clássica dos Testes com análise de
   coerência pedagógica. Classifica os itens em 5 faixas de dificuldade pela
   taxa de acerto da turma e ajusta a nota de cada aluno pelos padrões de
   coerência. Feedback em tempo real.
2. **Motor Científico** (`app/motores/cientifico.py`) — TRI 2PL/3PL de
   verdade. Calibra os parâmetros dos itens (a, b) via MML com quadratura
   gaussiana + L-BFGS-B, estima o theta via EAP com prior N(0,1) e então
   ancora as notas nas tabelas de referência do ENEM. Usado nos fechamentos
   oficiais de simulados e no fluxo de CSV do frontend.

### Pipeline de ancoragem de notas

`app/tabela_referencia.py` carrega dois conjuntos de dados de referência de
`app/referencias/` na inicialização:

- **Ano único** (`tri_tabela_referencia_oficial.json`) — endpoints da API V2/V3
- **Multi-ano** (`tri_tabela_referencia_multiano.json`) — 10 anos
  (2009, 2015–2022, 2024) usados pelo fluxo de CSV do frontend

Cada tabela mapeia `(area, acertos) -> {tri_min, tri_med, tri_max}`; o motor
científico usa `norm.cdf(theta)` para interpolar dentro de cada faixa. O
envelope de referência por ano/acertos é o limite primário de corte.

### Módulos de apoio

- `app/csv_parser.py` — faz o parse de CSVs binários de respostas do ENEM
  (vírgula/ponto-e-vírgula, cabeçalhos numéricos ou "Questao N", XLSX,
  variantes TAB/sem cabeçalho). Dia 1: questões 1–90 = LC (1–45) + CH
  (46–90); Dia 2: 91–180 = CN (91–135) + MT (136–180). Ignora colunas de
  metadados (ID, Idioma, Acerto, TRI 1/2, CALI1/2).
- `app/irt/` — calibração unidimensional conjunta MML/EM (`unidim_em.py`,
  substitui a calibração item a item mantendo o contrato `{a, b, c,
  taxa_acerto, saturado}`), ancoragem em escala reduzida, escoragem,
  transformações.
- `app/mirt/` + `app/scripts/mirt_*.R` — worker de TRI multidimensional em
  R: o Python prepara a entrada em disco, versiona artefatos por
  ano/área/modelo/job, chama o `Rscript` via subprocesso e valida a saída
  normalizada.
- `app/runtime/` + `app/storage/` — registro local de modelos/artefatos
  armazenado em `~/Library/Application Support/XTRI Engine/` (modelos,
  cache, jobs, logs), fora do bundle do app.
- `app/microdata/` — pipeline de microdados oficiais do ENEM: loader de CSV
  em chunks com streaming (~1 GB de pico de memória) → cache Parquet →
  calibração → dashboards matplotlib (PNG em base64) → exportação Excel/CSV
  → relatório narrativo opcional via Claude.
- `app/evaluation/` — arnês de acurácia k-fold (`python -m app.evaluation`):
  mede a generalização calibrando num split de treino e escorando um split
  hold-out; nunca mistura exames/áreas num mesmo modelo.
- `app/jobs/` — dispatcher ThreadPoolExecutor dedicado para calibrações
  pesadas em background, isolado do event loop.
- `app/db/supabase_client.py` — cliente Supabase singleton (chave
  service_role).
- `app/enriquecimento.py` — monta a saída enriquecida de 9 dimensões por
  aluno (bruta, TRI, desempenho, nível pedagógico, posição relativa,
  projeção de aprovação SISU, recomendações, PRIMI, referências INEP).
- `app/models.py` — todos os schemas Pydantic de requisição/resposta.

### App desktop (`desktop/xtri-engine/`)

Projeto BeeWare Briefcase (Toga). Depende do serviço via caminho relativo
(`../../python_tri_v3_service` no seu `pyproject.toml` — mantenha o layout
do repositório ou atualize o caminho e rode `briefcase update` novamente).
Em runtime ele inicia o uvicorn **numa worker thread** (o binário GUI-stub
do Briefcase não é um interpretador Python genérico, então um subprocesso
`python -m uvicorn` trava num .app empacotado), faz polling em `/health` e
então abre uma WebView apontada para o backend. O build macOS é apenas
arm64, macOS mínimo 12.0 (restrições dos wheels de scipy/pandas/numpy).

## Regras de Domínio Fundamentais

- **Áreas**: LC (Linguagens), CH (Ciências Humanas), CN (Ciências da
  Natureza), MT (Matemática) — sempre códigos de duas letras maiúsculas.
- **Regra inviolável**: cada calibração é escopada a uma combinação
  simulado × área. Nunca misture áreas ou simulados num mesmo modelo.
- **Parâmetro c** (chute) é fixo em 0.20 (5 alternativas) para amostras
  < 500.
- **Estados do aluno** por nota ancorada: AVANCADO ≥ 700, PROFICIENTE ≥
  600, BASICO ≥ 500, ELEMENTAR ≥ 400, INICIANTE < 400.
- **Referência multi-ano**: LC vai de 0–44 acertos (não 0–45) em todos os
  anos; alguns anos têm medianas não-monotônicas em acertos extremos
  (artefato de amostra pequena nos dados reais do ENEM).

## Convenções

- Comentários de código e docstrings em `app/` são predominantemente em
  **português** (brasileiro); strings visíveis ao usuário, estados do aluno
  e nomes de campos da API também. Mantenha código novo consistente com o
  idioma e o estilo ao redor.
- Siga o estilo de docstring de módulo existente (um resumo curto em
  português das responsabilidades) e `from __future__ import annotations`
  nos módulos mais novos.
- Deploy: o `Dockerfile` de produção usa `python:3.11-slim`, instala a
  partir do `requirements.txt`, roda como usuário não-root com 2 workers
  uvicorn e faz health-check em `/health`. O Compose monta `app/` como
  somente-leitura para hot-reload e o diretório de microdados do host em
  `/data:ro`.
- Segurança: nunca commite microdados reais do INEP, `.env` ou chaves do
  Supabase — `microdados-2025/` e `microdata/` são áreas de staging
  ignoradas pelo git; o serviço lê segredos apenas de variáveis de ambiente.
