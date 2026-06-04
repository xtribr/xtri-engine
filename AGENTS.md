# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

XTRI Engine API (TRI V3) — A FastAPI service that estimates IRT (Item Response Theory / TRI) scores for ENEM exam simulations. Includes a web frontend for CSV upload + batch processing. Part of the Gabaritai/XTRI ecosystem, consumed by a Node.js backend via `PYTHON_TRI_URL`.

## Build & Run Commands

```bash
# Run locally (dev with hot-reload)
cd python_tri_v3_service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 5003 --reload

# Docker
cd python_tri_v3_service
docker compose up --build

# Frontend UI
# http://localhost:5003/

# Interactive API docs
# http://localhost:5003/docs
```

### Testing

```bash
cd python_tri_v3_service

# Full suite with coverage
docker compose -f docker-compose.test.yml run --rm test

# Single file
docker compose -f docker-compose.test.yml run --rm test pytest tests/test_cientifico.py -v

# By keyword
docker compose -f docker-compose.test.yml run --rm test pytest -k "test_eap" -v

# By marker (unit | integration | api)
docker compose -f docker-compose.test.yml run --rm test pytest -m unit
```

106 tests, 92% coverage. No rebuild needed when editing — source is volume-mounted.

## Architecture

All application code lives under `python_tri_v3_service/app/`.

### Two Estimation Engines

1. **Motor Heuristico** (`app/motores/heuristico.py`) — Fast, V2-compatible engine using Classical Test Theory (TCT) with pedagogical coherence analysis. Classifies items into 5 difficulty bands by class-wide accuracy, then adjusts each student's score based on coherence patterns. Used for real-time feedback.

2. **Motor Cientifico** (`app/motores/cientifico.py`) — Real 2PL/3PL IRT model. Calibrates item parameters (a, b) via MML with Gaussian quadrature + L-BFGS-B optimization, estimates student ability (theta) via EAP with N(0,1) prior, then anchors scores to ENEM reference tables. Used for official simulation closings and the frontend CSV workflow.

### Score Anchoring Pipeline

Both engines anchor scores using `app/tabela_referencia.py`. At startup, two sets of reference data load:
- **Single-year** (`tri_tabela_referencia_oficial.json`) — used by V2/V3 API endpoints
- **Multi-year** (`tri_tabela_referencia_multiano.json`) — 9 years (2009, 2015-2022) used by the frontend CSV workflow. Users select which year's scale to use.

Each table maps `(area, acertos) -> {tri_min, tri_med, tri_max}`. The scientific engine uses `norm.cdf(theta)` to interpolate within each range. Hard ceilings from `TRI_MAXIMA_OFICIAL` are always enforced.

### CSV Parser (`app/csv_parser.py`)

Parses ENEM binary response CSVs in multiple formats:
- Day 1 (questions 1-90): LC (1-45) + CH (46-90)
- Day 2 (questions 91-180): CN (91-135) + MT (136-180)
- Handles numeric headers (`1`, `2`), "Questao N" headers, comma/semicolon delimiters
- Skips metadata columns (ID, Idioma, Acerto, TRI 1/2, CALI1/2)

### API Routes (defined in `app/main.py`)

| Route | Engine | Purpose |
|---|---|---|
| `GET /` | — | Frontend UI (single-page HTML app) |
| `POST /api/calcular-tri` | Heuristic | V2 drop-in replacement, accepts raw JSON with `qN` fields |
| `POST /api/v3/calibrar-cientifico` | Scientific | Full calibration, requires min 10 students, binary response matrix |
| `POST /api/v3/estimar-theta` | Scientific | Single-student scoring with pre-calibrated item parameters |
| `GET /api/v3/referencias/{area}` | — | Returns official reference table for an area |
| `GET /api/v3/anos-referencia` | — | Available reference years for the dropdown |
| `POST /api/v3/processar-csv` | Scientific | Batch CSV upload: parses files, calibrates per area, returns anchored scores |

### Key Domain Rules

- **Areas**: LC (Linguagens), CH (Ciencias Humanas), CN (Ciencias da Natureza), MT (Matematica) — always uppercase two-letter codes.
- **Inviolable rule**: Each calibration is scoped to one simulado x area combination. Never mix areas or simulations in one model.
- **Parameter c** (guessing) is fixed at 0.20 (5 alternatives) for samples < 500.
- **Student states** (CHAMPION >= 650, ON_FIRE >= 580, STUDYING >= 480, NEEDS_REVIEW >= 380, NEW_STUDENT < 380) are assigned based on anchored score.
- **Multi-year reference**: LC goes 0-44 acertos (not 0-45) in all years. Some years have non-monotonic medians at extreme acertos (small-sample artifact in real ENEM data).

## Port

The service runs on **port 5003** (retrocompatible with V2 Flask service).
