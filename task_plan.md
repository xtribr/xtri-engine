# Task Plan: Stop the Patch Cycle

## Goal

Eliminate the pattern of "each new file = new code patch". Build a robust foundation so any reasonable simulado format works without code changes.

## The Problem (User's Concern)

Each new dataset uploaded has triggered a code change:

| Dataset | Patch Required | Reason |
|---------|---------------|--------|
| `mat.csv` | None (initial) | Baseline format |
| `dados-mat.xlsx` | XLSX support | New format |
| `cn.txt` | TAB delimiter + headerless detection | New format |
| `LC.txt` | Empirical ceiling correction | Wrong constant |
| `CH.txt` | None (used same fix) | — |
| `mat.txt` | TBD | TBD |

This is brittle. Root cause: we patch symptoms instead of building a principled foundation.

## Phases

### Phase 1: Diagnose Brittleness Sources [in_progress]

- [x] List all formats encountered so far
- [x] Identify patches made for each
- [ ] Categorize patches: format/parser vs constants vs anchoring vs other
- [ ] Identify the ONE root cause if there is one

### Phase 2: Test mat.txt Without Code Changes [complete]

- [x] Run mat.txt with current parser (V3 Local, MT area, 2022)
- [x] **RESULT: Worked without any code changes** — TAB delimiter and
  headerless detection (added for cn.txt) generalized correctly to mat.txt
- [x] 49 students parsed, no errors, 0 above teto
- [x] BUT the strong-population inflation is back: 73.5% AVANCADO,
  4.1% above P99.9 nacional (only 0.1% should be there in real ENEM)
- [x] Inflation is NOT a parser issue — it's an inherent property of
  local calibration with non-representative populations

### Phase 3: Define the Robust Foundation [pending]

Three principled fixes:

1. **Parser robustness**:
   - Accept CSV (comma), TSV (tab), SCSV (semicolon)
   - Accept XLSX (with/without explicit area columns)
   - Accept files with header OR headerless (auto-detect from first row)
   - Accept files with ID column OR without (synthesize aluno_N)
   - Validate column count matches expected per area

2. **Constants from microdata, not Excel**:
   - All ceilings, P99.9, mediana derived from actual ENEM microdata
   - Documented per-year per-area
   - Versioned and tested

3. **Test suite covering format variations**:
   - One test per (format × header × ID-column × area) combination
   - Minimum 12 format combinations covered
   - Any new file format must be added as a test, not just a patch

### Phase 4: Implement the Fix [SUPERSEDED by V2.1]

The user has chosen a more ambitious path: full data enrichment + UX feedback.

### Phase 5: V2.1 Enriched Output + UX Methodology Feedback [in_progress]

User decisions:
- Schema: Option B (parallel — keep flat `resultados`, add rich `resultados_estruturados`)
- UX: All 3 (1A banner + 1B per-student modal + 1C methodology panel)
- Score types: 8 dimensions (raw, irt, performance, desempenho_pedagogico, posicao_relativa, projecao_aprovacao, recomendacoes, primi) + referencia_inep metadata
- Always connect to INEP official (citations, validation metrics, methodology links)
- Single-user, future macOS native app

#### Phase 5A: Backend Schema [pending]
- New Pydantic models for the 8 dimensions
- New helper functions: compute_z_cohort, compute_percentil_nacional, classify_pedagogico, sisu_projection, primi_analysis
- New referencia table for SISU cutoffs and pedagogical interpretations
- Endpoint /api/v3/processar-csv adds `resultados_estruturados` field

#### Phase 5B: UX 1A — Methodology Banner [pending]
- After processing, show a banner with method, cohort stats, warnings
- "validado contra 360K alunos INEP, MAE 0.025"
- Warn if cohort acertos avg > P95 nacional

#### Phase 5C: UX 1B — Per-Student Modal [pending]
- Click student row → modal showing all 8 dimensions
- Charts: position in cohort, position nationally, breakdown by dimension
- Specific study recommendations

#### Phase 5D: UX 1C — Methodology Panel [pending]
- Collapsible "Diagnóstico Metodológico" at bottom
- Formula display
- Reference table visualization
- Empirical sigma/mu derivation
- Cited papers with links

### Phase 6: Validate Everything [pending]
- All 119 tests still passing
- New schema tests (10+)
- All 5 corpus files still process correctly
- New endpoint returns enriched data

## Files Created/Modified Tracking

| File | Status | Notes |
|------|--------|-------|
| task_plan.md | created | This file |
| findings.md | created | Analysis of brittleness |
| progress.md | created | Session log |
