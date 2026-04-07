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

### Phase 4: Implement the Fix [pending]

Only AFTER Phase 1-3 give clarity. No premature coding.

### Phase 5: Validate with All Existing Files [pending]

- [ ] Re-test mat.csv, dados-mat.xlsx, cn.txt, LC.txt, CH.txt, mat.txt
- [ ] All must work with zero code changes after Phase 4
- [ ] Run full pytest suite (must stay 119+ passing)

## Files Created/Modified Tracking

| File | Status | Notes |
|------|--------|-------|
| task_plan.md | created | This file |
| findings.md | created | Analysis of brittleness |
| progress.md | created | Session log |
