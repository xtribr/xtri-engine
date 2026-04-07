# Session Progress

## Current Phase: Phase 2 — Test mat.txt without code changes

## Decisions Made

- **2026-04-07**: User raised concern about per-file patching cycle. Created planning files. Will test mat.txt with current parser before making any changes.

## Test Log

| Time | Test | Result |
|------|------|--------|
| 2026-04-07 | mat.txt with V3 Local, MT, ref 2022 | ✓ SUCCESS — no code changes needed. 49 students, max 949.9 (within teto). Same strong-population inflation pattern as mat.csv (73.5% AVANCADO). |
| 2026-04-07 | User identified DESIGN bug: Rank Intragrupo + smoothing pushed all students toward tri_med, ignoring that low-theta students should land near tri_min. The whole point of MIN/MED/MAX in the table is to have a range. |
| 2026-04-07 | Replaced ancorar_notas() with "Cohort-Anchored Distribution": z-score normalization + percentile mapping into [tri_min, tri_med, tri_max] |
| 2026-04-07 | Re-tested all 5 corpus files: low-theta students now correctly land at tri_min (cn.txt 366.3, LC.txt 294, CH.txt 318.6, mat.txt 338.4, mat.csv 336.8). High-theta students correctly land at tri_max for their acertos. mat.csv still inflated for strong populations (inherent property, not bug). |
| 2026-04-07 | All 119 tests passing after the change. |

## Errors Encountered

(none yet)

## Files Created/Modified This Session

- `task_plan.md` (created)
- `findings.md` (created)  
- `progress.md` (this file, created)
