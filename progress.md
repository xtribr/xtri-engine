# Session Progress

## Current Phase: Phase 8 — MICRODATA Analysis + Drill-down (P1+P2+P3+P4 done, dashboards rendering 4×3 PNGs in ~3s; P5 pending = backend `/api/v5/microdata/*` routes)

## Prior Phase: Phase 7 — Native macOS App (Tier 1 MVN, **steps 1–12 complete**, optional step 13 = clean rebuild via `briefcase create macOS`)

**Progress**: Steps 1–12 complete end-to-end. `XTRI Engine.app` exists in three places (build dir, `/Applications/`, `~/Desktop/`), each 353 MB, arm64-thin, ad-hoc signed, identifier `com.gabaritai.xtri-engine`. Runs cleanly: launch → port 5003 bound by main process → /health 200 → WebView with X-TRI logo in Dock and header. `app.py` runs uvicorn in a worker thread (NOT subprocess — Briefcase GUI-Stub is not a real Python interpreter). 2024 reference scale ingested into JSON, constants updated to round-down PDF values. V3 Local validated end-to-end via LC.txt smoke test. Slider math fixed (shift/scale apply post-z-score for Cohort-Anchored, post-percentile for Percentil). UI layout fixes for dropdown overflow + chart x-axis label collision. 119 tests passing.

## Key Commits

| SHA | Phase | Summary |
|---|---|---|
| `57258ef` | pre-V2 | Adaptive smoothing + parser improvements |
| `8278b81` | pre-V2 | Empirical `TRI_MAXIMA_OFICIAL` from real microdata |
| `c732dad` | **V2.0** | Core scoring: Cohort-Anchored Distribution |
| `d80d0cd` | **V2.1** | Enriched 8-dimensional per-student output (current HEAD) |

## Decisions Log

- **2026-04-07**: User raised per-file patching concern → created planning docs.
- **2026-04-07**: Replaced `ancorar_notas()` with Cohort-Anchored Distribution (z-score → percentile mapping into `[tri_min, tri_med, tri_max]`).
- **2026-04-07**: V2.1 enriched output committed (`d80d0cd`). 4 files, 782 insertions. 7 new Pydantic models + `enriquecimento.py` (22.7 KB).
- **2026-04-07**: User decided to transform this into a **single-user native Mac app**. No distribution, no signing, WKWebView wraps existing HTML.
- **2026-04-07**: Tier 1 architecture decisions confirmed — Briefcase + WKWebView + Python 3.11 + `desktop/` subdir. `MICRODATA_PATH` stays dev-only (not bundled).
- **2026-04-07**: Two rounds of drift — user pasted AI-generated "authorized sequences" that contradicted Tier 1. Flagged both times, user confirmed **"Lets go with A"** (Tier 1 sequence as written in `task_plan.md` Phase 7).

## Test Log

| Date | Test | Result |
|------|------|--------|
| 2026-04-07 | mat.txt (V3 Local, MT, ref 2022) | ✓ SUCCESS — no code changes, 49 students |
| 2026-04-07 | 5 corpus files with Cohort-Anchored Distribution | ✓ Low-theta → tri_min (LC=294, CH=318.6, CN=366.3, mat.txt=338.4, mat.csv=336.8) |
| 2026-04-07 | 119 tests after V2.0 anchoring change | ✓ all passing |
| 2026-04-07 | V2.1 endpoint curl on LC/CH/CN/MT | ✓ all 3 new fields populated, 8 dimensions present, cohort floor exact |
| 2026-04-07 | 119 tests after V2.1 (commit `d80d0cd`) | ✓ all passing |
| 2026-04-07 | `briefcase dev` first run (15s timeout) | ✗ TIMEOUT — cold imports ~25s |
| 2026-04-07 | `briefcase dev` after timeout bump to 60s + try/except guard | ✓ uvicorn up, /health 200, /api/v3/anos-referencia returns 9 years, WebView showing frontend |
| 2026-04-07 | Window close after first on_exit attempt (assigned `self.on_exit = handler` + `self.exit()` in on_close) | ✗ ORPHAN — uvicorn 22752 reparented to launchd (PPID=1), held port 5003 |
| 2026-04-07 | Window close after `on_exit` override (proper class method) + `request_exit()` in on_close | ✓ briefcase dev exit 0, port 5003 free, no orphan |
| 2026-04-07 | `briefcase build macOS` iter 1 (git license) | ✗ Briefcase couldn't run git (Apple Xcode license unaccepted) — fixed via `sudo xcodebuild -license accept` |
| 2026-04-07 | `briefcase build macOS` iter 2 (scipy wheel) | ✗ scipy 1.11+ arm64 needs macOS 12+, default 11.0 — fixed via `min_os_version="12.0"` + `universal_build=false` |
| 2026-04-07 | `briefcase build macOS` iter 3 (icon warning) | ⚠ Built but no app icon — fixed via `sips`/`iconutil` generating proper `.icns` from 800px PNG |
| 2026-04-07 | Packaged `.app` first launch (subprocess uvicorn) | ✗ Stuck in `_wait_for_ready`, uvicorn children orphaned to launchd — root cause: Briefcase GUI-Stub binary is NOT a Python interpreter |
| 2026-04-07 | Packaged `.app` after thread-based uvicorn rewrite | ✓ Port 5003 bound by main process (PID 32650, no children), /health 200, WebView visible with logo |
| 2026-04-07 | Packaged `.app` V4 INEP mode CO_PROVA dropdown | ✗ "(nenhuma disponível)" — `/api/v4/enem/provas/2022/MT` returned HTTP 503 "Supabase não configurado" |
| 2026-04-07 | Packaged `.app` V4 endpoint after `.env` loader added to launcher | ✓ HTTP 200, returns real prova colors (AZUL/AMARELA/ROSA/CINZA/...) |
| 2026-04-07 | LC.txt processed via packaged .app with 2024 scale | ✓ 54 students, lowest theta exact tri_min[9]=303.2, highest near tri_max for their acertos band |
| 2026-04-07 | User identified all 4 `TRI_MAXIMA_OFICIAL` constants higher than empirical 2024 PDF maxes | LC=800→790, CH=835→815, CN=875→865, MT=990→960 (round-down conservative) |
| 2026-04-07 | 2024 year added to multi-year JSON (LC pooled INGLÊS+ESPANHOL) | 119 tests pass after `test_anos_esperados` updated to include 2024 |
| 2026-04-07 | Theta Shift / Theta Scale sliders frozen on Cohort-Anchored AND Percentil | ✗ Math bug — both methods normalize against cohort, uniform shift cancels out |
| 2026-04-07 | Slider fix: apply shift/scale post-z-score (Cohort) and post-percentile (Percentil) | ✓ Sliders now move notas visibly, validated by user across both methods |
| 2026-04-07 | Dropdown text "Cohort-Anchored (z-score → tri_min/med/max)" overflow | ✗ Overlapped next slider's "Theta Shift" label |
| 2026-04-07 | Layout fix: shorten dropdown to "Cohort-Anchored", add `min-width:0; overflow:hidden` | ✓ No more overlap |
| 2026-04-07 | Chart x-axis "Numero de Acertos" colliding with footer credit | ✗ Both rendered at y=531/532, 1px apart |
| 2026-04-07 | Chart fix: bump `pad.bottom` 55→90, reposition both labels relative to plot bottom | ✓ ~22px gap, no overlap |
| 2026-04-07 | `ditto` to `/Applications/XTRI Engine.app` AND `~/Desktop/XTRI Engine.app` | ✓ Both 353 MB, X-TRI logo, ad-hoc signed |
| 2026-04-08 | Phase 8 P1-P2: microdata loader + Parquet cache package | ✓ 20 unit tests, 139 total green |
| 2026-04-08 | Smoke test on real 1.6 GB MICRODADOS_ENEM_2024.csv | ✓ 36.6s cold stream (4.3M rows → 2.86M valid → 200K sample), 24 MB parquet, 0.08s reload |
| 2026-04-08 | Finding: `IN_TREINEIRO` removed from 2024+ microdata → loose filter removed, strict-only via Supabase whitelist | Per user "skip IN_TREINEIRO, never use it" |
| 2026-04-08 | Finding: `NO_ENTIDADE` removed from 2024+ microdata → school names via Supabase `inep_escolas` (27,328 rows, codigo_inep → nome/uf/dependencia/localizacao) | Ready for Phase 7 drill-down join |
| 2026-04-08 | Added `load_primeira_aplicacao_provas(year)` helper that queries Supabase `enem_codigos_prova` for the strict whitelist | Reuses existing supabase_client infra |
| 2026-04-08 | P3: app/microdata/calibration.py — V3 Local pipeline grouped by (area, co_prova, tp_lingua) | 18 unit tests, 157 total green |
| 2026-04-08 | Finding: INEP 2024 LC `TX_GABARITO_LC` is 50 chars (5 EN + 5 ES + 40 common), `TX_RESPOSTAS_LC` is 45 chars | Added `_effective_lc_gabarito` to slice per TP_LINGUA |
| 2026-04-08 | Finding: cientifico.py `_neg_log_lik_item` has Python loop over students → O(N×K) per L-BFGS iter, makes 200K calibration take >1 hour | Added `calibrar_itens_fast` with O(K) likelihood collapsed via (n_correct, n_wrong) counts; verified mathematically equivalent |
| 2026-04-08 | Smoke test on real 200K calibrated sample after both vectorizations | ✓ 51.3s total, LC r=0.9495, CH r=0.9711, CN r=0.9300, MT r=0.9489 vs official INEP, 99.8% coverage |
| 2026-04-08 | P4: app/microdata/dashboards.py — server-side matplotlib PNG generator (Agg backend) | 12 unit tests, 169 total green |
| 2026-04-08 | Smoke test rendering 9 PNGs from real 200K calibrated sample | ✓ Full 4×3 report 415 KB in 3.4s, per-area dashboards 105-118 KB each in <1s, layout matches mente_mat_resultados.png |
| 2026-04-08 | matplotlib>=3.7.0 added to pyproject.toml (installed in dev venv: 3.10.8) | +60-100 MB to .app bundle on next rebuild |

## Files Created/Modified

- `task_plan.md` — Phase 5/6 complete, Phase 7 Tier 1 added
- `findings.md` — root-cause analysis (historical reference, unchanged)
- `progress.md` — this file
- `python_tri_v3_service/app/models.py` — +129 lines (7 V2.1 models, 3 response fields)
- `python_tri_v3_service/app/enriquecimento.py` — **NEW** (22.7 KB, 8-dim builders + NACIONAL_STATS)
- `python_tri_v3_service/app/main.py` — imports `enriquecimento`, wires fields into `/api/v3/processar-csv`

## Open Items

- **Uncommitted changes** (Phase 7 scaffold, not yet committed):
  - `python_tri_v3_service/pyproject.toml` — added `[build-system]` + `[project]` + `[tool.setuptools.*]`
  - `desktop/` — entire directory is new (Briefcase tooling venv + scaffolded `xtri-engine/`)
  - `desktop/xtri-engine/src/xtri_engine/app.py` — Toga shell, threaded uvicorn (`uvicorn.Server` + `_XTRIServer` no-op signal handlers), `on_exit` override, `_on_main_window_close` → `request_exit()`, and a `load_dotenv` block at module-import time that searches `~/Library/Application Support/XTRI Engine/.env` then `/Volumes/KINGSTON/apps/apps/MIRT/.env`
  - `desktop/xtri-engine/pyproject.toml` — added `xtri-service @ file://...` to `requires`, `icon = "icons/xtri-engine"`, `min_os_version = "12.0"`, `universal_build = false`
  - `desktop/xtri-engine/icons/xtri-engine.png` — XTRI logo (800×800 RGBA, copied from `~/Desktop/logotipo.png`) used as macOS app icon at build time
  - `python_tri_v3_service/app/static/logotipo.png` — same file, served at `/static/logotipo.png`
  - `python_tri_v3_service/app/static/index.html` — added `<img src="/static/logotipo.png">` before `<h1>` in header
  - `python_tri_v3_service/pyproject.toml` — package-data widened from `static/*.html` to `static/*` so PNGs ship with the wheel
- Untracked `index.html` at repo root (unrelated, predates V2.1). Left for separate triage.
- Git identity warning from V2.1 commit: committer is `home@Alexandres-MacBook-Pro.local`. Not changed (destructive, needs explicit user request).
- `httpx` is used in `app.py` for the `/health` readiness probe but is **not directly listed** in `xtri-service` deps — it lands in the venv transitively via `supabase`. If supabase ever drops httpx, the readiness probe breaks silently. Add `httpx` to `python_tri_v3_service/pyproject.toml [project].dependencies` if you want belt-and-suspenders. Stdlib `urllib.request` is the zero-dep alternative.
- `file:///Volumes/KINGSTON/apps/apps/MIRT/python_tri_v3_service` is an absolute path hardcoded in `desktop/xtri-engine/pyproject.toml`. If the repo ever moves off KINGSTON, this must be updated and `briefcase update` re-run.
- Homebrew-installed Python 3.11 (`/opt/homebrew/bin/python3.11`) now exists on the host system. Reversible via `brew uninstall python@3.11`.
