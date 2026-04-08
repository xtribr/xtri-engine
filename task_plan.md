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

### Phase 5: V2.1 Enriched Output [complete — commit d80d0cd]

Backend delivered. UX layers deferred to Phase 7 (native app absorbs the frontend work).

- [x] **5A Backend Schema** — 7 new Pydantic models + `enriquecimento.py` builders (22.7 KB). `/api/v3/processar-csv` returns `resultados_estruturados`, `metodologia`, `coorte_stats`. 4 files, 782 insertions.
- [x] Validated across LC/CH/CN/MT — all 8 dimensions populated, cohort-anchoring confirmed (LC bottom = exact area floor 294.0), 5 pedagogical levels present.
- [~] **5B/5C/5D UX layers** — deferred. Native app wraps existing HTML via WKWebView; banner/modal/methodology panel get polished in that context.

### Phase 6: Validate [complete]
- [x] 119 tests passing after V2.1
- [x] 4 corpus areas confirmed (LC/CH/CN/MT)
- [x] Enriched endpoint verified via curl against running uvicorn

### Phase 7: Native macOS App — Tier 1 MVN [in_progress — blocked on scaffold]

**Goal**: single-user native Mac `.app` that wraps the existing FastAPI service + HTML frontend. Zero functional regression. No distribution, no code signing, no multi-tenancy.

**Architecture (Tier 1 "Shell-Wrapping")**:
- Swift/AppKit shell + WKWebView pointing at `http://127.0.0.1:5003`
- FastAPI runs as child process inside the bundled Python runtime
- Existing 909-line `static/index.html` reused as-is
- Reference JSONs bundled in app resources (176 KB total)

**Confirmed decisions** (resolved 2026-04-07):
| Q | Decision | Rationale |
|---|---|---|
| Q2 Bundling tool | **Briefcase (BeeWare)** | Best chance with scipy/numpy/pandas on Apple Silicon |
| Q3 Frontend in Tier 1 | **WKWebView → existing `static/index.html`** | Preserves 909 lines of working UI |
| Q4 Supabase | **Keep bundled, optional** | Already gracefully degrades via `is_supabase_configured()` |
| Q5 Python version | **3.11** | Matches Dockerfile, wheel maturity |
| Q6 `MICRODATA_PATH` | **NOT bundled** — dev-only | Runtime has zero refs; only `app/scripts/calcular_transformacao.py` + `validacao_massa.py` use it |
| Q7 Repo strategy | **Keep FastAPI service + add `desktop/` subdir** | Existing CI/Docker unaffected |

**What the `.app` bundles**:
- Python 3.11 runtime + site-packages (fastapi, uvicorn, numpy, scipy, pandas, openpyxl, pydantic, supabase, python-dotenv)
- `python_tri_v3_service/app/` (all source)
- `python_tri_v3_service/app/referencias/tri_tabela_referencia_oficial.json` (16 KB)
- `python_tri_v3_service/app/referencias/tri_tabela_referencia_multiano.json` (160 KB)
- Swift/AppKit shell + WKWebView

**What stays outside the `.app`** (dev-only):
- `MICRODATA_PATH` (INEP CSVs — unknown size, lives wherever user put it)
- `app/scripts/calcular_transformacao.py` (offline recalibration)
- `app/scripts/validacao_massa.py` (offline validation)
- `.env` at repo root (for when dev scripts run from terminal)
- `docker-compose.yml`, `docker-compose.test.yml` (CI/dev)

**Architecture update (2026-04-07)**: Dropped raw Swift/AppKit in favor of **Toga** (BeeWare's Python GUI toolkit). Toga.WebView on macOS is literally a `WKWebView` wrapped in Python — same rendering, zero Swift code, single Python stack trace for debugging. Briefcase 0.4.1 defaults to Toga anyway.

**Tier 1 execution sequence**:

1. [x] Kill running uvicorn (PID 89649) — port 5003 freed
2. [x] `mkdir desktop/` at repo root
3. [x] `brew install python@3.11` → `/opt/homebrew/bin/python3.11` (Python 3.11.15)
4. [x] Create isolated venv for Briefcase tooling at `desktop/.venv` (Python 3.11.15)
5. [x] `pip install briefcase` → Briefcase 0.4.1 + 40 transitive deps (rich, cookiecutter, dmgbuild, etc.)
6. [x] `briefcase new --no-input -Q ...` → scaffold at `desktop/xtri-engine/` (44 KB, Toga-based)
7. [x] Make `python_tri_v3_service/` pip-installable (**Option 1** chosen over copy/symlink/external-sources):
    - Added `[build-system]`, `[project name="xtri-service" version="2.1.0"]`, `[tool.setuptools.packages.find]`, `[tool.setuptools.package-data]` to `python_tri_v3_service/pyproject.toml`
    - Existing `[tool.pytest.ini_options]` + `[tool.coverage.*]` preserved
    - Referenced from Briefcase as `"xtri-service @ file:///Volumes/KINGSTON/apps/apps/MIRT/python_tri_v3_service"` in `[tool.briefcase.app.xtri-engine].requires`
    - Package data: `referencias/*.json`, `static/*.html`
    - **Validated end-to-end**: fresh Python 3.11 venv installed xtri-service + full dep tree in 25 seconds, zero compilation, all wheels (scipy-1.17.1, numpy-2.4.4, pandas-3.0.2, fastapi-0.135.3, cryptography-46.0.6). Smoke test imported `from app.main import app` → 22 routes, title "XTRI Engine API". Reference JSONs + static HTML resolved via `os.path.dirname(__file__)`.
    - **Zero functional regression**: Dockerfile still uses `requirements.txt`. 119 tests unaffected. 
8. [x] Rewrite `desktop/xtri-engine/src/xtri_engine/app.py` with Toga WebView + FastAPI subprocess launcher (~135 lines). Verified `GET /health` exists at `app/main.py:170`, so readiness probe uses it. Defaults applied: Q8=A (port 5003), Q9=A (`atexit` + `SIGTERM`/`SIGINT` handlers + SIGTERM→SIGKILL on child), Q10=B (`sys.executable -m uvicorn`). One latent dep: `httpx` is used for the readiness poll but is not directly listed in `xtri-service` deps — it comes in transitively via `supabase`. If supabase ever drops httpx, add it to `python_tri_v3_service/pyproject.toml [project].dependencies`. Stdlib `urllib.request` is the cheap fallback if we want zero deps.
9. [x] `briefcase dev` → first run with real FastAPI backend + WebView. **Three iterations required**:
    - **Iter 1 (timeout)**: first attempt timed out at 15s waiting for `/health` because cold imports of scipy/numpy/pandas/supabase + JSON loads in the bundled python-build-standalone runtime took ~25s. Worse: the `TimeoutError` raised inside Toga's `CFTimer` callback was swallowed by rubicon-objc as "Exception ignored", so atexit never fired and we got an orphaned Toga app + live uvicorn child. **Patches**: bumped `STARTUP_TIMEOUT_S` 15 → 60s; wrapped `_start_backend` + `_wait_for_ready` in `try/except BaseException` inside `startup()`, calling `_stop_backend()` then `sys.exit(1)` on failure.
    - **Iter 2 (success, but orphan on shutdown)**: deps cached, uvicorn bound 5003 well under timeout, all endpoints 200, WebView visible, logo wired (see step 9b below). User closed window → briefcase dev exited 0 → BUT uvicorn child was reparented to launchd (PPID=1) and continued holding port 5003. Root cause: Toga's macOS shutdown goes through `NSApp.terminate_()` which bypasses Python's `atexit` hooks; the SIGTERM handlers I installed in the parent never fire because nothing sends a signal to the parent. **First (broken) patch attempt**: assigned `self.on_exit = self._on_exit` in `startup()` AND set `self.main_window.on_close = self._on_main_window_close` calling `self.exit()`. Both wrong: (a) Toga 0.5 `on_exit` is a method that the constructor wraps via `simple_handler`; assigning a callable in `startup()` after construction replaces the wrapper with an unwrapped method whose signature `(self, app, **kwargs)` doesn't match what `wrapped_handler` passes — so it never actually fires correctly. (b) `app.exit()` is the **unconditional** exit per `toga/app.py:450-456`; it does NOT invoke `on_exit`. The correct API is `app.request_exit()` which fires `on_exit` first.
    - **Iter 3 (success)**: rewrote to **override `on_exit` as a class method** in `XTRIEngine` (Toga's `__init__` then wraps it via `simple_handler` respecting MRO) and changed `_on_main_window_close` to call `self.request_exit()` instead of `self.exit()`. Verified end-to-end: launched, uvicorn at PID 23828, user closed window, briefcase dev exited 0, port 5003 free, no orphan. Defense in depth still in place: try/except wrapper in startup, on_exit override, on_close → request_exit, atexit + signal handlers as last-resort fallbacks.

9b. [x] **Logo wiring** (user provided `~/Desktop/logotipo.png`, 800×800 RGBA X-TRI logo):
    - `desktop/xtri-engine/icons/xtri-engine.png` — bundled as macOS app icon at build time. Briefcase reads via explicit `icon = "icons/xtri-engine"` field added to `[tool.briefcase.app.xtri-engine]`. Will only show in Dock/Finder/About panel after `briefcase build macOS` (step 10), NOT in `briefcase dev` mode.
    - `python_tri_v3_service/app/static/logotipo.png` — served at `/static/logotipo.png` (200 OK, 76016 bytes verified).
    - `python_tri_v3_service/app/static/index.html` — added `<img src="/static/logotipo.png" alt="X-TRI logo" style="height:48px;width:auto;margin-right:.8rem">` before `<h1>` in the existing `<header>`.
    - `python_tri_v3_service/pyproject.toml` — package-data widened from `static/*.html` to `static/*` so PNGs (and any future css/js) ship with the wheel.
    - **Mirror caveat**: in dev mode, the running uvicorn loads `app.main` from the briefcase-managed venv (`.briefcase/xtri-engine/dev.cpython-311-darwin/lib/python3.11/site-packages/app/`), not from the source tree. Both `index.html` and `logotipo.png` had to be mirrored into that location for the change to be visible without reinstalling. Cleaner long-term: install xtri-service as `pip install -e` editable in the briefcase venv. Tracked in Open Items.
10. [x] `briefcase build macOS` → first runnable `.app`. **Four iterations required**:
    - **Iter 1 (git license)**: Briefcase failed with "git not installed" — actually git was at `/usr/bin/git` but Apple's `xcrun` shim was blocking it because the Xcode license hadn't been accepted. Briefcase's GitPython probe got the license error and reported it as missing-git. **Fix**: user ran `sudo xcodebuild -license accept` in their own Terminal.app (the `!` shell in Claude has no TTY so sudo can't prompt — important harness limitation).
    - **Iter 2 (scipy wheel)**: Briefcase tried to resolve scipy with default min macOS 11.0 for arm64; scipy 1.11+ arm64 wheels require macOS 12.0+. **Fix**: added `min_os_version = "12.0"` AND `universal_build = false` to `[tool.briefcase.app.xtri-engine.macOS]` (single-user is arm64-only, no need for universal binaries which would also need x86_64 wheels for the same min OS).
    - **Iter 3 (icon warning)**: Build succeeded but warned "Unable to find icons/xtri-engine.icns". Briefcase 0.4 does NOT auto-convert PNG → ICNS for macOS — must supply `.icns` directly. **Fix**: generated `desktop/xtri-engine/icons/xtri-engine.icns` from the 800×800 PNG via `sips` + `iconutil` with all 10 standard sizes (16, 16@2x, 32, 32@2x, 128, 128@2x, 256, 256@2x, 512, 512@2x — the @2x of 512 is upscaled from 800 to 1024). Verified `Info.plist.CFBundleIconFile = "xtri-engine.icns"` and the `.icns` is at `Contents/Resources/`.
    - **Iter 4 (subprocess hangs in packaged app)**: First launch via `open` left the main app stuck in `_wait_for_ready` for 3 minutes; uvicorn child PIDs orphaned to launchd in state S, port never bound. **Root cause**: the dev-mode pattern `[sys.executable, "-m", "uvicorn", "app.main:app", ...]` is structurally broken in a packaged Briefcase app because **`sys.executable` resolves to the GUI-Stub binary, which is NOT a generic Python interpreter** — it ignores `-c` and `-m` flags (verified empirically: `XTRI Engine -c "print(1)"` returns nothing, the stub silently ignores the flag). The dev-mode test passed because dev mode runs against the host's real Python at `sys.executable`. **Fix**: rewrote `app.py` to drop subprocess entirely. uvicorn now runs **in a worker thread** within the main process via `uvicorn.Server` + `uvicorn.Config`, with a `_XTRIServer` subclass that no-ops `install_signal_handlers` (signal.signal only works on the main thread, and Toga owns it anyway). Cleanup is `server.should_exit = True` + `thread.join(timeout=5)`. No subprocess, no orphan risk, smaller mental model.
    - **Verified end-to-end**: `briefcase build macOS` produces `desktop/xtri-engine/build/xtri-engine/macos/app/XTRI Engine.app` (353 MB, arm64-thin, ad-hoc signed, identifier `com.gabaritai.xtri-engine`). Launching via `open` → port 5003 bound by the **main app process itself** (PID 32650, no children) → `/health` HTTP 200 in <1ms → `/static/logotipo.png` HTTP 200 (76016 bytes) → Toga MainWindow + WebView visible with X-TRI logo in both Dock (.icns) and WebView header (.png).
11. [x] Smoke test on real data:
    - LC.txt (54 students) processed via packaged .app with 2024 reference scale → all sanity checks passed against the PDF tables (lowest-theta @ tri_min, highest-theta near tri_max for their acertos band).
    - Data layer iteration during smoke test: user identified all four `TRI_MAXIMA_OFICIAL` constants were higher than the empirical 2024 maxes from `~/Desktop/ENEM 2024 - dificuldades.pdf`. Compared old (LC=800/CH=835/CN=875/MT=990) vs PDF (795.8/819.7/867.2/961.9). Updated to round-down conservative values: **LC=790, CH=815, CN=865, MT=960**.
    - Added a new `2024` year to `tri_tabela_referencia_multiano.json` with full 0-45 acertos data for LC (pooled INGLÊS+ESPANHOL: min=min, max=max, med=mean), CH, MT, and 0-44 for CN (PDF has no row 45). Build script archived at `/tmp/build_2024_year.py`.
    - **Slider math bug** caught during V3 Local validation: Theta Shift / Theta Scale were no-ops for Cohort-Anchored AND Percentil methods because both normalize against the cohort, and a uniform shift cancels out. Fixed by applying shift/scale **post-z-score** in `anchorCohortAnchored()` and **post-percentile** in `anchorPercentilTurma()` (via `normCdf` mapping). Confirmed via instrumented debug panel showing reanchor fired with correct values; chart now updates visibly when sliders move.
    - **UI layout fixes** caught alongside slider bug: dropdown text "Cohort-Anchored (z-score → tri_min/med/max)" was overflowing into adjacent grid cell. Shortened option label to "Cohort-Anchored" and added `min-width:0; overflow:hidden` containment to `.tune-item`. Chart x-axis label "Numero de Acertos" was rendering at the same y as the footer credit (1px apart) — fixed by bumping `pad.bottom` from 55 to 90 and repositioning both labels relative to plot bottom.
12. [x] `cp -r` (via `ditto`, more macOS-correct) to `/Applications/XTRI Engine.app` AND `~/Desktop/XTRI Engine.app` (Desktop staging copy for AirDrop to home Mac mini M4). Both 353 MB, ad-hoc signed, identifier `com.gabaritai.xtri-engine`. The `.icns` icon survived the rebuild via manual swap (Briefcase 0.4 only copies icons during `create`, not on subsequent `build` runs — known quirk).
13. [ ] **Optional cleanup**: `briefcase create macOS` for a clean from-scratch rebuild that doesn't depend on the manual icon swap surviving across rebuilds.

### Phase 8 — MICRODATA Analysis + Drill-down Navigator + Claude Reports [in_progress]

Two features described in the user's April 8 request (shared alongside `mente_mat_resultados.png` and `ENEM 2024 - dificuldades.pdf`):

**Feature A** — New third section that ingests the full ENEM microdata CSV (~1.6 GB), filters valid first-application students, samples 100-200K, runs V3 Local calibration, and renders per-area dashboards (scatter V3 vs reference, histogram, acertos-vs-nota) matching the reference PNG.

**Feature B** — Drill-down UI navigator by UF → City → School → Student with Claude API integration for natural-language reports at each level. School names resolved via the user's Supabase `inep_escolas` table (27,328 schools, keyed on `codigo_inep = CO_ESCOLA`).

**Phased plan** (9 sub-phases, ~13-14 sessions total). See full plan in the earlier `/plan` output. Condensed:

- P1 — Microdata loader module
- P2 — Parquet cache layer
- P3 — Sample V3 Local calibration pipeline
- P4 — matplotlib dashboard generator (+matplotlib dep, ~60 MB bundle growth)
- P5 — Backend `/api/v5/microdata/*` routes
- P6 — Frontend `/microdata` page
- P7 — Drill-down data API (`/api/v5/drilldown/*`, joins with `inep_escolas`)
- P8 — Frontend `/drilldown` navigator
- P9 — Claude API reports integration (raw httpx, no anthropic SDK dep)

**Phase 8 P1 + P2 complete (2026-04-08, commit pending)**:

- `python_tri_v3_service/app/microdata/` package created with 5 modules (`__init__.py`, `schema.py`, `filters.py`, `loader.py`, `cache.py`)
- **Smoke-tested against real 1.6 GB MICRODADOS_ENEM_2024.csv**: streams + filters + samples in 36.6s, returns 200K rows. Cache writes to 24 MB parquet in 0.3s, reloads in 0.08s (457x speedup over cold). 4,332,944 total rows read, 2,856,647 survived valid-student filter (65.9%). Area maxes match the 2024 PDF exactly (LC 795.8, CH 819.7, CN 867.2, MT 961.9).
- `pyarrow>=14.0.0` added to `python_tri_v3_service/pyproject.toml` for Parquet I/O. Installed wheel is 34 MB (smaller than the 50 MB estimate).
- 20 unit tests added in `tests/test_microdata_loader.py`, all passing. Full suite still 139 green.
- **Two real-world findings from the smoke test**:
  - `IN_TREINEIRO` is **not in 2024+ microdata** (INEP removed it along with `TP_SEXO`, `NU_IDADE`, etc). User confirmed: skip IN_TREINEIRO entirely. The loose primeira-aplicação fallback was removed — `is_primeira_aplicacao(provas_primeira=None)` now explicitly logs a warning and returns all-True (no filter). Strict mode via Supabase whitelist is the only reliable filter for 2024+.
  - `NO_ENTIDADE` (school name) is also **not in 2024+ microdata** — only `CO_ESCOLA` (numeric). User has the `inep_escolas` Supabase table (27,328 rows) with `codigo_inep → nome, uf, dependencia, localizacao` which covers this for Phase 7 drill-down.
- New helper: `load_primeira_aplicacao_provas(year, area=None)` in `loader.py` queries Supabase `enem_codigos_prova` for the whitelist (reuses the pattern from `calcular_transformacao.load_item_bank`).

**Phase 8 P3 complete (2026-04-08, commit pending)**:

- `python_tri_v3_service/app/microdata/calibration.py` — V3 Local calibration pipeline for the sample. Groups by `(area, co_prova [, tp_lingua])`, calibrates items per group, estimates per-student theta + nota, anchors via cohort method, joins with `nota_ref_*` from microdata.
- **Two LC-specific findings**:
  - INEP 2024 LC `TX_GABARITO_LC` is **50 chars** (5 English + 5 Spanish + 40 common), but `TX_RESPOSTAS_LC` is **45 chars** (one language + common). Added `_effective_lc_gabarito(full_gab, tp_lingua)` that slices the 50-char gabarito to the student's effective 45-char key. Tests for both English (TP_LINGUA=0) and Spanish (TP_LINGUA=1) layouts.
  - Per-area grouping for LC must include `TP_LINGUA` (not just `CO_PROVA_LC`) because English and Spanish students received different first-5 items even within the same prova color.
- **Two performance optimizations** (made smoke test 70-100x faster):
  - `calibrar_itens_fast` (in `calibration.py`, NOT touching `cientifico.py`): original `_neg_log_lik_item` looped Python-level over every student per likelihood eval (O(N×K)). Insight: the marginal log-likelihood for one item only depends on `(n_correct, n_wrong)` counts, so it collapses to `n_correct·log(marg_correct) + n_wrong·log(marg_wrong)` — **O(K)**. Mathematically equivalent.
  - `eap_theta_batch` (in `calibration.py`): original `eap_theta` was per-student. Replaced with one matmul `(N students, J items) @ (J items, K quad) = (N, K)` for all students at once.
  - Both are unit-tested to match the scalar versions within 1e-4 / 0.05 tolerance.
- **Smoke test on real 200K sample**: 51.3s total (LC 13s, CH 12s, CN 13s, MT 13s). Down from a >1 hour estimate with the original loops.
- **Calibration accuracy** (V3 Local cohort-anchored vs official INEP NU_NOTA_*):
  - LC: r=0.9495, MAE=43.5, RMSE=57.7
  - CH: r=0.9711, MAE=43.7, RMSE=55.5
  - CN: r=0.9300, MAE=35.3, RMSE=42.5
  - MT: r=0.9489, MAE=37.1, RMSE=46.7
  - Coverage: 199,585/200,000 (99.8%) — the 0.2% missing are from sub-500-student reaplicação groups.
- 18 new unit tests (`test_microdata_calibration.py`), all passing. Full suite 157 green.
- Calibrated parquet cache: 36.6 MB at `~/Library/Application Support/XTRI Engine/microdata_cache/2024/<src>_<filt>_calibrated.parquet`. Reload in 0.08s.

**Phase 8 P4 complete (2026-04-08, commit pending)**:

- `python_tri_v3_service/app/microdata/dashboards.py` — server-side matplotlib PNG generator. `matplotlib.use('Agg')` set at module-import time before any pyplot import (FastAPI worker threads have no GUI backend).
- Public functions: `render_area_dashboard(df, area)` (1×3 PNG per area), `render_full_report(df)` (4×3 PNG matching the user's `mente_mat_resultados.png` reference layout), `render_difficulty_table_png(df, area)`, `compute_difficulty_table(df, area)` (data-only, used by frontend).
- Per-area color palette matching V3 conventions: LC purple, CH green, CN blue, MT orange.
- `matplotlib>=3.7.0` added to xtri-service deps. Installed wheel is matplotlib 3.10.8 + dependencies (contourpy, cycler, kiwisolver, pillow, fonttools). +60-100 MB to .app bundle on next rebuild.
- 12 new unit tests covering valid PNG bytes (PNG magic header check), per-area rendering, full report, difficulty table, empty-data graceful handling, palette completeness. Full suite 169 green.
- **Smoke test on real 200K calibrated sample**:
  - Full report (4×3 PNG, 4 areas): **3.4s**, 415 KB
  - Per-area dashboards: 0.7-1.1s each, 105-118 KB
  - Difficulty tables: <0.5s each, 100-109 KB
  - Total render time for all 9 PNGs: ~7s
  - Output saved to `/tmp/microdata_dashboards/` for inspection
  - Visual: matches the user's reference image structure exactly. Correlations from the calibration are visible in scatter titles (LC r=0.950, CH r=0.971, CN r=0.930, MT r=0.949).

**Tier 2 (future, optional)**: native menu bar, file dialogs, Dock drag-drop, macOS notifications, Spotlight.

### Step 8 sub-decisions (recommended defaults, unless user overrides)

**Q8 — Port selection**: **(A) Hardcode 5003** — matches dev workflow, user is single-user.
- Alt (C): try 5003 first, fall back to random if busy. Switch to (C) only if 5003 conflicts in practice.

**Q9 — Subprocess lifecycle on app crash**: **(A) `atexit` + SIGTERM/SIGKILL signal handlers** — good enough for Tier 1, minimal code.
- Alt (B): pidfile at `~/Library/Application Support/XTRI Engine/uvicorn.pid` cleaned on next launch.

**Q10 — How to invoke uvicorn in dev + bundled**: **(B) `sys.executable -m uvicorn`** — bulletproof across `briefcase dev` and bundled `.app`.
- Alt (A): `uvicorn.run()` in a `multiprocessing` child. Clean but adds complexity.

### Step 8 implementation sketch

`desktop/xtri-engine/src/xtri_engine/app.py` becomes ~150 lines:

```python
import subprocess, sys, atexit, signal, time, socket
import httpx
import toga
from toga.style.pack import COLUMN, Pack

PORT = 5003  # Q8=A
HOST = "127.0.0.1"
HEALTH_URL = f"http://{HOST}:{PORT}/health"
STARTUP_TIMEOUT_S = 15

class XTRIEngine(toga.App):
    def startup(self):
        self._uvicorn: subprocess.Popen | None = None
        self._start_backend()          # Q10=B: sys.executable -m uvicorn
        self._wait_for_ready()         # poll /health
        self.main_window = toga.MainWindow(title=self.formal_name, size=(1400, 900))
        self.main_window.content = toga.WebView(
            url=f"http://{HOST}:{PORT}/",
            style=Pack(flex=1),
        )
        self.main_window.show()

    def _start_backend(self):
        self._uvicorn = subprocess.Popen(
            [sys.executable, "-m", "uvicorn",
             "app.main:app",
             "--host", HOST,
             "--port", str(PORT),
             "--log-level", "warning"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        atexit.register(self._stop_backend)  # Q9=A

    def _wait_for_ready(self):
        deadline = time.time() + STARTUP_TIMEOUT_S
        while time.time() < deadline:
            try:
                r = httpx.get(HEALTH_URL, timeout=1.0)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if self._uvicorn.poll() is not None:
                stderr = self._uvicorn.stderr.read().decode(errors="replace")
                raise RuntimeError(f"uvicorn exited early:\n{stderr}")
            time.sleep(0.2)
        raise TimeoutError(f"Backend didn't reach {HEALTH_URL} within {STARTUP_TIMEOUT_S}s")

    def _stop_backend(self):
        if self._uvicorn and self._uvicorn.poll() is None:
            self._uvicorn.send_signal(signal.SIGTERM)
            try:
                self._uvicorn.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._uvicorn.kill()

def main():
    return XTRIEngine()
```

**Step 8 open questions — resolved**:
- `/health` endpoint: **exists** at `app/main.py:170`. `_wait_for_ready` polls it directly.
- `httpx` dependency: comes in transitively via `supabase` (supabase-py uses httpx internally). Acceptable for Tier 1; flag in Open Items so future dep upgrades don't silently break the readiness probe.

**Current blocker (resolved 2026-04-07, round 2)**: Twice the user pasted AI-generated "authorized sequences" that contradicted Tier 1 (generic Python app scaffold with `requests`/`pandas`/`app.py`/`utils/`/`assets/`). Flagged both times. User confirmed "Lets go with A" → Tier 1 sequence above. User later picked Toga defaults, Option 1 package install, and said "dont worry about" an accidentally-pasted password.

## Files Created/Modified Tracking

| File | Status | Notes |
|------|--------|-------|
| task_plan.md | created | This file |
| findings.md | created | Analysis of brittleness |
| progress.md | created | Session log |
