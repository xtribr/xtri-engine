"""
Endpoints V5 — Microdata Analysis (Phase 8) + status do runtime local.

Estas rotas ligam o pipeline app/microdata/ (loader → cache → calibração →
dashboards matplotlib) à superfície FastAPI. O frontend dispara o pipeline
longo via POST /prepare, faz poll em /job/{id} e busca os dashboards PNG em
base64 quando status==done.
"""

import base64
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

import io

# Indireção proposital: os testes fazem mock.patch em "app.main.default_csv_path"
# e "app.main.submit_microdata_job"; resolver via módulo app.main preserva esse
# contrato.
from .. import main as _main
from .. import state
from ..microdata import (
    build_microdata_csv,
    build_microdata_excel,
    calibrate_all_areas,
    calibration_summary,
    compute_breakdown,
    compute_dashboard_data,
    compute_difficulty_table,
    detect_schema,
    generate_full_report,
    generate_report,
    load_microdata,
    render_area_dashboard,
    render_full_report,
)
from ..microdata.cache import (
    cache_key,
    load_cached,
    save_cached,
)
from ..microdata.jobs import (
    JOB_DONE,
    JOB_ERROR,
    JOB_QUEUED,
    JOB_RUNNING,
    job_manager,
)
from ..microdata.loader import (
    discover_available_years,
    load_primeira_aplicacao_provas,
    MICRODATA_DIR,
)
from ..models import (
    MicrodataAreaDashboardResponse,
    MicrodataAreaSummary,
    MicrodataAvailabilityResponse,
    MicrodataDifficultyRow,
    MicrodataFullReportResponse,
    MicrodataJobStatusResponse,
    MicrodataPrepareRequest,
    MicrodataPrepareResponse,
    MicrodataYearStatus,
    RuntimeDesktopStatusResponse,
    RuntimeDirectoryStatus,
    RuntimeJobStatus,
    RuntimeModelStatus,
)
from ..runtime.local_paths import describe_runtime_layout, ensure_app_support_layout
from ..storage.model_registry import get_active_model, list_active_models

logger = logging.getLogger(__name__)

router = APIRouter()


def _runtime_directory_models() -> dict[str, RuntimeDirectoryStatus]:
    """Normaliza o snapshot de diretórios locais para o response model."""

    snapshot = describe_runtime_layout()
    return {
        name: RuntimeDirectoryStatus(**payload)
        for name, payload in snapshot.items()
    }


def _runtime_job_counts(jobs: list) -> dict[str, int]:
    """Conta jobs por status para o desktop não depender de lógica no cliente."""

    counts = {
        "queued": 0,
        "running": 0,
        "done": 0,
        "error": 0,
    }
    for job in jobs:
        counts[job.status] = counts.get(job.status, 0) + 1
    return counts


def _runtime_model_statuses() -> list[RuntimeModelStatus]:
    """Converte o registry bruto em uma lista ordenada de modelos ativos."""

    models: list[RuntimeModelStatus] = []
    for entry in list_active_models().values():
        artifact_dir = Path(entry["artifact_dir"])
        status = "pronto" if artifact_dir.is_dir() else "desatualizado"
        models.append(
            RuntimeModelStatus(
                ano_referencia=int(entry["ano_referencia"]),
                area=str(entry["area"]),
                scope_id=str(entry["scope_id"]),
                modelo=str(entry["modelo"]),
                artifact_id=entry.get("artifact_id"),
                artifact_dir=str(artifact_dir),
                saved_at_utc=entry.get("saved_at_utc"),
                status=status,
            )
        )
    return sorted(
        models,
        key=lambda model: (
            model.ano_referencia,
            model.area,
            model.scope_id,
            model.saved_at_utc or "",
        ),
        reverse=True,
    )


def _runtime_area_status(
    *,
    ano_referencia: int,
    simulado_id: Optional[str],
    jobs: list,
) -> dict[str, str]:
    """Deriva o status operacional por área para o desktop."""

    if not simulado_id:
        return {}

    has_running = any(job.status in {JOB_QUEUED, JOB_RUNNING} for job in jobs)
    has_error = any(job.status == JOB_ERROR for job in jobs)
    statuses: dict[str, str] = {}

    for area in ("LC", "CH", "CN", "MT"):
        active = get_active_model(
            ano_referencia=ano_referencia,
            area=area,
            scope_id=simulado_id,
        )
        if active is not None:
            statuses[area] = (
                "pronto" if Path(active["artifact_dir"]).is_dir() else "desatualizado"
            )
        elif has_running:
            statuses[area] = "calibrando"
        elif has_error:
            statuses[area] = "falhou"
        else:
            statuses[area] = "fallback_heuristico"

    return statuses


def _run_microdata_pipeline(
    job_id: str, year: int, sample_size: int, rng_seed: int, force_rerun: bool
) -> None:
    """Background runner for one microdata pipeline job.

    Runs in a thread (via asyncio.run_in_executor) so it doesn't block the
    uvicorn worker. Updates job state via the singleton job_manager.
    """
    try:
        job_manager.mark_running(job_id)
        csv_path = _main.default_csv_path(year)
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Microdata CSV not found at {csv_path}. "
                f"Drop MICRODADOS_ENEM_{year}.csv in {MICRODATA_DIR}/"
            )

        raw_key = cache_key(csv_path, year=year, sample_size=sample_size, rng_seed=rng_seed)
        calib_key = raw_key.with_stage("calibrated")

        # Stage 1: load + filter + sample (or load from cache)
        job_manager.update_progress(job_id, progress=0.05, stage="loading microdata")
        if not force_rerun:
            cached_calib = load_cached(calib_key)
            if cached_calib is not None:
                # Calibrated cache hit — done immediately
                job_manager.mark_done(job_id, result_key=calib_key)
                return

        df = None
        if not force_rerun:
            df = load_cached(raw_key)
        if df is None:
            job_manager.update_progress(job_id, progress=0.10, stage="streaming CSV")
            df = load_microdata(
                year=year,
                sample_size=sample_size,
                rng_seed=rng_seed,
                provas_primeira=load_primeira_aplicacao_provas(year),
            )
            save_cached(df, raw_key, source_path=csv_path, filters_description="v5 raw")
        job_manager.update_progress(job_id, progress=0.30, stage="raw sample ready")

        # Stage 2: calibrate all areas
        schema = detect_schema(csv_path)
        tabelas_multiano = state.get_tabelas_multiano()
        tabela = tabelas_multiano.get(year)
        if tabela is None:
            raise ValueError(
                f"No reference table loaded for year {year}. "
                f"Available: {sorted(tabelas_multiano.keys())}"
            )

        def on_progress(area, idx, total, n_done):
            # Each area is one quarter of the calibration phase
            area_idx = ["LC", "CH", "CN", "MT"].index(area)
            within_area = idx / total if total > 0 else 1.0
            overall = 0.30 + 0.65 * (area_idx + within_area) / 4
            job_manager.update_progress(
                job_id, progress=overall, stage=f"calibrating {area} ({idx}/{total})"
            )

        enriched = calibrate_all_areas(df, schema, tabela, progress_callback=on_progress)
        job_manager.update_progress(job_id, progress=0.95, stage="caching results")
        save_cached(
            enriched, calib_key, source_path=csv_path, filters_description="v5 calibrated"
        )

        job_manager.mark_done(job_id, result_key=calib_key)
    except Exception as exc:  # noqa: BLE001
        job_manager.mark_error(job_id, f"{type(exc).__name__}: {exc}")


def _summary_to_model(summary_dict: dict) -> MicrodataAreaSummary:
    """Convert calibration_summary() output for one area into the response model."""
    return MicrodataAreaSummary(
        n=int(summary_dict.get("n", 0)),
        correlation=summary_dict.get("correlation"),
        mae=summary_dict.get("mae"),
        rmse=summary_dict.get("rmse"),
        mean_v3=summary_dict.get("mean_v3"),
        mean_ref=summary_dict.get("mean_ref"),
        mean_shift=summary_dict.get("mean_shift"),
    )


@router.get("/api/v5/microdata/anos-disponiveis", response_model=MicrodataAvailabilityResponse)
async def v5_anos_disponiveis():
    """List ENEM years that have a MICRODADOS_ENEM_<year>.csv on disk."""
    return MicrodataAvailabilityResponse(
        anos=discover_available_years(),
        microdata_dir=str(MICRODATA_DIR),
    )


@router.get("/api/v5/microdata/status/{year}", response_model=MicrodataYearStatus)
async def v5_microdata_status(year: int):
    """Report file presence + cache state for one year. Always 200 (no 404)."""
    csv_path = _main.default_csv_path(year)
    csv_exists = csv_path.is_file()

    csv_size_mb = None
    csv_mtime = None
    raw_present = False
    calib_present = False
    n_sample = None

    if csv_exists:
        stat = csv_path.stat()
        csv_size_mb = round(stat.st_size / (1024 * 1024), 1)
        csv_mtime = stat.st_mtime
        # Check both raw and calibrated cache existence for the default 200K sample
        try:
            raw_key = cache_key(csv_path, year=year, sample_size=200_000, rng_seed=42)
            raw_present = raw_key.parquet_path.is_file()
            calib_present = raw_key.with_stage("calibrated").parquet_path.is_file()
            if calib_present:
                n_sample = 200_000
            elif raw_present:
                n_sample = 200_000
        except Exception:  # noqa: BLE001
            pass

    return MicrodataYearStatus(
        year=year,
        csv_path=str(csv_path),
        csv_exists=csv_exists,
        csv_size_mb=csv_size_mb,
        csv_mtime=csv_mtime,
        cache_raw_present=raw_present,
        cache_calibrated_present=calib_present,
        n_sample=n_sample,
    )


@router.post("/api/v5/microdata/prepare", response_model=MicrodataPrepareResponse)
async def v5_microdata_prepare(req: MicrodataPrepareRequest):
    """Kick off the load + calibrate pipeline asynchronously.

    Returns immediately with a job_id. Frontend polls /job/{id} for progress.
    If the calibrated cache for this (year, sample_size, seed) already exists
    AND force_rerun is False, the job completes synchronously inside the
    executor and the next /job/{id} call returns ``done``.
    """
    csv_path = _main.default_csv_path(req.year)
    if not csv_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Microdata CSV not found at {csv_path}. "
                f"Download MICRODADOS_ENEM_{req.year}.csv from INEP and copy to "
                f"{MICRODATA_DIR}/"
            ),
        )

    job = job_manager.create(
        year=req.year, sample_size=req.sample_size, rng_seed=req.rng_seed
    )

    # Enfileirar no executor dedicado do runtime local.
    _main.submit_microdata_job(
        _run_microdata_pipeline,
        job_id=job.job_id,
        year=req.year,
        sample_size=req.sample_size,
        rng_seed=req.rng_seed,
        force_rerun=req.force_rerun,
    )

    return MicrodataPrepareResponse(
        job_id=job.job_id,
        status=job.status,
        message=f"Pipeline started for year {req.year}, sample={req.sample_size}",
    )


@router.get("/api/v5/microdata/job/{job_id}", response_model=MicrodataJobStatusResponse)
async def v5_microdata_job(job_id: str):
    """Poll a running pipeline job."""
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return MicrodataJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        stage=job.stage,
        started_at=job.started_at,
        elapsed_seconds=job.elapsed_seconds,
        error=job.error,
    )


@router.get("/api/v5/runtime/status", response_model=RuntimeDesktopStatusResponse)
async def v5_runtime_status(
    ano_referencia: int = 2024,
    simulado_id: Optional[str] = None,
):
    """Snapshot do runtime local para o desktop/WebView."""

    ensure_app_support_layout()
    jobs = job_manager.snapshot()
    directories = _runtime_directory_models()

    return RuntimeDesktopStatusResponse(
        app_support_dir=directories["app_support"].path,
        directories=directories,
        models=_runtime_model_statuses(),
        job_counts=_runtime_job_counts(jobs),
        jobs=[
            RuntimeJobStatus(
                job_id=job.job_id,
                year=job.year,
                sample_size=job.sample_size,
                rng_seed=job.rng_seed,
                status=job.status,
                progress=job.progress,
                stage=job.stage,
                elapsed_seconds=job.elapsed_seconds,
                error=job.error,
            )
            for job in jobs
        ],
        ano_referencia=ano_referencia,
        simulado_id=simulado_id,
        status_areas=_runtime_area_status(
            ano_referencia=ano_referencia,
            simulado_id=simulado_id,
            jobs=jobs,
        ),
    )


def _load_calibrated_or_404(year: int) -> tuple:
    """Helper: load the calibrated DataFrame for a year or raise 404.

    Returns ``(df, csv_path)``. Used by both dashboard endpoints below.
    """
    csv_path = _main.default_csv_path(year)
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail=f"Microdata CSV not found for year {year}")
    raw_key = cache_key(csv_path, year=year, sample_size=200_000, rng_seed=42)
    calib_key = raw_key.with_stage("calibrated")
    df = load_cached(calib_key)
    if df is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No calibrated cache for year {year}. "
                f"POST /api/v5/microdata/prepare first."
            ),
        )
    return df, csv_path


@router.get(
    "/api/v5/microdata/dashboard/{year}",
    response_model=MicrodataAreaDashboardResponse,
)
async def v5_microdata_dashboard(year: int, area: str):
    """Single-area dashboard: 1×3 PNG + summary + difficulty table."""
    area = area.upper()
    if area not in ("LC", "CH", "CN", "MT"):
        raise HTTPException(status_code=400, detail=f"Invalid area: {area}")

    df, _ = _load_calibrated_or_404(year)

    summary = calibration_summary(df).get(area.lower(), {"n": 0})
    png_bytes = render_area_dashboard(df, area)
    png_b64 = base64.b64encode(png_bytes).decode("ascii")
    diff_rows = compute_difficulty_table(df, area)

    return MicrodataAreaDashboardResponse(
        year=year,
        area=area,
        n_students=int(summary.get("n", 0)),
        summary=_summary_to_model(summary),
        dashboard_png_base64=png_b64,
        difficulty_table=[MicrodataDifficultyRow(**r) for r in diff_rows],
    )


@router.get(
    "/api/v5/microdata/full-report/{year}",
    response_model=MicrodataFullReportResponse,
)
async def v5_microdata_full_report(year: int):
    """4×3 full-report PNG + per-area summaries."""
    df, _ = _load_calibrated_or_404(year)

    summaries_raw = calibration_summary(df)
    summaries_models = {area: _summary_to_model(s) for area, s in summaries_raw.items()}
    png_bytes = render_full_report(df)
    png_b64 = base64.b64encode(png_bytes).decode("ascii")

    return MicrodataFullReportResponse(
        year=year,
        n_students=len(df),
        summaries=summaries_models,
        full_report_png_base64=png_b64,
    )


@router.get("/api/v5/microdata/dashboard-data/{year}")
async def v5_microdata_dashboard_data(year: int, area: str):
    """INEP-only JSON payload for the interactive Plotly frontend.

    Returns histogram bins, scatter points (acertos × nota colored by acert_dif),
    and per-acertos boxplot data — all computed from the official ``NU_NOTA_*``
    columns in the microdata. NO V3 Local references; this endpoint is for the
    Phase 8 microdata page which is strictly an INEP-only analysis.

    Use this instead of /dashboard/{year} (which returns a static base64 PNG).
    """
    area = area.upper()
    if area not in ("LC", "CH", "CN", "MT"):
        raise HTTPException(status_code=400, detail=f"Invalid area: {area}")
    df, _ = _load_calibrated_or_404(year)
    payload = compute_dashboard_data(df, area)
    payload["year"] = year
    return payload


@router.get("/api/v5/microdata/breakdown/{year}")
async def v5_microdata_breakdown(
    year: int,
    area: str,
    by: str = "uf",
    min_n: Optional[int] = None,
    limit: Optional[int] = None,
):
    """Per-group INEP aggregation for the drill-down UI.

    Query params:
        area:  LC | CH | CN | MT
        by:    uf | municipio | escola
        min_n: drop groups with fewer students (default 1 for uf, 5 otherwise)
        limit: max rows (default None for uf, 500 municipio, 1000 escola)

    Returns a list of dicts sorted by median desc. Stats come exclusively from
    the official ``NU_NOTA_*`` columns (``nota_ref_*``) — V3 is ignored.
    """
    area = area.upper()
    if area not in ("LC", "CH", "CN", "MT"):
        raise HTTPException(status_code=400, detail=f"Invalid area: {area}")
    by_lower = by.lower()
    if by_lower not in ("uf", "municipio", "escola"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid by: {by} (must be uf, municipio, or escola)",
        )
    df, _ = _load_calibrated_or_404(year)
    rows = compute_breakdown(
        df, area, group_by=by_lower, min_n=min_n, limit=limit  # type: ignore[arg-type]
    )
    return {
        "year": year,
        "area": area,
        "by": by_lower,
        "count": len(rows),
        "rows": rows,
    }


@router.post("/api/v5/microdata/report/{year}")
async def v5_microdata_report(year: int, area: str):
    """Generate a Claude-authored markdown narrative for one area of one year.

    Assembles INEP summary stats + difficulty-table highlights + top-10 UFs,
    cities and schools into a grounded prompt, then calls the Anthropic API
    (model defaults to ``claude-sonnet-4-6``, overridable via ``XTRI_REPORT_MODEL``).

    Requires ``ANTHROPIC_API_KEY`` to be set either in the environment or in
    ``~/Library/Application Support/XTRI Engine/.env``. Returns ``501`` when
    the key is missing so the frontend can show a clear fallback message.
    """
    area = area.upper()
    if area not in ("LC", "CH", "CN", "MT"):
        raise HTTPException(status_code=400, detail=f"Invalid area: {area}")
    df, _ = _load_calibrated_or_404(year)

    try:
        result = generate_report(df, year, area)
    except RuntimeError as exc:
        # Missing API key → 501 Not Implemented (feature unavailable)
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="anthropic package not installed in this build",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("AI report generation failed")
        raise HTTPException(
            status_code=502, detail=f"Claude API call failed: {exc}"
        ) from exc

    return result


@router.post("/api/v5/microdata/report/{year}/full")
async def v5_microdata_report_full(year: int):
    """Generate a single Claude narrative covering all four ENEM areas.

    Builds payloads for LC, CH, CN and MT simultaneously, then sends one
    comprehensive prompt to Claude Opus requesting a comparative analysis.
    Expect 30–90 s of processing time.  Returns ``501`` when the API key is
    missing, ``502`` on API failure.
    """
    df, _ = _load_calibrated_or_404(year)

    try:
        result = generate_full_report(df, year)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="anthropic package not installed in this build",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Full AI report generation failed")
        raise HTTPException(
            status_code=502, detail=f"Claude API call failed: {exc}"
        ) from exc

    return result


@router.get("/api/v5/microdata/export-excel/{year}")
async def v5_microdata_export_excel(year: int, include_students: bool = False):
    """Stream the multi-sheet Excel report for one ENEM year.

    Sheets:
        Resumo               — title + per-area metrics
        Relatorio Completo   — embedded 4×3 full report PNG
        <area> sheets        — embedded per-area dashboard PNG + INEP empirical
                               difficulty table (one per LC/CH/CN/MT)
        Alunos (optional)    — full per-student data, only when
                               ``?include_students=true``. Adds ~50s + 25 MB.
                               For raw per-student data the CSV endpoint is
                               much faster — see /api/v5/microdata/export-csv/

    Default mode (without ``include_students``) builds in ~10s and is ~5 MB.

    Returns 404 if the calibrated cache for ``year`` is not on disk.
    """
    df, _ = _load_calibrated_or_404(year)
    summaries = calibration_summary(df)
    xlsx_bytes = build_microdata_excel(
        df, year, summaries, include_students=include_students
    )

    suffix = "_full" if include_students else ""
    filename = f"xtri_microdados_{year}{suffix}.xlsx"
    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(xlsx_bytes)),
        },
    )


@router.get("/api/v5/microdata/export-csv/{year}")
async def v5_microdata_export_csv(year: int):
    """Stream the per-student CSV with diff columns for one ENEM year.

    Much faster than the Excel "Alunos" sheet (pandas to_csv vs openpyxl
    serialization is ~30x). UTF-8 with BOM so Excel auto-detects encoding.
    Use this when you need the raw per-student data; use the Excel endpoint
    for the dashboards + summary report.

    Returns 404 if the calibrated cache for ``year`` is not on disk.
    """
    df, _ = _load_calibrated_or_404(year)
    csv_bytes = build_microdata_csv(df)
    filename = f"xtri_microdados_alunos_{year}.csv"
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(csv_bytes)),
        },
    )
