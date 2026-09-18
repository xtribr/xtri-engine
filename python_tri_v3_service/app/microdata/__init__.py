"""
Microdata analysis package — streams the official ENEM microdata CSV from INEP,
filters to valid first-application students, random-samples, calibrates via
V3 Local, and generates per-area validation dashboards.

Public surface:
    from app.microdata import load_microdata, MicrodataSchema
    from app.microdata.cache import load_cached, save_cached, cache_key
    from app.microdata.filters import is_valid_student, is_primeira_aplicacao

See Phase 7 notes in task_plan.md for the phased implementation plan.
"""

from app.microdata.breakdown import compute_breakdown
from app.microdata.calibration import (
    calibrate_all_areas,
    calibrate_area_group,
    calibration_summary,
)
from app.microdata.dashboard_data import compute_dashboard_data
from app.microdata.dashboards import (
    compute_difficulty_table,
    render_area_dashboard,
    render_difficulty_table_png,
    render_full_report,
)
from app.microdata.excel_export import build_microdata_csv, build_microdata_excel
from app.microdata.loader import load_microdata
from app.microdata.psychometrics_2025 import (
    Enem2025Paths,
    PsychometricReport,
    analyze_enem_2025_area,
    export_psychometric_report,
    resolve_enem_2025_paths,
)
from app.microdata.report import (
    build_full_payload,
    build_full_prompt,
    build_payload,
    build_prompt,
    generate_full_report,
    generate_report,
)
from app.microdata.schema import MicrodataSchema, detect_schema

__all__ = [
    "load_microdata",
    "MicrodataSchema",
    "detect_schema",
    "calibrate_all_areas",
    "calibrate_area_group",
    "calibration_summary",
    "compute_difficulty_table",
    "render_area_dashboard",
    "render_full_report",
    "render_difficulty_table_png",
    "build_microdata_excel",
    "build_microdata_csv",
    "compute_dashboard_data",
    "compute_breakdown",
    "build_full_payload",
    "build_full_prompt",
    "build_payload",
    "build_prompt",
    "generate_full_report",
    "generate_report",
    "Enem2025Paths",
    "PsychometricReport",
    "analyze_enem_2025_area",
    "export_psychometric_report",
    "resolve_enem_2025_paths",
]
