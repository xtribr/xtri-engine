"""
Claude-powered narrative report for a microdata year+area.

Builds a compact, factual summary of the INEP microdata (dashboard stats,
difficulty table key points, top-5 UFs / cities / schools) and asks
Claude to write a structured pedagogical narrative in Portuguese. The
output is markdown intended for rendering in the frontend or saving as .md.

The module never fabricates data — the prompt is filled with real aggregated
numbers from :mod:`app.microdata.dashboard_data`, :mod:`app.microdata.dashboards`,
and :mod:`app.microdata.breakdown`. Claude is explicitly told to stay grounded
in these numbers and not speculate.

API key lookup order:
1. ``ANTHROPIC_API_KEY`` environment variable
2. ``~/Library/Application Support/XTRI Engine/.env`` via python-dotenv

Raises :class:`RuntimeError` with a clear message when the key is missing so
the FastAPI endpoint can surface a 501 Not Implemented.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from app.microdata.breakdown import compute_breakdown
from app.microdata.dashboard_data import compute_dashboard_data
from app.microdata.dashboards import compute_difficulty_table
from app.microdata.schema import AREAS

logger = logging.getLogger(__name__)

# Default model for report generation. Sonnet 4.6 is the cost/quality sweet
# spot for structured narrative writing with rich data. Can be overridden via
# the ``XTRI_REPORT_MODEL`` environment variable for experimentation.
DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

AREA_LABELS = {
    "LC": "Linguagens, Códigos e suas Tecnologias",
    "CH": "Ciências Humanas e suas Tecnologias",
    "CN": "Ciências da Natureza e suas Tecnologias",
    "MT": "Matemática e suas Tecnologias",
}


@dataclass(frozen=True)
class ReportPayload:
    """Input payload for :func:`build_prompt` — all the aggregated stats."""

    year: int
    area: str
    summary: dict[str, Any]
    difficulty_highlights: list[dict[str, Any]]
    top_ufs: list[dict[str, Any]]
    top_municipios: list[dict[str, Any]]
    top_escolas: list[dict[str, Any]]


def build_payload(df: pd.DataFrame, year: int, area: str) -> ReportPayload:
    """Assemble the facts Claude will summarize, pulling from the cache df.

    Args:
        df: Calibrated DataFrame (same as the one feeding the dashboards).
        year: ENEM year (for display in the prompt).
        area: Area code (LC, CH, CN, MT). Must be uppercase.

    Returns:
        A :class:`ReportPayload` ready to feed into :func:`build_prompt`.

    Raises:
        ValueError: If ``area`` is unknown.
    """
    if area not in AREAS:
        raise ValueError(f"Unknown area: {area}")

    dash = compute_dashboard_data(df, area)
    summary = dash["summary"]

    # Difficulty table highlights: pick ~6 rows at evenly-spaced acertos counts
    diff_rows = compute_difficulty_table(df, area)
    highlights = _pick_difficulty_highlights(diff_rows)

    top_ufs = compute_breakdown(df, area, group_by="uf", limit=10)[:10]
    top_municipios = compute_breakdown(df, area, group_by="municipio", limit=10)[:10]
    top_escolas = compute_breakdown(df, area, group_by="escola", limit=10)[:10]

    return ReportPayload(
        year=year,
        area=area,
        summary=summary,
        difficulty_highlights=highlights,
        top_ufs=top_ufs,
        top_municipios=top_municipios,
        top_escolas=top_escolas,
    )


def build_prompt(payload: ReportPayload) -> str:
    """Construct the Claude user prompt with all factual anchors.

    We embed the numbers as tight tables so Claude doesn't have to re-format
    them, and give explicit instructions to stay grounded.
    """
    s = payload.summary
    area_label = AREA_LABELS.get(payload.area, payload.area)

    lines: list[str] = []
    lines.append(
        f"Você é um analista pedagógico sênior do INEP. Escreva um relatório narrativo em"
        f" **português do Brasil** sobre os microdados oficiais do ENEM {payload.year} —"
        f" área de **{area_label}** ({payload.area})."
    )
    lines.append("")
    lines.append(
        "Baseie-se EXCLUSIVAMENTE nos números abaixo. Não invente estatísticas, não"
        " generalize para outras áreas, não cite estudos externos. Se um valor não"
        " estiver listado, diga explicitamente que o dado não foi fornecido."
    )
    lines.append("")
    lines.append("## Estatísticas Descritivas (NU_NOTA oficial INEP)")
    lines.append("")
    lines.append(f"- N alunos válidos: **{s.get('n', 0):,}**".replace(",", "."))
    lines.append(f"- Média: **{s.get('mean')}**")
    lines.append(f"- Mediana: **{s.get('median')}**")
    lines.append(f"- P25: {s.get('p25')}")
    lines.append(f"- P75: {s.get('p75')}")
    lines.append(f"- Mínimo: {s.get('min')}")
    lines.append(f"- Máximo: {s.get('max')}")
    lines.append("")
    lines.append("## Tabela de Dificuldades (INEP empírico)")
    lines.append("")
    lines.append("| Acertos | Freq | TRI min | TRI med | TRI max |")
    lines.append("|--------:|-----:|--------:|--------:|--------:|")
    for row in payload.difficulty_highlights:
        lines.append(
            f"| {row.get('acertos')} | {row.get('freq')} | "
            f"{row.get('tri_min')} | **{row.get('tri_med')}** | {row.get('tri_max')} |"
        )
    lines.append("")
    lines.append("## Top 10 Estados (UF) por mediana")
    lines.append("")
    lines.append("| UF | N | Mediana | Média | P25 | P75 |")
    lines.append("|:--|--:|--:|--:|--:|--:|")
    for r in payload.top_ufs:
        lines.append(
            f"| {r['label']} | {r['n']} | **{r['median']}** | "
            f"{r['mean']} | {r['p25']} | {r['p75']} |"
        )
    lines.append("")
    lines.append("## Top 10 Municípios por mediana")
    lines.append("")
    lines.append("| Município | UF | N | Mediana | Média |")
    lines.append("|:--|:--|--:|--:|--:|")
    for r in payload.top_municipios:
        lines.append(
            f"| {r['label']} | {r.get('uf', '—')} | {r['n']} | "
            f"**{r['median']}** | {r['mean']} |"
        )
    lines.append("")
    lines.append("## Top 10 Escolas por mediana")
    lines.append("")
    lines.append("| Código | Município | UF | Rede | Local | N | Mediana |")
    lines.append("|:--|:--|:--|:--|:--|--:|--:|")
    for r in payload.top_escolas:
        lines.append(
            f"| {r['label']} | {r.get('municipio', '—')} | {r.get('uf', '—')} | "
            f"{r.get('dependencia', '—')} | {r.get('localizacao', '—')} | "
            f"{r['n']} | **{r['median']}** |"
        )
    lines.append("")
    lines.append("## Instruções para o relatório")
    lines.append("")
    lines.append(
        "Estruture a resposta em markdown com as seções:\n"
        "\n"
        "1. **Resumo Executivo** — 2-3 parágrafos sintetizando as principais descobertas.\n"
        "2. **Distribuição Geral** — interpretação da média, mediana, amplitude (max−min)"
        " e da dispersão (IQR = P75−P25). Comente se a distribuição aparenta ser"
        " assimétrica.\n"
        "3. **Análise da Tabela de Dificuldades** — explique o que significa a progressão"
        " TRI med conforme acertos sobe, e aponte qualquer anomalia (plateaus, quebras).\n"
        "4. **Ranking Geográfico** — compare os top 10 Estados, destacando disparidades"
        " regionais. Identifique se há concentração por região (Sul/Sudeste vs. Norte/Nordeste).\n"
        "5. **Perfil dos Municípios e Escolas de Alto Desempenho** — descreva o perfil"
        " dos top 10 escolas (rede pública/privada, localização urbana/rural, concentração"
        " geográfica). Cite municípios específicos.\n"
        "6. **Implicações Pedagógicas** — 2-3 insights acionáveis para gestores educacionais,"
        " sem moralizar ou sugerir políticas públicas específicas.\n"
        "\n"
        "IMPORTANTE: use apenas os dados da amostra acima. Se for mencionar um valor,"
        " escreva o número exato. Não arredonde exagerado nem cite estatísticas"
        " hipotéticas. Idioma: português do Brasil. Tom: técnico mas acessível."
    )

    return "\n".join(lines)


def generate_report(
    df: pd.DataFrame,
    year: int,
    area: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Build the prompt and call the Claude API.

    Returns a dict with the generated markdown + metadata::

        {
            "year": 2024,
            "area": "MT",
            "model": "claude-sonnet-4-5",
            "prompt_tokens": 1234,   # best-effort, may be absent
            "output_tokens": 567,
            "markdown": "# Resumo Executivo\\n..."
        }

    Raises:
        RuntimeError: If ``ANTHROPIC_API_KEY`` is not set and no explicit
            ``api_key`` is passed. The FastAPI endpoint maps this to 501.
        ImportError: If the ``anthropic`` package isn't installed.
    """
    resolved_key = api_key or _load_api_key()
    if not resolved_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not configured. Set it in the environment or in "
            "~/Library/Application Support/XTRI Engine/.env to enable AI reports."
        )

    try:
        from anthropic import Anthropic  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is not installed. Run `pip install anthropic`"
            " or rebuild the Docker image."
        ) from exc

    payload = build_payload(df, year, area.upper())
    prompt = build_prompt(payload)
    model_name = model or os.environ.get("XTRI_REPORT_MODEL", DEFAULT_MODEL)

    logger.info(
        "Generating report year=%d area=%s model=%s prompt_chars=%d",
        year,
        area,
        model_name,
        len(prompt),
    )

    client = Anthropic(api_key=resolved_key)
    msg = client.messages.create(
        model=model_name,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    markdown_parts: list[str] = []
    for block in msg.content:
        text = getattr(block, "text", None)
        if text:
            markdown_parts.append(text)
    markdown = "\n".join(markdown_parts).strip()

    usage = getattr(msg, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", None) if usage else None
    output_tokens = getattr(usage, "output_tokens", None) if usage else None

    return {
        "year": year,
        "area": area.upper(),
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "markdown": markdown,
    }


def _load_api_key() -> Optional[str]:
    """Look up the Anthropic API key from env or the XTRI .env file."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key

    try:
        from dotenv import dotenv_values  # noqa: PLC0415
    except ImportError:
        return None

    env_path = _xtri_env_path()
    if not env_path.is_file():
        return None
    values = dotenv_values(env_path)
    return values.get("ANTHROPIC_API_KEY") or None


def _xtri_env_path():
    from pathlib import Path  # noqa: PLC0415

    return Path.home() / "Library" / "Application Support" / "XTRI Engine" / ".env"


def _pick_difficulty_highlights(
    rows: list[dict[str, Any]], n_target: int = 8
) -> list[dict[str, Any]]:
    """Sample ~N rows from a difficulty table at evenly-spaced acertos counts.

    Always includes the first and last rows. If the table is already smaller
    than ``n_target``, returns it unchanged.
    """
    if len(rows) <= n_target:
        return rows
    sorted_rows = sorted(rows, key=lambda r: r.get("acertos", 0))
    step = max(1, (len(sorted_rows) - 1) // (n_target - 1))
    picked = sorted_rows[::step]
    # Ensure last row is included
    if picked[-1] is not sorted_rows[-1]:
        picked.append(sorted_rows[-1])
    return picked
