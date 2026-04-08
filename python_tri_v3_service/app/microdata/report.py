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

# Default model for report generation. Opus 4.6 delivers the depth needed for
# a rigorous pedagogical narrative. Override via ``XTRI_REPORT_MODEL`` env var.
DEFAULT_MODEL = "claude-opus-4-6"
MAX_TOKENS = 8192       # single-area report
FULL_MAX_TOKENS = 16384  # full 4-area comparative report

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
    difficulty_highlights: list[dict[str, Any]]  # ~8-row sample for the prompt
    difficulty_table: list[dict[str, Any]]        # full table for charts
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

    # Full difficulty table for charts; sparse highlights for the prompt
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
        difficulty_table=diff_rows,
        top_ufs=top_ufs,
        top_municipios=top_municipios,
        top_escolas=top_escolas,
    )


def build_prompt(payload: ReportPayload) -> str:
    """Construct the Claude user prompt with all factual anchors."""
    s = payload.summary
    area_label = AREA_LABELS.get(payload.area, payload.area)
    iqr = round((s.get("p75") or 0) - (s.get("p25") or 0), 1)
    amplitude = round((s.get("max") or 0) - (s.get("min") or 0), 1)

    lines: list[str] = []
    lines.append(
        f"Você é um analista pedagógico sênior do INEP. Escreva um relatório analítico "
        f"detalhado em **português do Brasil** sobre os microdados oficiais do ENEM "
        f"{payload.year} — área de **{area_label}** ({payload.area})."
    )
    lines.append("")
    lines.append(
        "Baseie-se EXCLUSIVAMENTE nos números abaixo. Não invente estatísticas, não "
        "generalize para outras áreas, não cite estudos externos. Se um valor não "
        "estiver listado, diga explicitamente que o dado não foi fornecido. "
        "Seja rigoroso, detalhado e cite os números exatos ao longo do texto."
    )
    lines.append("")
    lines.append("## Estatísticas Descritivas (NU_NOTA oficial INEP)")
    lines.append("")
    lines.append(f"- N alunos válidos: **{s.get('n', 0):,}**".replace(",", "."))
    lines.append(f"- Média: **{s.get('mean')}**")
    lines.append(f"- Mediana: **{s.get('median')}**")
    lines.append(f"- Desvio-padrão: {s.get('std', 'não disponível')}")
    lines.append(f"- P10: {s.get('p10', '—')}  |  P25: {s.get('p25')}  |  P75: {s.get('p75')}  |  P90: {s.get('p90', '—')}")
    lines.append(f"- IQR (P75−P25): **{iqr}**")
    lines.append(f"- Amplitude (max−min): **{amplitude}**")
    lines.append(f"- Mínimo: {s.get('min')}  |  Máximo: {s.get('max')}")
    lines.append("")
    lines.append("## Tabela de Dificuldades (escala TRI INEP empírica)")
    lines.append("")
    lines.append(
        "Cada linha: número de acertos na prova → frequência de alunos → "
        "nota TRI mínima / mediana / máxima observada para esse nível."
    )
    lines.append("")
    lines.append("| Acertos | Freq | TRI min | TRI med | TRI max |")
    lines.append("|--------:|-----:|--------:|--------:|--------:|")
    for row in payload.difficulty_highlights:
        lines.append(
            f"| {row.get('acertos')} | {row.get('freq')} | "
            f"{row.get('tri_min')} | **{row.get('tri_med')}** | {row.get('tri_max')} |"
        )
    lines.append("")
    lines.append("## Top 10 Estados (UF) por mediana INEP")
    lines.append("")
    lines.append("| UF | N | Mediana | Média | P25 | P75 | IQR |")
    lines.append("|:--|--:|--:|--:|--:|--:|--:|")
    for r in payload.top_ufs:
        uf_iqr = round((r.get("p75") or 0) - (r.get("p25") or 0), 1)
        lines.append(
            f"| {r['label']} | {r['n']} | **{r['median']}** | "
            f"{r['mean']} | {r['p25']} | {r['p75']} | {uf_iqr} |"
        )
    lines.append("")
    lines.append("## Top 10 Municípios por mediana INEP")
    lines.append("")
    lines.append("| Município | UF | N | Mediana | Média | P25 | P75 |")
    lines.append("|:--|:--|--:|--:|--:|--:|--:|")
    for r in payload.top_municipios:
        lines.append(
            f"| {r['label']} | {r.get('uf', '—')} | {r['n']} | "
            f"**{r['median']}** | {r['mean']} | {r.get('p25', '—')} | {r.get('p75', '—')} |"
        )
    lines.append("")
    lines.append("## Top 10 Escolas por mediana INEP")
    lines.append("")
    lines.append("| Código | Município | UF | Rede | Localização | N | Mediana | Média |")
    lines.append("|:--|:--|:--|:--|:--|--:|--:|--:|")
    for r in payload.top_escolas:
        lines.append(
            f"| {r['label']} | {r.get('municipio', '—')} | {r.get('uf', '—')} | "
            f"{r.get('dependencia', '—')} | {r.get('localizacao', '—')} | "
            f"{r['n']} | **{r['median']}** | {r.get('mean', '—')} |"
        )
    lines.append("")
    lines.append("## Instruções para o Relatório")
    lines.append("")
    lines.append(
        "Produza um relatório extenso e detalhado em markdown, cobrindo obrigatoriamente:\n"
        "\n"
        "### 1. Resumo Executivo\n"
        "3–4 parágrafos. Sintetize o panorama geral da área, os achados mais relevantes "
        "e as disparidades mais marcantes encontradas nos dados. Cite valores numéricos.\n"
        "\n"
        "### 2. Análise da Distribuição de Notas\n"
        "- Interprete média vs. mediana: a diferença indica assimetria? Em qual direção?\n"
        "- Calcule e comente o coeficiente de variação aproximado (std/média × 100%) se "
        "o desvio-padrão estiver disponível.\n"
        "- Interprete o IQR e a amplitude: a distribuição é concentrada ou dispersa?\n"
        "- Quais faixas de nota concentram a maior parte dos alunos?\n"
        "\n"
        "### 3. Análise Detalhada da Tabela de Dificuldades\n"
        "- Descreva a progressão da TRI med à medida que acertos aumentam.\n"
        "- Identifique e explique plateaus (acertos adjacentes com TRI med similar) e "
        "quebras abruptas (saltos grandes de TRI entre acertos consecutivos).\n"
        "- Comente a amplitude do intervalo [TRI min, TRI max] em cada nível — "
        "intervalos largos indicam maior variabilidade de desempenho naquele nível.\n"
        "- Qual é o 'ponto de inflexão' onde a dificuldade cresce mais rapidamente?\n"
        "\n"
        "### 4. Análise Geográfica — Estados\n"
        "- Ranking dos top 10 UFs: cite as medianas exatas.\n"
        "- Há concentração regional (Sul/Sudeste vs. Norte/Nordeste/Centro-Oeste)?\n"
        "- Compare IQR entre os estados: quais têm distribuição mais homogênea? Mais dispersa?\n"
        "- Identifique o gap entre o melhor e o pior estado do ranking.\n"
        "\n"
        "### 5. Perfil dos Municípios e Escolas de Topo\n"
        "- Cite municípios específicos com medianas excepcionais.\n"
        "- Analise o perfil das top 10 escolas: rede (Federal/Estadual/Municipal/Privada), "
        "localização (Urbana/Rural), concentração geográfica por estado.\n"
        "- Que padrões emergem? Existe dominância de algum tipo de rede ou região?\n"
        "- Quais escolas se destacam com N alto (base robusta), não apenas com N pequeno?\n"
        "\n"
        "### 6. Implicações Pedagógicas\n"
        "3–4 insights concretos e acionáveis para gestores e coordenadores pedagógicos, "
        "baseados estritamente nos dados apresentados. Não moralize nem sugira "
        "políticas públicas genéricas. Foque em padrões observáveis.\n"
        "\n"
        "REGRAS: Use apenas os dados fornecidos. Cite números exatos. "
        "Idioma: português do Brasil formal. Tom: técnico, analítico e objetivo. "
        "Extensão mínima: 800 palavras."
    )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Full 4-area report
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FullReportPayload:
    """Holds payloads for all four ENEM areas."""

    year: int
    areas: dict  # area_code -> ReportPayload


def build_full_payload(df: pd.DataFrame, year: int) -> FullReportPayload:
    """Build :class:`ReportPayload` for every area and bundle them."""
    return FullReportPayload(
        year=year,
        areas={area: build_payload(df, year, area) for area in AREAS},
    )


def build_full_prompt(full: FullReportPayload) -> str:
    """Build a comparative prompt covering all four areas in one Claude call."""
    lines: list[str] = []
    lines.append(
        f"Você é um analista pedagógico sênior do INEP. Escreva um relatório analítico "
        f"completo e aprofundado em **português do Brasil** sobre os microdados oficiais "
        f"do ENEM {full.year}, cobrindo as **quatro áreas de conhecimento**: "
        "Linguagens (LC), Ciências Humanas (CH), Ciências da Natureza (CN) e "
        "Matemática (MT)."
    )
    lines.append("")
    lines.append(
        "Baseie-se EXCLUSIVAMENTE nos dados abaixo. Não invente estatísticas nem "
        "cite estudos externos. Cite números exatos ao longo do texto. "
        "Seja rigoroso, detalhado e comparativo entre as áreas."
    )
    lines.append("")

    # ── Comparative overview table ──────────────────────────────────────────
    lines.append("## Quadro Comparativo — 4 Áreas de Conhecimento")
    lines.append("")
    lines.append("| Área | N | Média | Mediana | P25 | P75 | IQR | Mín | Máx |")
    lines.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
    for code in AREAS:
        p = full.areas[code]
        s = p.summary
        iqr = round((s.get("p75") or 0) - (s.get("p25") or 0), 1)
        label = AREA_LABELS.get(code, code)
        lines.append(
            f"| **{code}** — {label} | {s.get('n', 0):,} | {s.get('mean')} | "
            f"**{s.get('median')}** | {s.get('p25')} | {s.get('p75')} | "
            f"{iqr} | {s.get('min')} | {s.get('max')} |".replace(",", ".")
        )
    lines.append("")

    # ── Per-area detail ─────────────────────────────────────────────────────
    for code in AREAS:
        p = full.areas[code]
        s = p.summary
        area_label = AREA_LABELS.get(code, code)
        lines.append(f"## Detalhamento — {code}: {area_label}")
        lines.append("")

        # Difficulty highlights
        lines.append("### Tabela de Dificuldades (TRI INEP empírica)")
        lines.append("")
        lines.append("| Acertos | Freq | TRI min | TRI med | TRI max |")
        lines.append("|--------:|-----:|--------:|--------:|--------:|")
        for row in p.difficulty_highlights:
            lines.append(
                f"| {row.get('acertos')} | {row.get('freq')} | "
                f"{row.get('tri_min')} | **{row.get('tri_med')}** | {row.get('tri_max')} |"
            )
        lines.append("")

        # Top UFs
        lines.append("### Top 10 Estados por mediana")
        lines.append("")
        lines.append("| UF | N | Mediana | Média | P25 | P75 |")
        lines.append("|:--|--:|--:|--:|--:|--:|")
        for r in p.top_ufs:
            lines.append(
                f"| {r['label']} | {r['n']} | **{r['median']}** | "
                f"{r['mean']} | {r['p25']} | {r['p75']} |"
            )
        lines.append("")

        # Top schools
        lines.append("### Top 10 Escolas por mediana")
        lines.append("")
        lines.append("| Código | Município | UF | Rede | N | Mediana |")
        lines.append("|:--|:--|:--|:--|--:|--:|")
        for r in p.top_escolas:
            lines.append(
                f"| {r['label']} | {r.get('municipio', '—')} | {r.get('uf', '—')} | "
                f"{r.get('dependencia', '—')} | {r['n']} | **{r['median']}** |"
            )
        lines.append("")

    # ── Instructions ────────────────────────────────────────────────────────
    lines.append("## Instruções para o Relatório")
    lines.append("")
    lines.append(
        "Produza um relatório extenso, estruturado e detalhado em markdown:\n"
        "\n"
        "### 1. Resumo Executivo Comparativo\n"
        "4–5 parágrafos. Compare o desempenho geral nas quatro áreas. Qual área tem "
        "maior mediana? Qual tem maior dispersão (IQR)? Onde se concentram as maiores "
        "desigualdades? Cite os valores exatos.\n"
        "\n"
        "### 2. Análise por Área (uma subseção para cada uma das 4 áreas)\n"
        "Para LC, CH, CN e MT individualmente:\n"
        "- Distribuição de notas: média vs. mediana, assimetria, IQR, amplitude.\n"
        "- Tabela de dificuldades: progressão TRI, plateaus, quebras, pontos de inflexão.\n"
        "- Top estados: padrão regional, gaps entre melhores e piores.\n"
        "- Top escolas: perfil de rede, localização, concentração geográfica.\n"
        "\n"
        "### 3. Análise Comparativa entre Áreas\n"
        "- Quais áreas são sistematicamente mais difíceis (menor mediana)?\n"
        "- A hierarquia de dificuldade é consistente com o esperado para o ENEM?\n"
        "- Compare a dispersão entre áreas: onde há mais homogeneidade? Mais desigualdade?\n"
        "- Há estados que aparecem no top 10 em múltiplas áreas? Quais?\n"
        "- Há escolas que aparecem no top 10 em mais de uma área? O que isso sugere?\n"
        "\n"
        "### 4. Padrões Geográficos Transversais\n"
        "- Identifique estados com desempenho consistentemente alto ou baixo em todas as áreas.\n"
        "- Analise o padrão regional: Sul/Sudeste vs. Norte/Nordeste/Centro-Oeste.\n"
        "- Há municípios que se destacam em múltiplas áreas?\n"
        "\n"
        "### 5. Perfil das Escolas de Excelência\n"
        "- Considerando as top 10 escolas de todas as áreas: qual rede domina (Federal/"
        "Estadual/Municipal/Privada)? Há escolas públicas federais entre as melhores?\n"
        "- Qual a distribuição geográfica dessas escolas?\n"
        "- Escolas com N alto e mediana elevada são mais significativas — destaque-as.\n"
        "\n"
        "### 6. Implicações Pedagógicas\n"
        "4–5 insights concretos e acionáveis baseados estritamente nos dados. "
        "Foque em padrões observáveis, não em recomendações políticas genéricas.\n"
        "\n"
        "REGRAS: Use apenas os dados fornecidos. Cite números exatos. "
        "Idioma: português do Brasil formal. Tom: técnico, analítico e objetivo. "
        "Extensão mínima: 1.500 palavras. Seja abrangente — este é o relatório principal "
        "do sistema XTRI."
    )

    return "\n".join(lines)


def generate_full_report(
    df: pd.DataFrame,
    year: int,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Generate a single Claude narrative covering all four ENEM areas.

    Returns the same dict shape as :func:`generate_report` but ``area`` is
    set to ``"ALL"`` and ``markdown`` contains the cross-area analysis.
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
            "The 'anthropic' package is not installed. Run `pip install anthropic`."
        ) from exc

    full_payload = build_full_payload(df, year)
    prompt = build_full_prompt(full_payload)
    model_name = model or os.environ.get("XTRI_REPORT_MODEL", DEFAULT_MODEL)

    logger.info(
        "Generating full 4-area report year=%d model=%s prompt_chars=%d",
        year,
        model_name,
        len(prompt),
    )

    client = Anthropic(api_key=resolved_key)
    msg = client.messages.create(
        model=model_name,
        max_tokens=FULL_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    markdown_parts: list[str] = []
    for block in msg.content:
        text = getattr(block, "text", None)
        if text:
            markdown_parts.append(text)
    markdown = "\n".join(markdown_parts).strip()

    usage = getattr(msg, "usage", None)
    return {
        "year": year,
        "area": "ALL",
        "model": model_name,
        "prompt_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        "markdown": markdown,
        "charts_data": {
            area_code: _charts_data(p)
            for area_code, p in full_payload.areas.items()
        },
    }


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
    return {
        "year": year,
        "area": area.upper(),
        "model": model_name,
        "prompt_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        "markdown": markdown,
        "charts_data": _charts_data(payload),
    }


def _charts_data(payload: ReportPayload) -> dict[str, Any]:
    """Serialize a ReportPayload into a JSON-safe dict for Plotly rendering."""
    return {
        "summary": payload.summary,
        "difficulty": payload.difficulty_table,
        "top_ufs": payload.top_ufs,
        "top_municipios": payload.top_municipios,
        "top_escolas": payload.top_escolas,
    }


def _load_api_key() -> Optional[str]:
    """Look up the Anthropic API key from env or a .env file.

    Search order:
    1. ``ANTHROPIC_API_KEY`` environment variable (already exported).
    2. ``~/Library/Application Support/XTRI Engine/.env`` — written by the
       macOS .app bundle installer / user setup.
    3. Any ``.env`` file found by walking up from the current working directory
       (python-dotenv ``find_dotenv``).  Covers dev mode where the key lives in
       the project root.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key

    try:
        from dotenv import dotenv_values, find_dotenv  # noqa: PLC0415
    except ImportError:
        return None

    # 2 — explicit macOS app-support path
    env_path = _xtri_env_path()
    if env_path.is_file():
        values = dotenv_values(env_path)
        key = values.get("ANTHROPIC_API_KEY") or None
        if key:
            return key

    # 3 — walk up from cwd (dev / server mode)
    found = find_dotenv(usecwd=True)
    if found:
        values = dotenv_values(found)
        return values.get("ANTHROPIC_API_KEY") or None

    return None


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
