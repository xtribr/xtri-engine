"""
Year-aware microdata column schema.

INEP has changed column names across ENEM years. Known differences:
- 2022/2023: primary ID is ``NU_INSCRICAO``
- 2024+:     primary ID is ``NU_SEQUENCIAL``

This module detects the schema dynamically from the CSV header so downstream
code doesn't have to hardcode year→column maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Areas used throughout the engine (uppercase two-letter codes).
AREAS: tuple[str, ...] = ("LC", "CH", "CN", "MT")


@dataclass(frozen=True)
class MicrodataSchema:
    """Normalized accessor for microdata columns, detected from the CSV header.

    All per-area columns are stored as dicts keyed by area code. Attributes that
    may be missing from older microdata dumps are Optional and callers must
    check before using.
    """

    id_col: str  # "NU_SEQUENCIAL" (2024+) or "NU_INSCRICAO" (<2024)
    tp_presenca: dict[str, str] = field(default_factory=dict)
    tx_respostas: dict[str, str] = field(default_factory=dict)
    co_prova: dict[str, str] = field(default_factory=dict)
    tx_gabarito: dict[str, str] = field(default_factory=dict)
    nu_nota: dict[str, str] = field(default_factory=dict)
    tp_status_redacao: str = "TP_STATUS_REDACAO"
    nu_nota_redacao: str = "NU_NOTA_REDACAO"
    tp_lingua: str = "TP_LINGUA"
    # Drill-down / demographic columns (optional — some years don't emit them).
    # NB: INEP removed IN_TREINEIRO and NO_ENTIDADE (school name) from ENEM
    # 2024+ microdata. School names are resolved in Phase 7 via the Supabase
    # `inep_escolas` table (27K schools, keyed on codigo_inep = CO_ESCOLA).
    co_uf_esc: Optional[str] = None
    sg_uf_esc: Optional[str] = None
    co_municipio_esc: Optional[str] = None
    no_municipio_esc: Optional[str] = None
    co_escola: Optional[str] = None
    tp_dependencia_adm_esc: Optional[str] = None
    tp_localizacao_esc: Optional[str] = None

    def all_needed_columns(self) -> list[str]:
        """Return the superset of columns this schema wants from the CSV.

        Used as the ``usecols=`` argument to ``pd.read_csv`` so we only read
        what we actually need (significant memory savings on a 1.5 GB CSV).
        """
        cols: list[str] = [self.id_col, self.tp_status_redacao, self.nu_nota_redacao, self.tp_lingua]
        for area in AREAS:
            cols.append(self.tp_presenca[area])
            cols.append(self.tx_respostas[area])
            cols.append(self.co_prova[area])
            cols.append(self.nu_nota[area])
            if area in self.tx_gabarito:
                cols.append(self.tx_gabarito[area])
        # Optional columns — include only if detected
        for optional in (
            self.co_uf_esc,
            self.sg_uf_esc,
            self.co_municipio_esc,
            self.no_municipio_esc,
            self.co_escola,
            self.tp_dependencia_adm_esc,
            self.tp_localizacao_esc,
        ):
            if optional is not None:
                cols.append(optional)
        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for col in cols:
            if col not in seen:
                seen.add(col)
                unique.append(col)
        return unique


def _detect_id_col(headers: list[str]) -> str:
    """Match the existing scripts' logic in ``calcular_transformacao.detect_id_column``."""
    if "NU_SEQUENCIAL" in headers:
        return "NU_SEQUENCIAL"
    if "NU_INSCRICAO" in headers:
        return "NU_INSCRICAO"
    # Fall back to the first column — better than crashing, at least gives caller
    # a working handle even if the header is unusual.
    return headers[0]


def detect_schema(csv_path: Path | str, encoding: str = "latin-1") -> MicrodataSchema:
    """Read the CSV header row and build a :class:`MicrodataSchema`.

    Reads only the first line of the file (minimal I/O) and constructs the
    schema based on which columns are present. Columns that are always
    expected (e.g. ``TP_PRESENCA_LC``) are assumed present and will raise a
    ``KeyError`` via ``MicrodataSchema.all_needed_columns`` if missing.

    Args:
        csv_path: Path to the microdata CSV. Must be readable.
        encoding: File encoding. INEP microdata is always latin-1.

    Returns:
        A :class:`MicrodataSchema` populated from the detected header columns.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Microdata CSV not found: {csv_path}")

    with open(csv_path, "r", encoding=encoding, errors="replace") as f:
        header_line = f.readline().rstrip("\n").rstrip("\r")
    headers = header_line.split(";")
    header_set = set(headers)

    def _maybe(col: str) -> Optional[str]:
        return col if col in header_set else None

    tp_presenca = {area: f"TP_PRESENCA_{area}" for area in AREAS}
    tx_respostas = {area: f"TX_RESPOSTAS_{area}" for area in AREAS}
    co_prova = {area: f"CO_PROVA_{area}" for area in AREAS}
    nu_nota = {area: f"NU_NOTA_{area}" for area in AREAS}
    tx_gabarito = {area: f"TX_GABARITO_{area}" for area in AREAS if f"TX_GABARITO_{area}" in header_set}

    return MicrodataSchema(
        id_col=_detect_id_col(headers),
        tp_presenca=tp_presenca,
        tx_respostas=tx_respostas,
        co_prova=co_prova,
        tx_gabarito=tx_gabarito,
        nu_nota=nu_nota,
        tp_status_redacao="TP_STATUS_REDACAO",
        nu_nota_redacao="NU_NOTA_REDACAO",
        tp_lingua="TP_LINGUA",
        co_uf_esc=_maybe("CO_UF_ESC"),
        sg_uf_esc=_maybe("SG_UF_ESC"),
        co_municipio_esc=_maybe("CO_MUNICIPIO_ESC"),
        no_municipio_esc=_maybe("NO_MUNICIPIO_ESC"),
        co_escola=_maybe("CO_ESCOLA"),
        tp_dependencia_adm_esc=_maybe("TP_DEPENDENCIA_ADM_ESC"),
        tp_localizacao_esc=_maybe("TP_LOCALIZACAO_ESC"),
    )
