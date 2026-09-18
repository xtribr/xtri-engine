"""Relatório psicométrico complementar para os microdados oficiais do ENEM 2025.

O pacote público de 2025 separa resultados, participantes e itens. Este módulo
trabalha somente com ``RESULTADOS_2025.csv`` e ``ITENS_PROVA_2025.csv`` porque
o INEP declara que ``NU_SEQUENCIAL`` não pode ser relacionado ao identificador
mascarado de participantes.

A análise é feita em streaming, por área e aplicação, sem misturar provas. Os
itens dos diferentes cadernos são alinhados pelo ``CO_ITEM`` oficial antes do
cálculo dos indicadores clássicos.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd


AREAS_2025 = ("LC", "CH", "CN", "MT")
APLICACOES_2025 = ("regular", "reaplicacao")
RESPONSE_CATEGORIES = ("A", "B", "C", "D", "E", "*", ".", "OUTRA")

# Códigos publicados pelo INEP nos arquivos de importação do pacote 2025.
# BAM é catalogado para contabilização/exclusão, mas não pode ser analisado no
# nível do item porque seus códigos não aparecem em ITENS_PROVA_2025.csv.
PROVA_GROUPS_2025: dict[str, dict[str, frozenset[str]]] = {
    "CN": {
        "regular": frozenset({"1483", "1484", "1485", "1486", "1487", "1488", "1490", "1491", "1498"}),
        "reaplicacao": frozenset({"1569", "1570", "1571", "1572"}),
        "bam": frozenset({"1619", "1620", "1621", "1622", "1623", "1634"}),
    },
    "CH": {
        "regular": frozenset({"1447", "1448", "1449", "1450", "1451", "1452", "1454", "1455", "1495"}),
        "reaplicacao": frozenset({"1539", "1541", "1542"}),
        "bam": frozenset({"1583", "1584", "1585", "1586", "1587", "1631"}),
    },
    "LC": {
        "regular": frozenset({"1459", "1460", "1461", "1462", "1463", "1464", "1466", "1467", "1496"}),
        "reaplicacao": frozenset({"1549", "1551", "1552"}),
        "bam": frozenset({"1595", "1596", "1597", "1598", "1599", "1632"}),
    },
    "MT": {
        "regular": frozenset({"1471", "1472", "1473", "1474", "1475", "1476", "1478", "1479", "1497"}),
        "reaplicacao": frozenset({"1559", "1560", "1561", "1562"}),
        "bam": frozenset({"1607", "1608", "1609", "1610", "1611", "1633"}),
    },
}


@dataclass(frozen=True)
class Enem2025Paths:
    """Arquivos necessários do pacote oficial de microdados 2025."""

    root: Path
    resultados: Path
    itens: Path
    participantes: Optional[Path]


@dataclass(frozen=True)
class FormLayout:
    """Ordem canônica dos 45 itens respondidos em um caderno."""

    co_prova: str
    tp_lingua: Optional[int]
    item_indices: np.ndarray
    keys: np.ndarray
    valid_mask: np.ndarray


@dataclass
class PsychometricReport:
    """Tabelas e metadados calculados para uma área/aplicação."""

    summary: dict[str, object]
    items: pd.DataFrame
    distractors: pd.DataFrame
    forms: pd.DataFrame
    quality: pd.DataFrame


def _candidate_roots(root: Optional[Path | str]) -> list[Path]:
    candidates: list[Path] = []
    if root is not None:
        candidates.append(Path(root).expanduser())
    env_root = os.getenv("ENEM_2025_ROOT") or os.getenv("MICRODATA_PATH")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            Path("/Volumes/Kingston/microdados últimos 5 anos/microdados_enem_2025"),
            Path("/Volumes/KINGSTON/microdados últimos 5 anos/microdados_enem_2025"),
            Path("/Volumes/Kingston/apps/MIRT/microdados-2025"),
        ]
    )
    return candidates


def resolve_enem_2025_paths(root: Optional[Path | str] = None) -> Enem2025Paths:
    """Localiza a estrutura separada do pacote oficial ENEM 2025.

    ``root`` pode apontar para a raiz extraída, para ``DADOS/`` ou diretamente
    para ``RESULTADOS_2025.csv``. Variáveis ``ENEM_2025_ROOT`` e
    ``MICRODATA_PATH`` são aceitas como alternativas configuráveis.
    """

    attempted: list[Path] = []
    for candidate in _candidate_roots(root):
        if candidate.is_file():
            data_dir = candidate.parent
            package_root = data_dir.parent if data_dir.name.upper() == "DADOS" else data_dir
        else:
            package_root = candidate
            data_dir = candidate / "DADOS" if (candidate / "DADOS").is_dir() else candidate

        resultados = data_dir / "RESULTADOS_2025.csv"
        itens = data_dir / "ITENS_PROVA_2025.csv"
        participantes = data_dir / "PARTICIPANTES_2025.csv"
        attempted.extend([resultados, itens])
        if resultados.is_file() and itens.is_file():
            return Enem2025Paths(
                root=package_root.resolve(),
                resultados=resultados.resolve(),
                itens=itens.resolve(),
                participantes=participantes.resolve() if participantes.is_file() else None,
            )

    tried = ", ".join(str(path) for path in attempted)
    raise FileNotFoundError(
        "Pacote ENEM 2025 não encontrado. Informe a raiz extraída em "
        f"ENEM_2025_ROOT. Caminhos verificados: {tried}"
    )


def _as_float(value: object) -> float:
    if value is None or pd.isna(value):
        return math.nan
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return math.nan


def _classify_code(area: str, code: str) -> str:
    for application, codes in PROVA_GROUPS_2025[area].items():
        if code in codes:
            return application
    return "desconhecida"


def _load_item_catalog(
    itens_path: Path | str,
    area: str,
    application: str,
) -> tuple[pd.DataFrame, dict[tuple[str, Optional[int]], FormLayout]]:
    area = area.upper()
    target_codes = PROVA_GROUPS_2025[area][application]
    raw = pd.read_csv(itens_path, sep=";", encoding="latin-1", dtype=str)
    raw = raw.loc[raw["SG_AREA"].eq(area) & raw["CO_PROVA"].isin(target_codes)].copy()
    if raw.empty:
        raise ValueError(f"Nenhum item encontrado para área={area} aplicação={application}")

    raw["CO_POSICAO"] = pd.to_numeric(raw["CO_POSICAO"], errors="raise").astype(int)
    raw["IN_ITEM_ABAN"] = raw["IN_ITEM_ABAN"].fillna("0")
    raw["_tp_lingua"] = pd.to_numeric(raw["TP_LINGUA"], errors="coerce")

    consistency_cols = [
        "TX_GABARITO",
        "CO_HABILIDADE",
        "IN_ITEM_ABAN",
        "NU_PARAM_A",
        "NU_PARAM_B",
        "NU_PARAM_C",
    ]
    for col in consistency_cols:
        nunique = raw.groupby("CO_ITEM", dropna=False)[col].nunique(dropna=False)
        inconsistent = nunique[nunique > 1]
        if not inconsistent.empty:
            raise ValueError(
                f"Metadado {col} inconsistente entre cadernos para itens: "
                f"{', '.join(inconsistent.index.astype(str)[:10])}"
            )

    meta = (
        raw.sort_values(["CO_ITEM", "CO_PROVA", "CO_POSICAO"])
        .drop_duplicates("CO_ITEM")
        .reset_index(drop=True)
    )
    meta["_item_index"] = np.arange(len(meta), dtype=int)
    index_by_item = dict(zip(meta["CO_ITEM"], meta["_item_index"]))

    layouts: dict[tuple[str, Optional[int]], FormLayout] = {}
    languages: Iterable[Optional[int]] = (0, 1) if area == "LC" else (None,)
    for code, form in raw.groupby("CO_PROVA", sort=True):
        for language in languages:
            if area == "LC":
                selected = form.loc[
                    (form["CO_POSICAO"] > 5) | form["_tp_lingua"].eq(language)
                ].copy()
            else:
                selected = form.copy()
            selected = selected.sort_values("CO_POSICAO")
            if len(selected) != 45 or selected["CO_POSICAO"].nunique() != 45:
                raise ValueError(
                    f"Caderno {code} área={area} língua={language} não possui 45 posições únicas"
                )
            indices = selected["CO_ITEM"].map(index_by_item).to_numpy(dtype=int)
            keys = selected["TX_GABARITO"].fillna("").to_numpy(dtype="S1")
            valid = (
                selected["IN_ITEM_ABAN"].ne("1")
                & selected["TX_GABARITO"].isin(list("ABCDE"))
            ).to_numpy(dtype=bool)
            layouts[(str(code), language)] = FormLayout(
                co_prova=str(code),
                tp_lingua=language,
                item_indices=indices,
                keys=keys,
                valid_mask=valid,
            )
    return meta, layouts


def _corr_from_sums(
    n: np.ndarray,
    sum_x: np.ndarray,
    sum_y: np.ndarray,
    sum_x2: np.ndarray,
    sum_y2: np.ndarray,
    sum_xy: np.ndarray,
) -> np.ndarray:
    out = np.full(len(n), np.nan, dtype=float)
    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt(
        np.maximum(n * sum_x2 - sum_x**2, 0.0)
        * np.maximum(n * sum_y2 - sum_y**2, 0.0)
    )
    mask = (n > 1) & (denominator > 0)
    out[mask] = numerator[mask] / denominator[mask]
    return out


def _difficulty_label(p_value: float) -> str:
    if math.isnan(p_value):
        return "N/A"
    if p_value >= 0.85:
        return "muito fácil"
    if p_value >= 0.65:
        return "fácil"
    if p_value >= 0.35:
        return "média"
    if p_value >= 0.15:
        return "difícil"
    return "muito difícil"


def _discrimination_label(value: float) -> str:
    if math.isnan(value):
        return "N/A"
    if value >= 0.40:
        return "muito boa"
    if value >= 0.30:
        return "boa"
    if value >= 0.20:
        return "aceitável"
    if value >= 0.0:
        return "baixa"
    return "negativa"


def _sample_variance(n: int, total: float, total_sq: float) -> float:
    if n <= 1:
        return math.nan
    return max((total_sq - total * total / n) / (n - 1), 0.0)


def analyze_enem_2025_area(
    *,
    results_path: Path | str,
    items_path: Path | str,
    area: str,
    application: str = "regular",
    chunksize: int = 100_000,
    max_rows: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> PsychometricReport:
    """Calcula indicadores psicométricos clássicos em streaming.

    O escopo é sempre uma combinação ``ano × aplicação × área``. Participantes
    ausentes/eliminados, respostas de tamanho inválido e cadernos BAM são
    contabilizados, mas não entram nas estimativas dos itens.
    """

    area = area.upper()
    application = application.lower()
    if area not in AREAS_2025:
        raise ValueError(f"Área inválida: {area}. Use {', '.join(AREAS_2025)}")
    if application not in APLICACOES_2025:
        raise ValueError(
            f"Aplicação inválida: {application}. Use {', '.join(APLICACOES_2025)}"
        )
    if chunksize <= 0:
        raise ValueError("chunksize deve ser positivo")

    item_meta, layouts = _load_item_catalog(items_path, area, application)
    n_items = len(item_meta)
    target_codes = PROVA_GROUPS_2025[area][application]
    presence_col = f"TP_PRESENCA_{area}"
    code_col = f"CO_PROVA_{area}"
    response_col = f"TX_RESPOSTAS_{area}"
    usecols = [presence_col, code_col, response_col]
    if area == "LC":
        usecols.append("TP_LINGUA")

    exposed = np.zeros(n_items, dtype=np.int64)
    valid_n = np.zeros(n_items, dtype=np.int64)
    sum_x = np.zeros(n_items, dtype=float)
    sum_y = np.zeros(n_items, dtype=float)
    sum_x2 = np.zeros(n_items, dtype=float)
    sum_y2 = np.zeros(n_items, dtype=float)
    sum_xy = np.zeros(n_items, dtype=float)
    category_counts = np.zeros((n_items, len(RESPONSE_CATEGORIES)), dtype=np.int64)
    category_rest_sum = np.zeros((n_items, len(RESPONSE_CATEGORIES)), dtype=float)

    form_stats: dict[tuple[str, Optional[int]], dict[str, object]] = {}
    presence_counts: Counter[str] = Counter()
    application_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    rows_read = 0
    rows_analyzed = 0

    reader = pd.read_csv(
        results_path,
        sep=";",
        encoding="latin-1",
        dtype=str,
        usecols=usecols,
        chunksize=chunksize,
        nrows=max_rows,
        low_memory=False,
    )
    for chunk in reader:
        rows_read += len(chunk)
        presence_counts.update(chunk[presence_col].fillna("<vazio>"))
        present = chunk.loc[chunk[presence_col].eq("1")].copy()
        for code, count in present[code_col].fillna("<vazio>").value_counts().items():
            application_counts[_classify_code(area, str(code))] += int(count)

        selected = present.loc[present[code_col].isin(target_codes)].copy()
        if selected.empty:
            if progress_callback is not None:
                progress_callback(rows_read, rows_analyzed)
            continue
        lengths = selected[response_col].fillna("").str.len()
        excluded_counts["resposta_tamanho_invalido"] += int(lengths.ne(45).sum())
        selected = selected.loc[lengths.eq(45)]

        group_columns = [code_col, "TP_LINGUA"] if area == "LC" else [code_col]
        for group_key, group in selected.groupby(group_columns, dropna=False, sort=False):
            if area == "LC":
                code, language_raw = group_key
                try:
                    language = int(str(language_raw))
                except (TypeError, ValueError):
                    excluded_counts["lingua_invalida"] += len(group)
                    continue
            else:
                code = group_key[0] if isinstance(group_key, tuple) else group_key
                language = None
            layout = layouts.get((str(code), language))
            if layout is None:
                excluded_counts["caderno_sem_itens"] += len(group)
                continue

            responses = group[response_col].to_numpy(dtype="S45")
            matrix = responses.view("S1").reshape(-1, 45)
            correct = matrix == layout.keys[None, :]
            correct[:, ~layout.valid_mask] = False
            k_valid = int(layout.valid_mask.sum())
            if k_valid < 2:
                raise ValueError(f"Caderno {code} possui menos de 2 itens válidos")
            totals = correct[:, layout.valid_mask].sum(axis=1).astype(float)
            rest_denominator = k_valid - 1
            n_group = len(group)
            rows_analyzed += n_group

            form_key = (str(code), language)
            stats = form_stats.setdefault(
                form_key,
                {
                    "n": 0,
                    "sum_total": 0.0,
                    "sum_total_sq": 0.0,
                    "sum_items": np.zeros(45, dtype=float),
                    "valid_mask": layout.valid_mask.copy(),
                },
            )
            stats["n"] = int(stats["n"]) + n_group
            stats["sum_total"] = float(stats["sum_total"]) + float(totals.sum())
            stats["sum_total_sq"] = float(stats["sum_total_sq"]) + float(np.square(totals).sum())
            stats["sum_items"] = np.asarray(stats["sum_items"]) + correct.sum(axis=0)

            for position in range(45):
                item_idx = int(layout.item_indices[position])
                exposed[item_idx] += n_group
                raw = matrix[:, position]
                category_idx = np.full(n_group, RESPONSE_CATEGORIES.index("OUTRA"), dtype=np.int8)
                for cat_idx, category in enumerate(RESPONSE_CATEGORIES[:-1]):
                    category_idx[raw == category.encode("ascii")] = cat_idx
                category_counts[item_idx] += np.bincount(
                    category_idx, minlength=len(RESPONSE_CATEGORIES)
                )

                if not layout.valid_mask[position]:
                    continue
                x = correct[:, position].astype(float)
                rest = (totals - x) / rest_denominator
                valid_n[item_idx] += n_group
                sum_x[item_idx] += x.sum()
                sum_x2[item_idx] += np.square(x).sum()
                sum_y[item_idx] += rest.sum()
                sum_y2[item_idx] += np.square(rest).sum()
                sum_xy[item_idx] += np.multiply(x, rest).sum()
                for cat_idx in range(len(RESPONSE_CATEGORIES)):
                    mask = category_idx == cat_idx
                    if mask.any():
                        category_rest_sum[item_idx, cat_idx] += rest[mask].sum()

        if progress_callback is not None:
            progress_callback(rows_read, rows_analyzed)

    correlations = _corr_from_sums(valid_n, sum_x, sum_y, sum_x2, sum_y2, sum_xy)
    p_values = np.divide(
        sum_x,
        valid_n,
        out=np.full(n_items, np.nan, dtype=float),
        where=valid_n > 0,
    )

    item_rows: list[dict[str, object]] = []
    distractor_rows: list[dict[str, object]] = []
    for idx, meta in item_meta.iterrows():
        p_value = float(p_values[idx])
        discrimination = float(correlations[idx])
        abandoned = str(meta["IN_ITEM_ABAN"]) == "1"
        alerts: list[str] = []
        if abandoned:
            alerts.append("item abandonado pelo INEP")
        elif not math.isnan(discrimination) and discrimination < 0:
            alerts.append("discriminação negativa")
        elif not math.isnan(discrimination) and discrimination < 0.20:
            alerts.append("discriminação baixa")
        if not math.isnan(p_value) and (p_value < 0.10 or p_value > 0.90):
            alerts.append("dificuldade extrema")
        if int(exposed[idx]) == 0:
            alerts.append("sem exposição nos resultados")

        item_rows.append(
            {
                "area": area,
                "aplicacao": application,
                "co_item": str(meta["CO_ITEM"]),
                "habilidade": meta.get("CO_HABILIDADE"),
                "gabarito": meta.get("TX_GABARITO"),
                "item_abandonado": abandoned,
                "n_expostos": int(exposed[idx]),
                "n_analisados": int(valid_n[idx]),
                "taxa_acerto": p_value,
                "dificuldade_ctt": 1.0 - p_value if not math.isnan(p_value) else math.nan,
                "classe_dificuldade": _difficulty_label(p_value),
                "correlacao_item_total_corrigida": discrimination,
                "classe_discriminacao": _discrimination_label(discrimination),
                "parametro_a_inep": _as_float(meta.get("NU_PARAM_A")),
                "parametro_b_inep": _as_float(meta.get("NU_PARAM_B")),
                "parametro_c_inep": _as_float(meta.get("NU_PARAM_C")),
                "alertas": "; ".join(alerts),
            }
        )

        for cat_idx, category in enumerate(RESPONSE_CATEGORIES):
            count = int(category_counts[idx, cat_idx])
            mean_rest = (
                float(category_rest_sum[idx, cat_idx] / count)
                if count > 0 and not abandoned
                else math.nan
            )
            distractor_rows.append(
                {
                    "co_item": str(meta["CO_ITEM"]),
                    "alternativa": category,
                    "tipo": "gabarito" if category == meta.get("TX_GABARITO") else "distrator",
                    "n": count,
                    "proporcao_expostos": count / exposed[idx] if exposed[idx] else math.nan,
                    "media_escore_restante": mean_rest,
                }
            )

    form_rows: list[dict[str, object]] = []
    for (code, language), stats in sorted(form_stats.items()):
        n = int(stats["n"])
        valid_mask = np.asarray(stats["valid_mask"], dtype=bool)
        k = int(valid_mask.sum())
        sum_items = np.asarray(stats["sum_items"], dtype=float)[valid_mask]
        total_var = _sample_variance(n, float(stats["sum_total"]), float(stats["sum_total_sq"]))
        item_vars = sum(
            _sample_variance(n, float(value), float(value)) for value in sum_items
        )
        alpha = (
            k / (k - 1) * (1.0 - item_vars / total_var)
            if k > 1 and not math.isnan(total_var) and total_var > 0
            else math.nan
        )
        form_rows.append(
            {
                "co_prova": code,
                "tp_lingua": language,
                "n_analisados": n,
                "itens_validos": k,
                "media_acertos": float(stats["sum_total"]) / n if n else math.nan,
                "desvio_acertos": math.sqrt(total_var) if not math.isnan(total_var) else math.nan,
                "alpha_cronbach": alpha,
            }
        )

    items_df = pd.DataFrame(item_rows).sort_values(
        ["item_abandonado", "correlacao_item_total_corrigida", "co_item"],
        na_position="last",
    )
    distractors_df = pd.DataFrame(distractor_rows)
    forms_df = pd.DataFrame(form_rows)
    analyzed_items = items_df.loc[~items_df["item_abandonado"] & items_df["n_analisados"].gt(0)]
    weighted_alpha = (
        np.average(
            forms_df.loc[forms_df["alpha_cronbach"].notna(), "alpha_cronbach"],
            weights=forms_df.loc[forms_df["alpha_cronbach"].notna(), "n_analisados"],
        )
        if not forms_df.empty and forms_df["alpha_cronbach"].notna().any()
        else math.nan
    )
    summary: dict[str, object] = {
        "ano": 2025,
        "area": area,
        "aplicacao": application,
        "linhas_lidas": rows_read,
        "participantes_presentes_area": int(presence_counts.get("1", 0)),
        "participantes_analisados": rows_analyzed,
        "participantes_por_aplicacao": dict(sorted(application_counts.items())),
        "exclusoes": dict(sorted(excluded_counts.items())),
        "cadernos_analisados": len(form_stats),
        "itens_canonicos": len(items_df),
        "itens_analisados": len(analyzed_items),
        "itens_abandonados": int(items_df["item_abandonado"].sum()),
        "taxa_acerto_mediana_itens": float(analyzed_items["taxa_acerto"].median()),
        "discriminacao_mediana": float(analyzed_items["correlacao_item_total_corrigida"].median()),
        "itens_discriminacao_baixa": int(
            analyzed_items["correlacao_item_total_corrigida"].lt(0.20).sum()
        ),
        "itens_discriminacao_negativa": int(
            analyzed_items["correlacao_item_total_corrigida"].lt(0).sum()
        ),
        "alpha_cronbach_ponderado_cadernos": float(weighted_alpha),
        "nota_metodologica": (
            "Indicadores CTT descritivos; não substituem a nota oficial TRI. "
            "Participantes e Resultados 2025 não foram relacionados."
        ),
    }
    quality_rows = [
        {"verificacao": "linhas_lidas", "valor": rows_read, "status": "OK"},
        {"verificacao": "presentes_na_area", "valor": int(presence_counts.get("1", 0)), "status": "OK"},
        {"verificacao": "analisados_na_aplicacao", "valor": rows_analyzed, "status": "OK" if rows_analyzed else "ERRO"},
        {"verificacao": "respostas_tamanho_invalido", "valor": int(excluded_counts.get("resposta_tamanho_invalido", 0)), "status": "OK" if not excluded_counts.get("resposta_tamanho_invalido") else "ATENÇÃO"},
        {"verificacao": "cadernos_sem_itens", "valor": int(excluded_counts.get("caderno_sem_itens", 0)), "status": "OK" if not excluded_counts.get("caderno_sem_itens") else "ATENÇÃO"},
        {"verificacao": "vinculo_participantes_resultados", "valor": "não permitido pelo INEP", "status": "BLOQUEADO"},
    ]
    return PsychometricReport(
        summary=summary,
        items=items_df.reset_index(drop=True),
        distractors=distractors_df.reset_index(drop=True),
        forms=forms_df.reset_index(drop=True),
        quality=pd.DataFrame(quality_rows),
    )


def export_psychometric_report(
    report: PsychometricReport,
    output_xlsx: Path | str,
    *,
    output_json: Optional[Path | str] = None,
) -> tuple[Path, Optional[Path]]:
    """Exporta o relatório validado para XLSX e, opcionalmente, JSON."""

    output_xlsx = Path(output_xlsx)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(
        [{"indicador": key, "valor": value} for key, value in report.summary.items()]
    )
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Resumo", index=False)
        report.items.to_excel(writer, sheet_name="Itens", index=False)
        report.distractors.to_excel(writer, sheet_name="Distratores", index=False)
        report.forms.to_excel(writer, sheet_name="Cadernos", index=False)
        report.quality.to_excel(writer, sheet_name="Qualidade", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                width = min(max(len(str(cell.value or "")) for cell in column) + 2, 48)
                sheet.column_dimensions[column[0].column_letter].width = width

    json_path: Optional[Path] = None
    if output_json is not None:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report.summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    return output_xlsx, json_path
