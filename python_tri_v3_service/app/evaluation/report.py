# app/evaluation/report.py
"""Monta e renderiza o relatório de acurácia (JSON + tabela texto)."""
from __future__ import annotations

import json
from pathlib import Path


def build_report(
    *, year: int, end_to_end: dict, attribution: dict, recovery: dict
) -> dict:
    return {
        "year": year,
        "end_to_end": end_to_end,
        "attribution": attribution,
        "recovery": recovery,
    }


def _fmt(x: object) -> str:
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def render_text_table(report: dict) -> str:
    lines: list[str] = []
    lines.append(f"=== ARNÊS DE ACURÁCIA TRI — ENEM {report['year']} ===")
    lines.append("")
    lines.append("[Nota ponta-a-ponta vs NU_NOTA oficial — k-fold hold-out]")
    lines.append(f"{'Área':<5} {'n':>7} {'MAE':>7} {'IC(MAE)':>16} {'RMSE':>7} {'r':>6}")
    for area in ("LC", "CH", "CN", "MT"):
        e = report["end_to_end"].get(area)
        if not e:
            continue
        lo, hi = e.get("mae_ci", (float("nan"), float("nan")))
        lines.append(
            f"{area:<5} {e['n']:>7} {_fmt(e['mae']):>7} "
            f"[{_fmt(lo)},{_fmt(hi)}]".rjust(16)
            + f" {_fmt(e['rmse']):>7} {_fmt(e['r']):>6}"
        )
    lines.append("")
    lines.append("[Atribuição de erro (CN/MT, AZUL) — quanto vem de calibração vs ancoragem]")
    lines.append(f"{'Área':<5} {'MAE_total':>10} {'MAE_anc':>10} {'calib':>8} {'IC(calib)':>16}")
    for area in ("CN", "MT"):
        a = report["attribution"].get(area)
        if not a:
            continue
        lo, hi = a["calib_contrib_ci"]
        lines.append(
            f"{area:<5} {_fmt(a['mae_full']):>10} {_fmt(a['mae_anchor']):>10} "
            f"{_fmt(a['calib_contrib_mean']):>8} "
            + f"[{_fmt(lo)},{_fmt(hi)}]".rjust(16)
        )
    lines.append("")
    lines.append("[Recuperação de parâmetro (CN/MT) — nosso a,b vs oficial INEP, por linking]")
    for area in ("CN", "MT"):
        rec = report["recovery"].get(area, {})
        for ano, by_link in rec.items():
            for link_name, by_param in by_link.items():
                ra = by_param["a"]
                rb = by_param["b"]
                lines.append(
                    f"{area} {ano} {link_name:<13} "
                    f"a: r={_fmt(ra['r'])} rmse={_fmt(ra['rmse'])} | "
                    f"b: r={_fmt(rb['r'])} rmse={_fmt(rb['rmse'])}"
                )
    return "\n".join(lines)


def write_report(report: dict, out_path: Path | str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
