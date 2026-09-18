"""CLI para gerar o relatório psicométrico complementar do ENEM 2025."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from app.microdata.psychometrics_2025 import (
    analyze_enem_2025_area,
    export_psychometric_report,
    resolve_enem_2025_paths,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", choices=("LC", "CH", "CN", "MT"), default="MT")
    parser.add_argument(
        "--aplicacao", choices=("regular", "reaplicacao"), default="regular"
    )
    parser.add_argument("--root", type=Path, default=None, help="Raiz extraída do ENEM 2025")
    parser.add_argument("--output", type=Path, required=True, help="Arquivo XLSX de saída")
    parser.add_argument("--json", type=Path, default=None, help="Resumo JSON opcional")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-rows", type=int, default=None, help="Limite para validação rápida")
    args = parser.parse_args()

    paths = resolve_enem_2025_paths(args.root)
    started = time.monotonic()
    last_million = -1

    def progress(rows_read: int, rows_analyzed: int) -> None:
        nonlocal last_million
        million = math.floor(rows_read / 1_000_000)
        if million != last_million:
            last_million = million
            print(
                f"Lidas {rows_read:,} linhas; analisadas {rows_analyzed:,}",
                flush=True,
            )

    report = analyze_enem_2025_area(
        results_path=paths.resultados,
        items_path=paths.itens,
        area=args.area,
        application=args.aplicacao,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
        progress_callback=progress,
    )
    xlsx, json_path = export_psychometric_report(
        report, args.output, output_json=args.json
    )
    print(f"Relatório: {xlsx}")
    if json_path is not None:
        print(f"Resumo JSON: {json_path}")
    print(f"Tempo total: {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
