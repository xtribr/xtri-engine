"""
One-time script to convert tri_acertos_2009_2022.xlsx to multi-year JSON.

Usage:
    python -m app.scripts.converter_excel_referencia /path/to/tri_acertos_2009_2022.xlsx

Output:
    app/referencias/tri_tabela_referencia_multiano.json
"""

import json
import os
import sys

import openpyxl


def converter(xlsx_path: str, output_path: str) -> None:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)

    # Use the summary sheet (first sheet)
    ws = wb[wb.sheetnames[0]]

    # Read header row
    header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    col_map = {name: idx for idx, name in enumerate(header)}

    required = {"area", "acertos", "min", "max", "mediana", "ano"}
    if not required.issubset(col_map.keys()):
        missing = required - set(col_map.keys())
        raise ValueError(f"Missing columns: {missing}. Found: {list(col_map.keys())}")

    # Parse rows
    data: dict = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        area = row[col_map["area"]]
        acertos = row[col_map["acertos"]]
        tri_min = row[col_map["min"]]
        tri_max = row[col_map["max"]]
        tri_med = row[col_map["mediana"]]
        ano = row[col_map["ano"]]

        if area is None or ano is None:
            continue

        ano_str = str(int(ano))
        acertos_str = str(int(acertos))

        if ano_str not in data:
            data[ano_str] = {}
        if area not in data[ano_str]:
            data[ano_str][area] = {}

        data[ano_str][area][acertos_str] = {
            "tri_min": round(float(tri_min), 1),
            "tri_med": round(float(tri_med), 1),
            "tri_max": round(float(tri_max), 1),
        }

    # Validate each year (warn on non-monotonic medians — real ENEM data has
    # small-sample quirks at the extremes, e.g. LC 2016 acertos 42-43)
    warnings = []
    for ano_str, areas in sorted(data.items()):
        for area, acertos_dict in areas.items():
            keys_sorted = sorted(int(k) for k in acertos_dict.keys())
            meds = [acertos_dict[str(k)]["tri_med"] for k in keys_sorted]
            if meds != sorted(meds):
                warnings.append(f"  WARN: Year {ano_str}, area {area}: mediana not strictly monotonic")
            for k in keys_sorted:
                entry = acertos_dict[str(k)]
                if not (entry["tri_min"] <= entry["tri_max"]):
                    warnings.append(
                        f"  WARN: Year {ano_str}, area {area}, acertos {k}: "
                        f"min={entry['tri_min']} > max={entry['tri_max']}"
                    )
        print(f"  Year {ano_str}: {list(areas.keys())} — OK")
    for w in warnings:
        print(w)

    # Sort years
    sorted_data = {k: data[k] for k in sorted(data.keys())}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated {output_path}")
    print(f"Years: {list(sorted_data.keys())}")
    total_entries = sum(
        len(acertos_dict)
        for areas in sorted_data.values()
        for acertos_dict in areas.values()
    )
    print(f"Total entries: {total_entries}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.scripts.converter_excel_referencia <xlsx_path>")
        sys.exit(1)

    xlsx_path = sys.argv[1]
    output_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "referencias",
        "tri_tabela_referencia_multiano.json",
    )
    converter(xlsx_path, output_path)
