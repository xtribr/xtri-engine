"""
TRI 3PL de um simulado ENEM binarizado (JSON), pela linha de comando.

Usa exatamente o mesmo código do endpoint `POST /api/v3/processar-json`
(`app/simulado_binarizado.py`): 3PL por área, c fixo em 0.20 e estimado por
item com ≥ 500 alunos na área, theta EAP e ancoragem condicional aos acertos
na escala do ENEM do ano de referência.

Saídas em `pasta_saida`:
- `TRI_3PL.json`  — mesma resposta da API
- `TRI_3PL.xlsx`  — abas Notas, Por turma, Resumo e Itens
- `TRI_3PL.csv`   — aba Notas (entrada do `primi.py`)

Uso:
    cd python_tri_v3_service
    .venv313/bin/python scripts_local/tri_binarizado/tri_3pl.py binarizado.json pasta_saida [--ano 2023]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SERVICO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SERVICO))

from app.simulado_binarizado import processar_simulado_binarizado  # noqa: E402
from app.tabela_referencia import TabelaReferenciaTRI  # noqa: E402


def tabela_notas(resultado: dict) -> pd.DataFrame:
    areas = list(resultado["areas"])
    linhas = []
    for a in resultado["alunos"]:
        linha = {"matricula": a["matricula"], "nome": a["nome"], "turma": a["turma"]}
        for area in areas:
            r = a["areas"].get(area, {})
            linha[f"Acertos {area}"] = r.get("acertos")
            linha[f"TRI {area}"] = r.get("nota_tri")
        linha["Média TRI"] = a["media_tri"]
        for campo, rotulo in (("nivel", "Nível"), ("theta", "Theta")):
            for area in areas:
                linha[f"{rotulo} {area}"] = a["areas"].get(area, {}).get(campo)
        linhas.append(linha)
    df = pd.DataFrame(linhas)
    for area in areas:
        df[f"Acertos {area}"] = df[f"Acertos {area}"].astype("Int64")
    return df.sort_values(["turma", "nome"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("entrada", type=Path)
    parser.add_argument("saida", type=Path)
    parser.add_argument("--ano", type=int, default=2023, help="ano da escala de referência (padrão: 2023)")
    args = parser.parse_args()

    tabelas = TabelaReferenciaTRI.carregar_multiano()
    if args.ano not in tabelas:
        sys.exit(f"Ano {args.ano} indisponível. Opções: {sorted(tabelas)}")
    resultado = processar_simulado_binarizado(json.loads(args.entrada.read_text()), tabelas[args.ano], args.ano)

    args.saida.mkdir(parents=True, exist_ok=True)
    (args.saida / "TRI_3PL.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2))

    notas = tabela_notas(resultado)
    cols_tri = [f"TRI {a}" for a in resultado["areas"]] + ["Média TRI"]
    turmas = notas.groupby("turma")[cols_tri].mean().round(1)
    turmas.insert(0, "alunos", notas.groupby("turma").size())
    resumo = pd.DataFrame([{"area": ar, "n_alunos": b["n_alunos"], "modelo": b["modelo"], **b["resumo"]}
                           for ar, b in resultado["areas"].items()])
    itens = pd.DataFrame([{"area": ar, **it} for ar, b in resultado["areas"].items() for it in b["itens"]])

    with pd.ExcelWriter(args.saida / "TRI_3PL.xlsx", engine="openpyxl") as w:
        notas.to_excel(w, sheet_name="Notas", index=False)
        turmas.to_excel(w, sheet_name="Por turma")
        resumo.to_excel(w, sheet_name="Resumo", index=False)
        itens.to_excel(w, sheet_name="Itens", index=False)
        w.sheets["Notas"].freeze_panes = "D2"
    notas.to_csv(args.saida / "TRI_3PL.csv", index=False)

    print(resumo.to_string(index=False))
    for aviso in resultado["avisos"]:
        print("aviso:", aviso)
    print(f"\nsalvo em {args.saida}")


if __name__ == "__main__":
    main()
