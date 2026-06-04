# app/evaluation/fredao_recovery.py
"""Recuperação de parâmetro no Simulado Fredão (2017 CN/MT, ordem AZUL).

Os xlsx do Fredão já são matrizes binárias de acerto (0/1), com colunas em
ordem de posição global (CN: 91-135, MT: 136-180) = ordem dos params oficiais
2017. Logo o alinhamento é 1:1 por posição, totalmente offline — reusa
``run_recovery_for_group`` passando o gabarito oficial (casa todas as posições).

Entrega RECUPERAÇÃO de parâmetro (nosso a,b calibrado vs. oficial INEP, sob dois
linkings). NÃO entrega atribuição ponta-a-ponta: alunos do Fredão não têm
NU_NOTA oficial, então não há nota-verdade para decompor o erro.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.evaluation.__main__ import run_recovery_for_group
from app.evaluation.ground_truth import load_official_params, official_items
from app.evaluation.report import write_report

FREDAO_DIR = Path("/Volumes/KINGSTON/apps/MIRT/Simulado Fredão ")
FREDAO_FILES = {"CN": "Natureza - Simulado 2.xlsx", "MT": "Matemática - Simulado 2.xlsx"}
N_ITENS = 45


def load_fredao_matrix(xlsx_path: Path | str) -> np.ndarray:
    """Lê o xlsx do Fredão → matriz 0/1 (n_alunos × 45) em ordem de posição.

    Ignora a coluna ``ID``, ordena as colunas restantes por posição numérica,
    e trata qualquer valor fora de {0,1} (vazio/NaN) como 0.
    """
    df = pd.read_excel(xlsx_path)
    # Preserva a ORDEM do arquivo (= ordem de posição); colunas anuladas vêm
    # rotuladas (ex.: "Anulada") no slot da posição correspondente e são puladas
    # depois via o gabarito oficial ('X'/'.') no alinhamento.
    cols = [c for c in df.columns if str(c).upper() != "ID"]
    if len(cols) != N_ITENS:
        raise ValueError(
            f"esperado {N_ITENS} itens, achei {len(cols)} colunas de posição"
        )
    mat = df[cols].to_numpy()
    mat = np.where(np.isin(mat, [0, 1]), mat, 0).astype(int)
    return mat


# Sanidade de alinhamento: se as colunas de resposta estão na ordem dos params
# oficiais, a taxa de acerto por item correlaciona FORTEMENTE (negativa) com o b
# oficial (itens mais difíceis → menos acertos). Perto de zero = desalinhado, e a
# recuperação seria sem sentido — então é suprimida.
_ALIGN_CORR_MAX = -0.30


def alignment_sanity_corr(matrix: np.ndarray, official: list[dict]) -> float:
    """corr(taxa de acerto por item, b oficial). Fortemente negativa se alinhado."""
    acerto = matrix.mean(axis=0)
    b = np.array([it["b"] for it in official], dtype=float)
    if acerto.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(acerto, b)[0, 1])


def run_fredao_recovery(year: int = 2017, fredao_dir: Path = FREDAO_DIR) -> dict:
    """Recuperação de parâmetro por área (CN/MT) contra os params oficiais.

    Cada área retorna ``sanity_corr_acerto_b`` e ``aligned``; a recuperação só é
    computada (chave do ano) quando o alinhamento passa na sanidade — caso
    contrário fica ``None`` e um aviso é emitido (nada de números enganosos).
    """
    params = load_official_params()
    recovery: dict = {}
    for area, fname in FREDAO_FILES.items():
        path = Path(fredao_dir) / fname
        if not path.is_file():
            print(f"AVISO: Fredão {area} ausente em {path} — pulado.", file=sys.stderr)
            continue
        official = official_items(params, year, area)
        matrix = load_fredao_matrix(path)
        sanity = alignment_sanity_corr(matrix, official)
        aligned = sanity <= _ALIGN_CORR_MAX
        entry = {"sanity_corr_acerto_b": sanity, "aligned": aligned, str(year): None}
        if aligned:
            gab = "".join(it["gabarito"] for it in official)
            entry[str(year)] = run_recovery_for_group(matrix, gab, official)
            print(f"Fredão {area}: {len(matrix)} alunos — alinhado (corr={sanity:.2f}).", file=sys.stderr)
        else:
            print(
                f"AVISO: Fredão {area} — alinhamento NÃO confiável "
                f"(corr acerto×b={sanity:.2f}, esperado ≤ {_ALIGN_CORR_MAX}). "
                f"As colunas de resposta não estão na ordem dos params oficiais; "
                f"recuperação suprimida (seria sem sentido).",
                file=sys.stderr,
            )
        recovery[area] = entry
    return recovery


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recuperação de parâmetro — Simulado Fredão (2017 CN/MT)"
    )
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--out", default="fredao_recovery_report.json")
    args = parser.parse_args(argv)

    recovery = run_fredao_recovery(args.year)
    report = {"source": "Simulado Fredão", "year": args.year, "recovery": recovery}
    write_report(report, args.out)

    print(f"=== RECUPERAÇÃO DE PARÂMETRO — Simulado Fredão {args.year} ===")
    print("(nosso a,b calibrado vs. oficial INEP, por método de linking)")
    for area, entry in recovery.items():
        sanity = entry["sanity_corr_acerto_b"]
        if not entry["aligned"]:
            print(
                f"{area}: alinhamento NÃO confiável (corr acerto×b={sanity:.2f}) "
                f"— recuperação suprimida (colunas fora da ordem oficial)."
            )
            continue
        print(f"{area} (alinhado, corr acerto×b={sanity:.2f}):")
        for link, bp in entry[str(args.year)].items():
            ra, rb = bp["a"], bp["b"]
            print(
                f"  {link:<13} a: r={ra['r']:.2f} rmse={ra['rmse']:.2f} | "
                f"b: r={rb['r']:.2f} rmse={rb['rmse']:.2f} (n={rb['n']})"
            )
    print(f"\nRelatório salvo em: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
