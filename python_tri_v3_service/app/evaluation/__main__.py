# app/evaluation/__main__.py
"""CLI do arnês de acurácia: python -m app.evaluation --year 2024

Orquestra: carrega microdata real → k-fold ponta-a-ponta (4 áreas) →
recuperação de parâmetro + atribuição (CN/MT/AZUL) → relatório JSON + tabela.

Nota: atribuição/recuperação exigem alinhar as respostas à ordem oficial
(posição global). O TX_GABARITO do microdata está na ordem embaralhada do
caderno de cada cor; sem o mapa cor+posição→item (Supabase enem_itens_prova),
nenhum caderno casa com a ordem oficial e essas seções são puladas com aviso.
O k-fold ponta-a-ponta NÃO precisa desse alinhamento (calibra na ordem do
próprio caderno) e sempre roda.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from app.evaluation.attribution import decompose_error, score_with_params
from app.evaluation.ground_truth import (
    OfficialParamsNotFoundError,
    align_official_to_gabarito,
    load_official_params,
    official_items,
)
from app.evaluation.kfold import (
    build_group_matrix,
    kfold_oof_predictions,
    run_kfold_area,
)
from app.evaluation.linking import (
    apply_linking,
    mean_sigma_linking,
    stocking_lord_linking,
)
from app.evaluation.metrics import bootstrap_ci, mae, param_recovery_stats, pearson_r, rmse
from app.evaluation.report import build_report, render_text_table, write_report
from app.microdata.calibration import calibrar_itens_fast
from app.microdata.loader import default_csv_path, load_microdata, load_primeira_aplicacao_provas
from app.microdata.schema import detect_schema
from app.tabela_referencia import TabelaReferenciaTRI

ATTR_AREAS = ("CN", "MT")
ALL_AREAS = ("LC", "CH", "CN", "MT")


def _recovery_stats_pair(ours_linked: list[dict], official: list[dict]) -> dict:
    a_ours = np.array([p["a"] for p in ours_linked])
    b_ours = np.array([p["b"] for p in ours_linked])
    a_off = np.array([p["a"] for p in official])
    b_off = np.array([p["b"] for p in official])
    return {
        "a": param_recovery_stats(a_ours, a_off),
        "b": param_recovery_stats(b_ours, b_off),
    }


def run_recovery_for_group(
    resp_matrix: np.ndarray, group_gabarito: str, official: list[dict]
) -> dict:
    """Calibra o grupo, valida/aligna ao oficial, linka e mede recuperação."""
    valid_cols, official_valid = align_official_to_gabarito(official, group_gabarito)
    ours_all = calibrar_itens_fast(resp_matrix, c_strategy="fixed")
    ours = [ours_all[j] for j in valid_cols]

    b_ours = np.array([p["b"] for p in ours])
    b_off = np.array([p["b"] for p in official_valid])
    link_ms = mean_sigma_linking(b_ours, b_off)
    link_sl = stocking_lord_linking(ours, official_valid)

    return {
        "mean_sigma": _recovery_stats_pair(apply_linking(ours, link_ms), official_valid),
        "stocking_lord": _recovery_stats_pair(apply_linking(ours, link_sl), official_valid),
    }


def _find_azul_group(df, schema, area, official):
    """Encontra o caderno cujo gabarito casa com a ordem oficial (posição global).

    Os params oficiais estão em ordem de posição global do banco de itens; o
    TX_GABARITO do microdata está na ordem embaralhada do caderno de cada cor.
    O caderno cujo gabarito casa quase totalmente com o oficial é o que está na
    ordem oficial. Offline (sem o mapa cor+posição→item do Supabase), nenhum
    caderno do microdata está nessa ordem → retorna None.

    Retorna (resp, nota_ref, valid_cols, official_valid, gab) ou None.
    """
    off_gab = "".join(it["gabarito"] for it in official)
    co_col = schema.co_prova[area]
    co_str = df[co_col].astype(str)
    best = None  # (n_match, co_prova, gab)
    for cp, n in co_str.value_counts().items():
        if n < 500:
            continue
        gab = df.loc[co_str == cp, schema.tx_gabarito[area]].astype(str).iloc[0]
        if len(gab) != len(off_gab):
            continue
        n_match = sum(1 for a, b in zip(off_gab, gab) if a == b)
        if best is None or n_match > best[0]:
            best = (n_match, cp, gab)
    # Exige casamento quase total (tolera poucos itens anulados); senão não há
    # caderno na ordem oficial neste dado.
    if best is None or best[0] < len(off_gab) - 2:
        return None
    cp, gab = best[1], best[2]
    sub = df[co_str == cp]
    resp, nota_ref = build_group_matrix(sub, schema, area)
    valid_cols, official_valid = align_official_to_gabarito(official, gab)
    return resp, nota_ref, valid_cols, official_valid, gab


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Arnês de acurácia TRI")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--c-strategy", default="fixed", choices=["fixed", "bayes"])
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--out", default="accuracy_report.json")
    args = parser.parse_args(argv)

    csv_path = args.csv_path or default_csv_path(args.year)
    from pathlib import Path
    if not Path(csv_path).is_file():
        print(
            f"ERRO: microdata não encontrado em {csv_path}. "
            f"Baixe MICRODADOS_ENEM_{args.year}.csv do INEP.",
            file=sys.stderr,
        )
        return 2

    print(f"Carregando microdata {args.year} (amostra {args.sample_size})...", file=sys.stderr)
    whitelist = load_primeira_aplicacao_provas(args.year)
    df = load_microdata(
        args.year, csv_path=csv_path, sample_size=args.sample_size,
        provas_primeira=whitelist,
    )
    schema = detect_schema(csv_path)

    tabelas = TabelaReferenciaTRI.carregar_multiano()
    tabela = tabelas.get(args.year) or TabelaReferenciaTRI()

    end_to_end: dict = {}
    for area in ALL_AREAS:
        print(f"k-fold {area}...", file=sys.stderr)
        r = run_kfold_area(df, schema, area, tabela, k=args.kfolds, c_strategy=args.c_strategy)
        if r["n"] == 0:
            continue
        lo, hi = bootstrap_ci(np.abs(r["pred"] - r["ref"]))
        end_to_end[area] = {
            "area": area, "n": r["n"], "mae": mae(r["pred"], r["ref"]),
            "rmse": rmse(r["pred"], r["ref"]), "r": pearson_r(r["pred"], r["ref"]),
            "mae_ci": (lo, hi),
        }

    attribution: dict = {}
    recovery: dict = {}
    try:
        params_oficiais = load_official_params()
    except OfficialParamsNotFoundError as exc:
        print(f"AVISO: {exc} Atribuição/recuperação puladas.", file=sys.stderr)
    else:
        for area in ATTR_AREAS:
            official = official_items(params_oficiais, args.year, area)
            found = _find_azul_group(df, schema, area, official)
            if found is None:
                print(
                    f"AVISO: atribuição/recuperação {area} puladas — nenhum caderno do "
                    f"microdata está na ordem oficial (posição global). É necessário o mapa "
                    f"cor+posição→item (Supabase enem_itens_prova), indisponível offline.",
                    file=sys.stderr,
                )
                continue
            resp, nota_ref, valid_cols, official_valid, gab = found
            if len(resp) < 500:
                print(f"AVISO: grupo AZUL {area} pequeno ({len(resp)}) — pulado.", file=sys.stderr)
                continue

            our_oof = kfold_oof_predictions(resp, area, tabela, k=args.kfolds, c_strategy=args.c_strategy)
            official_pred = score_with_params(resp[:, valid_cols], official_valid, area, tabela)
            mask = ~np.isnan(our_oof) & ~np.isnan(nota_ref) & (nota_ref > 0)
            attribution[area] = decompose_error(our_oof[mask], official_pred[mask], nota_ref[mask])

            recovery[area] = {str(args.year): run_recovery_for_group(resp, gab, official)}

    report = build_report(
        year=args.year, end_to_end=end_to_end,
        attribution=attribution, recovery=recovery,
    )
    write_report(report, args.out)
    print(render_text_table(report))
    print(f"\nRelatório JSON salvo em: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
