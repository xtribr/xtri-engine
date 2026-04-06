"""
Validação em massa do XTRI Engine contra os microdados completos do INEP.

Para cada (ano, área), processa centenas de milhares de alunos reais do
ENEM e mede a precisão do XTRI Engine calculando:

  • MAE, RMSE, correlação vs NU_NOTA_{area} oficial
  • Distribuição de diferenças (percentis 5, 25, 50, 75, 95)
  • Acurácia da classificação de níveis pedagógicos

Este é o benchmark científico definitivo usado no paper acadêmico.

Uso:
    python -m app.scripts.validacao_massa --ano 2024 --area MT --max-alunos 100000
    python -m app.scripts.validacao_massa --all --max-alunos 50000

Saída:
    • Console: relatório detalhado
    • JSON: /tmp/validacao_{ano}_{area}.json com todos os pares (theta, real, v4)
    • enem_validacoes: tabela opcional no Supabase com o resumo (se existir)
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    root_env = Path(__file__).resolve().parents[3] / ".env"
    if root_env.exists():
        load_dotenv(root_env)
except ImportError:
    pass

from supabase import Client, create_client


# ════════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════════

ENCODING = "latin-1"

RESULTADOS_FILES = {
    2022: "dados 2022/RESULTADO_2022.csv",
    2023: "dados 2023/MICRODADOS_ENEM_2023.csv",
    2024: "dados 2024/RESULTADOS_2024.csv",
}

POS_OFFSET = {"LC": 1, "CH": 46, "CN": 91, "MT": 136}


# ════════════════════════════════════════════════════════════════════════════════
# SUPABASE
# ════════════════════════════════════════════════════════════════════════════════

def get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("ERRO: SUPABASE_URL/SUPABASE_KEY não definidos")
        sys.exit(1)
    return create_client(url, key.strip())


def fetch_all(client: Client, table: str, filter_eq: Optional[Dict] = None,
              select: str = "*", page_size: int = 1000) -> List[dict]:
    rows = []
    offset = 0
    while True:
        q = client.table(table).select(select).range(offset, offset + page_size - 1)
        if filter_eq:
            for k, v in filter_eq.items():
                q = q.eq(k, v)
        r = q.execute()
        if not r.data:
            break
        rows.extend(r.data)
        if len(r.data) < page_size:
            break
        offset += page_size
    return rows


def load_item_bank_and_transformacao(
    client: Client, ano: int, area: str
) -> Tuple[Dict[int, List[dict]], Dict[str, float]]:
    """Carrega itens por prova e a transformação (σ, μ)."""
    itens = fetch_all(client, "enem_itens", filter_eq={"ano": ano, "area": area})
    itens_idx = {it["co_item"]: it for it in itens}

    maps = fetch_all(client, "enem_itens_prova", filter_eq={"ano": ano})
    provas: Dict[int, List[dict]] = defaultdict(list)
    for m in maps:
        it = itens_idx.get(m["co_item"])
        if not it or it.get("item_abandonado"):
            continue
        if it.get("param_discriminacao") is None:
            continue
        provas[m["co_prova"]].append({
            "posicao": m["posicao"],
            "a": float(it["param_discriminacao"]),
            "b": float(it["param_dificuldade"]),
            "c": float(it.get("param_acerto_casual") or 0.20),
            "gabarito": it.get("gabarito"),
            "tp_lingua": it.get("tipo_lingua"),
        })
    for cp in provas:
        provas[cp].sort(key=lambda x: x["posicao"])

    # Transformação
    t = client.table("enem_transformacoes").select("*").eq("ano", ano).eq("area", area).limit(1).execute()
    if not t.data:
        raise ValueError(f"Transformação não encontrada para {area}/{ano}")
    transf = t.data[0]
    return dict(provas), {
        "sigma": float(transf["sigma"]),
        "mu": float(transf["mu"]),
        "n_calibracao": transf["n_alunos"] or 0,
    }


# ════════════════════════════════════════════════════════════════════════════════
# EAP
# ════════════════════════════════════════════════════════════════════════════════

def eap_theta(responses: np.ndarray, item_params: List[dict], n_quad: int = 61) -> float:
    theta_grid = np.linspace(-4, 4, n_quad)
    prior = norm.pdf(theta_grid, 0, 1)
    lk = np.ones(n_quad)
    for j, p in enumerate(item_params):
        a, b, c = p["a"], p["b"], p["c"]
        prob = c + (1 - c) / (1 + np.exp(-a * (theta_grid - b)))
        prob = np.clip(prob, 1e-9, 1 - 1e-9)
        lk *= np.where(responses[j] == 1, prob, 1 - prob)
    posterior = lk * prior
    s = posterior.sum()
    if s < 1e-300:
        return 0.0
    posterior /= s
    return float(np.sum(theta_grid * posterior))


# ════════════════════════════════════════════════════════════════════════════════
# NIVEIS PEDAGÓGICOS (mesmo mapeamento do motor cientifico)
# ════════════════════════════════════════════════════════════════════════════════

def estado_tri(nota: float) -> str:
    if nota >= 700: return "AVANCADO"
    if nota >= 600: return "PROFICIENTE"
    if nota >= 500: return "BASICO"
    if nota >= 400: return "ELEMENTAR"
    return "INICIANTE"


# ════════════════════════════════════════════════════════════════════════════════
# PROCESSAMENTO DE UM ALUNO
# ════════════════════════════════════════════════════════════════════════════════

def filter_items_for_student(items: List[dict], tp_lingua: Optional[int]) -> List[dict]:
    has_lingua = any(it.get("tp_lingua") is not None for it in items)
    if not has_lingua:
        return items
    return [it for it in items if it.get("tp_lingua") is None or it.get("tp_lingua") == tp_lingua]


def processar_aluno(
    resp_str: str,
    items_prova: List[dict],
    area: str,
    tp_lingua: Optional[int],
) -> Optional[Tuple[float, int]]:
    """Computa theta de um aluno. Retorna (theta, acertos) ou None."""
    if not resp_str or len(resp_str) != 45:
        return None

    items = filter_items_for_student(items_prova, tp_lingua)
    if not items:
        return None

    offset = POS_OFFSET[area]
    binary = []
    valid_items = []
    for it in items:
        str_idx = it["posicao"] - offset
        if not (0 <= str_idx < 45):
            continue
        student_answer = resp_str[str_idx]
        gab = it.get("gabarito")
        if not gab or gab in "X.":
            binary.append(1)
        elif student_answer == gab:
            binary.append(1)
        else:
            binary.append(0)
        valid_items.append(it)

    if len(binary) < 30:
        return None

    theta = eap_theta(np.array(binary), valid_items)
    return theta, sum(binary)


# ════════════════════════════════════════════════════════════════════════════════
# PIPELINE DE VALIDAÇÃO EM MASSA
# ════════════════════════════════════════════════════════════════════════════════

def validar_ano_area(
    client: Client,
    ano: int,
    area: str,
    microdata_path: str,
    max_alunos: int,
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Processa e valida um par (ano, área) contra os microdados."""
    print(f"\n{'─'*70}")
    print(f"  VALIDACAO MASSA: {area} / {ano}")
    print(f"{'─'*70}")

    # 1. Carregar item bank + transformação
    item_bank, transf = load_item_bank_and_transformacao(client, ano, area)
    sigma, mu = transf["sigma"], transf["mu"]
    print(f"  • σ={sigma:.4f}  μ={mu:.4f}  (calibrada com {transf['n_calibracao']} alunos)")
    print(f"  • {len(item_bank)} provas no banco")

    # 2. Localizar microdata
    rel = RESULTADOS_FILES.get(ano)
    if not rel:
        raise ValueError(f"Sem mapeamento de arquivo para ano {ano}")
    csv_path = os.path.join(microdata_path, rel)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    print(f"  • Streaming {csv_path}")

    # 3. Stream + validação
    t0 = time.time()
    pairs = []  # (theta_xtri, nota_xtri, nota_inep, acertos, nivel_xtri, nivel_inep)
    n_scanned = 0
    n_valid = 0
    n_skipped_presenca = 0
    n_skipped_items = 0
    n_skipped_theta = 0

    with open(csv_path, "r", encoding=ENCODING, errors="replace") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            n_scanned += 1

            # Filtro: primeira aplicação + 5 provas + redação OK
            if not (row.get(f"TP_PRESENCA_{area}") == "1" and
                    row.get("TP_PRESENCA_LC") == "1" and
                    row.get("TP_PRESENCA_CH") == "1" and
                    row.get("TP_PRESENCA_CN") == "1" and
                    row.get("TP_PRESENCA_MT") == "1" and
                    row.get("TP_STATUS_REDACAO") == "1"):
                n_skipped_presenca += 1
                continue

            try:
                co_prova = int(row[f"CO_PROVA_{area}"])
            except (KeyError, ValueError, TypeError):
                continue

            items_prova = item_bank.get(co_prova)
            if not items_prova:
                n_skipped_items += 1
                continue

            resp_str = row.get(f"TX_RESPOSTAS_{area}", "")
            tp_lingua_str = row.get("TP_LINGUA", "")
            tp_lingua = int(tp_lingua_str) if tp_lingua_str else None

            result = processar_aluno(resp_str, items_prova, area, tp_lingua)
            if result is None:
                n_skipped_theta += 1
                continue
            theta_xtri, acertos = result

            nota_xtri = sigma * theta_xtri + mu

            try:
                nota_inep = float(row[f"NU_NOTA_{area}"].replace(",", "."))
            except (ValueError, AttributeError, KeyError):
                continue
            if nota_inep <= 0:
                continue

            nivel_xtri = estado_tri(nota_xtri)
            nivel_inep = estado_tri(nota_inep)

            pairs.append((theta_xtri, nota_xtri, nota_inep, acertos, nivel_xtri, nivel_inep))
            n_valid += 1

            if n_valid >= max_alunos:
                break

            if n_scanned % 100000 == 0:
                elapsed = time.time() - t0
                rate = n_scanned / elapsed
                print(f"  • Processados: {n_scanned:,} | válidos: {n_valid:,} | {rate:.0f} rows/s")

    elapsed = time.time() - t0
    print(f"\n  ▸ Stream concluído em {elapsed:.0f}s")
    print(f"    • Lidos: {n_scanned:,}")
    print(f"    • Válidos: {n_valid:,}")
    print(f"    • Filtrados (presença): {n_skipped_presenca:,}")
    print(f"    • Filtrados (item bank): {n_skipped_items:,}")
    print(f"    • Filtrados (theta): {n_skipped_theta:,}")

    if n_valid < 100:
        raise ValueError(f"Amostra muito pequena: {n_valid}")

    # 4. Calcular métricas
    thetas = np.array([p[0] for p in pairs])
    notas_xtri = np.array([p[1] for p in pairs])
    notas_inep = np.array([p[2] for p in pairs])
    acertos_arr = np.array([p[3] for p in pairs])
    diffs = notas_xtri - notas_inep

    mae = float(np.mean(np.abs(diffs)))
    rmse = float(np.sqrt(np.mean(diffs ** 2)))
    mean_diff = float(np.mean(diffs))
    max_abs_diff = float(np.max(np.abs(diffs)))
    correlacao = float(np.corrcoef(notas_xtri, notas_inep)[0, 1])

    # Percentis das diferenças absolutas
    abs_diffs = np.abs(diffs)
    pct5 = float(np.percentile(abs_diffs, 5))
    pct25 = float(np.percentile(abs_diffs, 25))
    pct50 = float(np.percentile(abs_diffs, 50))
    pct75 = float(np.percentile(abs_diffs, 75))
    pct95 = float(np.percentile(abs_diffs, 95))

    # Acurácia de nível
    nivel_xtri_list = [p[4] for p in pairs]
    nivel_inep_list = [p[5] for p in pairs]
    concordancias = sum(1 for a, b in zip(nivel_xtri_list, nivel_inep_list) if a == b)
    acc_nivel = concordancias / len(pairs) * 100

    # Diferença por nível de acertos
    print(f"\n  ▸ MÉTRICAS GLOBAIS")
    print(f"    • N: {n_valid:,}")
    print(f"    • Correlação: {correlacao:.8f}")
    print(f"    • MAE: {mae:.4f}")
    print(f"    • RMSE: {rmse:.4f}")
    print(f"    • Mean diff: {mean_diff:+.4f}")
    print(f"    • Max |diff|: {max_abs_diff:.4f}")
    print(f"    • Percentis |diff|: p5={pct5:.3f} p25={pct25:.3f} p50={pct50:.3f} p75={pct75:.3f} p95={pct95:.3f}")
    print(f"    • Acurácia de nível: {acc_nivel:.2f}%")

    resumo = {
        "ano": ano,
        "area": area,
        "n_alunos": n_valid,
        "sigma": sigma,
        "mu": mu,
        "correlacao": correlacao,
        "mae": mae,
        "rmse": rmse,
        "mean_diff": mean_diff,
        "max_abs_diff": max_abs_diff,
        "pct5": pct5,
        "pct25": pct25,
        "pct50": pct50,
        "pct75": pct75,
        "pct95": pct95,
        "acc_nivel_pct": acc_nivel,
        "tempo_segundos": elapsed,
    }

    if output_json:
        # Salvar todos os pares para análise posterior
        dump = {
            "meta": resumo,
            "pairs": [
                {"theta": t, "nota_xtri": nx, "nota_inep": ni, "acertos": ac}
                for t, nx, ni, ac, _, _ in pairs
            ],
        }
        with open(output_json, "w") as f:
            json.dump(dump, f)
        print(f"\n  ✓ Dados salvos em {output_json}")

    return resumo


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Validação em massa do XTRI Engine vs microdados INEP")
    parser.add_argument("--ano", type=int, help="Ano específico")
    parser.add_argument("--area", type=str, help="Área (LC, CH, CN, MT)")
    parser.add_argument("--all", action="store_true", help="Todos os anos × áreas")
    parser.add_argument("--max-alunos", type=int, default=100_000,
                        help="Máximo de alunos por combinação (default: 100000)")
    parser.add_argument("--save-json", action="store_true", help="Salvar pares em JSON em /tmp/")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  XTRI Engine — Validação em Massa vs Microdados INEP")
    print(f"{'='*70}")

    microdata_path = os.getenv("MICRODATA_PATH")
    if not microdata_path:
        print("ERRO: MICRODATA_PATH não definido")
        sys.exit(1)
    print(f"  Microdata: {microdata_path}")
    print(f"  Max alunos por combo: {args.max_alunos:,}")

    client = get_client()

    todas_areas = ["LC", "CH", "CN", "MT"]
    todos_anos = [2022, 2023, 2024]

    if args.all:
        combinacoes = [(a, ar) for a in todos_anos for ar in todas_areas]
    elif args.ano and args.area:
        combinacoes = [(args.ano, args.area.upper())]
    elif args.ano:
        combinacoes = [(args.ano, ar) for ar in todas_areas]
    else:
        print("ERRO: especifique --ano, --ano+--area, ou --all")
        sys.exit(1)

    print(f"  Combinações: {len(combinacoes)}")

    resultados = []
    t_total = time.time()

    for ano, area in combinacoes:
        try:
            json_out = f"/tmp/validacao_{ano}_{area}.json" if args.save_json else None
            r = validar_ano_area(client, ano, area, microdata_path, args.max_alunos, json_out)
            resultados.append(r)
        except Exception as e:
            import traceback
            print(f"\n  ✗ ERRO em {area}/{ano}: {e}")
            traceback.print_exc()

    # Relatório final
    print(f"\n{'='*70}")
    print(f"  RELATÓRIO FINAL ({time.time()-t_total:.0f}s)")
    print(f"{'='*70}\n")
    print(f"  {'Ano':>5} {'Area':>5} {'N':>10} {'Corr':>10} {'MAE':>8} {'RMSE':>8} {'MeanDiff':>10} {'MaxDiff':>10} {'Acc%':>8}")
    print(f"  {'-'*5} {'-'*5} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for r in resultados:
        print(f"  {r['ano']:>5} {r['area']:>5} {r['n_alunos']:>10,} "
              f"{r['correlacao']:>10.6f} {r['mae']:>8.3f} {r['rmse']:>8.3f} "
              f"{r['mean_diff']:>+10.3f} {r['max_abs_diff']:>10.2f} {r['acc_nivel_pct']:>7.2f}%")

    if resultados:
        avg_corr = np.mean([r["correlacao"] for r in resultados])
        avg_mae = np.mean([r["mae"] for r in resultados])
        avg_acc = np.mean([r["acc_nivel_pct"] for r in resultados])
        print(f"\n  MÉDIAS:")
        print(f"    • Correlação média: {avg_corr:.6f}")
        print(f"    • MAE médio:        {avg_mae:.4f}")
        print(f"    • Acurácia nível:   {avg_acc:.2f}%")

    # Salvar resumo final
    if resultados:
        out_file = f"/tmp/validacao_massa_resumo_{int(time.time())}.json"
        with open(out_file, "w") as f:
            json.dump(resultados, f, indent=2)
        print(f"\n  ✓ Resumo salvo em {out_file}")


if __name__ == "__main__":
    main()
