"""
Calculador de transformação empírica do ENEM.

Para cada (ano, área), deriva os parâmetros (σ, μ) da transformação
linear que converte theta na escala ENEM:

    Nota_ENEM = σ * theta + μ

Onde:
  • theta é estimado via EAP usando os parâmetros oficiais do INEP
  • Nota_ENEM é a NU_NOTA_{area} dos microdados (verdade fundamental)
  • A regressão usa apenas alunos de PRIMEIRA APLICAÇÃO com 5 provas válidas

Resultado salvo em enem_transformacoes (uma linha por ano × área).

Uso:
    python -m app.scripts.calcular_transformacao --ano 2024 --area MT
    python -m app.scripts.calcular_transformacao --ano 2024            # todas as áreas
    python -m app.scripts.calcular_transformacao --all                 # todos anos × todas áreas
    python -m app.scripts.calcular_transformacao --ano 2024 --max-alunos 10000

Variáveis de ambiente:
    SUPABASE_URL, SUPABASE_KEY, MICRODATA_PATH
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

# Carrega .env
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

# Mapeamento ano → arquivo de microdados (no MICRODATA_PATH)
RESULTADOS_FILES = {
    2022: "dados 2022/RESULTADO_2022.csv",
    2023: "dados 2023/MICRODADOS_ENEM_2023.csv",
    2024: "dados 2024/RESULTADOS_2024.csv",
}

# Códigos de prova de primeira aplicação por ano e área
# Identificamos via TX_COR no banco — provas que NÃO contém "Reaplic" no nome
# Como cache: aqui ficam os códigos conhecidos. Se vazio, descobrimos do banco.
DEFAULT_MAX_ALUNOS = 5000  # amostra suficiente para correlação 1.0 (já validamos com 500)


# ════════════════════════════════════════════════════════════════════════════════
# CLIENTE SUPABASE
# ════════════════════════════════════════════════════════════════════════════════

def get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("ERRO: SUPABASE_URL ou SUPABASE_KEY não definidos. Verifique .env")
        sys.exit(1)
    return create_client(url, key.strip())


def fetch_all(client: Client, table: str, filter_eq: Optional[Dict] = None,
              select: str = "*", page_size: int = 1000) -> List[dict]:
    """Pagina todas as rows de uma tabela com filtros opcionais."""
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


# ════════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DO ITEM BANK
# ════════════════════════════════════════════════════════════════════════════════

def load_item_bank(client: Client, ano: int, area: str) -> Tuple[Dict[int, List[dict]], set]:
    """
    Carrega parâmetros TRI por (co_prova → lista de itens ordenados por posição).

    Filtra para primeira aplicação apenas (is_reaplicacao=False na metadata).

    Returns:
        provas: {co_prova: [{posicao, co_item, a, b, c, gabarito, tp_lingua}]}
        provas_primeira: set de co_prova de primeira aplicação
    """
    print(f"  ▸ Carregando item bank do Supabase para {area}/{ano}...")

    # 1. Códigos de prova de primeira aplicação
    codigos = fetch_all(client, "enem_codigos_prova",
                        filter_eq={"ano": ano, "area": area})
    provas_primeira = {c["co_prova"] for c in codigos if not c.get("is_reaplicacao", False)}
    print(f"    • {len(codigos)} cores totais, {len(provas_primeira)} de primeira aplicação")

    # 2. Itens com parâmetros (deduplicado por co_item)
    itens = fetch_all(client, "enem_itens",
                      filter_eq={"ano": ano, "area": area})
    itens_idx = {it["co_item"]: it for it in itens}
    print(f"    • {len(itens)} itens com parâmetros TRI")

    # 3. Mapeamento item → prova × posição
    mapeamentos = fetch_all(client, "enem_itens_prova", filter_eq={"ano": ano})
    # Filtrar para apenas as provas dessa área
    mapeamentos_area = [m for m in mapeamentos if m["co_item"] in itens_idx]
    print(f"    • {len(mapeamentos_area)} mapeamentos item→prova para a área")

    # 4. Construir índice por co_prova
    provas: Dict[int, List[dict]] = defaultdict(list)
    for m in mapeamentos_area:
        co_item = m["co_item"]
        item = itens_idx.get(co_item)
        if not item:
            continue
        # Pula itens abandonados ou sem parâmetros
        if item.get("item_abandonado"):
            continue
        if item.get("param_discriminacao") is None:
            continue
        provas[m["co_prova"]].append({
            "posicao": m["posicao"],
            "co_item": co_item,
            "a": float(item["param_discriminacao"]),
            "b": float(item["param_dificuldade"]),
            "c": float(item.get("param_acerto_casual") or 0.20),
            "gabarito": item.get("gabarito"),
            "tp_lingua": item.get("tipo_lingua"),
            "abandonado": False,
        })

    # Ordena cada prova por posição
    for cp in provas:
        provas[cp].sort(key=lambda x: x["posicao"])

    return dict(provas), provas_primeira


# ════════════════════════════════════════════════════════════════════════════════
# EAP THETA
# ════════════════════════════════════════════════════════════════════════════════

def eap_theta(responses: np.ndarray, item_params: List[dict], n_quad: int = 61) -> float:
    """EAP estimation com prior N(0,1) e quadratura gaussiana."""
    theta_grid = np.linspace(-5, 5, n_quad)
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
# PROCESSAMENTO DE UM ALUNO
# ════════════════════════════════════════════════════════════════════════════════

# Offset por área no exam-wide CO_POSICAO (1..180)
POS_OFFSET = {"LC": 1, "CH": 46, "CN": 91, "MT": 136}


def filter_items_for_student(items: List[dict], tp_lingua: Optional[int]) -> List[dict]:
    """
    Para LC: mantém só os itens da língua do aluno (5 itens) + itens comuns (40 itens).
    Para outras áreas: retorna todos os itens.
    """
    if not items:
        return items
    # Detecta se há itens com tp_lingua definido (caso LC)
    has_lingua = any(it.get("tp_lingua") is not None for it in items)
    if not has_lingua:
        return items
    return [it for it in items if it.get("tp_lingua") is None or it.get("tp_lingua") == tp_lingua]


def calcular_theta_aluno(
    resp_str: str,
    items_prova: List[dict],
    area: str,
    tp_lingua: Optional[int],
) -> Optional[float]:
    """
    Reconstrói o vetor binário de respostas e calcula theta via EAP.
    Usa o gabarito armazenado em cada item (não a string TX_GABARITO_*),
    o que evita problemas com o tamanho variável de TX_GABARITO_LC (50 chars).

    Retorna None se os dados forem inválidos.
    """
    if not resp_str or len(resp_str) != 45:
        return None

    # Filtra itens para a língua do aluno (relevante apenas para LC)
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
        # Item anulado ou sem gabarito: crédito a todos
        if not gab or gab in "X.":
            binary.append(1)
        elif student_answer == gab:
            binary.append(1)
        else:
            binary.append(0)
        valid_items.append(it)

    if len(binary) < 30:  # menos de 30 itens válidos é insuficiente
        return None

    return eap_theta(np.array(binary), valid_items)


# ════════════════════════════════════════════════════════════════════════════════
# STREAMING DOS MICRODADOS
# ════════════════════════════════════════════════════════════════════════════════

def detect_id_column(headers: List[str]) -> str:
    """Detecta a coluna de ID: NU_SEQUENCIAL (2024+) ou NU_INSCRICAO (2022/2023)."""
    if "NU_SEQUENCIAL" in headers:
        return "NU_SEQUENCIAL"
    if "NU_INSCRICAO" in headers:
        return "NU_INSCRICAO"
    return headers[0]


def processar_microdados(
    csv_path: str,
    ano: int,
    area: str,
    item_bank: Dict[int, List[dict]],
    provas_primeira: set,
    max_alunos: int,
    progress_every: int = 50000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stream do RESULTADOS_{ano}.csv. Para cada aluno válido, computa theta e
    coleta a nota oficial (NU_NOTA_{area}).

    Retorna (thetas, notas) como arrays numpy.
    """
    print(f"  ▸ Streaming microdados de {csv_path}")
    print(f"    • Coletando até {max_alunos:,} alunos válidos")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    thetas: List[float] = []
    notas: List[float] = []

    n_total = 0
    n_filtrados_presenca = 0
    n_filtrados_prova = 0
    n_processados = 0
    t0 = time.time()

    with open(csv_path, "r", encoding=ENCODING, errors="replace") as f:
        reader = csv.DictReader(f, delimiter=";")
        id_col = detect_id_column(reader.fieldnames)

        for row in reader:
            n_total += 1

            # 1. Filtro de presença em todas as 4 áreas + redação OK
            if not (row.get(f"TP_PRESENCA_{area}") == "1" and
                    row.get("TP_PRESENCA_LC") == "1" and
                    row.get("TP_PRESENCA_CH") == "1" and
                    row.get("TP_PRESENCA_CN") == "1" and
                    row.get("TP_PRESENCA_MT") == "1" and
                    row.get("TP_STATUS_REDACAO") == "1"):
                n_filtrados_presenca += 1
                continue

            # 2. Filtro de primeira aplicação
            try:
                co_prova = int(row[f"CO_PROVA_{area}"])
            except (KeyError, ValueError, TypeError):
                continue

            if co_prova not in provas_primeira:
                n_filtrados_prova += 1
                continue

            # 3. Verificar se temos os parâmetros dessa prova
            items_prova = item_bank.get(co_prova)
            if not items_prova:
                continue

            # 4. Reconstruir respostas (usa gabarito do item bank, não TX_GABARITO_*)
            resp_str = row.get(f"TX_RESPOSTAS_{area}", "")
            tp_lingua_str = row.get("TP_LINGUA", "")
            tp_lingua = int(tp_lingua_str) if tp_lingua_str else None

            theta = calcular_theta_aluno(resp_str, items_prova, area, tp_lingua)
            if theta is None:
                continue

            try:
                nota = float(row[f"NU_NOTA_{area}"].replace(",", "."))
            except (ValueError, AttributeError):
                continue
            if nota <= 0:
                continue

            thetas.append(theta)
            notas.append(nota)
            n_processados += 1

            if n_processados >= max_alunos:
                break

            if n_total % progress_every == 0:
                elapsed = time.time() - t0
                rate = n_total / elapsed
                print(f"    • Lidos {n_total:,} | válidos {n_processados:,} | {rate:.0f} rows/s")

    elapsed = time.time() - t0
    print(f"    • Total lido: {n_total:,} rows em {elapsed:.1f}s")
    print(f"    • Filtrados (presença): {n_filtrados_presenca:,}")
    print(f"    • Filtrados (prova reaplic.): {n_filtrados_prova:,}")
    print(f"    • Válidos coletados: {n_processados:,}")

    return np.array(thetas), np.array(notas)


# ════════════════════════════════════════════════════════════════════════════════
# REGRESSÃO E PERSISTÊNCIA
# ════════════════════════════════════════════════════════════════════════════════

def calcular_regressao(thetas: np.ndarray, notas: np.ndarray) -> Dict[str, float]:
    """Regressão linear nota = σ * theta + μ."""
    if len(thetas) < 10:
        raise ValueError(f"Amostra insuficiente: {len(thetas)} alunos")

    if thetas.std() < 1e-6:
        raise ValueError("Variância de theta = 0 — algo está errado")

    sigma, mu = np.polyfit(thetas, notas, 1)
    pred = sigma * thetas + mu
    mae = float(np.mean(np.abs(pred - notas)))
    corr = float(np.corrcoef(thetas, notas)[0, 1])

    return {
        "sigma": float(sigma),
        "mu": float(mu),
        "mae": mae,
        "correlacao": corr,
        "n_alunos": int(len(thetas)),
        "theta_mean": float(thetas.mean()),
        "theta_std": float(thetas.std()),
        "nota_mean": float(notas.mean()),
        "nota_std": float(notas.std()),
    }


def salvar_transformacao(client: Client, ano: int, area: str, result: Dict[str, float]) -> None:
    """Insert ou update na tabela enem_transformacoes."""
    payload = {
        "ano": ano,
        "area": area,
        "sigma": round(result["sigma"], 4),
        "mu": round(result["mu"], 4),
        "n_alunos": result["n_alunos"],
        "correlacao": round(result["correlacao"], 6),
        "mae": round(result["mae"], 4),
        "observacoes": (
            f"theta_mean={result['theta_mean']:.3f} "
            f"theta_std={result['theta_std']:.3f} "
            f"nota_mean={result['nota_mean']:.1f} "
            f"nota_std={result['nota_std']:.1f}"
        ),
    }

    # Tenta upsert: delete + insert (já que não temos ON CONFLICT direto)
    try:
        client.table("enem_transformacoes").delete().eq("ano", ano).eq("area", area).execute()
    except Exception:
        pass

    client.table("enem_transformacoes").insert(payload).execute()
    print(f"  ✓ Salvo em enem_transformacoes")


# ════════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO PARA UM (ANO, ÁREA)
# ════════════════════════════════════════════════════════════════════════════════

def processar_ano_area(
    client: Client,
    ano: int,
    area: str,
    microdata_path: str,
    max_alunos: int,
    dry_run: bool,
) -> Optional[Dict[str, float]]:
    """Executa o pipeline completo para um par (ano, área)."""
    print(f"\n{'─'*70}")
    print(f"  {area} / {ano}")
    print(f"{'─'*70}")

    # 1. Carregar item bank
    try:
        item_bank, provas_primeira = load_item_bank(client, ano, area)
    except Exception as e:
        print(f"  ✗ Erro ao carregar item bank: {e}")
        return None

    if not item_bank:
        print(f"  ✗ Sem itens no banco para {area}/{ano}")
        return None

    # 2. Localizar microdata
    rel_path = RESULTADOS_FILES.get(ano)
    if not rel_path:
        print(f"  ✗ Sem mapeamento de arquivo para ano {ano}")
        return None
    csv_path = os.path.join(microdata_path, rel_path)

    # 3. Stream + EAP
    try:
        thetas, notas = processar_microdados(csv_path, ano, area, item_bank, provas_primeira, max_alunos)
    except FileNotFoundError as e:
        print(f"  ✗ Arquivo não encontrado: {e}")
        return None

    if len(thetas) < 100:
        print(f"  ✗ Amostra muito pequena: {len(thetas)}")
        return None

    # 4. Regressão
    print(f"\n  ▸ Regressão linear...")
    result = calcular_regressao(thetas, notas)
    print(f"    • Nota = {result['sigma']:.4f} * theta + {result['mu']:.4f}")
    print(f"    • Correlação: {result['correlacao']:.6f}")
    print(f"    • MAE: {result['mae']:.4f}")
    print(f"    • N: {result['n_alunos']}")
    print(f"    • theta: mean={result['theta_mean']:.3f} std={result['theta_std']:.3f}")
    print(f"    • nota:  mean={result['nota_mean']:.1f} std={result['nota_std']:.1f}")

    # Sanity check
    if result["correlacao"] < 0.95:
        print(f"  ⚠ AVISO: correlação baixa ({result['correlacao']:.4f}) — algo pode estar errado")
    if result["mae"] > 5.0:
        print(f"  ⚠ AVISO: MAE alto ({result['mae']:.2f}) — algo pode estar errado")

    # 5. Persistência
    if dry_run:
        print(f"  [DRY-RUN] Não salvo no banco")
    else:
        salvar_transformacao(client, ano, area, result)

    return result


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Calculador de transformação empírica ENEM")
    parser.add_argument("--ano", type=int, help="Ano específico (ex: 2024)")
    parser.add_argument("--area", type=str, help="Área específica (LC, CH, CN, MT)")
    parser.add_argument("--all", action="store_true", help="Processar todos os anos e áreas")
    parser.add_argument("--max-alunos", type=int, default=DEFAULT_MAX_ALUNOS,
                        help=f"Máximo de alunos por ano×área (padrão: {DEFAULT_MAX_ALUNOS})")
    parser.add_argument("--dry-run", action="store_true", help="Calcula sem salvar no banco")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  XTRI Engine — Calculador de Transformação Empírica ENEM")
    print(f"{'='*70}")

    microdata_path = os.getenv("MICRODATA_PATH")
    if not microdata_path:
        print("ERRO: MICRODATA_PATH não definido no .env")
        sys.exit(1)
    print(f"  Microdata path: {microdata_path}")
    print(f"  Max alunos: {args.max_alunos}")
    print(f"  Dry-run: {args.dry_run}")

    client = get_client()

    # Determinar combinações a processar
    todas_areas = ["LC", "CH", "CN", "MT"]
    todos_anos = [2022, 2023, 2024]

    if args.all:
        combinacoes = [(a, ar) for a in todos_anos for ar in todas_areas]
    elif args.ano and args.area:
        combinacoes = [(args.ano, args.area.upper())]
    elif args.ano:
        combinacoes = [(args.ano, ar) for ar in todas_areas]
    else:
        print("ERRO: especifique --ano, --ano + --area, ou --all")
        sys.exit(1)

    print(f"  Combinações: {len(combinacoes)}")
    print(f"  Lista: {combinacoes}")

    resultados = []
    t_inicio = time.time()
    for ano, area in combinacoes:
        try:
            r = processar_ano_area(client, ano, area, microdata_path, args.max_alunos, args.dry_run)
            if r:
                resultados.append((ano, area, r))
        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário")
            break
        except Exception as e:
            import traceback
            print(f"\n  ✗ ERRO em {area}/{ano}: {e}")
            traceback.print_exc()

    # Sumário final
    print(f"\n{'='*70}")
    print(f"  RESUMO FINAL ({time.time()-t_inicio:.0f}s)")
    print(f"{'='*70}")
    print(f"\n  {'Ano':>5} {'Area':>5} {'Sigma':>10} {'Mu':>10} {'Corr':>10} {'MAE':>8} {'N':>8}")
    print(f"  {'-'*5} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for ano, area, r in resultados:
        print(f"  {ano:>5} {area:>5} {r['sigma']:>10.4f} {r['mu']:>10.4f} "
              f"{r['correlacao']:>10.6f} {r['mae']:>8.4f} {r['n_alunos']:>8}")

    print(f"\n  ✓ {len(resultados)}/{len(combinacoes)} combinações processadas com sucesso")


if __name__ == "__main__":
    main()
