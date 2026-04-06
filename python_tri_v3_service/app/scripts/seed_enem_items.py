"""
Seeder do banco de itens ENEM (item bank) no Supabase.

Lê o arquivo ITENS_PROVA_{ano}.csv (publicado pelo INEP) e popula 3 tabelas:

  • enem_itens         — um item único por (co_item, ano) com parâmetros TRI
  • enem_itens_prova   — mapeamento (co_item × co_prova × posicao × cor)
  • enem_codigos_prova — metadados de cada versão de prova (cor, reaplicação)

Uso:
    python -m app.scripts.seed_enem_items --ano 2024 --file /data/dados\\ 2024/ITENS_PROVA_2024.csv
    python -m app.scripts.seed_enem_items --ano 2024 --file /data/... --reset

A flag --reset apaga primeiro todos os registros do ano antes de inserir.

Variáveis de ambiente:
    SUPABASE_URL, SUPABASE_KEY (carregadas do .env automaticamente)
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Carrega .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    # Tenta também carregar do diretório raiz do projeto
    root_env = Path(__file__).resolve().parents[3] / ".env"
    if root_env.exists():
        load_dotenv(root_env)
except ImportError:
    pass

from supabase import Client, create_client

BATCH_SIZE = 500
ENCODING = "latin-1"  # INEP usa Latin-1 nos CSVs


# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTES — códigos de prova de primeira aplicação vs reaplicação
# ════════════════════════════════════════════════════════════════════════════════
# Baseado no script INPUT_R_RESULTADOS_2024.R do INEP

PRIMEIRA_APLICACAO = {
    # Todos os anos seguem padrão: códigos altos = primeira aplicação,
    # códigos baixos = reaplicação. Mas o range varia por ano.
    # Para identificar reaplicação, usamos a coluna TX_COR que contém o sufixo.
}


def is_reaplicacao(tx_cor: str) -> bool:
    """Identifica se uma cor de prova é reaplicação."""
    if not tx_cor:
        return False
    s = tx_cor.upper()
    return "REAPLIC" in s or "REAP." in s


# ════════════════════════════════════════════════════════════════════════════════
# PARSER DO CSV
# ════════════════════════════════════════════════════════════════════════════════

def parse_itens_prova_csv(csv_path: str, ano: int) -> Dict[str, Any]:
    """
    Parse o arquivo ITENS_PROVA_{ano}.csv e retorna estruturas prontas para insert.

    Returns:
        {
            'itens_unicos': [...],         # uma linha por (co_item, ano)
            'itens_prova': [...],          # uma linha por (co_item, co_prova, posicao, cor)
            'codigos_prova': [...],        # uma linha por (ano, area, co_prova)
            'stats': {area: {n_itens: int, n_provas: int}}
        }
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    itens_unicos: Dict[tuple, dict] = {}
    itens_prova: List[dict] = []
    codigos_prova: Dict[tuple, dict] = {}
    stats: Dict[str, Dict[str, set]] = defaultdict(lambda: {"itens": set(), "provas": set()})

    n_rows = 0
    n_skipped = 0

    with open(csv_path, "r", encoding=ENCODING, errors="replace") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            n_rows += 1
            try:
                co_item = int(row["CO_ITEM"])
                co_prova = int(row["CO_PROVA"])
                area = row["SG_AREA"].strip()
                posicao = int(row["CO_POSICAO"])
                cor = (row.get("TX_COR") or "").strip()
                gabarito = (row.get("TX_GABARITO") or "").strip() or None
                hab = row.get("CO_HABILIDADE")
                hab_int = int(hab) if hab and hab.strip() else None
                aban_str = (row.get("IN_ITEM_ABAN") or "0").strip()
                aban = aban_str == "1"
                motivo = (row.get("TX_MOTIVO_ABAN") or "").strip() or None
                tp_lingua_str = (row.get("TP_LINGUA") or "").strip()
                tp_lingua = int(tp_lingua_str) if tp_lingua_str else None

                # Parâmetros TRI (podem estar vazios para itens abandonados)
                a_str = (row.get("NU_PARAM_A") or "").strip()
                b_str = (row.get("NU_PARAM_B") or "").strip()
                c_str = (row.get("NU_PARAM_C") or "").strip()
                a_val = float(a_str) if a_str else None
                b_val = float(b_str) if b_str else None
                c_val = float(c_str) if c_str else None
            except (KeyError, ValueError) as e:
                n_skipped += 1
                if n_skipped <= 3:
                    print(f"  ! Linha {n_rows}: erro de parsing — {e}")
                continue

            stats[area]["itens"].add(co_item)
            stats[area]["provas"].add(co_prova)

            # Item único (por co_item, ano) — só insere se ainda não existe na tabela memória
            key = (co_item, ano)
            if key not in itens_unicos:
                itens_unicos[key] = {
                    "co_item": co_item,
                    "ano": ano,
                    "area": area,
                    "gabarito": gabarito,
                    "numero_habilidade": hab_int,
                    "param_discriminacao": a_val,
                    "param_dificuldade": b_val,
                    "param_acerto_casual": c_val,
                    "tipo_lingua": tp_lingua,
                    "item_abandonado": aban,
                    "motivo_abandono": motivo,
                    "is_reaplicacao": is_reaplicacao(cor),
                }

            # Mapping item × prova × posição
            itens_prova.append({
                "co_item": co_item,
                "co_prova": co_prova,
                "ano": ano,
                "posicao": posicao,
                "cor_caderno": cor or None,
            })

            # Metadado da prova (uma linha por co_prova)
            cp_key = (ano, co_prova)
            if cp_key not in codigos_prova:
                codigos_prova[cp_key] = {
                    "ano": ano,
                    "area": area,
                    "co_prova": co_prova,
                    "categoria": "DADOS DA PROVA OBJETIVA",
                    "descricao_categoria": cor or None,
                    "is_reaplicacao": is_reaplicacao(cor),
                }

    return {
        "itens_unicos": list(itens_unicos.values()),
        "itens_prova": itens_prova,
        "codigos_prova": list(codigos_prova.values()),
        "stats": {a: {"n_itens": len(d["itens"]), "n_provas": len(d["provas"])} for a, d in stats.items()},
        "n_rows": n_rows,
        "n_skipped": n_skipped,
    }


# ════════════════════════════════════════════════════════════════════════════════
# UPLOAD PARA SUPABASE
# ════════════════════════════════════════════════════════════════════════════════

def get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("ERRO: SUPABASE_URL ou SUPABASE_KEY não definidos no ambiente")
        print("      Verifique seu .env")
        sys.exit(1)
    return create_client(url, key.strip())


def reset_year(client: Client, ano: int) -> None:
    """Apaga todos os registros de um ano específico nas 3 tabelas."""
    print(f"\n[RESET] Apagando registros do ano {ano}...")
    for table in ("enem_itens_prova", "enem_codigos_prova", "enem_itens"):
        try:
            r = client.table(table).delete().eq("ano", ano).execute()
            n = len(r.data) if r.data else 0
            print(f"  • {table}: {n} linhas removidas")
        except Exception as e:
            print(f"  ! {table}: erro ao limpar — {str(e)[:100]}")


def insert_batched(client: Client, table: str, rows: List[dict]) -> int:
    """Insere uma lista de rows em batches. Retorna total inserido."""
    if not rows:
        return 0
    total = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        try:
            r = client.table(table).insert(batch).execute()
            total += len(r.data)
            print(f"  • {table}: inseridas {total}/{len(rows)}")
        except Exception as e:
            msg = str(e)[:300]
            print(f"  ! Erro batch {i}-{i+len(batch)}: {msg}")
            # Tenta linha por linha pra não perder tudo
            for r1 in batch:
                try:
                    client.table(table).insert(r1).execute()
                    total += 1
                except Exception as e2:
                    pass
    return total


def validate_year(client: Client, ano: int) -> None:
    """Valida as contagens após o seed."""
    print(f"\n[VALIDAÇÃO] Contagens para ano {ano}:")
    for table in ("enem_itens", "enem_itens_prova", "enem_codigos_prova"):
        try:
            r = client.table(table).select("id", count="exact", head=True).eq("ano", ano).execute()
            print(f"  • {table}: {r.count} rows")
        except Exception as e:
            print(f"  ! {table}: erro — {str(e)[:120]}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Seeder do item bank ENEM no Supabase")
    parser.add_argument("--ano", type=int, required=True, help="Ano do ENEM (ex: 2024)")
    parser.add_argument("--file", type=str, required=True, help="Caminho para ITENS_PROVA_{ano}.csv")
    parser.add_argument("--reset", action="store_true", help="Apaga registros existentes do ano antes de inserir")
    parser.add_argument("--dry-run", action="store_true", help="Apenas parse o CSV sem inserir no banco")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  XTRI Engine — Seeder ENEM Item Bank")
    print(f"{'='*70}")
    print(f"  Ano: {args.ano}")
    print(f"  CSV: {args.file}")
    print(f"  Reset: {args.reset}")
    print(f"  Dry-run: {args.dry_run}")

    print(f"\n[1/3] Parsing CSV...")
    parsed = parse_itens_prova_csv(args.file, args.ano)
    print(f"  • Linhas processadas: {parsed['n_rows']}")
    if parsed["n_skipped"]:
        print(f"  • Linhas com erro: {parsed['n_skipped']}")
    print(f"  • Itens únicos: {len(parsed['itens_unicos'])}")
    print(f"  • Item-prova mappings: {len(parsed['itens_prova'])}")
    print(f"  • Códigos de prova: {len(parsed['codigos_prova'])}")
    print(f"\n  Estatísticas por área:")
    for area in sorted(parsed["stats"].keys()):
        s = parsed["stats"][area]
        print(f"    {area}: {s['n_itens']} itens únicos, {s['n_provas']} provas")

    if args.dry_run:
        print("\n[DRY-RUN] Nenhuma alteração no banco.")
        return

    client = get_client()

    if args.reset:
        reset_year(client, args.ano)

    print(f"\n[2/3] Inserindo dados...")
    n1 = insert_batched(client, "enem_itens", parsed["itens_unicos"])
    n2 = insert_batched(client, "enem_codigos_prova", parsed["codigos_prova"])
    n3 = insert_batched(client, "enem_itens_prova", parsed["itens_prova"])
    print(f"\n  Total inserido:")
    print(f"    enem_itens:         {n1}")
    print(f"    enem_codigos_prova: {n2}")
    print(f"    enem_itens_prova:   {n3}")

    print(f"\n[3/3] Validando...")
    validate_year(client, args.ano)
    print(f"\n✓ Seeder concluído com sucesso.\n")


if __name__ == "__main__":
    main()
