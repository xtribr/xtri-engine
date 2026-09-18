"""CLI de correção das provas Marista.

Uso:

    python -m app.scripts.corrigir_marista \
        --bloco EM --nivel "1a serie" \
        --respostas caminho/marcacoes.csv \
        --saida-alunos notas.csv --saida-itens itens.csv

O arquivo de marcações tem uma linha por estudante, colunas de identificação
(aluno_id, nome, turma, matrícula) e uma coluna por questão (1..N, q1.., etc.).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from app.marista.correcao import ResultadoTurma, corrigir
from app.marista.estrutura import obter_prova
from app.marista.referencia import ParametrosEnvelope
from app.marista.respostas import ler_respostas


def _escrever_alunos(resultado: ResultadoTurma, destino: Path) -> None:
    areas = sorted({a for aluno in resultado.alunos for a in aluno.por_area})
    campos = ["bloco", "nivel", "aluno", "turma", "confiavel"]
    for area in areas:
        campos += [f"{area}_acertos", f"{area}_min", f"{area}_med", f"{area}_max", f"{area}_obs"]
    campos += ["GERAL_acertos", "GERAL_min", "GERAL_med", "GERAL_max", "GERAL_obs"]

    with destino.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=campos)
        escritor.writeheader()
        for aluno in resultado.alunos:
            linha = {
                "bloco": aluno.bloco,
                "nivel": aluno.nivel,
                "aluno": aluno.rotulo,
                "turma": aluno.identificacao.get("turma", ""),
                "confiavel": "sim" if aluno.confiavel else "nao",
            }
            for area in areas:
                r = aluno.por_area[area]
                linha[f"{area}_acertos"] = r.acertos
                linha[f"{area}_min"] = r.nota_minima
                linha[f"{area}_med"] = r.nota_mediana
                linha[f"{area}_max"] = r.nota_maxima
                linha[f"{area}_obs"] = r.observacao
            g = aluno.geral
            linha.update(
                {
                    "GERAL_acertos": g.acertos,
                    "GERAL_min": g.nota_minima,
                    "GERAL_med": g.nota_mediana,
                    "GERAL_max": g.nota_maxima,
                    "GERAL_obs": g.observacao,
                }
            )
            escritor.writerow(linha)


def _escrever_itens(resultado: ResultadoTurma, destino: Path) -> None:
    with destino.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(
            ["questao", "componente", "area", "gabarito", "respondentes", "acertos", "taxa_acerto", "discriminacao", "observacao"]
        )
        for i in resultado.itens:
            escritor.writerow(
                [i.questao, i.componente, i.area, i.gabarito, i.respondentes, i.acertos, i.taxa_acerto,
                 "" if i.discriminacao is None else i.discriminacao, i.observacao]
            )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Corrige provas Marista por área, com ancoragem provisória.")
    p.add_argument("--bloco", required=True, help="EF ou EM")
    p.add_argument("--nivel", required=True, help="ex.: '6o ano', '1a serie'")
    p.add_argument("--respostas", required=True, type=Path)
    p.add_argument("--saida-alunos", type=Path, default=Path("notas_alunos.csv"))
    p.add_argument("--saida-itens", type=Path, default=Path("estatistica_itens.csv"))
    p.add_argument("--limite-inferior", type=float, default=200.0)
    p.add_argument("--limite-superior", type=float, default=800.0)
    p.add_argument("--dispersao-base", type=float, default=85.0)
    p.add_argument("--deslocamento", type=float, default=0.0)
    args = p.parse_args(argv)

    prova = obter_prova(args.bloco, args.nivel)
    respostas = ler_respostas(args.respostas, prova)
    params = ParametrosEnvelope(
        limite_inferior=args.limite_inferior,
        limite_superior=args.limite_superior,
        dispersao_base=args.dispersao_base,
        deslocamento=args.deslocamento,
    )
    resultado = corrigir(respostas, prova, params)

    _escrever_alunos(resultado, args.saida_alunos)
    _escrever_itens(resultado, args.saida_itens)

    print(f"{prova.bloco}/{prova.nivel}: {resultado.n_alunos} estudantes, {prova.n_itens} itens")
    print(f"composição por área: {prova.composicao} | alternativas: {prova.alternativas} (c = {prova.chute_c:.2f})")
    print(f"notas por aluno  -> {args.saida_alunos}")
    print(f"estatística item -> {args.saida_itens}")
    for aviso in resultado.avisos:
        print(f"AVISO: {aviso}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
