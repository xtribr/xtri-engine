# TRI 3PL de simulado binarizado

Versão de linha de comando do endpoint `POST /api/v3/processar-json`
(guia de integração: `python_tri_v3_service/INTEGRACAO_PROCESSAR_JSON.md`).

Correção TRI de um simulado ENEM (180 questões) exportado como JSON binarizado,
na escala pedagógica do ENEM 2023.

## Uso

```bash
cd python_tri_v3_service
.venv313/bin/python scripts_local/tri_binarizado/tri_3pl.py ~/Downloads/binarizado.json ../outputs/binarizado_2023
```

`--ano 2024` troca a escala de referência (só anos com percentis por acertos: 2023 e 2024).

## Entrada

```json
{
  "total_questoes": 180,
  "areas": { "LC": [1, 45], "CH": [46, 90], "CN": [91, 135], "MT": [136, 180] },
  "alunos": [
    { "id": "...", "matricula": "...", "nome": "...", "turma": "EM1AM",
      "fez_area": { "LC": true, "CH": true, "CN": true, "MT": true },
      "respostas": [0, 1, ...] }
  ]
}
```

## Método

1. Calibração 3PL por área (`app/simulado_binarizado.py`, o mesmo código do
   endpoint `POST /api/v3/processar-json`): c fixo em 0.20; com ≥ 500 alunos
   na área, c estimado por item com prior Beta(10, 40).
2. Theta EAP com prior N(0,1).
3. Posição no grupo de mesmo acerto: `u = Φ(w·(θ − m(k))/s(k))`, com `m(k)`, `s(k)`
   simulados dos parâmetros dos itens (`app/irt/conditional.py`).
4. Nota = quantil `u` das notas oficiais do ENEM 2023 com `k` acertos
   (`app/referencias/tri_quantis_referencia.json`), com o piso oficial da área.

Não é nota oficial do INEP. LC é calibrada sem separar idioma (a entrada não o informa).

## Saídas

| Arquivo | Conteúdo |
|---|---|
| `TRI_3PL.json` | método, itens (a, b, c), curva condicional, notas por aluno e área |
| `TRI_3PL.xlsx` | abas Notas, Por turma, Resumo, Itens |
| `TRI_3PL.csv` | aba Notas |

As saídas contêm nomes e matrículas: mantenha-as em `outputs/`, fora do git.
