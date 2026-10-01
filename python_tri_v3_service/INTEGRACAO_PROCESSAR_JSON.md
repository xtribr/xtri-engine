# Integração: correção TRI de simulado via JSON

Endpoint do XTRI Engine que recebe um simulado ENEM completo, já binarizado
(0 = erro, 1 = acerto), e devolve a TRI 3PL de cada área com a nota na escala
pedagógica do ENEM do ano escolhido (padrão: 2023).

É o mesmo cálculo usado na correção do arquivo `binarizado.json` — o script
`scripts_local/tri_binarizado/tri_3pl.py` chama exatamente o mesmo código
(`app/simulado_binarizado.py`).

## 1. Subir o serviço

```bash
cd python_tri_v3_service
docker compose up --build -d          # porta 5003
curl http://localhost:5003/health     # {"status": "online", ...}
```

Sem Docker (Python ≥ 3.11):

```bash
cd python_tri_v3_service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 5003 --workers 2
```

O R/`mirt` **não** é necessário para este endpoint; o Supabase também não.
Documentação interativa (Swagger): `http://localhost:5003/docs`.

## 2. Chamada

```
POST /api/v3/processar-json
Content-Type: application/json
```

### Entrada

O mesmo formato do `binarizado.json` gerado pela plataforma, com o campo
opcional `ano_referencia`:

```json
{
  "ano_referencia": 2023,
  "total_questoes": 180,
  "areas": { "LC": [1, 45], "CH": [46, 90], "CN": [91, 135], "MT": [136, 180] },
  "alunos": [
    {
      "id": "aluno-001",
      "matricula": "000000001",
      "nome": "Nome do Aluno",
      "turma": "EM1AM",
      "fez_area": { "LC": true, "CH": true, "CN": true, "MT": true },
      "respostas": [0, 0, 1, 1, ...]
    }
  ]
}
```

| Campo | Regra |
|---|---|
| `ano_referencia` | Opcional, padrão 2023. Anos válidos: `GET /api/v3/anos-referencia`. 2023 e 2024 usam percentis completos por acertos; os demais, mín/med/máx |
| `areas` | Faixa `[início, fim]` (1-indexada, inclusiva) de cada área; códigos `LC`, `CH`, `CN`, `MT`. Padrão: o layout ENEM de 180 questões |
| `alunos[].respostas` | `0` ou `1` em todas as questões das áreas feitas. Áreas não feitas podem vir com `null` |
| `alunos[].fez_area` | `false` tira o aluno da calibração e da nota daquela área. Área ausente conta como feita |
| `id`, `matricula`, `nome`, `turma` | Opcionais; só são devolvidos. **Recomendação:** envie só `id` e cruze os dados no seu backend |
| `acertos`, `total_alunos` | Ignorados (recalculados) |

Limites: até 50 000 alunos por chamada; mínimo de 10 alunos por área para
calibrar (áreas abaixo disso ficam sem nota e geram aviso).

### Saída (resumo)

```json
{
  "status": "sucesso",
  "ano_referencia": 2023,
  "metodo": { "modelo": "...", "parametro_c": {"LC": "3PL c estimado (prior Beta 10,40)", ...}, ... },
  "total_alunos": 607,
  "total_questoes": 180,
  "areas": {
    "MT": {
      "questoes": [136, 180],
      "n_alunos": 584,
      "modelo": "3PL c estimado (prior Beta 10,40)",
      "resumo": { "acertos_medio": 18.98, "nota_media": 603.6, "nota_dp": 119.6, ... },
      "curva_condicional": { ... },
      "itens": [ { "questao": 136, "a": 1.37, "b": -0.37, "c": 0.19, "taxa_acerto": 0.67, "r_item_resto": 0.36, "saturado": false } ]
    }
  },
  "alunos": [
    {
      "id": "...", "matricula": "...", "nome": "...", "turma": "...",
      "fez_area": { ... },
      "areas": {
        "MT": { "acertos": 11, "theta": -1.17, "se_theta": 0.54, "posicao_no_grupo": 0.39, "nota_tri": 440.5, "nivel": "ELEMENTAR" }
      },
      "media_tri": 513.8
    }
  ],
  "avisos": [ "MT: questões com correlação item-resto < 0,10 (conferir gabarito): 137, 156" ],
  "tempo_processamento": 7.3
}
```

- `alunos` volta **na mesma ordem** da entrada.
- `media_tri` é `null` quando o aluno não tem nota em todas as áreas enviadas.
- `nivel`: `AVANCADO` ≥ 700, `PROFICIENTE` ≥ 600, `BASICO` ≥ 500, `ELEMENTAR` ≥ 400, `INICIANTE` < 400.
- `avisos` lista áreas não calibradas e questões suspeitas (possível erro de gabarito).

### Erros

| HTTP | Quando |
|---|---|
| 400 | Área inválida, resposta diferente de 0/1 numa área feita, vetor de respostas curto, ano indisponível, nenhuma área com ≥ 10 alunos |
| 422 | JSON fora do schema (campo obrigatório ausente, tipo errado) |
| 500 | Falha inesperada na calibração (detalhe em `detail`) |

## 3. Método

Cada área é um modelo independente (nunca se misturam áreas):

1. **Calibração 3PL** por MML/EM (61 pontos de quadratura).
   - Área com **menos de 500 alunos**: `c` fixo em 0,20 (5 alternativas).
   - Área com **500 alunos ou mais**: `c` estimado por item, com prior Beta(10, 40).
2. **Theta** por EAP com prior N(0,1).
3. **Ancoragem condicional aos acertos**: o aluno é posicionado entre quem teve
   o mesmo número de acertos, `u = Φ(w·(θ − m(k))/s(k))`, e recebe o percentil
   `u` das notas oficiais do ENEM do ano de referência para `k` acertos, com o
   piso oficial da área.

A nota é uma **projeção pedagógica** na escala do ENEM; não é nota oficial do
INEP. A calibração é refeita a cada chamada a partir dos alunos enviados —
envie a coorte inteira do simulado numa única requisição, não turma por turma.

## 4. Desempenho e operação

- ~7 s para 607 alunos × 180 questões numa máquina de desenvolvimento.
  O tempo cresce aproximadamente linear com o número de alunos.
- A rota é síncrona: configure o timeout do cliente HTTP para pelo menos 120 s.
- O endpoint roda no threadpool do servidor; com `--workers 2` atende
  chamadas paralelas sem travar `/health`.

## 5. Segurança

- Esta rota **não exige API key** (como as demais rotas V3). Não exponha a
  porta 5003 na internet: chame-a somente do seu backend, numa rede privada
  ou atrás de um proxy com autenticação.
- O serviço não grava as respostas nem os nomes em disco; tudo fica só na
  memória durante a requisição.
- Para reduzir dados pessoais em trânsito, envie apenas `id` (sem `nome` e
  `matricula`).

## 6. Exemplo em Node.js

```js
const resp = await fetch(`${process.env.PYTHON_TRI_URL}/api/v3/processar-json`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ ano_referencia: 2023, ...binarizado }),
  signal: AbortSignal.timeout(120_000),
});
if (!resp.ok) throw new Error(`TRI ${resp.status}: ${(await resp.json()).detail}`);
const tri = await resp.json();
for (const aluno of tri.alunos) {
  // aluno.id, aluno.areas.MT?.nota_tri, aluno.media_tri ...
}
```
