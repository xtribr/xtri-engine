# XTRI Engine API — TRI V3

API de estimação TRI para simulados ENEM com dois motores:

| Motor | Rota | Uso |
|---|---|---|
| **Heurístico** (V2 compat.) | `POST /api/calcular-tri` | Feedback em tempo real, turmas pequenas |
| **Científico** (MML + EAP) | `POST /api/v3/calibrar-cientifico` | Fechamento de simulados, banco de itens |
| **Theta Individual** | `POST /api/v3/estimar-theta` | Alunos retardatários, OMR assíncrono |
| **Calibração MIRT** | `POST /api/v5/mirt/calibracoes` | Ajuste multidimensional assíncrono por simulado × área |
| **Scoring MIRT em lote** | `POST /api/v5/mirt/estimar-theta-lote` | Correção de até 10 mil alunos com um artefato calibrado |
| **Referências** | `GET /api/v3/referencias/{area}` | Tabela oficial para frontend |

## Quick Start

```bash
# Docker
docker compose up --build

# Ou direto com Python
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 5003 --reload
```

## Endpoints

### Health Check
```bash
curl http://localhost:5003/health
```

### Motor Heurístico (compatível com V2)
```bash
curl -X POST http://localhost:5003/api/calcular-tri \
  -H "Content-Type: application/json" \
  -d '{
    "alunos": [{"id": "1", "nome": "João", "q1": "A", "q2": "B", ...}],
    "gabarito": {"1": "A", "2": "B", ...},
    "areas_config": {"LC": [1, 45]}
  }'
```

### Motor Científico
```bash
curl -X POST http://localhost:5003/api/v3/calibrar-cientifico \
  -H "Content-Type: application/json" \
  -d '{
    "simulado_id": "sim_2026_01",
    "area": "LC",
    "alunos": [
      {"aluno_id": "uuid-1", "nome": "João", "respostas": [1,0,1,1,0,...]}
    ]
  }'
```

### Swagger UI
Acesse `http://localhost:5003/docs` para documentação interativa.

## API MIRT para integração web

O fluxo público separa o ajuste pesado do scoring cotidiano:

1. Envie ao `POST /api/v5/mirt/calibracoes` uma coorte com pelo menos 100
   alunos, uma única área e a Q-matrix real. No bifator, cada item carrega o
   fator geral e exatamente um específico; cada específico precisa de pelo
   menos três itens.
2. Consulte `GET /api/v5/mirt/calibracoes/{job_id}` até `status=done`.
3. Use exatamente um entre `simulado_id` e `artefato_id` no
   `POST /api/v5/mirt/estimar-theta-lote`.

Respostas usam `0=erro`, `1=acerto` e `null=ausente`. O MIRT recebe ausentes
como `NA`; eles nunca viram erro silenciosamente. Quando um aluno tem itens
ausentes, a API devolve os scores latentes, mas omite a nota ENEM ancorada.
A nota também é omitida em provas com menos de 45 itens; isso evita tratar,
por exemplo, 4/4 como 4 acertos numa prova ENEM completa. O lote exige
`item_ids` na ordem exata da Q-matrix para impedir troca silenciosa de itens.
O endpoint individual exige a mesma ordem explícita e não faz fallback
posicional para artefatos unidimensionais; use `/api/v3/estimar-theta`
deliberadamente quando precisar do motor unidimensional.

Uma Q-matrix automática por habilidade só é identificável quando cada fator
tem indicadores suficientes. Para provas com uma ou duas questões por
habilidade, forneça uma matriz manual baseada em agrupamentos pedagógicos mais
amplos e reais (competência, domínio ou testlet); a API não inventa esses
agrupamentos.

Configure em produção:

```bash
XTRI_API_KEY='uma-chave-longa-e-aleatoria'
XTRI_CORS_ORIGINS='https://seu-site.com.br'
XTRI_APP_SUPPORT_DIR='/var/lib/xtri'
# Somente em desenvolvimento local sem chave:
# XTRI_MIRT_ALLOW_UNAUTHENTICATED='true'
# Opcional: preserve os CSVs com IDs para auditoria (padrão: apagar)
XTRI_MIRT_RETAIN_RESPONSES='false'
XTRI_MIRT_MAX_ACTIVE_JOBS='1'
XTRI_MIRT_MAX_SCORING_PROCESSES='2'
XTRI_MIRT_SCORE_TIMEOUT_SECONDS='300'
```

Nas rotas MIRT, envie a chave pelo header `X-API-Key`. Sem `XTRI_API_KEY`, elas
falham de forma segura; o modo aberto exige
`XTRI_MIRT_ALLOW_UNAUTHENTICATED=true` e deve ser usado somente no ambiente de
desenvolvimento local. A imagem Docker instala R, `jsonlite` e `mirt`, usa uma
fila de calibração por processo e persiste modelos/jobs no volume
`xtri-runtime`.

`XTRI_API_KEY` é um segredo **server-to-server**: mantenha-o no backend do seu
site (por exemplo, Node.js → FastAPI) e nunca o exponha no JavaScript ou no
navegador. CORS limita origens no browser, mas não protege a chave nem substitui
autenticação.

O `model.rds` do pacote `mirt` contém padrões de resposta desidentificados
necessários ao objeto ajustado, embora os CSVs com `aluno_id` sejam apagados
por padrão. Trate o volume `xtri-runtime` como dado sensível: acesso restrito,
criptografia no host e política de retenção. A imagem fixa `mirt` 1.46.1 e o
scoring rejeita artefatos criados por outra versão do pacote.

Exemplo de scoring em lote:

```bash
curl -X POST http://localhost:5003/api/v5/mirt/estimar-theta-lote \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: uma-chave-longa-e-aleatoria' \
  -d '{
    "simulado_id": "sim_2026_01",
    "area": "MT",
    "ano_referencia": 2024,
    "item_ids": ["mt_001", "mt_002", "mt_003", "mt_004"],
    "alunos": [
      {"aluno_id": "123", "respostas": [1, 0, 1, 1]}
    ]
  }'
```

## Migração V2 → V3

A rota `/api/calcular-tri` é um **drop-in replacement** da V2 (Flask).
Basta alterar a variável `PYTHON_TRI_URL` no backend Node.js para apontar para a V3.

## Stack

- **FastAPI** + **Pydantic v2** (validação automática)
- **NumPy** + **SciPy** (motor científico)
- **Docker** (deploy containerizado)
- Porta: **5003** (retrocompatível)
