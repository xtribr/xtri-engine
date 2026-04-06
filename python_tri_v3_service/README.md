# XTRI Engine API — TRI V3

API de estimação TRI para simulados ENEM com dois motores:

| Motor | Rota | Uso |
|---|---|---|
| **Heurístico** (V2 compat.) | `POST /api/calcular-tri` | Feedback em tempo real, turmas pequenas |
| **Científico** (MML + EAP) | `POST /api/v3/calibrar-cientifico` | Fechamento de simulados, banco de itens |
| **Theta Individual** | `POST /api/v3/estimar-theta` | Alunos retardatários, OMR assíncrono |
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

## Migração V2 → V3

A rota `/api/calcular-tri` é um **drop-in replacement** da V2 (Flask).
Basta alterar a variável `PYTHON_TRI_URL` no backend Node.js para apontar para a V3.

## Stack

- **FastAPI** + **Pydantic v2** (validação automática)
- **NumPy** + **SciPy** (motor científico)
- **Docker** (deploy containerizado)
- Porta: **5003** (retrocompatível)
