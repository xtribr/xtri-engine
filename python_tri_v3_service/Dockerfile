# ═══════════════════════════════════════════════════════════════════════════════
# XTRI Engine API — TRI V3 (FastAPI)
# ═══════════════════════════════════════════════════════════════════════════════
# Build: docker build -t xtri-tri-v3 .
# Run:   docker run -p 5003:5003 xtri-tri-v3
# ═══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim AS base

# Variáveis de ambiente para otimizar Python em container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Instalar dependências primeiro (cache de camada Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY app/ ./app/

# Criar usuário não-root para segurança
RUN useradd --create-home appuser
USER appuser

# Porta da API (retrocompatível com V2)
EXPOSE 5003

# Healthcheck para Docker / orquestradores
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5003/health')" || exit 1

# Iniciar com Uvicorn (2 workers, timeout 60s)
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "5003", \
     "--workers", "2", \
     "--timeout-keep-alive", "60", \
     "--log-level", "info"]
