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
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XTRI_APP_SUPPORT_DIR=/var/lib/xtri \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# Runtime R/MIRT. As dependências de compilação são removidas depois da
# instalação dos pacotes; R e as bibliotecas compiladas permanecem na imagem.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        r-base r-base-dev build-essential gfortran cmake ca-certificates \
        libcurl4-openssl-dev libssl-dev libxml2-dev libuv1-dev libgfortran5 \
    && Rscript --vanilla -e "install.packages('remotes', repos='https://cloud.r-project.org'); remotes::install_version('jsonlite', version='2.0.0', repos='https://cloud.r-project.org', upgrade='never'); remotes::install_version('mirt', version='1.46.1', repos='https://cloud.r-project.org', dependencies=NA, upgrade='never')" \
    && Rscript --vanilla -e "stopifnot(requireNamespace('jsonlite', quietly=TRUE), requireNamespace('mirt', quietly=TRUE))" \
    && apt-get purge -y r-base-dev build-essential gfortran cmake \
        libcurl4-openssl-dev libssl-dev libxml2-dev libuv1-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Dependências transitivas compiladas do MIRT ligam dinamicamente contra
# libuv (`fs`) e TBB (`RcppParallel`/`qs2`). Os pacotes -dev também mantêm
# os symlinks sem versão exigidos pelos binários R distribuídos no pacote.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libuv1-dev libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências primeiro (cache de camada Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY app/ ./app/

# Versão do motor gravada em cada execução (bloco "execucao" da resposta).
# Build: docker build --build-arg GIT_COMMIT=$(git rev-parse --short=12 HEAD) .
ARG GIT_COMMIT=desconhecida
ENV XTRI_MOTOR_VERSAO=$GIT_COMMIT

# Criar usuário não-root para segurança
RUN useradd --create-home appuser \
    && mkdir -p /var/lib/xtri \
    && chown -R appuser:appuser /var/lib/xtri
USER appuser

# Porta da API (retrocompatível com V2)
EXPOSE 5003

# Healthcheck para Docker / orquestradores
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5003/health')" || exit 1

# Um worker evita duas filas R independentes disputando CPU/registry local.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "5003", \
     "--workers", "1", \
     "--timeout-keep-alive", "60", \
     "--log-level", "info"]
