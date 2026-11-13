# ---------------------------------------------------------------- build stage
# Dependencies are installed into a venv that is copied alone into the runtime
# image: no compilers, pip caches or build files in the final layer.
FROM python:3.14-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install -r requirements.txt

# ---------------------------------------------------------------- runtime stage
FROM python:3.14-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY app ./app
# Self-contained corpus with its full lineage: raw (as shipped by Syntax),
# cleaned text, chunks and the manifest. The embeddings are computed at
# start-up with the injected OPENAI_EMBEDDING_MODEL (no key at build time).
COPY data/raw ./data/raw
COPY data/cleaned ./data/cleaned
COPY data/chunks ./data/chunks
COPY data/manifest.json ./data/manifest.json
RUN mkdir -p data/cache data/traces && chown -R app:app /app/data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health').status != 200)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
