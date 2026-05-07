# syntax=docker/dockerfile:1.7

# ── Etapa 1: build del frontend ───────────────────────────────────────────────
FROM node:20-alpine AS frontend
WORKDIR /front
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ── Etapa 2: runtime Python ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Dependencias del sistema mínimas para faiss/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Limpia el toolchain del runtime para reducir tamaño
RUN apt-get purge -y --auto-remove build-essential

COPY backend/app /app/app
COPY backend/indexer /app/indexer

# Frontend buildado en /app/static
COPY --from=frontend /front/dist /app/static

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
