# Aegis — full-stack image: churn API + statically exported dashboard.
#
# Stage 1 compiles the Next.js dashboard to plain HTML/JS; stage 2 serves it
# and the FastAPI app from a single uvicorn process (see the static mount at
# the bottom of api/main.py).

# ── Stage 1: dashboard static export ────────────────────────────────────────
FROM node:20-alpine AS dashboard

WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
# Baked in at build time: "" = same origin (the API serves the dashboard)
ENV NEXT_PUBLIC_API_URL=""
RUN npm run build

# ── Stage 2: API + static serving ────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# libgomp1 provides libgomp.so.1, the OpenMP runtime that the XGBoost (and
# LightGBM) native libraries load at import time. python:*-slim does not ship
# it, so install it explicitly rather than relying on the wheel to vendor a copy.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install lean serving dependencies first for layer caching
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Application package and trained model artifacts
COPY src/ src/
COPY models/ models/

# Dashboard static export (served by FastAPI at /)
COPY --from=dashboard /web/out web/out

# The package lives under src/, so it must be importable
ENV PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "churn_prediction.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
