# Aegis — full-stack demo image for Hugging Face Spaces.
# Runs the FastAPI prediction API (localhost:8000) AND the Streamlit dashboard
# (public port 7860) in a single container via start.sh.
#
# NOTE: This is the Space/demo image and lives only on the `huggingface`
# branch. The lean API-only image used by CI lives on `main`.
FROM python:3.12-slim

WORKDIR /app

# libgomp1: OpenMP runtime required by XGBoost. curl: used by the healthcheck loop.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Full dependencies (dashboard + API + model libraries)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY start.sh .

RUN chmod +x start.sh
# The package lives under src/; the dashboard's home dir must be writable
# (Hugging Face may run the container as a non-root user).
ENV PYTHONPATH=/app/src
ENV HOME=/tmp
ENV API_URL=http://127.0.0.1:8000
ENV API_KEY=demo_key_public

EXPOSE 7860

CMD ["./start.sh"]
