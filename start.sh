#!/usr/bin/env bash
# Full-stack entrypoint for the Hugging Face Space:
# starts the FastAPI prediction API in the background, waits for it to become
# healthy, then launches the Streamlit dashboard on the HF Spaces port.
set -e

export PYTHONPATH=/app/src

python -m uvicorn churn_prediction.api.main:app --host 127.0.0.1 --port 8000 &

echo "Waiting for prediction API to become healthy..."
for i in $(seq 1 40); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "Prediction API ready."
    break
  fi
  sleep 1
done

exec streamlit run src/churn_prediction/dashboard/app.py \
  --server.port=7860 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
