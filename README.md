# Aegis — SaaS Customer Churn Prediction API

[![CI](https://github.com/Lucas-Maingi/aegis-churn-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/Lucas-Maingi/aegis-churn-analytics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)

Aegis is a production-shaped churn prediction service: a **FastAPI** REST API that scores a customer's probability of churning and returns **SHAP-based, plain-English explanations** of *why*, plus a Streamlit dashboard for non-technical operators. It is built around the model-serving concerns that separate a notebook from a product — API-key auth, request validation, rate limiting, background persistence, and reproducible artifacts.

> **Scope & honesty note:** This is a portfolio project trained on the public [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (~7,000 customers). Metrics below are real, measured on a held-out test split — not inflated. See [Known Limitations](#-known-limitations).

---

## 🎯 The Business Problem

Acquiring a new SaaS customer costs far more than retaining an existing one, and churn is often only noticed *after* the cancellation. The value of a churn model is not the accuracy number — it is **giving a retention team a ranked list of at-risk accounts early enough to act, with a reason attached to each one.** Aegis is designed around that workflow: every prediction ships with the top-3 drivers so a success manager knows whether to offer a discount, fix a support issue, or push an annual contract.

---

## 📊 Model Performance (IBM Telco, held-out test split)

The winning model is a tuned **XGBoost classifier** selected via randomized search with 5-fold cross-validation, benchmarked against LightGBM and logistic-regression baselines and tracked in MLflow.

| Metric | Score | Why it matters here |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.847** | Ranking quality — how well the model separates churners from non-churners. |
| **Recall (churn class)** | **0.807** | Of customers who actually churn, ~81% are caught. Missing a churner is the expensive error. |
| **Precision (churn class)** | **0.516** | Of those flagged, ~52% truly churn — an intentional trade to prioritize recall. |
| **F1 (churn class)** | **0.630** | Balance of the two. |

**Why recall is deliberately favored over precision:** the cost of a *missed* churner (lost lifetime revenue) is much higher than the cost of a *false alarm* (a retention email to a happy customer). The decision threshold is tuned accordingly rather than left at the default 0.5. These are honest numbers for Telco — ~0.84 AUC is close to the practical ceiling widely reported on this dataset.

---

## 🏗️ Architecture

```
                +---------------------------+
                |   Streamlit Dashboard     |
                |  (retention-team UI)      |
                +-------------+-------------+
                              | HTTP + X-API-Key
                              v
                +---------------------------+       +------------------+
                |        FastAPI API        |       |  Supabase        |
                |  /predict  /predict/batch |------>|  (prediction log |
                |  /health                  |  bg   |   for monitoring)|
                +-------------+-------------+       +------------------+
                              |
          +-------------------+-------------------+
          |                   |                   |
   Preprocessor         XGBoost model       SHAP explainer
   (fitted pipeline)    (joblib artifact)   (top-3 drivers)
```

### Engineering features that make it "product", not "notebook"
- **API-key authentication** (`X-API-Key`) guarding all prediction routes.
- **Strict request validation** via Pydantic schemas — invalid categories or negative tenure are rejected with `422` and a structured error body.
- **Sliding-window rate limiting** (60 req/min/IP) implemented as ASGI middleware.
- **Non-blocking persistence** — predictions are logged to Supabase via FastAPI `BackgroundTasks` so logging never adds latency to a response.
- **Lifespan artifact loading** — model, preprocessor, feature names, and metadata are loaded once at startup and fail fast if any artifact is missing.
- **Reproducibility** — training pins a `dataset_hash` and `random_state`, and logs runs to MLflow.

---

## 🚀 Quickstart

### Local (Python)
```bash
pip install -r requirements.txt

# Run the API (package lives under src/)
PYTHONPATH=src uvicorn churn_prediction.api.main:app --reload --port 8000

# In another terminal, run the dashboard
API_URL=http://127.0.0.1:8000 streamlit run src/churn_prediction/dashboard/app.py
```
Interactive API docs: `http://localhost:8000/docs`.

### Docker
```bash
docker build -t aegis-churn-api .
docker run -p 8000:8000 -e API_KEY=your-key aegis-churn-api
```

### Example request
```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: test_api_key_1234" \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "9999-EXAMPLE",
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 89.5, "TotalCharges": 268.5
  }'
```
Returns a churn probability, a `LOW`/`MEDIUM`/`HIGH` risk tier, and the top-3 SHAP drivers in plain English.

---

## 🧪 Tests

```bash
PYTHONPATH=src pytest --cov=churn_prediction
```
The suite covers the preprocessing pipeline and the full API surface — auth (missing/invalid key), schema validation, single and batch prediction, and rate-limit enforcement. CI runs lint + tests + a Docker health-check smoke test on every push.

---

## ⚠️ Known Limitations

Being explicit about the gap between this and a real production deployment:

- **Single public dataset.** Telco is clean, static, and balanced-ish. Real SaaS churn data is messier, drifts over time, and needs its own re-benchmarking — expect lower and moving numbers.
- **No live retraining loop.** Retraining is a manual pipeline run; there is no automated drift detection or scheduled retraining yet.
- **Rate limiting is in-process.** The sliding window is per-process and would need Redis (or similar) to work correctly behind multiple replicas.
- **Explanation cost.** SHAP is computed per request; at high throughput this would move to a cached/approximate explainer.

---

## 🗂️ Project Structure

```
src/churn_prediction/
├── api/            # FastAPI app, auth, middleware, schemas, DB logging
├── data/           # dataset loader + preprocessing pipeline
├── models/         # trainer + SHAP explainer
├── dashboard/      # Streamlit operator UI
├── utils/          # metrics helpers
└── config.py       # centralized paths, features, thresholds
tests/              # API + preprocessing tests
notebooks/          # EDA and modelling narrative
```

## 📄 License

MIT.
