# Aegis — Churn Intelligence for Subscription Businesses

[![CI](https://github.com/Lucas-Maingi/aegis-churn-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/Lucas-Maingi/aegis-churn-analytics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Next.js](https://img.shields.io/badge/next.js-16-black)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)

Aegis is a **multi-tenant churn-prevention SaaS** aimed at small ISPs, WISPs, and telecom operators — businesses with Telco-shaped customer data and no data science team. An operator signs up, uploads a CSV export from their billing system, and gets every customer scored for churn risk with **plain-English explanations of why** — then wins at-risk customers back with **one-click retention offers**.

**The workflow:** sign up → import your customer CSV (columns auto-matched) → see a ranked list of at-risk customers and revenue at risk → open any customer to read the top-3 churn drivers → click once to send a loyalty discount, annual-plan offer, or personal check-in.

It began as a single-tenant model-serving project and evolved into a product: organizations with isolated data, JWT dashboard sessions, CSV ingestion with column mapping and value normalization, vectorized batch scoring with SHAP explanations, and an outreach engine with real (Resend) or simulated email delivery.

> **Scope & honesty note:** The model is trained on the public [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (~7,000 customers). It transfers to telecom/ISP-shaped businesses because the churn drivers (contract type, tenure, price, payment method) are the same; it is **not** a general-purpose churn model for arbitrary SaaS. Metrics below are real, measured on a held-out test split. See [Known Limitations](#-known-limitations).

---

## 🎥 Live Demo

**▶️ Try it live:** https://huggingface.co/spaces/lucas-maingi/aegis-churn-analytics

The live demo runs the full SaaS — the Next.js dashboard and the FastAPI scoring engine — in a single always-on container. Click **"Use demo account"** on the login page (`demo@aegis.app` / `aegis-demo-2026`) to land on a demo ISP with 60 customers already scored, or create your own organization and import a CSV. Demo storage is ephemeral and resets on Space restarts.

![Aegis churn dashboard](docs/demo.gif)

> _Recording the GIF? See [`docs/HOW_TO_RECORD_DEMO.md`](docs/HOW_TO_RECORD_DEMO.md)._

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
        +-----------------------------+
        |   Next.js Dashboard (web/)  |
        |  signup · CSV import wizard |
        |  ranked risk list · drawer  |
        |  one-click outreach         |
        +--------------+--------------+
                       | HTTP + Bearer JWT
                       v
        +-----------------------------+        +---------------------+
        |         FastAPI API         |        |  SQL database       |
        |  /api/v1/auth/*             |------->|  orgs · users       |
        |  /api/v1/customers/*        |        |  customers + scores |
        |  /api/v1/outreach/*         |        |  outreach log       |
        |  /predict (legacy, API key) |        |  (SQLite locally,   |
        +--------------+--------------+        |   Postgres in prod) |
                       |                       +---------------------+
     +-----------------+------------------+
     |                 |                  |            +-----------+
  Preprocessor    XGBoost model     SHAP explainer --->|  Resend   |
  (column mapper  (joblib           (top-3 drivers,    |  (email   |
   + pipeline)     artifact)         plain English)    |  offers)  |
                                                       +-----------+
```

### Engineering features that make it "product", not "notebook"
- **Multi-tenancy** — organizations with row-level data isolation; every query is scoped to the authenticated org, with tests proving one tenant can never read or message another tenant's customers.
- **JWT auth** — PBKDF2-hashed passwords, 7-day signed sessions; the legacy `/predict` routes keep `X-API-Key` auth.
- **CSV ingestion that meets operators where they are** — arbitrary billing-export headers are auto-matched to the model schema, and messy real-world values are normalized (`fiber` → `Fiber optic`, `mpesa` → `Electronic check`, `1 year` → `One year`). Re-uploads update rather than duplicate.
- **Vectorized batch scoring** — one preprocessor/model/SHAP pass for the whole upload; probability, risk tier, and top-3 drivers (with raw, human-readable values) persist per customer.
- **One-click outreach** — three retention templates rendered per customer; delivered via Resend when configured, recorded as `simulated` otherwise so the full workflow demos without credentials.
- **Strict request validation** via Pydantic schemas — invalid categories or negative tenure are rejected with `422` and a structured error body.
- **Sliding-window rate limiting** (60 req/min/IP) implemented as ASGI middleware.
- **Lifespan artifact loading** — model, preprocessor, feature names, and metadata are loaded once at startup and fail fast if any artifact is missing.
- **Reproducibility** — training pins a `dataset_hash` and `random_state`, and logs runs to MLflow.

---

## 🚀 Quickstart

### Local (full SaaS: API + dashboard)
```bash
pip install -r requirements.txt

# 1. Run the API (package lives under src/)
PYTHONPATH=src uvicorn churn_prediction.api.main:app --reload --port 8000

# 2. In another terminal, run the Next.js dashboard
cd web && npm install && npm run dev
```
Open `http://localhost:3000`, create an organization, and import
[`docs/sample_customers.csv`](docs/sample_customers.csv) — a realistic ISP
billing export with messy headers — to see the full flow. Interactive API
docs: `http://localhost:8000/docs`.

Optional environment (see `.env.example`): `DATABASE_URL` for Postgres,
`JWT_SECRET` for sessions, `RESEND_API_KEY` to send real retention emails
(unset = simulated sends, fully functional demo).

### Legacy single-prediction stack (Streamlit)
```bash
API_URL=http://127.0.0.1:8000 streamlit run src/churn_prediction/dashboard/app.py
```

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
The suite covers the preprocessing pipeline, the prediction API surface (auth, schema validation, single/batch prediction, rate limiting), and the SaaS layer — signup/login, tenant isolation, CSV upload with column mapping, ranked listing, and one-click outreach. CI runs lint + tests + a Docker health-check smoke test on every push.

---

## ⚠️ Known Limitations

Being explicit about the gap between this and a real production deployment:

- **Single public dataset.** Telco is clean, static, and balanced-ish. A real carrier's data is messier and drifts — the honest path is scoring with this base model first, then fine-tuning per tenant once a few months of labeled outcomes accumulate. There is no per-tenant fine-tuning loop yet.
- **Telecom-shaped businesses only.** The feature schema (contract, tenure, service type, charges) transfers across ISPs/telcos, not to e-commerce or general SaaS.
- **No billing/subscriptions.** Tenants are free; there is no Stripe metering of the platform itself yet.
- **Rate limiting is in-process.** The sliding window is per-process and would need Redis (or similar) to work correctly behind multiple replicas.
- **Explanation cost.** SHAP is computed per upload batch (not per request), but very large tenants would want a cached/approximate explainer.

---

## 🗂️ Project Structure

```
src/churn_prediction/
├── api/            # FastAPI app, legacy predict routes, middleware
├── saas/           # multi-tenant layer: auth, orgs, CSV mapping,
│                   # batch scoring, outreach (Resend/simulated)
├── data/           # dataset loader + preprocessing pipeline
├── models/         # trainer + SHAP explainer
├── dashboard/      # legacy Streamlit operator UI
├── utils/          # metrics helpers
└── config.py       # centralized paths, features, thresholds
web/                # Next.js dashboard (signup, import wizard,
                    # ranked risk list, one-click outreach)
tests/              # preprocessing + API + SaaS-layer tests
notebooks/          # EDA and modelling narrative
docs/               # demo assets + sample ISP billing CSV
```

## 📄 License

MIT.
