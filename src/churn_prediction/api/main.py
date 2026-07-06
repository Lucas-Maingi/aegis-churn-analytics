"""
FastAPI REST API Server
=======================
Primary application entrypoint defining the FastAPI instance, routes,
lifespan loading of model artifacts, and request handling logic.
"""

import json
import logging
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Security, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from churn_prediction import config
from churn_prediction.api.auth import verify_api_key
from churn_prediction.api.database import get_supabase_client, log_prediction, log_prediction_batch
from churn_prediction.api.middleware import (
    RateLimitMiddleware,
    custom_validation_exception_handler,
    global_exception_handler,
)
from churn_prediction.api.schemas import (
    BatchPredictionResponse,
    CustomerBatchInput,
    CustomerInput,
    ExplanationItem,
    PredictionResponse,
)
from churn_prediction.config import get_risk_tier
from churn_prediction.models.explainer import ChurnExplainer
from churn_prediction.saas.db import init_db
from churn_prediction.saas.routes_auth import router as auth_router
from churn_prediction.saas.routes_customers import router as customers_router
from churn_prediction.saas.routes_outreach import router as outreach_router

# Set up logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to load model artifacts on startup and clear them on shutdown."""
    logger.info("Initializing API application lifespan...")

    model_path = config.MODELS_DIR / "model.joblib"
    preprocessor_path = config.MODELS_DIR / "preprocessor.joblib"
    feature_names_path = config.MODELS_DIR / "feature_names.json"
    metadata_path = config.MODELS_DIR / "model_metadata.json"

    # Verify all serialized files exist before booting up the server
    missing_files = []
    for name, path in [
        ("model", model_path),
        ("preprocessor", preprocessor_path),
        ("feature_names", feature_names_path),
        ("metadata", metadata_path),
    ]:
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        err_msg = f"Missing model artifacts for initialization: {', '.join(missing_files)}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    # Load artifacts into memory
    app.state.model = joblib.load(model_path)
    app.state.preprocessor = joblib.load(preprocessor_path)

    with open(feature_names_path, "r", encoding="utf-8") as f:
        app.state.feature_names = json.load(f)

    with open(metadata_path, "r", encoding="utf-8") as f:
        app.state.metadata = json.load(f)

    logger.info(
        "Successfully loaded all model artifacts. Winning model type: %s",
        type(app.state.model).__name__,
    )

    # Create SaaS tables (organizations, users, customers, outreach) if absent
    init_db()

    yield

    # Clean up states on shutdown
    app.state.model = None
    app.state.preprocessor = None
    app.state.feature_names = None
    app.state.metadata = None
    logger.info("API application lifespan teardown complete.")


app = FastAPI(
    title="Churn Prediction API",
    description="Commercial-grade SaaS REST API predicting user churn probabilities and explanations.",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin frontend dashboard connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom sliding-window Rate Limiting middleware (60 requests per minute per client IP)
app.add_middleware(RateLimitMiddleware, requests_limit=60, window_seconds=60)

# Native exception handlers mapping custom and general errors to standard JSON schemas
app.add_exception_handler(RequestValidationError, custom_validation_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

# Multi-tenant SaaS routes (dashboard auth, customer ingestion, outreach)
app.include_router(auth_router)
app.include_router(customers_router)
app.include_router(outreach_router)


# ── API Routes ───────────────────────────────────────────────────────────────


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Retrieve system health, api version, and loaded model metadata."""
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    db_connected = get_supabase_client() is not None
    return {
        "status": "healthy",
        "version": "0.1.0",
        "model_loaded": model_loaded,
        "db_connected": db_connected,
        "metadata": getattr(app.state, "metadata", None),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Security(verify_api_key)],
)
def predict_single(customer: CustomerInput, background_tasks: BackgroundTasks):
    """Predict churn probability and explain risk drivers for a single customer."""
    model = app.state.model
    preprocessor = app.state.preprocessor
    feature_names = app.state.feature_names

    # Convert request schema to raw Pandas DataFrame matching expected data loader columns
    raw_df = pd.DataFrame([customer.to_pandas_dict()])

    # Transform using preprocessor
    X_transformed = preprocessor.transform(raw_df)

    # Scoring (0.0 to 1.0 probability)
    prob = float(model.predict_proba(X_transformed)[:, 1][0])
    prob_pct = round(prob * 100.0, 2)
    risk_tier = get_risk_tier(prob_pct)

    # Compute SHAP explanation drivers
    explainer = ChurnExplainer(model, feature_names)
    shap_vals = explainer.compute_shap_values(X_transformed)[0]

    # Find top 3 driver indices by absolute SHAP values
    top_3_indices = np.argsort(np.abs(shap_vals))[::-1][:3]
    flat_features = X_transformed[0]

    explanations = []
    for idx in top_3_indices:
        feat = feature_names[idx]
        sv = float(shap_vals[idx])
        fv = flat_features[idx]
        direction = "increases" if sv > 0 else "decreases"
        plain = explainer._render_explanation(feat, fv)

        explanations.append(
            ExplanationItem(
                feature_name=feat,
                shap_value=sv,
                feature_value=fv,
                direction=f"{direction} churn risk",
                plain_english=plain,
            )
        )

    # Log to Supabase in the background
    background_tasks.add_task(
        log_prediction,
        customer_id=customer.customerID,
        churn_probability=prob_pct,
        risk_tier=risk_tier
    )

    return PredictionResponse(
        customerID=customer.customerID,
        churn_probability=prob_pct,
        risk_tier=risk_tier,
        explanations=explanations,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Security(verify_api_key)],
)
def predict_batch(batch: CustomerBatchInput, background_tasks: BackgroundTasks):
    """Process and score a list of customer records concurrently in a single vectorized pass."""
    model = app.state.model
    preprocessor = app.state.preprocessor
    feature_names = app.state.feature_names

    # Build a combined raw DataFrame from the batch inputs
    raw_df = pd.DataFrame([cust.to_pandas_dict() for cust in batch.customers])

    # Batch transform and inference
    X_transformed = preprocessor.transform(raw_df)
    probs = model.predict_proba(X_transformed)[:, 1]

    # Pre-compute SHAP values for the entire batch in one vectorized pass
    explainer = ChurnExplainer(model, feature_names)
    shap_values = explainer.compute_shap_values(X_transformed)

    predictions = []
    db_log_batch = []
    for i, customer in enumerate(batch.customers):
        prob_pct = round(float(probs[i] * 100.0), 2)
        risk_tier = get_risk_tier(prob_pct)

        row_shap = shap_values[i]
        row_features = X_transformed[i]

        # Extract top 3 drivers for this row
        top_3_indices = np.argsort(np.abs(row_shap))[::-1][:3]
        explanations = []
        for idx in top_3_indices:
            feat = feature_names[idx]
            sv = float(row_shap[idx])
            fv = row_features[idx]
            direction = "increases" if sv > 0 else "decreases"
            plain = explainer._render_explanation(feat, fv)

            explanations.append(
                ExplanationItem(
                    feature_name=feat,
                    shap_value=sv,
                    feature_value=fv,
                    direction=f"{direction} churn risk",
                    plain_english=plain,
                )
            )

        predictions.append(
            PredictionResponse(
                customerID=customer.customerID,
                churn_probability=prob_pct,
                risk_tier=risk_tier,
                explanations=explanations,
            )
        )
        
        db_log_batch.append({
            "customer_id": customer.customerID if customer.customerID else "UNKNOWN",
            "churn_probability": prob_pct,
            "risk_tier": risk_tier
        })

    # Log entire batch to Supabase in the background
    background_tasks.add_task(log_prediction_batch, db_log_batch)

    return BatchPredictionResponse(predictions=predictions)


# ── Dashboard static export ──────────────────────────────────────────────────
# In production the Next.js dashboard is statically exported to web/out and
# served from this same process (single-container deploy). Starlette matches
# routes in registration order, so this catch-all mount MUST be registered
# after every API route above or it shadows them. Skipped in dev, where the
# dashboard runs on its own server.

_WEB_OUT = config.PROJECT_ROOT / "web" / "out"
if _WEB_OUT.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_WEB_OUT, html=True), name="dashboard")
    logger.info("Serving dashboard static export from %s", _WEB_OUT)
