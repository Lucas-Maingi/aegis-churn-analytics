"""
Feedback-Loop Integration Tests
==============================
Covers ground-truth outcome recording, the model scorecard, retrain gating,
and tenant isolation of outcomes.
"""

import io
import json
import random

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from churn_prediction.api.main import app
from churn_prediction.api.middleware import InMemoryRateLimiter
from churn_prediction.saas import db as saas_db
from churn_prediction.saas.db import Base
from churn_prediction.saas.retrain import MIN_OUTCOMES_FOR_RETRAIN


@pytest.fixture(name="client", scope="module")
def client_fixture():
    mp = pytest.MonkeyPatch()
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    mp.setattr(saas_db, "SessionLocal", TestSession)
    mp.setattr(InMemoryRateLimiter, "check_limit", lambda self, key: True)
    with TestClient(app) as c:
        yield c
    mp.undo()


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _signup(client: TestClient, email: str) -> str:
    res = client.post(
        "/api/v1/auth/signup",
        json={"organization_name": f"Org {email}", "email": email, "password": "password123"},
    )
    assert res.status_code == 201
    return res.json()["access_token"]


def _make_csv(n: int) -> str:
    """Build a CSV of n customers with a learnable tenure/plan churn signal."""
    rng = random.Random(3)
    lines = ["subscriber_id,months_active,monthly_fee,plan_type,contract_type,payment_method"]
    for i in range(1, n + 1):
        plan = rng.choice(["fiber", "dsl"])
        tenure = rng.choice([1, 3, 6, 12, 24, 48, 60])
        fee = round(rng.uniform(25, 90), 2)
        contract = rng.choice(["monthly", "1 year", "2 year"])
        lines.append(f"SUB-{i:03d},{tenure},{fee},{plan},{contract},mpesa")
    return "\n".join(lines) + "\n"


_MAPPING = {
    "external_id": "subscriber_id",
    "tenure": "months_active",
    "MonthlyCharges": "monthly_fee",
    "InternetService": "plan_type",
    "Contract": "contract_type",
    "PaymentMethod": "payment_method",
}


def _upload(client: TestClient, token: str, csv_text: str):
    return client.post(
        "/api/v1/customers/upload",
        headers=_headers(token),
        files={"file": ("c.csv", io.BytesIO(csv_text.encode()), "text/csv")},
        data={"mapping": json.dumps(_MAPPING)},
    )


@pytest.fixture(name="token", scope="module")
def token_fixture(client):
    token = _signup(client, "loop@example.com")
    assert _upload(client, token, _make_csv(80)).status_code == 200
    return token


def _all_customers(client, token):
    return client.get(
        "/api/v1/customers?page_size=100", headers=_headers(token)
    ).json()["items"]


# ── Outcome recording ────────────────────────────────────────────────────────


def test_record_outcome_and_reflect_in_detail(client, token):
    cust = _all_customers(client, token)[0]
    res = client.post(
        "/api/v1/model/outcome",
        headers=_headers(token),
        json={"customer_id": cust["id"], "outcome": "churned"},
    )
    assert res.status_code == 200
    detail = client.get(
        f"/api/v1/customers/{cust['id']}", headers=_headers(token)
    ).json()
    assert detail["actual_outcome"] == "churned"


def test_invalid_outcome_rejected(client, token):
    cust = _all_customers(client, token)[0]
    res = client.post(
        "/api/v1/model/outcome",
        headers=_headers(token),
        json={"customer_id": cust["id"], "outcome": "maybe"},
    )
    assert res.status_code == 422


def test_outcome_requires_auth(client):
    assert client.post("/api/v1/model/outcome", json={}).status_code == 401


# ── Scorecard + retrain gating ───────────────────────────────────────────────


def test_scorecard_and_retrain_gate_before_enough_outcomes(client):
    """A fresh org with no outcomes cannot retrain and has an empty scorecard."""
    token = _signup(client, "fresh@example.com")
    _upload(client, token, _make_csv(50))

    card = client.get("/api/v1/model/scorecard", headers=_headers(token)).json()
    assert card["active_model"] == "base"
    assert card["n_outcomes"] == 0
    assert card["can_retrain"] is False
    assert card["min_outcomes_for_retrain"] == MIN_OUTCOMES_FOR_RETRAIN

    res = client.post("/api/v1/model/retrain", headers=_headers(token)).json()
    assert res["trained"] is False
    assert res["promoted"] is False


def test_scorecard_and_retrain_with_enough_outcomes(client):
    """Label enough customers to pass the gate; retrain trains a model."""
    token = _signup(client, "labeled@example.com")
    _upload(client, token, _make_csv(90))
    customers = _all_customers(client, token)

    # Label with a real signal: short-tenure customers churned, long-tenure retained.
    labeled = 0
    for c in customers:
        if labeled >= MIN_OUTCOMES_FOR_RETRAIN + 20:
            break
        outcome = "churned" if c["tenure"] <= 12 else "retained"
        client.post(
            "/api/v1/model/outcome",
            headers=_headers(token),
            json={"customer_id": c["id"], "outcome": outcome},
        )
        labeled += 1

    card = client.get("/api/v1/model/scorecard", headers=_headers(token)).json()
    assert card["n_outcomes"] == labeled
    assert card["accuracy"] is not None
    assert card["confusion"] is not None
    assert card["can_retrain"] is True

    res = client.post("/api/v1/model/retrain", headers=_headers(token)).json()
    assert res["trained"] is True
    assert isinstance(res["promoted"], bool)
    assert res["base_auc"] is not None
    assert res["tenant_auc"] is not None
    assert res["n_train"] + res["n_eval"] == labeled


# ── Isolation ────────────────────────────────────────────────────────────────


def test_outcomes_are_tenant_isolated(client, token):
    """Another org sees none of this org's outcomes and can't label its customers."""
    other = _signup(client, "intruder@example.com")
    victim = _all_customers(client, token)[0]

    res = client.post(
        "/api/v1/model/outcome",
        headers=_headers(other),
        json={"customer_id": victim["id"], "outcome": "churned"},
    )
    assert res.status_code == 404

    card = client.get("/api/v1/model/scorecard", headers=_headers(other)).json()
    assert card["n_outcomes"] == 0
