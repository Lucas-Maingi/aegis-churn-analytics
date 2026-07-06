"""
SaaS Layer Integration Tests
============================
Covers signup/login, tenant isolation, CSV upload + mapping + scoring,
the ranked customer list, and one-click outreach.
"""

import io
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from churn_prediction.api.main import app
from churn_prediction.api.middleware import InMemoryRateLimiter
from churn_prediction.saas import db as saas_db
from churn_prediction.saas.db import Base


@pytest.fixture(name="client", scope="module")
def client_fixture():
    """TestClient with an isolated in-memory database and no rate limiting.

    The rate-limit test in test_api.py intentionally fills the shared
    sliding window, so this module bypasses the limiter entirely.
    """
    mp = pytest.MonkeyPatch()

    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    Base.metadata.create_all(bind=test_engine)

    mp.setattr(saas_db, "SessionLocal", TestSession)
    mp.setattr(InMemoryRateLimiter, "check_limit", lambda self, key: True)

    with TestClient(app) as c:
        yield c

    mp.undo()


# ── Helpers ──────────────────────────────────────────────────────────────────

SIGNUP_PAYLOAD = {
    "organization_name": "Nairobi Fiber Co",
    "industry": "ISP",
    "full_name": "Lucas M",
    "email": "owner@nairobifiber.example",
    "password": "s3cure-password",
}

# A billing export with non-Telco headers and messy values, as a real
# ISP would produce it.
CSV_CONTENT = (
    "subscriber_id,customer name,email address,months_active,monthly_fee,"
    "plan type,contract type,payment method\n"
    "SUB-001,Alice W,alice@example.com,2,45.00,fiber,monthly,mpesa\n"
    "SUB-002,Brian K,brian@example.com,48,30.00,dsl,2 year,direct debit\n"
    "SUB-003,Carol N,carol@example.com,6,80.00,fiber,monthly,mpesa\n"
    ",No ID,noid@example.com,3,20.00,fiber,monthly,cash\n"
)

CSV_MAPPING = {
    "external_id": "subscriber_id",
    "name": "customer name",
    "email": "email address",
    "tenure": "months_active",
    "MonthlyCharges": "monthly_fee",
    "InternetService": "plan type",
    "Contract": "contract type",
    "PaymentMethod": "payment method",
}


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _upload(client: TestClient, token: str, csv_text: str = CSV_CONTENT):
    return client.post(
        "/api/v1/customers/upload",
        headers=_auth_headers(token),
        files={"file": ("customers.csv", io.BytesIO(csv_text.encode()), "text/csv")},
        data={"mapping": json.dumps(CSV_MAPPING)},
    )


@pytest.fixture(name="token", scope="module")
def token_fixture(client: TestClient) -> str:
    res = client.post("/api/v1/auth/signup", json=SIGNUP_PAYLOAD)
    assert res.status_code == 201
    return res.json()["access_token"]


# ── Auth ─────────────────────────────────────────────────────────────────────


def test_signup_returns_token_and_org(client, token):
    # token fixture already asserted the 201; verify a login round-trip
    res = client.post(
        "/api/v1/auth/login",
        json={"email": SIGNUP_PAYLOAD["email"], "password": SIGNUP_PAYLOAD["password"]},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["organization_name"] == "Nairobi Fiber Co"
    assert body["access_token"]


def test_duplicate_signup_conflicts(client, token):
    res = client.post("/api/v1/auth/signup", json=SIGNUP_PAYLOAD)
    assert res.status_code == 409


def test_login_wrong_password_rejected(client, token):
    res = client.post(
        "/api/v1/auth/login",
        json={"email": SIGNUP_PAYLOAD["email"], "password": "wrong-password"},
    )
    assert res.status_code == 401


def test_customers_require_auth(client):
    assert client.get("/api/v1/customers").status_code == 401


# ── Upload & scoring ─────────────────────────────────────────────────────────


def test_upload_preview_suggests_mapping(client, token):
    res = client.post(
        "/api/v1/customers/upload/preview",
        headers=_auth_headers(token),
        files={"file": ("customers.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["row_count"] == 4
    suggested = body["suggested_mapping"]
    assert suggested.get("external_id") == "subscriber_id"
    assert suggested.get("MonthlyCharges") == "monthly_fee"
    assert suggested.get("Contract") == "contract type"


def test_upload_imports_and_scores(client, token):
    res = _upload(client, token)
    assert res.status_code == 200
    body = res.json()
    assert body["imported"] == 3
    assert body["scored"] == 3
    assert body["skipped"] == 1  # row without a subscriber_id


def test_reupload_updates_instead_of_duplicating(client, token):
    res = _upload(client, token)
    assert res.status_code == 200
    body = res.json()
    assert body["imported"] == 0
    assert body["updated"] == 3


def test_customer_list_is_ranked_with_summary(client, token):
    res = client.get("/api/v1/customers", headers=_auth_headers(token))
    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 3

    probs = [item["churn_probability"] for item in body["items"]]
    assert probs == sorted(probs, reverse=True)

    counts = body["tier_counts"]
    assert counts["HIGH"] + counts["MEDIUM"] + counts["LOW"] == 3

    # Short-tenure month-to-month fiber customers should outrank the
    # 48-month two-year DSL customer.
    top = body["items"][0]
    assert top["contract"] == "Month-to-month"
    assert body["items"][-1]["external_id"] == "SUB-002"


def test_customer_detail_has_explanations(client, token):
    listing = client.get("/api/v1/customers", headers=_auth_headers(token)).json()
    cust_id = listing["items"][0]["id"]

    res = client.get(f"/api/v1/customers/{cust_id}", headers=_auth_headers(token))
    assert res.status_code == 200
    body = res.json()
    assert len(body["explanations"]) == 3
    for exp in body["explanations"]:
        assert exp["plain_english"]
        assert "churn risk" in exp["direction"]
    assert body["features"]["Contract"] == "Month-to-month"
    assert body["features"]["PaymentMethod"] == "Electronic check"  # mpesa mapped


def test_tenant_isolation(client, token):
    res = client.post(
        "/api/v1/auth/signup",
        json={
            "organization_name": "Other ISP",
            "email": "owner@otherisp.example",
            "password": "another-password",
        },
    )
    assert res.status_code == 201
    other_token = res.json()["access_token"]

    listing = client.get("/api/v1/customers", headers=_auth_headers(other_token)).json()
    assert listing["total"] == 0


# ── Outreach ─────────────────────────────────────────────────────────────────


def test_templates_available(client, token):
    res = client.get("/api/v1/outreach/templates", headers=_auth_headers(token))
    assert res.status_code == 200
    keys = [tpl["key"] for tpl in res.json()]
    assert set(keys) == {"discount_offer", "contract_upgrade", "check_in"}


def test_one_click_send_and_history(client, token):
    listing = client.get("/api/v1/customers", headers=_auth_headers(token)).json()
    top = listing["items"][0]

    res = client.post(
        "/api/v1/outreach/send",
        headers=_auth_headers(token),
        json={"customer_id": top["id"], "template_key": "discount_offer"},
    )
    assert res.status_code == 200
    body = res.json()
    # No RESEND_API_KEY in the test environment → simulated delivery
    assert body["message"]["status"] == "simulated"
    assert "Nairobi Fiber Co" in body["message"]["subject"]

    history = client.get(
        "/api/v1/outreach/history", headers=_auth_headers(token)
    ).json()
    assert len(history) == 1
    assert history[0]["customer_id"] == top["id"]
    assert history[0]["template_key"] == "discount_offer"


def test_send_to_foreign_customer_blocked(client, token):
    other_login = client.post(
        "/api/v1/auth/login",
        json={"email": "owner@otherisp.example", "password": "another-password"},
    ).json()

    listing = client.get("/api/v1/customers", headers=_auth_headers(token)).json()
    top_id = listing["items"][0]["id"]

    res = client.post(
        "/api/v1/outreach/send",
        headers=_auth_headers(other_login["access_token"]),
        json={"customer_id": top_id, "template_key": "check_in"},
    )
    assert res.status_code == 404
