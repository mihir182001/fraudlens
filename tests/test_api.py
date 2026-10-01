"""
Tests for src.api.main, the FastAPI serving layer.

api_client is a module-scoped fixture: it trains small (fast) fraud and
credit artifacts once per test module with
src.serving.train_serving_models' own functions (the exact code path
train_serving_models.py's CLI uses, just with a smaller n for test speed),
saves them into a tmp_path_factory directory, points FRAUDLENS_MODELS_DIR
and FRAUDLENS_AUDIT_DB at that temporary location BEFORE importing
src.api.main (see that module's docstring: both are read fresh inside
lifespan, not at import time, specifically so a test can do this), then
opens the app as a context manager so its lifespan actually runs.

Nothing here ever touches this project's real models/ directory or the
real audit_log.db a locally running instance would use.
"""

from __future__ import annotations

import json

import joblib
import pytest
from fastapi.testclient import TestClient

from src.serving.train_serving_models import (
    train_credit_serving_scorecard,
    train_fraud_serving_model,
)


@pytest.fixture(scope="module")
def api_client(tmp_path_factory, monkeypatch_module):
    models_dir = tmp_path_factory.mktemp("models")
    audit_db_path = tmp_path_factory.mktemp("audit") / "audit_log.db"

    fraud_artifact = train_fraud_serving_model(n_transactions=1200, seed=1)
    joblib.dump(fraud_artifact, models_dir / "fraud_model.joblib")

    credit_artifact = train_credit_serving_scorecard(n_applications=1200, seed=1)
    joblib.dump(credit_artifact, models_dir / "credit_scorecard.joblib")

    monkeypatch_module.setenv("FRAUDLENS_MODELS_DIR", str(models_dir))
    monkeypatch_module.setenv("FRAUDLENS_AUDIT_DB", str(audit_db_path))

    from src.api.main import app

    with TestClient(app) as client:
        client.fraud_artifact = fraud_artifact
        client.credit_artifact = credit_artifact
        client.audit_db_path = audit_db_path
        yield client


@pytest.fixture(scope="module")
def monkeypatch_module():
    # pytest's built-in monkeypatch fixture is function-scoped; this module
    # needs a module-scoped equivalent so api_client (also module-scoped,
    # to avoid retraining models for every single test) can set environment
    # variables that last for the whole module.
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_health_reports_both_models_loaded(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["fraud_model_loaded"] is True
    assert body["credit_model_loaded"] is True


def test_fraud_schema_lists_feature_columns(api_client):
    response = api_client.get("/models/fraud/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["feature_columns"] == api_client.fraud_artifact["feature_columns"]
    assert 0.0 <= body["review_threshold"] < body["decline_threshold"] <= 1.0


def test_credit_schema_lists_feature_columns(api_client):
    response = api_client.get("/models/credit/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["feature_columns"] == api_client.credit_artifact["feature_columns"]
    assert body["decline_cutoff"] < body["approve_cutoff"]


def test_models_comparison_returns_both_metrics(api_client):
    response = api_client.get("/models/comparison")
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["fraud_model"]["test_auc_roc"] <= 1.0
    assert 0.0 <= body["credit_model"]["test_auc_roc"] <= 1.0


def test_score_fraud_returns_valid_decision_and_logs_audit_row(api_client):
    feature_columns = api_client.fraud_artifact["feature_columns"]
    features = {c: 0.0 for c in feature_columns}
    response = api_client.post("/score/fraud", json={"features": features})
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["decision"] in ("approve", "review", "decline")

    from src.serving.audit_log import fetch_recent_decisions
    recent = fetch_recent_decisions(db_path=api_client.audit_db_path, model_type="fraud")
    assert len(recent) >= 1
    logged_features = json.loads(recent.iloc[0]["input_json"])
    assert logged_features == features


def test_score_fraud_rejects_missing_required_feature(api_client):
    feature_columns = api_client.fraud_artifact["feature_columns"]
    incomplete = {c: 0.0 for c in feature_columns[:-1]}  # drop the last required feature.
    response = api_client.post("/score/fraud", json={"features": incomplete})
    assert response.status_code == 422
    assert feature_columns[-1] in response.json()["detail"]


def test_score_fraud_accepts_null_for_a_legitimately_missing_feature(api_client):
    # Regression test for a real, disclosed schema gap: JSON has no NaN
    # literal, so a caller cannot send NaN directly for a feature that is
    # genuinely missing (dist1, dist2, and some D-family columns are
    # legitimately NaN in real IEEE-CIS transactions). A null is how a
    # caller expresses that same missingness over HTTP, and score_fraud
    # must convert it back to a real NaN for the pipeline's own
    # SimpleImputer to handle, exactly as it does during training, rather
    # than rejecting the request or crashing on it.
    feature_columns = api_client.fraud_artifact["feature_columns"]
    features = {c: 0.0 for c in feature_columns}
    features[feature_columns[0]] = None  # sent as JSON null.

    response = api_client.post("/score/fraud", json={"features": features})
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["decision"] in ("approve", "review", "decline")

    # The key was present (with a null value), so it must NOT be treated
    # as a missing required feature. fetch_recent_decisions orders newest
    # first (see its own docstring), so the row just logged is iloc[0].
    from src.serving.audit_log import fetch_recent_decisions
    recent = fetch_recent_decisions(db_path=api_client.audit_db_path, model_type="fraud")
    logged_features = json.loads(recent.iloc[0]["input_json"])
    assert logged_features[feature_columns[0]] is None


def test_score_credit_returns_valid_decision_and_logs_audit_row(api_client):
    payload = {
        "AMT_INCOME_TOTAL": 150000.0, "AMT_CREDIT": 450000.0, "AMT_ANNUITY": 22000.0,
        "AMT_GOODS_PRICE": 420000.0, "DAYS_BIRTH": -14000, "DAYS_EMPLOYED": -2000,
        "NAME_EDUCATION_TYPE": "Higher education", "EXT_SOURCE_1": 0.6,
        "EXT_SOURCE_2": 0.7, "EXT_SOURCE_3": 0.55,
    }
    response = api_client.post("/score/credit", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] in ("approve", "review", "decline")
    assert 0.0 <= body["probability"] <= 1.0

    from src.serving.audit_log import fetch_recent_decisions
    recent = fetch_recent_decisions(db_path=api_client.audit_db_path, model_type="credit")
    assert len(recent) >= 1


def test_score_credit_handles_missing_ext_source_via_woe_missing_bin(api_client):
    # No EXT_SOURCE_1/2/3 at all: a real, disclosed possibility (see
    # generate_synthetic_credit.py), and this must resolve to woe_encoding's
    # own MISSING_LABEL bin rather than raising or crashing the endpoint.
    payload = {"AMT_INCOME_TOTAL": 90000.0, "DAYS_BIRTH": -9000}
    response = api_client.post("/score/credit", json=payload)
    assert response.status_code == 200
    assert response.json()["decision"] in ("approve", "review", "decline")


def test_audit_log_captures_both_fraud_and_credit_decisions(api_client):
    from src.serving.audit_log import decision_counts
    counts = decision_counts(db_path=api_client.audit_db_path)
    model_types = set(counts["model_type"])
    assert "fraud" in model_types
    assert "credit" in model_types
