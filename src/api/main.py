"""
main.py, FastAPI serving module: the fraud scoring endpoint, the credit
risk endpoint, a model comparison endpoint, and risk decision audit
logging, the four items Week 11 of this project's spec asks for.

Both models are loaded ONCE at startup (see lifespan below), from
models/fraud_model.joblib and models/credit_scorecard.joblib
(src/serving/train_serving_models.py writes these; run that script first if
this app fails to start with a "run train_serving_models" error). Neither
model is ever refit inside a request: every request is pure inference
against whatever was fit offline, the same fit-once-serve-many discipline
real model serving requires and this project's own train/test splits have
followed since Week 2.

FRAUDLENS_MODELS_DIR and FRAUDLENS_AUDIT_DB (both read fresh every time the
app starts, inside lifespan, not once at import time) let tests point this
app at a temporary directory and a temporary audit database without
touching the real models/ or audit_log.db a locally running instance uses.

Run locally with:
    uvicorn src.api.main:app --reload
then open http://127.0.0.1:8000/docs for the interactive OpenAPI page.

score_fraud converts each null value in the request body's features to a
real float("nan") before calling the pipeline (see the comment at that
conversion, and schemas.py's own docstring for why FraudScoringRequest
types features as Dict[str, Optional[float]] rather than Dict[str, float]
in the first place): JSON cannot carry NaN directly, and several of this
model's own real engineered features are legitimately missing in practice.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.credit_risk.scorecard_model import score_applications
from src.credit_risk.woe_encoding import transform_woe
from src.serving.audit_log import DEFAULT_AUDIT_DB_PATH, init_audit_db, log_decision
from src.serving.decision_policy import credit_decision, fraud_decision
from src.serving.schemas import (
    CreditScoringRequest,
    CreditScoringResponse,
    FraudScoringRequest,
    FraudScoringResponse,
    ModelComparisonResponse,
    ModelMetrics,
)

FRAUD_MODEL_FILENAME = "fraud_model.joblib"
CREDIT_MODEL_FILENAME = "credit_scorecard.joblib"


def _models_dir() -> Path:
    return Path(os.environ.get("FRAUDLENS_MODELS_DIR", "models"))


def _audit_db_path() -> Path:
    return Path(os.environ.get("FRAUDLENS_AUDIT_DB", str(DEFAULT_AUDIT_DB_PATH)))


def _load_artifact(models_dir: Path, filename: str) -> Dict[str, Any]:
    path = models_dir / filename
    if not path.exists():
        raise RuntimeError(
            f"{path} not found. Run `python -m src.serving.train_serving_models` "
            "first to create the serving model artifacts."
        )
    return joblib.load(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = _models_dir()
    app.state.fraud_artifact = _load_artifact(models_dir, FRAUD_MODEL_FILENAME)
    app.state.credit_artifact = _load_artifact(models_dir, CREDIT_MODEL_FILENAME)
    app.state.audit_db_path = _audit_db_path()
    init_audit_db(app.state.audit_db_path)
    yield


app = FastAPI(
    title="FraudLens Risk Scoring API",
    description="Fraud transaction scoring and credit application scoring, "
    "backed by this project's own trained models, with every decision "
    "logged to an audit trail.",
    lifespan=lifespan,
)


def _metrics_from_artifact(artifact: Dict[str, Any]) -> ModelMetrics:
    return ModelMetrics(
        trained_at=artifact["trained_at"],
        test_auc_roc=artifact["test_auc_roc"],
        test_auc_pr=artifact["test_auc_pr"],
        test_ks_statistic=artifact["test_ks_statistic"],
        test_gini=artifact["test_gini"],
        n_train=artifact["n_train"],
        n_test=artifact["n_test"],
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "fraud_model_loaded": getattr(app.state, "fraud_artifact", None) is not None,
        "credit_model_loaded": getattr(app.state, "credit_artifact", None) is not None,
    }


@app.get("/models/fraud/schema")
def fraud_schema() -> Dict[str, Any]:
    artifact = app.state.fraud_artifact
    return {
        "feature_columns": artifact["feature_columns"],
        "review_threshold": artifact["thresholds"].review_threshold,
        "decline_threshold": artifact["thresholds"].decline_threshold,
        "trained_at": artifact["trained_at"],
    }


@app.get("/models/credit/schema")
def credit_schema() -> Dict[str, Any]:
    artifact = app.state.credit_artifact
    return {
        "feature_columns": artifact["feature_columns"],
        "approve_cutoff": artifact["cutoffs"].approve_cutoff,
        "decline_cutoff": artifact["cutoffs"].decline_cutoff,
        "trained_at": artifact["trained_at"],
    }


@app.get("/models/comparison", response_model=ModelComparisonResponse)
def models_comparison() -> ModelComparisonResponse:
    return ModelComparisonResponse(
        fraud_model=_metrics_from_artifact(app.state.fraud_artifact),
        credit_model=_metrics_from_artifact(app.state.credit_artifact),
    )


@app.post("/score/fraud", response_model=FraudScoringResponse)
def score_fraud(request: FraudScoringRequest) -> FraudScoringResponse:
    artifact = app.state.fraud_artifact
    feature_columns = artifact["feature_columns"]

    missing = [c for c in feature_columns if c not in request.features]
    if missing:
        raise HTTPException(
            status_code=422, detail=f"Missing required feature(s): {missing}",
        )

    # JSON has no NaN literal, so a legitimately-missing engineered feature
    # (dist1, dist2, some D-family columns; see schemas.py's own docstring)
    # arrives here as None, not NaN. Converting it to a real float("nan")
    # explicitly, rather than leaving it as None, matters for two reasons:
    # it keeps every column's dtype numeric (a lone None in a single-row
    # DataFrame would otherwise infer as an object-dtype column, which the
    # pipeline's SimpleImputer cannot process), and it reproduces exactly
    # the NaN the imputer was fit against during training.
    row = pd.DataFrame(
        [{c: (float("nan") if request.features[c] is None else request.features[c]) for c in feature_columns}]
    )
    probability = float(artifact["pipeline"].predict_proba(row)[:, 1][0])
    decision = fraud_decision(probability, artifact["thresholds"])

    log_decision(
        model_type="fraud",
        model_version=artifact["trained_at"],
        score=probability,
        decision=decision,
        input_payload=request.features,
        db_path=app.state.audit_db_path,
    )

    return FraudScoringResponse(
        probability=probability,
        decision=decision,
        model_version=artifact["trained_at"],
        review_threshold=artifact["thresholds"].review_threshold,
        decline_threshold=artifact["thresholds"].decline_threshold,
    )


@app.post("/score/credit", response_model=CreditScoringResponse)
def score_credit(request: CreditScoringRequest) -> CreditScoringResponse:
    artifact = app.state.credit_artifact
    payload = request.model_dump()
    row = pd.DataFrame([payload])

    woe_row = transform_woe(row, artifact["woe_bins"])
    woe_columns = [f"{f}_woe" for f in artifact["feature_columns"]]
    probability = float(artifact["model"].predict_proba(woe_row[woe_columns])[:, 1][0])

    score = float(
        score_applications(
            row, artifact["feature_columns"], artifact["woe_bins"],
            artifact["base_points"], artifact["points_tables"],
        )[0]
    )
    decision = credit_decision(score, artifact["cutoffs"])

    log_decision(
        model_type="credit",
        model_version=artifact["trained_at"],
        score=score,
        decision=decision,
        input_payload=payload,
        db_path=app.state.audit_db_path,
    )

    return CreditScoringResponse(
        score=score,
        probability=probability,
        decision=decision,
        model_version=artifact["trained_at"],
        approve_cutoff=artifact["cutoffs"].approve_cutoff,
        decline_cutoff=artifact["cutoffs"].decline_cutoff,
    )
