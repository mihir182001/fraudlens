"""
schemas.py, FastAPI serving module: pydantic request/response models for
the fraud and credit scoring endpoints (src/api/main.py).

The two request schemas are deliberately shaped differently, for a real
reason rather than an inconsistency: the fraud model's feature columns
(velocity features, card-usage features, hour-of-day aggregates,
frequency-encoded categoricals; see prepare_model_data.py) require
historical transaction state a single stateless HTTP request does not
carry, so FraudScoringRequest accepts an already-engineered feature vector
by name, exactly as a real deployment's upstream feature-serving layer
would hand off to a stateless scoring service. GET /models/fraud/schema
(see src/api/main.py) reports the exact feature names the loaded model
expects. The credit scorecard, by contrast, needs only the applicant's own
attributes (no history required), so CreditScoringRequest exposes the real
Home Credit application schema (generate_synthetic_credit.py) directly as
named fields.

Every CreditScoringRequest field is Optional, defaulting to None, which
becomes a real, meaningfully handled missing value once it reaches
woe_encoding.py's own MISSING_LABEL bin, exactly the same missingness this
project's synthetic generator already reproduces for EXT_SOURCE_1 and
DAYS_EMPLOYED. A field genuinely missing from a real application is not an
error here; it is data the scorecard was built to handle.

FraudScoringRequest.features is typed Dict[str, Optional[float]], not
Dict[str, float], for a real reason: several of the fraud model's own
engineered columns (dist1, dist2, some D-family columns; see
prepare_train_test_features) are legitimately NaN in real transactions, and
the trained pipeline's own SimpleImputer step (see
build_advanced_pipeline in advanced_models.py) exists specifically to
handle that. JSON itself has no NaN literal, so a caller cannot send NaN
directly; sending null for that key is how a caller expresses the same
missingness over HTTP, and src/api/main.py converts each null back to a
real NaN before calling the pipeline, so the imputer sees exactly what it
would during training. A key still has to be PRESENT, with either a number
or null, since which features exist is part of the model's contract; only
the value may be null.
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class FraudScoringRequest(BaseModel):
    features: Dict[str, Optional[float]] = Field(
        ...,
        description="Already-engineered feature vector, keyed by the exact "
        "column names GET /models/fraud/schema reports. Any required "
        "column missing from this dict returns a 422. A column whose real "
        "value is missing (e.g. dist1, dist2) is sent as null, which the "
        "API converts to NaN for the model's own imputer to handle, the "
        "same way it does during training.",
    )


class FraudScoringResponse(BaseModel):
    probability: float
    decision: str
    model_version: str
    review_threshold: float
    decline_threshold: float


class CreditScoringRequest(BaseModel):
    AMT_INCOME_TOTAL: Optional[float] = None
    AMT_CREDIT: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    DAYS_BIRTH: Optional[float] = None
    DAYS_EMPLOYED: Optional[float] = None
    CODE_GENDER: Optional[str] = None
    FLAG_OWN_CAR: Optional[str] = None
    FLAG_OWN_REALTY: Optional[str] = None
    CNT_CHILDREN: Optional[float] = None
    CNT_FAM_MEMBERS: Optional[float] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    NAME_INCOME_TYPE: Optional[str] = None
    NAME_HOUSING_TYPE: Optional[str] = None
    REGION_RATING_CLIENT: Optional[float] = None
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None


class CreditScoringResponse(BaseModel):
    score: float
    probability: float
    decision: str
    model_version: str
    approve_cutoff: float
    decline_cutoff: float


class ModelMetrics(BaseModel):
    trained_at: str
    test_auc_roc: float
    test_auc_pr: float
    test_ks_statistic: float
    test_gini: float
    n_train: int
    n_test: int


class ModelComparisonResponse(BaseModel):
    fraud_model: ModelMetrics
    credit_model: ModelMetrics
