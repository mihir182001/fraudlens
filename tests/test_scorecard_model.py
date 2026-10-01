"""
Tests for src.credit_risk.scorecard_model.

test_final_score_is_a_monotonic_transform_of_predicted_probability is this
module's central claim (see its own module docstring): converting a
logistic regression to a points scorecard must not change what the model
can discriminate, only how a business user reads it. Checked directly by
comparing AUC-ROC computed from the raw predicted probability against
AUC-ROC computed from the points score (negated, since a higher score means
lower risk): they must come out identical, not merely correlated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.credit_risk.scorecard_model import (
    build_points_scorecard,
    fit_scorecard_logistic_regression,
    score_applications,
)
from src.credit_risk.woe_encoding import fit_woe_bins, transform_woe

FEATURE_COLUMNS = ["EXT_SOURCE", "DAYS_EMPLOYED"]


def _toy_credit_frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ext_source = rng.uniform(0, 1, size=n)
    days_employed = -rng.integers(30, 5000, size=n).astype(float)
    default_prob = 1.0 / (1.0 + np.exp(-(2.0 - 4.0 * ext_source - 0.0003 * (-days_employed))))
    target = (rng.random(n) < default_prob).astype(int)
    return pd.DataFrame({"EXT_SOURCE": ext_source, "DAYS_EMPLOYED": days_employed, "TARGET": target})


def _fit_pipeline(seed=0):
    df = _toy_credit_frame(seed=seed)
    train_df, test_df = train_test_split(df, test_size=0.25, stratify=df["TARGET"], random_state=0)
    woe_bins = fit_woe_bins(train_df, FEATURE_COLUMNS, n_bins=5)
    woe_train = transform_woe(train_df, woe_bins)
    woe_test = transform_woe(test_df, woe_bins)
    model = fit_scorecard_logistic_regression(woe_train, FEATURE_COLUMNS)
    return model, woe_bins, woe_train, woe_test, train_df, test_df


def test_scorecard_coefficients_are_negative_for_a_predictive_woe_feature():
    model, *_ = _fit_pipeline()
    # EXT_SOURCE's WoE is constructed so higher WoE means safer (see
    # woe_encoding.py's convention); a well-behaved model predicting
    # TARGET=1 (bad) should therefore assign it a negative coefficient.
    assert model.coef_[0][0] < 0


def test_points_increase_monotonically_with_woe_within_a_feature():
    model, woe_bins, *_ = _fit_pipeline()
    base_points, points_tables = build_points_scorecard(model, FEATURE_COLUMNS, woe_bins)

    table = points_tables["EXT_SOURCE"].sort_values("woe").reset_index(drop=True)
    assert list(table["points"]) == sorted(table["points"])


def test_final_score_is_a_monotonic_transform_of_predicted_probability():
    model, woe_bins, woe_train, woe_test, train_df, test_df = _fit_pipeline()
    base_points, points_tables = build_points_scorecard(model, FEATURE_COLUMNS, woe_bins)

    woe_columns = [f"{f}_woe" for f in FEATURE_COLUMNS]
    test_proba = model.predict_proba(woe_test[woe_columns])[:, 1]
    test_scores = score_applications(test_df, FEATURE_COLUMNS, woe_bins, base_points, points_tables)

    auc_from_proba = roc_auc_score(test_df["TARGET"], test_proba)
    auc_from_score = roc_auc_score(test_df["TARGET"], -test_scores)
    assert auc_from_proba == pytest.approx(auc_from_score, abs=1e-9)


def test_base_points_shifts_with_base_score_argument():
    model, woe_bins, *_ = _fit_pipeline()
    base_600, _ = build_points_scorecard(model, FEATURE_COLUMNS, woe_bins, base_score=600.0)
    base_700, _ = build_points_scorecard(model, FEATURE_COLUMNS, woe_bins, base_score=700.0)
    assert base_700 - base_600 == pytest.approx(100.0)


def test_fit_scorecard_logistic_regression_rejects_missing_woe_column():
    df = _toy_credit_frame(n=200)
    with pytest.raises(ValueError, match="missing"):
        fit_scorecard_logistic_regression(df, FEATURE_COLUMNS)


def test_build_points_scorecard_rejects_non_positive_pdo():
    model, woe_bins, *_ = _fit_pipeline()
    with pytest.raises(ValueError, match="pdo must be positive"):
        build_points_scorecard(model, FEATURE_COLUMNS, woe_bins, pdo=0)


def test_build_points_scorecard_rejects_non_positive_base_odds():
    model, woe_bins, *_ = _fit_pipeline()
    with pytest.raises(ValueError, match="base_odds must be positive"):
        build_points_scorecard(model, FEATURE_COLUMNS, woe_bins, base_odds=0)
