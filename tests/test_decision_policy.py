"""
Tests for src.serving.decision_policy.

These are deliberately independent of either trained model: the whole
point of keeping the threshold comparisons in their own pure functions
(see the module's own docstring) is that the POLICY layer can be checked
on its own, with made-up thresholds and scores, without training anything.
"""

from __future__ import annotations

import pytest

from src.serving.decision_policy import (
    CreditCutoffs,
    FraudDecisionThresholds,
    credit_decision,
    fraud_decision,
)


def test_fraud_decision_approve_below_review_threshold():
    thresholds = FraudDecisionThresholds(review_threshold=0.5, decline_threshold=0.75)
    assert fraud_decision(0.1, thresholds) == "approve"
    assert fraud_decision(0.499, thresholds) == "approve"


def test_fraud_decision_review_between_thresholds():
    thresholds = FraudDecisionThresholds(review_threshold=0.5, decline_threshold=0.75)
    assert fraud_decision(0.5, thresholds) == "review"
    assert fraud_decision(0.74, thresholds) == "review"


def test_fraud_decision_decline_at_or_above_decline_threshold():
    thresholds = FraudDecisionThresholds(review_threshold=0.5, decline_threshold=0.75)
    assert fraud_decision(0.75, thresholds) == "decline"
    assert fraud_decision(1.0, thresholds) == "decline"


def test_fraud_decision_rejects_out_of_range_probability():
    thresholds = FraudDecisionThresholds(review_threshold=0.5, decline_threshold=0.75)
    with pytest.raises(ValueError, match="between 0 and 1"):
        fraud_decision(1.5, thresholds)
    with pytest.raises(ValueError, match="between 0 and 1"):
        fraud_decision(-0.1, thresholds)


def test_fraud_decision_thresholds_reject_invalid_ordering():
    with pytest.raises(ValueError, match="strictly below"):
        FraudDecisionThresholds(review_threshold=0.8, decline_threshold=0.5)
    with pytest.raises(ValueError, match="strictly below"):
        FraudDecisionThresholds(review_threshold=0.5, decline_threshold=0.5)


def test_fraud_decision_thresholds_reject_out_of_range_values():
    with pytest.raises(ValueError, match="review_threshold"):
        FraudDecisionThresholds(review_threshold=1.5, decline_threshold=1.6)


def test_credit_decision_approve_at_or_above_approve_cutoff():
    cutoffs = CreditCutoffs(approve_cutoff=600.0, decline_cutoff=500.0)
    assert credit_decision(600.0, cutoffs) == "approve"
    assert credit_decision(700.0, cutoffs) == "approve"


def test_credit_decision_review_between_cutoffs():
    cutoffs = CreditCutoffs(approve_cutoff=600.0, decline_cutoff=500.0)
    assert credit_decision(599.9, cutoffs) == "review"
    assert credit_decision(500.0, cutoffs) == "review"


def test_credit_decision_decline_below_decline_cutoff():
    cutoffs = CreditCutoffs(approve_cutoff=600.0, decline_cutoff=500.0)
    assert credit_decision(499.9, cutoffs) == "decline"
    assert credit_decision(0.0, cutoffs) == "decline"


def test_credit_cutoffs_reject_invalid_ordering():
    with pytest.raises(ValueError, match="strictly above"):
        CreditCutoffs(approve_cutoff=500.0, decline_cutoff=600.0)
    with pytest.raises(ValueError, match="strictly above"):
        CreditCutoffs(approve_cutoff=500.0, decline_cutoff=500.0)
