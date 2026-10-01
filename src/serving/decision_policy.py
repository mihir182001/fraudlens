"""
decision_policy.py, FastAPI/Streamlit serving module: turns a raw model
output (a fraud probability, or a scorecard points score) into one of three
business decisions: "approve", "review", or "decline".

This is a deliberately thin, separately testable layer on top of each
model's own output, for a reason worth stating plainly: neither model
produces a decision by itself. A model produces a probability or a score; a
BUSINESS then decides what to do at each level of that probability or
score, based on a cost tradeoff (the cost of a false decline vs. a false
approve) that has nothing to do with the model's own training. Keeping that
tradeoff in its own small, pure functions here, instead of scattering
threshold comparisons inside the API endpoints, makes the actual business
policy visible in one place and unit-testable on its own, independent of
whichever model happens to be loaded.

The specific thresholds and cutoffs used by train_serving_models.py (see
that module) are illustrative choices for this synthetic-data demo, not a
claim about any real institution's actual risk appetite:
  - Fraud: the F1-optimal threshold from Week 4's own metrics module
    (already a real, measured cut point on this model's own test set) is
    used as the REVIEW boundary; a second, higher DECLINE boundary is set
    halfway between that and 1.0, on the reasoning that only the most
    confident fraud predictions should be auto-blocked without a human
    look, while anything above the F1-optimal cut point but below that is
    worth a manual review rather than a silent pass-through.
  - Credit: APPROVE/REVIEW/DECLINE cutoffs are set from percentiles of the
    scorecard's own TRAIN score distribution (60th and 20th percentile
    respectively), never from test or from live traffic, the same
    fit-on-train-only discipline this project has followed since Week 2.

A real deployment would set both models' thresholds from an explicit,
often regulator-reviewed cost-benefit analysis, not a percentile pulled
from a synthetic training set; that analysis is outside this project's
scope, and the numbers below should be read as a working illustration of
the mechanism, not a recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

DECISION_APPROVE = "approve"
DECISION_REVIEW = "review"
DECISION_DECLINE = "decline"


@dataclass(frozen=True)
class FraudDecisionThresholds:
    """review_threshold and decline_threshold are both fraud PROBABILITIES
    in [0, 1]; review_threshold must be strictly below decline_threshold.
    """

    review_threshold: float
    decline_threshold: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.review_threshold <= 1.0:
            raise ValueError("review_threshold must be between 0 and 1.")
        if not 0.0 <= self.decline_threshold <= 1.0:
            raise ValueError("decline_threshold must be between 0 and 1.")
        if self.review_threshold >= self.decline_threshold:
            raise ValueError("review_threshold must be strictly below decline_threshold.")


@dataclass(frozen=True)
class CreditCutoffs:
    """approve_cutoff and decline_cutoff are both scorecard POINTS scores;
    a HIGHER score is safer (see scorecard_model.py's own convention), so
    approve_cutoff must be strictly above decline_cutoff.
    """

    approve_cutoff: float
    decline_cutoff: float

    def __post_init__(self) -> None:
        if self.approve_cutoff <= self.decline_cutoff:
            raise ValueError("approve_cutoff must be strictly above decline_cutoff.")


def fraud_decision(probability: float, thresholds: FraudDecisionThresholds) -> str:
    """A higher fraud probability is riskier: below review_threshold is
    "approve", from review_threshold up to decline_threshold is "review",
    and decline_threshold or above is "decline".
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be between 0 and 1, got {probability!r}.")
    if probability >= thresholds.decline_threshold:
        return DECISION_DECLINE
    if probability >= thresholds.review_threshold:
        return DECISION_REVIEW
    return DECISION_APPROVE


def credit_decision(score: float, cutoffs: CreditCutoffs) -> str:
    """A higher scorecard score is safer: at or above approve_cutoff is
    "approve", at or above decline_cutoff (but below approve_cutoff) is
    "review", and below decline_cutoff is "decline".
    """
    if score >= cutoffs.approve_cutoff:
        return DECISION_APPROVE
    if score >= cutoffs.decline_cutoff:
        return DECISION_REVIEW
    return DECISION_DECLINE
