"""
uplift_models.py, Week 6: two standard ways to estimate individual-level
uplift, P(convert | treated) - P(convert | not treated), from randomized
campaign data (src/data/generate_synthetic_campaign.py).

Both approaches turn the causal problem into a plain classification problem
solvable with any scikit-learn-compatible classifier, then combine two
predicted probabilities into one uplift score. Neither can be trained or
scored with a single call to .fit()/.predict() the way Weeks 1-5's models
were, which is why each is its own small class here rather than a bare
function like build_pipeline in src/models/baseline_models.py.

SLearner ("single model") adds treatment as one more input feature and fits
one classifier P(Y=1 | X, T) on everyone. To score a customer's uplift, it
asks that one model two counterfactual questions on the same feature row:
"if this customer were treated" and "if this customer were not," and
subtracts the answers. It is simple and data-efficient (all rows train one
model), but a model with many features can easily let treatment's effect
get diluted or masked by everything else it is fitting, especially when the
true, treatment-driven signal is small relative to natural row-to-row
variation, exactly why sleeping_dog and lost_cause customers exist in the
test data below and are the hardest segments for the S-learner to keep
straight.

TLearner ("two models") fits two completely separate classifiers, one on
only the treated rows, one on only the control rows, then subtracts their
predictions. It cannot let one group's patterns leak into the other's
model by construction, which usually makes it better at finding a real
treatment effect, at the cost of each model seeing only half the data (a
real weakness on a small campaign) and, more subtly, of the two models
potentially learning to weigh even irrelevant features differently between
groups, which can inject noise into the subtraction. Both weaknesses are
real trade-offs in the causal-inference literature, not defects unique to
this implementation; a more sophisticated approach (the X-learner) exists
specifically to address them and is a natural extension beyond this
project's scope.

Both classes default to LogisticRegression, deterministic and needing no
random_state to reproduce, so the tests below don't have to account for
run-to-run training randomness; either accepts any other scikit-learn
classifier (a RandomForestClassifier, for instance) via base_estimator.

That default comes with a real, tested limitation worth disclosing rather
than hiding behind a better default: a plain LogisticRegression S-learner
adds treatment as one more additive term in the log-odds, with no
interaction against the other features, so its treatment coefficient is a
single number shared by every row. Confirmed directly (see
tests/test_uplift_models.py's
test_slearner_with_additive_base_estimator_cannot_recover_heterogeneous_effect):
on a dataset with a true, strongly heterogeneous effect (one segment at
+0.5 uplift, another at -0.3), a LogisticRegression S-learner predicts
nearly the same uplift, around +0.08, for both segments, essentially the
population-average effect, not each segment's real effect. A TLearner does
not share this failure mode (it fits two entirely separate models, so each
group's coefficients differ freely), and an S-learner built on a base
estimator that can express interactions on its own, such as a decision
tree or gradient-boosted trees, recovers the true heterogeneity correctly,
because it can split on treatment together with other features. This is
why a tree-based or boosted base_estimator is the more common practical
choice for an S-learner in production, despite LogisticRegression being
this module's default for the sake of deterministic, reproducible tests.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression

TREATMENT_COLUMN = "__treatment__"


def _validate_fit_inputs(X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> None:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")
    if TREATMENT_COLUMN in X.columns:
        raise ValueError(
            f"X must not already contain a column named {TREATMENT_COLUMN!r}; "
            "it is reserved for this module's internal use."
        )
    treatment = np.asarray(treatment)
    outcome = np.asarray(outcome)
    if not (len(X) == len(treatment) == len(outcome)):
        raise ValueError("X, treatment, and outcome must have the same length.")
    if len(X) == 0:
        raise ValueError("X must be non-empty.")
    if not set(np.unique(treatment)).issubset({0, 1}):
        raise ValueError("treatment must be binary (0/1).")
    if len(np.unique(treatment)) < 2:
        raise ValueError("treatment must contain both treated (1) and control (0) rows.")
    if not set(np.unique(outcome)).issubset({0, 1}):
        raise ValueError("outcome must be binary (0/1).")


class SLearner:
    """A single classifier with treatment as an extra feature; see module
    docstring for the counterfactual-scoring approach and its trade-offs.
    """

    def __init__(self, base_estimator: Optional[ClassifierMixin] = None) -> None:
        self.base_estimator = base_estimator if base_estimator is not None else LogisticRegression(
            max_iter=1000
        )

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> "SLearner":
        _validate_fit_inputs(X, treatment, outcome)
        self.feature_columns_ = list(X.columns)

        X_with_treatment = X.copy()
        X_with_treatment[TREATMENT_COLUMN] = np.asarray(treatment)

        self.model_ = clone(self.base_estimator)
        self.model_.fit(X_with_treatment[self.feature_columns_ + [TREATMENT_COLUMN]], outcome)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predicted P(Y=1 | X, T=1) - P(Y=1 | X, T=0) for each row."""
        if not hasattr(self, "model_"):
            raise ValueError("SLearner must be fit before predict_uplift can be called.")

        X_treated = X[self.feature_columns_].copy()
        X_treated[TREATMENT_COLUMN] = 1
        X_control = X[self.feature_columns_].copy()
        X_control[TREATMENT_COLUMN] = 0

        columns = self.feature_columns_ + [TREATMENT_COLUMN]
        p_treated = self.model_.predict_proba(X_treated[columns])[:, 1]
        p_control = self.model_.predict_proba(X_control[columns])[:, 1]
        return p_treated - p_control


class TLearner:
    """Two independent classifiers, one per treatment arm; see module
    docstring for why this avoids the S-learner's dilution risk at the cost
    of splitting the training data in two.
    """

    def __init__(self, base_estimator: Optional[ClassifierMixin] = None) -> None:
        self.base_estimator = base_estimator if base_estimator is not None else LogisticRegression(
            max_iter=1000
        )

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> "TLearner":
        _validate_fit_inputs(X, treatment, outcome)
        self.feature_columns_ = list(X.columns)

        treatment = np.asarray(treatment)
        outcome = np.asarray(outcome)
        treated_mask = treatment == 1
        control_mask = treatment == 0

        self.treated_model_ = clone(self.base_estimator)
        self.treated_model_.fit(X.loc[treated_mask, self.feature_columns_], outcome[treated_mask])

        self.control_model_ = clone(self.base_estimator)
        self.control_model_.fit(X.loc[control_mask, self.feature_columns_], outcome[control_mask])
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predicted P(Y=1 | X, treated model) - P(Y=1 | X, control
        model) for each row.
        """
        if not hasattr(self, "treated_model_"):
            raise ValueError("TLearner must be fit before predict_uplift can be called.")

        X_features = X[self.feature_columns_]
        p_treated = self.treated_model_.predict_proba(X_features)[:, 1]
        p_control = self.control_model_.predict_proba(X_features)[:, 1]
        return p_treated - p_control
