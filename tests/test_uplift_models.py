"""
Tests for src.uplift.uplift_models.

The central test for each learner (test_*_recovers_segment_uplift_ordering)
uses a hand-built dataset where a single feature deterministically reveals
which of two true segments a row belongs to: a "persuadable"-like segment
(large positive true effect) and a "sleeping_dog"-like segment (negative
true effect). A learner that is wired correctly must recover that ordering
in its predicted uplift; this is the same "does it recover a known,
designed ground truth" discipline used throughout this project (Week 5's
ring-only graph test, Week 1's synthetic fraud rate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.uplift.uplift_models import SLearner, TLearner


def _two_segment_frame(n_per_segment: int = 1000, seed: int = 0):
    """group == 1 is a strong persuadable (p0=0.1, p1=0.6, true uplift
    +0.5); group == 0 is a sleeping dog (p0=0.5, p1=0.2, true uplift -0.3).
    "signal" reveals the group exactly; "noise" carries no information at
    all, so a learner that is fitting real signal, not memorizing rows,
    should give it near-zero importance in the resulting ranking.
    """
    rng = np.random.default_rng(seed)
    n = n_per_segment * 2
    group = np.array([1] * n_per_segment + [0] * n_per_segment)
    rng.shuffle(group)

    signal = group.astype(float) + rng.normal(scale=0.01, size=n)
    noise = rng.normal(size=n)
    X = pd.DataFrame({"signal": signal, "noise": noise})

    treatment = rng.integers(0, 2, size=n)
    p0 = np.where(group == 1, 0.1, 0.5)
    p1 = np.where(group == 1, 0.6, 0.2)
    conversion_prob = np.where(treatment == 1, p1, p0)
    outcome = (rng.random(n) < conversion_prob).astype(int)

    return X, treatment, outcome, group


def test_slearner_recovers_segment_uplift_ordering():
    # A plain LogisticRegression S-learner cannot recover this (see
    # test_slearner_with_additive_base_estimator_cannot_recover_heterogeneous_effect
    # below and the module docstring), because it has no way to let
    # treatment interact with "signal": its treatment coefficient is one
    # number shared by every row. A base estimator that can express
    # interactions on its own, such as a decision tree, is what an
    # S-learner needs to find heterogeneous effects.
    from sklearn.tree import DecisionTreeClassifier

    X, treatment, outcome, group = _two_segment_frame(seed=1)
    model = SLearner(base_estimator=DecisionTreeClassifier(max_depth=4, random_state=0)).fit(
        X, treatment, outcome
    )
    uplift = model.predict_uplift(X)

    assert uplift[group == 1].mean() > 0.2
    assert uplift[group == 0].mean() < 0.0
    assert uplift[group == 1].mean() > uplift[group == 0].mean()


def test_slearner_with_additive_base_estimator_cannot_recover_heterogeneous_effect():
    # Documents a real, confirmed limitation (see module docstring): a
    # LogisticRegression S-learner adds treatment as one more additive
    # log-odds term with no interaction against the other features, so it
    # can only learn a single, population-average treatment effect. On this
    # dataset's true, strongly heterogeneous effect (+0.5 for group 1, -0.3
    # for group 0), it predicts nearly the same small positive uplift for
    # both groups instead of recovering the true opposite-signed effects.
    X, treatment, outcome, group = _two_segment_frame(seed=1)
    model = SLearner().fit(X, treatment, outcome)
    uplift = model.predict_uplift(X)

    difference_between_groups = abs(uplift[group == 1].mean() - uplift[group == 0].mean())
    assert difference_between_groups < 0.05
    # It still lands somewhere near the true population-average effect
    # rather than at an arbitrary value.
    assert 0.0 < uplift.mean() < 0.2


def test_tlearner_recovers_segment_uplift_ordering():
    X, treatment, outcome, group = _two_segment_frame(seed=2)
    model = TLearner().fit(X, treatment, outcome)
    uplift = model.predict_uplift(X)

    assert uplift[group == 1].mean() > 0.2
    assert uplift[group == 0].mean() < 0.0
    assert uplift[group == 1].mean() > uplift[group == 0].mean()


def test_predict_uplift_shape_matches_input_rows():
    X, treatment, outcome, _ = _two_segment_frame(n_per_segment=50, seed=3)
    for model in (SLearner().fit(X, treatment, outcome), TLearner().fit(X, treatment, outcome)):
        assert model.predict_uplift(X).shape == (len(X),)


def test_slearner_and_tlearner_are_reproducible_with_deterministic_base_estimator():
    X, treatment, outcome, _ = _two_segment_frame(n_per_segment=50, seed=4)
    for cls in (SLearner, TLearner):
        uplift_a = cls().fit(X, treatment, outcome).predict_uplift(X)
        uplift_b = cls().fit(X, treatment, outcome).predict_uplift(X)
        assert uplift_a == pytest.approx(uplift_b)


def test_predict_uplift_before_fit_raises():
    X, _, _, _ = _two_segment_frame(n_per_segment=10, seed=5)
    with pytest.raises(ValueError, match="must be fit"):
        SLearner().predict_uplift(X)
    with pytest.raises(ValueError, match="must be fit"):
        TLearner().predict_uplift(X)


def test_fit_rejects_non_dataframe_X():
    _, treatment, outcome, _ = _two_segment_frame(n_per_segment=10, seed=6)
    with pytest.raises(ValueError, match="DataFrame"):
        SLearner().fit(np.zeros((len(treatment), 2)), treatment, outcome)


def test_fit_rejects_reserved_treatment_column_name():
    X, treatment, outcome, _ = _two_segment_frame(n_per_segment=10, seed=7)
    X_bad = X.copy()
    X_bad["__treatment__"] = 0
    with pytest.raises(ValueError, match="reserved"):
        SLearner().fit(X_bad, treatment, outcome)


def test_fit_rejects_treatment_with_only_one_group():
    X, treatment, outcome, _ = _two_segment_frame(n_per_segment=10, seed=8)
    all_treated = np.ones_like(treatment)
    with pytest.raises(ValueError, match="both treated"):
        TLearner().fit(X, all_treated, outcome)


def test_fit_rejects_non_binary_outcome():
    X, treatment, outcome, _ = _two_segment_frame(n_per_segment=10, seed=9)
    bad_outcome = outcome.copy()
    bad_outcome[0] = 2
    with pytest.raises(ValueError, match="outcome must be binary"):
        SLearner().fit(X, treatment, bad_outcome)


def test_fit_rejects_mismatched_lengths():
    X, treatment, outcome, _ = _two_segment_frame(n_per_segment=10, seed=10)
    with pytest.raises(ValueError, match="same length"):
        SLearner().fit(X, treatment[:-1], outcome)


def test_custom_base_estimator_is_used():
    from sklearn.tree import DecisionTreeClassifier

    X, treatment, outcome, _ = _two_segment_frame(n_per_segment=50, seed=11)
    model = TLearner(base_estimator=DecisionTreeClassifier(random_state=0)).fit(
        X, treatment, outcome
    )
    assert isinstance(model.treated_model_, DecisionTreeClassifier)
    assert isinstance(model.control_model_, DecisionTreeClassifier)
