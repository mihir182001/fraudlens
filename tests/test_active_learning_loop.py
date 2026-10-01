"""
Tests for src.active_learning.active_learning_loop.

test_entropy_strategy_beats_random_strategy is this module's central claim:
on a dataset with a genuine, learnable, imbalanced decision boundary,
uncertainty sampling should reach better test AUC-PR than random sampling
at the same labeling budget, and should also naturally surface more of the
rare positive class along the way, without ever using the label to decide
what to query. Both are checked against a hand-built logistic dataset
(seed fixed, verified directly before writing this test) rather than
asserted on faith.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.active_learning.active_learning_loop import run_active_learning_simulation

FEATURE_COLUMNS = ["f1", "f2"]


def _imbalanced_logistic_frame(n_total: int = 6000, n_test: int = 1000, seed: int = 7):
    """A genuinely learnable, imbalanced (about 15% positive) binary
    classification dataset: isFraud follows a real logistic relationship in
    f1/f2, not noise, so a model that sees more informative rows should
    measurably improve.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n_total)
    x2 = rng.normal(size=n_total)
    logit = 1.5 * x1 - 1.0 * x2 - 2.5
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n_total) < p).astype(int)
    df = pd.DataFrame({"f1": x1, "f2": x2, "isFraud": y})

    train_df = df.iloc[: n_total - n_test].reset_index(drop=True)
    test_df = df.iloc[n_total - n_test :].reset_index(drop=True)
    return train_df, test_df


def test_entropy_strategy_beats_random_strategy_on_final_round_auc_pr():
    train_df, test_df = _imbalanced_logistic_frame()
    common_kwargs = dict(
        train_df=train_df, test_df=test_df, feature_columns=FEATURE_COLUMNS,
        n_initial_labeled=30, query_batch_size=20, n_rounds=8, random_state=42,
    )
    entropy_history = run_active_learning_simulation(strategy="entropy", **common_kwargs)
    random_history = run_active_learning_simulation(strategy="random", **common_kwargs)

    # Round 0 uses the identical initial labeled set for both strategies
    # (the strategy choice only affects which rows get queried afterward),
    # so it must match exactly.
    assert entropy_history[0]["n_labeled"] == random_history[0]["n_labeled"]
    assert entropy_history[0]["test_auc_pr"] == pytest.approx(random_history[0]["test_auc_pr"])

    assert entropy_history[-1]["test_auc_pr"] > random_history[-1]["test_auc_pr"]


def test_entropy_strategy_surfaces_more_fraud_labels_than_random():
    # A secondary, intuitive finding: uncertainty sampling naturally
    # over-represents the minority class relative to random sampling on
    # imbalanced data, purely as a side effect of fraud rows disproportio-
    # nately sitting near the decision boundary, never by looking at labels
    # to decide what to query.
    train_df, test_df = _imbalanced_logistic_frame()
    common_kwargs = dict(
        train_df=train_df, test_df=test_df, feature_columns=FEATURE_COLUMNS,
        n_initial_labeled=30, query_batch_size=20, n_rounds=8, random_state=42,
    )
    entropy_history = run_active_learning_simulation(strategy="entropy", **common_kwargs)
    random_history = run_active_learning_simulation(strategy="random", **common_kwargs)

    assert entropy_history[-1]["n_labeled_fraud"] > random_history[-1]["n_labeled_fraud"]


def test_simulation_is_reproducible_given_same_random_state():
    train_df, test_df = _imbalanced_logistic_frame(n_total=1000, n_test=200)
    kwargs = dict(
        train_df=train_df, test_df=test_df, feature_columns=FEATURE_COLUMNS,
        strategy="entropy", n_initial_labeled=30, query_batch_size=20,
        n_rounds=4, random_state=5,
    )
    history_a = run_active_learning_simulation(**kwargs)
    history_b = run_active_learning_simulation(**kwargs)
    assert history_a == history_b


def test_history_has_one_entry_per_round_plus_initial():
    train_df, test_df = _imbalanced_logistic_frame(n_total=1000, n_test=200)
    history = run_active_learning_simulation(
        train_df, test_df, FEATURE_COLUMNS, n_initial_labeled=30,
        query_batch_size=20, n_rounds=5, random_state=1,
    )
    assert len(history) == 6
    assert [h["round"] for h in history] == list(range(6))
    assert [h["n_labeled"] for h in history] == [30, 50, 70, 90, 110, 130]


def test_loop_stops_early_when_pool_is_exhausted():
    rng = np.random.default_rng(1)
    n = 150
    train_df = pd.DataFrame({
        "f1": rng.normal(size=n), "isFraud": (rng.random(n) < 0.3).astype(int),
    })
    test_df = pd.DataFrame({
        "f1": rng.normal(size=50), "isFraud": (rng.random(50) < 0.3).astype(int),
    })

    history = run_active_learning_simulation(
        train_df, test_df, ["f1"], strategy="entropy", n_initial_labeled=20,
        query_batch_size=30, n_rounds=10, random_state=3,
    )
    # The pool (130 rows) is exhausted well before 10 rounds of 30 each.
    assert len(history) < 11
    assert history[-1]["n_labeled"] == len(train_df)


def test_rejects_non_positive_n_initial_labeled():
    train_df, test_df = _imbalanced_logistic_frame(n_total=200, n_test=50)
    with pytest.raises(ValueError, match="n_initial_labeled must be positive"):
        run_active_learning_simulation(train_df, test_df, FEATURE_COLUMNS, n_initial_labeled=0)


def test_rejects_non_positive_query_batch_size():
    train_df, test_df = _imbalanced_logistic_frame(n_total=200, n_test=50)
    with pytest.raises(ValueError, match="query_batch_size must be positive"):
        run_active_learning_simulation(
            train_df, test_df, FEATURE_COLUMNS, n_initial_labeled=30, query_batch_size=0,
        )


def test_rejects_n_initial_labeled_larger_than_train_df():
    train_df, test_df = _imbalanced_logistic_frame(n_total=200, n_test=50)
    with pytest.raises(ValueError, match="cannot exceed"):
        run_active_learning_simulation(
            train_df, test_df, FEATURE_COLUMNS, n_initial_labeled=len(train_df) + 1,
        )


def test_rejects_labeled_set_with_only_one_class():
    n = 100
    train_df = pd.DataFrame({"f1": np.arange(n, dtype=float), "isFraud": [0] * n})
    test_df = pd.DataFrame({"f1": [1.0, 2.0], "isFraud": [0, 1]})
    with pytest.raises(ValueError, match="only one class"):
        run_active_learning_simulation(
            train_df, test_df, ["f1"], n_initial_labeled=10, query_batch_size=5, n_rounds=1,
        )
