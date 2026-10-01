"""
active_learning_loop.py, Week 7: pool-based active learning simulation for
FraudLens's transaction fraud model.

This simulates the real workflow a fraud team faces: a small set of
already-labeled transactions to start from, and a much larger pool of
transactions whose true fraud status is only found out by having a human
investigator review them, an expensive, limited resource. Instead of
picking which transactions to send to investigators at random, active
learning uses the CURRENT model's own uncertainty (src/active_learning/
query_strategies.py) to prioritize the transactions it is least sure about:
a transaction the model already scores near 0 or 1 confirms what it already
believes, while one scored near the decision boundary is where a label does
the most to correct that boundary.

Every simulation round: (1) fits build_pipeline("logistic_regression") on
the current labeled set, reusing Week 3's exact preprocessing so this
module's numbers are directly comparable to Week 3/4's own results on the
same data; (2) scores that round's model on the one FIXED, held-out test
set, never touched by any query strategy, so every round's reported
test AUC-ROC/AUC-PR is a fair, apples-to-apples comparison across rounds
and across strategies; (3) scores the remaining pool and selects the next
batch to "label" via the chosen query strategy, or purely at random for the
baseline every real strategy is compared against; (4) reveals those rows'
already-known true labels and moves them from the pool into the labeled
set. The labels being "already known" the whole time is what makes this a
simulation rather than a real deployment: the loop is careful to use them
only at this reveal step, never to influence which rows the query strategy
picks, exactly the same discipline a real held-out test label gets
throughout this project.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.active_learning.query_strategies import select_query_batch
from src.models.baseline_models import build_pipeline
from src.models.metrics import auc_pr


def run_active_learning_simulation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    target_col: str = "isFraud",
    strategy: str = "entropy",
    n_initial_labeled: int = 200,
    query_batch_size: int = 100,
    n_rounds: int = 10,
    random_state: int = 42,
) -> List[Dict]:
    """Runs one pool-based active learning simulation, drawing the initial
    labeled set and unlabeled pool from train_df and evaluating every
    round's model against the fixed test_df.

    strategy is "entropy", "least_confidence", or "margin" (see
    query_strategies.py; for this project's binary isFraud target, these
    three are mathematically equivalent, see that module's docstring), or
    "random", which selects the next batch uniformly at random from the
    remaining pool instead of by predicted uncertainty. "random" is the
    baseline every real strategy in this project is compared against.

    Returns a list with one dict per round (n_rounds + 1 entries, rounds 0
    through n_rounds), each with round, n_labeled, n_labeled_fraud,
    test_auc_roc, and test_auc_pr. The loop stops early, returning fewer
    than n_rounds + 1 entries, if the pool is exhausted first.

    Raises ValueError if n_initial_labeled or query_batch_size is not
    positive, if n_initial_labeled exceeds len(train_df), or if any round's
    labeled set ends up with only one class present (a real risk with
    fraud's severe class imbalance and a small n_initial_labeled): AUC is
    undefined without both classes, so this is raised rather than silently
    producing a meaningless number.
    """
    if n_initial_labeled <= 0:
        raise ValueError("n_initial_labeled must be positive.")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive.")
    if n_initial_labeled > len(train_df):
        raise ValueError("n_initial_labeled cannot exceed the size of train_df.")

    rng = np.random.default_rng(random_state)
    shuffled_index = rng.permutation(train_df.index.to_numpy())

    labeled_index = list(shuffled_index[:n_initial_labeled])
    pool_index = list(shuffled_index[n_initial_labeled:])

    x_test = test_df[feature_columns]
    y_test = test_df[target_col].to_numpy()

    history: List[Dict] = []

    for round_number in range(n_rounds + 1):
        labeled_df = train_df.loc[labeled_index]
        y_labeled = labeled_df[target_col]

        if y_labeled.nunique() < 2:
            raise ValueError(
                f"Round {round_number}: the labeled set has only one class "
                f"present among {len(labeled_df)} rows; AUC is undefined. "
                "Try a larger n_initial_labeled or a different random_state."
            )

        pipeline = build_pipeline("logistic_regression")
        pipeline.fit(labeled_df[feature_columns], y_labeled)

        test_proba = pipeline.predict_proba(x_test)[:, 1]
        history.append({
            "round": round_number,
            "n_labeled": len(labeled_index),
            "n_labeled_fraud": int(y_labeled.sum()),
            "test_auc_roc": float(roc_auc_score(y_test, test_proba)),
            "test_auc_pr": auc_pr(y_test, test_proba),
        })

        if round_number == n_rounds or not pool_index:
            break

        pool_df = train_df.loc[pool_index]
        batch_size = min(query_batch_size, len(pool_index))

        if strategy == "random":
            chosen_positions = rng.choice(len(pool_index), size=batch_size, replace=False)
        else:
            pool_proba = pipeline.predict_proba(pool_df[feature_columns])
            chosen_positions = select_query_batch(pool_proba, batch_size, strategy=strategy)

        chosen_positions_set = set(int(p) for p in chosen_positions)
        chosen_index = [pool_index[i] for i in chosen_positions]
        labeled_index.extend(chosen_index)
        pool_index = [idx for i, idx in enumerate(pool_index) if i not in chosen_positions_set]

    return history
