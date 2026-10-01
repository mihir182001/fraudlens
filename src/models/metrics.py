"""
metrics.py, Week 4: the credit-risk-style metrics the spec adds on top of
plain AUC-ROC, the KS statistic, the Gini coefficient, and AUC-PR, plus a
threshold optimizer, since a probability score alone is not a decision.

AUC-ROC (used throughout Week 3) treats false positives and false
negatives symmetrically and is insensitive to class balance, which is why
credit and fraud teams also report these three numbers:

KS statistic (Kolmogorov-Smirnov): the maximum gap between the cumulative
distribution of scores for the positive class and the negative class, when
both are sorted from the highest predicted score to the lowest. This is
the classic credit-scoring definition, not scipy's two-sample test, since
this project's audience (Zopa, Amex, Citi, JPMorgan-style credit risk
teams) reports it this way; it also comes with a natural companion, the
score threshold where that maximum gap occurs, which doubles as one
reasonable operating threshold.

Gini coefficient: 2 * AUC-ROC minus 1, the standard credit-scoring
transformation of AUC-ROC onto a scale where 0 is a random model and 1 is
a perfect one, matching how AUC-ROC is usually reported in that industry.

AUC-PR (area under the precision-recall curve, also called average
precision): unlike AUC-ROC, this is sensitive to class imbalance, so it
tends to look much lower than AUC-ROC on a rare-event problem like fraud,
which is itself useful information, not a bug in the number.

Threshold optimization: every model above returns a continuous score, but
a real system has to decide, at some point, to flag a transaction or not.
optimal_threshold_by_f1 picks the score cutoff that maximizes F1 (the
harmonic mean of precision and recall) on the data it is given, so it must
be called on a validation or test split, never on the same split used to
report the final metric, the same train/validation separation discipline
used throughout this project.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _validate_binary_inputs(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    """Shared validation for the metrics below: both classes must be
    present, or the metric they compute is undefined, and y_true/y_proba
    must be the same length.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if len(y_true) != len(y_proba):
        raise ValueError(
            f"y_true and y_proba must be the same length, got {len(y_true)} "
            f"and {len(y_proba)}."
        )
    if len(y_true) == 0:
        raise ValueError("y_true and y_proba must not be empty.")
    unique_labels = set(np.unique(y_true).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"y_true must be binary (0/1), got values {sorted(unique_labels)}.")
    if len(unique_labels) < 2:
        raise ValueError(
            "y_true has only one class present; this metric is undefined "
            "without both classes."
        )


def ks_statistic(y_true, y_proba) -> Dict[str, float]:
    """Returns the KS statistic and the score threshold where it occurs.

    Sorts rows by predicted score from highest to lowest, then tracks what
    fraction of all positives and what fraction of all negatives have been
    "captured" so far at each point in that ranking. The KS statistic is
    the largest gap between those two running fractions; ks_threshold is
    the score at the row where that gap is largest, a natural candidate
    cutoff since it is the point where the model best separates the two
    classes.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    _validate_binary_inputs(y_true, y_proba)

    order = pd.DataFrame({"y": y_true, "p": y_proba}).sort_values(
        "p", ascending=False, kind="mergesort"
    ).reset_index(drop=True)

    total_pos = order["y"].sum()
    total_neg = len(order) - total_pos

    cum_pos_rate = order["y"].cumsum() / total_pos
    cum_neg_rate = (1 - order["y"]).cumsum() / total_neg
    gap = (cum_pos_rate - cum_neg_rate).abs()

    best_index = int(gap.idxmax())
    return {
        "ks_statistic": float(gap.iloc[best_index]),
        "ks_threshold": float(order["p"].iloc[best_index]),
    }


def gini_coefficient(y_true, y_proba) -> float:
    """Returns the Gini coefficient, 2 * AUC-ROC minus 1.

    This is a linear rescaling of AUC-ROC, not an independent measurement,
    so it will always move in lockstep with the AUC-ROC this project has
    already been reporting since Week 3; it is included because credit
    risk teams conventionally report Gini rather than, or alongside,
    AUC-ROC directly.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    _validate_binary_inputs(y_true, y_proba)
    return 2.0 * roc_auc_score(y_true, y_proba) - 1.0


def auc_pr(y_true, y_proba) -> float:
    """Returns the area under the precision-recall curve (average
    precision).

    A thin, named wrapper around scikit-learn's average_precision_score,
    kept as its own function so every metric this project reports for a
    model comes from this one module with a consistent validation and
    naming convention, rather than some being sklearn calls made ad hoc at
    the call site.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    _validate_binary_inputs(y_true, y_proba)
    return float(average_precision_score(y_true, y_proba))


def optimal_threshold_by_f1(y_true, y_proba) -> Dict[str, float]:
    """Returns the score threshold that maximizes F1 on the data given,
    along with the precision, recall, and F1 at that threshold.

    Must be called on a validation or test split, not on the same split a
    final metric will be reported on, the same discipline this project has
    followed for every fit-based step since Week 3.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    _validate_binary_inputs(y_true, y_proba)

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision and recall have one more entry than thresholds (the final
    # point corresponds to no threshold at all, i.e. flagging nothing), so
    # only the first len(thresholds) entries of each pair with a real cutoff.
    precision = precision[: len(thresholds)]
    recall = recall[: len(thresholds)]

    denominator = precision + recall
    f1_scores = np.where(denominator > 0, 2 * precision * recall / np.where(denominator == 0, 1, denominator), 0.0)

    best_index = int(np.argmax(f1_scores))
    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(f1_scores[best_index]),
    }
