"""
Tests for src.models.metrics.

Every metric here has at least one test built from values worked out by
hand (not just checked against sklearn's own functions), so a passing test
confirms the actual numbers, not just that this module calls sklearn the
same way sklearn would call itself.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.models.metrics import auc_pr, gini_coefficient, ks_statistic, optimal_threshold_by_f1


def test_ks_statistic_exact_known_value():
    # Sorted by score descending already: (0.9, pos), (0.8, neg), (0.6,
    # pos), (0.2, neg). Two positives, two negatives.
    # cumulative positive rate after each row: 0.5, 0.5, 1.0, 1.0
    # cumulative negative rate after each row: 0.0, 0.5, 0.5, 1.0
    # gap:                                     0.5, 0.0, 0.5, 0.0
    # Largest gap is 0.5, first reached at the row scored 0.9.
    y_true = [1, 0, 1, 0]
    y_proba = [0.9, 0.8, 0.6, 0.2]
    result = ks_statistic(y_true, y_proba)
    assert result["ks_statistic"] == pytest.approx(0.5)
    assert result["ks_threshold"] == pytest.approx(0.9)


def test_ks_statistic_perfect_separation_is_one():
    y_true = [1, 1, 0, 0]
    y_proba = [0.9, 0.8, 0.2, 0.1]
    result = ks_statistic(y_true, y_proba)
    assert result["ks_statistic"] == pytest.approx(1.0)


def test_ks_statistic_requires_both_classes():
    with pytest.raises(ValueError, match="only one class"):
        ks_statistic([1, 1, 1], [0.9, 0.8, 0.7])


def test_ks_statistic_requires_matching_lengths():
    with pytest.raises(ValueError, match="same length"):
        ks_statistic([1, 0], [0.9, 0.8, 0.1])


def test_gini_coefficient_matches_known_auc_relationship():
    y_true = [1, 0, 1, 0]
    y_proba = [0.9, 0.8, 0.6, 0.2]
    # Worked out by hand: the two positives are ranked (0.9, 0.6) and the
    # two negatives (0.8, 0.2). Of the four positive-negative pairs, three
    # rank the positive above the negative (0.9>0.8, 0.9>0.2, 0.6>0.2) and
    # one does not (0.6<0.8), giving AUC-ROC = 3/4 = 0.75.
    auc = roc_auc_score(y_true, y_proba)
    assert auc == pytest.approx(0.75)
    assert gini_coefficient(y_true, y_proba) == pytest.approx(2 * 0.75 - 1)


def test_gini_coefficient_is_zero_for_random_ranking():
    # A model that ranks positives and negatives with no separation at all
    # (every score tied) has AUC-ROC = 0.5, so Gini = 0.
    y_true = [1, 0, 1, 0]
    y_proba = [0.5, 0.5, 0.5, 0.5]
    assert gini_coefficient(y_true, y_proba) == pytest.approx(0.0)


def test_gini_coefficient_requires_both_classes():
    with pytest.raises(ValueError, match="only one class"):
        gini_coefficient([0, 0, 0], [0.1, 0.2, 0.3])


def test_auc_pr_matches_sklearn_average_precision_directly():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=50)
    y_proba = rng.random(size=50)
    # This confirms the wrapper passes its inputs through unchanged, so a
    # future refactor of this module cannot silently drift from sklearn's
    # own definition without a test catching it.
    assert auc_pr(y_true, y_proba) == pytest.approx(average_precision_score(y_true, y_proba))


def test_auc_pr_is_higher_for_a_clearly_separable_case():
    y_true = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # 20 percent positive rate
    good_scores = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    random_scores = [0.5] * 10
    assert auc_pr(y_true, good_scores) > auc_pr(y_true, random_scores)


def test_auc_pr_requires_both_classes():
    with pytest.raises(ValueError, match="only one class"):
        auc_pr([1, 1, 1], [0.9, 0.8, 0.7])


def test_optimal_threshold_by_f1_matches_hand_worked_sklearn_example():
    # This is scikit-learn's own precision_recall_curve documentation
    # example, chosen because its precision/recall/threshold values are
    # independently published and easy to verify by hand:
    # precision = [0.5, 0.66666667, 0.5, 1., 1.]
    # recall    = [1.,  1.,         0.5, 0.5, 0.]
    # thresholds= [0.1, 0.35,       0.4, 0.8]
    # F1 at each threshold: 0.6667, 0.8, 0.5, 0.6667. The maximum is 0.8 at
    # threshold 0.35 (precision 2/3, recall 1.0).
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.4, 0.35, 0.8]
    result = optimal_threshold_by_f1(y_true, y_proba)
    assert result["threshold"] == pytest.approx(0.35)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(0.8)


def test_optimal_threshold_by_f1_requires_both_classes():
    with pytest.raises(ValueError, match="only one class"):
        optimal_threshold_by_f1([0, 0, 0], [0.1, 0.2, 0.3])
