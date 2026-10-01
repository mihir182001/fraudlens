"""
Tests for src.uplift.uplift_metrics.

The hand-worked example used throughout has its qini_curve values and
qini_coefficient computed by hand; see the comments below for the
arithmetic. This mirrors tests/test_metrics.py's Week 4 precedent of exact
hand-worked values for KS/Gini rather than only checking that the
functions run.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.uplift.uplift_metrics import qini_coefficient, qini_curve, uplift_at_k

# Already sorted by descending score, so order == identity; this keeps the
# hand computation simple while still exercising the sort (ties/order are
# covered by a separate test below).
#           idx:    0  1  2  3  4  5  6  7
_OUTCOME = np.array([1, 0, 1, 0, 1, 0, 1, 0])
_TREATMENT = np.array([1, 1, 0, 0, 1, 1, 0, 0])
_SCORE = np.array([8, 7, 6, 5, 4, 3, 2, 1])

# By hand:
# cum_treated            = [1, 2, 2, 2, 3, 4, 4, 4]
# cum_control            = [0, 0, 1, 2, 2, 2, 3, 4]
# cum_treated_responders = [1, 1, 1, 1, 2, 2, 2, 2]
# cum_control_responders = [0, 0, 1, 1, 1, 1, 2, 2]
# ratio (treated/control, 0 where control==0) = [-, -, 2, 1, 1.5, 2, 4/3, 1]
# qini = treated_resp - control_resp * ratio, except where cum_control == 0
#        (use treated_resp alone there):
#   i=0: cum_control=0 -> 1
#   i=1: cum_control=0 -> 1
#   i=2: 1 - 1*2 = -1
#   i=3: 1 - 1*1 = 0
#   i=4: 2 - 1*1.5 = 0.5
#   i=5: 2 - 1*2 = 0
#   i=6: 2 - 2*(4/3) = -2/3
#   i=7: 2 - 2*1 = 0
_EXPECTED_QINI_VALUES = np.array([1.0, 1.0, -1.0, 0.0, 0.5, 0.0, -2.0 / 3.0, 0.0])
_EXPECTED_FRACTIONS = np.arange(1, 9) / 8.0

# qini_coefficient: overall_qini = qini_values[-1] = 0, so the random
# baseline is the zero line, and the raw area is exactly the trapezoidal
# area under the model's curve (through (0, 0)):
#   area = (1/8) * ((y0 + y8) / 2 + sum(y1..y7))
#        = (1/8) * ((0 + 0) / 2 + (1 + 1 - 1 + 0 + 0.5 + 0 - 2/3))
#        = (1/8) * (5/6) = 5/48
# qini_coefficient divides that raw area by n = 8: (5/48) / 8 = 5/384.
_EXPECTED_QINI_COEFFICIENT = 5.0 / 384.0


def test_qini_curve_matches_hand_worked_example():
    fractions, qini_values = qini_curve(_OUTCOME, _TREATMENT, _SCORE)
    assert fractions == pytest.approx(_EXPECTED_FRACTIONS)
    assert qini_values == pytest.approx(_EXPECTED_QINI_VALUES)


def test_qini_coefficient_matches_hand_worked_example():
    result = qini_coefficient(_OUTCOME, _TREATMENT, _SCORE)
    assert result == pytest.approx(_EXPECTED_QINI_COEFFICIENT)


def test_qini_curve_is_invariant_to_score_scale_only_order_matters():
    # Multiplying every score by a positive constant must not change the
    # ranking, and therefore must not change the curve at all.
    fractions_a, qini_a = qini_curve(_OUTCOME, _TREATMENT, _SCORE)
    fractions_b, qini_b = qini_curve(_OUTCOME, _TREATMENT, _SCORE * 100.0)
    assert fractions_a == pytest.approx(fractions_b)
    assert qini_a == pytest.approx(qini_b)


def test_uplift_at_k_top_half_matches_hand_computation():
    # Top 4 rows by score: indices 0-3. treatment=[1,1,0,0], outcome=[1,0,1,0].
    # treated_rate = mean([1, 0]) = 0.5; control_rate = mean([1, 0]) = 0.5.
    result = uplift_at_k(_OUTCOME, _TREATMENT, _SCORE, k=0.5)
    assert result == pytest.approx(0.0)


def test_uplift_at_k_top_quarter_raises_when_one_group_missing():
    # Top 2 rows by score: indices 0-1, both treatment == 1. No control rows
    # in the slice, so the rate difference is undefined.
    with pytest.raises(ValueError, match="only one treatment group"):
        uplift_at_k(_OUTCOME, _TREATMENT, _SCORE, k=0.25)


def test_uplift_at_k_full_population():
    # All 8 rows: treated=[idx0,1,4,5] outcomes=[1,0,1,0] -> rate 0.5.
    # control=[idx2,3,6,7] outcomes=[1,0,1,0] -> rate 0.5. uplift = 0.
    result = uplift_at_k(_OUTCOME, _TREATMENT, _SCORE, k=1.0)
    assert result == pytest.approx(0.0)


def test_uplift_at_k_rejects_k_out_of_range():
    with pytest.raises(ValueError, match="k must be in"):
        uplift_at_k(_OUTCOME, _TREATMENT, _SCORE, k=0.0)
    with pytest.raises(ValueError, match="k must be in"):
        uplift_at_k(_OUTCOME, _TREATMENT, _SCORE, k=1.5)


def test_random_ranking_has_qini_coefficient_near_zero():
    # A score with no relationship to outcome or treatment should score
    # close to zero on a large enough sample (random targeting baseline).
    rng = np.random.default_rng(0)
    n = 20000
    treatment = rng.integers(0, 2, size=n)
    outcome = rng.integers(0, 2, size=n)
    random_score = rng.normal(size=n)
    result = qini_coefficient(outcome, treatment, random_score)
    assert abs(result) < 0.01


def test_perfect_ranking_beats_random_ranking():
    # A score that ranks true persuadables (treatment helps) above everyone
    # else should score a clearly positive qini_coefficient, well above the
    # near-zero random baseline.
    rng = np.random.default_rng(1)
    n = 5000
    treatment = rng.integers(0, 2, size=n)
    # First half of customers are true persuadables: p0=0.1, p1=0.6.
    # Second half are true nulls: p0=p1=0.3.
    is_persuadable = np.arange(n) < n // 2
    p0 = np.where(is_persuadable, 0.1, 0.3)
    p1 = np.where(is_persuadable, 0.6, 0.3)
    conversion_prob = np.where(treatment == 1, p1, p0)
    outcome = (rng.random(n) < conversion_prob).astype(int)

    perfect_score = is_persuadable.astype(float)
    result = qini_coefficient(outcome, treatment, perfect_score)
    assert result > 0.02


def test_validate_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        qini_curve(_OUTCOME, _TREATMENT, _SCORE[:-1])


def test_validate_rejects_non_binary_outcome():
    bad_outcome = _OUTCOME.copy()
    bad_outcome[0] = 2
    with pytest.raises(ValueError, match="outcome must be binary"):
        qini_curve(bad_outcome, _TREATMENT, _SCORE)


def test_validate_rejects_treatment_with_only_one_group():
    all_treated = np.ones_like(_TREATMENT)
    with pytest.raises(ValueError, match="both treated"):
        qini_curve(_OUTCOME, all_treated, _SCORE)


def test_validate_rejects_empty_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        qini_curve(np.array([]), np.array([]), np.array([]))
