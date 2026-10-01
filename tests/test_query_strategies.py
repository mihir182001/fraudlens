"""
Tests for src.active_learning.query_strategies.

test_binary_strategies_produce_identical_rankings checks a fact the module
docstring states as a mathematical guarantee, not an empirical tendency:
for binary classification, least_confidence, margin, and entropy all rank
the same set of rows identically. If this test ever failed, it would mean
one of the three implementations has a real bug, not that the claim itself
was ever approximate.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.active_learning.query_strategies import (
    entropy_score,
    least_confidence_score,
    margin_score,
    select_query_batch,
)

# Hand-worked example. Rows, as [P(class0), P(class1)]:
#   row 0: [0.5, 0.5]  -> maximally uncertain
#   row 1: [0.9, 0.1]  -> confident in class 0
#   row 2: [0.5, 0.5]  -> tied with row 0
#   row 3: [0.0, 1.0]  -> completely certain
#   row 4: [0.7, 0.3]  -> moderately confident
_PROBABILITIES = np.array([
    [0.5, 0.5],
    [0.9, 0.1],
    [0.5, 0.5],
    [0.0, 1.0],
    [0.7, 0.3],
])


def test_least_confidence_score_matches_hand_computation():
    # 1 - max(p): row0=0.5, row1=0.1, row2=0.5, row3=0.0, row4=0.3
    result = least_confidence_score(_PROBABILITIES)
    assert result == pytest.approx([0.5, 0.1, 0.5, 0.0, 0.3])


def test_margin_score_matches_hand_computation():
    # 1 - (top - second): row0=1-(0.5-0.5)=1.0, row1=1-(0.9-0.1)=0.2,
    # row2=1.0, row3=1-(1.0-0.0)=0.0, row4=1-(0.7-0.3)=0.6
    result = margin_score(_PROBABILITIES)
    assert result == pytest.approx([1.0, 0.2, 1.0, 0.0, 0.6])


def test_entropy_score_matches_hand_computation():
    # -sum(p*log p): row0 = -2*(0.5*log(0.5)) = log(2) ~ 0.6931
    # row3 = 0 (both terms have a zero factor)
    ln2 = np.log(2.0)
    expected_row0 = -2 * (0.5 * np.log(0.5))
    result = entropy_score(_PROBABILITIES)
    assert result[0] == pytest.approx(expected_row0)
    assert result[0] == pytest.approx(ln2)
    assert result[3] == pytest.approx(0.0)


def test_binary_strategies_produce_identical_rankings():
    # See module docstring: for binary classification this is guaranteed,
    # not just usually true. Checked here against 1000 random probability
    # vectors, not just the small hand-worked example above.
    rng = np.random.default_rng(0)
    p1 = rng.random(1000)
    probabilities = np.column_stack([1 - p1, p1])

    order_lc = np.argsort(-least_confidence_score(probabilities), kind="stable")
    order_margin = np.argsort(-margin_score(probabilities), kind="stable")
    order_entropy = np.argsort(-entropy_score(probabilities), kind="stable")

    assert np.array_equal(order_lc, order_margin)
    assert np.array_equal(order_lc, order_entropy)


def test_select_query_batch_returns_most_uncertain_rows_first():
    # Rows 0 and 2 are tied for most uncertain, then row 4, then row 1,
    # then row 3.
    batch = select_query_batch(_PROBABILITIES, batch_size=3, strategy="entropy")
    assert set(batch[:2].tolist()) == {0, 2}
    assert batch[2] == 4


def test_ties_are_broken_by_original_order():
    # Rows 0 and 2 are exactly tied on every strategy; row 0 (the earlier
    # index) must come first in the returned batch.
    batch = select_query_batch(_PROBABILITIES, batch_size=2, strategy="entropy")
    assert list(batch) == [0, 2]


def test_select_query_batch_clips_batch_size_to_available_rows():
    batch = select_query_batch(_PROBABILITIES, batch_size=100, strategy="entropy")
    assert len(batch) == len(_PROBABILITIES)


def test_select_query_batch_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        select_query_batch(_PROBABILITIES, batch_size=2, strategy="not_a_real_strategy")


def test_select_query_batch_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        select_query_batch(_PROBABILITIES, batch_size=0, strategy="entropy")


def test_validate_rejects_rows_not_summing_to_one():
    bad = np.array([[0.5, 0.6]])
    with pytest.raises(ValueError, match="sum to 1"):
        least_confidence_score(bad)


def test_validate_rejects_values_outside_zero_one():
    bad = np.array([[1.5, -0.5]])
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        margin_score(bad)


def test_validate_rejects_one_dimensional_input():
    with pytest.raises(ValueError, match="2D array"):
        entropy_score(np.array([0.5, 0.5]))


def test_validate_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        entropy_score(np.zeros((0, 2)))


def test_multiclass_probabilities_are_supported():
    # A 3-class row where classes 0 and 1 are tied for most likely and
    # class 2 is ruled out entirely.
    probabilities = np.array([[0.45, 0.45, 0.10]])
    lc = least_confidence_score(probabilities)
    margin = margin_score(probabilities)
    assert lc[0] == pytest.approx(0.55)
    assert margin[0] == pytest.approx(1.0)
