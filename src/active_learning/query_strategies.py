"""
query_strategies.py, Week 7: uncertainty measures used to decide which
unlabeled transactions are most worth sending to a human investigator next.

Active learning's central idea is that not every unlabeled example is
equally worth labeling. A transaction the current model already scores at
0.01 or 0.99 fraud probability confirms what the model already believes;
one it scores near 0.5 is exactly where the model's decision boundary is
uncertain, and getting that one label corrected does the most to move the
boundary. All three strategies below turn "how uncertain is the model about
this row" into a single per-row number, higher meaning more worth labeling.

These are computed from a full probability distribution over classes
(shape (n_samples, n_classes)) so they generalize beyond this project's
binary fraud/not-fraud task, matching how they are defined in the active
learning literature:
    - least_confidence_score: 1 - max_class_probability. Zero when the
      model is completely sure of one class; largest when it is not.
    - margin_score: 1 - (top class probability - second-place class
      probability). Small margin between the top two candidates means the
      model is genuinely torn between them.
    - entropy_score: -sum(p * log(p)) over classes, the standard
      information-theoretic uncertainty measure. Zero when one class has
      probability 1; largest when probability is spread evenly.

A fact worth stating plainly rather than leaving implicit: for BINARY
classification specifically (this project's isFraud task), all three
strategies are mathematically guaranteed to rank samples identically. Every
one of them is a strictly decreasing function of |p - 0.5| where p is the
predicted probability of the positive class:
    least_confidence = 1 - max(p, 1-p) = 0.5 - |p - 0.5|
    margin           = 1 - |p - (1-p)| = 1 - 2|p - 0.5|
    entropy          = -(p*log(p) + (1-p)*log(1-p)), symmetric around
                        p=0.5 and monotonically increasing as |p - 0.5|
                        shrinks.
Confirmed directly in tests/test_query_strategies.py's
test_binary_strategies_produce_identical_rankings, this is not a
coincidence of this project's data, it holds for any binary probabilities.
The three strategies only diverge once there are three or more classes,
which this project's fraud/not-fraud task never has. They are kept as
three separate, named functions anyway because the choice of name is part
of how this technique is normally discussed and reported, and because a
future multi-class extension of this project (a fraud TYPE classifier,
say) would need the distinction back.
"""

from __future__ import annotations

import numpy as np

_EPSILON = 1e-12


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array of shape (n_samples, n_classes).")
    if probabilities.shape[0] == 0:
        raise ValueError("probabilities must be non-empty.")
    if probabilities.shape[1] < 2:
        raise ValueError("probabilities must have at least 2 classes.")
    if np.any(probabilities < -1e-9) or np.any(probabilities > 1 + 1e-9):
        raise ValueError("probabilities must all lie in [0, 1].")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("each row of probabilities must sum to 1.")
    return probabilities


def least_confidence_score(probabilities: np.ndarray) -> np.ndarray:
    """1 - the predicted probability of each row's most likely class."""
    probabilities = _validate_probabilities(probabilities)
    return 1.0 - probabilities.max(axis=1)


def margin_score(probabilities: np.ndarray) -> np.ndarray:
    """1 - (top class probability - second-place class probability)."""
    probabilities = _validate_probabilities(probabilities)
    sorted_desc = np.sort(probabilities, axis=1)[:, ::-1]
    return 1.0 - (sorted_desc[:, 0] - sorted_desc[:, 1])


def entropy_score(probabilities: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row's class distribution, with the 0*log(0)
    convention taken as 0 (a class with probability exactly 0 contributes
    nothing to the sum).
    """
    probabilities = _validate_probabilities(probabilities)
    safe = np.clip(probabilities, _EPSILON, 1.0)
    return -np.sum(probabilities * np.log(safe), axis=1)


_STRATEGIES = {
    "least_confidence": least_confidence_score,
    "margin": margin_score,
    "entropy": entropy_score,
}


def select_query_batch(
    probabilities: np.ndarray, batch_size: int, strategy: str = "entropy"
) -> np.ndarray:
    """Returns the row indices (into probabilities) of the batch_size most
    uncertain samples under the named strategy, highest uncertainty first.

    strategy must be one of "least_confidence", "margin", or "entropy" (see
    module docstring; for binary classification these all return the same
    ranking). batch_size is clipped to the number of available rows if it
    exceeds it, since a shrinking unlabeled pool is the normal end state of
    an active learning loop, not an error.

    Raises ValueError for an unknown strategy or a non-positive batch_size.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}; expected one of {sorted(_STRATEGIES)}."
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    probabilities = _validate_probabilities(probabilities)
    scores = _STRATEGIES[strategy](probabilities)

    n = len(scores)
    effective_batch_size = min(batch_size, n)
    # Stable sort on the NEGATED scores, rather than sorting ascending and
    # reversing the whole array: reversing a stable ascending sort would
    # also reverse the original tie order among equal scores. Sorting -
    # scores directly gives descending-by-score order while keeping tied
    # rows in their original index order, a real distinction confirmed by
    # tests/test_query_strategies.py's test_ties_are_broken_by_original_order.
    order = np.argsort(-scores, kind="stable")
    return order[:effective_batch_size]
