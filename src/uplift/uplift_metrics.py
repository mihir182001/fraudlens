"""
uplift_metrics.py, Week 6: evaluation metrics for uplift models, the causal
counterpart to src/models/metrics.py's KS/Gini/AUC-PR for plain classifiers.

A plain classification metric like AUC-ROC cannot evaluate an uplift model
at all: it needs a single ground-truth label per row, but no row has an
observed "true uplift" (that would require observing the same customer both
treated and untreated). What every metric here does instead is compare, at
each ranking cutoff, the observed conversion rate among TREATED customers in
that slice against the observed conversion rate among CONTROL customers in
that slice. That comparison is only a valid estimate of the ranking's real
quality because treatment was assigned at random (see
src/data/generate_synthetic_campaign.py's module docstring); on data where
treatment was not randomized, these formulas would silently produce
confounded, meaningless numbers.

qini_curve is this module's building block: at each population fraction phi
(customers sorted by descending predicted uplift), it reports
    Y_t(phi) - Y_c(phi) * N_t(phi) / N_c(phi)
where Y_t/Y_c are cumulative treated/control responders in that top phi
slice and N_t/N_c are cumulative treated/control counts. The N_t/N_c
reweighting exists because a top-phi slice will rarely contain exactly equal
numbers of treated and control customers (whoever the model ranks highest),
so Y_c is rescaled to what it would be if the control group in that slice
were the same size as the treated group, making the two directly
comparable. This is the standard Qini curve definition (Radcliffe, 2007).
Before the first control customer appears in the ranking (N_c(phi) == 0),
the ratio is undefined; this implementation defines the curve as
Y_t(phi) alone at those points, since there is no control baseline yet to
subtract.

qini_coefficient is the area between the model's Qini curve and the
diagonal line from (0, 0) to (1, Qini(1)), the curve a model that ranks
customers in a completely uninformative order would trace out in
expectation, divided by the number of customers n. This is the direct
causal analogue of Gini's "area between the ROC curve and the
random-classifier diagonal", src/models/metrics.py's gini_coefficient; the
division by n is there for the same reason Gini and AUC are already
scale-free: Y_t(phi) and Y_c(phi) are raw responder counts, so the
un-normalized area grows with the size of the dataset and is not
comparable across runs with different n. Dividing by n turns it into an
average incremental converted customer per capita, gained by using this
model's ranking instead of random targeting, a small number typically well
under 1.0 and comparable across dataset sizes. A positive qini_coefficient
means the model's ranking finds more real incremental converters, sorted to
the top, than a random ranking would; zero or negative means it does no
better than random.

uplift_at_k reports something more directly actionable: the plain observed
uplift (treated conversion rate minus control conversion rate) restricted
to the top k fraction of customers by predicted uplift, the number a
marketer would actually see if told "target your top k% by this model's
score."
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _validate_uplift_eval_inputs(
    outcome: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    outcome = np.asarray(outcome)
    treatment = np.asarray(treatment)
    uplift_score = np.asarray(uplift_score)

    if not (len(outcome) == len(treatment) == len(uplift_score)):
        raise ValueError("outcome, treatment, and uplift_score must have the same length.")
    if len(outcome) == 0:
        raise ValueError("outcome, treatment, and uplift_score must be non-empty.")
    if not set(np.unique(outcome)).issubset({0, 1}):
        raise ValueError("outcome must be binary (0/1).")
    if not set(np.unique(treatment)).issubset({0, 1}):
        raise ValueError("treatment must be binary (0/1).")
    if len(np.unique(treatment)) < 2:
        raise ValueError("treatment must contain both treated (1) and control (0) rows.")

    return outcome, treatment, uplift_score


def qini_curve(
    outcome: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (fractions, qini_values), both length n, sorted by descending
    uplift_score (ties broken by original order, via a stable sort).
    fractions[i] = (i + 1) / n, the cumulative population fraction targeted
    when including the top i + 1 customers by predicted uplift.
    qini_values[i] is this project's Qini curve value at that fraction; see
    module docstring for the formula and the zero-control-so-far edge case.
    """
    outcome, treatment, uplift_score = _validate_uplift_eval_inputs(
        outcome, treatment, uplift_score
    )
    n = len(outcome)

    order = np.argsort(-uplift_score, kind="mergesort")
    outcome_sorted = outcome[order]
    treatment_sorted = treatment[order]

    is_treated = treatment_sorted == 1
    is_control = ~is_treated

    cum_treated = np.cumsum(is_treated)
    cum_control = np.cumsum(is_control)
    cum_treated_responders = np.cumsum(is_treated & (outcome_sorted == 1))
    cum_control_responders = np.cumsum(is_control & (outcome_sorted == 1))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cum_control > 0, cum_treated / np.maximum(cum_control, 1), 0.0)

    qini_values = np.where(
        cum_control > 0,
        cum_treated_responders - cum_control_responders * ratio,
        cum_treated_responders.astype(float),
    )

    fractions = np.arange(1, n + 1) / n
    return fractions, qini_values.astype(float)


def qini_coefficient(
    outcome: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray
) -> float:
    """The area between the model's Qini curve and the random-targeting
    diagonal, divided by n (see module docstring for why). Positive means
    better-than-random targeting; zero or negative means no better than
    random.
    """
    fractions, qini_values = qini_curve(outcome, treatment, uplift_score)
    n = len(qini_values)
    overall_qini = qini_values[-1]

    baseline_values = fractions * overall_qini

    x = np.concatenate([[0.0], fractions])
    model_curve = np.concatenate([[0.0], qini_values])
    baseline_curve = np.concatenate([[0.0], baseline_values])

    area_model = float(np.trapezoid(model_curve, x))
    area_baseline = float(np.trapezoid(baseline_curve, x))
    return (area_model - area_baseline) / n


def uplift_at_k(
    outcome: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray, k: float
) -> float:
    """The observed uplift (treated conversion rate minus control conversion
    rate) among the top k fraction of customers by predicted uplift_score.

    Raises ValueError if k is not in (0, 1], or if the top-k slice does not
    contain at least one treated and one control customer (the rate
    difference is undefined without both).
    """
    outcome, treatment, uplift_score = _validate_uplift_eval_inputs(
        outcome, treatment, uplift_score
    )
    if not (0.0 < k <= 1.0):
        raise ValueError("k must be in (0, 1].")

    n = len(outcome)
    n_top = max(1, int(round(k * n)))
    order = np.argsort(-uplift_score, kind="mergesort")
    top_idx = order[:n_top]

    treatment_top = treatment[top_idx]
    outcome_top = outcome[top_idx]
    treated_mask = treatment_top == 1
    control_mask = ~treated_mask

    if treated_mask.sum() == 0 or control_mask.sum() == 0:
        raise ValueError(
            f"The top {k:.1%} slice ({n_top} rows) contains only one "
            "treatment group; uplift is undefined without both."
        )

    treated_rate = float(outcome_top[treated_mask].mean())
    control_rate = float(outcome_top[control_mask].mean())
    return treated_rate - control_rate
