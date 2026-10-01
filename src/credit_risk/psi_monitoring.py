"""
psi_monitoring.py, Credit Risk Scorecard module: Population Stability Index
(PSI), the standard metric credit risk teams use to monitor whether a
deployed scorecard's score distribution has drifted away from the
population it was built and validated on.

PSI compares two distributions of the SAME score (or any single variable):
an "expected" distribution (conventionally the training/development
population the scorecard was built and validated against) and an "actual"
distribution (a later population, e.g. this month's new applicants). Both
are binned using edges fit on the EXPECTED distribution only, exactly the
same fit-on-one-side-only discipline this project uses for WoE binning,
frequency encoding, and every train/test split since Week 2: refitting bin
edges on the actual population would defeat the point, since PSI is
specifically measuring whether the actual population still falls into the
expected population's own bins the way it used to.

For bin i:
    PSI_i = (pct_actual_i - pct_expected_i) * ln(pct_actual_i / pct_expected_i)
    PSI = sum_i(PSI_i)

The customary interpretation bands (published in essentially every credit
risk monitoring reference that discusses PSI, and used here unchanged):
    PSI < 0.10            : no significant population shift.
    0.10 <= PSI < 0.25     : moderate shift; investigate before relying on
                             the scorecard's existing thresholds unchanged.
    PSI >= 0.25            : significant shift; the scorecard was built on a
                             population that no longer resembles who is
                             actually applying, and is a real candidate for
                             retraining or at least a fresh validation.

A perfectly stable population still yields a small positive PSI (a finite
sample is never a bit for bit match to itself), which is why the
interpretation is a band around a threshold, not a strict "PSI==0 means
stable" test.

Missing values get their own explicit bin (MISSING_LABEL), the same
discipline woe_encoding.py uses, rather than being silently dropped or
crashing the whole computation. This was a real bug caught by direct
testing against this project's own EXT_SOURCE columns (which carry
substantial, disclosed real-world-style missingness, see
generate_synthetic_credit.py): numpy's np.quantile propagates NaN through
every quantile level when the input contains any NaN, which collapsed
every bin edge to NaN and made _bin_edges_from_expected raise "too little
variation" on a column that in fact had plenty of variation, just also
some missing values. Fixed by computing quantile edges from the non-missing
values only, then explicitly assigning every missing value (in EITHER
population) to its own bin, so a genuine change in a feature's missing
rate between the expected and actual populations is itself something PSI
can detect, rather than being invisible to it.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

_EPSILON = 1e-4  # additive smoothing so an empty bin never causes ln(0) or a division by zero.
MISSING_LABEL = "__MISSING__"

PSI_STABLE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25


def _bin_edges_from_expected(expected_non_missing: np.ndarray, bins: int) -> np.ndarray:
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected_non_missing, quantiles))
    if len(edges) < 2:
        raise ValueError(
            "expected has too little variation to form even one PSI bin "
            "(all its non-missing values are identical)."
        )
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _assign_psi_bins(values: np.ndarray, edges: np.ndarray) -> pd.Series:
    values = np.asarray(values, dtype=float)
    labels = np.empty(len(values), dtype=object)
    is_missing = np.isnan(values)
    labels[is_missing] = MISSING_LABEL
    if (~is_missing).any():
        cut = pd.cut(values[~is_missing], bins=edges, include_lowest=True)
        labels[~is_missing] = cut.astype(str)
    return pd.Series(labels)


def psi_report(expected, actual, bins: int = 10) -> pd.DataFrame:
    """Returns a per-bin PSI breakdown DataFrame with columns [bin,
    pct_expected, pct_actual, psi_contribution], bin edges fit on expected's
    non-missing values only (see module docstring for why missing values
    get their own bin instead). The overall PSI is
    psi_report(...)["psi_contribution"].sum(); population_stability_index
    below is a thin convenience wrapper around that sum.

    Raises ValueError if expected or actual is empty, or if expected's
    non-missing values are too few or too uniform to form even one bin.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(expected) == 0:
        raise ValueError("expected must not be empty.")
    if len(actual) == 0:
        raise ValueError("actual must not be empty.")

    expected_non_missing = expected[~np.isnan(expected)]
    if len(expected_non_missing) == 0:
        raise ValueError("expected has no non-missing values to bin.")

    edges = _bin_edges_from_expected(expected_non_missing, bins)
    expected_labels = _assign_psi_bins(expected, edges)
    actual_labels = _assign_psi_bins(actual, edges)

    # Fixed bin order from expected's OWN labels (falling back to whatever
    # actual introduces, e.g. a MISSING_LABEL expected never had), so both
    # sides are always compared over the same set of bins even when one
    # side has a bin the other technically has zero rows in.
    all_bins = list(dict.fromkeys(list(expected_labels.unique()) + list(actual_labels.unique())))

    expected_counts = expected_labels.value_counts().reindex(all_bins, fill_value=0)
    actual_counts = actual_labels.value_counts().reindex(all_bins, fill_value=0)

    pct_expected = (expected_counts + _EPSILON) / (len(expected) + _EPSILON * len(all_bins))
    pct_actual = (actual_counts + _EPSILON) / (len(actual) + _EPSILON * len(all_bins))

    report = pd.DataFrame({
        "bin": all_bins,
        "pct_expected": pct_expected.to_numpy(),
        "pct_actual": pct_actual.to_numpy(),
    })
    report["psi_contribution"] = (
        (report["pct_actual"] - report["pct_expected"])
        * np.log(report["pct_actual"] / report["pct_expected"])
    )
    return report


def population_stability_index(expected, actual, bins: int = 10) -> float:
    """Returns the overall PSI comparing actual against expected (bin edges
    fit on expected only). See module docstring for the formula and the
    customary interpretation bands (psi_verdict).
    """
    return float(psi_report(expected, actual, bins=bins)["psi_contribution"].sum())


def psi_verdict(psi_value: float) -> str:
    """Returns "stable" (PSI < 0.10), "moderate_shift" (0.10-0.25), or
    "significant_shift" (>= 0.25), the customary bands described in the
    module docstring.
    """
    if psi_value < PSI_STABLE_THRESHOLD:
        return "stable"
    if psi_value < PSI_SIGNIFICANT_THRESHOLD:
        return "moderate_shift"
    return "significant_shift"


def psi_by_feature(
    expected_df: pd.DataFrame, actual_df: pd.DataFrame, columns, bins: int = 10,
) -> pd.DataFrame:
    """Runs population_stability_index independently for each of columns,
    returning a DataFrame with [feature, psi, verdict], sorted by
    descending PSI. Useful for finding WHICH input features drove an
    overall score PSI shift, not just that the score shifted.
    """
    rows = []
    for column in columns:
        psi_value = population_stability_index(
            expected_df[column].to_numpy(), actual_df[column].to_numpy(), bins=bins,
        )
        rows.append({"feature": column, "psi": psi_value, "verdict": psi_verdict(psi_value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
