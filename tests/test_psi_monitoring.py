"""
Tests for src.credit_risk.psi_monitoring.

test_psi_matches_hand_computed_four_bin_example is the module's central
check: expected is built so quantile binning lands on exactly four clean,
equal-sized bins (25% each), and actual is built with a known, different
bin split (10/20/30/40), so the whole PSI can be computed in closed form
and compared directly against population_stability_index's own output,
the same discipline test_woe_encoding.py uses for its hand-worked IV
example.

test_psi_report_handles_missing_values_without_crashing and
test_psi_detects_a_shift_in_missing_rate_via_missing_bin cover the real bug
this module's own docstring documents: numpy's np.quantile propagates NaN
through every quantile level, which used to collapse every bin edge to NaN
and made a column with genuine real-world-style missingness (like
EXT_SOURCE_1) raise a spurious "too little variation" error. These tests
guard against that regressing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.credit_risk.psi_monitoring import (
    MISSING_LABEL,
    PSI_SIGNIFICANT_THRESHOLD,
    PSI_STABLE_THRESHOLD,
    population_stability_index,
    psi_by_feature,
    psi_report,
    psi_verdict,
)


def _four_clean_bins(n_per_bin: int = 1000) -> np.ndarray:
    # Four distinct values, equal counts, so quantile edges at [0, .25, .5,
    # .75, 1] fall strictly between the value blocks and every value lands
    # in the bin its own block belongs to, with no ambiguous edge cases.
    return np.concatenate([
        np.full(n_per_bin, 0.0), np.full(n_per_bin, 1.0),
        np.full(n_per_bin, 2.0), np.full(n_per_bin, 3.0),
    ])


def test_psi_matches_hand_computed_four_bin_example():
    expected = _four_clean_bins(n_per_bin=1000)  # 25% / 25% / 25% / 25%.
    actual = np.concatenate([
        np.full(100, 0.0), np.full(200, 1.0), np.full(300, 2.0), np.full(400, 3.0),
    ])  # 10% / 20% / 30% / 40%, same four discrete values.

    pct_expected = [0.25, 0.25, 0.25, 0.25]
    pct_actual = [0.10, 0.20, 0.30, 0.40]
    hand_psi = sum(
        (a - e) * math.log(a / e) for e, a in zip(pct_expected, pct_actual)
    )

    psi = population_stability_index(expected, actual, bins=4)
    assert psi == pytest.approx(hand_psi, abs=1e-3)


def test_identical_distributions_give_zero_psi():
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, size=2000)
    assert population_stability_index(values, values, bins=5) == 0.0


@pytest.mark.parametrize("psi_value,expected_verdict", [
    (0.0, "stable"),
    (0.05, "stable"),
    (PSI_STABLE_THRESHOLD, "moderate_shift"),
    (0.15, "moderate_shift"),
    (PSI_SIGNIFICANT_THRESHOLD, "significant_shift"),
    (0.50, "significant_shift"),
])
def test_psi_verdict_thresholds(psi_value, expected_verdict):
    assert psi_verdict(psi_value) == expected_verdict


def test_psi_report_handles_missing_values_without_crashing():
    rng = np.random.default_rng(1)
    expected = rng.uniform(0, 1, size=2000)
    expected[rng.random(2000) < 0.3] = np.nan  # real-world-style missingness, e.g. EXT_SOURCE_1.
    actual = rng.uniform(0, 1, size=2000)
    actual[rng.random(2000) < 0.3] = np.nan

    report = psi_report(expected, actual, bins=5)
    assert MISSING_LABEL in set(report["bin"])
    assert not report.isna().any().any()


def test_psi_detects_a_shift_in_missing_rate_via_missing_bin():
    rng = np.random.default_rng(2)
    expected = rng.uniform(0, 1, size=3000)
    expected[rng.random(3000) < 0.05] = np.nan
    actual = rng.uniform(0, 1, size=3000)
    actual[rng.random(3000) < 0.30] = np.nan  # missingness rate itself has shifted a lot.

    report = psi_report(expected, actual, bins=5)
    missing_row = report[report["bin"] == MISSING_LABEL].iloc[0]
    assert missing_row["pct_actual"] > missing_row["pct_expected"] + 0.15

    psi = population_stability_index(expected, actual, bins=5)
    assert psi_verdict(psi) in ("moderate_shift", "significant_shift")


def test_psi_by_feature_ranks_shifted_feature_above_stable_feature():
    rng = np.random.default_rng(3)
    n = 3000
    stable_expected = rng.uniform(0, 1, size=n)
    stable_actual = rng.uniform(0, 1, size=n)
    shifted_expected = rng.normal(0, 1, size=n)
    shifted_actual = rng.normal(2.5, 1, size=n)  # a genuinely different distribution.

    import pandas as pd
    expected_df = pd.DataFrame({"STABLE": stable_expected, "SHIFTED": shifted_expected})
    actual_df = pd.DataFrame({"STABLE": stable_actual, "SHIFTED": shifted_actual})

    result = psi_by_feature(expected_df, actual_df, columns=["STABLE", "SHIFTED"], bins=10)
    assert result.iloc[0]["feature"] == "SHIFTED"
    assert result.iloc[0]["psi"] > result.iloc[1]["psi"]
    assert result.iloc[0]["verdict"] == "significant_shift"


def test_psi_report_rejects_empty_expected():
    with pytest.raises(ValueError, match="expected must not be empty"):
        psi_report(np.array([]), np.array([1.0, 2.0]))


def test_psi_report_rejects_empty_actual():
    with pytest.raises(ValueError, match="actual must not be empty"):
        psi_report(np.array([1.0, 2.0]), np.array([]))


def test_psi_report_rejects_expected_with_no_variation():
    expected = np.full(100, 5.0)
    actual = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="too little variation"):
        psi_report(expected, actual, bins=4)
