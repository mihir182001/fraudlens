"""
woe_encoding.py, Credit Risk Scorecard module: Weight of Evidence (WoE)
binning and Information Value (IV) computation, built directly on
pandas/numpy rather than the scorecardpy package the original spec named.

scorecardpy was tried first and rejected for a real, reproducible reason,
not a style preference: it imports pkg_resources at import time
(scorecardpy/germancredit.py), a module setuptools has now fully removed
(confirmed directly: setuptools 84.0.0, this project's installed version,
raises ModuleNotFoundError on import). Downgrading setuptools far enough to
restore pkg_resources (setuptools<77) was confirmed to work for
scorecardpy's own woebin() call, but that downgrade directly conflicts with
a dependency ALREADY in this project: torch 2.14.0 (Week 5's GNN module)
requires setuptools>=77.0.3. There is no single setuptools version that
satisfies both packages in this environment. On top of that, scorecardpy's
woebin() itself throws a dozen+ pandas FutureWarnings even once it imports
(deprecated groupby/agg patterns throughout woebin.py), meaning it is
already flagged to break outright on a future pandas release. Given a real
dependency conflict plus a library visibly heading toward breakage, WoE and
IV are implemented here directly instead: both are simple, well-defined
arithmetic (see below), not something that benefits from an opaque
dependency, and building them directly means this module's own tests pin
down the exact formulas rather than trusting an unmaintained package's.

Convention used throughout (the standard one in credit scoring literature,
e.g. Siddiqi's "Credit Risk Scorecards", and the same one scorecardpy
itself uses): "good" means TARGET == 0 (no payment difficulty), "bad" means
TARGET == 1 (a default). For bin i:
    WoE_i = ln( distr_good_i / distr_bad_i )
    IV_i  = (distr_good_i - distr_bad_i) * WoE_i
    total_IV = sum_i(IV_i)
where distr_good_i and distr_bad_i are bin i's share of ALL good rows and
ALL bad rows respectively (not a share of bin i's own rows). A bin with
more goods than its overall share gets positive WoE (safer than average); a
bin with more bads than its overall share gets negative WoE (riskier than
average). IV is the industry's usual predictive-power ruler for one
feature; see IV_INTERPRETATION_THRESHOLDS below for the customary bands.

Every bin's edges and its WoE table are fit on TRAIN ONLY and applied
(never refit) to test, the identical fit-on-train-only discipline this
project has followed since Week 2's frequency_encode: fit_woe_bins /
transform_woe mirror that module's fit_frequencies / transform_frequencies
split by name on purpose.

Special values (fit_woe_bins' special_values argument) exist because of a
real, disclosed data-quality fact this project's own synthetic generator
reproduces on purpose (see generate_synthetic_credit.py's module
docstring): DAYS_EMPLOYED uses 365243 as a placeholder for "not employed,"
not a real employment length. Quantile-binning that value in with genuine
employment lengths would put every unemployed/pensioner applicant in
whatever bin happens to catch an extreme outlier, diluting whatever real
signal employment length carries for people who ARE employed. Declaring it
a special value gives it its own bin and its own WoE, exactly like a
missing value gets its own bin below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

MISSING_LABEL = "__MISSING__"
UNSEEN_LABEL = "__UNSEEN__"
_EPSILON = 0.5  # additive (Laplace-style) smoothing count; see _woe_iv_table.

IV_INTERPRETATION_THRESHOLDS = {
    "not_useful": 0.02,
    "weak": 0.10,
    "medium": 0.30,
    "strong": 0.50,
}


def iv_strength_label(total_iv: float) -> str:
    """Returns the customary credit-scoring interpretation band for an IV
    value (see IV_INTERPRETATION_THRESHOLDS): "not_useful" (<0.02), "weak"
    (0.02-0.1), "medium" (0.1-0.3), "strong" (0.3-0.5), or "suspicious"
    (>0.5, conventionally read as too good to be true, worth checking for
    target leakage rather than celebrating).
    """
    if total_iv < IV_INTERPRETATION_THRESHOLDS["not_useful"]:
        return "not_useful"
    if total_iv < IV_INTERPRETATION_THRESHOLDS["weak"]:
        return "weak"
    if total_iv < IV_INTERPRETATION_THRESHOLDS["medium"]:
        return "medium"
    if total_iv < IV_INTERPRETATION_THRESHOLDS["strong"]:
        return "strong"
    return "suspicious"


@dataclass
class WoeBinning:
    """One feature's fitted binning: how to assign any future row to a bin
    label, and that bin's WoE value, both fit on TRAIN ONLY.
    """

    feature: str
    is_numeric: bool
    bin_edges: Optional[np.ndarray]  # None for categorical features
    special_values: List[float] = field(default_factory=list)
    known_categories: Optional[List[str]] = None  # None for numeric features
    bin_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_iv: float = 0.0

    def assign_bins(self, values: pd.Series) -> pd.Series:
        """Returns the bin label for every row of values, using bin edges
        (or categories) fit on train. Never refits anything.
        """
        if self.is_numeric:
            return _assign_numeric_bins(values, self.bin_edges, self.special_values)
        return _assign_categorical_bins(values, self.known_categories)

    def woe_map(self) -> Dict[str, float]:
        return dict(zip(self.bin_stats["bin"], self.bin_stats["woe"]))


def _assign_numeric_bins(
    values: pd.Series, bin_edges: np.ndarray, special_values: Sequence[float],
) -> pd.Series:
    values = pd.Series(values).reset_index(drop=True)
    labels = pd.Series(index=values.index, dtype=object)

    is_missing = values.isna()
    labels[is_missing] = MISSING_LABEL

    is_special = pd.Series(False, index=values.index)
    for special in special_values:
        match = (~is_missing) & (values == special)
        is_special = is_special | match
        labels[match] = f"special={special!r}"

    remaining = ~is_missing & ~is_special
    if remaining.any() and len(bin_edges) >= 2:
        cut = pd.cut(values[remaining], bins=bin_edges, include_lowest=True)
        labels[remaining] = cut.astype(str)
    return labels


def _assign_categorical_bins(values: pd.Series, known_categories: Sequence[str]) -> pd.Series:
    values = pd.Series(values).reset_index(drop=True)
    labels = values.astype(object)
    is_missing = values.isna()
    labels[is_missing] = MISSING_LABEL
    known = set(known_categories)
    is_unseen = (~is_missing) & (~values.isin(known))
    labels[is_unseen] = UNSEEN_LABEL
    return labels


def _woe_iv_table(bin_labels: pd.Series, target: pd.Series) -> pd.DataFrame:
    """Builds the per-bin WoE/IV table for one already-binned feature.

    _EPSILON is added to every bin's good and bad count before computing
    shares (Laplace-style smoothing), so a bin with zero goods or zero bads
    gets a large-but-finite WoE instead of ln(0) or a division by zero; a
    perfectly pure bin is a real, if extreme, possibility with a small
    synthetic sample or a genuinely rare category, and this project would
    rather report a large finite WoE for it than crash.
    """
    target = pd.Series(target).reset_index(drop=True).astype(int)
    frame = pd.DataFrame({"bin": bin_labels.reset_index(drop=True), "target": target})

    total_good = float((frame["target"] == 0).sum())
    total_bad = float((frame["target"] == 1).sum())
    if total_good == 0 or total_bad == 0:
        raise ValueError(
            "Both classes (TARGET 0 and 1) must be present to compute WoE/IV."
        )

    grouped = frame.groupby("bin")["target"].agg(
        count="count", bad="sum",
    ).reset_index()
    grouped["good"] = grouped["count"] - grouped["bad"]

    smoothed_good = grouped["good"] + _EPSILON
    smoothed_bad = grouped["bad"] + _EPSILON
    distr_good = smoothed_good / (total_good + _EPSILON * len(grouped))
    distr_bad = smoothed_bad / (total_bad + _EPSILON * len(grouped))

    grouped["distr_good"] = distr_good
    grouped["distr_bad"] = distr_bad
    grouped["woe"] = np.log(distr_good / distr_bad)
    grouped["iv"] = (distr_good - distr_bad) * grouped["woe"]
    return grouped.sort_values("bin").reset_index(drop=True)


def fit_woe_bins(
    train_df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_col: str = "TARGET",
    n_bins: int = 5,
    special_values: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, WoeBinning]:
    """Fits a WoeBinning for every column in feature_columns, using ONLY
    train_df. special_values maps a feature name to a list of sentinel
    values (e.g. {"DAYS_EMPLOYED": [365243]}) that get pulled out into
    their own bin before quantile binning the rest; a feature with no entry
    gets no special-value handling.

    Numeric columns are binned by TRAIN quantiles (pd.qcut with
    duplicates="drop", since a low-cardinality or skewed numeric column can
    genuinely have fewer than n_bins distinct quantile edges); categorical
    (object/category/bool) columns get one bin per distinct TRAIN category.
    """
    if special_values is None:
        special_values = {}

    bins: Dict[str, WoeBinning] = {}
    for feature in feature_columns:
        column = train_df[feature]
        feature_specials = special_values.get(feature, [])
        is_numeric = pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column)

        if is_numeric:
            non_special = column[~column.isin(feature_specials) & column.notna()]
            if non_special.nunique() < 2:
                edges = np.array([-np.inf, np.inf])
            else:
                _, edges = pd.qcut(
                    non_special, q=n_bins, retbins=True, duplicates="drop",
                )
                edges = edges.copy()
                edges[0] = -np.inf
                edges[-1] = np.inf
            binning = WoeBinning(
                feature=feature, is_numeric=True, bin_edges=edges,
                special_values=list(feature_specials),
            )
        else:
            known_categories = sorted(column.dropna().unique().tolist())
            binning = WoeBinning(
                feature=feature, is_numeric=False, bin_edges=None,
                known_categories=known_categories,
            )

        bin_labels = binning.assign_bins(column)
        stats = _woe_iv_table(bin_labels, train_df[target_col])
        binning.bin_stats = stats
        binning.total_iv = float(stats["iv"].sum())
        bins[feature] = binning

    return bins


def transform_woe(df: pd.DataFrame, woe_bins: Dict[str, WoeBinning]) -> pd.DataFrame:
    """Returns a COPY of df with one new `<feature>_woe` column per entry in
    woe_bins, applying each feature's already-fitted bins (never refitting
    on df). A bin label present in df but never seen in the fitting data
    (possible for a categorical column's UNSEEN_LABEL, or in principle a
    numeric special value that never appeared in train) maps to WoE 0.0,
    a neutral "no information" value, rather than raising, since this can
    legitimately happen on real held-out data.
    """
    out = df.copy()
    for feature, binning in woe_bins.items():
        bin_labels = binning.assign_bins(df[feature])
        woe_map = binning.woe_map()
        out[f"{feature}_woe"] = bin_labels.map(woe_map).fillna(0.0).to_numpy()
    return out


def select_features_by_iv(
    woe_bins: Dict[str, WoeBinning], min_iv: float = IV_INTERPRETATION_THRESHOLDS["not_useful"],
) -> List[str]:
    """Returns the feature names whose total_iv is at least min_iv, sorted
    by descending IV. Features above IV_INTERPRETATION_THRESHOLDS["strong"]
    are kept (a high IV is not itself disqualifying) but flagged separately
    by iv_report, since a suspiciously high IV is conventionally a prompt
    to check for target leakage, not a reason to silently drop the feature.
    """
    return [
        feature for feature, _ in sorted(
            woe_bins.items(), key=lambda item: item[1].total_iv, reverse=True,
        )
        if woe_bins[feature].total_iv >= min_iv
    ]


def iv_report(woe_bins: Dict[str, WoeBinning]) -> pd.DataFrame:
    """Returns a one-row-per-feature summary: feature, total_iv, and its
    iv_strength_label, sorted by descending IV, for printing or logging.
    """
    rows = [
        {"feature": feature, "total_iv": binning.total_iv, "strength": iv_strength_label(binning.total_iv)}
        for feature, binning in woe_bins.items()
    ]
    return pd.DataFrame(rows).sort_values("total_iv", ascending=False).reset_index(drop=True)
