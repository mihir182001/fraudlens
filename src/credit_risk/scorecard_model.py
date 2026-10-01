"""
scorecard_model.py, Credit Risk Scorecard module: fits a logistic
regression on WoE-encoded features (woe_encoding.py) and converts it into
the points-based scorecard format credit risk teams actually deploy,
following the standard transformation from Naeem Siddiqi's "Credit Risk
Scorecards" (the industry-standard reference for this exact conversion,
reproduced in some form by essentially every practitioner write-up and by
scorecardpy's own scorecard() function).

Why a logistic regression on WoE-encoded features specifically, rather than
any other model: this is the point of the whole WoE step. Once every
feature is replaced by its bin's WoE value, a logistic regression's own
linear score, ln(odds), becomes an exact sum of per-bin contributions:

    ln(odds of bad) = intercept + sum_j( coef_j * WoE_j )

That additive structure is what makes a points-based scorecard possible at
all: each bin of each feature can be assigned a fixed number of points
ahead of time, and a new applicant's score is just those points added up,
computable by a clerk with a calculator (or a lookup table in a live
system) rather than a model call. Swapping in XGBoost here (this project's
model of choice everywhere else) would not offer this property: nothing
about a tree ensemble's output decomposes into fixed per-bin point
contributions, which is exactly why real credit scorecards are still built
this way even at institutions that use gradient boosting elsewhere in the
business.

The points transformation (see build_points_scorecard) uses the
conventional Offset/Factor formulation:
    Factor = pdo / ln(2)
    Offset = base_score - Factor * ln(base_odds)
    Points_i (feature j, bin i) = -Factor * coef_j * WoE_i
    Base points = Offset - Factor * intercept
where pdo ("points to double the odds"), base_score, and base_odds are
illustrative conventions this project chose (pdo=20, base_score=600 at
base_odds=50 good:bad), not a universal standard; different institutions
pick different anchor points for the same underlying formula. The minus
sign is not a typo: this project's logistic regression predicts TARGET=1
(bad/default), and woe_encoding.py's WoE convention is ln(good/bad), so a
well-behaved model's coefficients come out negative (a higher, safer WoE
lowers the odds of bad); negating them turns "more WoE, less risk" into
"more WoE, more points," matching the real-world convention that a HIGHER
score means LOWER risk. This project confirmed the sign directly rather
than assuming it (see tests/test_scorecard_model.py), and also confirmed
that a resulting applicant's score is a strictly monotonic transform of the
logistic regression's own predicted probability, so AUC-ROC/KS/Gini
computed on the points score are IDENTICAL to the same metrics computed on
the raw predicted probability; the scorecard changes what a business user
reads, not what the model can discriminate.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.credit_risk.woe_encoding import WoeBinning

DEFAULT_PDO = 20.0
DEFAULT_BASE_SCORE = 600.0
DEFAULT_BASE_ODDS = 50.0


def fit_scorecard_logistic_regression(
    woe_train_df: pd.DataFrame, feature_columns: List[str], target_col: str = "TARGET",
) -> LogisticRegression:
    """Fits a plain LogisticRegression on the `<feature>_woe` columns for
    feature_columns (woe_train_df is the output of woe_encoding.transform_woe
    on the TRAIN split). No regularization tuning here on purpose: WoE
    binning has already done the project's real "smoothing" work per
    feature, and this project's own comparison is about the scorecard
    conversion, not about squeezing extra AUC out of the logistic
    regression itself.
    """
    woe_columns = [f"{feature}_woe" for feature in feature_columns]
    missing = [c for c in woe_columns if c not in woe_train_df.columns]
    if missing:
        raise ValueError(
            f"woe_train_df is missing {missing}; call woe_encoding.transform_woe "
            "on the same feature_columns first."
        )
    model = LogisticRegression(max_iter=1000)
    model.fit(woe_train_df[woe_columns], woe_train_df[target_col])
    return model


def build_points_scorecard(
    model: LogisticRegression,
    feature_columns: List[str],
    woe_bins: Dict[str, WoeBinning],
    pdo: float = DEFAULT_PDO,
    base_score: float = DEFAULT_BASE_SCORE,
    base_odds: float = DEFAULT_BASE_ODDS,
) -> Tuple[float, Dict[str, pd.DataFrame]]:
    """Converts a fitted logistic regression on WoE features into a
    points-based scorecard: a base point value plus a per-bin point table
    for every feature. See module docstring for the formula and why the
    sign is what it is.

    Returns (base_points, points_tables), where points_tables maps each
    feature to a DataFrame with columns [bin, woe, points].
    """
    if pdo <= 0:
        raise ValueError("pdo must be positive.")
    if base_odds <= 0:
        raise ValueError("base_odds must be positive.")

    factor = pdo / math.log(2)
    offset = base_score - factor * math.log(base_odds)
    base_points = offset - factor * float(model.intercept_[0])

    points_tables: Dict[str, pd.DataFrame] = {}
    for feature, coef in zip(feature_columns, model.coef_[0]):
        binning = woe_bins[feature]
        table = binning.bin_stats[["bin", "woe"]].copy()
        table["points"] = -factor * float(coef) * table["woe"]
        points_tables[feature] = table.reset_index(drop=True)

    return base_points, points_tables


def score_applications(
    df: pd.DataFrame,
    feature_columns: List[str],
    woe_bins: Dict[str, WoeBinning],
    base_points: float,
    points_tables: Dict[str, pd.DataFrame],
) -> np.ndarray:
    """Computes each row of df's total scorecard score: base_points plus
    each selected feature's bin's points, using the SAME fitted bins
    (woe_bins) the points_tables were built from. Never refits a bin.
    """
    total = np.full(len(df), base_points, dtype=float)
    for feature in feature_columns:
        binning = woe_bins[feature]
        bin_labels = binning.assign_bins(df[feature])
        points_map = dict(zip(points_tables[feature]["bin"], points_tables[feature]["points"]))
        total = total + bin_labels.map(points_map).fillna(0.0).to_numpy()
    return total
