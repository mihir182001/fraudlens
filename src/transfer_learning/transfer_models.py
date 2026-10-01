"""
transfer_models.py, Week 8: transfer learning between a mature SOURCE
fraud domain and a small, newly-launched TARGET domain, via XGBoost's
warm-start continuation, compared against two baselines.

Three ways of getting a model for the target domain, all built from the
same DomainModel bundle (a fitted SimpleImputer plus a fitted XGBClassifier,
kept together because scoring new rows requires applying the exact same
imputer the model was fit behind):

  - apply_zero_shot: score the target domain directly with the SOURCE
    model, no target labels used at all. This is the "do nothing" baseline;
    it is expected to do WORSE on the target domain than a model that has
    seen any target data, precisely because the two domains are built with
    a deliberate prevalence and pattern shift (see train_transfer_learning.py
    for the exact numbers this project measured).
  - warm_start_transfer: continue training the SOURCE model's existing
    XGBoost booster on a small amount of TARGET training data, via
    XGBoost's own xgb_model= continuation parameter. Confirmed directly in
    this project's own sandbox testing (see Week 8's module docstring
    history) that passing xgb_model=<source booster> to a fresh
    XGBClassifier.fit() call ADDS new trees on top of the existing ones
    (a source with 100 boosted rounds continued with n_estimators=30 and
    xgb_model=<source booster> produced a model with 130 total rounds, not
    30), which is exactly the "start from what the source domain already
    learned, then adapt using only the little target data available"
    behavior transfer learning is meant to give.
  - train_from_scratch: fit a brand new imputer and a brand new
    XGBClassifier using ONLY the target domain's own training data, with no
    knowledge of the source domain at all. This is the realistic
    alternative to transfer learning: what a team would do if no source
    model existed. It is expected to do WORSE than warm_start_transfer
    specifically when the target training set is very small (there is not
    enough target data to learn a good decision boundary from nothing),
    and the gap is expected to close as the target training set grows.

A preprocessing discipline worth stating plainly, because it is easy to get
backwards: warm_start_transfer reuses the SOURCE model's already-fitted
imputer (never refits one on the small target training set), because the
source booster's trees were built under that imputer's specific fill
values and numeric scale; refitting a new imputer on a tiny target sample
before continuing training would silently shift every feature's fill value
and hand the existing trees data on a scale they were never built for.
train_from_scratch, by contrast, fits its own imputer on the target
training set only, exactly as it would have to in a real deployment with
no source model available at all. This mirrors a design decision this
project already made once before, in Week 5's build_transaction_graph.py
(the transfer-vs-from-scratch preprocessing split there was analogous), and
is deliberately not "fixed" to use one imputer everywhere: using the
source's imputer for the from-scratch baseline would make it not actually
"from scratch" any more, and would overstate how much transfer learning
helps by handicapping the very comparison it is measured against.

A second, real limitation this project found only by directly inspecting
the continued booster's own trees rather than trusting its aggregate test
metrics: XGBoost's default min_child_weight (1.0) can make warm-start
continuation a complete, silent no-op on exactly the small-target-budget
case transfer learning exists for. min_child_weight is compared against
the SUM OF HESSIANS in a node, not a row count, and a mature source
model's Hessian per row (p * (1 - p) for log loss) shrinks toward zero as
its predictions become more confident. Confirmed directly (both through
the sklearn XGBClassifier wrapper and through the low-level
xgboost.train() API, so this is not specific to how this module calls
either one): continuing this project's own 200-round source booster on a
30-row target batch produced a total Hessian of 0.1648, below the 1.0
default, and every one of the new trees came back as a single root leaf
with weight EXACTLY 0.0, confirmed by reading get_booster().trees_to_dataframe()
directly, not merely inferred from unchanged metrics; predictions on that
batch, and on the fixed target test set, were then bit-for-bit identical
to the un-continued source model, no matter how many additional
estimators were requested, since a zero-weight tree's gradients never
change and every subsequent round recomputes the identical zero-weight
result. Lowering min_child_weight to 0.1 (chosen by testing 1.0, 0.1, 0.01,
and 0.0 directly against this same 30-row batch; 0.1 was the smallest
value that reliably produced a real, non-degenerate leaf weight without
also removing the guard entirely) fixed this, confirmed by the same
trees_to_dataframe check now showing genuinely fit, non-zero leaf weights.
This is why warm_start_transfer's default is 0.1, not XGBoost's own
library default of 1.0.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

RANDOM_STATE = 42


class DomainModel:
    """A fitted imputer paired with a fitted XGBClassifier, plus the exact
    feature column order both were fit against. Kept together because
    scoring new rows always means applying THIS imputer before THIS model,
    never one fit on different data (see module docstring).
    """

    def __init__(self, imputer: SimpleImputer, model: XGBClassifier, feature_columns: List[str]):
        self.imputer = imputer
        self.model = model
        self.feature_columns = list(feature_columns)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the predicted probability of the positive class for each
        row of df. Raises ValueError if df is missing any column this
        model's imputer was fit against (see _check_columns_present).
        """
        _check_columns_present(self, df)
        x = self.imputer.transform(df[self.feature_columns])
        return self.model.predict_proba(x)[:, 1]


def _check_columns_present(domain_model: DomainModel, df: pd.DataFrame) -> None:
    missing = [c for c in domain_model.feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"df is missing {len(missing)} column(s) this model was fit "
            f"against: {missing}. The source and target domains must share "
            "the same feature schema for transfer learning to apply."
        )


def _check_fitted(domain_model: Optional[DomainModel], what: str) -> None:
    if domain_model is None:
        raise ValueError(
            f"{what} must be a fitted DomainModel; call train_source_model "
            "first."
        )


def _validate_training_frame(df: pd.DataFrame, target_col: str, what: str) -> None:
    if df.empty:
        raise ValueError(f"{what} is empty.")
    if target_col not in df.columns:
        raise ValueError(f"{what} has no column {target_col!r}.")
    if df[target_col].nunique() < 2:
        raise ValueError(
            f"{what} has only one class present in {target_col!r}; a "
            "classifier cannot be fit or continued without both classes."
        )


def train_source_model(
    train_df: pd.DataFrame,
    feature_columns: List[str],
    target_col: str = "isFraud",
    n_estimators: int = 200,
    max_depth: int = 4,
    random_state: int = RANDOM_STATE,
) -> DomainModel:
    """Fits the SOURCE domain's model from scratch: a SimpleImputer fit on
    train_df's own features, and an XGBClassifier fit on the imputed
    result. This is the model every other function in this module either
    scores directly (apply_zero_shot) or continues training from
    (warm_start_transfer).
    """
    _validate_training_frame(train_df, target_col, "train_df")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train_df[feature_columns])
    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, eval_metric="logloss", n_jobs=-1,
    )
    model.fit(x_train, train_df[target_col])
    return DomainModel(imputer, model, feature_columns)


def apply_zero_shot(source_model: Optional[DomainModel], target_df: pd.DataFrame) -> np.ndarray:
    """Scores target_df directly with the SOURCE model, using no target
    labels at all: the "do nothing" baseline every real transfer or
    from-scratch approach is compared against.
    """
    _check_fitted(source_model, "source_model")
    return source_model.predict_proba(target_df)


def warm_start_transfer(
    source_model: Optional[DomainModel],
    target_train_df: pd.DataFrame,
    target_col: str = "isFraud",
    n_additional_estimators: int = 50,
    max_depth: int = 4,
    min_child_weight: float = 0.1,
    random_state: int = RANDOM_STATE,
) -> DomainModel:
    """Continues training the SOURCE model's existing XGBoost booster on
    TARGET training data, via xgb_model=<source booster>. Returns a NEW
    DomainModel that reuses the SOURCE model's already-fitted imputer (see
    module docstring for why) paired with the continued booster.

    n_additional_estimators is how many NEW trees this call adds on top of
    the source booster's existing trees, confirmed directly in this
    project's own sandbox testing (see module docstring): it is not a
    replacement tree count.

    min_child_weight defaults to 0.1, not XGBoost's own library default of
    1.0, because of a real, measured failure mode this project found: with
    a small target_train_df and an already-confident source model, the
    default can make this whole function a silent no-op. See the module
    docstring's dedicated explanation before changing this default back up.

    Raises ValueError if source_model has not been fit yet, if
    n_additional_estimators is not positive, if target_train_df is empty or
    has only one class present, or if target_train_df is missing any
    feature column the source model was fit against.
    """
    _check_fitted(source_model, "source_model")
    _check_columns_present(source_model, target_train_df)
    if n_additional_estimators <= 0:
        raise ValueError("n_additional_estimators must be positive.")
    _validate_training_frame(target_train_df, target_col, "target_train_df")

    x_target = source_model.imputer.transform(target_train_df[source_model.feature_columns])
    y_target = target_train_df[target_col]

    continued_model = XGBClassifier(
        n_estimators=n_additional_estimators, max_depth=max_depth,
        min_child_weight=min_child_weight,
        random_state=random_state, eval_metric="logloss", n_jobs=-1,
    )
    continued_model.fit(x_target, y_target, xgb_model=source_model.model.get_booster())
    return DomainModel(source_model.imputer, continued_model, source_model.feature_columns)


def train_from_scratch(
    target_train_df: pd.DataFrame,
    feature_columns: List[str],
    target_col: str = "isFraud",
    n_estimators: int = 50,
    max_depth: int = 4,
    random_state: int = RANDOM_STATE,
) -> DomainModel:
    """Fits a brand new imputer and a brand new XGBClassifier using ONLY
    target_train_df, with no knowledge of any source model. The realistic
    alternative to transfer learning when no source model is available;
    see module docstring for why this fits its OWN imputer rather than
    reusing the source's.
    """
    _validate_training_frame(target_train_df, target_col, "target_train_df")
    imputer = SimpleImputer(strategy="median")
    x_target = imputer.fit_transform(target_train_df[feature_columns])
    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, eval_metric="logloss", n_jobs=-1,
    )
    model.fit(x_target, target_train_df[target_col])
    return DomainModel(imputer, model, feature_columns)
