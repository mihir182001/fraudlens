"""
Tests for src.transfer_learning.transfer_models.

Three tests carry this module's central, measured claims (see
transfer_models.py's and train_transfer_learning.py's module docstrings for
the full numbers this project found before writing these assertions):

  - test_zero_shot_degrades_under_a_real_pattern_shift: the source model
    applied zero-shot must score WORSE on a target domain built with a real
    pattern shift (domain_data.py's shift_columns) than on an otherwise
    identical target domain built without one. Without this, Week 8's whole
    premise (a domain shift transfer learning has to work across) would not
    actually be real.
  - test_warm_start_beats_from_scratch_on_small_target_budget (at two
    different small budgets): the entire reason to prefer transfer over
    training from scratch is that it should win when target data is scarce.
  - test_default_min_child_weight_avoids_the_measured_xgboost_no_op and
    test_min_child_weight_one_reproduces_the_measured_no_op: a real,
    separately investigated XGBoost anomaly (see transfer_models.py's
    module docstring), confirmed here as two directly opposite outcomes
    from the same continuation call, one with the library's own default
    min_child_weight and one with this project's chosen default.

All fixtures use a deliberately small but non-trivial domain size (source:
4000 transactions/150 estimators, target: 800 transactions), chosen by
direct testing in this project's own sandbox to keep the full test file
fast (a few seconds) while still reproducing every measured finding above;
see the two module docstrings referenced above for the larger-scale numbers
this project actually reports as Week 8's real result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.models.metrics import auc_pr
from src.transfer_learning.domain_data import build_domain
from src.transfer_learning.transfer_models import (
    DomainModel,
    apply_zero_shot,
    train_from_scratch,
    train_source_model,
    warm_start_transfer,
)

SHIFT_COLUMNS = [f"V{i}" for i in range(1, 11)]


def _build_source():
    return build_domain(n_transactions=4000, fraud_rate=0.04, seed=1, start_transaction_id=1_000_000)


def _build_target(shift: bool = True):
    shift_columns = SHIFT_COLUMNS if shift else None
    return build_domain(
        n_transactions=800, fraud_rate=0.07, seed=2, start_transaction_id=5_000_000,
        shift_columns=shift_columns, shift_seed=99,
    )


def _fit_source_model():
    source_train, source_test, source_features = _build_source()
    source_model = train_source_model(source_train, source_features, n_estimators=150)
    return source_model, source_features


def _subsample(target_train, budget, random_state=0):
    sub_train, _ = train_test_split(
        target_train, train_size=budget, stratify=target_train["isFraud"], random_state=random_state,
    )
    return sub_train


def test_zero_shot_degrades_under_a_real_pattern_shift():
    source_model, _ = _fit_source_model()

    _, target_test_shifted, _ = _build_target(shift=True)
    _, target_test_plain, _ = _build_target(shift=False)

    shifted_proba = apply_zero_shot(source_model, target_test_shifted)
    plain_proba = apply_zero_shot(source_model, target_test_plain)

    shifted_auc_pr = auc_pr(target_test_shifted["isFraud"], shifted_proba)
    plain_auc_pr = auc_pr(target_test_plain["isFraud"], plain_proba)

    assert shifted_auc_pr < plain_auc_pr


@pytest.mark.parametrize("budget", [40, 80])
def test_warm_start_beats_from_scratch_on_small_target_budget(budget):
    source_model, source_features = _fit_source_model()
    target_train, target_test, target_features = _build_target(shift=True)
    assert source_features == target_features

    sub_train = _subsample(target_train, budget)

    warm_model = warm_start_transfer(source_model, sub_train, n_additional_estimators=50)
    scratch_model = train_from_scratch(sub_train, target_features, n_estimators=50)

    warm_auc_pr = auc_pr(target_test["isFraud"], warm_model.predict_proba(target_test))
    scratch_auc_pr = auc_pr(target_test["isFraud"], scratch_model.predict_proba(target_test))

    assert warm_auc_pr > scratch_auc_pr


def test_default_min_child_weight_avoids_the_measured_xgboost_no_op():
    source_model, _ = _fit_source_model()
    target_train, target_test, _ = _build_target(shift=True)
    sub_train = _subsample(target_train, 40)

    zero_shot_proba = apply_zero_shot(source_model, target_test)
    warm_model = warm_start_transfer(source_model, sub_train, n_additional_estimators=50)
    warm_proba = warm_model.predict_proba(target_test)

    # See transfer_models.py's module docstring: with the library's own
    # min_child_weight default this would come back bit-for-bit identical
    # to zero_shot_proba, a real, measured no-op. This project's chosen
    # default (0.1) must actually move the predictions.
    assert np.max(np.abs(warm_proba - zero_shot_proba)) > 0.1


def test_min_child_weight_one_reproduces_the_measured_no_op():
    # A direct regression test for the anomaly transfer_models.py's module
    # docstring describes: continuing training with XGBoost's own
    # min_child_weight default (1.0) on a small enough batch produces
    # trees whose leaves are all exactly 0.0, making the continued model
    # bit-for-bit identical to the un-continued source model. Built here
    # with the raw XGBClassifier rather than warm_start_transfer, since
    # warm_start_transfer itself no longer defaults to 1.0.
    source_model, source_features = _fit_source_model()
    target_train, target_test, _ = _build_target(shift=True)
    sub_train = _subsample(target_train, 40)

    zero_shot_proba = apply_zero_shot(source_model, target_test)

    x_target = source_model.imputer.transform(sub_train[source_features])
    y_target = sub_train["isFraud"]
    no_op_model = XGBClassifier(
        n_estimators=50, max_depth=4, min_child_weight=1.0,
        random_state=42, eval_metric="logloss", n_jobs=-1,
    )
    no_op_model.fit(x_target, y_target, xgb_model=source_model.model.get_booster())
    no_op_bundle = DomainModel(source_model.imputer, no_op_model, source_features)

    no_op_proba = no_op_bundle.predict_proba(target_test)
    assert np.max(np.abs(no_op_proba - zero_shot_proba)) == pytest.approx(0.0, abs=1e-9)


def test_warm_start_transfer_continues_the_existing_booster_not_replaces_it():
    source_model, _ = _fit_source_model()
    target_train, _, _ = _build_target(shift=True)
    sub_train = _subsample(target_train, 80)

    source_rounds = source_model.model.get_booster().num_boosted_rounds()
    warm_model = warm_start_transfer(source_model, sub_train, n_additional_estimators=30)
    continued_rounds = warm_model.model.get_booster().num_boosted_rounds()

    assert continued_rounds == source_rounds + 30


def test_train_source_model_and_train_from_scratch_are_reproducible():
    source_train, _, source_features = _build_source()
    model_a = train_source_model(source_train, source_features, n_estimators=20)
    model_b = train_source_model(source_train, source_features, n_estimators=20)

    target_train, _, target_features = _build_target(shift=True)
    scratch_a = train_from_scratch(target_train, target_features, n_estimators=20)
    scratch_b = train_from_scratch(target_train, target_features, n_estimators=20)

    np.testing.assert_array_equal(
        model_a.predict_proba(source_train), model_b.predict_proba(source_train)
    )
    np.testing.assert_array_equal(
        scratch_a.predict_proba(target_train), scratch_b.predict_proba(target_train)
    )


def test_train_source_model_rejects_single_class_training_frame():
    train_df = pd.DataFrame({"f1": np.arange(20, dtype=float), "isFraud": [0] * 20})
    with pytest.raises(ValueError, match="only one class"):
        train_source_model(train_df, ["f1"])


def test_train_from_scratch_rejects_empty_training_frame():
    empty_df = pd.DataFrame({"f1": [], "isFraud": []})
    with pytest.raises(ValueError, match="is empty"):
        train_from_scratch(empty_df, ["f1"])


def test_warm_start_transfer_rejects_non_positive_n_additional_estimators():
    source_model, _ = _fit_source_model()
    target_train, _, _ = _build_target(shift=True)
    with pytest.raises(ValueError, match="n_additional_estimators must be positive"):
        warm_start_transfer(source_model, target_train, n_additional_estimators=0)


def test_warm_start_transfer_rejects_empty_target_frame():
    source_model, source_features = _fit_source_model()
    empty_df = pd.DataFrame({col: [] for col in source_features + ["isFraud"]})
    with pytest.raises(ValueError, match="is empty"):
        warm_start_transfer(source_model, empty_df)


def test_apply_zero_shot_rejects_unfitted_source_model():
    _, target_test, _ = _build_target(shift=True)
    with pytest.raises(ValueError, match="must be a fitted DomainModel"):
        apply_zero_shot(None, target_test)


def test_warm_start_transfer_rejects_unfitted_source_model():
    target_train, _, _ = _build_target(shift=True)
    with pytest.raises(ValueError, match="must be a fitted DomainModel"):
        warm_start_transfer(None, target_train)


def test_mismatched_feature_columns_between_domains_raises():
    source_model, source_features = _fit_source_model()
    _, target_test, _ = _build_target(shift=True)
    target_missing_a_column = target_test.drop(columns=[source_features[0]])

    with pytest.raises(ValueError, match="missing"):
        apply_zero_shot(source_model, target_missing_a_column)


def test_domain_model_predict_proba_rejects_missing_columns_directly():
    source_model, source_features = _fit_source_model()
    _, target_test, _ = _build_target(shift=True)
    target_missing_a_column = target_test.drop(columns=[source_features[-1]])

    with pytest.raises(ValueError, match="missing"):
        source_model.predict_proba(target_missing_a_column)
