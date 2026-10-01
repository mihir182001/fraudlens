"""
Tests for src.models.advanced_models.

The synthetic data helper below places the minority (fraud) class on a
fixed, evenly spaced pattern rather than shuffling it in randomly. This
matters specifically for the tuning tests: RandomizedSearchCVs
TimeSeriesSplit folds are contiguous chunks of the data in row order, and
SMOTE needs at least a handful of minority rows in every training fold it
runs on (its default nearest-neighbor search needs more neighbors than
that to work at all). An evenly spaced pattern guarantees every contiguous
chunk above a small minimum size has its fair share of minority rows,
so these tests exercise the real SMOTE-plus-cross-validation path
without depending on a random shuffle happening to cooperate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.advanced_models import (
    LIGHTGBM_PARAM_DISTRIBUTIONS,
    SKOPS_TRUSTED_TYPES,
    XGBOOST_PARAM_DISTRIBUTIONS,
    build_advanced_pipeline,
    run_all_advanced_models,
    train_and_evaluate_advanced,
    tune_hyperparameters,
)

FEATURE_COLUMNS = ["signal", "noise"]


def _interleaved_imbalanced_sample(n: int = 300, minority_period: int = 4, seed: int = 0) -> pd.DataFrame:
    """A learnable, imbalanced dataset with the minority class placed on a
    fixed period (every minority_period-th row), not shuffled, so any
    contiguous block of rows has a predictable, even share of the minority
    class. signal is shifted for the minority class and is real, learnable
    separation; noise carries no information about the label.
    """
    rng = np.random.default_rng(seed)
    is_fraud = np.array([1 if i % minority_period == 0 else 0 for i in range(n)])
    signal = rng.normal(loc=0.0, scale=1.0, size=n) + is_fraud * 4.0
    noise = rng.normal(loc=0.0, scale=1.0, size=n)
    return pd.DataFrame({"isFraud": is_fraud, "signal": signal, "noise": noise})


@pytest.mark.parametrize("model_name", ["xgboost", "lightgbm"])
@pytest.mark.parametrize("use_smote", [True, False])
def test_build_advanced_pipeline_returns_expected_steps(model_name, use_smote):
    pipeline = build_advanced_pipeline(model_name, use_smote=use_smote)
    assert "imputer" in pipeline.named_steps
    assert "model" in pipeline.named_steps
    assert ("smote" in pipeline.named_steps) == use_smote


def test_build_advanced_pipeline_rejects_unknown_model_name():
    with pytest.raises(ValueError):
        build_advanced_pipeline("not_a_real_model", use_smote=True)


@pytest.mark.parametrize("model_name", ["xgboost", "lightgbm"])
@pytest.mark.parametrize("use_smote", [True, False])
def test_train_and_evaluate_advanced_learns_real_signal_untuned(model_name, use_smote):
    df = _interleaved_imbalanced_sample(n=200)
    train_df = df.iloc[:150].reset_index(drop=True)
    test_df = df.iloc[150:].reset_index(drop=True)

    result = train_and_evaluate_advanced(
        model_name, train_df, test_df, FEATURE_COLUMNS,
        use_smote=use_smote, tune=False,
    )

    # A clearly separable signal should give a strong test AUC-ROC. 0.75 is
    # a deliberately generous floor, not a tuned target.
    assert result["test_auc_roc"] > 0.75
    assert 0.0 <= result["test_auc_pr"] <= 1.0
    assert 0.0 <= result["test_ks_statistic"] <= 1.0
    assert -1.0 <= result["test_gini"] <= 1.0
    assert result["n_train"] == 150
    assert result["n_test"] == 50
    assert result["use_smote"] == use_smote
    assert result["tuned"] is False
    assert result["best_params"] is None


def test_train_and_evaluate_advanced_raises_when_train_has_one_class():
    df = _interleaved_imbalanced_sample(n=200)
    single_class_train = df[df["isFraud"] == 0].reset_index(drop=True)
    test_df = df.iloc[:20].reset_index(drop=True)
    with pytest.raises(ValueError, match="Training set"):
        train_and_evaluate_advanced(
            "xgboost", single_class_train, test_df, FEATURE_COLUMNS, tune=False,
        )


def test_train_and_evaluate_advanced_raises_when_test_has_one_class():
    df = _interleaved_imbalanced_sample(n=200)
    train_df = df.iloc[:150].reset_index(drop=True)
    single_class_test = df[df["isFraud"] == 0].iloc[:10].reset_index(drop=True)
    with pytest.raises(ValueError, match="Test set"):
        train_and_evaluate_advanced(
            "xgboost", train_df, single_class_test, FEATURE_COLUMNS, tune=False,
        )


@pytest.mark.parametrize("model_name", ["xgboost", "lightgbm"])
def test_tune_hyperparameters_returns_fitted_search_with_expected_params(model_name):
    df = _interleaved_imbalanced_sample(n=200)
    train_df = df.iloc[:150].reset_index(drop=True)
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["isFraud"]

    search = tune_hyperparameters(
        model_name, x_train, y_train, use_smote=False, n_iter=2, n_splits=2,
    )

    expected_keys = set(
        (XGBOOST_PARAM_DISTRIBUTIONS if model_name == "xgboost" else LIGHTGBM_PARAM_DISTRIBUTIONS).keys()
    )
    assert hasattr(search, "best_estimator_")
    assert set(search.best_params_.keys()) == expected_keys


def test_train_and_evaluate_advanced_runs_tuning_with_smote():
    # Exercises the real SMOTE-inside-cross-validation path: every
    # RandomizedSearchCV fold must have enough minority rows for SMOTEs
    # neighbor search to succeed, which is exactly what the evenly spaced
    # minority pattern in _interleaved_imbalanced_sample guarantees.
    df = _interleaved_imbalanced_sample(n=300)
    train_df = df.iloc[:240].reset_index(drop=True)
    test_df = df.iloc[240:].reset_index(drop=True)

    result = train_and_evaluate_advanced(
        "xgboost", train_df, test_df, FEATURE_COLUMNS,
        use_smote=True, tune=True, n_iter=2, n_splits=2,
    )

    assert result["tuned"] is True
    assert result["use_smote"] is True
    assert result["best_params"] is not None
    assert set(result["best_params"].keys()) == set(XGBOOST_PARAM_DISTRIBUTIONS.keys())
    assert result["test_auc_roc"] > 0.5


def test_run_all_advanced_models_returns_floor_and_tuned_for_both_models():
    df = _interleaved_imbalanced_sample(n=300)
    train_df = df.iloc[:240].reset_index(drop=True)
    test_df = df.iloc[240:].reset_index(drop=True)

    results = run_all_advanced_models(
        train_df, test_df, FEATURE_COLUMNS, tune=True, n_iter=2, n_splits=2,
    )

    assert len(results) == 4
    configs = [(r["model_name"], r["use_smote"], r["tuned"]) for r in results]
    assert configs == [
        ("xgboost", False, False),
        ("xgboost", True, True),
        ("lightgbm", False, False),
        ("lightgbm", True, True),
    ]


@pytest.mark.parametrize("model_name", ["xgboost", "lightgbm"])
@pytest.mark.parametrize("use_smote", [True, False])
def test_mlflow_log_model_round_trip_for_advanced_models(tmp_path, model_name, use_smote):
    """Exercises mlflow.sklearn.log_model for every advanced-model pipeline
    variant this project trains, confirming SKOPS_TRUSTED_TYPES is
    actually sufficient for imblearn's SMOTE step plus XGBoost/LightGBM's
    own fitted model objects, not just for the plainer Week 3 models.
    """
    mlflow = pytest.importorskip("mlflow")

    df = _interleaved_imbalanced_sample(n=200)
    train_df = df.iloc[:150].reset_index(drop=True)
    test_df = df.iloc[150:].reset_index(drop=True)
    result = train_and_evaluate_advanced(
        model_name, train_df, test_df, FEATURE_COLUMNS,
        use_smote=use_smote, tune=False,
    )

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow_advanced_test.db")
    mlflow.set_experiment("fraudlens_advanced_model_log_test")
    with mlflow.start_run(run_name=f"{model_name}_{use_smote}_model_log_test"):
        model_info = mlflow.sklearn.log_model(
            result["pipeline"], name="model", skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    loaded_pipeline = mlflow.sklearn.load_model(model_info.model_uri)
    x_test = test_df[FEATURE_COLUMNS]
    loaded_proba = loaded_pipeline.predict_proba(x_test)[:, 1]
    original_proba = result["pipeline"].predict_proba(x_test)[:, 1]
    assert np.allclose(loaded_proba, original_proba)
