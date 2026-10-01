"""Tests for src.models.baseline_models.

Uses a small synthetic dataset built so the target is genuinely learnable
from the features (not random), so a passing "the model actually learns
something" test is checking real signal, not just that the code runs.
MLflow logging is exercised against a temporary local sqlite tracking
database (never the projects real mlflow.db), so these tests never depend
on or pollute any real experiment history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baseline_models import (
    SKOPS_TRUSTED_TYPES,
    build_pipeline,
    run_all_baselines,
    train_and_evaluate,
)


def _learnable_sample(n_per_class: int = 60, seed: int = 0) -> pd.DataFrame:
    """A small, genuinely learnable dataset: signal is drawn from a
    clearly different distribution per class, plus a noise column with no
    relationship to the label, plus a NaN-containing column (to exercise
    the imputer) that also carries a little real signal.
    """
    rng = np.random.default_rng(seed)
    n = n_per_class * 2
    is_fraud = np.array([0] * n_per_class + [1] * n_per_class)
    signal = rng.normal(loc=0.0, scale=1.0, size=n) + is_fraud * 4.0
    noise = rng.normal(loc=0.0, scale=1.0, size=n)
    with_gaps = rng.normal(loc=0.0, scale=1.0, size=n) + is_fraud * 3.0
    with_gaps[rng.random(n) < 0.1] = np.nan

    df = pd.DataFrame({
        "isFraud": is_fraud,
        "signal": signal,
        "noise": noise,
        "with_gaps": with_gaps,
    })
    # Shuffle row order so the model does not trivially see all-0-then-all-1.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_build_pipeline_returns_expected_steps_for_each_model():
    for name in ("logistic_regression", "decision_tree", "random_forest"):
        pipeline = build_pipeline(name)
        assert "imputer" in pipeline.named_steps
        assert "model" in pipeline.named_steps
    assert "scaler" in build_pipeline("logistic_regression").named_steps
    assert "scaler" not in build_pipeline("decision_tree").named_steps


def test_build_pipeline_rejects_unknown_model_name():
    with pytest.raises(ValueError):
        build_pipeline("not_a_real_model")


@pytest.mark.parametrize("model_name", ["logistic_regression", "decision_tree", "random_forest"])
def test_train_and_evaluate_learns_real_signal(model_name):
    df = _learnable_sample()
    train_df = df.iloc[:80].reset_index(drop=True)
    test_df = df.iloc[80:].reset_index(drop=True)

    result = train_and_evaluate(
        model_name, train_df, test_df, feature_columns=["signal", "noise", "with_gaps"]
    )

    # A clearly separable signal should give a strong, not a coin-flip,
    # test AUC-ROC. 0.8 is a deliberately generous floor, not a tuned
    # target: the point is confirming the pipeline actually learns from
    # real signal, not pinning an exact score.
    assert result["test_auc_roc"] > 0.8
    assert result["n_train"] == 80
    assert result["n_test"] == 40
    assert result["n_features"] == 3


def test_train_and_evaluate_reports_traceable_counts():
    df = _learnable_sample()
    train_df = df.iloc[:80].reset_index(drop=True)
    test_df = df.iloc[80:].reset_index(drop=True)
    result = train_and_evaluate(
        "logistic_regression", train_df, test_df, feature_columns=["signal"]
    )
    assert result["n_train_fraud"] == int(train_df["isFraud"].sum())
    assert result["n_test_fraud"] == int(test_df["isFraud"].sum())


def test_train_and_evaluate_raises_when_train_has_one_class():
    df = _learnable_sample()
    single_class_train = df[df["isFraud"] == 0].reset_index(drop=True)
    test_df = df.iloc[:10].reset_index(drop=True)
    with pytest.raises(ValueError, match="Training set"):
        train_and_evaluate(
            "logistic_regression", single_class_train, test_df, feature_columns=["signal"]
        )


def test_train_and_evaluate_raises_when_test_has_one_class():
    df = _learnable_sample()
    train_df = df.iloc[:80].reset_index(drop=True)
    single_class_test = df[df["isFraud"] == 0].iloc[:10].reset_index(drop=True)
    with pytest.raises(ValueError, match="Test set"):
        train_and_evaluate(
            "logistic_regression", train_df, single_class_test, feature_columns=["signal"]
        )


def test_run_all_baselines_returns_all_three_models_in_spec_order():
    df = _learnable_sample()
    train_df = df.iloc[:80].reset_index(drop=True)
    test_df = df.iloc[80:].reset_index(drop=True)
    results = run_all_baselines(train_df, test_df, feature_columns=["signal", "noise", "with_gaps"])
    assert [r["model_name"] for r in results] == [
        "logistic_regression", "decision_tree", "random_forest",
    ]
    assert all("test_auc_roc" in r for r in results)


def test_mlflow_logging_round_trip(tmp_path):
    """Exercises the actual MLflow logging path (not just the training
    code) against a temporary local sqlite tracking database, confirming a
    run can be logged and then read back with the metric we wrote.

    A plain filesystem store (the "file:..." URI style) is used nowhere in
    this project: mlflow 3.16.1, the version actually pinned in
    requirements.txt, was confirmed directly (by running this exact test
    against a file: URI first) to raise MlflowException for that backend,
    since MLflow has put the filesystem store into maintenance mode. A
    local sqlite database is just as dependency-free and server-free, so
    that is what this project uses everywhere it logs to MLflow.
    """
    mlflow = pytest.importorskip("mlflow")

    df = _learnable_sample()
    train_df = df.iloc[:80].reset_index(drop=True)
    test_df = df.iloc[80:].reset_index(drop=True)
    result = train_and_evaluate(
        "logistic_regression", train_df, test_df, feature_columns=["signal", "noise"]
    )

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow_test.db")
    mlflow.set_experiment("fraudlens_test_experiment")
    with mlflow.start_run(run_name="logistic_regression_test"):
        mlflow.log_param("model_name", "logistic_regression")
        mlflow.log_metric("test_auc_roc", result["test_auc_roc"])

    runs = mlflow.search_runs(experiment_names=["fraudlens_test_experiment"])
    assert len(runs) == 1
    assert runs.iloc[0]["metrics.test_auc_roc"] == pytest.approx(result["test_auc_roc"])


@pytest.mark.parametrize("model_name", ["logistic_regression", "decision_tree", "random_forest"])
def test_mlflow_log_model_round_trip_for_every_model_type(tmp_path, model_name):
    """Exercises mlflow.sklearn.log_model itself, not just log_param and
    log_metric, for every model type this project trains.

    This is its own test rather than folded into the round trip above
    because mlflow 3.16.1 was found (by actually running the real pipeline
    end to end, not just this files earlier, narrower tests) to serialize
    sklearn models with skops, which refuses to load a saved model back
    unless every type inside it is explicitly trusted. Decision trees and
    random forests store an extra type (their fitted tree structure) that
    logistic regression does not, so all three model types are checked
    here rather than assuming logistic regressions success covers them.
    """
    mlflow = pytest.importorskip("mlflow")

    df = _learnable_sample()
    train_df = df.iloc[:80].reset_index(drop=True)
    test_df = df.iloc[80:].reset_index(drop=True)
    result = train_and_evaluate(
        model_name, train_df, test_df, feature_columns=["signal", "noise", "with_gaps"]
    )

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow_model_test.db")
    mlflow.set_experiment("fraudlens_model_log_test")
    with mlflow.start_run(run_name=f"{model_name}_model_log_test"):
        model_info = mlflow.sklearn.log_model(
            result["pipeline"], name="model", skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    loaded_pipeline = mlflow.sklearn.load_model(model_info.model_uri)
    x_test = test_df[["signal", "noise", "with_gaps"]]
    loaded_proba = loaded_pipeline.predict_proba(x_test)[:, 1]
    original_proba = result["pipeline"].predict_proba(x_test)[:, 1]
    assert (loaded_proba == original_proba).all()
