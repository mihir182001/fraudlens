"""
Tests for src.serving.model_registry.

Every test here uses its own temporary sqlite-backed MLflow tracking URI
(tmp_path), so none of them ever touch this project's real mlflow.db.
register_and_promote is tested against two real model shapes this project
actually produces: a plain scikit-learn LogisticRegression (the credit
scorecard's own model) needing no skops_trusted_types, and the fraud
model's imblearn Pipeline (SMOTE plus XGBoost) needing the same
skops_trusted_types list src.models.advanced_models already established,
matching what model_registry.py's own module docstring says was
empirically confirmed before that module was written.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.serving.model_registry import (
    CHAMPION_ALIAS,
    get_champion_version,
    list_versions,
    register_and_promote,
)


def _tracking_uri(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def _tiny_logistic_regression():
    x = np.random.RandomState(0).rand(40, 3)
    y = np.random.RandomState(0).randint(0, 2, 40)
    return LogisticRegression().fit(x, y)


def test_register_and_promote_creates_version_and_champion_alias(tmp_path):
    tracking_uri = _tracking_uri(tmp_path)
    model = _tiny_logistic_regression()

    result = register_and_promote(
        model,
        registered_model_name="test_model",
        params={"n_features": 3},
        metrics={"test_auc_roc": 0.75},
        tracking_uri=tracking_uri,
        experiment_name="test_experiment",
        run_name="test_run",
    )

    assert result["registered_model_name"] == "test_model"
    assert result["registered_model_version"] == 1
    assert result["alias"] == CHAMPION_ALIAS
    assert result["run_id"]

    champion = get_champion_version("test_model", tracking_uri)
    assert champion is not None
    assert int(champion.version) == 1

    versions = list_versions("test_model", tracking_uri)
    assert len(versions) == 1
    assert int(versions[0].version) == 1


def test_register_and_promote_moves_champion_alias_to_newest_version(tmp_path):
    tracking_uri = _tracking_uri(tmp_path)

    first = register_and_promote(
        _tiny_logistic_regression(), registered_model_name="test_model",
        params={}, metrics={"test_auc_roc": 0.70}, tracking_uri=tracking_uri,
        experiment_name="test_experiment", run_name="run_1",
    )
    second = register_and_promote(
        _tiny_logistic_regression(), registered_model_name="test_model",
        params={}, metrics={"test_auc_roc": 0.80}, tracking_uri=tracking_uri,
        experiment_name="test_experiment", run_name="run_2",
    )

    assert first["registered_model_version"] == 1
    assert second["registered_model_version"] == 2

    champion = get_champion_version("test_model", tracking_uri)
    assert int(champion.version) == 2

    versions = list_versions("test_model", tracking_uri)
    assert [int(v.version) for v in versions] == [2, 1]  # Newest first.


def test_get_champion_version_returns_none_for_unregistered_model(tmp_path):
    assert get_champion_version("does_not_exist", _tracking_uri(tmp_path)) is None


def test_list_versions_returns_empty_list_for_unregistered_model(tmp_path):
    assert list_versions("does_not_exist", _tracking_uri(tmp_path)) == []


def test_register_and_promote_handles_the_real_fraud_pipeline_shape(tmp_path):
    # Regression test for a real, empirically-confirmed requirement: MLflow
    # 3.x serializes scikit-learn-compatible models with skops, which
    # refuses to load a file back unless every non-default type inside it
    # was explicitly trusted at logging time. The fraud model's own
    # pipeline (imblearn's Pipeline wrapping SMOTE and XGBoost) is exactly
    # the shape src.models.advanced_models.SKOPS_TRUSTED_TYPES was already
    # built for. This confirms register_and_promote's own
    # skops_trusted_types passthrough actually works end to end for that
    # pipeline, not only for the simpler LogisticRegression case above.
    from src.models.advanced_models import SKOPS_TRUSTED_TYPES
    from src.serving.train_serving_models import train_fraud_serving_model

    artifact = train_fraud_serving_model(n_transactions=1200, seed=1)
    tracking_uri = _tracking_uri(tmp_path)

    result = register_and_promote(
        artifact["pipeline"], registered_model_name="test_fraud_model",
        params={"n_features": len(artifact["feature_columns"])},
        metrics={"test_auc_roc": artifact["test_auc_roc"]},
        tracking_uri=tracking_uri, experiment_name="test_experiment",
        run_name="fraud_run", skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )

    assert result["registered_model_version"] == 1
    champion = get_champion_version("test_fraud_model", tracking_uri)
    assert champion is not None

    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    loaded = mlflow.sklearn.load_model(f"models:/test_fraud_model/{result['registered_model_version']}")
    assert loaded is not None
