"""
model_registry.py, FastAPI/Streamlit serving module: registers a trained
model in MLflow's own Model Registry and promotes it with a "champion"
alias, on top of the mlflow.db this project's other training scripts
already write metrics to.

Why this exists on top of the joblib files train_serving_models.py already
writes: those files are what src/api/main.py actually loads at startup
(fast, no database round trip needed just to start serving), and that
stays unchanged here. What was missing before this module is version
history: every time train_serving_models.py's CLI is rerun, the previous
fraud_model.joblib or credit_scorecard.joblib was silently overwritten
with no record of what was serving before, and no link back to exactly
which MLflow run produced the file currently on disk. Registering each
trained model as a new MLflow Model Registry version, with the run's own
params and metrics attached, answers both: mlflow ui (pointed at the same
sqlite:///mlflow.db every other script here uses) shows the full version
history of each registered model, and the "champion" alias set on the
newest run's version is a single place that always means "the version
this project currently considers correct to serve," separate from merely
being the most recent version registered.

MLflow deprecated its older "stage" concept (Staging/Production/Archived)
in favor of named aliases, so "champion" here is a plain alias, not a
stage.

train_serving_models.py's own CLI calls register_and_promote after it
already has each artifact's trained model in hand, and embeds the
returned run_id, registered_model_name, and registered_model_version into
the same joblib dict before writing it to disk, so the file src/api/main.py
loads carries a durable pointer back to the exact MLflow run and registry
version that produced it, without src/api/main.py itself needing to talk
to MLflow at all.

Every function here was verified directly against a real, temporary
sqlite-backed MLflow store before being wired into train_serving_models.py:
both this project's plain scikit-learn LogisticRegression (the credit
scorecard's model) and its imblearn Pipeline with SMOTE and XGBoost (the
fraud model's pipeline) were confirmed to register, alias, and load back
successfully, the fraud pipeline needing the same skops_trusted_types list
src.models.advanced_models.SKOPS_TRUSTED_TYPES already established for
exactly this pipeline shape, and the credit model needing none at all.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

CHAMPION_ALIAS = "champion"


def register_and_promote(
    model: Any,
    registered_model_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    skops_trusted_types: Optional[List[str]] = None,
    alias: str = CHAMPION_ALIAS,
) -> Dict[str, Any]:
    """Logs model as a new MLflow run (with params/metrics attached) under
    experiment_name, registers it as a new version of registered_model_name
    in the Model Registry, and points `alias` (default "champion") at that
    new version, moving it off whichever version the alias pointed to
    before, if any. Returns a dict with run_id, registered_model_name,
    registered_model_version, and alias, meant to be persisted alongside
    the caller's own joblib artifact for lineage.

    skops_trusted_types is passed straight through to
    mlflow.sklearn.log_model: MLflow 3.x serializes scikit-learn-compatible
    models with skops, which refuses to LOAD a file back unless every
    non-default type inside it was explicitly trusted at logging time (see
    src.models.advanced_models.SKOPS_TRUSTED_TYPES and
    src.models.baseline_models.SKOPS_TRUSTED_TYPES for this project's own
    prior, directly-confirmed examples of exactly which types each kind of
    pipeline here needs). Left as None for a model, like this project's own
    plain LogisticRegression credit scorecard, that was confirmed not to
    need any.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    log_model_kwargs: Dict[str, Any] = {}
    if skops_trusted_types is not None:
        log_model_kwargs["skops_trusted_types"] = skops_trusted_types

    with mlflow.start_run(run_name=run_name) as run:
        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        model_info = mlflow.sklearn.log_model(
            model, name="model", registered_model_name=registered_model_name, **log_model_kwargs,
        )
        run_id = run.info.run_id

    client = MlflowClient(tracking_uri=tracking_uri)
    version = model_info.registered_model_version
    client.set_registered_model_alias(registered_model_name, alias, version)

    return {
        "run_id": run_id,
        "registered_model_name": registered_model_name,
        "registered_model_version": version,
        "alias": alias,
    }


def get_champion_version(
    registered_model_name: str, tracking_uri: str, alias: str = CHAMPION_ALIAS,
):
    """Returns the MLflow ModelVersion object currently aliased `alias`
    (default "champion") for registered_model_name, or None if that
    registered model or alias does not exist yet. A fresh mlflow.db, or one
    where this model has never been registered, has neither; that is a
    normal state, not an error, which is why this returns None instead of
    letting MlflowException propagate.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        return client.get_model_version_by_alias(registered_model_name, alias)
    except MlflowException:
        return None


def list_versions(registered_model_name: str, tracking_uri: str) -> List[Any]:
    """Returns every registered version of registered_model_name, newest
    version number first, as a list of MLflow ModelVersion objects. Empty
    list, not an error, if the registered model does not exist yet.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        versions = client.search_model_versions(f"name='{registered_model_name}'")
    except MlflowException:
        return []
    return sorted(versions, key=lambda v: int(v.version), reverse=True)
