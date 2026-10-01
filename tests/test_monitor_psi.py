"""
Tests for src.credit_risk.monitor_psi.

Each test trains a small, real credit scorecard artifact with
src.serving.train_serving_models.train_credit_serving_scorecard (the exact
function train_serving_models.py's CLI uses), saves it to a tmp_path, and
runs monitor_psi.main() against it with a temporary sqlite MLflow store, so
none of these ever touch this project's real models/credit_scorecard.joblib
or mlflow.db.

The "significant shift" and "stable" monitoring batches below reuse this
project's own already-established shift knobs from generate_synthetic_
credit.py (unemployment_shift, ext_source_shift): train_scorecard.py's PSI
demo already relies on its own default values of these (0.15, -0.12)
reliably producing a real "significant_shift" verdict (confirmed in this
project's own prior sessions), so the same defaults are reused here rather
than picked arbitrarily.
"""

from __future__ import annotations

import joblib
import mlflow
import pytest

from src.credit_risk.monitor_psi import EXIT_COULD_NOT_RUN, EXIT_OK, EXIT_SIGNIFICANT_SHIFT, main
from src.data.generate_synthetic_credit import generate_synthetic_credit_applications
from src.serving.train_serving_models import train_credit_serving_scorecard


def _write_artifact(tmp_path):
    artifact = train_credit_serving_scorecard(n_applications=1500, seed=1)
    model_path = tmp_path / "credit_scorecard.joblib"
    joblib.dump(artifact, model_path)
    return model_path


def test_exits_ok_for_a_stable_population(tmp_path):
    model_path = _write_artifact(tmp_path)
    csv_path = tmp_path / "stable_batch.csv"
    generate_synthetic_credit_applications(
        n_applications=1500, default_rate=0.08, seed=99, start_id=900001,
    ).to_csv(csv_path, index=False)

    with pytest.raises(SystemExit) as excinfo:
        main([
            "--model-path", str(model_path),
            "--input-csv", str(csv_path),
            "--skip-mlflow",
        ])

    assert excinfo.value.code == EXIT_OK


def test_exits_significant_shift_for_a_deliberately_shifted_population(tmp_path):
    model_path = _write_artifact(tmp_path)
    csv_path = tmp_path / "shifted_batch.csv"
    generate_synthetic_credit_applications(
        n_applications=1500, default_rate=0.08, seed=99, start_id=900001,
        unemployment_shift=0.15, ext_source_shift=-0.12,
    ).to_csv(csv_path, index=False)

    with pytest.raises(SystemExit) as excinfo:
        main([
            "--model-path", str(model_path),
            "--input-csv", str(csv_path),
            "--skip-mlflow",
        ])

    assert excinfo.value.code == EXIT_SIGNIFICANT_SHIFT


def test_exits_could_not_run_when_model_file_is_missing(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main([
            "--model-path", str(tmp_path / "does_not_exist.joblib"),
            "--skip-mlflow",
        ])

    assert excinfo.value.code == EXIT_COULD_NOT_RUN


def test_exits_could_not_run_when_artifact_has_no_reference_scores(tmp_path):
    # A model trained before this feature existed: exercises the explicit,
    # disclosed error path rather than an obscure KeyError.
    model_path = tmp_path / "old_style_credit_scorecard.joblib"
    artifact = train_credit_serving_scorecard(n_applications=400, seed=1)
    del artifact["reference_scores"]
    joblib.dump(artifact, model_path)

    with pytest.raises(SystemExit) as excinfo:
        main(["--model-path", str(model_path), "--skip-mlflow"])

    assert excinfo.value.code == EXIT_COULD_NOT_RUN


def test_generates_a_synthetic_batch_and_logs_to_mlflow_when_no_csv_given(tmp_path):
    model_path = _write_artifact(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    with pytest.raises(SystemExit) as excinfo:
        main([
            "--model-path", str(model_path),
            "--n-applications", "500",
            "--mlflow-tracking-uri", tracking_uri,
        ])

    assert excinfo.value.code in (EXIT_OK, EXIT_SIGNIFICANT_SHIFT)

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("fraudlens_credit_scorecard")
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert "metrics.score_psi" in runs.columns
    logged_psi = runs.iloc[0]["metrics.score_psi"]
    assert logged_psi == logged_psi  # Not NaN.
    assert logged_psi >= 0.0
