"""
app.py, Streamlit serving module: the "model comparison dashboard" item
from the project spec, plus a view onto the risk decision audit log
src/api/main.py writes to.

Deliberately decoupled from the FastAPI service: this dashboard reads
directly from the two durable stores every earlier week already writes to
(mlflow.db for every model's own logged metrics, audit_log.db for every
scored decision), rather than calling the API itself. That mirrors how a
real monitoring dashboard usually works, reading from a shared data store
rather than depending on a specific service instance being up, and means
this page still works even if the FastAPI service is not currently running.

FRAUDLENS_MLFLOW_URI and FRAUDLENS_AUDIT_DB (both read fresh at the top of
each Streamlit rerun, matching src/api/main.py's own env-var pattern) let
tests point this script at a temporary tracking store and audit database.

Run locally with:
    streamlit run src/dashboard/app.py

The sys.path fixup below is required, not defensive boilerplate: `streamlit
run <path>` executes this file the same way plain `python src/dashboard/
app.py` would, which makes Python treat src/dashboard/ (this file's own
directory) as the import root, NOT the repository root. Without it,
`from src.serving...` fails with "ModuleNotFoundError: No module named
'src'" the moment a real browser connects and the script actually runs
(confirmed directly: this only surfaces once a client triggers a real run,
which is why a plain server-started/curl-based check can miss it, and why
`python -m src.api.main`/pytest/uvicorn never hit this, since each of those
already puts the repository root on sys.path in a way `streamlit run`'s own
plain-script execution does not).

A second real, directly-confirmed rendering issue: st.line_chart draws
nothing meaningful for a history of exactly one point (a line needs two
points), and was confirmed to render an empty plot with a stray,
mislabeled axis tick in that case rather than the real logged value. render()
below checks for this (len(psi_history) == 1) and shows a plain st.metric
instead, since a single run has no trend to plot in the first place.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mlflow
import pandas as pd
import streamlit as st

from src.credit_risk.psi_monitoring import psi_verdict
from src.serving.audit_log import DEFAULT_AUDIT_DB_PATH, decision_counts, fetch_recent_decisions

MODEL_EXPERIMENTS = [
    "fraudlens_baseline_models",
    "fraudlens_advanced_models",
    "fraudlens_credit_scorecard",
]
METRIC_COLUMNS = [
    "metrics.test_auc_roc", "metrics.test_auc_pr", "metrics.test_ks_statistic", "metrics.test_gini",
]


def _mlflow_uri() -> str:
    return os.environ.get("FRAUDLENS_MLFLOW_URI", "sqlite:///mlflow.db")


def _audit_db_path():
    return os.environ.get("FRAUDLENS_AUDIT_DB", str(DEFAULT_AUDIT_DB_PATH))


def load_model_comparison() -> pd.DataFrame:
    """Returns one row per MLflow run across every model-training
    experiment this project has, with a run's own experiment name and
    run name attached, and only the metric columns this dashboard displays.
    Empty (no columns assumed) if no experiment has been run yet.
    """
    mlflow.set_tracking_uri(_mlflow_uri())
    client = mlflow.tracking.MlflowClient()

    frames = []
    for experiment_name in MODEL_EXPERIMENTS:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            continue
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if runs.empty:
            continue
        runs = runs.copy()
        runs["experiment"] = experiment_name
        runs["run_name"] = runs.get("tags.mlflow.runName", pd.Series(dtype=str))
        frames.append(runs)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    present_metric_columns = [c for c in METRIC_COLUMNS if c in combined.columns]
    # A run that never logged one of these metrics (e.g. a psi_monitoring
    # run, which only logs score_psi) legitimately has no value for it;
    # such rows are kept but will show blank in that column rather than
    # being dropped, since they are still real, relevant model-comparison
    # rows for whichever metrics they DID log.
    return combined[["experiment", "run_name", "start_time"] + present_metric_columns].sort_values(
        "start_time", ascending=False,
    )


def load_psi_history() -> pd.DataFrame:
    """Returns every logged score_psi value from the credit scorecard
    experiment's psi_monitoring runs, in chronological order, so repeated
    runs of train_scorecard.py show up as a real history rather than only
    the latest number.
    """
    mlflow.set_tracking_uri(_mlflow_uri())
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("fraudlens_credit_scorecard")
    if experiment is None:
        return pd.DataFrame()

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty or "metrics.score_psi" not in runs.columns:
        return pd.DataFrame()

    psi_runs = runs[runs["metrics.score_psi"].notna()].copy()
    if psi_runs.empty:
        return pd.DataFrame()

    psi_runs["start_time"] = pd.to_datetime(psi_runs["start_time"])
    return psi_runs[["start_time", "metrics.score_psi"]].sort_values("start_time")


def render() -> None:
    st.set_page_config(page_title="FraudLens Risk Monitoring", layout="wide")
    st.title("FraudLens Risk Monitoring")
    st.caption(
        "Model comparison across every trained fraud and credit model, and "
        "the risk decision audit trail logged by the scoring API. All "
        "numbers come from this project's own MLflow tracking store and "
        "audit log; nothing on this page is a live model call."
    )

    st.header("Model comparison")
    comparison = load_model_comparison()
    if comparison.empty:
        st.info(
            "No model runs logged yet. Run src/models/baseline_models.py, "
            "src/models/advanced_models.py, or "
            "src/credit_risk/train_scorecard.py first."
        )
    else:
        st.dataframe(comparison, width="stretch")
        if "metrics.test_auc_roc" in comparison.columns:
            auc_chart = comparison.dropna(subset=["metrics.test_auc_roc"]).set_index("run_name")[
                "metrics.test_auc_roc"
            ]
            if not auc_chart.empty:
                st.bar_chart(auc_chart)

    st.header("PSI monitoring history")
    psi_history = load_psi_history()
    if psi_history.empty:
        st.info(
            "No PSI monitoring runs logged yet. Run "
            "src/credit_risk/train_scorecard.py to generate one."
        )
    elif len(psi_history) == 1:
        # A line chart of exactly one point draws nothing meaningful (a
        # line needs two points), and Streamlit/Altair's own single-point
        # line_chart rendering has been confirmed directly to show an
        # empty plot with a stray, mislabeled axis tick instead of the
        # real value, which reads as a bug in the number rather than in
        # the chart choice. A single run has no trend to show anyway, so
        # it is shown as a plain number instead, which is both correct and
        # cannot misrender.
        latest_psi = float(psi_history["metrics.score_psi"].iloc[0])
        st.metric("Latest PSI (only one run logged so far)", f"{latest_psi:.4f}", help=psi_verdict(latest_psi))
    else:
        st.line_chart(psi_history.set_index("start_time")["metrics.score_psi"])
        st.caption(
            "PSI >= 0.25 (see src/credit_risk/psi_monitoring.py) is "
            "conventionally read as a significant population shift, worth "
            "a fresh scorecard validation."
        )

    st.header("Risk decision audit log")
    counts = decision_counts(db_path=_audit_db_path())
    if counts.empty:
        st.info(
            "No risk decisions logged yet. Run the FastAPI service "
            "(src/api/main.py) and call /score/fraud or /score/credit."
        )
    else:
        pivot = counts.pivot(index="model_type", columns="decision", values="count").fillna(0)
        st.bar_chart(pivot)

        recent = fetch_recent_decisions(limit=200, db_path=_audit_db_path())
        st.subheader("Most recent decisions")
        st.dataframe(
            recent[["logged_at", "model_type", "model_version", "score", "decision"]],
            width="stretch",
        )


# Streamlit executes this whole script top-to-bottom on every rerun
# regardless of __name__ (unlike a plain Python script run standalone), so
# render() is called unconditionally here; this is also what lets
# streamlit.testing.v1.AppTest, which runs this file the same way `streamlit
# run` does, exercise the exact same page in tests.
render()
