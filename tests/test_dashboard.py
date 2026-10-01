"""
Tests for src.dashboard.app, using Streamlit's own AppTest harness
(streamlit.testing.v1), which runs the real script file exactly like
`streamlit run` does and lets a test inspect what it rendered.

Both AppTest-based tests point FRAUDLENS_MLFLOW_URI and FRAUDLENS_AUDIT_DB
at tmp_path locations (see the dashboard module's own docstring: both are
read fresh on every rerun), so neither test ever touches this project's
real mlflow.db or audit_log.db.

test_dashboard_module_is_importable_from_a_different_working_directory is a
real regression test for a real bug found by hand, not a hypothetical: a
first version of src/dashboard/app.py raised "ModuleNotFoundError: No
module named 'src'" the moment a real browser connected to `streamlit run
src/dashboard/app.py`, even though AppTest.from_file above ran the exact
same file without error. The two disagreed because AppTest executes the
script in-process, inside whatever pytest process already has this
project's repository root on sys.path (from how pytest itself was
launched); a real `streamlit run` (or plain `python src/dashboard/app.py`)
subprocess does not inherit that, and Python's own script-execution rule
puts the SCRIPT's own directory (src/dashboard/) on sys.path, not the
repository root, so `from src.serving...` had nothing to resolve against.
The fix (see app.py's own docstring) inserts the repository root into
sys.path at the top of the file before any project import. This test
spawns a real subprocess with its OWN working directory set somewhere
else entirely, which is exactly the condition that exposed the bug, so it
would fail again immediately if that fix ever regressed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mlflow
import pytest
from streamlit.testing.v1 import AppTest

from src.serving.audit_log import log_decision

# AppTest.from_file resolves a relative path against the file that CALLS
# it (this test file's own directory), not the current working directory,
# so this is built as an absolute path from this test file's own location.
APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "dashboard" / "app.py")


def test_dashboard_renders_empty_state_without_exceptions(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUDLENS_MLFLOW_URI", f"sqlite:///{tmp_path / 'empty_mlflow.db'}")
    monkeypatch.setenv("FRAUDLENS_AUDIT_DB", str(tmp_path / "empty_audit.db"))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    assert not at.exception
    info_messages = " ".join(info.value for info in at.info)
    assert "No model runs logged yet" in info_messages
    assert "No risk decisions logged yet" in info_messages


def test_dashboard_renders_populated_state_without_exceptions(tmp_path, monkeypatch):
    mlflow_uri = f"sqlite:///{tmp_path / 'populated_mlflow.db'}"
    audit_db_path = tmp_path / "populated_audit.db"

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("fraudlens_credit_scorecard")
    with mlflow.start_run(run_name="scorecard"):
        mlflow.log_metric("test_auc_roc", 0.739)
        mlflow.log_metric("test_auc_pr", 0.222)
        mlflow.log_metric("test_ks_statistic", 0.345)
        mlflow.log_metric("test_gini", 0.478)
    with mlflow.start_run(run_name="psi_monitoring"):
        mlflow.log_metric("score_psi", 0.4194)

    log_decision(
        model_type="credit", model_version="v1", score=580.0, decision="approve",
        input_payload={"AMT_INCOME_TOTAL": 100000.0}, db_path=audit_db_path,
    )
    log_decision(
        model_type="fraud", model_version="v1", score=0.91, decision="decline",
        input_payload={"amount": 999.0}, db_path=audit_db_path,
    )

    monkeypatch.setenv("FRAUDLENS_MLFLOW_URI", mlflow_uri)
    monkeypatch.setenv("FRAUDLENS_AUDIT_DB", str(audit_db_path))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    assert not at.exception
    assert len(at.dataframe) >= 2  # model comparison table + recent-decisions table.
    assert len(at.subheader) >= 1  # "Most recent decisions".
    # Exactly one psi_monitoring run was logged above, so this must take the
    # single-point st.metric path (see test_psi_history_with_one_point_uses_a_metric_not_a_chart
    # below for why a chart is wrong here), not silently show nothing.
    assert len(at.metric) == 1
    assert at.metric[0].value == "0.4194"


def test_psi_history_with_one_point_uses_a_metric_not_a_chart(tmp_path, monkeypatch):
    # Regression test for a real, directly-confirmed rendering bug: st.line_chart
    # draws nothing meaningful for a history of exactly one point (a line needs
    # two points) and was confirmed to render an empty plot with a stray,
    # mislabeled axis tick in that case, which reads as a wrong NUMBER rather
    # than a wrong chart choice. app.py works around this by showing a plain
    # st.metric instead whenever there is only one psi_monitoring run so far.
    mlflow_uri = f"sqlite:///{tmp_path / 'one_point_mlflow.db'}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("fraudlens_credit_scorecard")
    with mlflow.start_run(run_name="psi_monitoring"):
        mlflow.log_metric("score_psi", 0.4194)

    monkeypatch.setenv("FRAUDLENS_MLFLOW_URI", mlflow_uri)
    monkeypatch.setenv("FRAUDLENS_AUDIT_DB", str(tmp_path / "audit.db"))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    assert not at.exception
    assert len(at.metric) == 1
    assert at.metric[0].value == "0.4194"


def test_psi_history_with_two_points_uses_a_chart_not_a_metric(tmp_path, monkeypatch):
    mlflow_uri = f"sqlite:///{tmp_path / 'two_point_mlflow.db'}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("fraudlens_credit_scorecard")
    with mlflow.start_run(run_name="psi_monitoring"):
        mlflow.log_metric("score_psi", 0.10)
    with mlflow.start_run(run_name="psi_monitoring"):
        mlflow.log_metric("score_psi", 0.4194)

    monkeypatch.setenv("FRAUDLENS_MLFLOW_URI", mlflow_uri)
    monkeypatch.setenv("FRAUDLENS_AUDIT_DB", str(tmp_path / "audit.db"))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    assert not at.exception
    assert len(at.metric) == 0  # two points: the line chart path is used instead.


def test_dashboard_module_is_importable_from_a_different_working_directory(tmp_path):
    # Deliberately run with cwd set to tmp_path, nowhere near the repository
    # root, and invoke the file directly with plain python rather than the
    # streamlit CLI: this hits the exact same "script's own directory on
    # sys.path" condition a real `streamlit run` subprocess does, without
    # needing a real Streamlit server or a browser to reproduce it (Python
    # applies that sys.path rule to any directly executed script, streamlit
    # or not). A clean exit means the module-level imports resolved.
    result = subprocess.run(
        [sys.executable, str(Path(APP_PATH))],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "ModuleNotFoundError" not in result.stderr
    assert result.returncode == 0
