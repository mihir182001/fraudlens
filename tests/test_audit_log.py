"""
Tests for src.serving.audit_log. Every test uses tmp_path so nothing here
ever touches a real audit_log.db on disk.
"""

from __future__ import annotations

from src.serving.audit_log import decision_counts, fetch_recent_decisions, log_decision


def test_log_decision_and_fetch_recent_roundtrip(tmp_path):
    db_path = tmp_path / "audit.db"
    row_id = log_decision(
        model_type="fraud", model_version="2026-01-01T00:00:00+00:00",
        score=0.83, decision="decline", input_payload={"amount": 250.0},
        db_path=db_path, logged_at="2026-01-01T12:00:00+00:00",
    )
    assert row_id == 1

    recent = fetch_recent_decisions(db_path=db_path)
    assert len(recent) == 1
    row = recent.iloc[0]
    assert row["model_type"] == "fraud"
    assert row["decision"] == "decline"
    assert row["score"] == 0.83
    assert row["logged_at"] == "2026-01-01T12:00:00+00:00"
    assert "250.0" in row["input_json"]


def test_fetch_recent_decisions_returns_empty_frame_when_db_missing(tmp_path):
    db_path = tmp_path / "never_created.db"
    recent = fetch_recent_decisions(db_path=db_path)
    assert len(recent) == 0
    assert list(recent.columns) == [
        "id", "logged_at", "model_type", "model_version", "score", "decision", "input_json",
    ]


def test_decision_counts_returns_empty_frame_when_db_missing(tmp_path):
    db_path = tmp_path / "never_created.db"
    counts = decision_counts(db_path=db_path)
    assert len(counts) == 0
    assert list(counts.columns) == ["model_type", "decision", "count"]


def test_fetch_recent_decisions_respects_limit_and_newest_first_order(tmp_path):
    db_path = tmp_path / "audit.db"
    for i in range(5):
        log_decision(
            model_type="credit", model_version="v1", score=float(i), decision="approve",
            input_payload={"i": i}, db_path=db_path, logged_at=f"2026-01-0{i + 1}T00:00:00+00:00",
        )

    recent = fetch_recent_decisions(limit=3, db_path=db_path)
    assert len(recent) == 3
    assert list(recent["score"]) == [4.0, 3.0, 2.0]  # newest (highest id) first.


def test_fetch_recent_decisions_filters_by_model_type(tmp_path):
    db_path = tmp_path / "audit.db"
    log_decision(model_type="fraud", model_version="v1", score=0.9, decision="decline",
                 input_payload={}, db_path=db_path)
    log_decision(model_type="credit", model_version="v1", score=550.0, decision="approve",
                 input_payload={}, db_path=db_path)

    fraud_only = fetch_recent_decisions(db_path=db_path, model_type="fraud")
    assert len(fraud_only) == 1
    assert fraud_only.iloc[0]["model_type"] == "fraud"


def test_decision_counts_groups_by_model_type_and_decision(tmp_path):
    db_path = tmp_path / "audit.db"
    log_decision(model_type="fraud", model_version="v1", score=0.9, decision="decline",
                 input_payload={}, db_path=db_path)
    log_decision(model_type="fraud", model_version="v1", score=0.95, decision="decline",
                 input_payload={}, db_path=db_path)
    log_decision(model_type="fraud", model_version="v1", score=0.1, decision="approve",
                 input_payload={}, db_path=db_path)
    log_decision(model_type="credit", model_version="v1", score=550.0, decision="approve",
                 input_payload={}, db_path=db_path)

    counts = decision_counts(db_path=db_path)
    lookup = {(row["model_type"], row["decision"]): row["count"] for _, row in counts.iterrows()}
    assert lookup[("fraud", "decline")] == 2
    assert lookup[("fraud", "approve")] == 1
    assert lookup[("credit", "approve")] == 1
