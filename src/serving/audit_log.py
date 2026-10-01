"""
audit_log.py, FastAPI/Streamlit serving module: a durable, queryable record
of every risk decision this project's API makes, the "risk decision audit
logging" item from the project spec.

Real credit and fraud risk audit trails exist for a genuine regulatory
reason: an institution has to be able to show, after the fact, exactly what
a model saw and decided for a specific applicant or transaction, which
model version made that call, and when. This module reproduces the shape of
that requirement (one durable row per decision, queryable by model type and
time) at a scale appropriate to a portfolio demo, using a plain sqlite
database rather than a new dependency, the same choice this project already
made for MLflow's own tracking store.

Two real simplifications worth stating plainly rather than glossing over:
  - This logs the full input payload as JSON. That is acceptable ONLY
    because every input this project ever scores is synthetic data with no
    real person behind it (see generate_synthetic_credit.py and
    generate_synthetic_ieee.py). A real production audit log handling real
    applicants would need to mask or hash personally identifying fields
    before writing them anywhere, which this module deliberately does not
    implement, since there is no real PII in this project to protect.
  - This is an ordinary, mutable sqlite table: a row can technically still
    be altered or deleted by anyone with file access to audit_log.db. A
    real regulated audit trail typically needs to be append-only and
    tamper-evident (write-once storage, or a cryptographic chain over
    rows), which is a real infrastructure concern beyond this project's
    scope, not something this module claims to provide.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

DEFAULT_AUDIT_DB_PATH = Path("audit_log.db")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS risk_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT NOT NULL,
    model_type TEXT NOT NULL,
    model_version TEXT NOT NULL,
    score REAL NOT NULL,
    decision TEXT NOT NULL,
    input_json TEXT NOT NULL
)
"""


def init_audit_db(db_path: Path = DEFAULT_AUDIT_DB_PATH) -> None:
    """Creates the risk_decisions table if it does not already exist.
    Safe to call every time the API starts up; never drops or alters an
    existing table.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_TABLE_SQL)


def log_decision(
    model_type: str,
    model_version: str,
    score: float,
    decision: str,
    input_payload: Dict[str, Any],
    db_path: Path = DEFAULT_AUDIT_DB_PATH,
    logged_at: Optional[str] = None,
) -> int:
    """Appends one risk decision row and returns its new row id.

    model_type is conventionally "fraud" or "credit"; model_version is
    whatever the serving artifact recorded as its own trained_at timestamp
    (see train_serving_models.py), standing in for a real model registry
    version number. logged_at defaults to the current UTC time in ISO 8601
    format if not supplied (a caller can pass one explicitly for
    reproducible tests).
    """
    if logged_at is None:
        logged_at = datetime.now(timezone.utc).isoformat()

    init_audit_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO risk_decisions "
            "(logged_at, model_type, model_version, score, decision, input_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (logged_at, model_type, model_version, float(score), decision, json.dumps(input_payload)),
        )
        conn.commit()
        return int(cursor.lastrowid)


def fetch_recent_decisions(
    limit: int = 50, db_path: Path = DEFAULT_AUDIT_DB_PATH, model_type: Optional[str] = None,
) -> pd.DataFrame:
    """Returns the most recent `limit` logged decisions (newest first) as a
    DataFrame with columns [id, logged_at, model_type, model_version,
    score, decision, input_json]. Optionally filtered to one model_type.
    Returns an empty DataFrame (same columns, zero rows) if the database or
    table does not exist yet, rather than raising, since "no decisions
    logged yet" is a normal state for a freshly started API.
    """
    columns = ["id", "logged_at", "model_type", "model_version", "score", "decision", "input_json"]
    if not Path(db_path).exists():
        return pd.DataFrame(columns=columns)

    init_audit_db(db_path)
    query = "SELECT * FROM risk_decisions"
    params: tuple = ()
    if model_type is not None:
        query += " WHERE model_type = ?"
        params = (model_type,)
    query += " ORDER BY id DESC LIMIT ?"
    params = params + (limit,)

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)


def decision_counts(db_path: Path = DEFAULT_AUDIT_DB_PATH) -> pd.DataFrame:
    """Returns a DataFrame with columns [model_type, decision, count],
    one row per (model_type, decision) pair actually logged so far, for a
    dashboard's summary view. Empty (same columns) if nothing logged yet.
    """
    columns = ["model_type", "decision", "count"]
    if not Path(db_path).exists():
        return pd.DataFrame(columns=columns)

    init_audit_db(db_path)
    with sqlite3.connect(db_path) as conn:
        result = pd.read_sql_query(
            "SELECT model_type, decision, COUNT(*) AS count FROM risk_decisions "
            "GROUP BY model_type, decision ORDER BY model_type, decision",
            conn,
        )
    return result
