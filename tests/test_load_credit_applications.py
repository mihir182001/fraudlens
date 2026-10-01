"""
Tests for src.data.load_credit_applications.

The real application_train.csv is not available in this environment (see
generate_synthetic_credit.py's own module docstring for why), so these
tests build a stand-in CSV that mimics the real file's actual shape: the
same 18 reduced columns this project's pipeline uses, PLUS extra columns a
real 122-column file would have but this project does not, to confirm the
subsetting logic actually drops them rather than merely working by
coincidence on a file that happens to already be the reduced shape.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.generate_synthetic_credit import (
    APPLICATION_FEATURE_COLUMNS,
    generate_synthetic_credit_applications,
)
from src.data.load_credit_applications import load_credit_applications


def _write_wide_csv(tmp_path, df: pd.DataFrame, extra_columns=None):
    wide = df.copy()
    for name, value in (extra_columns or {}).items():
        wide[name] = value
    path = tmp_path / "application_train.csv"
    wide.to_csv(path, index=False)
    return path


def test_loads_and_subsets_a_wider_real_shaped_file(tmp_path):
    df = generate_synthetic_credit_applications(n_applications=200, seed=1)
    path = _write_wide_csv(
        tmp_path, df,
        extra_columns={
            "FLAG_MOBIL": 1,
            "FLAG_EMP_PHONE": 1,
            "OWN_CAR_AGE": 5.0,
        },
    )

    loaded = load_credit_applications(path)

    assert list(loaded.columns) == ["SK_ID_CURR", "TARGET"] + APPLICATION_FEATURE_COLUMNS
    assert "FLAG_MOBIL" not in loaded.columns
    assert len(loaded) == 200


def test_raises_file_not_found_for_a_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="Credit application file not found"):
        load_credit_applications(tmp_path / "does_not_exist.csv")


def test_raises_value_error_for_missing_required_columns(tmp_path):
    df = generate_synthetic_credit_applications(n_applications=50, seed=1)
    df = df.drop(columns=["EXT_SOURCE_1"])
    path = tmp_path / "application_train.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required column"):
        load_credit_applications(path)


def test_raises_value_error_for_duplicate_sk_id_curr(tmp_path):
    df = generate_synthetic_credit_applications(n_applications=50, seed=1)
    duplicated = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    path = tmp_path / "application_train.csv"
    duplicated.to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate SK_ID_CURR"):
        load_credit_applications(path)
