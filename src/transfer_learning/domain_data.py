"""
domain_data.py, Week 8: builds one domain's engineered train/test feature
set for transfer learning, by calling Week 1's synthetic generator and
Week 1/3's merge and feature-preparation functions once per domain with
different parameters, rather than splitting one dataset in half.

Two calls to build_domain with different n_transactions/fraud_rate/seed
produce two genuinely separate transaction streams: different card pools,
different realized fraud patterns, different chronological ranges. Each
domain gets its own independent chronological train/test split and its own
independently-fit hour-of-day aggregates and frequency encodings (via
prepare_train_test_features), exactly as it would for two genuinely
separate customer populations in production, never one continuous stream
artificially cut in two.

That alone was NOT enough domain shift to give transfer learning something
real to do, and this project found that out by direct measurement rather
than assuming it: generate_synthetic_ieee.py's V1-V20 columns are built as
`base_noise + isFraud * fixed_shift` with the SAME fixed_shift distribution
regardless of seed, n_transactions, or fraud_rate (see that module's source),
so two domains built only by varying those three arguments share the
IDENTICAL feature-to-label relationship; only the class PREVALENCE differs
between them. Measured directly in this project's own sandbox testing: a
source model trained on an 8000-row/3.5%-fraud domain and applied with zero
target labels (apply_zero_shot) to a 1200-row/6%-fraud domain scored
AUC-PR=0.9658, actually slightly ABOVE the source model's own 0.8858 test
AUC-PR on its own domain, because AUC-based ranking metrics do not care
about a shift in prevalence alone (they are threshold-invariant), only
about whether the pattern being ranked is the same. A prevalence-only shift
gives zero-shot transfer nothing real to fail at, and gives warm-start
nothing real to fix, which would have made Week 8's whole comparison
uninteresting.

`shift_columns` exists to fix that, the same way Week 5's
generate_synthetic_ieee.py deliberately injects fraud rings so a graph
model has real structure to find: it independently shuffles the given
columns' values across rows, AFTER generation, breaking whatever
correlation those specific columns had with isFraud while leaving every
other column's relationship intact. Passed a subset of the V-columns for
the TARGET domain only (never the source domain), this creates a genuine,
disclosed pattern shift: several columns the source model learned to rely
on carry no signal at all in the target domain. Measured directly: shuffling
V1-V10 (half of this generator's 20 signal columns) for the target domain
dropped zero-shot transfer's AUC-PR from 0.9658 to 0.6058, a real, large
drop this project's own transfer_models.py functions are then measured
against.

start_transaction_id defaults differ (source vs target) only so that, if
anyone ever wanted to concatenate the two domains' raw transaction tables
for inspection, TransactionID would still be unique across them; nothing in
this module or Week 8's transfer_models.py actually relies on that.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.generate_synthetic_ieee import (
    generate_synthetic_identity,
    generate_synthetic_transactions,
)
from src.data.load_and_merge import merge_transaction_identity
from src.models.prepare_model_data import prepare_train_test_features, select_feature_columns


def build_domain(
    n_transactions: int,
    fraud_rate: float,
    seed: int,
    test_size: float = 0.2,
    identity_coverage: float = 0.24,
    start_transaction_id: int = 2987000,
    shift_columns: Optional[Sequence[str]] = None,
    shift_seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Generates one synthetic domain end to end: raw transactions and
    identity records, merged, then chronologically split into engineered
    train/test feature frames.

    shift_columns, if given, are independently row-shuffled in the raw
    transactions table right after generation, before any feature
    engineering, breaking their correlation with isFraud while leaving
    their own marginal distribution unchanged. See the module docstring for
    why this is needed at all and what this project measured it to do; pass
    it only for a domain meant to represent a genuine pattern shift from
    another domain built without it.

    Returns (train_df, test_df, feature_columns). feature_columns is
    computed from train_df, the same discipline prepare_model_data.py's own
    callers already follow, since it must never be computed from test_df
    (that would leak which columns test rows happen to have non-null
    values in).
    """
    transactions = generate_synthetic_transactions(
        n_transactions=n_transactions,
        fraud_rate=fraud_rate,
        seed=seed,
        start_transaction_id=start_transaction_id,
    )
    if shift_columns:
        rng = np.random.default_rng(shift_seed)
        for column in shift_columns:
            if column not in transactions.columns:
                raise ValueError(f"shift_columns contains unknown column {column!r}.")
            transactions[column] = rng.permutation(transactions[column].to_numpy())

    identity = generate_synthetic_identity(
        transactions["TransactionID"].to_numpy(),
        coverage_rate=identity_coverage,
        seed=seed + 1,
    )
    merged = merge_transaction_identity(transactions, identity)
    train_df, test_df = prepare_train_test_features(merged, test_size=test_size)
    feature_columns = select_feature_columns(train_df)
    return train_df, test_df, feature_columns
