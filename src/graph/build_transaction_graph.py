"""
build_transaction_graph.py, Week 5: turns the tabular train/test features
from Weeks 2 through 4 into a heterogeneous transaction graph, the input a
graph neural network needs.

The graph has two kinds of nodes. "transaction" nodes are the actual rows
this project has been modeling all along, carrying the same engineered
feature columns Week 3/4's tabular models used. Entity nodes (one type per
column in entity_columns, by default card1, addr1, and P_emaildomain) each
represent one distinct real-world value of that column, for example one
particular card number. An edge connects a transaction node to an entity
node whenever that transaction used that card, address, or email domain.
Two transactions that share a card become reachable from one another
through that shared card1 entity node, which is exactly the structure a
graph neural network can use to flag a transaction as suspicious because
of who it is connected to, not only what its own row looks like. Week 3
and 4's tabular models have no way to see this: a row-by-row feature
vector cannot represent "this transaction's card was also used by three
other transactions currently flagged as fraud."

Entity nodes are given a constant, uninformative feature (a single 1.0),
deliberately. Giving an entity node a feature computed from the labels of
the transactions connected to it (such as "how many of this card's
transactions are fraud") would leak test-set labels into the graph itself,
since a shared entity can connect train and test transactions. Keeping
entity features constant means every signal the model uses about an
entity has to come from actually passing messages along its edges during
training, never from a precomputed shortcut.

The train and test transactions are combined into a single graph rather
than two separate ones. This is standard practice for graph neural
networks (called transductive node classification) and is not a leak by
itself: which card, address, or email domain a transaction used is known
the moment that transaction happens, in production exactly as much as
here, so sharing that structure across the train and test period does not
give the model information it would not really have. What must and does
stay separated is the LABEL: train_mask and test_mask mark which
transaction nodes' isFraud values the training loop is allowed to look at,
and src/graph/gnn_model.py's training loop computes its loss only over
train_mask, never test_mask.

Missing feature values (several Week 2 features are NaN by construction,
such as a card's first transaction having no prior average) are imputed
with the training set's median, and every feature is standardized, both
fit on the training transactions only and applied to both splits, the
same train-only-fit discipline used since Week 3. Standardization matters
more here than it did for Week 3's tree models: a neural network's
gradient-based training is sensitive to feature scale in a way a decision
tree split is not.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData

DEFAULT_ENTITY_COLUMNS = ["card1", "addr1", "P_emaildomain"]


def build_hetero_graph(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    entity_columns: Optional[List[str]] = None,
    target_col: str = "isFraud",
) -> HeteroData:
    """Builds one HeteroData graph covering both train_df and test_df.

    Raises if either frame is empty, if feature_columns/entity_columns/
    target_col are missing from either frame, or if train_df and test_df
    share any TransactionID (they must be a real, non-overlapping split;
    see src/models/prepare_model_data.py's chronological_train_test_split).
    """
    if entity_columns is None:
        entity_columns = DEFAULT_ENTITY_COLUMNS

    if train_df.empty or test_df.empty:
        raise ValueError("train_df and test_df must both be non-empty.")

    required_columns = set(feature_columns) | set(entity_columns) | {target_col}
    for name, df in [("train_df", train_df), ("test_df", test_df)]:
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}.")

    if "TransactionID" in train_df.columns and "TransactionID" in test_df.columns:
        overlap = set(train_df["TransactionID"]) & set(test_df["TransactionID"])
        if overlap:
            raise ValueError(
                f"train_df and test_df share {len(overlap)} TransactionID "
                "values; they must be a real, non-overlapping split."
            )

    combined = pd.concat([train_df, test_df], ignore_index=True)
    n_train = len(train_df)
    n_total = len(combined)
    train_mask = np.zeros(n_total, dtype=bool)
    train_mask[:n_train] = True
    test_mask = ~train_mask

    # Fit-on-train-only imputation and scaling, same discipline as Week 3's
    # models, applied here since a neural network needs both no missing
    # values and comparable feature scales to train well.
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_df[feature_columns])
    scaler = StandardScaler()
    scaler.fit(imputer.transform(train_df[feature_columns]))

    x = scaler.transform(imputer.transform(combined[feature_columns])).astype(np.float32)
    y = combined[target_col].to_numpy().astype(np.float32)

    data = HeteroData()
    data["transaction"].x = torch.from_numpy(x)
    data["transaction"].y = torch.from_numpy(y)
    data["transaction"].train_mask = torch.from_numpy(train_mask)
    data["transaction"].test_mask = torch.from_numpy(test_mask)

    for entity_col in entity_columns:
        values = combined[entity_col]
        distinct_values = sorted(values.dropna().unique().tolist(), key=str)
        value_to_index = {value: index for index, value in enumerate(distinct_values)}

        entity_index = values.map(value_to_index)
        present = entity_index.notna().to_numpy()
        transaction_indices = np.nonzero(present)[0]
        entity_indices = entity_index[present].to_numpy().astype(np.int64)

        # Constant, label-free feature; see module docstring for why.
        data[entity_col].x = torch.ones((len(distinct_values), 1), dtype=torch.float32)

        forward_edge_index = torch.from_numpy(
            np.stack([transaction_indices, entity_indices]).astype(np.int64)
        )
        backward_edge_index = torch.from_numpy(
            np.stack([entity_indices, transaction_indices]).astype(np.int64)
        )
        data[("transaction", f"uses_{entity_col}", entity_col)].edge_index = forward_edge_index
        data[(entity_col, f"rev_uses_{entity_col}", "transaction")].edge_index = backward_edge_index

    return data
