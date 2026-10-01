"""
gnn_model.py, Week 5: a heterogeneous graph neural network that classifies
transaction nodes as fraud or not, trained and evaluated on the graph
src/graph/build_transaction_graph.py builds.

FraudGNN alternates two things across num_layers rounds: each edge type
(uses_card1, rev_uses_card1, uses_addr1, ...) gets its own GraphSAGE
convolution, and HeteroConv combines whatever each convolution produces
for a given node type by summing it. In plain terms, after one round every
transaction node's representation has absorbed a summary of the entity
nodes it touches, and every entity node's representation has absorbed a
summary of the transactions that touch it; after two rounds, a
transaction's representation has also picked up a hint of the OTHER
transactions that share its entities, one hop further out, which is
exactly the "fraud ring" signal this module exists to find. A final linear
layer turns each transaction node's representation into one fraud score.

Every SAGEConv here is built with lazy input dimensions ((-1, -1)), a
PyTorch Geometric feature that infers each layer's real input size from
the first batch of data it sees rather than requiring it to be hardcoded
ahead of time; this is why train_gnn always runs one throwaway forward
pass before creating the optimizer, so every lazily created parameter
already exists and is included in what the optimizer trains.

Training uses BCEWithLogitsLoss computed ONLY over the training
transaction nodes (data["transaction"].train_mask), never the test nodes,
even though every node's features and edges are visible to the model
during every forward pass (see build_transaction_graph.py's docstring for
why sharing structure, but not labels, is not a leak). Evaluation reuses
Week 4's own metrics module (src/models/metrics.py), so a Week 5 graph
model's AUC-ROC, AUC-PR, KS statistic, and Gini coefficient are directly
comparable to Week 3 and Week 4's tabular numbers on the same test rows.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from src.models.metrics import auc_pr, gini_coefficient, ks_statistic


class FraudGNN(torch.nn.Module):
    """A stack of HeteroConv/GraphSAGE layers followed by a linear fraud
    score head applied to the "transaction" node type's output embedding.
    """

    def __init__(
        self,
        edge_types: List[Tuple[str, str, str]],
        hidden_channels: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {edge_type: SAGEConv((-1, -1), hidden_channels) for edge_type in edge_types},
                aggr="sum",
            )
            self.convs.append(conv)
        self.dropout = dropout
        self.classifier = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict) -> torch.Tensor:
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(value) for key, value in x_dict.items()}
            x_dict = {
                key: F.dropout(value, p=self.dropout, training=self.training)
                for key, value in x_dict.items()
            }
        return self.classifier(x_dict["transaction"]).squeeze(-1)


def _validate_mask_has_both_classes(y: torch.Tensor, mask: torch.Tensor, mask_name: str) -> None:
    labels = y[mask]
    if labels.numel() == 0:
        raise ValueError(f"{mask_name} selects zero nodes.")
    unique_labels = set(labels.unique().tolist())
    if len(unique_labels) < 2:
        raise ValueError(
            f"{mask_name} has only one class present; training/evaluation "
            "metrics are undefined without both classes."
        )


def train_gnn(
    data: HeteroData,
    hidden_channels: int = 32,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    random_state: int = 42,
) -> Tuple[FraudGNN, List[float]]:
    """Trains a FraudGNN on data, using only data["transaction"].train_mask
    for the loss, and returns the trained model plus the per-epoch training
    loss history (for anyone who wants to plot or sanity-check convergence,
    not used anywhere else in this project).
    """
    torch.manual_seed(random_state)

    y = data["transaction"].y
    train_mask = data["transaction"].train_mask
    _validate_mask_has_both_classes(y, train_mask, "train_mask")

    model = FraudGNN(
        data.edge_types, hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout,
    )
    # One throwaway forward pass so every lazily initialized SAGEConv
    # weight actually exists before the optimizer is built; see module
    # docstring.
    with torch.no_grad():
        model(data.x_dict, data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    history: List[float] = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x_dict, data.edge_index_dict)
        loss = loss_fn(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))

    # Left in eval mode (dropout off) on return: evaluate_gnn also sets
    # this itself, but returning a model still in train mode would give
    # non-deterministic, dropout-affected output to anyone who calls it
    # directly instead, a real bug caught by testing mlflow round-trip
    # logging (two forward passes on the same untouched model disagreed
    # until eval() was added here).
    model.eval()
    return model, history


def evaluate_gnn(model: FraudGNN, data: HeteroData, split: str = "test") -> Dict:
    """Evaluates a trained FraudGNN on data["transaction"][f"{split}_mask"],
    returning the same metric set (AUC-ROC, AUC-PR, KS statistic and its
    threshold, Gini) Week 4's tabular models report, plus the row and
    fraud counts the numbers are traceable back to.
    """
    mask_attr = f"{split}_mask"
    if not hasattr(data["transaction"], mask_attr):
        raise ValueError(f"data['transaction'] has no {mask_attr!r}.")

    y = data["transaction"].y
    mask = getattr(data["transaction"], mask_attr)
    _validate_mask_has_both_classes(y, mask, mask_attr)

    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        proba = torch.sigmoid(logits)[mask].numpy()
    y_true = y[mask].numpy()

    ks = ks_statistic(y_true, proba)

    return {
        "auc_roc": roc_auc_score(y_true, proba),
        "auc_pr": auc_pr(y_true, proba),
        "ks_statistic": ks["ks_statistic"],
        "ks_threshold": ks["ks_threshold"],
        "gini": gini_coefficient(y_true, proba),
        "n": int(mask.sum().item()),
        "n_fraud": int(y_true.sum()),
    }
