"""
GNN model architectures for the OGBN-Arxiv node classification project.

This is the single source of truth for model definitions. Any notebook
or script that needs a GCN or GraphSAGE model should import from here
rather than redefining the classes locally — that way, an architecture
change (e.g. adding BatchNorm) only needs to happen once, and every
downstream task (training, evaluation, explainability) picks it up
automatically the next time they pull `main`.

Usage:
    from models.gnn_models import GCN, GraphSAGE

    model = GCN(in_channels=128, hidden_channels=256,
                out_channels=40, dropout=0.5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class GCN(nn.Module):
    """
    Baseline spectral-style GNN (Kipf & Welling, 2017).

    Architecture:
        Input -> GCNConv -> BatchNorm -> ReLU -> Dropout
              -> GCNConv -> BatchNorm -> ReLU -> Dropout
              -> GCNConv -> logits

    Design choices:
        - 3 layers: enough to aggregate 3-hop neighborhood info
          without over-smoothing (a known GCN failure mode at
          higher depth on citation graphs).
        - BatchNorm after each conv, before activation: stabilizes
          training and matches the standard OGB baseline
          implementation for this dataset.
        - ReLU: standard, cheap, avoids vanishing gradients.
        - Dropout after each hidden layer: regularizes against
          overfitting on the small labeled fraction of nodes.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)  # raw logits, softmax applied in loss
        return x


class GraphSAGE(nn.Module):
    """
    Inductive, neighbor-sampling-friendly GNN (Hamilton et al., 2017).

    Architecture:
        Input -> SAGEConv -> BatchNorm -> ReLU -> Dropout
              -> SAGEConv -> BatchNorm -> ReLU -> Dropout
              -> SAGEConv -> logits

    Design choices vs. GCN:
        - SAGEConv aggregates neighbor features via a learned
          aggregation rather than a fixed spectral normalization —
          inductive, scales to large graphs via mini-batch sampling.
        - BatchNorm added for the same stabilization reason as GCN,
          keeping the GCN vs. GraphSAGE comparison fair (identical
          regularization treatment, isolating the effect of the
          aggregation mechanism itself).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        return x


# Registry so config-driven code can instantiate a model by name,
# e.g. MODEL_REGISTRY[config["class"]](**config_without_class_key)
MODEL_REGISTRY = {
    "GCN": GCN,
    "GraphSAGE": GraphSAGE,
}


def build_model_from_config(model_config: dict) -> nn.Module:
    """
    Instantiate a model directly from a model_config.json entry.

    Example:
        with open("models_checkpoints/model_config.json") as f:
            full_config = json.load(f)
        gcn = build_model_from_config(full_config["gcn"])
    """
    config = dict(model_config)  # don't mutate the caller's dict
    model_class_name = config.pop("class")
    model_class = MODEL_REGISTRY[model_class_name]
    return model_class(**config)
