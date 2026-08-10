import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


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


class GAT(nn.Module):
    """
    Graph Attention Network (Veličković et al., 2018).

    Architecture:
        Input -> GATConv (multi-head) -> BatchNorm -> ELU -> Dropout
              -> GATConv (single-head output) -> logits

    Design choices vs. GCN / GraphSAGE:
        - Learns per-edge attention weights instead of a fixed
          spectral normalization (GCN) or uniform/learned-but-
          unweighted aggregation (GraphSAGE) — each neighbor's
          contribution is weighted by a learned attention score,
          not treated equally.
        - Multi-head attention (default 4 heads) in the first layer
          lets the model attend to different aspects of the
          neighborhood in parallel, concatenated into a wider
          hidden representation.
        - Second layer uses a single head with averaging
          (concat=False) to produce final class logits directly,
          following the original GAT paper's design for the
          output layer.
        - ELU activation (not ReLU): matches the original GAT paper;
          ELU's smooth negative-region gradient tends to pair better
          with attention-weighted aggregation than ReLU.
        - Implemented as a third architecture — an "Advanced GNN
          Architecture" beyond the two required models — and to
          enable genuine attention-weight-based explainability in
          Task 07, which GCN and GraphSAGE cannot provide since
          neither has a learned per-neighbor importance score.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        heads: int = 4,
    ):
        super().__init__()
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=heads, dropout=dropout
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)

        self.conv2 = GATConv(
            hidden_channels * heads, out_channels, heads=1,
            concat=False, dropout=dropout
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def forward_with_attention(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Same forward pass, but also returns attention weights from
        both layers — used by Task 07 for attention-based
        explainability (which nodes/edges the model actually
        attended to when making a prediction).

        Returns:
            logits, (edge_index_1, alpha_1), (edge_index_2, alpha_2)
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, (edge_index_1, alpha_1) = self.conv1(
            x, edge_index, return_attention_weights=True
        )
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, (edge_index_2, alpha_2) = self.conv2(
            x, edge_index, return_attention_weights=True
        )
        return x, (edge_index_1, alpha_1), (edge_index_2, alpha_2)



MODEL_REGISTRY = {
    "GCN": GCN,
    "GraphSAGE": GraphSAGE,
    "GAT": GAT,
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
