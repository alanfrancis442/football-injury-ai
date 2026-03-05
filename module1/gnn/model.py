# module1/gnn/model.py
#
# Stage 2 — Graph Attention Network (GAT) Model
# ──────────────────────────────────────────────
# Takes a per-frame skeleton graph and outputs a 64-dimensional spatial
# feature vector that captures "what position is the body in right now
# and what does it mean biomechanically".
#
# Architecture:
#   Input  → GAT Layer 1 (64 dims)
#          → GAT Layer 2 (64 dims)
#          → GAT Layer 3 (64 dims)
#          → Global Mean Pooling
#   Output → 64-dim spatial feature vector
#
# The attention mechanism automatically learns which joint relationships
# matter most for injury risk — no manual feature engineering required.
#
# Hyperparameters are loaded from configs/module1.yaml

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class SkeletonGAT(nn.Module):
    """
    Graph Attention Network over the human skeleton.

    Parameters
    ----------
    in_channels  : int   — input feature size per node (default: 10)
    hidden_dim   : int   — hidden layer size (default: 64)
    out_dim      : int   — output spatial feature size (default: 64)
    heads        : int   — number of attention heads per layer (default: 4)
    dropout      : float — dropout rate applied to attention weights
    """

    def __init__(
        self,
        in_channels: int = 10,
        hidden_dim: int = 64,
        out_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout

        # Layer 1: in_channels → hidden_dim (multi-head)
        # concat=True means output is heads × hidden_dim per node
        self.gat1 = GATConv(
            in_channels,
            hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True,
        )

        # Layer 2: hidden_dim → hidden_dim
        self.gat2 = GATConv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True,
        )

        # Layer 3: hidden_dim → out_dim (single head, no concat)
        self.gat3 = GATConv(
            hidden_dim,
            out_dim,
            heads=1,
            dropout=dropout,
            concat=False,
        )

        # Batch normalisation after each layer for training stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (N, in_channels)  — node features for all graphs in the batch
        edge_index : (2, E)            — edge list
        batch      : (N,)              — maps each node to its graph index in the batch

        Returns
        -------
        out : (B, out_dim)  — one 64-dim vector per graph (player) in the batch
        """
        # Layer 1
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3
        x = self.gat3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)

        # Global pooling: reduce 33 node vectors → 1 graph vector
        out = global_mean_pool(x, batch)  # (B, out_dim)

        return out


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch_geometric.data import Data, Batch
    from module1.gnn.graph import EDGE_INDEX

    # Simulate a batch of 2 skeleton graphs
    batch_data = []
    for _ in range(2):
        x = torch.rand(33, 10)
        data = Data(x=x, edge_index=EDGE_INDEX.clone())
        batch_data.append(data)

    batch = Batch.from_data_list(batch_data)

    model = SkeletonGAT()
    model.eval()

    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.batch)

    print(f"Input  node features : {batch.x.shape}")  # (66, 10) — 2 graphs × 33 nodes
    print(f"Output spatial vecs  : {out.shape}")  # (2, 64)  — 2 graphs × 64 dims
    print("SkeletonGAT forward pass OK.")
