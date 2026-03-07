# module1/pretrain/model.py
#
# Stage 1 — SpatioTemporalEncoder + ActionClassifier
# ────────────────────────────────────────────────────
# End-to-end model that encodes a sequence of skeleton graphs:
#
#   Input : (B, T, J, F)  — batch, frames, joints, features (pos+vel+acc = 9)
#
#   Spatial stage — time-distributed GAT (shared weights across all T frames):
#     GATConv( F → 16, heads=4, concat=True)  → 64-dim per node + BN + ELU
#     GATConv(64 → 16, heads=4, concat=True)  → 64-dim per node + BN + ELU
#     GATConv(64 → 64, heads=1, concat=False) → 64-dim per node + BN + ELU
#     global_mean_pool over joints             → (B*T, 64)
#     reshape                                  → (B, T, 64)
#
#   Temporal stage — Transformer encoder:
#     input_proj Linear(64 → d_model)
#     prepend CLS token
#     sinusoidal positional encoding
#     2 × TransformerEncoderLayer (pre-norm, nhead=4, ff=256)
#     LayerNorm
#     extract CLS output                       → (B, d_model)
#
#   ActionClassifier wraps SpatioTemporalEncoder with a linear head for
#   102-class ANUBIS pretraining.  After pretraining, save only the encoder
#   (encoder.state_dict()) and attach new fine-tuning heads.

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

from utils.logger import get_logger

log = get_logger(__name__)


# ── Positional Encoding ────────────────────────────────────────────────────────


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).

    Parameters
    ----------
    d_model  : int   embedding dimension
    max_len  : int   maximum sequence length (including CLS token)
    dropout  : float dropout applied after adding the encoding
    """

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, T, d_model)

        Returns
        -------
        out : torch.Tensor  shape (B, T, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ── SpatioTemporalEncoder ─────────────────────────────────────────────────────


class SpatioTemporalEncoder(nn.Module):
    """
    End-to-end spatiotemporal encoder: time-distributed GAT + Transformer.

    The encoder is the shared backbone used across all training stages.
    It outputs a single CLS embedding per sequence.

    Parameters
    ----------
    num_joints   : int   number of skeleton joints (32 for ANUBIS)
    in_features  : int   feature dimension per joint (9: pos+vel+acc)
    edge_index   : torch.Tensor  shape (2, E)  skeleton connectivity;
                   registered as a buffer so it moves to device automatically
    gat_hidden   : int   output dim of every GAT block (default 64)
    gat_heads    : int   attention heads for GAT layers 1 & 2 (default 4)
    gat_dropout  : float dropout on GAT attention coefficients
    d_model      : int   transformer embedding dimension (default 128)
    nhead        : int   transformer attention heads
    num_layers   : int   number of TransformerEncoderLayer blocks
    dim_ff       : int   feedforward expansion inside transformer
    dropout      : float transformer dropout
    seq_len      : int   expected sequence length (used to size pos. encoding)
    cls_std      : float trunc_normal std for CLS-token initialisation
    """

    def __init__(
        self,
        num_joints: int,
        in_features: int,
        edge_index: torch.Tensor,
        gat_hidden: int = 64,
        gat_heads: int = 4,
        gat_dropout: float = 0.1,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
        seq_len: int = 60,
        cls_std: float = 0.02,
    ) -> None:
        super().__init__()

        self.num_joints = num_joints
        self.gat_hidden = gat_hidden
        self.gat_dropout = gat_dropout
        self.d_model = d_model

        # Register skeleton topology as a non-learnable buffer
        self.register_buffer("edge_index", edge_index)  # (2, E)

        # ── GAT layers (shared across all time steps) ──────────────────────
        per_head_dim = gat_hidden // gat_heads  # 16
        self.gat1 = GATConv(
            in_features, per_head_dim, heads=gat_heads, concat=True, dropout=gat_dropout
        )
        self.gat2 = GATConv(
            gat_hidden, per_head_dim, heads=gat_heads, concat=True, dropout=gat_dropout
        )
        self.gat3 = GATConv(
            gat_hidden, gat_hidden, heads=1, concat=False, dropout=gat_dropout
        )

        self.bn1 = nn.BatchNorm1d(gat_hidden)
        self.bn2 = nn.BatchNorm1d(gat_hidden)
        self.bn3 = nn.BatchNorm1d(gat_hidden)

        # ── Transformer ────────────────────────────────────────────────────
        self.input_proj = nn.Linear(gat_hidden, d_model)

        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=cls_std)

        # max_len = seq_len + 1 to accommodate the CLS token
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm variant — more stable gradients
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _gat_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process all T frames through the GAT in a single batched pass.

        Each (frame, sequence) pair is treated as an independent graph.
        All B*T graphs share the same edge topology (self.edge_index).

        Parameters
        ----------
        x : torch.Tensor  shape (B, T, J, F)

        Returns
        -------
        spatial : torch.Tensor  shape (B, T, gat_hidden)
        """
        B, T, J, Fin = x.shape
        num_graphs = B * T

        # Flatten nodes: (B*T*J, Fin)
        x_nodes = x.reshape(num_graphs * J, Fin)

        # Build batched edge index by offsetting each graph's node indices by J
        # edge_index: (2, E)  →  broadcast to (2, num_graphs, E)
        E = self.edge_index.shape[1]
        offsets = torch.arange(num_graphs, device=x.device) * J  # (num_graphs,)
        edge_batch = self.edge_index.unsqueeze(1).expand(
            -1, num_graphs, -1
        )  # (2, G, E)
        edge_batch = edge_batch + offsets.view(1, -1, 1)  # (2, G, E)
        edge_flat = edge_batch.reshape(2, num_graphs * E)  # (2, G*E)

        # batch_tensor: maps each node to its graph index
        batch_tensor = torch.arange(num_graphs, device=x.device).repeat_interleave(J)

        # ── GAT layer 1 ────────────────────────────────────────────────────
        h = self.gat1(x_nodes, edge_flat)  # (B*T*J, gat_hidden)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.gat_dropout, training=self.training)

        # ── GAT layer 2 ────────────────────────────────────────────────────
        h = self.gat2(h, edge_flat)
        h = self.bn2(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.gat_dropout, training=self.training)

        # ── GAT layer 3 ────────────────────────────────────────────────────
        h = self.gat3(h, edge_flat)
        h = self.bn3(h)
        h = F.elu(h)

        # Global mean pool: (B*T*J, gat_hidden) → (B*T, gat_hidden)
        spatial = global_mean_pool(h, batch_tensor)

        return spatial.reshape(B, T, self.gat_hidden)  # (B, T, gat_hidden)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of skeleton sequences.

        Parameters
        ----------
        x : torch.Tensor  shape (B, T, J, F)
            Joint features: pos + vel + acc (F = 9).

        Returns
        -------
        cls_embedding : torch.Tensor  shape (B, d_model)
            Sequence-level representation extracted from the CLS token.
        """
        B = x.shape[0]

        # Stage 1: spatial encoding (time-distributed GAT)
        spatial = self._gat_forward(x)  # (B, T, gat_hidden)

        # Stage 2: temporal encoding
        h = self.input_proj(spatial)  # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        h = torch.cat([cls, h], dim=1)  # (B, T+1, d_model)

        # Positional encoding
        h = self.pos_enc(h)  # (B, T+1, d_model)

        # Transformer encoder
        h = self.transformer(h)  # (B, T+1, d_model)
        h = self.norm(h)

        # Extract CLS token (position 0)
        return h[:, 0, :]  # (B, d_model)


# ── ActionClassifier ──────────────────────────────────────────────────────────


class ActionClassifier(nn.Module):
    """
    Full pretraining model: SpatioTemporalEncoder + linear classification head.

    Used only for Stage 1 (ANUBIS action classification pretraining).
    After pretraining, save encoder.state_dict() and discard the head.

    Parameters
    ----------
    encoder     : SpatioTemporalEncoder
    num_classes : int   number of action classes (102 for ANUBIS)
    """

    def __init__(self, encoder: SpatioTemporalEncoder, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, T, J, F)

        Returns
        -------
        logits : torch.Tensor  shape (B, num_classes)
        """
        cls_emb = self.encoder(x)  # (B, d_model)
        return self.head(cls_emb)  # (B, num_classes)


# ── Factory helper ────────────────────────────────────────────────────────────


def build_model(
    cfg: dict, edge_index: torch.Tensor
) -> Tuple[SpatioTemporalEncoder, ActionClassifier]:
    """
    Construct encoder and full classifier from config dict.

    Parameters
    ----------
    cfg        : dict               full config loaded from configs/pretrain.yaml
    edge_index : torch.Tensor (2,E) skeleton edge index

    Returns
    -------
    encoder    : SpatioTemporalEncoder
    classifier : ActionClassifier
    """
    m = cfg["model"]
    d = cfg["data"]
    encoder = SpatioTemporalEncoder(
        num_joints=d["num_joints"],
        in_features=m["in_features"],
        edge_index=edge_index,
        gat_hidden=m["gat_hidden_dim"],
        gat_heads=m["gat_heads"],
        gat_dropout=m["gat_dropout"],
        d_model=m["d_model"],
        nhead=m["nhead"],
        num_layers=m["num_transformer_layers"],
        dim_ff=m["dim_feedforward"],
        dropout=m["transformer_dropout"],
        seq_len=d["seq_len"],
        cls_std=m["cls_std"],
    )
    classifier = ActionClassifier(encoder, num_classes=d["num_classes"])
    total = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    log.info("Model built: %d trainable parameters", total)
    return encoder, classifier


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from module1.data.joint_config import EDGE_INDEX, NUM_JOINTS

    B, T, J, F_IN = 4, 60, NUM_JOINTS, 9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = SpatioTemporalEncoder(
        num_joints=J,
        in_features=F_IN,
        edge_index=EDGE_INDEX,
        seq_len=T,
    ).to(device)

    classifier = ActionClassifier(encoder, num_classes=102).to(device)
    classifier.eval()

    dummy = torch.randn(B, T, J, F_IN, device=device)

    with torch.no_grad():
        cls_emb = encoder(dummy)
        logits = classifier(dummy)

    assert cls_emb.shape == (B, 128), f"Expected (4,128), got {cls_emb.shape}"
    assert logits.shape == (B, 102), f"Expected (4,102), got {logits.shape}"
    assert not torch.isnan(cls_emb).any(), "NaN in encoder output"
    assert not torch.isnan(logits).any(), "NaN in classifier logits"

    n_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Encoder output shape    : {cls_emb.shape}")
    print(f"Classifier output shape : {logits.shape}")
    print(f"Trainable parameters    : {n_params:,}")
    print("model.py OK.")
