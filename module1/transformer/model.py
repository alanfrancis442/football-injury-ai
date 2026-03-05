# module1/transformer/model.py
#
# Stage 3 — Temporal Transformer Model
# ──────────────────────────────────────
# Takes a sequence of 50 GNN spatial feature vectors (one per frame, 2 seconds
# of movement history) and classifies the biomechanical risk level.
#
# The key insight: a single frame shows WHERE a joint is.
#                  A sequence shows WHERE it's HEADING — which is what matters.
#
# Architecture:
#   Input sequence : (T=50, 64-dim GNN vectors)
#   Positional Encoding → 2× TransformerEncoderLayer (4 heads)
#   [CLS] token → Linear head → risk score + body region
#
# Output:
#   risk_score  : float [0,1]  — probability of biomechanical injury risk
#   region_logits : (11,)      — which body part is at risk

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ── Body region labels ─────────────────────────────────────────────────────────
BODY_REGIONS = [
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "left_hamstring",
    "right_hamstring",
    "left_ankle",
    "right_ankle",
    "spine",
    "shoulder",
    "contact_impact",
]
NUM_REGIONS = len(BODY_REGIONS)  # 11


# ── Sinusoidal Positional Encoding ────────────────────────────────────────────


class PositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding from "Attention Is All You Need".
    Adds a unique positional signal to each frame in the sequence so the model
    knows the order of events (frame 1 vs frame 50 is meaningful for risk).
    """

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the encoding matrix once (not learnable)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as buffer so it moves with .to(device) but isn't a parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ── Biomechanical Transformer ─────────────────────────────────────────────────


class BiomechanicalTransformer(nn.Module):
    """
    Transformer-based temporal risk classifier.

    Parameters
    ----------
    input_dim   : int   — GNN output dimension (default: 64)
    d_model     : int   — internal transformer width (default: 128)
    nhead       : int   — attention heads (default: 4)
    num_layers  : int   — number of encoder layers (default: 2)
    dim_ff      : int   — feedforward expansion size (default: 256)
    dropout     : float — dropout rate
    seq_len     : int   — temporal window length in frames (default: 50)
    """

    def __init__(
        self,
        input_dim: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
        seq_len: int = 50,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Project GNN 64-dim → transformer d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # [CLS] token — prepended to each sequence, its output summarises the whole window
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,  # input is (B, T, d_model)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)

        # Output heads
        self.risk_head = nn.Linear(d_model, 1)  # → risk score (0–1)
        self.region_head = nn.Linear(d_model, NUM_REGIONS)  # → body region

    def forward(self, x: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, T, input_dim)
            Sequence of GNN spatial feature vectors.
            B = batch size, T = sequence length (50 frames), input_dim = 64.

        Returns
        -------
        dict with keys:
            'risk_score'     : (B, 1)        — sigmoid probability [0,1]
            'region_logits'  : (B, NUM_REGIONS) — raw logits for body region
            'cls_embedding'  : (B, d_model)  — full CLS embedding for inspection
        """
        B, T, _ = x.shape

        # Project to d_model
        x = self.input_proj(x)  # (B, T, d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)

        # Positional encoding
        x = self.pos_enc(x)  # (B, T+1, d_model)

        # Transformer encoder
        x = self.transformer(x)  # (B, T+1, d_model)
        x = self.norm(x)

        # Extract [CLS] output (index 0)
        cls_out = x[:, 0, :]  # (B, d_model)

        # Heads
        risk_score = torch.sigmoid(self.risk_head(cls_out))  # (B, 1)
        region_logits = self.region_head(cls_out)  # (B, 11)

        return {
            "risk_score": risk_score,
            "region_logits": region_logits,
            "cls_embedding": cls_out,
        }


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = BiomechanicalTransformer()
    model.eval()

    # Simulate a batch of 4 players, each with 50 frames of 64-dim GNN vectors
    dummy_input = torch.rand(4, 50, 64)

    with torch.no_grad():
        out = model(dummy_input)

    print(f"Input shape        : {dummy_input.shape}")  # (4, 50, 64)
    print(f"Risk score shape   : {out['risk_score'].shape}")  # (4, 1)
    print(f"Region logits shape: {out['region_logits'].shape}")  # (4, 11)
    print(f"CLS embedding shape: {out['cls_embedding'].shape}")  # (4, 128)
    print("BiomechanicalTransformer forward pass OK.")
