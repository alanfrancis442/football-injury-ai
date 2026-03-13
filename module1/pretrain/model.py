# module1/pretrain/model.py
#
# Stage 1 — SpatioTemporalEncoder + ActionClassifier
# ───────────────────────────────────────────────────
# Encodes ANUBIS skeleton sequences with a shared spatial GAT trunk and a
# configurable temporal head. Supported experiment heads are:
#   - Transformer with absolute positional encoding
#   - Dilated Temporal Convolution Network (TCN) with attention pooling

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

from utils.logger import get_logger

log = get_logger(__name__)


# ── Positional Encoding ───────────────────────────────────────────────────────


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence models.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_len : int
        Maximum supported sequence length.
    dropout : float
        Dropout applied after adding the positional signal.
    """

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pe: torch.Tensor
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = cast(torch.Tensor, self.pe)
        x = x + pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned absolute positional encoding for sequence models.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_len : int
        Maximum supported sequence length.
    dropout : float
        Dropout applied after adding the positional signal.
    """

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.embedding[:, : x.size(1), :]
        return self.dropout(x)


def build_positional_encoder(
    encoding: str,
    d_model: int,
    max_len: int,
    dropout: float,
) -> Optional[nn.Module]:
    """Construct an absolute positional encoder or return None."""
    encoding_name = encoding.lower()
    if encoding_name == "none":
        return None
    if encoding_name == "sinusoidal":
        return PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
    if encoding_name == "learned":
        return LearnedPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )
    raise ValueError(
        f"Unsupported positional_encoding='{encoding}'. "
        "Use 'none', 'sinusoidal', or 'learned'."
    )


# ── Temporal Heads ────────────────────────────────────────────────────────────


class TemporalAttentionPool(nn.Module):
    """Attention-weighted pooling over the temporal axis."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        hidden = max(d_model // 2, 32)
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a sequence tensor of shape (B, T, D) into (B, D)."""
        h = self.norm(x)
        weights = torch.softmax(self.score(h), dim=1)
        return torch.sum(h * weights, dim=1)


class TemporalConvBlock(nn.Module):
    """Residual dilated temporal convolution block."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"TCN kernel size must be odd for same-length padding, got {kernel_size}."
            )

        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(d_model)
        self.temporal = nn.Sequential(
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x).transpose(1, 2)
        h = self.temporal(h).transpose(1, 2)
        return residual + h


class TemporalConvEncoder(nn.Module):
    """Dilated TCN encoder with attention pooling."""

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        kernel_size: int,
        dilation_growth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = []
        dilation = 1
        for _ in range(num_layers):
            blocks.append(
                TemporalConvBlock(
                    d_model=d_model,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            dilation *= max(1, dilation_growth)
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)
        self.pool = TemporalAttentionPool(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.pool(x)


# ── Stage 1 Encoder ───────────────────────────────────────────────────────────


class SpatioTemporalEncoder(nn.Module):
    """
    End-to-end spatiotemporal encoder: GAT trunk + temporal head.

    Supported temporal backbones:
    - ``transformer``: CLS token + absolute positional encoding.
    - ``tcn``: dilated temporal convolution network + attention pooling.
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
        temporal_backbone: str = "transformer",
        positional_encoding: str = "sinusoidal",
        num_tcn_layers: int = 4,
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.1,
        tcn_dilation_growth: int = 2,
    ) -> None:
        super().__init__()

        self.num_joints = num_joints
        self.gat_hidden = gat_hidden
        self.gat_dropout = gat_dropout
        self.d_model = d_model
        self.temporal_backbone = temporal_backbone.lower()
        self.positional_encoding = positional_encoding.lower()

        self.edge_index: torch.Tensor
        self.register_buffer("edge_index", edge_index)

        per_head_dim = gat_hidden // gat_heads
        self.gat1 = GATConv(
            in_features,
            per_head_dim,
            heads=gat_heads,
            concat=True,
            dropout=gat_dropout,
        )
        self.gat2 = GATConv(
            gat_hidden,
            per_head_dim,
            heads=gat_heads,
            concat=True,
            dropout=gat_dropout,
        )
        self.gat3 = GATConv(
            gat_hidden,
            gat_hidden,
            heads=1,
            concat=False,
            dropout=gat_dropout,
        )

        self.bn1 = nn.BatchNorm1d(gat_hidden)
        self.bn2 = nn.BatchNorm1d(gat_hidden)
        self.bn3 = nn.BatchNorm1d(gat_hidden)

        self.input_proj = nn.Linear(gat_hidden, d_model)

        self.cls_token: Optional[nn.Parameter] = None
        self.temporal_pos_enc: Optional[nn.Module] = None
        self.transformer: Optional[nn.Module] = None
        self.tcn: Optional[nn.Module] = None
        self.temporal_norm: Optional[nn.Module] = None

        if self.temporal_backbone == "transformer":
            self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=cls_std)

            self.temporal_pos_enc = build_positional_encoder(
                encoding=self.positional_encoding,
                d_model=d_model,
                max_len=seq_len + 1,
                dropout=dropout,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
            self.temporal_norm = nn.LayerNorm(d_model)

        elif self.temporal_backbone == "tcn":
            self.temporal_pos_enc = build_positional_encoder(
                encoding=self.positional_encoding,
                d_model=d_model,
                max_len=seq_len,
                dropout=tcn_dropout,
            )
            self.tcn = TemporalConvEncoder(
                d_model=d_model,
                num_layers=num_tcn_layers,
                kernel_size=tcn_kernel_size,
                dilation_growth=tcn_dilation_growth,
                dropout=tcn_dropout,
            )
            self.temporal_norm = nn.LayerNorm(d_model)

        else:
            raise ValueError(
                f"Unsupported temporal_backbone='{temporal_backbone}'. "
                "Use 'transformer' or 'tcn'."
            )

    def _gat_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all B*T graphs through the spatial GAT trunk in one pass."""
        B, T, J, fin = x.shape
        num_graphs = B * T

        x_nodes = x.reshape(num_graphs * J, fin)

        edge_index = cast(torch.Tensor, self.edge_index)
        num_edges = int(edge_index.shape[1])
        offsets = torch.arange(num_graphs, device=x.device) * J
        edge_batch = edge_index.unsqueeze(1).expand(-1, num_graphs, -1)
        edge_batch = edge_batch + offsets.view(1, -1, 1)
        edge_flat = edge_batch.reshape(2, num_graphs * num_edges)

        batch_tensor = torch.arange(num_graphs, device=x.device).repeat_interleave(J)

        h = self.gat1(x_nodes, edge_flat)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.gat_dropout, training=self.training)

        h = self.gat2(h, edge_flat)
        h = self.bn2(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.gat_dropout, training=self.training)

        h = self.gat3(h, edge_flat)
        h = self.bn3(h)
        h = F.elu(h)

        spatial = global_mean_pool(h, batch_tensor)
        return spatial.reshape(B, T, self.gat_hidden)

    def _forward_transformer(self, h: torch.Tensor) -> torch.Tensor:
        """Temporal forward pass for the transformer experiment."""
        if (
            self.cls_token is None
            or self.transformer is None
            or self.temporal_norm is None
        ):
            raise RuntimeError("Transformer temporal modules are not initialised.")

        batch_size = h.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls, h], dim=1)
        if self.temporal_pos_enc is not None:
            h = self.temporal_pos_enc(h)
        h = self.transformer(h)
        h = self.temporal_norm(h)
        return h[:, 0, :]

    def _forward_tcn(self, h: torch.Tensor) -> torch.Tensor:
        """Temporal forward pass for the dilated TCN experiment."""
        if self.tcn is None or self.temporal_norm is None:
            raise RuntimeError("TCN temporal modules are not initialised.")
        if self.temporal_pos_enc is not None:
            h = self.temporal_pos_enc(h)
        h = self.temporal_norm(h)
        return self.tcn(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of skeleton sequences.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, J, F).

        Returns
        -------
        cls_embedding : torch.Tensor
            Shape (B, d_model).
        """
        spatial = self._gat_forward(x)
        h = self.input_proj(spatial)

        if self.temporal_backbone == "transformer":
            return self._forward_transformer(h)
        return self._forward_tcn(h)


# ── Action Classifier ─────────────────────────────────────────────────────────


class ActionClassifier(nn.Module):
    """SpatioTemporalEncoder plus linear classification head."""

    def __init__(self, encoder: SpatioTemporalEncoder, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_emb = self.encoder(x)
        return self.head(cls_emb)


# ── Factory Helper ────────────────────────────────────────────────────────────


def build_model(
    cfg: Dict,
    edge_index: torch.Tensor,
) -> Tuple[SpatioTemporalEncoder, ActionClassifier]:
    """Construct encoder and classifier from a config dictionary."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    encoder = SpatioTemporalEncoder(
        num_joints=data_cfg["num_joints"],
        in_features=model_cfg["in_features"],
        edge_index=edge_index,
        gat_hidden=model_cfg["gat_hidden_dim"],
        gat_heads=model_cfg["gat_heads"],
        gat_dropout=model_cfg["gat_dropout"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_transformer_layers"],
        dim_ff=model_cfg["dim_feedforward"],
        dropout=model_cfg["transformer_dropout"],
        seq_len=data_cfg["seq_len"],
        cls_std=model_cfg["cls_std"],
        temporal_backbone=model_cfg.get("temporal_backbone", "transformer"),
        positional_encoding=model_cfg.get("positional_encoding", "sinusoidal"),
        num_tcn_layers=model_cfg.get("num_tcn_layers", 4),
        tcn_kernel_size=model_cfg.get("tcn_kernel_size", 3),
        tcn_dropout=model_cfg.get("tcn_dropout", model_cfg["transformer_dropout"]),
        tcn_dilation_growth=model_cfg.get("tcn_dilation_growth", 2),
    )
    classifier = ActionClassifier(encoder, num_classes=data_cfg["num_classes"])
    total = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    log.info("Model built: %d trainable parameters", total)
    return encoder, classifier


# ── Quick Sanity Check ────────────────────────────────────────────────────────


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
