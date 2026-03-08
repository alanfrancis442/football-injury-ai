# module1/pretrain/model.py
#
# Stage 1 - SpatioTemporalEncoder + ActionClassifier
# ---------------------------------------------------
# End-to-end model that encodes a sequence of skeleton graphs.
#
# Input: (B, T, J, F) where F=9 [x,y,z,vx,vy,vz,ax,ay,az].
#
# Spatial stage (shared across all T frames):
#   GATConv -> BN -> ELU -> Dropout
#   GATConv -> BN -> ELU -> Dropout
#   GATConv -> BN -> ELU
#   global_mean_pool over joints -> (B, T, gat_hidden)
#
# Temporal stage (configurable):
#   1) Transformer (baseline): CLS + sinusoidal position + TransformerEncoder
#   2) Perceiver latent bottleneck: cross-attn + latent self-attn (+ optional RoPE)

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

from utils.logger import get_logger

log = get_logger(__name__)


# -- Positional encoding -------------------------------------------------------


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_len : int
        Maximum sequence length.
    dropout : float
        Dropout applied after adding positional encoding.
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
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate even/odd features for rotary embedding."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(-2)


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embedding helper for attention Q/K tensors.

    Parameters
    ----------
    dim : int
        Attention head dimension (must be even).
    base : float
        RoPE base frequency.
    """

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires even head dimension, got dim={dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def get_cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cosine and sine tensors shaped for (B, H, T, D)."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, D)
        cos = emb.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        sin = emb.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        return cos, sin


# -- Attention blocks ----------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention supporting self-attention and cross-attention.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    nhead : int
        Number of attention heads.
    dropout : float
        Dropout on attention probabilities and output projection.
    cross_attention : bool
        If True, query comes from x and key/value comes from context.
    use_rope : bool
        If True, apply RoPE to q/k tensors.
    rope_base : float
        RoPE base frequency.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        cross_attention: bool = False,
        use_rope: bool = False,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5
        self.cross_attention = cross_attention
        self.use_rope = use_rope

        if cross_attention:
            self.q_proj = nn.Linear(d_model, d_model)
            self.kv_proj = nn.Linear(d_model, 2 * d_model)
        else:
            self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, base=rope_base)
        else:
            self.rope = None

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, H, T, Dh)"""
        B, T, _ = x.shape
        x = x.view(B, T, self.nhead, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, Dh) -> (B, T, D)"""
        B, H, T, Dh = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * Dh)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Query tensor, shape (B, Tq, D).
        context : torch.Tensor, optional
            Key/value source, shape (B, Tk, D). Required for cross-attention.

        Returns
        -------
        out : torch.Tensor
            Attention output, shape (B, Tq, D).
        """
        if self.cross_attention:
            if context is None:
                raise ValueError("context is required for cross-attention")
            q = self.q_proj(x)
            kv = self.kv_proj(context)
            k, v = torch.chunk(kv, 2, dim=-1)
        else:
            qkv = self.qkv_proj(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        if self.rope is not None:
            cos_q, sin_q = self.rope.get_cos_sin(q.shape[2], q.device, q.dtype)
            cos_k, sin_k = self.rope.get_cos_sin(k.shape[2], k.device, k.dtype)
            q = (q * cos_q) + (_rotate_half(q) * sin_q)
            k = (k * cos_k) + (_rotate_half(k) * sin_k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        out = torch.matmul(attn_probs, v)
        out = self._merge_heads(out)
        out = self.out_proj(out)
        out = self.out_dropout(out)
        return out


class FeedForward(nn.Module):
    """Transformer-style feedforward block."""

    def __init__(self, d_model: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverLayer(nn.Module):
    """
    One Perceiver layer: cross-attention + latent self-attention + FFN.
    """

    def __init__(
        self,
        d_model: int,
        cross_attn_heads: int,
        self_attn_heads: int,
        dim_ff: int,
        dropout: float,
        use_rope: bool,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.norm_latent_cross = nn.LayerNorm(d_model)
        self.norm_input_cross = nn.LayerNorm(d_model)
        self.norm_latent_self = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            nhead=cross_attn_heads,
            dropout=dropout,
            cross_attention=True,
            use_rope=use_rope,
            rope_base=rope_base,
        )
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            nhead=self_attn_heads,
            dropout=dropout,
            cross_attention=False,
            use_rope=use_rope,
            rope_base=rope_base,
        )
        self.ff = FeedForward(d_model=d_model, dim_ff=dim_ff, dropout=dropout)

    def forward(self, latents: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        latents = latents + self.cross_attn(
            x=self.norm_latent_cross(latents),
            context=self.norm_input_cross(tokens),
        )
        latents = latents + self.self_attn(self.norm_latent_self(latents))
        latents = latents + self.ff(self.norm_ff(latents))
        return latents


# -- SpatioTemporalEncoder -----------------------------------------------------


class SpatioTemporalEncoder(nn.Module):
    """
    End-to-end spatiotemporal encoder: GAT + temporal backbone.

    Temporal backbones:
    - "transformer": baseline with CLS token and sinusoidal encoding.
    - "perceiver": learnable latent bottleneck; supports sinusoidal or RoPE.
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
        num_latents: int = 64,
        num_perceiver_layers: int = 2,
        cross_attn_heads: int = 4,
        self_attn_heads: int = 4,
        perceiver_dim_ff: int = 256,
        perceiver_dropout: float = 0.1,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()

        self.num_joints = num_joints
        self.gat_hidden = gat_hidden
        self.gat_dropout = gat_dropout
        self.d_model = d_model
        self.temporal_backbone = temporal_backbone.lower()
        self.positional_encoding = positional_encoding.lower()

        self.register_buffer("edge_index", edge_index)

        # Spatial GAT stack
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

        # Optional temporal modules initialised per-backbone.
        self.pos_enc: Optional[nn.Module] = None
        self.transformer: Optional[nn.Module] = None
        self.latents: Optional[nn.Parameter] = None
        self.perceiver_layers: Optional[nn.ModuleList] = None

        # Temporal backbone selection
        if self.temporal_backbone == "transformer":
            if self.positional_encoding != "sinusoidal":
                raise ValueError(
                    "Transformer backbone currently supports only "
                    "positional_encoding='sinusoidal'."
                )

            self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=cls_std)

            self.pos_enc = PositionalEncoding(
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
                encoder_layer, num_layers=num_layers
            )
            self.norm = nn.LayerNorm(d_model)

        elif self.temporal_backbone == "perceiver":
            if self.positional_encoding not in ("sinusoidal", "rope"):
                raise ValueError(
                    "Perceiver backbone supports positional_encoding='sinusoidal' "
                    "or 'rope'."
                )

            self.latents = nn.Parameter(torch.empty(1, num_latents, d_model))
            nn.init.trunc_normal_(self.latents, std=cls_std)

            self.pos_enc = None
            if self.positional_encoding == "sinusoidal":
                self.pos_enc = PositionalEncoding(
                    d_model=d_model,
                    max_len=seq_len,
                    dropout=perceiver_dropout,
                )

            use_rope = self.positional_encoding == "rope"
            self.perceiver_layers = nn.ModuleList(
                [
                    PerceiverLayer(
                        d_model=d_model,
                        cross_attn_heads=cross_attn_heads,
                        self_attn_heads=self_attn_heads,
                        dim_ff=perceiver_dim_ff,
                        dropout=perceiver_dropout,
                        use_rope=use_rope,
                        rope_base=rope_base,
                    )
                    for _ in range(num_perceiver_layers)
                ]
            )
            self.norm = nn.LayerNorm(d_model)

        else:
            raise ValueError(
                f"Unsupported temporal_backbone='{temporal_backbone}'. "
                "Use 'transformer' or 'perceiver'."
            )

    def _gat_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all B*T graphs through GAT in one batched pass."""
        B, T, J, Fin = x.shape
        num_graphs = B * T

        x_nodes = x.reshape(num_graphs * J, Fin)

        E = self.edge_index.shape[1]
        offsets = torch.arange(num_graphs, device=x.device) * J
        edge_batch = self.edge_index.unsqueeze(1).expand(-1, num_graphs, -1)
        edge_batch = edge_batch + offsets.view(1, -1, 1)
        edge_flat = edge_batch.reshape(2, num_graphs * E)

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
        """Temporal forward for baseline transformer path."""
        if self.pos_enc is None or self.transformer is None:
            raise RuntimeError("Transformer temporal modules are not initialised.")
        B = h.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos_enc(h)
        h = self.transformer(h)
        h = self.norm(h)
        return h[:, 0, :]

    def _forward_perceiver(self, h: torch.Tensor) -> torch.Tensor:
        """Temporal forward for Perceiver latent bottleneck path."""
        if self.latents is None or self.perceiver_layers is None:
            raise RuntimeError("Perceiver temporal modules are not initialised.")
        if self.pos_enc is not None:
            h = self.pos_enc(h)

        B = h.shape[0]
        latents = self.latents.expand(B, -1, -1)
        for layer in self.perceiver_layers:
            latents = layer(latents, h)

        latents = self.norm(latents)
        return latents[:, 0, :]

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
        return self._forward_perceiver(h)


# -- ActionClassifier ----------------------------------------------------------


class ActionClassifier(nn.Module):
    """SpatioTemporalEncoder + linear classification head."""

    def __init__(self, encoder: SpatioTemporalEncoder, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_emb = self.encoder(x)
        return self.head(cls_emb)


# -- Factory helper ------------------------------------------------------------


def build_model(
    cfg: dict,
    edge_index: torch.Tensor,
) -> Tuple[SpatioTemporalEncoder, ActionClassifier]:
    """Construct encoder and classifier from config."""
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
        temporal_backbone=m.get("temporal_backbone", "transformer"),
        positional_encoding=m.get("positional_encoding", "sinusoidal"),
        num_latents=m.get("num_latents", 64),
        num_perceiver_layers=m.get("num_perceiver_layers", 2),
        cross_attn_heads=m.get("cross_attn_heads", m["nhead"]),
        self_attn_heads=m.get("self_attn_heads", m["nhead"]),
        perceiver_dim_ff=m.get("perceiver_dim_feedforward", m["dim_feedforward"]),
        perceiver_dropout=m.get("perceiver_dropout", m["transformer_dropout"]),
        rope_base=m.get("rope_base", 10000.0),
    )
    classifier = ActionClassifier(encoder, num_classes=d["num_classes"])
    total = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    log.info("Model built: %d trainable parameters", total)
    return encoder, classifier


# -- Quick sanity check --------------------------------------------------------


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
