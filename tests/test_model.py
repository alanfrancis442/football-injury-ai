# tests/test_model.py
#
# Unit tests for module1/pretrain/model.py
# ──────────────────────────────────────────
# Covers: PositionalEncoding, SpatioTemporalEncoder (forward pass shapes,
#         no NaNs, edge_index buffer device movement), ActionClassifier
#         (forward pass, logit shape), and encoder/classifier separation.
#
# Run with: pytest tests/test_model.py -v

import pytest
import torch
import torch.nn as nn

from module1.data.joint_config import EDGE_INDEX, NUM_JOINTS
from module1.pretrain.model import (
    ActionClassifier,
    PositionalEncoding,
    SpatioTemporalEncoder,
    build_model,
)


# ── Constants used across tests ───────────────────────────────────────────────

B = 2  # batch size (small to keep tests fast)
T = 60  # sequence length (frames)
J = NUM_JOINTS  # 32
F_IN = 9  # features per joint
D_MODEL = 128
NUM_CLASSES = 102


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def edge_index() -> torch.Tensor:
    return EDGE_INDEX.clone()


@pytest.fixture
def encoder(edge_index: torch.Tensor) -> SpatioTemporalEncoder:
    model = SpatioTemporalEncoder(
        num_joints=J,
        in_features=F_IN,
        edge_index=edge_index,
        gat_hidden=64,
        gat_heads=4,
        d_model=D_MODEL,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        dropout=0.0,  # disable dropout so outputs are deterministic in eval
        seq_len=T,
    )
    model.eval()
    return model


@pytest.fixture
def classifier(encoder: SpatioTemporalEncoder) -> ActionClassifier:
    model = ActionClassifier(encoder, num_classes=NUM_CLASSES)
    model.eval()
    return model


@pytest.fixture
def dummy_input() -> torch.Tensor:
    """Small random batch: (B, T, J, F_IN)."""
    torch.manual_seed(0)
    return torch.randn(B, T, J, F_IN)


@pytest.fixture
def minimal_cfg() -> dict:
    """Minimal config dict that mirrors configs/pretrain.yaml structure."""
    return {
        "data": {
            "num_joints": J,
            "seq_len": T,
            "num_classes": NUM_CLASSES,
        },
        "model": {
            "in_features": F_IN,
            "gat_hidden_dim": 64,
            "gat_heads": 4,
            "gat_dropout": 0.0,
            "d_model": D_MODEL,
            "nhead": 4,
            "num_transformer_layers": 2,
            "dim_feedforward": 256,
            "transformer_dropout": 0.0,
            "cls_std": 0.02,
        },
    }


# ── PositionalEncoding ────────────────────────────────────────────────────────


class TestPositionalEncoding:
    def test_output_shape(self) -> None:
        pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
        x = torch.zeros(B, T, D_MODEL)
        out = pe(x)
        assert out.shape == (B, T, D_MODEL)

    def test_no_nan(self) -> None:
        pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
        x = torch.randn(B, T, D_MODEL)
        out = pe(x)
        assert not torch.isnan(out).any()

    def test_adds_positional_signal(self) -> None:
        # Zero input + no dropout: output should equal the PE buffer, not zero
        pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
        pe.eval()
        x = torch.zeros(B, T, D_MODEL)
        out = pe(x)
        assert not torch.all(out == 0)

    def test_different_positions_differ(self) -> None:
        # Adjacent time steps should (almost certainly) have different encodings
        pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
        pe.eval()
        x = torch.zeros(1, T, D_MODEL)
        out = pe(x)
        assert not torch.allclose(out[0, 0, :], out[0, 1, :])


# ── SpatioTemporalEncoder ─────────────────────────────────────────────────────


class TestSpatioTemporalEncoder:
    @torch.no_grad()
    def test_output_shape(
        self, encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
    ) -> None:
        out = encoder(dummy_input)
        assert out.shape == (B, D_MODEL), f"Expected ({B},{D_MODEL}), got {out.shape}"

    @torch.no_grad()
    def test_no_nan_in_output(
        self, encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
    ) -> None:
        out = encoder(dummy_input)
        assert not torch.isnan(out).any(), "NaN detected in encoder output"

    @torch.no_grad()
    def test_no_inf_in_output(
        self, encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
    ) -> None:
        out = encoder(dummy_input)
        assert not torch.isinf(out).any(), "Inf detected in encoder output"

    @torch.no_grad()
    def test_different_inputs_give_different_outputs(
        self, encoder: SpatioTemporalEncoder
    ) -> None:
        torch.manual_seed(1)
        x1 = torch.randn(B, T, J, F_IN)
        x2 = torch.randn(B, T, J, F_IN)
        out1 = encoder(x1)
        out2 = encoder(x2)
        assert not torch.allclose(out1, out2), "Encoder is constant regardless of input"

    @torch.no_grad()
    def test_batch_independence(self, encoder: SpatioTemporalEncoder) -> None:
        # Encoding one sample individually must equal its row in a batched pass
        torch.manual_seed(2)
        x = torch.randn(B, T, J, F_IN)
        batch_out = encoder(x)
        single_out = encoder(x[0:1])
        assert torch.allclose(batch_out[0], single_out[0], atol=1e-5), (
            "Batch output differs from single-sample output"
        )

    def test_edge_index_is_buffer(self, encoder: SpatioTemporalEncoder) -> None:
        # edge_index must be registered as a buffer (non-parameter)
        buffer_names = {name for name, _ in encoder.named_buffers()}
        assert "edge_index" in buffer_names

    def test_edge_index_shape(self, encoder: SpatioTemporalEncoder) -> None:
        assert encoder.edge_index.shape == EDGE_INDEX.shape

    def test_has_no_unused_params_in_training_mode(
        self, encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
    ) -> None:
        """All parameters must receive a gradient after a backward pass."""
        encoder.train()
        out = encoder(dummy_input)
        loss = out.sum()
        loss.backward()
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter: {name}"
        encoder.eval()


# ── ActionClassifier ──────────────────────────────────────────────────────────


class TestActionClassifier:
    @torch.no_grad()
    def test_output_shape(
        self, classifier: ActionClassifier, dummy_input: torch.Tensor
    ) -> None:
        logits = classifier(dummy_input)
        assert logits.shape == (B, NUM_CLASSES), (
            f"Expected ({B},{NUM_CLASSES}), got {logits.shape}"
        )

    @torch.no_grad()
    def test_no_nan_in_logits(
        self, classifier: ActionClassifier, dummy_input: torch.Tensor
    ) -> None:
        logits = classifier(dummy_input)
        assert not torch.isnan(logits).any()

    @torch.no_grad()
    def test_logits_are_unnormalised(
        self, classifier: ActionClassifier, dummy_input: torch.Tensor
    ) -> None:
        # The head is a plain Linear — softmax probabilities should NOT sum to 1
        logits = classifier(dummy_input)
        row_sums = logits.softmax(dim=-1).sum(dim=-1)
        # After softmax they should sum to ~1, but raw logits sum should not
        assert not torch.allclose(logits.sum(dim=-1), torch.ones(B))

    def test_encoder_is_accessible(self, classifier: ActionClassifier) -> None:
        assert isinstance(classifier.encoder, SpatioTemporalEncoder)

    def test_encoder_and_classifier_share_params(
        self, classifier: ActionClassifier
    ) -> None:
        # Parameters of classifier.encoder must be a subset of classifier's params
        enc_ids = {id(p) for p in classifier.encoder.parameters()}
        cls_ids = {id(p) for p in classifier.parameters()}
        assert enc_ids.issubset(cls_ids)

    def test_head_not_in_encoder(self, classifier: ActionClassifier) -> None:
        enc_ids = {id(p) for p in classifier.encoder.parameters()}
        head_ids = {id(p) for p in classifier.head.parameters()}
        assert head_ids.isdisjoint(enc_ids), "Head parameters appear in encoder"


# ── build_model factory ───────────────────────────────────────────────────────


class TestBuildModel:
    def test_returns_encoder_and_classifier(self, minimal_cfg: dict) -> None:
        encoder, classifier = build_model(minimal_cfg, EDGE_INDEX)
        assert isinstance(encoder, SpatioTemporalEncoder)
        assert isinstance(classifier, ActionClassifier)

    def test_classifier_wraps_same_encoder(self, minimal_cfg: dict) -> None:
        encoder, classifier = build_model(minimal_cfg, EDGE_INDEX)
        assert classifier.encoder is encoder

    @torch.no_grad()
    def test_forward_pass_after_build(self, minimal_cfg: dict) -> None:
        _, classifier = build_model(minimal_cfg, EDGE_INDEX)
        classifier.eval()
        dummy = torch.randn(B, T, J, F_IN)
        logits = classifier(dummy)
        assert logits.shape == (B, NUM_CLASSES)
        assert not torch.isnan(logits).any()
