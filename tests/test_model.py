# tests/test_model.py
#
# Unit tests for module1/pretrain/model.py
# -----------------------------------------
# Covers absolute positional encodings, transformer and TCN temporal heads,
# encoder/classifier forward shapes, and the build_model factory.
#
# Run with: pytest tests/test_model.py -v

from __future__ import annotations

import copy

import pytest
import torch

from module1.data.joint_config import EDGE_INDEX, NUM_JOINTS
from module1.pretrain.model import (
    ActionClassifier,
    LearnedPositionalEncoding,
    PositionalEncoding,
    SpatioTemporalEncoder,
    build_model,
)


B = 2
T = 60
J = NUM_JOINTS
F_IN = 9
D_MODEL = 128
NUM_CLASSES = 102


@pytest.fixture
def dummy_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, T, J, F_IN)


@pytest.fixture
def minimal_cfg() -> dict:
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
            "temporal_backbone": "transformer",
            "positional_encoding": "sinusoidal",
        },
    }


@pytest.fixture
def transformer_encoder() -> SpatioTemporalEncoder:
    model = SpatioTemporalEncoder(
        num_joints=J,
        in_features=F_IN,
        edge_index=EDGE_INDEX.clone(),
        gat_hidden=64,
        gat_heads=4,
        d_model=D_MODEL,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        dropout=0.0,
        seq_len=T,
        temporal_backbone="transformer",
        positional_encoding="sinusoidal",
    )
    model.eval()
    return model


@pytest.fixture
def tcn_encoder() -> SpatioTemporalEncoder:
    model = SpatioTemporalEncoder(
        num_joints=J,
        in_features=F_IN,
        edge_index=EDGE_INDEX.clone(),
        gat_hidden=64,
        gat_heads=4,
        d_model=D_MODEL,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        dropout=0.0,
        seq_len=T,
        temporal_backbone="tcn",
        positional_encoding="learned",
        num_tcn_layers=3,
        tcn_kernel_size=3,
        tcn_dropout=0.0,
        tcn_dilation_growth=2,
    )
    model.eval()
    return model


class TestPositionalEncoding:
    def test_sinusoidal_shape(self) -> None:
        pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
        out = pe(torch.zeros(B, T, D_MODEL))
        assert out.shape == (B, T, D_MODEL)

    def test_sinusoidal_changes_zero_input(self) -> None:
        pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
        out = pe(torch.zeros(1, T, D_MODEL))
        assert not torch.all(out == 0)

    def test_learned_shape(self) -> None:
        pe = LearnedPositionalEncoding(d_model=D_MODEL, max_len=T, dropout=0.0)
        out = pe(torch.zeros(B, T, D_MODEL))
        assert out.shape == (B, T, D_MODEL)

    def test_learned_differs_by_position(self) -> None:
        pe = LearnedPositionalEncoding(d_model=D_MODEL, max_len=T, dropout=0.0)
        out = pe(torch.zeros(1, T, D_MODEL))
        assert not torch.allclose(out[0, 0], out[0, 1])


class TestTransformerEncoder:
    @torch.no_grad()
    def test_output_shape(
        self,
        transformer_encoder: SpatioTemporalEncoder,
        dummy_input: torch.Tensor,
    ) -> None:
        out = transformer_encoder(dummy_input)
        assert out.shape == (B, D_MODEL)

    @torch.no_grad()
    def test_no_nan(
        self, transformer_encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
    ) -> None:
        out = transformer_encoder(dummy_input)
        assert not torch.isnan(out).any()


class TestTcnEncoder:
    @torch.no_grad()
    def test_output_shape(
        self,
        tcn_encoder: SpatioTemporalEncoder,
        dummy_input: torch.Tensor,
    ) -> None:
        out = tcn_encoder(dummy_input)
        assert out.shape == (B, D_MODEL)

    @torch.no_grad()
    def test_no_nan(
        self, tcn_encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
    ) -> None:
        out = tcn_encoder(dummy_input)
        assert not torch.isnan(out).any()

    def test_gradients_reach_tcn(
        self,
        tcn_encoder: SpatioTemporalEncoder,
        dummy_input: torch.Tensor,
    ) -> None:
        tcn_encoder.train()
        loss = tcn_encoder(dummy_input).sum()
        loss.backward()
        grad_found = any(
            param.grad is not None
            for name, param in tcn_encoder.named_parameters()
            if name.startswith("tcn")
        )
        assert grad_found


class TestActionClassifier:
    @torch.no_grad()
    def test_output_shape(
        self, transformer_encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
    ) -> None:
        classifier = ActionClassifier(transformer_encoder, num_classes=NUM_CLASSES)
        classifier.eval()
        logits = classifier(dummy_input)
        assert logits.shape == (B, NUM_CLASSES)

    def test_encoder_exposed(self, transformer_encoder: SpatioTemporalEncoder) -> None:
        classifier = ActionClassifier(transformer_encoder, num_classes=NUM_CLASSES)
        assert isinstance(classifier.encoder, SpatioTemporalEncoder)


class TestBuildModel:
    def test_transformer_factory(self, minimal_cfg: dict) -> None:
        encoder, classifier = build_model(minimal_cfg, EDGE_INDEX)
        assert isinstance(encoder, SpatioTemporalEncoder)
        assert classifier.encoder is encoder

    @torch.no_grad()
    def test_transformer_factory_forward(self, minimal_cfg: dict) -> None:
        _, classifier = build_model(minimal_cfg, EDGE_INDEX)
        classifier.eval()
        logits = classifier(torch.randn(B, T, J, F_IN))
        assert logits.shape == (B, NUM_CLASSES)

    @torch.no_grad()
    def test_tcn_factory_forward(self, minimal_cfg: dict) -> None:
        cfg = copy.deepcopy(minimal_cfg)
        cfg["model"].update(
            {
                "temporal_backbone": "tcn",
                "positional_encoding": "learned",
                "num_tcn_layers": 3,
                "tcn_kernel_size": 3,
                "tcn_dropout": 0.0,
                "tcn_dilation_growth": 2,
            }
        )
        _, classifier = build_model(cfg, EDGE_INDEX)
        classifier.eval()
        logits = classifier(torch.randn(B, T, J, F_IN))
        assert logits.shape == (B, NUM_CLASSES)

    def test_invalid_backbone_raises(self, minimal_cfg: dict) -> None:
        cfg = copy.deepcopy(minimal_cfg)
        cfg["model"]["temporal_backbone"] = "perceiver"
        with pytest.raises(ValueError):
            build_model(cfg, EDGE_INDEX)
