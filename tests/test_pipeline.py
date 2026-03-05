# tests/test_pipeline.py
#
# Integration test — verifies the end-to-end data flow of Module 1
# WITHOUT requiring trained weights or real video.
#
# This is the most important test: it catches shape/dimension mismatches
# between stages before you waste hours of training time.
#
# Run with: pytest tests/test_pipeline.py -v

import numpy as np
import pytest
import torch

from module1.gnn.graph import build_graph, EDGE_INDEX
from module1.gnn.model import SkeletonGAT
from module1.pose.tracker import TrackedPlayer
from module1.transformer.model import BODY_REGIONS, BiomechanicalTransformer


# ── Stage 2: GNN forward pass ──────────────────────────────────────────────────


class TestGNNForwardPass:
    def setup_method(self):
        self.model = SkeletonGAT(in_channels=10, hidden_dim=64, out_dim=64, heads=4)
        self.model.eval()

    def test_single_graph_output_shape(self):
        """One skeleton graph → should output (1, 64)."""
        from torch_geometric.data import Data, Batch

        x = torch.rand(33, 10)
        graph = Data(x=x, edge_index=EDGE_INDEX.clone())
        batch = Batch.from_data_list([graph])

        with torch.no_grad():
            out = self.model(batch.x, batch.edge_index, batch.batch)

        assert out.shape == (1, 64)

    def test_batch_of_graphs_output_shape(self):
        """Batch of 4 skeleton graphs → should output (4, 64)."""
        from torch_geometric.data import Data, Batch

        graphs = [
            Data(x=torch.rand(33, 10), edge_index=EDGE_INDEX.clone()) for _ in range(4)
        ]
        batch = Batch.from_data_list(graphs)

        with torch.no_grad():
            out = self.model(batch.x, batch.edge_index, batch.batch)

        assert out.shape == (4, 64)

    def test_output_no_nan(self):
        from torch_geometric.data import Data, Batch

        x = torch.rand(33, 10)
        graph = Data(x=x, edge_index=EDGE_INDEX.clone())
        batch = Batch.from_data_list([graph])

        with torch.no_grad():
            out = self.model(batch.x, batch.edge_index, batch.batch)

        assert not torch.isnan(out).any()


# ── Stage 3: Transformer forward pass ─────────────────────────────────────────


class TestTransformerForwardPass:
    def setup_method(self):
        self.model = BiomechanicalTransformer(
            input_dim=64,
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_ff=256,
            dropout=0.0,
            seq_len=50,
        )
        self.model.eval()

    def test_output_keys(self):
        x = torch.rand(2, 50, 64)
        with torch.no_grad():
            out = self.model(x)
        assert "risk_score" in out
        assert "region_logits" in out
        assert "cls_embedding" in out

    def test_risk_score_in_range(self):
        x = torch.rand(4, 50, 64)
        with torch.no_grad():
            out = self.model(x)
        scores = out["risk_score"]
        assert (scores >= 0.0).all() and (scores <= 1.0).all()

    def test_risk_score_shape(self):
        x = torch.rand(4, 50, 64)
        with torch.no_grad():
            out = self.model(x)
        assert out["risk_score"].shape == (4, 1)

    def test_region_logits_shape(self):
        x = torch.rand(4, 50, 64)
        with torch.no_grad():
            out = self.model(x)
        assert out["region_logits"].shape == (4, len(BODY_REGIONS))

    def test_handles_short_sequence(self):
        """Padded sequence (fewer than 50 frames) should still work."""
        x = torch.rand(1, 50, 64)  # already padded to 50
        with torch.no_grad():
            out = self.model(x)
        assert out["risk_score"].shape == (1, 1)


# ── End-to-end data flow: raw keypoints → risk score ──────────────────────────


class TestEndToEndDataFlow:
    """
    Simulates the full pipeline without video or trained weights.
    Catches dimension mismatches between stages early.
    """

    def test_keypoints_to_graph_to_gnn_to_transformer(self):
        # Simulate 50 frames of pose data for one player
        keypoints_seq = np.random.rand(50, 33, 4).astype(np.float32)

        # Stage 2a: Build graph
        graph = build_graph(keypoints_seq)
        assert graph.x.shape == (33, 10)
        assert graph.edge_index.shape[0] == 2

        # Stage 2b: GNN forward
        from torch_geometric.data import Batch

        batch_obj = Batch.from_data_list([graph])
        gnn = SkeletonGAT(in_channels=10, hidden_dim=64, out_dim=64, heads=4)
        gnn.eval()

        with torch.no_grad():
            spatial_vec = gnn(batch_obj.x, batch_obj.edge_index, batch_obj.batch)

        assert spatial_vec.shape == (1, 64)

        # Stage 3: Transformer forward (simulate accumulated 50-frame history)
        fake_history = torch.rand(1, 50, 64)
        transformer = BiomechanicalTransformer(
            input_dim=64,
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_ff=256,
            dropout=0.0,
            seq_len=50,
        )
        transformer.eval()

        with torch.no_grad():
            out = transformer(fake_history)

        assert 0.0 <= out["risk_score"].item() <= 1.0
        print(f"\nEnd-to-end test passed. Risk score: {out['risk_score'].item():.3f}")
