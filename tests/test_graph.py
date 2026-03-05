# tests/test_graph.py
#
# Unit tests for Stage 2 — Skeleton Graph Construction
# Run with: pytest tests/test_graph.py -v

import numpy as np
import pytest
import torch

from module1.gnn.graph import (
    EDGE_INDEX,
    NUM_KEYPOINTS,
    build_graph,
    build_node_features,
    compute_joint_angle,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def dummy_seq():
    """Random keypoint sequence — shape (50, 33, 4)."""
    return np.random.rand(50, 33, 4).astype(np.float32)


@pytest.fixture
def short_seq():
    """Only 2 frames — tests padding behaviour."""
    return np.random.rand(2, 33, 4).astype(np.float32)


# ── Edge index ─────────────────────────────────────────────────────────────────


class TestEdgeIndex:
    def test_shape(self):
        assert EDGE_INDEX.shape[0] == 2

    def test_bidirectional(self):
        """Every edge must appear in both directions."""
        num_edges = EDGE_INDEX.shape[1]
        assert num_edges % 2 == 0, "Expected even number of edges (bidirectional)"

    def test_node_indices_in_range(self):
        assert EDGE_INDEX.min() >= 0
        assert EDGE_INDEX.max() < NUM_KEYPOINTS


# ── Node features ──────────────────────────────────────────────────────────────


class TestBuildNodeFeatures:
    def test_output_shape(self, dummy_seq):
        features = build_node_features(dummy_seq)
        assert features.shape == (33, 10)

    def test_output_dtype(self, dummy_seq):
        features = build_node_features(dummy_seq)
        assert features.dtype == np.float32

    def test_works_with_short_seq(self, short_seq):
        """Should not crash when fewer than 3 frames are available."""
        features = build_node_features(short_seq)
        assert features.shape == (33, 10)

    def test_no_nan_values(self, dummy_seq):
        features = build_node_features(dummy_seq)
        assert not np.isnan(features).any()


# ── Joint angle computation ────────────────────────────────────────────────────


class TestJointAngle:
    def test_straight_line_is_180(self):
        """Joint in middle of a straight line → angle should be ~180°."""
        kp = np.zeros((33, 4), dtype=np.float32)
        kp[0] = [0.0, 0.0, 0.0, 1.0]  # joint
        kp[1] = [1.0, 0.0, 0.0, 1.0]  # parent
        kp[2] = [-1.0, 0.0, 0.0, 1.0]  # child
        angle = compute_joint_angle(kp, joint_idx=0, parent_idx=1, child_idx=2)
        assert abs(angle - 180.0) < 1.0

    def test_right_angle_is_90(self):
        """Perpendicular vectors → 90°."""
        kp = np.zeros((33, 4), dtype=np.float32)
        kp[0] = [0.0, 0.0, 0.0, 1.0]
        kp[1] = [1.0, 0.0, 0.0, 1.0]
        kp[2] = [0.0, 1.0, 0.0, 1.0]
        angle = compute_joint_angle(kp, joint_idx=0, parent_idx=1, child_idx=2)
        assert abs(angle - 90.0) < 1.0

    def test_degenerate_returns_zero(self):
        """All keypoints at the same location → should return 0.0, not crash."""
        kp = np.zeros((33, 4), dtype=np.float32)
        angle = compute_joint_angle(kp, joint_idx=0, parent_idx=1, child_idx=2)
        assert angle == 0.0


# ── Graph building ─────────────────────────────────────────────────────────────


class TestBuildGraph:
    def test_node_feature_shape(self, dummy_seq):
        graph = build_graph(dummy_seq)
        assert graph.x.shape == (33, 10)

    def test_edge_index_present(self, dummy_seq):
        graph = build_graph(dummy_seq)
        assert graph.edge_index is not None
        assert graph.edge_index.shape[0] == 2

    def test_label_attached(self, dummy_seq):
        graph = build_graph(dummy_seq, label=1)
        assert graph.y.item() == 1

    def test_no_label(self, dummy_seq):
        graph = build_graph(dummy_seq)
        assert not hasattr(graph, "y") or graph.y is None

    def test_tensor_dtype(self, dummy_seq):
        graph = build_graph(dummy_seq)
        assert graph.x.dtype == torch.float32
