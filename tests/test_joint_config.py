# tests/test_joint_config.py
#
# Unit tests for module1/data/joint_config.py
# ─────────────────────────────────────────────
# Covers: joint count, edge validity, bidirectionality, mirror pairs,
#         flip feature indices, angle triplets, and body-region mappings.
#
# Run with: pytest tests/test_joint_config.py -v

import pytest
import torch

from module1.data.joint_config import (
    ANGLE_TRIPLETS,
    BODY_REGIONS,
    EDGE_INDEX,
    FLIP_FEATURE_INDICES,
    JOINT_NAMES,
    MIRROR_JOINT_PAIRS,
    NUM_EDGES,
    NUM_JOINTS,
    NUM_REGIONS,
    REGION_NAMES,
    SKELETON_EDGES,
    build_edge_index,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def edge_index() -> torch.Tensor:
    """Freshly built edge index tensor."""
    return build_edge_index()


# ── Joint names ───────────────────────────────────────────────────────────────


class TestJointNames:
    def test_count(self) -> None:
        assert len(JOINT_NAMES) == 32

    def test_num_joints_constant(self) -> None:
        assert NUM_JOINTS == 32

    def test_all_strings(self) -> None:
        for name in JOINT_NAMES:
            assert isinstance(name, str) and name

    def test_unique_names(self) -> None:
        assert len(JOINT_NAMES) == len(set(JOINT_NAMES)), "Duplicate joint names"

    def test_pelvis_is_root(self) -> None:
        assert JOINT_NAMES[0] == "Pelvis"


# ── Skeleton edges ────────────────────────────────────────────────────────────


class TestSkeletonEdges:
    def test_num_edges_constant_matches_list(self) -> None:
        assert NUM_EDGES == len(SKELETON_EDGES)

    def test_expected_edge_count(self) -> None:
        # 31 unique undirected edges (as defined in joint_config.py)
        assert NUM_EDGES == 31

    def test_no_self_loops(self) -> None:
        for src, dst in SKELETON_EDGES:
            assert src != dst, f"Self-loop at joint {src}"

    def test_node_indices_in_range(self) -> None:
        for src, dst in SKELETON_EDGES:
            assert 0 <= src < NUM_JOINTS, f"src {src} out of range"
            assert 0 <= dst < NUM_JOINTS, f"dst {dst} out of range"

    def test_no_duplicate_edges(self) -> None:
        # Count both (a,b) and (b,a) as the same undirected edge
        undirected = {frozenset((s, d)) for s, d in SKELETON_EDGES}
        assert len(undirected) == len(SKELETON_EDGES), "Duplicate edges detected"


# ── build_edge_index ──────────────────────────────────────────────────────────


class TestBuildEdgeIndex:
    def test_shape(self, edge_index: torch.Tensor) -> None:
        assert edge_index.shape == (2, 2 * NUM_EDGES)

    def test_dtype(self, edge_index: torch.Tensor) -> None:
        assert edge_index.dtype == torch.long

    def test_node_range(self, edge_index: torch.Tensor) -> None:
        assert int(edge_index.min()) >= 0
        assert int(edge_index.max()) < NUM_JOINTS

    def test_bidirectional(self, edge_index: torch.Tensor) -> None:
        # For every forward edge (s→d) there must be a reverse edge (d→s)
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        for s, d in SKELETON_EDGES:
            assert (s, d) in edges, f"Missing forward edge ({s}→{d})"
            assert (d, s) in edges, f"Missing reverse edge ({d}→{s})"

    def test_prebuilt_constant_matches(self, edge_index: torch.Tensor) -> None:
        assert torch.equal(edge_index, EDGE_INDEX)


# ── Mirror pairs ──────────────────────────────────────────────────────────────


class TestMirrorPairs:
    def test_indices_in_range(self) -> None:
        for l, r in MIRROR_JOINT_PAIRS:
            assert 0 <= l < NUM_JOINTS, f"Left mirror index {l} out of range"
            assert 0 <= r < NUM_JOINTS, f"Right mirror index {r} out of range"

    def test_no_self_pairs(self) -> None:
        for l, r in MIRROR_JOINT_PAIRS:
            assert l != r, f"Self-pair at joint {l}"

    def test_no_duplicate_joint_appearances(self) -> None:
        # Each joint should appear at most once across all pairs
        flat = [j for pair in MIRROR_JOINT_PAIRS for j in pair]
        assert len(flat) == len(set(flat)), "A joint appears in multiple mirror pairs"

    def test_flip_feature_indices(self) -> None:
        # Must include x (0), vx (3), ax (6)
        assert set(FLIP_FEATURE_INDICES) == {0, 3, 6}


# ── Angle triplets ────────────────────────────────────────────────────────────


class TestAngleTriplets:
    def test_joint_indices_in_range(self) -> None:
        for joint_idx, (prox, dist) in ANGLE_TRIPLETS.items():
            assert 0 <= joint_idx < NUM_JOINTS
            assert 0 <= prox < NUM_JOINTS
            assert 0 <= dist < NUM_JOINTS

    def test_no_degenerate_triplets(self) -> None:
        # joint_idx must differ from both neighbours
        for joint_idx, (prox, dist) in ANGLE_TRIPLETS.items():
            assert joint_idx != prox
            assert joint_idx != dist


# ── Body regions ──────────────────────────────────────────────────────────────


class TestBodyRegions:
    def test_num_regions_constant(self) -> None:
        assert NUM_REGIONS == len(BODY_REGIONS) == len(REGION_NAMES)

    def test_expected_region_count(self) -> None:
        assert NUM_REGIONS == 11

    def test_all_joint_indices_in_range(self) -> None:
        for region, joints in BODY_REGIONS.items():
            for j in joints:
                assert 0 <= j < NUM_JOINTS, (
                    f"Region '{region}' contains out-of-range joint {j}"
                )

    def test_no_empty_regions(self) -> None:
        for region, joints in BODY_REGIONS.items():
            assert len(joints) > 0, f"Region '{region}' is empty"

    def test_region_names_list_matches_dict(self) -> None:
        assert REGION_NAMES == list(BODY_REGIONS.keys())
