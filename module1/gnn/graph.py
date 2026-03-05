# module1/gnn/graph.py
#
# Stage 2 — Skeleton Graph Definition
# ─────────────────────────────────────
# Defines the human skeleton as a graph:
#   Nodes  = 33 body joints (MediaPipe keypoints)
#   Edges  = anatomical bone connections (kinematic chain)
#
# Also computes per-node features from raw keypoints:
#   - 3D position  (x, y, z)
#   - Velocity     (dx, dy, dz)     — finite difference between frames
#   - Acceleration (ddx, ddy, ddz)  — finite difference of velocity
#   - Joint angle                   — angle at this joint from adjacent bones
#   - Visibility / confidence score
#
# Total per-node features: 10
#
# Output is a PyTorch Geometric `Data` object — the standard format for GNNs.

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Data


# ── Skeleton edge definition ───────────────────────────────────────────────────
# Each tuple is (parent_joint_index, child_joint_index)
# Based on the MediaPipe Pose kinematic chain.

SKELETON_EDGES = [
    # Spine / torso
    (11, 12),  # left shoulder  — right shoulder
    (11, 23),  # left shoulder  — left hip
    (12, 24),  # right shoulder — right hip
    (23, 24),  # left hip       — right hip
    # Left arm
    (11, 13),  # left shoulder — left elbow
    (13, 15),  # left elbow    — left wrist
    # Right arm
    (12, 14),  # right shoulder — right elbow
    (14, 16),  # right elbow    — right wrist
    # Left leg
    (23, 25),  # left hip   — left knee
    (25, 27),  # left knee  — left ankle
    (27, 29),  # left ankle — left heel
    (27, 31),  # left ankle — left foot index
    # Right leg
    (24, 26),  # right hip   — right knee
    (26, 28),  # right knee  — right ankle
    (28, 30),  # right ankle — right heel
    (28, 32),  # right ankle — right foot index
    # Head
    (0, 11),  # nose — left shoulder
    (0, 12),  # nose — right shoulder
]


# Build edge index tensor (2, num_edges) — bidirectional
def _build_edge_index() -> torch.Tensor:
    src, dst = zip(*SKELETON_EDGES)
    # Add reverse edges so the graph is undirected
    edge_src = list(src) + list(dst)
    edge_dst = list(dst) + list(src)
    return torch.tensor([edge_src, edge_dst], dtype=torch.long)


EDGE_INDEX = _build_edge_index()  # computed once at import time


# ── Feature computation ────────────────────────────────────────────────────────


def compute_joint_angle(
    kp: np.ndarray,
    joint_idx: int,
    parent_idx: int,
    child_idx: int,
) -> float:
    """
    Compute the angle at `joint_idx` formed by the vectors:
        joint → parent  and  joint → child

    Returns angle in degrees. Returns 0.0 if any keypoint is missing.
    """
    j = kp[joint_idx, :3]
    p = kp[parent_idx, :3]
    c = kp[child_idx, :3]

    v1 = p - j
    v2 = c - j

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0

    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# Joint angle triplets: (joint, parent_towards_torso, child_away_from_torso)
# Used to compute meaningful biomechanical angles at key joints
ANGLE_TRIPLETS = {
    13: (11, 15),  # left elbow  (shoulder–elbow–wrist)
    14: (12, 16),  # right elbow
    25: (23, 27),  # left knee   (hip–knee–ankle)
    26: (24, 28),  # right knee
    23: (11, 25),  # left hip    (shoulder–hip–knee)
    24: (12, 26),  # right hip
    27: (25, 29),  # left ankle  (knee–ankle–heel)
    28: (26, 30),  # right ankle
}


def build_node_features(
    keypoints_seq: np.ndarray,  # shape (T, 33, 4) — T frames of (x,y,z,vis)
) -> np.ndarray:
    """
    Compute per-node feature matrix for the LAST frame in the sequence,
    using the sequence history for velocity and acceleration.

    Returns
    -------
    features : np.ndarray  shape (33, 10)
        Per joint: [x, y, z, vx, vy, vz, ax, ay, az, joint_angle, visibility]
        Note: 10 features total.
    """
    T = keypoints_seq.shape[0]

    kp_now = keypoints_seq[-1]  # (33, 4)
    kp_prev = keypoints_seq[-2] if T >= 2 else kp_now  # (33, 4)
    kp_pp = keypoints_seq[-3] if T >= 3 else kp_prev  # (33, 4)

    pos = kp_now[:, :3]  # (33, 3)
    vel = kp_now[:, :3] - kp_prev[:, :3]  # (33, 3)
    acc = vel - (kp_prev[:, :3] - kp_pp[:, :3])  # (33, 3)
    vis = kp_now[:, 3:4]  # (33, 1)

    # Joint angles (one per joint, 0.0 for joints without a defined triplet)
    angles = np.zeros((33, 1), dtype=np.float32)
    for joint, (parent, child) in ANGLE_TRIPLETS.items():
        angles[joint, 0] = compute_joint_angle(kp_now, joint, parent, child)

    features = np.concatenate([pos, vel, acc, angles, vis], axis=1)  # (33, 10)
    return features.astype(np.float32)


# ── PyTorch Geometric Data builder ────────────────────────────────────────────


def build_graph(
    keypoints_seq: np.ndarray,  # shape (T, 33, 4)
    label: Optional[int] = None,  # 0 = no risk, 1 = risk (for training)
) -> Data:
    """
    Build a PyTorch Geometric Data object for one player's current frame.

    Parameters
    ----------
    keypoints_seq : np.ndarray  shape (T, 33, 4)
        Sequence of keypoint frames (uses last frame + history for features).
    label : int, optional
        Ground-truth injury risk label for training.

    Returns
    -------
    torch_geometric.data.Data
        .x           : node features (33, 10)
        .edge_index  : skeleton connectivity (2, num_edges)
        .y           : label tensor (if provided)
    """
    node_features = build_node_features(keypoints_seq)  # (33, 10)

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = EDGE_INDEX.clone()

    data = Data(x=x, edge_index=edge_index)

    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)

    return data


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create a dummy sequence (3 frames, 33 joints, 4 values)
    dummy_seq = np.random.rand(3, 33, 4).astype(np.float32)
    graph = build_graph(dummy_seq, label=0)

    print(f"Node features shape : {graph.x.shape}")  # (33, 10)
    print(f"Edge index shape    : {graph.edge_index.shape}")  # (2, n_edges)
    print(f"Label               : {graph.y}")
    print("Graph construction OK.")
