# module1/data/joint_config.py
#
# Stage 2 — ANUBIS 32-Joint Skeleton Configuration
# ──────────────────────────────────────────────────
# Defines the 32-joint skeleton used by the preprocessed ANUBIS dataset:
#   • Joint names and indices
#   • Skeleton edges for the Graph Attention Network
#   • Left/right mirror pairs for horizontal-flip augmentation
#   • Biomechanical angle triplets for diagnostics
#   • Body-region groupings for fine-tuning (injury localisation head)
#
# IMPORTANT — verify joint indices against your preprocessing notebook.
# The indices here follow a standard 32-joint motion-capture hierarchy
# (pelvis-rooted, pelvis coordinate already subtracted during preprocessing).
# If your ANUBIS preprocessing used a different joint ordering, update
# JOINT_NAMES and SKELETON_EDGES accordingly; all downstream code will
# automatically pick up the change.

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


# ── Joint names (index → label) ───────────────────────────────────────────────
# Hierarchy:
#   Pelvis (0)
#   ├─ Spine chain:  0→1→2→3→4→5   (Pelvis→Spine1→Spine2→Spine3→Neck→Head)
#   ├─ Left leg:     0→6→7→8        (Pelvis→L_Hip→L_Knee→L_Ankle)
#   │                  8→9          (L_Ankle→L_Heel)
#   │                  8→10         (L_Ankle→L_FootIndex)
#   │                  10→28        (L_FootIndex→L_Toes)
#   ├─ Right leg:    0→11→12→13     (Pelvis→R_Hip→R_Knee→R_Ankle)
#   │                  13→14        (R_Ankle→R_Heel)
#   │                  13→15        (R_Ankle→R_FootIndex)
#   │                  15→29        (R_FootIndex→R_Toes)
#   ├─ Left arm:     3→16→17→18→19→20→21
#   │               (Spine3→L_Collar→L_Shoulder→L_Elbow→L_Wrist→L_Hand→L_Fingertip)
#   └─ Right arm:    3→22→23→24→25→26→27
#                   (Spine3→R_Collar→R_Shoulder→R_Elbow→R_Wrist→R_Hand→R_Fingertip)
#   Extra:           28→30 (L_Toes→L_ThumbToe), 29→31 (R_Toes→R_ThumbToe)

JOINT_NAMES: List[str] = [
    "Pelvis",  # 0  — root, always at origin after normalisation
    "Spine1",  # 1  — lower lumbar
    "Spine2",  # 2  — mid lumbar
    "Spine3",  # 3  — thorax / upper spine
    "Neck",  # 4
    "Head",  # 5
    "L_Hip",  # 6
    "L_Knee",  # 7
    "L_Ankle",  # 8
    "L_Heel",  # 9
    "L_FootIndex",  # 10 — ball of left foot
    "R_Hip",  # 11
    "R_Knee",  # 12
    "R_Ankle",  # 13
    "R_Heel",  # 14
    "R_FootIndex",  # 15 — ball of right foot
    "L_Collar",  # 16 — left clavicle
    "L_Shoulder",  # 17
    "L_Elbow",  # 18
    "L_Wrist",  # 19
    "L_Hand",  # 20
    "L_Fingertip",  # 21
    "R_Collar",  # 22 — right clavicle
    "R_Shoulder",  # 23
    "R_Elbow",  # 24
    "R_Wrist",  # 25
    "R_Hand",  # 26
    "R_Fingertip",  # 27
    "L_Toes",  # 28 — left toe base
    "R_Toes",  # 29 — right toe base
    "L_ThumbToe",  # 30 — left big-toe tip
    "R_ThumbToe",  # 31 — right big-toe tip
]

NUM_JOINTS: int = len(JOINT_NAMES)  # 32


# ── Skeleton edges ─────────────────────────────────────────────────────────────
# Each tuple is (parent_index, child_index).
# Edges are made bidirectional inside _build_edge_index().

SKELETON_EDGES: List[Tuple[int, int]] = [
    # Spine / head chain
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    # Pelvis to hips
    (0, 6),
    (0, 11),
    # Left leg
    (6, 7),  # hip → knee
    (7, 8),  # knee → ankle
    (8, 9),  # ankle → heel
    (8, 10),  # ankle → ball of foot
    (10, 28),  # ball of foot → toe base
    (28, 30),  # toe base → big-toe tip
    # Right leg
    (11, 12),
    (12, 13),
    (13, 14),
    (13, 15),
    (15, 29),
    (29, 31),
    # Spine3 to collars
    (3, 16),
    (3, 22),
    # Left arm
    (16, 17),  # collar → shoulder
    (17, 18),  # shoulder → elbow
    (18, 19),  # elbow → wrist
    (19, 20),  # wrist → hand
    (20, 21),  # hand → fingertip
    # Right arm
    (22, 23),
    (23, 24),
    (24, 25),
    (25, 26),
    (26, 27),
]

NUM_EDGES: int = len(SKELETON_EDGES)  # 31 unique edges → 62 directed


def build_edge_index() -> torch.Tensor:
    """
    Build a bidirectional edge index tensor for the ANUBIS skeleton.

    Returns
    -------
    edge_index : torch.Tensor  shape (2, 2 * NUM_EDGES)
        Directed edge index compatible with PyTorch Geometric.
    """
    src, dst = zip(*SKELETON_EDGES)
    edge_src = list(src) + list(dst)  # forward + reverse
    edge_dst = list(dst) + list(src)
    return torch.tensor([edge_src, edge_dst], dtype=torch.long)


# Pre-built edge index — computed once at import time, CPU tensor.
# The model registers this as a buffer so it moves to GPU automatically.
EDGE_INDEX: torch.Tensor = build_edge_index()  # shape (2, 62)


# ── Left / right mirror pairs ─────────────────────────────────────────────────
# Used by the horizontal-flip augmentation: swap these joint indices AND
# negate the x-coordinate (feature index 0) + vx (index 3) + ax (index 6).

MIRROR_JOINT_PAIRS: List[Tuple[int, int]] = [
    (6, 11),  # L_Hip        ↔ R_Hip
    (7, 12),  # L_Knee       ↔ R_Knee
    (8, 13),  # L_Ankle      ↔ R_Ankle
    (9, 14),  # L_Heel       ↔ R_Heel
    (10, 15),  # L_FootIndex  ↔ R_FootIndex
    (16, 22),  # L_Collar     ↔ R_Collar
    (17, 23),  # L_Shoulder   ↔ R_Shoulder
    (18, 24),  # L_Elbow      ↔ R_Elbow
    (19, 25),  # L_Wrist      ↔ R_Wrist
    (20, 26),  # L_Hand       ↔ R_Hand
    (21, 27),  # L_Fingertip  ↔ R_Fingertip
    (28, 29),  # L_Toes       ↔ R_Toes
    (30, 31),  # L_ThumbToe   ↔ R_ThumbToe
]

# Feature indices that flip sign when the skeleton is mirrored left-right.
# For feature layout [x, y, z, vx, vy, vz, ax, ay, az]: x→0, vx→3, ax→6.
FLIP_FEATURE_INDICES: List[int] = [0, 3, 6]


# ── Biomechanical angle triplets ──────────────────────────────────────────────
# (joint_idx, proximal_neighbour, distal_neighbour)
# Angle at joint_idx is the angle formed by the vectors:
#   joint → proximal   and   joint → distal
# Used for diagnostic visualisation and optional feature engineering.

ANGLE_TRIPLETS: Dict[int, Tuple[int, int]] = {
    7: (6, 8),  # L_Knee:   L_Hip  → L_Knee  → L_Ankle
    12: (11, 13),  # R_Knee:   R_Hip  → R_Knee  → R_Ankle
    6: (0, 7),  # L_Hip:    Pelvis → L_Hip   → L_Knee
    11: (0, 12),  # R_Hip:    Pelvis → R_Hip   → R_Knee
    8: (7, 9),  # L_Ankle:  L_Knee → L_Ankle → L_Heel
    13: (12, 14),  # R_Ankle:  R_Knee → R_Ankle → R_Heel
    18: (17, 19),  # L_Elbow:  L_Shoulder → L_Elbow → L_Wrist
    24: (23, 25),  # R_Elbow:  R_Shoulder → R_Elbow → R_Wrist
    1: (0, 2),  # Spine1:   Pelvis → Spine1 → Spine2  (trunk flexion)
}


# ── Body-region groupings (for fine-tuning injury-localisation head) ──────────
# Maps region name → list of joint indices most representative of that region.
# The transformer's CLS embedding is supervised against these labels in Stage 3.

BODY_REGIONS: Dict[str, List[int]] = {
    "left_hip": [6, 7],  # L_Hip + L_Knee (proximal hamstring)
    "right_hip": [11, 12],  # R_Hip + R_Knee
    "left_hamstring": [6, 7, 8],  # hip–knee–ankle chain
    "right_hamstring": [11, 12, 13],
    "left_knee": [7, 8],  # L_Knee + L_Ankle
    "right_knee": [12, 13],
    "left_ankle": [8, 9, 10],  # L_Ankle + L_Heel + L_FootIndex
    "right_ankle": [13, 14, 15],
    "spine": [0, 1, 2, 3],
    "left_shoulder": [16, 17],
    "right_shoulder": [22, 23],
}

REGION_NAMES: List[str] = list(BODY_REGIONS.keys())
NUM_REGIONS: int = len(REGION_NAMES)  # 11


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    ei = build_edge_index()
    assert ei.shape == (2, 2 * NUM_EDGES), f"Unexpected edge_index shape: {ei.shape}"
    assert len(JOINT_NAMES) == NUM_JOINTS == 32

    # Every edge node index must be in [0, NUM_JOINTS)
    assert ei.max().item() < NUM_JOINTS
    assert ei.min().item() >= 0

    # Every mirror pair must reference valid joints
    for l, r in MIRROR_JOINT_PAIRS:
        assert 0 <= l < NUM_JOINTS and 0 <= r < NUM_JOINTS

    print(f"NUM_JOINTS  : {NUM_JOINTS}")
    print(f"NUM_EDGES   : {NUM_EDGES} unique ({2 * NUM_EDGES} directed)")
    print(f"NUM_REGIONS : {NUM_REGIONS}")
    print("joint_config OK.")
