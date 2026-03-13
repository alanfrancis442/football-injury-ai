# module1/pretrain/live_utils.py
#
# Stage 1 — Live Inference Utilities
# ───────────────────────────────────
# Shared helpers for the live ANUBIS action pipeline: source parsing, feature
# construction, label-map loading, probability smoothing, motion gating, and
# stable prediction hysteresis for pause/background handling.

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import yaml


BACKGROUND_LABEL = "background / pause"
BACKGROUND_INDEX = -1


@dataclass
class DecodedPrediction:
    """Decoded live prediction before hysteresis is applied."""

    label: str
    index: int
    score: float
    margin: float
    motion_energy: float
    is_background: bool
    reason: str


@dataclass
class PredictionTracker:
    """State holder for stable live prediction display."""

    displayed_label: str = BACKGROUND_LABEL
    displayed_index: int = BACKGROUND_INDEX
    displayed_score: float = 0.0
    candidate_label: str = BACKGROUND_LABEL
    candidate_index: int = BACKGROUND_INDEX
    candidate_score: float = 0.0
    streak: int = 0
    reason: str = "warming_up"


def parse_video_source(source: str) -> Union[int, str]:
    """Convert a CLI source string into OpenCV's expected type."""
    stripped = source.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def normalise_pelvis_center(
    positions: np.ndarray,
    pelvis_idx: int = 0,
) -> np.ndarray:
    """Pelvis-centre a window of joint coordinates."""
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(
            f"Expected positions shape (T, J, 3), got {tuple(positions.shape)}"
        )
    if not (0 <= pelvis_idx < positions.shape[1]):
        raise ValueError(
            f"Invalid pelvis_idx={pelvis_idx} for J={positions.shape[1]} joints."
        )

    centered = positions.astype(np.float32, copy=True)
    centered -= centered[:, pelvis_idx : pelvis_idx + 1, :]
    return centered


def compute_velocity(positions: np.ndarray) -> np.ndarray:
    """Compute forward finite-difference velocity along time."""
    vel = np.zeros_like(positions, dtype=np.float32)
    if positions.shape[0] > 1:
        vel[:-1] = positions[1:] - positions[:-1]
        vel[-1] = vel[-2]
    return vel


def compute_acceleration(velocity: np.ndarray) -> np.ndarray:
    """Compute forward finite-difference acceleration along time."""
    acc = np.zeros_like(velocity, dtype=np.float32)
    if velocity.shape[0] > 1:
        acc[:-1] = velocity[1:] - velocity[:-1]
        acc[-1] = acc[-2]
    return acc


def build_feature_window(
    positions: np.ndarray,
    pelvis_idx: int = 0,
) -> torch.Tensor:
    """Build [pos, vel, acc] features with shape (T, J, 9)."""
    centered = normalise_pelvis_center(positions, pelvis_idx=pelvis_idx)
    vel = compute_velocity(centered)
    acc = compute_acceleration(vel)
    feat = np.concatenate([centered, vel, acc], axis=2).astype(np.float32, copy=False)
    return torch.from_numpy(feat)


def resolve_default_label_map_path(num_classes: int) -> Optional[str]:
    """Resolve the built-in ANUBIS label map when available."""
    if num_classes != 102:
        return None

    default_map = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "anubis_class_names.yaml",
        )
    )
    return default_map if os.path.exists(default_map) else None


def _from_mapping_dict(mapping: Dict, num_classes: int) -> List[str]:
    names = [f"class_{i}" for i in range(num_classes)]
    for key, value in mapping.items():
        idx = int(key)
        if 0 <= idx < num_classes:
            names[idx] = str(value)
    return names


def load_class_names(
    num_classes: int,
    label_map_path: Optional[str] = None,
) -> List[str]:
    """Load class names from a text, YAML, or JSON file."""
    fallback = [f"class_{i}" for i in range(num_classes)]
    if label_map_path is None:
        label_map_path = resolve_default_label_map_path(num_classes)
        if label_map_path is None:
            return fallback

    if not os.path.exists(label_map_path):
        raise FileNotFoundError(
            f"Label map not found: {label_map_path}\n"
            "Provide a valid --label-map path or omit this flag."
        )

    ext = os.path.splitext(label_map_path)[1].lower()

    if ext == ".txt":
        with open(label_map_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        names = [line for line in lines if line]
        merged = fallback[:]
        limit = min(num_classes, len(names))
        merged[:limit] = names[:limit]
        return merged

    if ext in (".yaml", ".yml"):
        with open(label_map_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
    elif ext == ".json":
        with open(label_map_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        raise ValueError(
            f"Unsupported label-map extension: {ext}. Use .txt, .yaml/.yml, or .json."
        )

    if isinstance(payload, list):
        merged = fallback[:]
        limit = min(num_classes, len(payload))
        for idx in range(limit):
            merged[idx] = str(payload[idx])
        return merged

    if isinstance(payload, dict):
        return _from_mapping_dict(payload, num_classes)

    raise ValueError(
        "Label map must be a list or dict in YAML/JSON format, "
        f"got type={type(payload).__name__}."
    )


def smooth_probs(
    previous: Optional[torch.Tensor],
    current: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Apply exponential moving-average smoothing to class probabilities."""
    clipped_alpha = float(min(max(alpha, 0.0), 0.999))
    if previous is None:
        return current
    return (previous * clipped_alpha) + (current * (1.0 - clipped_alpha))


def compute_motion_energy(positions: np.ndarray, pelvis_idx: int = 0) -> float:
    """Estimate how much the current window is moving."""
    centered = normalise_pelvis_center(positions, pelvis_idx=pelvis_idx)
    velocity = compute_velocity(centered)
    return float(np.linalg.norm(velocity, axis=2).mean())


def decode_prediction(
    probs: torch.Tensor,
    class_names: List[str],
    motion_energy: float,
    min_confidence: float,
    min_margin: float,
    min_motion_energy: float,
    background_label: str = BACKGROUND_LABEL,
) -> DecodedPrediction:
    """Map raw probabilities to a gated live prediction."""
    top_values, top_indices = torch.topk(probs, k=min(2, probs.numel()))
    top1_score = float(top_values[0].item())
    top1_index = int(top_indices[0].item())
    top1_label = (
        class_names[top1_index]
        if 0 <= top1_index < len(class_names)
        else f"class_{top1_index}"
    )
    top2_score = float(top_values[1].item()) if top_values.numel() > 1 else 0.0
    margin = top1_score - top2_score

    if motion_energy < min_motion_energy:
        return DecodedPrediction(
            label=background_label,
            index=BACKGROUND_INDEX,
            score=top1_score,
            margin=margin,
            motion_energy=motion_energy,
            is_background=True,
            reason="low_motion",
        )

    if top1_score < min_confidence:
        return DecodedPrediction(
            label=background_label,
            index=BACKGROUND_INDEX,
            score=top1_score,
            margin=margin,
            motion_energy=motion_energy,
            is_background=True,
            reason="low_confidence",
        )

    if margin < min_margin:
        return DecodedPrediction(
            label=background_label,
            index=BACKGROUND_INDEX,
            score=top1_score,
            margin=margin,
            motion_energy=motion_energy,
            is_background=True,
            reason="low_margin",
        )

    return DecodedPrediction(
        label=top1_label,
        index=top1_index,
        score=top1_score,
        margin=margin,
        motion_energy=motion_energy,
        is_background=False,
        reason="accepted",
    )


def update_prediction_tracker(
    tracker: PredictionTracker,
    decoded: DecodedPrediction,
    hysteresis_frames: int,
) -> PredictionTracker:
    """Update the stable displayed label using simple hysteresis."""
    if decoded.is_background:
        tracker.displayed_label = decoded.label
        tracker.displayed_index = decoded.index
        tracker.displayed_score = decoded.score
        tracker.candidate_label = decoded.label
        tracker.candidate_index = decoded.index
        tracker.candidate_score = decoded.score
        tracker.streak = 0
        tracker.reason = decoded.reason
        return tracker

    if decoded.label == tracker.displayed_label:
        tracker.displayed_index = decoded.index
        tracker.displayed_score = decoded.score
        tracker.candidate_label = decoded.label
        tracker.candidate_index = decoded.index
        tracker.candidate_score = decoded.score
        tracker.streak = 0
        tracker.reason = decoded.reason
        return tracker

    if decoded.label != tracker.candidate_label:
        tracker.candidate_label = decoded.label
        tracker.candidate_index = decoded.index
        tracker.candidate_score = decoded.score
        tracker.streak = 1
        tracker.reason = f"candidate:{decoded.reason}"
        return tracker

    tracker.streak += 1
    tracker.candidate_score = decoded.score
    tracker.reason = f"candidate:{decoded.reason}"

    if tracker.streak >= max(1, hysteresis_frames):
        tracker.displayed_label = decoded.label
        tracker.displayed_index = decoded.index
        tracker.displayed_score = decoded.score
        tracker.reason = decoded.reason
        tracker.streak = 0

    return tracker


if __name__ == "__main__":
    demo = np.random.randn(60, 32, 3).astype(np.float32)
    feat = build_feature_window(demo)
    assert feat.shape == (60, 32, 9)
    energy = compute_motion_energy(demo)
    assert energy >= 0.0
    print("live_utils OK.")
