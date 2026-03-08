# module1/pretrain/live_utils.py
#
# Stage 1 - Live Inference Utilities
# ----------------------------------
# Helper functions shared by the live ANUBIS inference pipeline:
#   - class-label loading
#   - pelvis-centred normalisation
#   - velocity / acceleration feature construction
#   - probability smoothing

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import yaml


def parse_video_source(source: str) -> Union[int, str]:
    """
    Convert a CLI source value to OpenCV's expected type.

    Parameters
    ----------
    source : str
        Webcam index (e.g. "0") or path/URL.

    Returns
    -------
    parsed_source : int | str
        Integer webcam index if input is numeric, otherwise the original string.
    """
    stripped = source.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def normalise_pelvis_center(
    positions: np.ndarray,
    pelvis_idx: int = 0,
) -> np.ndarray:
    """
    Pelvis-centre a window of joint coordinates.

    Parameters
    ----------
    positions : np.ndarray  shape (T, J, 3)
        Raw joint coordinates in normalised image space.
    pelvis_idx : int
        Pelvis joint index in the skeleton definition.

    Returns
    -------
    centered : np.ndarray  shape (T, J, 3)
        Pelvis-centred coordinates.
    """
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
    """
    Compute velocity via forward finite difference along time.

    Boundary handling follows the ANUBIS preprocessing style:
    last velocity frame duplicates the previous one.

    Parameters
    ----------
    positions : np.ndarray  shape (T, J, 3)

    Returns
    -------
    velocity : np.ndarray  shape (T, J, 3)
    """
    vel = np.zeros_like(positions, dtype=np.float32)
    if positions.shape[0] > 1:
        vel[:-1] = positions[1:] - positions[:-1]
        vel[-1] = vel[-2]
    return vel


def compute_acceleration(velocity: np.ndarray) -> np.ndarray:
    """
    Compute acceleration via forward finite difference of velocity.

    Boundary handling follows the ANUBIS preprocessing style:
    last acceleration frame duplicates the previous one.

    Parameters
    ----------
    velocity : np.ndarray  shape (T, J, 3)

    Returns
    -------
    acceleration : np.ndarray  shape (T, J, 3)
    """
    acc = np.zeros_like(velocity, dtype=np.float32)
    if velocity.shape[0] > 1:
        acc[:-1] = velocity[1:] - velocity[:-1]
        acc[-1] = acc[-2]
    return acc


def build_feature_window(
    positions: np.ndarray,
    pelvis_idx: int = 0,
) -> torch.Tensor:
    """
    Build model-ready per-joint features from a position window.

    Output feature order per joint matches training:
      [x, y, z, vx, vy, vz, ax, ay, az]

    Parameters
    ----------
    positions : np.ndarray  shape (T, J, 3)
        Raw 3D joint coordinates for one temporal window.
    pelvis_idx : int
        Pelvis index for centring.

    Returns
    -------
    features : torch.Tensor  shape (T, J, 9)
        Float32 tensor ready to be batched as (B, T, J, 9).
    """
    centered = normalise_pelvis_center(positions, pelvis_idx=pelvis_idx)
    vel = compute_velocity(centered)
    acc = compute_acceleration(vel)
    feat = np.concatenate([centered, vel, acc], axis=2).astype(np.float32, copy=False)
    return torch.from_numpy(feat)


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
    """
    Load class names for display from text / YAML / JSON mapping.

    Supported formats
    -----------------
    - .txt  : one class name per line
    - .yaml : list of names OR dict {class_id: class_name}
    - .json : list of names OR dict {class_id: class_name}

    Parameters
    ----------
    num_classes : int
        Number of model output classes.
    label_map_path : str, optional
        Optional file path containing class labels.

    Returns
    -------
    class_names : list[str]
        Length == num_classes.
    """
    fallback = [f"class_{i}" for i in range(num_classes)]
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
    """
    Exponential moving average over class probabilities.

    Parameters
    ----------
    previous : torch.Tensor, optional  shape (C,)
        Previous smoothed probability vector.
    current : torch.Tensor  shape (C,)
        Current probability vector.
    alpha : float
        EMA factor in [0, 1). Higher values produce smoother but slower output.

    Returns
    -------
    smoothed : torch.Tensor  shape (C,)
    """
    clipped_alpha = float(min(max(alpha, 0.0), 0.999))
    if previous is None:
        return current
    return (previous * clipped_alpha) + (current * (1.0 - clipped_alpha))


if __name__ == "__main__":
    demo = np.random.randn(60, 32, 3).astype(np.float32)
    feat = build_feature_window(demo)
    assert feat.shape == (60, 32, 9)
    print("live_utils OK.")
