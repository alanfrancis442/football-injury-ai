# module1/data/preprocess.py
#
# Stage 1 — Data Preprocessing and Augmentation
# ───────────────────────────────────────────────
# Stateless functions that operate on individual samples.
# Input tensors have shape (T, J, F):
#   T = number of frames
#   J = number of joints (32)
#   F = number of features per joint (9: pos + vel + acc)
#
# Feature layout:   [x, y, z,  vx, vy, vz,  ax, ay, az]
#   indices:         0  1  2    3   4   5    6   7   8
#
# All functions are pure (no in-place mutation) and return new tensors.
# They are called inside ANUBISDataset.__getitem__ during training.

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ── Acceleration computation ───────────────────────────────────────────────────


def compute_acceleration(
    features: torch.Tensor,  # (N, 6, T, J) -> returns (N, 9, T, J)
) -> torch.Tensor:
    """
    Compute acceleration (finite difference of velocity) and append as channels.

    Called once when the dataset is loaded to augment the raw 6-channel data
    (pos + vel) to 9-channel data (pos + vel + acc).

    Parameters
    ----------
    features : torch.Tensor  shape (N, 6, T, J)
        Channels: [x, y, z, vx, vy, vz]

    Returns
    -------
    features_9 : torch.Tensor  shape (N, 9, T, J)
        Channels: [x, y, z, vx, vy, vz, ax, ay, az]
        Acceleration is computed with a forward finite difference along time,
        and the last frame is duplicated from the previous frame to match
        the ANUBIS preprocessing notebook's boundary handling style.
    """
    vel = features[:, 3:6, :, :]  # (N, 3, T, J)
    acc = torch.zeros_like(vel)  # (N, 3, T, J)
    acc[:, :, :-1, :] = vel[:, :, 1:, :] - vel[:, :, :-1, :]
    if vel.shape[2] > 1:
        acc[:, :, -1, :] = acc[:, :, -2, :]
    return torch.cat([features, acc], dim=1)  # (N, 9, T, J)


# ── Window cropping ────────────────────────────────────────────────────────────


def random_crop(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Randomly crop a contiguous window of ``seq_len`` frames from ``x``.

    Parameters
    ----------
    x       : torch.Tensor  shape (T, J, F)
    seq_len : int           must be <= T

    Returns
    -------
    cropped : torch.Tensor  shape (seq_len, J, F)
    """
    T = x.shape[0]
    if seq_len >= T:
        return x
    start = torch.randint(0, T - seq_len + 1, (1,)).item()
    return x[start : start + seq_len]


def center_crop(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Take a centred window of ``seq_len`` frames from ``x`` (used at validation).

    Parameters
    ----------
    x       : torch.Tensor  shape (T, J, F)
    seq_len : int

    Returns
    -------
    cropped : torch.Tensor  shape (seq_len, J, F)
    """
    T = x.shape[0]
    if seq_len >= T:
        return x
    start = (T - seq_len) // 2
    return x[start : start + seq_len]


# ── Augmentation ───────────────────────────────────────────────────────────────


def flip_sequence(
    x: torch.Tensor,  # (T, J, F)
    mirror_pairs: List[Tuple[int, int]],
    flip_feature_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Mirror the skeleton left-right by swapping paired joints and negating the
    x-axis features (x, vx, ax at indices 0, 3, 6 for F=9).

    Parameters
    ----------
    x                   : torch.Tensor  shape (T, J, F)
    mirror_pairs        : list of (left_joint_idx, right_joint_idx)
    flip_feature_indices: feature dimensions to negate; defaults to [0, 3, 6]

    Returns
    -------
    flipped : torch.Tensor  shape (T, J, F)
    """
    if flip_feature_indices is None:
        flip_feature_indices = [0, 3, 6]

    x = x.clone()

    # Swap joint pairs
    for l_idx, r_idx in mirror_pairs:
        left = x[:, l_idx, :].clone()
        right = x[:, r_idx, :].clone()
        x[:, l_idx, :] = right
        x[:, r_idx, :] = left

    # Negate lateral (x-axis) features
    x[:, :, flip_feature_indices] = -x[:, :, flip_feature_indices]

    return x


def joint_jitter(x: torch.Tensor, std: float) -> torch.Tensor:
    """
    Add zero-mean Gaussian noise to all joint features.

    Parameters
    ----------
    x   : torch.Tensor  shape (T, J, F)
    std : float         noise standard deviation

    Returns
    -------
    noisy : torch.Tensor  shape (T, J, F)
    """
    return x + torch.randn_like(x) * std


def time_mask(x: torch.Tensor, max_mask_frames: int) -> torch.Tensor:
    """
    Replace a short contiguous time span with a nearby frame.

    Parameters
    ----------
    x : torch.Tensor  shape (T, J, F)
    max_mask_frames : int
        Maximum number of consecutive frames to mask.

    Returns
    -------
    masked : torch.Tensor  shape (T, J, F)
    """
    if max_mask_frames <= 0 or x.shape[0] < 2:
        return x

    T = x.shape[0]
    span = int(torch.randint(1, min(max_mask_frames, T - 1) + 1, (1,)).item())
    start = int(torch.randint(0, T - span + 1, (1,)).item())

    masked = x.clone()
    source_idx = start - 1 if start > 0 else min(T - 1, start + span)
    masked[start : start + span] = masked[source_idx].unsqueeze(0).expand(span, -1, -1)
    return masked


def joint_dropout(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """
    Randomly zero out whole joints across the full temporal window.

    Parameters
    ----------
    x : torch.Tensor  shape (T, J, F)
    drop_prob : float
        Probability of dropping each joint.

    Returns
    -------
    dropped : torch.Tensor  shape (T, J, F)
    """
    clipped_prob = float(min(max(drop_prob, 0.0), 1.0))
    if clipped_prob <= 0.0:
        return x

    keep_mask = torch.rand(x.shape[1], device=x.device) >= clipped_prob
    if not torch.any(keep_mask):
        keep_mask[torch.randint(0, x.shape[1], (1,), device=x.device)] = True

    dropped = x.clone()
    dropped[:, ~keep_mask, :] = 0.0
    return dropped


def speed_perturb(
    x: torch.Tensor,  # (T, J, F)
    factor_range: Tuple[float, float],
    target_len: int,
) -> torch.Tensor:
    """
    Temporally stretch or compress a sequence by a random factor, then
    crop / pad back to ``target_len`` frames.

    Uses linear interpolation along the time axis.

    Parameters
    ----------
    x            : torch.Tensor  shape (T, J, F)
    factor_range : (min_factor, max_factor)  e.g. (0.8, 1.2)
    target_len   : output sequence length after the perturbation

    Returns
    -------
    perturbed : torch.Tensor  shape (target_len, J, F)
    """
    T, J, Fin = x.shape
    lo, hi = factor_range
    factor = lo + torch.rand(1).item() * (hi - lo)
    new_T = max(2, int(round(T * factor)))  # must be at least 2 for interpolate

    # F.interpolate expects (N, C, L) — treat joints×features as channels
    x_t = x.permute(1, 2, 0).reshape(1, J * Fin, T).float()
    x_resampled = F.interpolate(x_t, size=new_T, mode="linear", align_corners=False)
    x_resampled = x_resampled.reshape(J, Fin, new_T).permute(2, 0, 1)  # (new_T, J, F)

    # Crop or zero-pad to target_len
    if new_T >= target_len:
        start = torch.randint(0, new_T - target_len + 1, (1,)).item()
        x_resampled = x_resampled[start : start + target_len]
    else:
        pad_len = target_len - new_T
        pad = x_resampled[-1:].expand(pad_len, -1, -1)
        x_resampled = torch.cat([x_resampled, pad], dim=0)

    return x_resampled


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from module1.data.joint_config import MIRROR_JOINT_PAIRS, FLIP_FEATURE_INDICES

    T, J, F_IN = 60, 32, 6
    N = 8
    dummy_raw = torch.randn(N, F_IN, T, J)

    # Acceleration
    feat9 = compute_acceleration(dummy_raw)
    assert feat9.shape == (N, 9, T, J), feat9.shape

    # One sample in (T, J, F) format
    sample = feat9[0].permute(1, 2, 0)  # (T, J, 9)
    assert sample.shape == (T, J, 9)

    # Random crop
    cropped = random_crop(sample, seq_len=50)
    assert cropped.shape == (50, J, 9)

    # Centre crop
    centred = center_crop(sample, seq_len=50)
    assert centred.shape == (50, J, 9)

    # Flip
    flipped = flip_sequence(sample, MIRROR_JOINT_PAIRS, FLIP_FEATURE_INDICES)
    assert flipped.shape == sample.shape
    # After double-flip we should get back the original
    double_flipped = flip_sequence(flipped, MIRROR_JOINT_PAIRS, FLIP_FEATURE_INDICES)
    assert torch.allclose(sample, double_flipped)

    # Jitter
    jittered = joint_jitter(sample, std=0.01)
    assert jittered.shape == sample.shape

    # Time mask
    masked = time_mask(sample, max_mask_frames=6)
    assert masked.shape == sample.shape

    # Joint dropout
    dropped = joint_dropout(sample, drop_prob=0.2)
    assert dropped.shape == sample.shape

    # Speed perturb
    perturbed = speed_perturb(sample, factor_range=(0.8, 1.2), target_len=60)
    assert perturbed.shape == (60, J, 9)

    print("preprocess OK.")
