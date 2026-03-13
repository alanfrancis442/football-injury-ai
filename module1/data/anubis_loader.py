# module1/data/anubis_loader.py
#
# Stage 1 — ANUBIS Dataset Loader
# ─────────────────────────────────
# Loads preprocessed ANUBIS skeleton data from .npy files and exposes it as a
# PyTorch Dataset / DataLoader pair ready for model training.
#
# Expected files on disk (configured in the Stage 1 experiment YAML):
#   trn_features_cleaned.npy  —  shape (N, 6, 60, 32)
#                                  OR (N, 6, 60, 32, M)
#                                  axes: (samples, channels, frames, joints[, persons])
#                                  channels: [x, y, z, vx, vy, vz]
#                                  coordinates are pelvis-centred.
#   trn_labels_cleaned.npy    —  shape (N,)  integer class index in [0, 101]
#
# The loader converts 6-channel data → 9-channel (pos + vel + acc), transposes
# to (N, T, J, F) format, and wraps in an ANUBISDataset that applies
# configurable augmentation during training.

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from module1.data.joint_config import (
    FLIP_FEATURE_INDICES,
    MIRROR_JOINT_PAIRS,
)
from module1.data.preprocess import (
    center_crop,
    compute_acceleration,
    flip_sequence,
    joint_dropout,
    joint_jitter,
    random_crop,
    speed_perturb,
    time_mask,
)
from utils.logger import get_logger

log = get_logger(__name__)


# ── Dataset ────────────────────────────────────────────────────────────────────


class ANUBISDataset(Dataset):
    """
    PyTorch Dataset wrapping preprocessed ANUBIS skeleton clips.

    Each item is a tuple ``(features, label)`` where:
      features : torch.Tensor  shape (seq_len, J, F)   float32
      label    : torch.Tensor  shape ()                 int64

    Parameters
    ----------
    features : torch.Tensor  shape (N, T, J, F)
        Pre-loaded and pre-processed feature tensor (pos + vel + acc).
    labels : torch.Tensor  shape (N,)
        Integer class indices in [0, num_classes).
    seq_len : int
        Number of frames per sample.  A random crop is applied during training
        and a centre crop during validation when seq_len < T.
    augment : bool
        Whether to apply data augmentation (training split only).
    aug_cfg : dict, optional
        Augmentation hyper-parameters.  Keys:
          flip_prob        (float, default 0.5)
          joint_jitter_std (float, default 0.01)
          speed_perturb      (bool,  default True)
          speed_range        (list,  default [0.8, 1.2])
          time_mask_frames   (int,   default 0)
          joint_drop_prob    (float, default 0.0)
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        seq_len: int,
        augment: bool = False,
        aug_cfg: Optional[Dict] = None,
    ) -> None:
        self.features = features  # (N, T, J, F)
        self.labels = labels  # (N,)
        self.seq_len = seq_len
        self.clip_len = features.shape[1]
        self.augment = augment
        self.aug_cfg = aug_cfg or {}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx].clone()  # (T, J, F)
        y = self.labels[idx]

        # ── Window crop ──────────────────────────────────────────────────────
        if self.seq_len < self.clip_len:
            x = (
                random_crop(x, self.seq_len)
                if self.augment
                else center_crop(x, self.seq_len)
            )

        # ── Speed perturbation (before other augmentations) ──────────────────
        if self.augment and self.aug_cfg.get("speed_perturb", True):
            sr = self.aug_cfg.get("speed_range", [0.8, 1.2])
            x = speed_perturb(x, factor_range=(sr[0], sr[1]), target_len=self.seq_len)

        # ── Horizontal flip ──────────────────────────────────────────────────
        if self.augment:
            prob = self.aug_cfg.get("flip_prob", 0.5)
            if torch.rand(1).item() < prob:
                x = flip_sequence(x, MIRROR_JOINT_PAIRS, FLIP_FEATURE_INDICES)

        # ── Joint jitter ─────────────────────────────────────────────────────
        if self.augment:
            std = self.aug_cfg.get("joint_jitter_std", 0.01)
            if std > 0:
                x = joint_jitter(x, std)

        # ── Time masking ───────────────────────────────────────────────────────
        if self.augment:
            max_mask_frames = int(self.aug_cfg.get("time_mask_frames", 0))
            if max_mask_frames > 0:
                x = time_mask(x, max_mask_frames=max_mask_frames)

        # ── Joint dropout ──────────────────────────────────────────────────────
        if self.augment:
            drop_prob = float(self.aug_cfg.get("joint_drop_prob", 0.0))
            if drop_prob > 0.0:
                x = joint_dropout(x, drop_prob=drop_prob)

        return x, y


# ── Data loading helpers ───────────────────────────────────────────────────────


def _load_npy(path: str, desc: str) -> np.ndarray:
    """Load a .npy file with a descriptive error if it does not exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{desc} not found: {path}\n"
            "Copy your preprocessed ANUBIS files to the anubis_dir specified in "
            "the experiment config, then run again."
        )
    arr = np.load(path)
    log.info("Loaded %s  shape=%s  dtype=%s", desc, arr.shape, arr.dtype)
    return arr


def load_anubis_tensors(
    features_path: str,
    labels_path: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load raw .npy files, compute acceleration, and return (features, labels).

    Parameters
    ----------
    features_path : str   path to .npy file of shape (N, 6, T, J)
                          or (N, 6, T, J, M)
    labels_path   : str   path to .npy file of shape (N,)

    Returns
    -------
    features : torch.Tensor  shape (N, T, J, 9)   float32
    labels   : torch.Tensor  shape (N,)            int64
    """
    raw = torch.from_numpy(_load_npy(features_path, "features").astype(np.float32))
    lbls = torch.from_numpy(_load_npy(labels_path, "labels").astype(np.int64))

    # Support both (N, C, T, J) and (N, C, T, J, M) layouts.
    # If multiple persons are present, follow the preprocessing notebook and
    # keep only the primary person at index 0.
    if raw.ndim == 5:
        num_persons = raw.shape[-1]
        if num_persons < 1:
            raise ValueError(
                "Expected person dimension M >= 1 in (N, C, T, J, M), got 0."
            )
        raw = raw[..., 0]
        log.info("Selected primary person index 0 from M=%d", num_persons)

    if raw.ndim != 4:
        raise ValueError(
            "Expected features to have 4 dimensions (N, C, T, J) or 5 dimensions "
            f"(N, C, T, J, M), got {raw.ndim}. Check that the file matches the "
            "format described in anubis_loader.py."
        )
    if raw.shape[1] != 6:
        raise ValueError(
            f"Expected 6 channels [x,y,z,vx,vy,vz], got {raw.shape[1]}. "
            "Acceleration will be computed here; do not pre-compute it."
        )

    # Compute acceleration: raw (N,6,T,J) → features (N,9,T,J)
    features_9 = compute_acceleration(raw)  # (N, 9, T, J)
    # Transpose to (N, T, J, 9) — time-first layout expected by the model
    features_9 = features_9.permute(0, 2, 3, 1).contiguous()

    log.info("Features after acc. computation: shape=%s", tuple(features_9.shape))
    return features_9, lbls


def build_dataloaders(
    cfg: Dict,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from config.

    Parameters
    ----------
    cfg : dict
        Full config dict loaded from the Stage 1 experiment YAML.
    num_workers : int
        DataLoader worker processes.
    pin_memory : bool
        Pin host memory for faster GPU transfer.

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    """
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})

    base_dir = data_cfg["anubis_dir"]
    seq_len = data_cfg["seq_len"]
    batch_size = cfg["training"]["batch_size"]
    val_split = data_cfg.get("val_split", 0.15)
    has_sep_val = data_cfg.get("has_separate_val", False)

    # ── Load training data ───────────────────────────────────────────────────
    trn_feat, trn_lbl = load_anubis_tensors(
        features_path=os.path.join(base_dir, data_cfg["train_features"]),
        labels_path=os.path.join(base_dir, data_cfg["train_labels"]),
    )

    # ── Validation split ─────────────────────────────────────────────────────
    if has_sep_val:
        val_feat, val_lbl = load_anubis_tensors(
            features_path=os.path.join(base_dir, data_cfg["val_features"]),
            labels_path=os.path.join(base_dir, data_cfg["val_labels"]),
        )
    else:
        # Random split of the training set
        N = len(trn_lbl)
        n_val = int(N * val_split)
        n_trn = N - n_val
        indices = torch.randperm(N)
        trn_idx, val_idx = indices[:n_trn], indices[n_trn:]
        val_feat = trn_feat[val_idx]
        val_lbl = trn_lbl[val_idx]
        trn_feat = trn_feat[trn_idx]
        trn_lbl = trn_lbl[trn_idx]
        log.info("Random val split: %d train / %d val", n_trn, n_val)

    # ── Datasets ─────────────────────────────────────────────────────────────
    aug_enabled = aug_cfg.get("enabled", True)
    train_ds = ANUBISDataset(
        trn_feat,
        trn_lbl,
        seq_len=seq_len,
        augment=aug_enabled,
        aug_cfg=aug_cfg,
    )
    val_ds = ANUBISDataset(
        val_feat,
        val_lbl,
        seq_len=seq_len,
        augment=False,
    )

    log.info("Train dataset: %d samples", len(train_ds))
    log.info("Val   dataset: %d samples", len(val_ds))

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    from module1.data.joint_config import NUM_JOINTS

    J, T, N = NUM_JOINTS, 60, 20
    # Create dummy raw data matching the .npy format
    raw = np.random.randn(N, 6, T, J).astype(np.float32)
    lbl = np.random.randint(0, 102, size=(N,)).astype(np.int64)

    with tempfile.TemporaryDirectory() as tmpdir:
        fp = os.path.join(tmpdir, "features.npy")
        lp = os.path.join(tmpdir, "labels.npy")
        np.save(fp, raw)
        np.save(lp, lbl)

        feats, labels = load_anubis_tensors(fp, lp)
        assert feats.shape == (N, T, J, 9), feats.shape
        assert labels.shape == (N,), labels.shape

        ds = ANUBISDataset(feats, labels, seq_len=50, augment=True)
        x, y = ds[0]
        assert x.shape == (50, J, 9), x.shape
        assert y.dtype == torch.int64

    print("anubis_loader OK.")
