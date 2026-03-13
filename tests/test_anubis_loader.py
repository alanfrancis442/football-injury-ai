# tests/test_anubis_loader.py
#
# Unit tests for module1/data/anubis_loader.py and module1/data/preprocess.py
# ─────────────────────────────────────────────────────────────────────────────
# Tests preprocessing functions (compute_acceleration, random/center_crop,
# flip_sequence, joint_jitter, time_mask, joint_dropout, speed_perturb) and the ANUBISDataset /
# load_anubis_tensors / build_dataloaders pipeline using in-memory dummy data.
#
# Run with: pytest tests/test_anubis_loader.py -v

from __future__ import annotations

import os
import tempfile
from typing import Iterator, Tuple

import numpy as np
import pytest
import torch

from module1.data.anubis_loader import ANUBISDataset, load_anubis_tensors
from module1.data.joint_config import FLIP_FEATURE_INDICES, MIRROR_JOINT_PAIRS
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


# ── Constants ─────────────────────────────────────────────────────────────────

N = 20  # samples
T = 60  # frames
J = 32  # joints
C = 6  # raw channels (pos + vel)
F = 9  # full channels (pos + vel + acc)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def raw_batch() -> torch.Tensor:
    """Random raw features: (N, 6, T, J)."""
    torch.manual_seed(42)
    return torch.randn(N, C, T, J)


@pytest.fixture
def sample() -> torch.Tensor:
    """Single sample after acc computation, transposed: (T, J, F)."""
    torch.manual_seed(0)
    raw = torch.randn(1, C, T, J)
    feat9 = compute_acceleration(raw)  # (1, 9, T, J)
    return feat9[0].permute(1, 2, 0).clone()  # (T, J, F)


@pytest.fixture
def tensor_pair() -> Tuple[torch.Tensor, torch.Tensor]:
    """(features (N, T, J, F), labels (N,)) as returned by load_anubis_tensors."""
    torch.manual_seed(7)
    raw = torch.randn(N, C, T, J)
    feat9 = compute_acceleration(raw)
    features = feat9.permute(0, 2, 3, 1).contiguous()  # (N, T, J, F)
    labels = torch.randint(0, 102, (N,), dtype=torch.long)
    return features, labels


@pytest.fixture
def tmp_npy_files() -> Iterator[Tuple[str, str, str]]:
    """Write dummy .npy files to a temp dir; yield (dir, feat_path, lbl_path)."""
    np.random.seed(1)
    raw = np.random.randn(N, C, T, J).astype(np.float32)
    lbl = np.random.randint(0, 102, size=(N,)).astype(np.int64)
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = os.path.join(tmpdir, "features.npy")
        lp = os.path.join(tmpdir, "labels.npy")
        np.save(fp, raw)
        np.save(lp, lbl)
        yield tmpdir, fp, lp


# ── compute_acceleration ──────────────────────────────────────────────────────


class TestComputeAcceleration:
    def test_output_shape(self, raw_batch: torch.Tensor) -> None:
        out = compute_acceleration(raw_batch)
        assert out.shape == (N, 9, T, J)

    def test_last_acc_frame_duplicates_previous(self, raw_batch: torch.Tensor) -> None:
        out = compute_acceleration(raw_batch)
        acc = out[:, 6:9, :, :]
        assert torch.allclose(acc[:, :, -1, :], acc[:, :, -2, :]), (
            "Last-frame acceleration must duplicate previous frame"
        )

    def test_acc_is_finite_difference_of_vel(self, raw_batch: torch.Tensor) -> None:
        out = compute_acceleration(raw_batch)
        vel = out[:, 3:6, :, :]
        acc = out[:, 6:9, :, :]
        expected_acc = vel[:, :, 1:, :] - vel[:, :, :-1, :]
        assert torch.allclose(acc[:, :, :-1, :], expected_acc)

    def test_pos_vel_channels_unchanged(self, raw_batch: torch.Tensor) -> None:
        out = compute_acceleration(raw_batch)
        assert torch.allclose(out[:, :6, :, :], raw_batch)

    def test_no_nan(self, raw_batch: torch.Tensor) -> None:
        out = compute_acceleration(raw_batch)
        assert not torch.isnan(out).any()


# ── random_crop / center_crop ─────────────────────────────────────────────────


class TestCroppingFunctions:
    def test_random_crop_output_shape(self, sample: torch.Tensor) -> None:
        cropped = random_crop(sample, seq_len=50)
        assert cropped.shape == (50, J, F)

    def test_center_crop_output_shape(self, sample: torch.Tensor) -> None:
        cropped = center_crop(sample, seq_len=50)
        assert cropped.shape == (50, J, F)

    def test_random_crop_no_crop_when_equal(self, sample: torch.Tensor) -> None:
        cropped = random_crop(sample, seq_len=T)
        assert cropped.shape == sample.shape

    def test_center_crop_no_crop_when_equal(self, sample: torch.Tensor) -> None:
        cropped = center_crop(sample, seq_len=T)
        assert cropped.shape == sample.shape

    def test_center_crop_is_centred(self, sample: torch.Tensor) -> None:
        seq_len = 40
        cropped = center_crop(sample, seq_len=seq_len)
        start = (T - seq_len) // 2
        assert torch.allclose(cropped, sample[start : start + seq_len])

    def test_random_crop_content_is_contiguous_window(
        self, sample: torch.Tensor
    ) -> None:
        seq_len = 30
        # Run several times — result must always be a contiguous window from sample
        for seed in range(5):
            torch.manual_seed(seed)
            cropped = random_crop(sample, seq_len=seq_len)
            found = False
            for start in range(T - seq_len + 1):
                if torch.allclose(sample[start : start + seq_len], cropped):
                    found = True
                    break
            assert found, f"random_crop result (seed={seed}) is not a contiguous window"


# ── flip_sequence ─────────────────────────────────────────────────────────────


class TestFlipSequence:
    def test_output_shape(self, sample: torch.Tensor) -> None:
        flipped = flip_sequence(sample, MIRROR_JOINT_PAIRS, FLIP_FEATURE_INDICES)
        assert flipped.shape == sample.shape

    def test_double_flip_is_identity(self, sample: torch.Tensor) -> None:
        flipped = flip_sequence(sample, MIRROR_JOINT_PAIRS, FLIP_FEATURE_INDICES)
        double = flip_sequence(flipped, MIRROR_JOINT_PAIRS, FLIP_FEATURE_INDICES)
        assert torch.allclose(sample, double), "Double-flip must restore original"

    def test_does_not_mutate_input(self, sample: torch.Tensor) -> None:
        original = sample.clone()
        flip_sequence(sample, MIRROR_JOINT_PAIRS, FLIP_FEATURE_INDICES)
        assert torch.allclose(sample, original), "flip_sequence must not modify input"

    def test_x_axis_features_negated(self, sample: torch.Tensor) -> None:
        # Use empty mirror pairs — only x-axis negation should happen
        flipped = flip_sequence(sample, [], FLIP_FEATURE_INDICES)
        for fi in FLIP_FEATURE_INDICES:
            assert torch.allclose(flipped[:, :, fi], -sample[:, :, fi]), (
                f"Feature index {fi} should be negated"
            )

    def test_non_flip_features_unchanged(self, sample: torch.Tensor) -> None:
        flipped = flip_sequence(sample, [], FLIP_FEATURE_INDICES)
        non_flip = [i for i in range(F) if i not in FLIP_FEATURE_INDICES]
        for fi in non_flip:
            assert torch.allclose(flipped[:, :, fi], sample[:, :, fi]), (
                f"Feature index {fi} should be unchanged after flip"
            )


# ── joint_jitter ──────────────────────────────────────────────────────────────


class TestJointJitter:
    def test_output_shape(self, sample: torch.Tensor) -> None:
        noisy = joint_jitter(sample, std=0.01)
        assert noisy.shape == sample.shape

    def test_adds_noise(self, sample: torch.Tensor) -> None:
        torch.manual_seed(99)
        noisy = joint_jitter(sample, std=0.01)
        assert not torch.allclose(sample, noisy)

    def test_zero_std_is_identity(self, sample: torch.Tensor) -> None:
        noisy = joint_jitter(sample, std=0.0)
        assert torch.allclose(sample, noisy)

    def test_does_not_mutate_input(self, sample: torch.Tensor) -> None:
        original = sample.clone()
        joint_jitter(sample, std=0.1)
        assert torch.allclose(sample, original)


class TestTimeMask:
    def test_output_shape(self, sample: torch.Tensor) -> None:
        masked = time_mask(sample, max_mask_frames=6)
        assert masked.shape == sample.shape

    def test_zero_frames_is_identity(self, sample: torch.Tensor) -> None:
        masked = time_mask(sample, max_mask_frames=0)
        assert torch.allclose(masked, sample)


class TestJointDropout:
    def test_output_shape(self, sample: torch.Tensor) -> None:
        dropped = joint_dropout(sample, drop_prob=0.2)
        assert dropped.shape == sample.shape

    def test_zero_prob_is_identity(self, sample: torch.Tensor) -> None:
        dropped = joint_dropout(sample, drop_prob=0.0)
        assert torch.allclose(dropped, sample)

    def test_some_joint_is_zeroed(self, sample: torch.Tensor) -> None:
        torch.manual_seed(0)
        dropped = joint_dropout(sample, drop_prob=0.9)
        zeroed = (dropped.abs().sum(dim=(0, 2)) == 0).any().item()
        assert zeroed


# ── speed_perturb ─────────────────────────────────────────────────────────────


class TestSpeedPerturb:
    def test_output_shape(self, sample: torch.Tensor) -> None:
        out = speed_perturb(sample, factor_range=(0.8, 1.2), target_len=T)
        assert out.shape == (T, J, F)

    def test_output_shape_different_target_len(self, sample: torch.Tensor) -> None:
        out = speed_perturb(sample, factor_range=(0.8, 1.2), target_len=50)
        assert out.shape == (50, J, F)

    def test_no_nan(self, sample: torch.Tensor) -> None:
        out = speed_perturb(sample, factor_range=(0.8, 1.2), target_len=T)
        assert not torch.isnan(out).any()

    def test_factor_1_is_near_identity(self, sample: torch.Tensor) -> None:
        # With factor fixed at exactly 1.0 the output should be very close
        out = speed_perturb(sample, factor_range=(1.0, 1.0), target_len=T)
        assert out.shape == (T, J, F)
        # Values may differ slightly due to linear interpolation rounding
        assert torch.allclose(out, sample, atol=1e-4)


# ── load_anubis_tensors ───────────────────────────────────────────────────────


class TestLoadAnubisTensors:
    def test_output_shapes(self, tmp_npy_files: Tuple[str, str, str]) -> None:
        _, fp, lp = tmp_npy_files
        feats, labels = load_anubis_tensors(fp, lp)
        assert feats.shape == (N, T, J, 9)
        assert labels.shape == (N,)

    def test_features_dtype(self, tmp_npy_files: Tuple[str, str, str]) -> None:
        _, fp, lp = tmp_npy_files
        feats, _ = load_anubis_tensors(fp, lp)
        assert feats.dtype == torch.float32

    def test_labels_dtype(self, tmp_npy_files: Tuple[str, str, str]) -> None:
        _, fp, lp = tmp_npy_files
        _, labels = load_anubis_tensors(fp, lp)
        assert labels.dtype == torch.int64

    def test_missing_features_raises(self, tmp_npy_files: Tuple[str, str, str]) -> None:
        _, _, lp = tmp_npy_files
        with pytest.raises(FileNotFoundError):
            load_anubis_tensors("/nonexistent/features.npy", lp)

    def test_missing_labels_raises(self, tmp_npy_files: Tuple[str, str, str]) -> None:
        _, fp, _ = tmp_npy_files
        with pytest.raises(FileNotFoundError):
            load_anubis_tensors(fp, "/nonexistent/labels.npy")

    def test_wrong_channel_count_raises(
        self, tmp_npy_files: Tuple[str, str, str]
    ) -> None:
        tmpdir, _, lp = tmp_npy_files
        bad = np.random.randn(N, 9, T, J).astype(np.float32)  # 9 channels, not 6
        bad_fp = os.path.join(tmpdir, "bad_features.npy")
        np.save(bad_fp, bad)
        with pytest.raises(ValueError, match="6 channels"):
            load_anubis_tensors(bad_fp, lp)

    def test_wrong_ndim_raises(self, tmp_npy_files: Tuple[str, str, str]) -> None:
        tmpdir, _, lp = tmp_npy_files
        bad = np.random.randn(N, T * J * C).astype(np.float32)  # 2D
        bad_fp = os.path.join(tmpdir, "bad_flat.npy")
        np.save(bad_fp, bad)
        with pytest.raises(ValueError, match="4 dimensions"):
            load_anubis_tensors(bad_fp, lp)

    def test_5d_input_selects_primary_person(
        self, tmp_npy_files: Tuple[str, str, str]
    ) -> None:
        tmpdir, _, lp = tmp_npy_files
        m = 2
        # Build deterministic data where person-0 is zeros and person-1 is ones.
        # Loader should select person-0 and thus return all zeros before acc concat.
        raw5d = np.ones((N, C, T, J, m), dtype=np.float32)
        raw5d[..., 0] = 0.0
        fp_5d = os.path.join(tmpdir, "features_5d.npy")
        np.save(fp_5d, raw5d)

        feats, _ = load_anubis_tensors(fp_5d, lp)
        # All channels should be zeros because person-0 contains zeros only.
        assert torch.allclose(feats, torch.zeros_like(feats))


# ── ANUBISDataset ─────────────────────────────────────────────────────────────


class TestANUBISDataset:
    def test_len(self, tensor_pair: Tuple[torch.Tensor, torch.Tensor]) -> None:
        features, labels = tensor_pair
        ds = ANUBISDataset(features, labels, seq_len=T)
        assert len(ds) == N

    def test_item_shapes_no_crop(
        self, tensor_pair: Tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        features, labels = tensor_pair
        ds = ANUBISDataset(features, labels, seq_len=T, augment=False)
        x, y = ds[0]
        assert x.shape == (T, J, F)
        assert y.shape == ()

    def test_item_shapes_with_crop(
        self, tensor_pair: Tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        features, labels = tensor_pair
        ds = ANUBISDataset(features, labels, seq_len=50, augment=False)
        x, y = ds[0]
        assert x.shape == (50, J, F), f"Expected (50,{J},{F}), got {x.shape}"

    def test_label_dtype(self, tensor_pair: Tuple[torch.Tensor, torch.Tensor]) -> None:
        features, labels = tensor_pair
        ds = ANUBISDataset(features, labels, seq_len=T)
        _, y = ds[0]
        assert y.dtype == torch.int64

    def test_augmentation_changes_output(
        self, tensor_pair: Tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        features, labels = tensor_pair
        ds_aug = ANUBISDataset(
            features,
            labels,
            seq_len=T,
            augment=True,
            aug_cfg={"flip_prob": 1.0, "joint_jitter_std": 0.1, "speed_perturb": False},
        )
        ds_no_aug = ANUBISDataset(features, labels, seq_len=T, augment=False)
        # With flip_prob=1 and jitter the augmented output must differ from clean
        torch.manual_seed(0)
        x_aug, _ = ds_aug[0]
        x_clean, _ = ds_no_aug[0]
        assert not torch.allclose(x_aug, x_clean), (
            "Augmented sample must differ from clean sample"
        )

    def test_getitem_does_not_mutate_stored_tensor(
        self, tensor_pair: Tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        features, labels = tensor_pair
        original = features[0].clone()
        ds = ANUBISDataset(
            features,
            labels,
            seq_len=T,
            augment=True,
            aug_cfg={"flip_prob": 1.0, "joint_jitter_std": 0.5, "speed_perturb": False},
        )
        _ = ds[0]
        assert torch.allclose(features[0], original), (
            "__getitem__ must not mutate the stored feature tensor"
        )
