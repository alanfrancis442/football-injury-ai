# tests/test_live_pipeline.py
#
# Unit tests for module1/pretrain/live_utils.py and module1/pose/extractor.py
# -----------------------------------------------------------------------------
# Covers: source parsing, feature-window construction (pos+vel+acc), label-map
# loading, EMA smoothing, and MediaPipe33->ANUBIS32 joint conversion.
#
# Run with: pytest tests/test_live_pipeline.py -v

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
import yaml

from module1.pose.extractor import mediapipe33_to_anubis32
from module1.pretrain.live_utils import (
    build_feature_window,
    compute_acceleration,
    compute_velocity,
    load_class_names,
    normalise_pelvis_center,
    parse_video_source,
    smooth_probs,
)


class TestParseVideoSource:
    def test_numeric_source_returns_int(self) -> None:
        assert parse_video_source("0") == 0
        assert parse_video_source("12") == 12

    def test_path_or_url_returns_str(self) -> None:
        assert parse_video_source("data/samples/clip.mp4") == "data/samples/clip.mp4"
        assert parse_video_source("rtsp://cam/live") == "rtsp://cam/live"


class TestFeatureConstruction:
    def test_normalise_pelvis_center(self) -> None:
        x = np.zeros((3, 4, 3), dtype=np.float32)
        x[:, 0, :] = np.array([2.0, -1.0, 0.5], dtype=np.float32)
        x[:, 1, :] = np.array([3.0, 2.0, 0.5], dtype=np.float32)

        out = normalise_pelvis_center(x, pelvis_idx=0)
        assert np.allclose(out[:, 0, :], 0.0)
        assert np.allclose(out[:, 1, :], np.array([1.0, 3.0, 0.0], dtype=np.float32))

    def test_velocity_and_acceleration_boundary_handling(self) -> None:
        pos = np.zeros((4, 1, 3), dtype=np.float32)
        pos[:, 0, 0] = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32)

        vel = compute_velocity(pos)
        acc = compute_acceleration(vel)

        # Forward differences on x channel
        assert np.allclose(
            vel[:, 0, 0], np.array([1.0, 2.0, 3.0, 3.0], dtype=np.float32)
        )
        assert np.allclose(
            acc[:, 0, 0], np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        )

    def test_build_feature_window_shape_and_order(self) -> None:
        T = 5
        J = 3
        pos = np.zeros((T, J, 3), dtype=np.float32)
        # Simple motion for joint 1 on x axis; pelvis is joint 0.
        pos[:, 0, :] = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        pos[:, 1, 0] = np.arange(T, dtype=np.float32)

        feat = build_feature_window(pos, pelvis_idx=0)
        assert feat.shape == (T, J, 9)

        # Channels 0:3 are centred positions. Pelvis must be zeros.
        assert torch.allclose(feat[:, 0, 0:3], torch.zeros((T, 3), dtype=torch.float32))
        # Channels 3:6 are velocity. Joint-1 x velocity is 1 except last duplicated.
        assert torch.allclose(
            feat[:, 1, 3], torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32)
        )


class TestClassNames:
    def test_default_fallback(self) -> None:
        names = load_class_names(num_classes=4, label_map_path=None)
        assert names == ["class_0", "class_1", "class_2", "class_3"]

    def test_load_from_txt(self, tmp_path: Path) -> None:
        p = tmp_path / "labels.txt"
        p.write_text("walk\nrun\njump\n", encoding="utf-8")
        names = load_class_names(num_classes=5, label_map_path=str(p))
        assert names[:3] == ["walk", "run", "jump"]
        assert names[4] == "class_4"

    def test_load_from_yaml_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "labels.yaml"
        payload = {0: "walk", 2: "kick"}
        p.write_text(yaml.safe_dump(payload), encoding="utf-8")
        names = load_class_names(num_classes=4, label_map_path=str(p))
        assert names == ["walk", "class_1", "kick", "class_3"]

    def test_load_from_json_list(self, tmp_path: Path) -> None:
        p = tmp_path / "labels.json"
        payload: List[str] = ["a", "b"]
        p.write_text(json.dumps(payload), encoding="utf-8")
        names = load_class_names(num_classes=3, label_map_path=str(p))
        assert names == ["a", "b", "class_2"]


class TestSmoothProbs:
    def test_first_step_returns_current(self) -> None:
        cur = torch.tensor([0.2, 0.8])
        out = smooth_probs(None, cur, alpha=0.7)
        assert torch.allclose(out, cur)

    def test_ema(self) -> None:
        prev = torch.tensor([0.5, 0.5])
        cur = torch.tensor([0.1, 0.9])
        out = smooth_probs(prev, cur, alpha=0.5)
        assert torch.allclose(out, torch.tensor([0.3, 0.7]))


class TestMediapipeConversion:
    def test_output_shape(self) -> None:
        mp = np.zeros((33, 4), dtype=np.float32)
        out = mediapipe33_to_anubis32(mp)
        assert out.shape == (32, 3)

    def test_pelvis_is_midpoint_of_hips(self) -> None:
        mp = np.zeros((33, 4), dtype=np.float32)
        mp[23, :3] = np.array([2.0, 0.0, 0.0], dtype=np.float32)  # left hip
        mp[24, :3] = np.array([4.0, 2.0, 0.0], dtype=np.float32)  # right hip
        out = mediapipe33_to_anubis32(mp)
        assert np.allclose(out[0], np.array([3.0, 1.0, 0.0], dtype=np.float32))

    def test_direct_joint_mappings(self) -> None:
        mp = np.zeros((33, 4), dtype=np.float32)
        mp[23, :3] = np.array([1, 2, 3], dtype=np.float32)  # L_Hip
        mp[24, :3] = np.array([4, 5, 6], dtype=np.float32)  # R_Hip
        mp[25, :3] = np.array([7, 8, 9], dtype=np.float32)  # L_Knee
        mp[26, :3] = np.array([10, 11, 12], dtype=np.float32)  # R_Knee
        out = mediapipe33_to_anubis32(mp)
        assert np.allclose(out[6], mp[23, :3])
        assert np.allclose(out[11], mp[24, :3])
        assert np.allclose(out[7], mp[25, :3])
        assert np.allclose(out[12], mp[26, :3])

    def test_invalid_shape_raises(self) -> None:
        bad = np.zeros((32, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            mediapipe33_to_anubis32(bad)
