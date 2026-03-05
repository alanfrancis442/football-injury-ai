# tests/test_pose.py
#
# Unit tests for Stage 1 — Pose Extraction & Tracking
# Run with: pytest tests/test_pose.py -v

import numpy as np
import pytest

from module1.pose.extractor import (
    KEYPOINT_DIM,
    NUM_KEYPOINTS,
    load_keypoints,
    save_keypoints,
)
from module1.pose.tracker import (
    SimpleTracker,
    TrackedPlayer,
    iou,
    keypoints_to_bbox,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def dummy_keypoints():
    """Random keypoints for one person — shape (33, 4)."""
    kp = np.random.rand(NUM_KEYPOINTS, KEYPOINT_DIM).astype(np.float32)
    kp[:, 3] = 0.9  # set all visibilities high
    return kp


@pytest.fixture
def zero_keypoints():
    """All-zero keypoints (no person detected)."""
    return np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)


# ── Extractor tests ────────────────────────────────────────────────────────────


class TestKeypoints:
    def test_shape(self, dummy_keypoints):
        assert dummy_keypoints.shape == (NUM_KEYPOINTS, KEYPOINT_DIM)

    def test_dtype(self, dummy_keypoints):
        assert dummy_keypoints.dtype == np.float32

    def test_save_and_load(self, dummy_keypoints, tmp_path):
        """Save keypoints to disk and reload — values must be identical."""
        seq = np.stack([dummy_keypoints] * 10)  # 10 frames
        save_keypoints(seq, str(tmp_path), "test_clip")
        loaded = load_keypoints(str(tmp_path / "test_clip_keypoints.npy"))
        np.testing.assert_array_equal(seq, loaded)


# ── Tracker tests ──────────────────────────────────────────────────────────────


class TestIoU:
    def test_identical_boxes(self):
        box = (10, 10, 100, 100)
        assert iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # Two 100×100 boxes overlapping by 50×100
        a = (0, 0, 100, 100)
        b = (50, 0, 150, 100)
        result = iou(a, b)
        assert 0.0 < result < 1.0


class TestBboxFromKeypoints:
    def test_output_is_tuple_of_4(self, dummy_keypoints):
        bbox = keypoints_to_bbox(dummy_keypoints, frame_w=1920, frame_h=1080)
        assert len(bbox) == 4

    def test_bbox_within_frame(self, dummy_keypoints):
        W, H = 1920, 1080
        x1, y1, x2, y2 = keypoints_to_bbox(dummy_keypoints, W, H)
        assert 0 <= x1 < x2 <= W
        assert 0 <= y1 < y2 <= H

    def test_zero_keypoints_returns_full_frame(self, zero_keypoints):
        W, H = 640, 480
        bbox = keypoints_to_bbox(zero_keypoints, W, H)
        assert bbox == (0, 0, W, H)


class TestSimpleTracker:
    def test_new_detection_creates_player(self, dummy_keypoints):
        tracker = SimpleTracker()
        active = tracker.update([dummy_keypoints], frame_w=640, frame_h=480)
        assert len(active) == 1

    def test_same_player_keeps_id(self, dummy_keypoints):
        tracker = SimpleTracker()
        active1 = tracker.update([dummy_keypoints], 640, 480)
        active2 = tracker.update([dummy_keypoints], 640, 480)
        assert list(active1.keys()) == list(active2.keys())

    def test_history_grows_up_to_max(self, dummy_keypoints):
        tracker = SimpleTracker()
        for _ in range(60):  # more than MAX_HISTORY (50)
            tracker.update([dummy_keypoints], 640, 480)

        pid = list(tracker.active_players.keys())[0]
        hist = tracker.active_players[pid].get_history_array()
        assert hist.shape == (TrackedPlayer.MAX_HISTORY, NUM_KEYPOINTS, KEYPOINT_DIM)

    def test_reset_clears_all_tracks(self, dummy_keypoints):
        tracker = SimpleTracker()
        tracker.update([dummy_keypoints], 640, 480)
        tracker.reset()
        assert len(tracker.active_players) == 0
