# module1/pose/tracker.py
#
# Stage 1 — Player Tracking
# ─────────────────────────
# Associates skeleton keypoints to persistent player IDs across frames.
# This is a lightweight bounding-box tracker built on top of pose detections.
#
# Why this matters:
#   MediaPipe gives you keypoints per frame, but has no memory between frames.
#   If player #7 moves from left to right across 200 frames, we need to know
#   it's the same player each time to build their movement history buffer.
#
# Approach:
#   - Derive a bounding box from the keypoints (min/max of x,y coordinates)
#   - Use IoU (Intersection over Union) to match boxes across consecutive frames
#   - Assign stable player IDs based on matching score
#
# For a more robust tracker (handles occlusions, crowded areas), the next
# upgrade is ByteTrack: https://github.com/ifzhang/ByteTrack
# Swap it in by replacing the `SimpleTracker` class below.

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Bounding box helpers ───────────────────────────────────────────────────────


def keypoints_to_bbox(
    keypoints: np.ndarray, frame_w: int, frame_h: int
) -> Tuple[int, int, int, int]:
    """
    Derive a bounding box (x1, y1, x2, y2) in pixel coords from keypoints.

    Parameters
    ----------
    keypoints : np.ndarray  shape (33, 4)  — normalised (x, y, z, visibility)
    frame_w, frame_h : int  — actual frame dimensions in pixels
    """
    # Only use keypoints with reasonable visibility
    visible = keypoints[keypoints[:, 3] > 0.3]
    if len(visible) == 0:
        return (0, 0, frame_w, frame_h)

    xs = visible[:, 0] * frame_w
    ys = visible[:, 1] * frame_h

    pad = 20  # pixel padding around the person
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(frame_w, int(xs.max()) + pad)
    y2 = min(frame_h, int(ys.max()) + pad)

    return (x1, y1, x2, y2)


def iou(box_a: Tuple, box_b: Tuple) -> float:
    """
    Compute Intersection over Union between two bounding boxes.
    Boxes are (x1, y1, x2, y2).
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# ── Tracked player dataclass ───────────────────────────────────────────────────


class TrackedPlayer:
    """
    Holds the state for one tracked player across frames.

    Attributes
    ----------
    player_id   : int   — unique ID assigned at first detection
    bbox        : tuple — current bounding box (x1, y1, x2, y2)
    keypoints   : np.ndarray shape (33, 4) — latest keypoints
    history     : list[np.ndarray] — keypoints for the last `max_history` frames
    frames_lost : int   — how many consecutive frames this player was not matched
    """

    MAX_HISTORY = 50  # 2 seconds at 25fps — matches the Transformer window size

    def __init__(self, player_id: int, bbox: Tuple, keypoints: np.ndarray):
        self.player_id = player_id
        self.bbox = bbox
        self.keypoints = keypoints
        self.history: List[np.ndarray] = [keypoints]
        self.frames_lost = 0

    def update(self, bbox: Tuple, keypoints: np.ndarray):
        self.bbox = bbox
        self.keypoints = keypoints
        self.frames_lost = 0
        self.history.append(keypoints)
        if len(self.history) > self.MAX_HISTORY:
            self.history.pop(0)

    def get_history_array(self) -> np.ndarray:
        """
        Returns keypoint history as array of shape (T, 33, 4).
        If fewer than MAX_HISTORY frames are available, pads with zeros at the front.
        """
        arr = np.array(self.history)  # (T, 33, 4)
        if len(arr) < self.MAX_HISTORY:
            pad = np.zeros((self.MAX_HISTORY - len(arr), 33, 4), dtype=np.float32)
            arr = np.concatenate([pad, arr], axis=0)
        return arr


# ── Simple IoU-based tracker ───────────────────────────────────────────────────


class SimpleTracker:
    """
    Assigns persistent player IDs using bounding-box IoU matching.

    Sufficient for single-player or controlled clips.
    For multi-player tracking in crowded scenes, replace with ByteTrack.

    Parameters
    ----------
    iou_threshold   : float — minimum IoU to consider two boxes the same player
    max_frames_lost : int   — remove a player after this many unmatched frames
    """

    def __init__(self, iou_threshold: float = 0.4, max_frames_lost: int = 10):
        self.iou_threshold = iou_threshold
        self.max_frames_lost = max_frames_lost
        self._next_id = 0
        self._players: Dict[int, TrackedPlayer] = {}

    @property
    def active_players(self) -> Dict[int, TrackedPlayer]:
        """Returns currently active (not lost) tracked players."""
        return {pid: p for pid, p in self._players.items() if p.frames_lost == 0}

    def update(
        self,
        detections: List[np.ndarray],  # list of (33, 4) keypoint arrays
        frame_w: int,
        frame_h: int,
    ) -> Dict[int, TrackedPlayer]:
        """
        Match new detections to existing tracks. Returns updated active players.

        Parameters
        ----------
        detections : list of np.ndarray (33, 4)
            One array per detected person in this frame.
        frame_w, frame_h : int
            Frame dimensions (needed to convert normalised coords to pixels).
        """
        # Convert detections to bounding boxes
        det_boxes = [keypoints_to_bbox(kp, frame_w, frame_h) for kp in detections]

        matched_track_ids = set()
        matched_det_indices = set()

        # Match each existing track to the best detection by IoU
        for pid, player in list(self._players.items()):
            if player.frames_lost >= self.max_frames_lost:
                del self._players[pid]
                continue

            best_iou = self.iou_threshold
            best_idx: Optional[int] = None

            for i, box in enumerate(det_boxes):
                if i in matched_det_indices:
                    continue
                score = iou(player.bbox, box)
                if score > best_iou:
                    best_iou = score
                    best_idx = i

            if best_idx is not None:
                player.update(det_boxes[best_idx], detections[best_idx])
                matched_track_ids.add(pid)
                matched_det_indices.add(best_idx)
            else:
                player.frames_lost += 1

        # Unmatched detections become new players
        for i, (box, kp) in enumerate(zip(det_boxes, detections)):
            if i not in matched_det_indices:
                new_player = TrackedPlayer(self._next_id, box, kp)
                self._players[self._next_id] = new_player
                self._next_id += 1

        return self.active_players

    def reset(self):
        """Clear all tracks (call between videos)."""
        self._players = {}
        self._next_id = 0
