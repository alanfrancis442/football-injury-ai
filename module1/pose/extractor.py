# module1/pose/extractor.py
#
# Stage 1 - Live Pose Extraction (MediaPipe -> ANUBIS-32)
# --------------------------------------------------------
# Converts MediaPipe Pose (33 landmarks) into the repository's 32-joint
# skeleton format used by the pretraining model.

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


NUM_MEDIAPIPE_JOINTS = 33
NUM_ANUBIS_JOINTS = 32


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


def mediapipe33_to_anubis32(keypoints_33: np.ndarray) -> np.ndarray:
    """
    Convert MediaPipe's 33-joint layout to the ANUBIS 32-joint layout.

    Parameters
    ----------
    keypoints_33 : np.ndarray  shape (33, 3) or (33, 4)
        MediaPipe landmark array. If a 4th column exists (visibility), it is
        ignored in the output conversion.

    Returns
    -------
    keypoints_32 : np.ndarray  shape (32, 3)
        Joint coordinates in ANUBIS order expected by module1/data/joint_config.py
    """
    if keypoints_33.ndim != 2 or keypoints_33.shape[0] != NUM_MEDIAPIPE_JOINTS:
        raise ValueError(
            f"Expected keypoints shape (33, 3|4), got {tuple(keypoints_33.shape)}"
        )
    if keypoints_33.shape[1] < 3:
        raise ValueError(
            f"Expected at least 3 coordinates per keypoint, got {keypoints_33.shape[1]}"
        )

    mp = keypoints_33[:, :3].astype(np.float32, copy=False)
    out = np.zeros((NUM_ANUBIS_JOINTS, 3), dtype=np.float32)

    # MediaPipe references
    l_shoulder = mp[11]
    r_shoulder = mp[12]
    l_elbow = mp[13]
    r_elbow = mp[14]
    l_wrist = mp[15]
    r_wrist = mp[16]
    l_pinky = mp[17]
    r_pinky = mp[18]
    l_index = mp[19]
    r_index = mp[20]
    l_thumb = mp[21]
    r_thumb = mp[22]
    l_hip = mp[23]
    r_hip = mp[24]
    l_knee = mp[25]
    r_knee = mp[26]
    l_ankle = mp[27]
    r_ankle = mp[28]
    l_heel = mp[29]
    r_heel = mp[30]
    l_foot_index = mp[31]
    r_foot_index = mp[32]

    pelvis = _midpoint(l_hip, r_hip)
    neck = _midpoint(l_shoulder, r_shoulder)
    head = (mp[0] + mp[7] + mp[8]) / 3.0

    # Spine interpolation between pelvis and neck.
    spine1 = pelvis * 0.75 + neck * 0.25
    spine2 = pelvis * 0.50 + neck * 0.50
    spine3 = pelvis * 0.25 + neck * 0.75

    # Left/right collar approximated between neck and shoulder.
    l_collar = _midpoint(neck, l_shoulder)
    r_collar = _midpoint(neck, r_shoulder)

    # Hand centroids and fingertips.
    l_hand = (l_pinky + l_index + l_thumb) / 3.0
    r_hand = (r_pinky + r_index + r_thumb) / 3.0
    l_fingertip = l_index
    r_fingertip = r_index

    # Fill ANUBIS-32 output layout.
    out[0] = pelvis
    out[1] = spine1
    out[2] = spine2
    out[3] = spine3
    out[4] = neck
    out[5] = head

    out[6] = l_hip
    out[7] = l_knee
    out[8] = l_ankle
    out[9] = l_heel
    out[10] = l_foot_index

    out[11] = r_hip
    out[12] = r_knee
    out[13] = r_ankle
    out[14] = r_heel
    out[15] = r_foot_index

    out[16] = l_collar
    out[17] = l_shoulder
    out[18] = l_elbow
    out[19] = l_wrist
    out[20] = l_hand
    out[21] = l_fingertip

    out[22] = r_collar
    out[23] = r_shoulder
    out[24] = r_elbow
    out[25] = r_wrist
    out[26] = r_hand
    out[27] = r_fingertip

    # ANUBIS has toe-base and thumb-toe nodes; MediaPipe has one foot-index point.
    out[28] = l_foot_index
    out[29] = r_foot_index
    out[30] = l_foot_index
    out[31] = r_foot_index

    return out


class PoseExtractor:
    """
    Lightweight MediaPipe Pose wrapper for frame-wise extraction.

    Usage
    -----
    with PoseExtractor() as extractor:
        joints_32 = extractor.extract_from_frame(frame_bgr)
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.model_complexity = int(model_complexity)

        self._mp_pose: Optional[Any] = None
        self._pose: Optional[Any] = None

    def __enter__(self) -> "PoseExtractor":
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "mediapipe is not installed. Install dependencies and retry."
            ) from exc

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._pose is not None:
            self._pose.close()
        self._pose = None
        self._mp_pose = None

    def extract_from_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract one ANUBIS-32 skeleton from a BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray  shape (H, W, 3)

        Returns
        -------
        joints_32 : np.ndarray  shape (32, 3), optional
            None if no pose is detected in this frame.
        """
        if self._pose is None:
            raise RuntimeError(
                "PoseExtractor is not initialised. Use it as a context manager."
            )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(frame_rgb)
        if results.pose_landmarks is None:
            return None

        keypoints_33 = np.array(
            [
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ],
            dtype=np.float32,
        )
        return mediapipe33_to_anubis32(keypoints_33)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam (index 0).")
    else:
        with PoseExtractor() as extractor:
            ok, frame = cap.read()
            if ok:
                joints = extractor.extract_from_frame(frame)
                if joints is not None:
                    print(f"Extracted shape: {joints.shape}")
                else:
                    print("No pose detected on first frame.")
    cap.release()
