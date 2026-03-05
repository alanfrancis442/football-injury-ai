# module1/pose/extractor.py
#
# Stage 1 — Pose Extraction
# ─────────────────────────
# Reads a video file (or webcam stream) frame-by-frame, runs MediaPipe Pose on
# each frame, and returns / saves the 33 keypoints per person per frame.
#
# Each keypoint:  (x, y, z, visibility)
#   x, y  — normalised [0,1] coordinates in the frame
#   z      — depth estimate relative to hips (negative = closer to camera)
#   visibility — confidence score [0,1]
#
# Usage (from project root):
#   python module1/pose/extractor.py --video data/samples/clip.mp4 --out data/processed/
#
# Output:
#   data/processed/<video_name>_keypoints.npy   shape: (n_frames, 33, 4)

import argparse
import os

import cv2
import mediapipe as mp
import numpy as np


# ── MediaPipe setup ────────────────────────────────────────────────────────────

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ── Joint index reference (MediaPipe 33-keypoint schema) ──────────────────────
# Use these constants when building the skeleton graph in module1/gnn/graph.py
KEYPOINT_NAMES = [
    "nose",  # 0
    "left_eye_inner",  # 1
    "left_eye",  # 2
    "left_eye_outer",  # 3
    "right_eye_inner",  # 4
    "right_eye",  # 5
    "right_eye_outer",  # 6
    "left_ear",  # 7
    "right_ear",  # 8
    "mouth_left",  # 9
    "mouth_right",  # 10
    "left_shoulder",  # 11
    "right_shoulder",  # 12
    "left_elbow",  # 13
    "right_elbow",  # 14
    "left_wrist",  # 15
    "right_wrist",  # 16
    "left_pinky",  # 17
    "right_pinky",  # 18
    "left_index",  # 19
    "right_index",  # 20
    "left_thumb",  # 21
    "right_thumb",  # 22
    "left_hip",  # 23
    "right_hip",  # 24
    "left_knee",  # 25
    "right_knee",  # 26
    "left_ankle",  # 27
    "right_ankle",  # 28
    "left_heel",  # 29
    "right_heel",  # 30
    "left_foot_index",  # 31
    "right_foot_index",  # 32
]

NUM_KEYPOINTS = 33  # total joints
KEYPOINT_DIM = 4  # (x, y, z, visibility)


# ── Core extractor class ───────────────────────────────────────────────────────


class PoseExtractor:
    """
    Extracts 33 body keypoints per frame using MediaPipe Pose.

    Parameters
    ----------
    min_detection_confidence : float
        Minimum confidence for initial person detection.
    min_tracking_confidence : float
        Minimum confidence to continue tracking across frames.
    visualise : bool
        If True, opens an OpenCV window showing the skeleton overlay.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        visualise: bool = False,
    ):
        self.visualise = visualise
        self._pose = mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_from_video(self, video_path: str) -> np.ndarray:
        """
        Process a full video file and return all keyframes as a numpy array.

        Returns
        -------
        keypoints : np.ndarray  shape (n_frames, 33, 4)
            Each frame contains 33 joints × (x, y, z, visibility).
            Frames where no person was detected are filled with zeros.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        all_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = self._process_frame(frame)
            all_frames.append(keypoints)

            if self.visualise:
                cv2.imshow("Pose Extraction", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if self.visualise:
            cv2.destroyAllWindows()

        return np.array(all_frames)  # (n_frames, 33, 4)

    def extract_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single BGR frame (from OpenCV) and return keypoints.

        Returns
        -------
        keypoints : np.ndarray  shape (33, 4)
        """
        return self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Internal: run MediaPipe on one frame, return (33, 4) array."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        keypoints = np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                keypoints[i] = [lm.x, lm.y, lm.z, lm.visibility]

            if self.visualise:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

        return keypoints

    def close(self):
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Save / load helpers ────────────────────────────────────────────────────────


def save_keypoints(keypoints: np.ndarray, out_dir: str, video_name: str) -> str:
    """Save extracted keypoints to a .npy file. Returns the saved path."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{video_name}_keypoints.npy")
    np.save(out_path, keypoints)
    print(f"Saved keypoints → {out_path}  shape={keypoints.shape}")
    return out_path


def load_keypoints(npy_path: str) -> np.ndarray:
    """Load previously saved keypoints from a .npy file."""
    return np.load(npy_path)


# ── CLI entrypoint ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Extract pose keypoints from a video.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--out", default="data/processed/", help="Output directory")
    parser.add_argument(
        "--visualise", action="store_true", help="Show skeleton overlay"
    )
    args = parser.parse_args()

    video_name = os.path.splitext(os.path.basename(args.video))[0]

    with PoseExtractor(visualise=args.visualise) as extractor:
        print(f"Processing: {args.video}")
        keypoints = extractor.extract_from_video(args.video)

    save_keypoints(keypoints, args.out, video_name)


if __name__ == "__main__":
    main()
