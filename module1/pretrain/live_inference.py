# module1/pretrain/live_inference.py
#
# Stage 1 - Live ANUBIS Motion Inference
# --------------------------------------
# Runs webcam/video/RTSP live inference using the pretrained ActionClassifier
# backbone. This is an action-understanding test pipeline for the pretrained
# model (102 classes), not the football injury-risk head.

from __future__ import annotations

import argparse
import collections
import os
import time
from typing import Deque, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from module1.data.joint_config import EDGE_INDEX
from module1.pose.extractor import PoseExtractor
from module1.pretrain.live_utils import (
    build_feature_window,
    load_class_names,
    parse_video_source,
    smooth_probs,
)
from module1.pretrain.model import ActionClassifier, build_model
from utils.config import load_config
from utils.logger import get_logger

log = get_logger(__name__)


def _draw_overlay(
    frame: np.ndarray,
    class_names: List[str],
    probs: torch.Tensor,
    topk: int,
    fps: float,
    frames_ready: int,
    seq_len: int,
) -> np.ndarray:
    """Draw prediction text overlay onto a frame."""
    canvas = frame.copy()
    _, W = canvas.shape[:2]

    # Background panel
    panel_h = 30 + (topk + 3) * 24
    cv2.rectangle(canvas, (10, 10), (min(W - 10, 680), 10 + panel_h), (0, 0, 0), -1)
    cv2.rectangle(canvas, (10, 10), (min(W - 10, 680), 10 + panel_h), (70, 70, 70), 1)

    y = 34
    cv2.putText(
        canvas,
        "Live ANUBIS Motion Inference (Pretraining Model)",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    y += 26
    cv2.putText(
        canvas,
        f"FPS: {fps:.1f} | Window: {frames_ready}/{seq_len}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (170, 255, 170),
        1,
        cv2.LINE_AA,
    )

    topv, topi = torch.topk(probs, k=min(topk, probs.numel()))
    for rank, (score, idx) in enumerate(zip(topv.tolist(), topi.tolist()), start=1):
        y += 24
        label = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        text = f"{rank}. [{idx:03d}] {label}  prob={score:.3f}"
        cv2.putText(
            canvas,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def _load_checkpoint(
    ckpt_path: str,
    classifier: ActionClassifier,
    device: torch.device,
) -> None:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run pretraining first or provide --ckpt path."
        )

    ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = None
    if isinstance(ckpt, dict):
        if "classifier" in ckpt:
            state_dict = ckpt["classifier"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            values = list(ckpt.values())
            if values and all(torch.is_tensor(v) for v in values):
                state_dict = ckpt
    elif hasattr(ckpt, "keys"):
        state_dict = ckpt

    if state_dict is None:
        raise ValueError(
            "Checkpoint format not recognised. Expected either a full training "
            "checkpoint with key 'classifier' or a classifier state_dict."
        )

    classifier.load_state_dict(state_dict)

    log.info(
        "Loaded checkpoint from %s (epoch=%s val_top1=%s)",
        ckpt_path,
        str(ckpt.get("epoch", "n/a")) if isinstance(ckpt, dict) else "n/a",
        (
            f"{ckpt.get('val_top1', float('nan')):.3f}"
            if isinstance(ckpt, dict) and "val_top1" in ckpt
            else "n/a"
        ),
    )


def run_live(
    config_path: str,
    ckpt_path: str,
    source: str,
    topk: int,
    smooth_alpha: float,
    label_map: Optional[str],
    show: bool,
) -> None:
    """
    Run live action inference from a camera/video stream.

    Parameters
    ----------
    config_path : str
    ckpt_path : str
    source : str
        Camera index (e.g. "0"), video path, or RTSP URL.
    topk : int
        Number of top predictions shown in the overlay.
    smooth_alpha : float
        EMA alpha for probability smoothing (0=no smoothing, close to 1=strong).
    label_map : str, optional
        Optional class-name map file (.txt/.yaml/.json).
    show : bool
        Whether to display OpenCV window. Disable for headless testing.
    """
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    edge_index = EDGE_INDEX.to(device)
    _, classifier = build_model(cfg, edge_index)
    classifier = classifier.to(device)
    classifier.eval()
    _load_checkpoint(ckpt_path, classifier, device)

    num_classes = int(cfg["data"]["num_classes"])
    seq_len = int(cfg["data"]["seq_len"])
    class_names = load_class_names(num_classes, label_map)

    parsed_source = parse_video_source(source)
    cap = cv2.VideoCapture(parsed_source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open source: {source}\nCheck camera index/path/URL and retry."
        )

    log.info(
        "Starting live inference from source=%s | seq_len=%d | topk=%d",
        str(source),
        seq_len,
        topk,
    )

    frame_buffer: Deque[np.ndarray] = collections.deque(maxlen=seq_len)
    smoothed_probs: Optional[torch.Tensor] = None
    fps = 0.0
    prev_time = time.time()

    with PoseExtractor() as extractor:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    log.info("Stream ended or frame read failed.")
                    break

                joints32 = extractor.extract_from_frame(frame)
                if joints32 is not None:
                    frame_buffer.append(joints32)

                # Default distribution before warm-up.
                probs = torch.full(
                    (num_classes,), 1.0 / num_classes, dtype=torch.float32
                )

                if len(frame_buffer) == seq_len:
                    window = np.stack(frame_buffer, axis=0)  # (T, J, 3)
                    features = build_feature_window(window, pelvis_idx=0)  # (T, J, 9)

                    with torch.no_grad():
                        x = features.unsqueeze(0).to(device)  # (1, T, J, 9)
                        logits = classifier(x).squeeze(0)
                        current_probs = F.softmax(logits, dim=0).cpu()

                    probs = smooth_probs(
                        smoothed_probs, current_probs, alpha=smooth_alpha
                    )
                    smoothed_probs = probs

                # FPS update
                now = time.time()
                delta = max(now - prev_time, 1e-6)
                fps = 0.9 * fps + 0.1 * (1.0 / delta)
                prev_time = now

                if show:
                    overlay = _draw_overlay(
                        frame=frame,
                        class_names=class_names,
                        probs=probs,
                        topk=topk,
                        fps=fps,
                        frames_ready=len(frame_buffer),
                        seq_len=seq_len,
                    )
                    cv2.imshow("Football Injury AI - Live Motion Test", overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

        finally:
            cap.release()
            if show:
                cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run live ANUBIS action inference from webcam/video stream"
    )
    parser.add_argument(
        "--config",
        default="configs/pretrain_final.yaml",
        help="Path to pretrain config YAML",
    )
    parser.add_argument(
        "--ckpt",
        default="outputs/checkpoints/pretrain/pretrain_best.pt",
        help="Path to classifier checkpoint",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index, file path, or RTSP URL",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top classes shown",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.7,
        help="EMA alpha for probability smoothing in [0,1)",
    )
    parser.add_argument(
        "--label-map",
        default=None,
        help="Optional class-name map (.txt/.yaml/.json)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable OpenCV preview window (headless)",
    )
    args = parser.parse_args()

    run_live(
        config_path=args.config,
        ckpt_path=args.ckpt,
        source=args.source,
        topk=max(1, args.topk),
        smooth_alpha=args.smooth_alpha,
        label_map=args.label_map,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
