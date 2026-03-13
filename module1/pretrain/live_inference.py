# module1/pretrain/live_inference.py
#
# Stage 1 — Live ANUBIS Motion Inference
# ──────────────────────────────────────
# Runs webcam/video/RTSP inference using the Stage 1 action classifier. The
# live path adds motion gating and hysteresis so idle frames are treated as
# background/pause instead of forcing an action label every frame.

from __future__ import annotations

import argparse
import collections
import os
import time
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from module1.data.joint_config import EDGE_INDEX, SKELETON_EDGES
from module1.pose.extractor import PoseExtractor
from module1.pretrain.live_utils import (
    BACKGROUND_LABEL,
    PredictionTracker,
    build_feature_window,
    compute_motion_energy,
    decode_prediction,
    load_class_names,
    parse_video_source,
    resolve_default_label_map_path,
    smooth_probs,
    update_prediction_tracker,
)
from module1.pretrain.model import ActionClassifier, build_model
from utils.config import load_config
from utils.logger import get_logger

log = get_logger(__name__)


def _draw_overlay(
    frame: np.ndarray,
    joints32: Optional[np.ndarray],
    class_names: List[str],
    probs: torch.Tensor,
    topk: int,
    fps: float,
    frames_ready: int,
    seq_len: int,
    tracker: PredictionTracker,
    status_text: str,
) -> np.ndarray:
    """Draw skeleton and prediction overlay onto the current frame."""
    canvas = frame.copy()

    if joints32 is not None:
        canvas = _draw_pose_skeleton(canvas, joints32)

    _, width = canvas.shape[:2]

    panel_h = 30 + (topk + 5) * 24
    cv2.rectangle(canvas, (10, 10), (min(width - 10, 760), 10 + panel_h), (0, 0, 0), -1)
    cv2.rectangle(
        canvas,
        (10, 10),
        (min(width - 10, 760), 10 + panel_h),
        (70, 70, 70),
        1,
    )

    y = 34
    cv2.putText(
        canvas,
        "Live ANUBIS Motion Inference (Stage 1)",
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

    y += 24
    top_color = (255, 225, 160) if tracker.displayed_index >= 0 else (180, 220, 255)
    cv2.putText(
        canvas,
        f"Displayed: {tracker.displayed_label}  score={tracker.displayed_score:.3f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        top_color,
        1,
        cv2.LINE_AA,
    )

    y += 24
    cv2.putText(
        canvas,
        status_text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (215, 215, 215),
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


def _draw_pose_skeleton(frame: np.ndarray, joints32: np.ndarray) -> np.ndarray:
    """Draw a 2D skeleton from ANUBIS-32 joint coordinates."""
    if joints32.ndim != 2 or joints32.shape[0] < 32 or joints32.shape[1] < 2:
        return frame

    canvas = frame.copy()
    height, width = canvas.shape[:2]

    points: List[Optional[Tuple[int, int]]] = []
    for joint in range(32):
        x = float(joints32[joint, 0])
        y = float(joints32[joint, 1])
        if not (np.isfinite(x) and np.isfinite(y)):
            points.append(None)
            continue
        px = int(np.clip(x * width, 0, max(0, width - 1)))
        py = int(np.clip(y * height, 0, max(0, height - 1)))
        points.append((px, py))

    for src, dst in SKELETON_EDGES:
        ps = points[src]
        pd = points[dst]
        if ps is None or pd is None:
            continue
        cv2.line(canvas, ps, pd, (60, 220, 60), 2, cv2.LINE_AA)

    for point in points:
        if point is None:
            continue
        cv2.circle(canvas, point, 3, (20, 120, 255), -1, cv2.LINE_AA)

    return canvas


def _load_checkpoint(
    ckpt_path: str,
    classifier: ActionClassifier,
    device: torch.device,
) -> None:
    """Load a compatible classifier checkpoint for live inference."""
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
        elif "model" in ckpt:
            state_dict = ckpt["model"]
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
            "Checkpoint format not recognised. Expected a training checkpoint or state_dict."
        )

    try:
        classifier.load_state_dict(state_dict)
    except RuntimeError as exc:
        keys = list(state_dict.keys())
        ckpt_has_tcn = any("encoder.tcn" in k for k in keys)
        ckpt_has_transformer = any("encoder.transformer.layers" in k for k in keys)

        expected_backbone = getattr(
            classifier.encoder, "temporal_backbone", "transformer"
        )
        inferred_backbone = "unknown"
        if ckpt_has_tcn and not ckpt_has_transformer:
            inferred_backbone = "tcn"
        elif ckpt_has_transformer and not ckpt_has_tcn:
            inferred_backbone = "transformer"

        if inferred_backbone != "unknown" and inferred_backbone != expected_backbone:
            raise RuntimeError(
                "Checkpoint/config architecture mismatch detected.\n"
                f"- Config builds: {expected_backbone}\n"
                f"- Checkpoint contains: {inferred_backbone}\n\n"
                "Use a config and checkpoint from the same experiment family.\n"
                "Examples:\n"
                "  Transformer baseline: --config configs/experiments/module1/baseline_transformer.yaml\n"
                "  TCN ablation:         --config configs/experiments/module1/tcn_ablation.yaml"
            ) from exc
        raise

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


def _resolve_label_map_path(
    label_map: Optional[str],
    num_classes: int,
) -> Optional[str]:
    """Resolve a user-provided or built-in label map path."""
    if label_map is not None:
        return label_map
    return resolve_default_label_map_path(num_classes)


def run_live(
    config_path: str,
    ckpt_path: str,
    source: str,
    topk: int,
    smooth_alpha: float,
    label_map: Optional[str],
    show: bool,
) -> None:
    """Run live action inference from a camera, video file, or RTSP stream."""
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
    label_map_path = _resolve_label_map_path(label_map, num_classes)
    class_names = load_class_names(num_classes, label_map_path)

    live_cfg = cfg.get("live", {})
    min_confidence = float(live_cfg.get("min_confidence", 0.40))
    min_margin = float(live_cfg.get("min_margin", 0.08))
    min_motion_energy = float(live_cfg.get("min_motion_energy", 0.015))
    hysteresis_frames = int(live_cfg.get("hysteresis_frames", 3))
    max_missing_frames = int(live_cfg.get("max_missing_frames", 8))

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
    tracker = PredictionTracker()
    missing_pose_frames = 0
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
                    missing_pose_frames = 0
                else:
                    missing_pose_frames += 1

                probs = (
                    smoothed_probs.clone()
                    if smoothed_probs is not None
                    else torch.full(
                        (num_classes,), 1.0 / num_classes, dtype=torch.float32
                    )
                )
                status_text = f"Displayed state: {tracker.reason}"

                if missing_pose_frames >= max_missing_frames:
                    frame_buffer.clear()
                    smoothed_probs = None
                    probs = torch.full(
                        (num_classes,), 1.0 / num_classes, dtype=torch.float32
                    )
                    tracker = PredictionTracker(reason="missing_pose_reset")
                    status_text = "State: missing pose -> reset to background"

                elif joints32 is not None and len(frame_buffer) == seq_len:
                    window = np.stack(frame_buffer, axis=0)
                    features = build_feature_window(window, pelvis_idx=0)
                    motion_energy = compute_motion_energy(window, pelvis_idx=0)

                    with torch.no_grad():
                        x = features.unsqueeze(0).to(device)
                        logits = classifier(x).squeeze(0)
                        current_probs = F.softmax(logits, dim=0).cpu()

                    probs = smooth_probs(
                        smoothed_probs, current_probs, alpha=smooth_alpha
                    )
                    smoothed_probs = probs

                    decoded = decode_prediction(
                        probs=probs,
                        class_names=class_names,
                        motion_energy=motion_energy,
                        min_confidence=min_confidence,
                        min_margin=min_margin,
                        min_motion_energy=min_motion_energy,
                        background_label=BACKGROUND_LABEL,
                    )
                    tracker = update_prediction_tracker(
                        tracker,
                        decoded,
                        hysteresis_frames=hysteresis_frames,
                    )
                    status_text = (
                        f"State: {tracker.reason} | motion={decoded.motion_energy:.3f} "
                        f"| margin={decoded.margin:.3f}"
                    )
                elif joints32 is None:
                    tracker.reason = f"pose_missing:{missing_pose_frames}"
                    status_text = f"State: pose missing ({missing_pose_frames}/{max_missing_frames})"

                now = time.time()
                delta = max(now - prev_time, 1e-6)
                fps = 0.9 * fps + 0.1 * (1.0 / delta)
                prev_time = now

                if show:
                    overlay = _draw_overlay(
                        frame=frame,
                        joints32=joints32,
                        class_names=class_names,
                        probs=probs,
                        topk=topk,
                        fps=fps,
                        frames_ready=len(frame_buffer),
                        seq_len=seq_len,
                        tracker=tracker,
                        status_text=status_text,
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
        default="configs/experiments/module1/baseline_transformer.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--ckpt",
        default="outputs/checkpoints/module1/baseline_transformer/pretrain_best.pt",
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
