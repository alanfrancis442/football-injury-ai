# module1/pipeline.py
#
# End-to-End Module 1 Pipeline
# ─────────────────────────────
# Connects all three stages into a single callable:
#
#   Video frame  →  PoseExtractor  →  tracker  →  GNN  →  Transformer
#                                                               ↓
#                                                       biomechanical_risk_score
#                                                       risk_body_region
#                                                       pattern_label
#
# This is the file you run to test the full pipeline on a clip.
# Each stage can also be tested independently from its own module.
#
# Usage:
#   python module1/pipeline.py --video data/samples/clip.mp4
#   python module1/pipeline.py --video data/samples/clip.mp4 --visualise

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch_geometric.data import Batch

from module1.gnn.graph import build_graph
from module1.gnn.model import SkeletonGAT
from module1.pose.extractor import PoseExtractor
from module1.pose.tracker import SimpleTracker
from module1.transformer.model import BODY_REGIONS, BiomechanicalTransformer
from utils.config import load_config
from utils.logger import get_logger

log = get_logger(__name__)


# ── Output dataclass ───────────────────────────────────────────────────────────


@dataclass
class PlayerRiskOutput:
    """Risk assessment for a single player at the current frame."""

    player_id: int
    risk_score: float  # 0.0 – 1.0
    risk_tier: str  # "GREEN" | "AMBER" | "ORANGE" | "RED"
    body_region: str  # e.g. "left_hip"
    pattern_label: str  # e.g. "hip_drop_progressive"
    frame_number: int


def score_to_tier(score: float) -> str:
    if score < 0.30:
        return "GREEN"
    if score < 0.55:
        return "AMBER"
    if score < 0.75:
        return "ORANGE"
    return "RED"


TIER_COLORS = {
    "GREEN": (0, 200, 0),
    "AMBER": (0, 200, 255),
    "ORANGE": (0, 128, 255),
    "RED": (0, 0, 220),
}


# ── Module 1 Pipeline ─────────────────────────────────────────────────────────


class Module1Pipeline:
    """
    Full biomechanical risk pipeline.

    Parameters
    ----------
    config_path : str   — path to configs/module1.yaml
    gnn_ckpt    : str   — path to trained GNN checkpoint (.pt)
    trans_ckpt  : str   — path to trained Transformer checkpoint (.pt)
    device      : str   — "cuda" or "cpu"

    Note: If no checkpoints are provided, the pipeline runs with randomly
    initialised weights — useful for testing the full data flow before training.
    """

    def __init__(
        self,
        config_path: str = "configs/module1.yaml",
        gnn_ckpt: Optional[str] = None,
        trans_ckpt: Optional[str] = None,
        device: str = "cpu",
    ):
        self.cfg = load_config(config_path)
        self.device = torch.device(device)

        gnn_cfg = self.cfg["gnn"]
        trans_cfg = self.cfg["transformer"]

        # ── Pose extractor + tracker ───────────────────────────────────────────
        self.extractor = PoseExtractor(
            min_detection_confidence=self.cfg["pose"]["min_detection_confidence"],
            min_tracking_confidence=self.cfg["pose"]["min_tracking_confidence"],
        )
        self.tracker = SimpleTracker(
            iou_threshold=self.cfg["pose"]["iou_threshold"],
            max_frames_lost=self.cfg["pose"]["max_frames_lost"],
        )

        # ── GNN ───────────────────────────────────────────────────────────────
        self.gnn = SkeletonGAT(
            in_channels=gnn_cfg["in_channels"],
            hidden_dim=gnn_cfg["hidden_dim"],
            out_dim=gnn_cfg["out_dim"],
            heads=gnn_cfg["heads"],
            dropout=0.0,  # no dropout at inference
        ).to(self.device)

        if gnn_ckpt and os.path.exists(gnn_ckpt):
            ckpt = torch.load(gnn_ckpt, map_location=self.device)
            self.gnn.load_state_dict(ckpt["model"])
            log.info(f"Loaded GNN weights from {gnn_ckpt}")
        else:
            log.warning("No GNN checkpoint — running with random weights (dev mode)")

        self.gnn.eval()

        # ── Transformer ───────────────────────────────────────────────────────
        self.transformer = BiomechanicalTransformer(
            input_dim=gnn_cfg["out_dim"],
            d_model=trans_cfg["d_model"],
            nhead=trans_cfg["nhead"],
            num_layers=trans_cfg["num_layers"],
            dim_ff=trans_cfg["dim_feedforward"],
            dropout=0.0,
            seq_len=trans_cfg["seq_len"],
        ).to(self.device)

        if trans_ckpt and os.path.exists(trans_ckpt):
            ckpt = torch.load(trans_ckpt, map_location=self.device)
            self.transformer.load_state_dict(ckpt["model"])
            log.info(f"Loaded Transformer weights from {trans_ckpt}")
        else:
            log.warning(
                "No Transformer checkpoint — running with random weights (dev mode)"
            )

        self.transformer.eval()

        # Per-player GNN feature history buffers: {player_id: list of 64-dim tensors}
        self._gnn_history: Dict[int, List[torch.Tensor]] = {}
        self._seq_len = trans_cfg["seq_len"]

    @torch.no_grad()
    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
    ) -> List[PlayerRiskOutput]:
        """
        Run the full pipeline on one BGR video frame.

        Returns a list of PlayerRiskOutput — one per tracked player.
        """
        h, w = frame.shape[:2]

        # Stage 1: Pose extraction
        keypoints = self.extractor.extract_from_frame(frame)  # (33, 4)

        # For simplicity, treat the frame as one player detection.
        # Multi-player: collect multiple (33,4) detections and pass as a list.
        detections = [keypoints] if keypoints.sum() > 0 else []
        active_players = self.tracker.update(detections, frame_w=w, frame_h=h)

        results: List[PlayerRiskOutput] = []

        for pid, player in active_players.items():
            # Stage 2: GNN — spatial feature for this frame
            history_arr = player.get_history_array()  # (50, 33, 4)
            graph = build_graph(history_arr)
            graph = graph.to(self.device)

            # Batch of one graph
            batch_tensor = torch.zeros(33, dtype=torch.long).to(self.device)
            spatial_vec = self.gnn(graph.x, graph.edge_index, batch_tensor)  # (1, 64)

            # Accumulate GNN history for the Transformer
            if pid not in self._gnn_history:
                self._gnn_history[pid] = []
            self._gnn_history[pid].append(spatial_vec.squeeze(0))

            # Keep only the last seq_len vectors
            if len(self._gnn_history[pid]) > self._seq_len:
                self._gnn_history[pid].pop(0)

            # Stage 3: Transformer — temporal risk
            history_list = self._gnn_history[pid]
            seq = torch.stack(history_list, dim=0)  # (T, 64)

            # Pad with zeros if fewer frames than seq_len
            if seq.shape[0] < self._seq_len:
                pad = torch.zeros(self._seq_len - seq.shape[0], seq.shape[1]).to(
                    self.device
                )
                seq = torch.cat([pad, seq], dim=0)

            seq = seq.unsqueeze(0)  # (1, T, 64)
            out = self.transformer(seq)

            risk_score = out["risk_score"].item()
            region_idx = out["region_logits"].argmax(dim=1).item()
            body_region = BODY_REGIONS[region_idx]

            results.append(
                PlayerRiskOutput(
                    player_id=pid,
                    risk_score=round(risk_score, 3),
                    risk_tier=score_to_tier(risk_score),
                    body_region=body_region,
                    pattern_label="pending_training",  # populated after model is trained
                    frame_number=frame_number,
                )
            )

        return results

    def process_video(
        self, video_path: str, visualise: bool = False
    ) -> List[List[PlayerRiskOutput]]:
        """
        Run the pipeline on every frame of a video file.

        Returns a list of frame results (each is a list of PlayerRiskOutput).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        all_results = []
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_results = self.process_frame(frame, frame_number=frame_num)
            all_results.append(frame_results)

            if visualise:
                self._draw_overlay(frame, frame_results)
                cv2.imshow("Module 1 — Biomechanical Risk", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_num += 1

        cap.release()
        if visualise:
            cv2.destroyAllWindows()

        return all_results

    def _draw_overlay(self, frame: np.ndarray, results: List[PlayerRiskOutput]):
        """Draw risk tier and score on the frame for visualisation."""
        for i, r in enumerate(results):
            color = TIER_COLORS[r.risk_tier]
            text = (
                f"P{r.player_id} | {r.risk_tier} {r.risk_score:.2f} | {r.body_region}"
            )
            y_pos = 30 + i * 30
            cv2.putText(
                frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

    def close(self):
        self.extractor.close()


# ── CLI entrypoint ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run Module 1 biomechanical risk pipeline."
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--config", default="configs/module1.yaml")
    parser.add_argument("--gnn_ckpt", default=None, help="GNN checkpoint path")
    parser.add_argument(
        "--trans_ckpt", default=None, help="Transformer checkpoint path"
    )
    parser.add_argument("--visualise", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    pipeline = Module1Pipeline(
        config_path=args.config,
        gnn_ckpt=args.gnn_ckpt,
        trans_ckpt=args.trans_ckpt,
        device=args.device,
    )

    log.info(f"Processing: {args.video}")
    all_results = pipeline.process_video(args.video, visualise=args.visualise)

    # Print summary of last frame results
    if all_results:
        log.info(f"\nFinal frame results ({len(all_results)} frames processed):")
        for r in all_results[-1]:
            log.info(
                f"  Player {r.player_id}: {r.risk_tier} ({r.risk_score:.3f}) — {r.body_region}"
            )

    pipeline.close()


if __name__ == "__main__":
    main()
