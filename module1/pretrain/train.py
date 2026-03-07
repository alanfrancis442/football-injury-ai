# module1/pretrain/train.py
#
# Stage 1 — ANUBIS Pretraining Loop
# ──────────────────────────────────
# Trains ActionClassifier end-to-end on ANUBIS 102-class action recognition.
# LR schedule: linear warm-up for warmup_epochs, then cosine annealing.
# After training, only encoder weights are needed for fine-tuning.
#
# Usage:
#   python module1/pretrain/train.py --config configs/pretrain.yaml
#
# Checkpoints are saved to paths.checkpoint_dir:
#   pretrain_best.pt     — best validation loss
#   pretrain_ep{N}.pt    — periodic saves every save_every_n_epochs epochs

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from module1.data.anubis_loader import build_dataloaders
from module1.data.joint_config import EDGE_INDEX
from module1.pretrain.model import ActionClassifier, SpatioTemporalEncoder, build_model
from utils.config import load_config
from utils.logger import get_logger

log = get_logger(__name__)


# ── LR schedule ───────────────────────────────────────────────────────────────


def _make_lr_lambda(warmup_epochs: int, total_epochs: int):
    """
    Return a LambdaLR multiplier function implementing linear warm-up followed
    by cosine annealing.

    Multiplier at epoch e:
      e < warmup_epochs  →  (e+1) / warmup_epochs   (linear ramp)
      e >= warmup_epochs →  0.5 * (1 + cos(π * progress))
    """

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


# ── Training / evaluation functions ──────────────────────────────────────────


def train_one_epoch(
    model: ActionClassifier,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    grad_clip: float,
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Parameters
    ----------
    model     : ActionClassifier
    loader    : DataLoader         training data
    optimiser : torch.optim.Optimizer
    criterion : nn.CrossEntropyLoss
    device    : torch.device
    grad_clip : float              max_norm for gradient clipping

    Returns
    -------
    avg_loss : float   mean CE loss over all batches
    accuracy : float   top-1 accuracy in [0, 1]
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in tqdm(loader, desc="  train", leave=False, unit="batch"):
        x = x.to(device, non_blocking=True)  # (B, T, J, F)
        y = y.to(device, non_blocking=True)  # (B,)

        optimiser.zero_grad()
        logits = model(x)  # (B, num_classes)
        loss = criterion(logits, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimiser.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += bs

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: ActionClassifier,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Evaluate the model on a data split.

    Parameters
    ----------
    model     : ActionClassifier
    loader    : DataLoader
    criterion : nn.CrossEntropyLoss  (label_smoothing=0 has no effect at eval)
    device    : torch.device

    Returns
    -------
    avg_loss  : float   mean CE loss
    top1_acc  : float   top-1 accuracy in [0, 1]
    top5_acc  : float   top-5 accuracy in [0, 1]
    """
    model.eval()
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    for x, y in tqdm(loader, desc="  val  ", leave=False, unit="batch"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)  # (B, num_classes)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs

        # Top-1
        total_top1 += (logits.argmax(dim=1) == y).sum().item()

        # Top-5
        top5_preds = logits.topk(k=min(5, logits.size(1)), dim=1).indices  # (B, 5)
        total_top5 += (top5_preds == y.unsqueeze(1)).any(dim=1).sum().item()

        total_samples += bs

    avg_loss = total_loss / total_samples
    top1_acc = total_top1 / total_samples
    top5_acc = total_top5 / total_samples
    return avg_loss, top1_acc, top5_acc


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def _save_checkpoint(
    path: str,
    epoch: int,
    encoder: SpatioTemporalEncoder,
    classifier: ActionClassifier,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metrics: Dict,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "classifier": classifier.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            **metrics,
        },
        path,
    )
    log.info("Saved checkpoint → %s", path)


# ── Main training entry point ─────────────────────────────────────────────────


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Data ────────────────────────────────────────────────────────────────
    log.info("Loading ANUBIS dataset …")
    train_loader, val_loader = build_dataloaders(cfg)

    # ── Model ───────────────────────────────────────────────────────────────
    edge_index = EDGE_INDEX.to(device)
    encoder, classifier = build_model(cfg, edge_index)
    classifier = classifier.to(device)

    # ── Optimiser ───────────────────────────────────────────────────────────
    t_cfg = cfg["training"]
    optimiser = torch.optim.AdamW(
        classifier.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
    )

    # Warm-up + cosine LR schedule
    total_epochs = t_cfg["epochs"]
    warmup_epochs = t_cfg.get("warmup_epochs", 5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lr_lambda=_make_lr_lambda(warmup_epochs, total_epochs),
    )

    # ── Loss ────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=t_cfg.get("label_smoothing", 0.1))
    eval_criterion = nn.CrossEntropyLoss()  # no smoothing for val metrics

    # ── Training loop ────────────────────────────────────────────────────────
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    save_every = t_cfg.get("save_every_n_epochs", 10)
    grad_clip = t_cfg.get("grad_clip_norm", 1.0)

    best_val_loss = float("inf")
    best_epoch = 0

    log.info("Starting pretraining for %d epochs", total_epochs)

    for epoch in range(1, total_epochs + 1):
        current_lr = optimiser.param_groups[0]["lr"]
        log.info("Epoch %d/%d  lr=%.6f", epoch, total_epochs, current_lr)

        # Train
        trn_loss, trn_acc = train_one_epoch(
            classifier, train_loader, optimiser, criterion, device, grad_clip
        )

        # Validate
        val_loss, val_top1, val_top5 = evaluate(
            classifier, val_loader, eval_criterion, device
        )

        scheduler.step()

        log.info(
            "  trn loss=%.4f  acc=%.3f | val loss=%.4f  top1=%.3f  top5=%.3f",
            trn_loss,
            trn_acc,
            val_loss,
            val_top1,
            val_top5,
        )

        metrics = {
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "trn_loss": trn_loss,
            "trn_acc": trn_acc,
        }

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            _save_checkpoint(
                os.path.join(ckpt_dir, "pretrain_best.pt"),
                epoch,
                encoder,
                classifier,
                optimiser,
                scheduler,
                metrics,
            )
            log.info("  *** New best at epoch %d  val_loss=%.4f ***", epoch, val_loss)

        # Periodic checkpoint
        if epoch % save_every == 0:
            _save_checkpoint(
                os.path.join(ckpt_dir, f"pretrain_ep{epoch:04d}.pt"),
                epoch,
                encoder,
                classifier,
                optimiser,
                scheduler,
                metrics,
            )

    log.info(
        "Pretraining complete.  Best epoch: %d  val_loss=%.4f",
        best_epoch,
        best_val_loss,
    )
    log.info(
        "Encoder weights saved in:  %s/pretrain_best.pt  (key: 'encoder')", ckpt_dir
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain SpatioTemporalEncoder on ANUBIS"
    )
    parser.add_argument(
        "--config",
        default="configs/pretrain.yaml",
        help="Path to pretrain config YAML",
    )
    args = parser.parse_args()
    main(args.config)
