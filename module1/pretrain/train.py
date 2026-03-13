# module1/pretrain/train.py
#
# Stage 1 — ANUBIS Pretraining Loop
# ──────────────────────────────────
# Trains the Stage 1 ActionClassifier on ANUBIS and stores experiment-scoped
# checkpoints/metrics so different temporal backbones can be compared cleanly.

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from module1.data.anubis_loader import build_dataloaders
from module1.data.joint_config import EDGE_INDEX
from module1.pretrain.experiments import (
    ensure_experiment_dirs,
    prepare_experiment_config,
    resolve_artifact_path,
    save_json,
    set_seed,
)
from module1.pretrain.model import ActionClassifier, build_model
from utils.config import load_config
from utils.logger import get_logger

log = get_logger(__name__)


# ── LR Schedule ───────────────────────────────────────────────────────────────


def _make_lr_lambda(warmup_epochs: int, total_epochs: int):
    """Return a linear-warmup + cosine-decay LR multiplier."""

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


# ── Train / Eval ──────────────────────────────────────────────────────────────


def train_one_epoch(
    model: ActionClassifier,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    grad_clip: float,
) -> Tuple[float, float]:
    """Run one training epoch and return mean loss and top-1 accuracy."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in tqdm(loader, desc="  train", leave=False, unit="batch"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimiser.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimiser.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: ActionClassifier,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model and return classification metrics."""
    model.eval()
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_samples = 0
    preds: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []

    for x, y in tqdm(loader, desc="  val  ", leave=False, unit="batch"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        pred = logits.argmax(dim=1)
        total_top1 += (pred == y).sum().item()

        top5_preds = logits.topk(k=min(5, logits.size(1)), dim=1).indices
        total_top5 += (top5_preds == y.unsqueeze(1)).any(dim=1).sum().item()
        total_samples += batch_size

        preds.append(pred.cpu())
        labels.append(y.cpu())

    pred_np = torch.cat(preds).numpy()
    label_np = torch.cat(labels).numpy()

    return {
        "val_loss": total_loss / total_samples,
        "val_top1": total_top1 / total_samples,
        "val_top5": total_top5 / total_samples,
        "val_macro_f1": float(f1_score(label_np, pred_np, average="macro")),
    }


# ── Checkpoints ───────────────────────────────────────────────────────────────


def _checkpoint_payload(
    epoch: int,
    classifier: ActionClassifier,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metrics: Dict[str, float],
    cfg: Dict,
) -> Dict:
    """Build a checkpoint payload that preserves both encoder and classifier."""
    exp_cfg = cfg.get("experiment", {})
    return {
        "epoch": epoch,
        "model": classifier.state_dict(),
        "encoder": classifier.encoder.state_dict(),
        "classifier": classifier.state_dict(),
        "optimiser": optimiser.state_dict(),
        "scheduler": scheduler.state_dict(),
        "experiment": {
            "group": exp_cfg.get("group", "module1"),
            "name": exp_cfg.get("name", "default"),
            "slug": exp_cfg.get("slug", "default"),
        },
        "config": cfg,
        **metrics,
    }


def _save_checkpoint(
    path: str,
    payload: Dict,
) -> None:
    """Save a training checkpoint to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    log.info("Saved checkpoint → %s", path)


# ── Main Entry Point ──────────────────────────────────────────────────────────


def main(config_path: str) -> None:
    raw_cfg = load_config(config_path)
    cfg, experiment = prepare_experiment_config(raw_cfg, config_path=config_path)
    ensure_experiment_dirs(experiment)

    seed = int(cfg.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    resolved_config_path = os.path.join(experiment.log_dir, "resolved_config.json")
    save_json(resolved_config_path, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    log.info(
        "Experiment: group=%s name=%s slug=%s",
        experiment.group,
        experiment.name,
        experiment.slug,
    )

    log.info("Loading ANUBIS dataset ...")
    train_loader, val_loader = build_dataloaders(cfg)

    edge_index = EDGE_INDEX.to(device)
    _, classifier = build_model(cfg, edge_index)
    classifier = classifier.to(device)

    train_cfg = cfg["training"]
    optimiser = torch.optim.AdamW(
        classifier.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    total_epochs = int(train_cfg["epochs"])
    warmup_epochs = int(train_cfg.get("warmup_epochs", 5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lr_lambda=_make_lr_lambda(warmup_epochs, total_epochs),
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0))
    )
    eval_criterion = nn.CrossEntropyLoss()

    save_every = int(train_cfg.get("save_every_n_epochs", 10))
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))

    history: List[Dict[str, float]] = []
    history_path = os.path.join(experiment.log_dir, "training_history.json")

    best_loss = float("inf")
    best_top1 = -1.0
    best_macro_f1 = -1.0
    best_epochs = {"loss": 0, "top1": 0, "macro_f1": 0}

    log.info("Starting pretraining for %d epochs", total_epochs)

    for epoch in range(1, total_epochs + 1):
        current_lr = optimiser.param_groups[0]["lr"]
        log.info("Epoch %d/%d  lr=%.6f", epoch, total_epochs, current_lr)

        trn_loss, trn_acc = train_one_epoch(
            classifier,
            train_loader,
            optimiser,
            criterion,
            device,
            grad_clip,
        )
        val_metrics = evaluate(classifier, val_loader, eval_criterion, device)
        scheduler.step()

        epoch_metrics = {
            "epoch": float(epoch),
            "trn_loss": float(trn_loss),
            "trn_acc": float(trn_acc),
            **{key: float(value) for key, value in val_metrics.items()},
        }
        history.append(epoch_metrics)
        save_json(
            history_path,
            {
                "experiment": {
                    "group": experiment.group,
                    "name": experiment.name,
                    "slug": experiment.slug,
                },
                "history": history,
            },
        )

        log.info(
            "  trn loss=%.4f acc=%.3f | val loss=%.4f top1=%.3f top5=%.3f macro_f1=%.3f",
            trn_loss,
            trn_acc,
            val_metrics["val_loss"],
            val_metrics["val_top1"],
            val_metrics["val_top5"],
            val_metrics["val_macro_f1"],
        )

        payload = _checkpoint_payload(
            epoch=epoch,
            classifier=classifier,
            optimiser=optimiser,
            scheduler=scheduler,
            metrics={key: float(value) for key, value in epoch_metrics.items()},
            cfg=cfg,
        )

        _save_checkpoint(
            os.path.join(experiment.checkpoint_dir, "pretrain_last.pt"), payload
        )

        if val_metrics["val_loss"] < best_loss:
            best_loss = float(val_metrics["val_loss"])
            best_epochs["loss"] = epoch
            _save_checkpoint(
                os.path.join(experiment.checkpoint_dir, "pretrain_best_loss.pt"),
                payload,
            )
            log.info(
                "  *** New best loss at epoch %d  val_loss=%.4f ***", epoch, best_loss
            )

        if val_metrics["val_top1"] > best_top1:
            best_top1 = float(val_metrics["val_top1"])
            best_epochs["top1"] = epoch
            best_top1_path = os.path.join(experiment.checkpoint_dir, "pretrain_best.pt")
            _save_checkpoint(best_top1_path, payload)
            _save_checkpoint(
                os.path.join(experiment.checkpoint_dir, "pretrain_best_top1.pt"),
                payload,
            )
            log.info(
                "  *** New best top1 at epoch %d  val_top1=%.4f ***", epoch, best_top1
            )

        if val_metrics["val_macro_f1"] > best_macro_f1:
            best_macro_f1 = float(val_metrics["val_macro_f1"])
            best_epochs["macro_f1"] = epoch
            _save_checkpoint(
                os.path.join(experiment.checkpoint_dir, "pretrain_best_macro_f1.pt"),
                payload,
            )
            log.info(
                "  *** New best macro F1 at epoch %d  val_macro_f1=%.4f ***",
                epoch,
                best_macro_f1,
            )

        if epoch % save_every == 0:
            _save_checkpoint(
                os.path.join(experiment.checkpoint_dir, f"pretrain_ep{epoch:04d}.pt"),
                payload,
            )

    summary_path = os.path.join(experiment.log_dir, "training_summary.json")
    save_json(
        summary_path,
        {
            "experiment": {
                "group": experiment.group,
                "name": experiment.name,
                "slug": experiment.slug,
            },
            "best": {
                "loss": {"epoch": best_epochs["loss"], "value": best_loss},
                "top1": {"epoch": best_epochs["top1"], "value": best_top1},
                "macro_f1": {
                    "epoch": best_epochs["macro_f1"],
                    "value": best_macro_f1,
                },
            },
            "artifacts": {
                "best_top1": resolve_artifact_path(
                    experiment.checkpoint_dir,
                    "pretrain_best.pt",
                ),
                "best_loss": resolve_artifact_path(
                    experiment.checkpoint_dir,
                    "pretrain_best_loss.pt",
                ),
                "best_macro_f1": resolve_artifact_path(
                    experiment.checkpoint_dir,
                    "pretrain_best_macro_f1.pt",
                ),
                "last": resolve_artifact_path(
                    experiment.checkpoint_dir,
                    "pretrain_last.pt",
                ),
            },
        },
    )

    log.info(
        "Pretraining complete. Best top1 epoch=%d top1=%.4f | best macro_f1 epoch=%d macro_f1=%.4f | best loss epoch=%d loss=%.4f",
        best_epochs["top1"],
        best_top1,
        best_epochs["macro_f1"],
        best_macro_f1,
        best_epochs["loss"],
        best_loss,
    )
    log.info("Experiment checkpoints: %s", experiment.checkpoint_dir)
    log.info("Experiment logs: %s", experiment.log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain the Stage 1 ActionClassifier on ANUBIS"
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/module1/baseline_transformer.yaml",
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()
    main(args.config)
