# module1/pretrain/evaluate.py
#
# Stage 1 — Pretraining Evaluation
# ──────────────────────────────────
# Loads a pretrained ActionClassifier checkpoint and computes:
#   • Top-1 and top-5 accuracy
#   • Per-class F1 score (macro and per-class breakdown)
#   • t-SNE visualisation of CLS embeddings coloured by action class
#
# Usage:
#   python module1/pretrain/evaluate.py \
#       --config  configs/pretrain.yaml \
#       --ckpt    outputs/checkpoints/pretrain/pretrain_best.pt \
#       --split   val          # or 'train'
#       --tsne                 # add this flag to generate the t-SNE plot

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from module1.data.anubis_loader import (
    build_dataloaders,
    load_anubis_tensors,
    ANUBISDataset,
)
from module1.data.joint_config import EDGE_INDEX
from module1.pretrain.model import ActionClassifier, build_model
from utils.config import load_config
from utils.logger import get_logger

log = get_logger(__name__)


# ── Metric computation ────────────────────────────────────────────────────────


@torch.no_grad()
def collect_predictions(
    model: ActionClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the model over a DataLoader and collect logits, predictions, and labels.

    Returns
    -------
    embeddings : np.ndarray  shape (N, d_model)   CLS embeddings
    preds      : np.ndarray  shape (N,)            top-1 predicted class
    labels     : np.ndarray  shape (N,)            ground-truth class
    """
    model.eval()
    all_embeddings: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for x, y in tqdm(loader, desc="  eval ", leave=False, unit="batch"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # CLS embedding from encoder
        emb = model.encoder(x)  # (B, d_model)
        logits = model.head(emb)  # (B, num_classes)
        preds = logits.argmax(dim=1)  # (B,)

        all_embeddings.append(emb.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return embeddings, preds, labels


def compute_accuracy_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    logits_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute top-1 accuracy and macro-averaged F1.

    Parameters
    ----------
    preds  : shape (N,) predicted class indices
    labels : shape (N,) ground-truth class indices

    Returns
    -------
    metrics : dict with keys 'top1_acc' and 'macro_f1'
    """
    top1 = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    return {"top1_acc": top1, "macro_f1": macro_f1}


def compute_top5_accuracy(
    model: ActionClassifier,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute top-5 accuracy by re-running the model.

    Returns
    -------
    top5_acc : float  in [0, 1]
    """
    model.eval()
    total_top5 = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="  top5 ", leave=False, unit="batch"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            top5 = logits.topk(k=min(5, logits.size(1)), dim=1).indices
            total_top5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)

    return total_top5 / total


# ── t-SNE visualisation ───────────────────────────────────────────────────────


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    n_samples: int = 2000,
    perplexity: float = 30.0,
) -> None:
    """
    Compute t-SNE on a random subset of CLS embeddings and save a scatter plot.

    Parameters
    ----------
    embeddings : np.ndarray  shape (N, D)
    labels     : np.ndarray  shape (N,)   integer class indices
    out_path   : str         path to save the PNG
    n_samples  : int         max samples to subsample (t-SNE is O(n²))
    perplexity : float       t-SNE perplexity parameter
    """
    N = len(labels)
    if N > n_samples:
        idx = np.random.choice(N, n_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    log.info("Running t-SNE on %d samples (d=%d) …", len(labels), embeddings.shape[1])
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(embeddings)  # (N, 2)

    num_classes = int(labels.max()) + 1
    cmap = plt.cm.get_cmap("tab20", num_classes)

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap=cmap,
        s=6,
        alpha=0.7,
    )
    plt.colorbar(scatter, ax=ax, label="Action class")
    ax.set_title("t-SNE of SpatioTemporalEncoder CLS embeddings (ANUBIS)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_xticks([])
    ax.set_yticks([])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("t-SNE plot saved → %s", out_path)


# ── Checkpoint loading ────────────────────────────────────────────────────────


def load_checkpoint(
    ckpt_path: str,
    classifier: ActionClassifier,
    device: torch.device,
) -> Dict:
    """
    Load a pretrain checkpoint into the classifier.

    Parameters
    ----------
    ckpt_path  : str
    classifier : ActionClassifier  (must be already constructed)
    device     : torch.device

    Returns
    -------
    ckpt : dict   full checkpoint dict (contains epoch, metrics, etc.)
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\nRun module1/pretrain/train.py first."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    classifier.load_state_dict(ckpt["classifier"])
    log.info(
        "Loaded checkpoint from %s  (epoch=%d  val_top1=%.3f)",
        ckpt_path,
        ckpt.get("epoch", -1),
        ckpt.get("val_top1", float("nan")),
    )
    return ckpt


# ── Main evaluation entry point ───────────────────────────────────────────────


def main(config_path: str, ckpt_path: str, split: str, run_tsne: bool) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Build model and load weights ─────────────────────────────────────
    edge_index = EDGE_INDEX.to(device)
    encoder, classifier = build_model(cfg, edge_index)
    classifier = classifier.to(device)
    load_checkpoint(ckpt_path, classifier, device)
    classifier.eval()

    # ── DataLoader ───────────────────────────────────────────────────────
    log.info("Loading data (split=%s) …", split)
    _, val_loader = build_dataloaders(cfg)
    loader = val_loader  # extend here if split='train' is needed

    # ── Collect predictions ──────────────────────────────────────────────
    embeddings, preds, labels = collect_predictions(classifier, loader, device)

    # ── Metrics ──────────────────────────────────────────────────────────
    metrics = compute_accuracy_metrics(preds, labels)
    top5 = compute_top5_accuracy(classifier, loader, device)
    metrics["top5_acc"] = top5

    log.info("Results:")
    log.info("  Top-1 accuracy : %.4f", metrics["top1_acc"])
    log.info("  Top-5 accuracy : %.4f", metrics["top5_acc"])
    log.info("  Macro F1       : %.4f", metrics["macro_f1"])

    # Detailed per-class report (printed to stdout)
    print("\nPer-class classification report (F1, precision, recall):")
    print(classification_report(labels, preds, zero_division=0))

    # ── t-SNE ────────────────────────────────────────────────────────────
    if run_tsne:
        tsne_path = os.path.join(cfg["paths"]["log_dir"], "pretrain_tsne.png")
        plot_tsne(embeddings, labels, out_path=tsne_path)

    return metrics


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained SpatioTemporalEncoder on ANUBIS"
    )
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument(
        "--ckpt", default="outputs/checkpoints/pretrain/pretrain_best.pt"
    )
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument(
        "--tsne",
        action="store_true",
        help="Generate t-SNE scatter plot of CLS embeddings",
    )
    args = parser.parse_args()
    main(args.config, args.ckpt, args.split, args.tsne)
