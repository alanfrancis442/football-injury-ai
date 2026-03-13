# module1/pretrain/evaluate.py
#
# Stage 1 — Pretraining Evaluation
# ──────────────────────────────────
# Loads a pretrained ActionClassifier checkpoint, computes aggregate metrics,
# and saves detailed diagnostics (classification report, confusion matrix,
# hardest confusion pairs, and optional t-SNE embeddings) into the experiment
# log directory.

from __future__ import annotations

import argparse
import inspect
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from module1.data.anubis_loader import build_dataloaders
from module1.data.joint_config import EDGE_INDEX
from module1.pretrain.experiments import (
    ensure_experiment_dirs,
    prepare_experiment_config,
    save_json,
)
from module1.pretrain.live_utils import load_class_names, resolve_default_label_map_path
from module1.pretrain.model import ActionClassifier, build_model
from utils.config import load_config
from utils.logger import get_logger

log = get_logger(__name__)


# ── Prediction Collection ─────────────────────────────────────────────────────


@torch.no_grad()
def collect_predictions(
    model: ActionClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect embeddings, logits, predictions, and labels for one split."""
    model.eval()
    all_embeddings: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for x, y in tqdm(loader, desc="  eval ", leave=False, unit="batch"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        emb = model.encoder(x)
        logits = model.head(emb)
        preds = logits.argmax(dim=1)

        all_embeddings.append(emb.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    return (
        np.concatenate(all_embeddings, axis=0),
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_labels, axis=0),
    )


def compute_accuracy_metrics(
    logits: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute top-k accuracy and F1 metrics from collected predictions."""
    top1 = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    weighted_f1 = float(f1_score(labels, preds, average="weighted"))

    topk = min(5, logits.shape[1])
    top5_idx = np.argpartition(logits, kth=logits.shape[1] - topk, axis=1)[:, -topk:]
    top5_acc = float(np.any(top5_idx == labels[:, None], axis=1).mean())

    return {
        "top1_acc": top1,
        "top5_acc": top5_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# ── Diagnostics ───────────────────────────────────────────────────────────────


def _top_confusions(
    matrix: np.ndarray,
    class_names: List[str],
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """Return the strongest off-diagonal confusion pairs."""
    pairs: List[Dict[str, Any]] = []
    for src in range(matrix.shape[0]):
        for dst in range(matrix.shape[1]):
            if src == dst or matrix[src, dst] <= 0:
                continue
            pairs.append(
                {
                    "count": int(matrix[src, dst]),
                    "source_index": src,
                    "source_label": class_names[src],
                    "target_index": dst,
                    "target_label": class_names[dst],
                }
            )
    pairs.sort(key=lambda item: item["count"], reverse=True)
    return pairs[:limit]


def save_diagnostics(
    log_dir: str,
    class_names: List[str],
    preds: np.ndarray,
    labels: np.ndarray,
    metrics: Dict[str, float],
    split: str,
    ckpt: Dict,
) -> None:
    """Persist evaluation diagnostics into the experiment log directory."""
    os.makedirs(log_dir, exist_ok=True)

    report_dict = classification_report(
        labels,
        preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
    )
    report_text = classification_report(
        labels,
        preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
    )
    matrix = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    confusions = _top_confusions(matrix, class_names)

    with open(
        os.path.join(log_dir, f"{split}_classification_report.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(report_text))

    with open(
        os.path.join(log_dir, f"{split}_classification_report.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report_dict, f, indent=2, sort_keys=True)

    np.save(os.path.join(log_dir, f"{split}_confusion_matrix.npy"), matrix)
    save_json(
        os.path.join(log_dir, f"{split}_top_confusions.json"), {"pairs": confusions}
    )
    save_json(
        os.path.join(log_dir, f"{split}_evaluation_summary.json"),
        {
            "split": split,
            "metrics": metrics,
            "checkpoint": {
                "epoch": int(ckpt.get("epoch", -1)),
                "experiment": ckpt.get("experiment", {}),
            },
        },
    )


# ── t-SNE ────────────────────────────────────────────────────────────────────


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    n_samples: int = 2000,
    perplexity: float = 30.0,
) -> None:
    """Compute a t-SNE plot from a random embedding subset."""
    N = len(labels)
    if N > n_samples:
        idx = np.random.choice(N, n_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    log.info("Running t-SNE on %d samples (d=%d) ...", len(labels), embeddings.shape[1])
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "random_state": 42,
    }
    if "max_iter" in inspect.signature(TSNE.__init__).parameters:
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000

    tsne = TSNE(**tsne_kwargs)
    coords = tsne.fit_transform(embeddings)

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
    ax.set_title("t-SNE of Stage 1 embeddings (ANUBIS)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_xticks([])
    ax.set_yticks([])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("t-SNE plot saved → %s", out_path)


# ── Checkpoint Loading ────────────────────────────────────────────────────────


def load_checkpoint(
    ckpt_path: str,
    classifier: ActionClassifier,
    device: torch.device,
) -> Dict:
    """Load a checkpoint into the classifier and return the full payload."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\nRun module1/pretrain/train.py first."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("classifier") or ckpt.get("model")
    if state_dict is None:
        raise ValueError(
            "Checkpoint format not recognised. Expected keys 'classifier' or 'model'."
        )
    classifier.load_state_dict(state_dict)
    log.info(
        "Loaded checkpoint from %s (epoch=%s val_top1=%s)",
        ckpt_path,
        str(ckpt.get("epoch", "n/a")),
        str(ckpt.get("val_top1", "n/a")),
    )
    return ckpt


# ── Main Entry Point ──────────────────────────────────────────────────────────


def main(
    config_path: str,
    ckpt_path: str,
    split: str,
    run_tsne: bool,
) -> Dict[str, float]:
    raw_cfg = load_config(config_path)
    cfg, experiment = prepare_experiment_config(raw_cfg, config_path=config_path)
    ensure_experiment_dirs(experiment)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    edge_index = EDGE_INDEX.to(device)
    _, classifier = build_model(cfg, edge_index)
    classifier = classifier.to(device)
    ckpt = load_checkpoint(ckpt_path, classifier, device)
    classifier.eval()

    log.info("Loading data (split=%s) ...", split)
    train_loader, val_loader = build_dataloaders(cfg)
    loader = train_loader if split == "train" else val_loader

    class_names = load_class_names(
        int(cfg["data"]["num_classes"]),
        resolve_default_label_map_path(int(cfg["data"]["num_classes"])),
    )

    embeddings, logits, preds, labels = collect_predictions(classifier, loader, device)
    metrics = compute_accuracy_metrics(logits, preds, labels)

    log.info("Results:")
    log.info("  Top-1 accuracy : %.4f", metrics["top1_acc"])
    log.info("  Top-5 accuracy : %.4f", metrics["top5_acc"])
    log.info("  Macro F1       : %.4f", metrics["macro_f1"])
    log.info("  Weighted F1    : %.4f", metrics["weighted_f1"])

    save_diagnostics(
        log_dir=experiment.log_dir,
        class_names=class_names,
        preds=preds,
        labels=labels,
        metrics=metrics,
        split=split,
        ckpt=ckpt,
    )

    if run_tsne:
        tsne_path = os.path.join(experiment.log_dir, f"{split}_pretrain_tsne.png")
        plot_tsne(embeddings, labels, out_path=tsne_path)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained Stage 1 ActionClassifier on ANUBIS"
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/module1/baseline_transformer.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--ckpt",
        default="outputs/checkpoints/module1/baseline_transformer/pretrain_best.pt",
    )
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument(
        "--tsne",
        action="store_true",
        help="Generate a t-SNE scatter plot of encoder embeddings",
    )
    args = parser.parse_args()
    main(args.config, args.ckpt, args.split, args.tsne)
