# module1/transformer/train.py
#
# Stage 3 — Transformer Training Script
# ───────────────────────────────────────
# Trains the BiomechanicalTransformer on sequences of GNN feature vectors.
#
# Dataset format expected:
#   data/processed/train_sequences.pt  — list of dicts:
#       { 'features': tensor(50, 64), 'risk_label': int, 'region_label': int }
#
# The GNN must be trained first (module1/gnn/train.py) to produce the 64-dim
# feature vectors that become the input to this model.
#
# Usage:
#   python module1/transformer/train.py --config configs/module1.yaml
#
# Checkpoints → outputs/checkpoints/transformer_best.pt

import argparse
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from module1.transformer.model import BiomechanicalTransformer
from utils.config import load_config
from utils.logger import get_logger

log = get_logger(__name__)


# ── Dataset wrapper ────────────────────────────────────────────────────────────


class SequenceDataset(Dataset):
    """
    Wraps a list of sequence dicts into a PyTorch Dataset.

    Each item:
        features      : tensor (T, 64)  — GNN output sequence
        risk_label    : int  — 0 (no risk) or 1 (at risk)
        region_label  : int  — body region index (see BODY_REGIONS)
    """

    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "features": item["features"],
            "risk_label": torch.tensor(item["risk_label"], dtype=torch.long),
            "region_label": torch.tensor(item["region_label"], dtype=torch.long),
        }


def load_sequences(processed_dir: str, split: str) -> list:
    path = os.path.join(processed_dir, f"{split}_sequences.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sequence data not found at {path}\n"
            "Run notebook 03_model_experiments.ipynb to generate it."
        )
    return torch.load(path)


# ── Training helpers ───────────────────────────────────────────────────────────


def train_one_epoch(
    model, loader, optimiser, risk_criterion, region_criterion, device, region_weight
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        features = batch["features"].to(device)  # (B, T, 64)
        risk_labels = batch["risk_label"].to(device)  # (B,)
        region_labels = batch["region_label"].to(device)  # (B,)

        optimiser.zero_grad()
        out = model(features)

        # Combined loss: primary risk classification + secondary region prediction
        risk_loss = risk_criterion(
            out["risk_score"].squeeze(1).float(), risk_labels.float()
        )
        region_loss = region_criterion(out["region_logits"], region_labels)
        loss = risk_loss + region_weight * region_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss += loss.item() * features.size(0)
        preds = (out["risk_score"].squeeze(1) > 0.5).long()
        correct += (preds == risk_labels).sum().item()
        total += features.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, risk_criterion, region_criterion, device, region_weight):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        features = batch["features"].to(device)
        risk_labels = batch["risk_label"].to(device)
        region_labels = batch["region_label"].to(device)

        out = model(features)
        risk_loss = risk_criterion(
            out["risk_score"].squeeze(1).float(), risk_labels.float()
        )
        region_loss = region_criterion(out["region_logits"], region_labels)
        loss = risk_loss + region_weight * region_loss

        total_loss += loss.item() * features.size(0)
        preds = (out["risk_score"].squeeze(1) > 0.5).long()
        correct += (preds == risk_labels).sum().item()
        total += features.size(0)

    return total_loss / total, correct / total


def main(config_path: str):
    cfg = load_config(config_path)
    trans_cfg = cfg["transformer"]
    train_cfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_data = load_sequences(cfg["paths"]["processed_dir"], "train")
    val_data = load_sequences(cfg["paths"]["processed_dir"], "val")

    train_loader = DataLoader(
        SequenceDataset(train_data), batch_size=train_cfg["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        SequenceDataset(val_data), batch_size=train_cfg["batch_size"]
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BiomechanicalTransformer(
        input_dim=cfg["gnn"]["out_dim"],
        d_model=trans_cfg["d_model"],
        nhead=trans_cfg["nhead"],
        num_layers=trans_cfg["num_layers"],
        dim_ff=trans_cfg["dim_feedforward"],
        dropout=trans_cfg["dropout"],
        seq_len=trans_cfg["seq_len"],
    ).to(device)

    # ── Loss and optimiser ────────────────────────────────────────────────────
    # BCELoss for binary risk classification
    risk_criterion = nn.BCELoss()
    # CrossEntropy for multi-class region prediction
    region_criterion = nn.CrossEntropyLoss()

    optimiser = Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=train_cfg["epochs"])

    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    region_weight = trans_cfg.get("region_loss_weight", 0.3)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimiser,
            risk_criterion,
            region_criterion,
            device,
            region_weight,
        )
        val_loss, val_acc = evaluate(
            model, val_loader, risk_criterion, region_criterion, device, region_weight
        )
        scheduler.step()

        log.info(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(checkpoint_dir, "transformer_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
            log.info(f"  -> Saved best checkpoint: {ckpt_path}")

    log.info("Transformer training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/module1.yaml")
    args = parser.parse_args()
    main(args.config)
