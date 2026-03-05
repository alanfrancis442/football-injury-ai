# module1/gnn/train.py
#
# Stage 2 — GNN Training Script
# ──────────────────────────────
# Trains the SkeletonGAT model on a dataset of labelled skeleton sequences.
#
# Dataset format expected:
#   data/processed/<split>_graphs.pt  — a list of torch_geometric Data objects
#   Each Data object has .x (33,10), .edge_index (2,E), .y (label: 0 or 1)
#
# Usage:
#   python module1/gnn/train.py --config configs/module1.yaml
#
# Checkpoints are saved to:
#   outputs/checkpoints/gnn_epoch_<n>.pt
#
# Tips for experimenting:
#   - Change model architecture in module1/gnn/model.py
#   - Change hyperparameters in configs/module1.yaml (never hardcode them here)
#   - Keep this file as the stable training loop — it should rarely need to change

import argparse
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from module1.gnn.model import SkeletonGAT
from utils.config import load_config
from utils.logger import get_logger

log = get_logger(__name__)


def load_dataset(processed_dir: str, split: str):
    """Load a pre-built list of PyG Data objects from disk."""
    path = os.path.join(processed_dir, f"{split}_graphs.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}\n"
            "Run the pose extraction and graph-building notebook first:\n"
            "  notebooks/02_graph_exploration.ipynb"
        )
    return torch.load(path)


def train_one_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        optimiser.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)  # (B, 64)
        # Classification head — add a simple linear layer on top of the 64-dim features
        # (In this training script the model output is the 64-dim spatial vector;
        #  the final classifier is the risk_head defined below)
        logits = model.risk_head(out)  # (B, 2)
        loss = criterion(logits, batch.y)

        loss.backward()
        optimiser.step()

        total_loss += loss.item() * batch.num_graphs
        preds = logits.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total += batch.num_graphs

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        logits = model.risk_head(out)
        loss = criterion(logits, batch.y)

        total_loss += loss.item() * batch.num_graphs
        preds = logits.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total += batch.num_graphs

    return total_loss / total, correct / total


def main(config_path: str):
    cfg = load_config(config_path)
    gnn_cfg = cfg["gnn"]
    train_cfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_data = load_dataset(cfg["paths"]["processed_dir"], "train")
    val_data = load_dataset(cfg["paths"]["processed_dir"], "val")

    train_loader = DataLoader(
        train_data, batch_size=train_cfg["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_data, batch_size=train_cfg["batch_size"], shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SkeletonGAT(
        in_channels=gnn_cfg["in_channels"],
        hidden_dim=gnn_cfg["hidden_dim"],
        out_dim=gnn_cfg["out_dim"],
        heads=gnn_cfg["heads"],
        dropout=gnn_cfg["dropout"],
    ).to(device)

    # Attach a simple 2-class risk head (healthy vs. at-risk)
    model.risk_head = nn.Linear(gnn_cfg["out_dim"], 2).to(device)

    # ── Training setup ────────────────────────────────────────────────────────
    # Focal loss weights — injury class (1) is rare, so we up-weight it
    class_weights = torch.tensor(
        [train_cfg["class_weight_healthy"], train_cfg["class_weight_injury"]],
        dtype=torch.float,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = Adam(
        list(model.parameters()) + list(model.risk_head.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimiser, mode="min", patience=5, factor=0.5)

    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimiser, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        log.info(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.3f}"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(checkpoint_dir, "gnn_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "risk_head": model.risk_head.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
            log.info(f"  -> Saved best checkpoint: {ckpt_path}")

        # Periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"gnn_epoch_{epoch:03d}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_path)

    log.info("GNN training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/module1.yaml")
    args = parser.parse_args()
    main(args.config)
