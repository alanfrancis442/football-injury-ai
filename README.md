# Football Player Injury Risk Prediction — Module 1

Biomechanical Risk Monitor using Pose Estimation, GNN, and Transformer.

---

## Project Structure

```
football-injury-ai/
├── module1/            # Biomechanical Risk Monitor (active build)
│   ├── pose/           # Stage 1 — keypoint extraction & player tracking
│   ├── data/           # Data loading & preprocessing
│   └── pretrain/       # Training, evaluation & tuning scripts
├── module2/            # Placeholder — Acute Workload Monitor
├── module3/            # Placeholder — Historical Risk Profile
├── fusion/             # Placeholder — Risk Fusion Engine
├── docs/               # Documentation
│   ├── COMMANDS.md     # All commands reference
│   └── LABELING_GUIDE.md  # How to label data
├── data/
│   ├── raw/            # Original video clips / downloaded datasets
│   ├── processed/      # Extracted keypoints (.npy / .csv)
│   ├── samples/        # Short clips for fast dev iteration
│   └── anubis/         # ANUBIS dataset files
├── notebooks/          # Jupyter experimentation sandbox
├── tests/              # Unit + integration tests
├── configs/            # Hyperparameters (edit here, not in code)
└── outputs/
    ├── checkpoints/    # Saved model weights (.pt)
    └── logs/          # Training logs
```

---

## Quick Links

| Guide | Description |
|-------|-------------|
| [docs/COMMANDS.md](docs/COMMANDS.md) | All commands for training, testing, evaluation |
| [docs/LABELING_GUIDE.md](docs/LABELING_GUIDE.md) | How to label football clips (for non-tech friends) |

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Install PyTorch Geometric extras

CPU:
```bash
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

GPU (adjust CUDA version):
```bash
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

---

## Common Commands

```bash
# Train ANUBIS pretraining
uv run module1/pretrain/train.py --config configs/pretrain.yaml

# Evaluate
uv run module1/pretrain/evaluate.py --config configs/pretrain.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt

# Run Optuna tuning
uv run module1/pretrain/tune_optuna.py --config configs/pretrain.yaml --n-trials 30

# Run tests
uv run pytest tests/

# Check data stats
uv run scripts/anubis_label_stats.py
uv run scripts/anubis_feature_stats.py
```

See [docs/COMMANDS.md](docs/COMMANDS.md) for the full command list.

---

## Development Workflow

1. **Experiment** in `notebooks/` — break things freely here
2. **Port clean code** into the matching `module1/` file once it works
3. **Adjust hyperparameters** in `configs/pretrain.yaml` — never hardcode them
4. **Verify nothing broke** with `uv run pytest tests/`
