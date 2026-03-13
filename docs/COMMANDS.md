# Football Injury AI — Command Reference

This guide covers all commands you need to train, evaluate, and work with the project.

---

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repo
git clone https://github.com/alanfrancis442/football-injury-ai.git
cd football-injury-ai

# Install all dependencies
uv sync

# Install PyTorch Geometric extras (CPU)
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# Or for GPU (adjust CUDA version)
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

---

## Data Setup

### Download ANUBIS Dataset

Place these files in `data/anubis/`:

```
data/anubis/
├── trn_features_cleaned.npy   # Training features
├── trn_labels_cleaned.npy     # Training labels
├── val_features_cleaned.npy   # Validation features
└── val_labels_cleaned.npy     # Validation labels
```

### Check Data Stats

```bash
# Label distribution
uv run scripts/anubis_label_stats.py

# Feature distribution
uv run scripts/anubis_feature_stats.py
```

---

## Training (Stage 1: ANUBIS Pretraining)

### Transformer Baseline

```bash
uv run module1/pretrain/train.py --config configs/experiments/module1/baseline_transformer.yaml
```

### TCN Ablation

```bash
uv run module1/pretrain/train.py --config configs/experiments/module1/tcn_ablation.yaml
```

### Hyperparameter Tuning (Optuna)

```bash
# Run 30 trials
uv run module1/pretrain/tune_optuna.py --config configs/experiments/module1/baseline_transformer.yaml --n-trials 30

# Run with custom timeout (e.g., 2 hours)
uv run module1/pretrain/tune_optuna.py --config configs/experiments/module1/baseline_transformer.yaml --timeout-sec 7200
```

### Train with Best Tuned Parameters

```bash
uv run module1/pretrain/train.py --config outputs/logs/module1/baseline_transformer/pretrain_optuna_best.yaml
```

---

## Evaluation

### Evaluate a Checkpoint

```bash
# Basic evaluation
uv run module1/pretrain/evaluate.py --config configs/experiments/module1/baseline_transformer.yaml --ckpt outputs/checkpoints/module1/baseline_transformer/pretrain_best.pt

# With t-SNE visualization
uv run module1/pretrain/evaluate.py --config configs/experiments/module1/baseline_transformer.yaml --ckpt outputs/checkpoints/module1/baseline_transformer/pretrain_best.pt --tsne

# TCN ablation evaluation
uv run module1/pretrain/evaluate.py --config configs/experiments/module1/tcn_ablation.yaml --ckpt outputs/checkpoints/module1/tcn_ablation/pretrain_best.pt --split val --tsne
```

### Checkpoint Locations

| Checkpoint | Path |
|------------|------|
| Baseline best (by val top-1) | `outputs/checkpoints/module1/baseline_transformer/pretrain_best.pt` |
| Baseline best macro F1 | `outputs/checkpoints/module1/baseline_transformer/pretrain_best_macro_f1.pt` |
| Baseline best loss | `outputs/checkpoints/module1/baseline_transformer/pretrain_best_loss.pt` |
| Baseline periodic | `outputs/checkpoints/module1/baseline_transformer/pretrain_ep{N:04d}.pt` |
| TCN best (by val top-1) | `outputs/checkpoints/module1/tcn_ablation/pretrain_best.pt` |
| Optuna best config | `outputs/logs/module1/baseline_transformer/pretrain_optuna_best.yaml` |

---

## Testing

### Live Inference (Webcam / Video / RTSP)

This tests the pretrained ANUBIS action model on live footage.

Important: config and checkpoint architecture must match.
- Transformer checkpoint -> `configs/experiments/module1/baseline_transformer.yaml`
- TCN checkpoint -> `configs/experiments/module1/tcn_ablation.yaml`

```bash
# Webcam (index 0)
uv run module1/pretrain/live_inference.py --config configs/experiments/module1/baseline_transformer.yaml --ckpt outputs/checkpoints/module1/baseline_transformer/pretrain_best.pt --source 0

# Webcam (TCN checkpoint)
uv run module1/pretrain/live_inference.py --config configs/experiments/module1/tcn_ablation.yaml --ckpt outputs/checkpoints/module1/tcn_ablation/pretrain_best.pt --source 0

# Video file
uv run module1/pretrain/live_inference.py --config configs/experiments/module1/baseline_transformer.yaml --ckpt outputs/checkpoints/module1/baseline_transformer/pretrain_best.pt --source data/samples/clip.mp4

# RTSP stream
uv run module1/pretrain/live_inference.py --config configs/experiments/module1/baseline_transformer.yaml --ckpt outputs/checkpoints/module1/baseline_transformer/pretrain_best.pt --source "rtsp://<user>:<pass>@<ip>:554/stream"
```

Optional flags:

```bash
# Show top-3 classes only
uv run module1/pretrain/live_inference.py --source 0 --topk 3

# Stronger prediction smoothing
uv run module1/pretrain/live_inference.py --source 0 --smooth-alpha 0.85

# Use custom class names (.txt/.yaml/.json)
uv run module1/pretrain/live_inference.py --source 0 --label-map outputs/logs/anubis_class_names.txt

# Headless mode (no OpenCV window)
uv run module1/pretrain/live_inference.py --source data/samples/clip.mp4 --no-show
```

Live window now shows:
- stable displayed label with pause/background rejection
- top-k ranked action predictions
- skeleton pose overlay drawn from detected joints

By default, for 102-class ANUBIS models, live inference auto-loads
`module1/data/anubis_class_names.yaml` so action names are shown directly.

Exit live mode with `q` or `Esc`.

### Run All Tests

```bash
uv run pytest tests/
```

### Run Specific Test File

```bash
uv run pytest tests/test_model.py -v
uv run pytest tests/test_anubis_loader.py -v
uv run pytest tests/test_joint_config.py -v
uv run pytest tests/test_live_pipeline.py -v
uv run pytest tests/test_live_label_map.py -v
```

### Run Single Test

```bash
uv run pytest tests/test_model.py::TestSpatioTemporalEncoder::test_output_shape -v
```

---

## Model Smoke Tests

Quick sanity checks without data:

```bash
# Test model forward pass
uv run module1/pretrain/model.py
```

---

## Project Structure

```
football-injury-ai/
├── module1/
│   ├── data/              # Data loading & preprocessing
│   ├── pose/              # Live pose extraction (MediaPipe -> ANUBIS-32)
│   ├── pretrain/          # Training & evaluation scripts
│   │   ├── live_inference.py
│   │   └── live_utils.py
│   └── ...
├── configs/
│   ├── experiments/module1/baseline_transformer.yaml
│   ├── experiments/module1/tcn_ablation.yaml
│   └── module1.yaml       # Fine-tuning config
├── scripts/               # Utility scripts
│   ├── anubis_label_stats.py
│   └── anubis_feature_stats.py
├── docs/
│   ├── LABELING_GUIDE.md  # How to label data
│   └── COMMANDS.md        # This file
├── outputs/
│   ├── checkpoints/       # Saved models
│   └── logs/              # Training logs
└── tests/                 # Unit tests
```

---

## Common Tasks

### Check GPU Availability

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### View Training Logs

```bash
cat outputs/logs/module1/baseline_transformer/training_history.json
```

### Delete Old Checkpoints

```bash
rm outputs/checkpoints/module1/baseline_transformer/pretrain_ep*.pt
```

---

## Troubleshooting

### "No module named 'torch'"

```bash
uv sync
```

### "No module named 'cv2'" or "No module named 'mediapipe'"

```bash
uv sync
```

### "CUDA out of memory"

Reduce batch size in config:
```yaml
training:
  batch_size: 32  # Try 16 or 8
```

### "Checkpoint not found"

Make sure you're running from the project root directory:
```bash
cd football-injury-ai
uv run module1/pretrain/train.py --config configs/experiments/module1/baseline_transformer.yaml
```

---

## Next Steps

After ANUBIS pretraining:

1. **Collect football clips** — see `docs/LABELING_GUIDE.md`
2. **Label the clips** — use Label Studio
3. **Fine-tune on football data** — see future updates

---

For questions, check the main README.md or open an issue on GitHub.
