# Football Player Injury Risk Prediction — Module 1

Biomechanical Risk Monitor using Pose Estimation, GNN, and Transformer.

---

## Project Structure

```
football-injury-ai/
├── module1/            # Biomechanical Risk Monitor (active build)
│   ├── pose/           # Stage 1 — keypoint extraction & player tracking
│   ├── gnn/            # Stage 2 — Graph Attention Network (spatial)
│   ├── transformer/    # Stage 3 — Transformer (temporal)
│   └── pipeline.py     # End-to-end: video → risk score
├── module2/            # Placeholder — Acute Workload Monitor
├── module3/            # Placeholder — Historical Risk Profile
├── fusion/             # Placeholder — Risk Fusion Engine
├── data/
│   ├── raw/            # Original video clips / downloaded datasets
│   ├── processed/      # Extracted keypoints (.npy / .csv)
│   └── samples/        # Short clips for fast dev iteration
├── notebooks/          # Jupyter experimentation sandbox
├── tests/              # Unit + integration tests
├── configs/            # Hyperparameters (edit here, not in code)
└── outputs/
    ├── checkpoints/    # Saved model weights (.pt)
    └── logs/           # Training logs
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install PyTorch Geometric extras (required for GNN)

Find the right command for your CUDA version at:
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

CPU-only (for development):
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

---

## Running the Pipeline

### Run on a single video clip
```bash
python module1/pipeline.py --video data/samples/clip.mp4
```

### Extract poses only
```bash
python module1/pose/extractor.py --video data/samples/clip.mp4 --out data/processed/
```

### Train the GNN
```bash
python module1/gnn/train.py --config configs/module1.yaml
```

### Train the Transformer
```bash
python module1/transformer/train.py --config configs/module1.yaml
```

### Run tests
```bash
pytest tests/
```

---

## Development Workflow

1. **Experiment** in `notebooks/` — break things freely here
2. **Port clean code** into the matching `module1/` file once it works
3. **Adjust hyperparameters** in `configs/module1.yaml` — never hardcode them
4. **Verify nothing broke** with `pytest tests/`
