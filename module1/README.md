# Module 1 - Biomechanical Risk Monitor

Football player injury risk prediction from video using pose estimation and graph neural networks.

## Structure

```
module1/
├── data/           # Data loading and preprocessing
├── pose/           # Pose extraction (MediaPipe / ViTPose)
├── models/         # GNN, ST-GCN, TCN, Transformer models
└── evaluation/     # Evaluation metrics and analysis
```

## Quick Start

1. Add football clips to `data/football/clips/`
2. Create annotations in `data/football/annotations/`
3. Run pose extraction
4. Train model
5. Evaluate on real footage

See `docs/module1_restart_plan.md` for the full plan.
