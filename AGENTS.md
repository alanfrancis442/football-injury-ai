# AGENTS.md — Football Injury AI

Guidance for agentic coding agents working in this repository.

---

## Project Overview

Python deep-learning research project: real-time football player injury risk
prediction. Module 1 (active) processes video → MediaPipe pose keypoints →
Graph Attention Network (spatial) → Transformer (temporal) → risk score + body
region. Modules 2, 3, and the Fusion Engine are placeholders for future work.

Stack: Python 3, PyTorch, PyTorch Geometric, MediaPipe, OpenCV, NumPy, pandas,
PyYAML, pytest. No JavaScript, no frontend.

---

## Commands

### Install dependencies
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# PyTorch Geometric extras (CPU dev):
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

### Run all tests
```bash
pytest tests/
```

### Run a single test file
```bash
pytest tests/test_graph.py -v
```

### Run a single test class
```bash
pytest tests/test_graph.py::TestBuildGraph -v
```

### Run a single test function
```bash
pytest tests/test_graph.py::TestBuildGraph::test_node_feature_shape -v
```

### Quick per-module smoke test (no pytest required)
```bash
python module1/gnn/model.py
python module1/transformer/model.py
python module1/gnn/graph.py
```

### Full pipeline
```bash
python module1/pipeline.py --video data/samples/clip.mp4
```

### Train GNN / Transformer
```bash
python module1/gnn/train.py --config configs/module1.yaml
python module1/transformer/train.py --config configs/module1.yaml
```

### No linter or formatter is configured. No CI/CD exists.

---

## Repository Layout

```
module1/         # Active module — Biomechanical Risk Monitor
  pose/          # Stage 1: keypoint extraction + IoU tracker
  gnn/           # Stage 2: skeleton graph + SkeletonGAT model + training
  transformer/   # Stage 3: BiomechanicalTransformer model + training
  pipeline.py    # End-to-end orchestrator (CLI entry point)
module2/         # Placeholder — Acute Workload Monitor
module3/         # Placeholder — Historical Risk Profile
fusion/          # Placeholder — Risk Fusion Engine
utils/           # config.py (YAML loader), logger.py (centralised logging)
configs/         # module1.yaml — ALL hyperparameters live here
tests/           # pytest unit + integration tests
notebooks/       # Jupyter exploration (not production code)
data/            # raw/, processed/, samples/ (all gitignored)
outputs/         # checkpoints/, logs/ (all gitignored)
```

---

## Code Style

### File headers
Every `.py` file begins with a comment block (not a docstring):
```python
# module1/gnn/model.py
#
# Stage 2 — Graph Attention Network (GAT) Model
# ──────────────────────────────────────────────
# One-paragraph description of purpose, inputs, outputs.
```

### Imports
Order: stdlib → third-party → local. One blank line between groups.
Use `from __future__ import annotations` at the top of files that need
forward references. No `__all__` declarations; all `__init__.py` files are empty.
```python
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

from utils.config import load_config
from utils.logger import get_logger
```

### Type hints
Always annotate function signatures. Use `typing` module style (Python <3.10):
`Dict`, `List`, `Optional`, `Tuple` — not `dict[...]`, `list[...]` builtins.
```python
def build_graph(seq: np.ndarray, label: Optional[int] = None) -> Data:
```

### Docstrings
NumPy style for all public classes and methods:
```python
def extract_from_video(self, video_path: str) -> np.ndarray:
    """
    Process a full video file and return all keyframes as a numpy array.

    Parameters
    ----------
    video_path : str — path to the video file

    Returns
    -------
    keypoints : np.ndarray  shape (n_frames, 33, 4)
    """
```

### Naming conventions
| Kind | Style | Example |
|---|---|---|
| Classes | PascalCase | `SkeletonGAT`, `BiomechanicalTransformer`, `PlayerRiskOutput` |
| Functions / methods | snake_case | `build_graph`, `train_one_epoch`, `score_to_tier` |
| Private methods / attrs | `_name` | `_process_frame`, `_next_id`, `_gnn_history` |
| Module-level constants | SCREAMING_SNAKE_CASE | `SKELETON_EDGES`, `BODY_REGIONS`, `NUM_KEYPOINTS` |
| Logger instance | `log` | `log = get_logger(__name__)` |
| Config dict variable | `cfg` | `cfg = load_config(...)` |

### Section separators
Use banner comments to divide logical sections within a file:
```python
# ── Stage 1: Pose Extraction ──────────────────────────────────────────────────
```

### Logging
Use the centralised logger everywhere except CLI `__main__` blocks:
```python
log = get_logger(__name__)   # at module top level, always "log" not "logger"
log.info("Training epoch %d", epoch)
```
`print()` is only acceptable inside `if __name__ == "__main__":` blocks and
the `extractor.py` save CLI.

### Error handling
Raise specific exceptions with descriptive messages and actionable hints.
Never use bare `except:`.
```python
if not os.path.exists(path):
    raise FileNotFoundError(
        f"Config not found: {path}\n"
        "Make sure you are running commands from the project root directory."
    )
```

### PyTorch conventions
- British spelling: `optimiser` (not `optimizer`)
- Use `@torch.no_grad()` decorator on eval functions, not the context manager
- Call `.eval()` before inference, `.train()` at start of training loops
- Device detection: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Gradient clipping: `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Checkpoint dicts always include `"epoch"`, `"model"` (state_dict), and metric keys
- Activation: ELU in GNN layers; standard transformer defaults elsewhere

### Hyperparameters
Never hardcode in Python files. All values live in `configs/module1.yaml`.
Access via `cfg["section"]["key"]`:
```python
cfg = load_config("configs/module1.yaml")
lr  = cfg["training"]["learning_rate"]
```

### Smoke tests in model files
Each model and graph file includes an `if __name__ == "__main__":` block that
runs a forward pass with dummy tensors and prints shape assertions. This is the
fastest way to verify a model change without running the full test suite.

### Dataclasses
Use `@dataclass` for output containers. Add inline comments on field types.

### Context managers
Implement `__enter__` / `__exit__` on classes that hold external resources
(e.g., `PoseExtractor` wraps a MediaPipe `Pose` object).

---

## Tests

Test files mirror the source they test: `tests/test_graph.py` → `module1/gnn/graph.py`.

Use classes to group related tests, and `@pytest.fixture` for shared test data:
```python
@pytest.fixture
def dummy_seq():
    """Random keypoint sequence — shape (50, 33, 4)."""
    return np.random.rand(50, 33, 4).astype(np.float32)

class TestBuildGraph:
    def test_node_feature_shape(self, dummy_seq):
        graph = build_graph(dummy_seq)
        assert graph.x.shape == (33, 10)
```

Each test file begins with a comment block identifying what it tests and how
to run it: `# Run with: pytest tests/test_graph.py -v`.

---

## Development Workflow

1. Experiment freely in `notebooks/` — break things here, not in source files
2. Port clean, typed, documented code into the matching `module1/` file
3. Adjust hyperparameters in `configs/module1.yaml` — never in source code
4. Verify nothing broke: `pytest tests/`
5. Run the in-file smoke test for any model you changed: `python module1/gnn/model.py`
