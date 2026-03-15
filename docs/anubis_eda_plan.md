# ANUBIS Dataset EDA Plan

## Purpose

Validate all assumptions before preprocessing the ANUBIS skeleton action dataset for pretraining. This EDA is designed to run safely on a laptop with ~16 GB RAM.

## Dataset Location

Download with:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Khat865/ANUBIS-Skeleton",
    repo_type="dataset",
    local_dir="./anubis_dataset",
    allow_patterns=["anubis/*.npy", "anubis/*.pkl"],
)
```

Expected files after download:

```
anubis_dataset/
└── anubis/
    ├── trn_data_all_action_front.npy    # ~7.2 GB
    ├── trn_label_all_action_front.pkl
    ├── val_data_all_action_back.npy     # ~7.0 GB
    └── val_label_all_action_back.pkl
```

## Memory Safety Rules

- Always use `mmap_mode="r"` for loading `.npy` files — never load full arrays into RAM
- Process in batches of 128-256 samples
- Store only summary statistics, counters, and histograms
- For visualization, sample only a few clips (not thousands)

---

## EDA Checklist

### 1. Metadata and File Inspection

**Goal:** Confirm file existence, shapes, and basic properties.

```python
from pathlib import Path
import numpy as np
import pickle

root = Path("./anubis_dataset/anubis")

# Check files exist and get sizes
for f in root.iterdir():
    print(f.name, f.stat().st_size / 1e9, "GB")

# Load shapes (memory-safe)
x_trn = np.load(root / "trn_data_all_action_front.npy", mmap_mode="r")
x_val = np.load(root / "val_data_all_action_back.npy", mmap_mode="r")

print("train shape:", x_trn.shape, "dtype:", x_trn.dtype)
print("val shape:  ", x_val.shape, "dtype:", x_val.dtype)

# Load labels
with open(root / "trn_label_all_action_front.pkl", "rb") as f:
    try:
        trn_names, trn_labels = pickle.load(f)
    except Exception:
        trn_names, trn_labels = pickle.load(f, encoding="latin1")

with open(root / "val_label_all_action_back.pkl", "rb") as f:
    try:
        val_names, val_labels = pickle.load(f)
    except Exception:
        val_names, val_labels = pickle.load(f, encoding="latin1")

print("\n--- Labels ---")
print("train samples:", len(trn_names), len(trn_labels))
print("train labels:  min", min(trn_labels), "max", max(trn_labels), "unique", len(set(trn_labels)))
print("val samples:  ", len(val_names), len(val_labels))
print("val labels:   min", min(val_labels), "max", max(val_labels), "unique", len(set(val_labels)))

# First few sample names
print("\nSample names (train):", trn_names[:5])
```

**Expected:**
- train: `(N, 3, 300, 32, 2)` — (samples, channels, frames, joints, persons)
- val: same shape
- 102 unique classes (0-101)

---

### 2. Class Distribution

**Goal:** Understand class balance and identify rare or missing classes.

```python
import numpy as np
from collections import Counter

# Class counts
trn_counts = Counter(trn_labels)
val_counts = Counter(val_labels)

print("--- Train class distribution ---")
print("most common:", trn_counts.most_common(5))
print("least common:", trn_counts.most_common()[-5:])

print("\n--- Val class distribution ---")
print("most common:", val_counts.most_common(5))
print("least common:", val_counts.most_common()[-5:])

# Check for missing classes in val
all_classes = set(range(102))
trn_classes = set(trn_counts.keys())
val_classes = set(val_counts.keys())

missing_in_val = all_classes - val_classes
missing_in_trn = all_classes - trn_classes

print("\nClasses missing in val:", len(missing_in_val), sorted(missing_in_val) if missing_in_val else "None")
print("Classes missing in trn:", len(missing_in_trn), sorted(missing_in_trn) if missing_in_trn else "None")
```

---

### 3. Integrity Scan

**Goal:** Find corrupted / empty clips — all zeros, NaN, inf, too few valid frames.

```python
import numpy as np

def scan_split(x, batch_size=256, desc=""):
    """Scan for corrupted clips."""
    n_samples = x.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    stats = {
        "all_zero": 0,
        "has_nan": 0,
        "has_inf": 0,
        "low_valid_frames": 0,
        "person0_empty": 0,
        "person1_empty": 0,
    }
    
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_samples)
        
        # Load one batch at a time
        batch = x[start:end]
        
        # Check each sample in batch
        for j in range(end - start):
            sample = batch[j]  # shape: (3, 300, 32, 2)
            
            # Person 0 and Person 1
            p0 = sample[:, :, :, 0]  # (3, 300, 32)
            p1 = sample[:, :, :, 1]  # (3, 300, 32)
            
            # All zero (person 0)
            if np.all(p0 == 0):
                stats["all_zero"] += 1
            
            # NaN / Inf
            if np.any(np.isnan(p0)) or np.any(np.isnan(p1)):
                stats["has_nan"] += 1
            if np.any(np.isinf(p0)) or np.any(np.isinf(p1)):
                stats["has_inf"] += 1
            
            # Valid frames (non-zero in at least one joint)
            valid_p0 = np.sum(np.any(p0 != 0, axis=(0, 2)))  # count frames with any non-zero
            if valid_p0 < 30:  # less than 10% of 300 frames
                stats["low_valid_frames"] += 1
            
            # Person occupancy
            if np.all(p0 == 0):
                stats["person0_empty"] += 1
            if np.all(p1 == 0):
                stats["person1_empty"] += 1
    
    # Print results
    print(f"\n{desc} Integrity Scan (n={n_samples})")
    for k, v in stats.items():
        pct = 100 * v / n_samples
        print(f"  {k}: {v} ({pct:.2f}%)")
    
    return stats

# Run on both splits
stats_trn = scan_split(x_trn, desc="Train")
stats_val = scan_split(x_val, desc="Val")
```

---

### 4. Valid Frame Statistics

**Goal:** Understand how many frames are actually valid per clip before resampling to 60.

```python
import numpy as np

def count_valid_frames(x, batch_size=256):
    """Count valid (non-zero) frames per sample."""
    n_samples = x.shape[0]
    valid_counts = []
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch = x[i:end, :, :, :, 0]  # person 0 only
        
        # Valid = at least one joint has non-zero values
        valid = np.any(batch != 0, axis=(1, 2))  # (batch, frames)
        valid_counts.append(np.sum(valid, axis=1))
    
    return np.concatenate(valid_counts)

print("Counting valid frames (train)...")
valid_trn = count_valid_frames(x_trn)
print("Counting valid frames (val)...")
valid_val = count_valid_frames(x_val)

print("\n--- Valid frame counts ---")
print("Train: min", valid_trn.min(), "max", valid_trn.max(), "mean", valid_trn.mean(), "median", np.median(valid_trn))
print("Val:   min", valid_val.min(), "max", valid_val.max(), "mean", valid_val.mean(), "median", np.median(valid_val))

# Histogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(valid_trn, bins=50, alpha=0.7)
axes[0].set_title("Train: Valid frames per clip")
axes[0].axvline(60, color='r', linestyle='--', label='target=60')
axes[0].legend()

axes[1].hist(valid_val, bins=50, alpha=0.7)
axes[1].set_title("Val: Valid frames per clip")
axes[1].axvline(60, color='r', linestyle='--', label='target=60')
axes[1].legend()

plt.tight_layout()
plt.savefig("anubis_valid_frames_hist.png", dpi=100)
print("\nSaved: anubis_valid_frames_hist.png")
```

---

### 5. Person 0 vs Person 1 Analysis

**Goal:** Decide whether dropping person 1 is safe or loses important data.

```python
import numpy as np
from collections import Counter

def analyze_person_occupancy(x, labels, names, batch_size=256):
    """Analyze person 1 usage."""
    n_samples = x.shape[0]
    
    both_present = 0
    only_p0 = 0
    only_p1 = 0
    both_empty = 0
    
    # Sample class-wise
    class_both = Counter()
    class_p0only = Counter()
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch = x[i:end]  # (batch, 3, 300, 32, 2)
        
        for j in range(end - i):
            p0 = batch[j, :, :, :, 0]
            p1 = batch[j, :, :, :, 1]
            
            p0_active = np.any(p0 != 0)
            p1_active = np.any(p1 != 0)
            
            label = labels[i + j]
            
            if p0_active and p1_active:
                both_present += 1
                class_both[label] += 1
            elif p0_active and not p1_active:
                only_p0 += 1
                class_p0only[label] += 1
            elif not p0_active and p1_active:
                only_p1 += 1
            else:
                both_empty += 1
    
    total = both_present + only_p0 + only_p1 + both_empty
    
    print(f"\nPerson Occupancy (n={total})")
    print(f"  Both present:  {both_present} ({100*both_present/total:.1f}%)")
    print(f"  Only P0:      {only_p0} ({100*only_p0/total:.1f}%)")
    print(f"  Only P1:      {only_p1} ({100*only_p1/total:.1f}%)")
    print(f"  Both empty:   {both_empty} ({100*both_empty/total:.1f}%)")
    
    print("\nClasses with most P1 usage (both present):")
    for label, count in class_both.most_common(10):
        print(f"  Class {label}: {count}")
    
    return {
        "both_present": both_present,
        "only_p0": only_p0,
        "only_p1": only_p1,
        "both_empty": both_empty,
    }

# Analyze both splits
print("=== TRAIN ===")
occ_trn = analyze_person_occupancy(x_trn, trn_labels, trn_names)

print("\n=== VAL ===")
occ_val = analyze_person_occupancy(x_val, val_labels, val_names)
```

---

### 6. Joint / Root Analysis

**Goal:** Validate that joint 0 is a stable root for pelvis-centering.

```python
import numpy as np

def analyze_joint_zero(x, batch_size=256, desc=""):
    """Analyze joint 0 (assumed pelvis) properties."""
    n_samples = x.shape[0]
    
    # Sample joint 0 for person 0
    joint0_vals = []
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch = x[i:end, :, :, 0, 0]  # (batch, 3, 300) — xyz, frames, joint 0
        
        joint0_vals.append(batch)
    
    joint0 = np.concatenate(joint0_vals, axis=0)  # (N, 3, 300)
    
    # Per-sample stats
    x_mean = joint0[:, 0, :].mean(axis=1)  # mean x per sample
    y_mean = joint0[:, 1, :].mean(axis=1)
    z_mean = joint0[:, 2, :].mean(axis=1)
    
    # Motion (std over frames)
    x_std = joint0[:, 0, :].std(axis=1)
    y_std = joint0[:, 1, :].std(axis=1)
    z_std = joint0[:, 2, :].std(axis=1)
    
    print(f"\n{desc} Joint 0 (Pelvis) Statistics")
    print(f"  Position mean: x={x_mean.mean():.3f}, y={y_mean.mean():.3f}, z={z_mean.mean():.3f}")
    print(f"  Position std:  x={x_mean.std():.3f}, y={y_mean.std():.3f}, z={z_mean.std():.3f}")
    print(f"  Motion (frame std) mean: x={x_std.mean():.3f}, y={y_std.mean():.3f}, z={z_std.mean():.3f}")
    
    # Check for unstable values
    unstable = np.sum((x_std > 10) | (y_std > 10) | (z_std > 10))
    print(f"  Unstable clips (std>10): {unstable} ({100*unstable/len(x_std):.2f}%)")
    
    return {"x_mean": x_mean, "y_mean": y_mean, "z_mean": z_mean}

_ = analyze_joint_zero(x_trn, desc="Train")
_ = analyze_joint_zero(x_val, desc="Val")
```

---

### 7. Train vs Val Shift (Front vs Back View)

**Goal:** Quantify the domain shift between train (front) and val (back) viewpoints.

```python
import numpy as np

def compare_splits(x_trn, x_val, batch_size=256):
    """Compare distributions between train and val."""
    
    # Sample a subset for comparison
    sample_size = min(5000, x_trn.shape[0], x_val.shape[0])
    idx_trn = np.random.choice(x_trn.shape[0], sample_size, replace=False)
    idx_val = np.random.choice(x_val.shape[0], sample_size, replace=False)
    
    # Compare coordinate ranges
    def get_stats(x, idx):
        batch = x[idx, :, :, :, 0]  # person 0
        return {
            "x_min": batch[:, 0, :, :].min(),
            "x_max": batch[:, 0, :, :].max(),
            "y_min": batch[:, 1, :, :].min(),
            "y_max": batch[:, 1, :, :].max(),
            "z_min": batch[:, 2, :, :].min(),
            "z_max": batch[:, 2, :, :].max(),
        }
    
    stats_trn = get_stats(x_trn, idx_trn)
    stats_val = get_stats(x_val, idx_val)
    
    print("\n--- Train (Front) vs Val (Back) Coordinate Range ---")
    print(f"Train: x=[{stats_trn['x_min']:.2f}, {stats_trn['x_max']:.2f}], "
          f"y=[{stats_trn['y_min']:.2f}, {stats_trn['y_max']:.2f}], "
          f"z=[{stats_trn['z_min']:.2f}, {stats_trn['z_max']:.2f}]")
    print(f"Val:   x=[{stats_val['x_min']:.2f}, {stats_val['x_max']:.2f}], "
          f"y=[{stats_val['y_min']:.2f}, {stats_val['y_max']:.2f}], "
          f"z=[{stats_val['z_min']:.2f}, {stats_val['z_max']:.2f}]")
    
    # Class overlap
    trn_classes = set(np.unique(trn_labels))
    val_classes = set(np.unique(val_labels))
    overlap = trn_classes & val_classes
    
    print(f"\nClass overlap: {len(overlap)} / 102 classes")
    print(f"Classes only in train: {len(trn_classes - val_classes)}")
    print(f"Classes only in val:   {len(val_classes - trn_classes)}")

compare_splits(x_trn, x_val)
```

---

### 8. Sampled Visualization (Optional)

**Goal:** Quick sanity-check plots of a few random clips.

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_sample_clips(x, labels, names, n_samples=6, save_path="anubis_samples.png"):
    """Plot skeleton trajectories for a few random clips."""
    n_samples = min(n_samples, x.shape[0])
    idx = np.random.choice(x.shape[0], n_samples, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, sample_idx in enumerate(idx):
        ax = axes[i]
        
        # Get sample: (3, 300, 32, 2) -> take person 0
        sample = x[sample_idx, :, :, :, 0]  # (3, 300, 32)
        
        # Mean position over time for visualization
        # Take first 10 frames, every 30th frame
        frames_to_show = sample[:, ::30, :]  # (3, 10, 32)
        
        # Plot x, y, z trajectories for a few joints
        joints_to_show = [0, 1, 2, 3, 4, 5]  # pelvis and spine
        
        for j in joints_to_show:
            ax.plot(frames_to_show[0, :, j], frames_to_show[1, :, j], 'o-', label=f"joint {j}", alpha=0.7)
        
        ax.set_title(f"Class {labels[sample_idx]}: {names[sample_idx][:30]}")
        ax.legend(fontsize=6, loc='upper right')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"Saved: {save_path}")

# Plot a few samples
np.random.seed(42)
plot_sample_clips(x_trn, trn_labels, trn_names, n_samples=6)
```

---

## Summary: Preprocessing Decision Gates

After running all EDA cells, you should be able to answer:

| Question | Expected Answer | Gate |
|----------|---------------|------|
| Raw shape matches (N,3,300,32,2)? | Yes | Proceed |
| Labels aligned with data count? | Yes | Proceed |
| Person 1 usage significant? | If >20% clips have both, note it | Keep or drop P1 |
| All-zero / corrupted clips >5%? | If yes, filter needed | Add filter |
| Valid frames mostly >=60? | If no, consider smarter sampling | Adjust resampling |
| Joint 0 stable as root? | If no, choose different root | Adjust centering |
| Train/val class overlap good? | If no, expect harder transfer | Adjust eval |

---

## Next Steps After EDA

1. If EDA confirms assumptions → write preprocessing script
2. If EDA reveals issues → adjust preprocessing plan first
3. Then generate cleaned files:
   - `trn_features_cleaned.npy` — (N, 6, 60, 32)
   - `trn_labels_cleaned.npy` — (N,)
   - `val_features_cleaned.npy` — (M, 6, 60, 32)
   - `val_labels_cleaned.npy` — (M,)
4. Run audit scripts:
   - `python scripts/anubis_label_stats.py`
   - `python scripts/anubis_feature_stats.py`
5. Then start pretraining
