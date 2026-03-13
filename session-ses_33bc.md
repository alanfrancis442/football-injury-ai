# Module 1 planning for football_injury_ai_system

**Session ID:** ses_33bca5dddffenaDWY1Dytgsmiy
**Created:** 3/6/2026, 11:02:55 PM
**Updated:** 3/8/2026, 11:25:22 PM

---

## Assistant (Build · claude-sonnet-4.6 · 15.6s)

_Thinking:_

The user doesn't want to run another Optuna or long training. They just want to move forward with what they have. The best checkpoint they have is from Run 1 (80 epochs, Optuna best params, random split) - `pretrain_best.pt` at epoch 75. 

The question is: what's the next step given they want to move forward without more training?

The answer is: **use the existing `pretrain_best.pt` checkpoint and start building the fine-tuning pipeline**. The pretraining is good enough - the encoder has learned meaningful representations (71.8% on in-distribution data). The cross-subject gap is a known limitation but doesn't prevent fine-tuning from working.


So I should set up the fine-tuning infrastructure now - creating the directory structure, config files, and a script that loads the pretrained encoder and adds a new classification head for risk prediction. Since the labeled football data isn't ready yet, building this framework in advance means everything will be ready to go as soon as the data arrives.

You don't need to. The existing `pretrain_best.pt` (epoch 75, Optuna params) is good enough to move forward.

26% on a 102-class cross-subject benchmark is reasonable. The encoder has learned real biomechanical features — that's all you need it for. The ANUBIS classification accuracy is not the end goal; it's just a means to pretrain the encoder before fine-tuning on football data.

---

## What's actually next

You're blocked on **data**, not training. The fine-tuning pipeline needs to be built and ready so that the moment your friends finish labelling football clips, you can plug it straight in.

The work to do now is:

1. **`module1/finetune/dataset.py`** — loader for your football labelled clips (CSV label file + video-extracted pose sequences)
2. **`module1/finetune/model.py`** — loads pretrained encoder, replaces classification head with a risk head (binary or 3-class: low/medium/high)
3. **`module1/finetune/train.py`** — fine-tuning loop with frozen encoder for first N epochs, then unfreezing
4. **`configs/finetune.yaml`** — fine-tuning hyperparameters

Do you want me to build this out now so it's ready when the labelled data arrives?

---

## User

the labeling is still not completed we are still on the pretraining face,but I want a more good pretrained model and it should be able to understand bacis human motions,what should I do now i am attaching the research papher of skeletion motion analysis again for your referal /home/alan/Downloads/research_papers/mini_project/fnins-18-1353257.pdf

---

## Assistant (Build · gpt-5.3-codex · 13.8s)

_Thinking:_

**Reading research paper for insights**

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/Downloads/research_papers/mini_project/fnins-18-1353257.pdf"
}
```

**Output:**
```
PDF read successfully
```

---

## Assistant (Build · gpt-5.3-codex · 8.5s)

_Thinking:_

**Planning model improvement steps**

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/configs/pretrain.yaml"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/configs/pretrain.yaml</path>
<type>file</type>
<content>1: # configs/pretrain.yaml
2: #
3: # Stage 1 — ANUBIS Pretraining Hyperparameters
4: # ─────────────────────────────────────────────
5: # All hyperparameters for pretraining the SpatioTemporalEncoder on the ANUBIS
6: # action-recognition dataset (102 classes, 32 joints, pelvis-centred).
7: # Fine-tuning / inference values belong in configs/module1.yaml.
8: # Never hardcode any of these values inside Python files.
9: 
10: # ── Data ──────────────────────────────────────────────────────────────────────
11: data:
12:   anubis_dir:         data/anubis/
13: 
14:   # File names inside anubis_dir (produced by your preprocessing notebook).
15:   # Shape convention: features (N, 6, 60, 32) — (samples, channels, frames, joints)
16:   #                   channels = [x, y, z, vx, vy, vz] (pelvis-centred, vel pre-computed)
17:   #                   labels   (N,) — integer class index in [0, 101]
18:   train_features:     trn_features_cleaned.npy
19:   train_labels:       trn_labels_cleaned.npy
20:   val_features:       val_features_cleaned.npy   # only used when has_separate_val=true
21:   val_labels:         val_labels_cleaned.npy
22: 
23:   has_separate_val:   true    # using separate val files
24:   val_split:          0.15    # fraction of training set used for val when has_separate_val=false
25: 
26:   num_classes:        102     # ANUBIS action categories
27:   num_joints:         32
28:   clip_frames:        60      # T stored in the .npy file
29:   seq_len:            60      # frames fed to the model; must be <= clip_frames
30:                               # if seq_len < clip_frames a random window is cropped
31: 
32: # ── Model ─────────────────────────────────────────────────────────────────────
33: model:
34:   in_features:           9    # pos(3) + vel(3) + acc(3) per joint
35:   gat_hidden_dim:       64    # output width of every GAT block (heads × per_head_dim)
36:   gat_heads:             4    # multi-head attention (first two GAT layers)
37:   gat_dropout:         0.2
38:   d_model:             128    # transformer embedding width
39:   nhead:                 4    # transformer attention heads
40:   num_transformer_layers: 2
41:   dim_feedforward:     256
42:   transformer_dropout: 0.2
43:   cls_std:            0.02    # trunc_normal std for CLS-token initialisation
44: 
45: # ── Training ──────────────────────────────────────────────────────────────────
46: training:
47:   epochs:              120
48:   batch_size:           64
49:   learning_rate:      0.0005
50:   weight_decay:      0.0005
51:   warmup_epochs:         5    # linear LR warm-up before cosine decay
52:   grad_clip_norm:      1.0
53:   label_smoothing:    0.05    # applied to CrossEntropyLoss
54:   save_every_n_epochs:  10
55: 
56: # ── Augmentation ──────────────────────────────────────────────────────────────
57: augmentation:
58:   enabled:            true
59:   flip_prob:          0.3     # probability of left-right joint flip per sample
60:   joint_jitter_std:  0.005    # Gaussian noise σ added to all coordinates
61:   speed_perturb:     false    # temporal speed perturbation
62:   speed_range:        [0.8, 1.2]   # [min_factor, max_factor] for temporal stretch
63: 
64: # ── Paths ─────────────────────────────────────────────────────────────────────
65: paths:
66:   checkpoint_dir:  outputs/checkpoints/pretrain/
67:   log_dir:         outputs/logs/
68: 
69: # ── Optuna Hyperparameter Search ──────────────────────────────────────────────
70: optuna:
71:   enabled:          false
72:   study_name:       anubis_pretrain_hpo
73:   direction:        maximize
74:   n_trials:         30
75:   timeout_sec:      null
76:   trial_epochs:     25
77:   seed:             42
78:   num_workers:      4
79:   pin_memory:       true
80:   storage:          null
81:   load_if_exists:   true
82: 
83:   output_best_params: outputs/logs/optuna_best_params.yaml
84:   output_best_config: outputs/logs/pretrain_optuna_best.yaml
85: 
86:   # Optional overrides for sampling ranges
87:   search_space:
88:     learning_rate:
89:       low: 0.0001
90:       high: 0.002
91:       log: true
92:     weight_decay:
93:       low: 0.00001
94:       high: 0.001
95:       log: true
96:     label_smoothing:
97:       low: 0.0
98:       high: 0.15
99:     gat_dropout:
100:       low: 0.05
101:       high: 0.35
102:     transformer_dropout:
103:       low: 0.05
104:       high: 0.35
105:     flip_prob:
106:       low: 0.1
107:       high: 0.6
108:     joint_jitter_std:
109:       low: 0.001
110:       high: 0.02
111:       log: true
112:     speed_perturb:
113:       choices: [true, false]

(End of file - total 113 lines)
</content>
```

---

## Assistant (Build · gpt-5.3-codex · 14.8s)

_Thinking:_

**Analyzing training and augmentation options**

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/module1/data/preprocess.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/module1/data/preprocess.py</path>
<type>file</type>
<content>1: # module1/data/preprocess.py
2: #
3: # Stage 1 — Data Preprocessing and Augmentation
4: # ───────────────────────────────────────────────
5: # Stateless functions that operate on individual samples.
6: # Input tensors have shape (T, J, F):
7: #   T = number of frames
8: #   J = number of joints (32)
9: #   F = number of features per joint (9: pos + vel + acc)
10: #
11: # Feature layout:   [x, y, z,  vx, vy, vz,  ax, ay, az]
12: #   indices:         0  1  2    3   4   5    6   7   8
13: #
14: # All functions are pure (no in-place mutation) and return new tensors.
15: # They are called inside ANUBISDataset.__getitem__ during training.
16: 
17: from __future__ import annotations
18: 
19: from typing import List, Optional, Tuple
20: 
21: import torch
22: import torch.nn.functional as F
23: 
24: 
25: # ── Acceleration computation ───────────────────────────────────────────────────
26: 
27: 
28: def compute_acceleration(
29:     features: torch.Tensor,  # (N, 6, T, J) -> returns (N, 9, T, J)
30: ) -> torch.Tensor:
31:     """
32:     Compute acceleration (finite difference of velocity) and append as channels.
33: 
34:     Called once when the dataset is loaded to augment the raw 6-channel data
35:     (pos + vel) to 9-channel data (pos + vel + acc).
36: 
37:     Parameters
38:     ----------
39:     features : torch.Tensor  shape (N, 6, T, J)
40:         Channels: [x, y, z, vx, vy, vz]
41: 
42:     Returns
43:     -------
44:     features_9 : torch.Tensor  shape (N, 9, T, J)
45:         Channels: [x, y, z, vx, vy, vz, ax, ay, az]
46:         Acceleration is computed with a forward finite difference along time,
47:         and the last frame is duplicated from the previous frame to match
48:         the ANUBIS preprocessing notebook's boundary handling style.
49:     """
50:     vel = features[:, 3:6, :, :]  # (N, 3, T, J)
51:     acc = torch.zeros_like(vel)  # (N, 3, T, J)
52:     acc[:, :, :-1, :] = vel[:, :, 1:, :] - vel[:, :, :-1, :]
53:     if vel.shape[2] > 1:
54:         acc[:, :, -1, :] = acc[:, :, -2, :]
55:     return torch.cat([features, acc], dim=1)  # (N, 9, T, J)
56: 
57: 
58: # ── Window cropping ────────────────────────────────────────────────────────────
59: 
60: 
61: def random_crop(x: torch.Tensor, seq_len: int) -> torch.Tensor:
62:     """
63:     Randomly crop a contiguous window of ``seq_len`` frames from ``x``.
64: 
65:     Parameters
66:     ----------
67:     x       : torch.Tensor  shape (T, J, F)
68:     seq_len : int           must be <= T
69: 
70:     Returns
71:     -------
72:     cropped : torch.Tensor  shape (seq_len, J, F)
73:     """
74:     T = x.shape[0]
75:     if seq_len >= T:
76:         return x
77:     start = torch.randint(0, T - seq_len + 1, (1,)).item()
78:     return x[start : start + seq_len]
79: 
80: 
81: def center_crop(x: torch.Tensor, seq_len: int) -> torch.Tensor:
82:     """
83:     Take a centred window of ``seq_len`` frames from ``x`` (used at validation).
84: 
85:     Parameters
86:     ----------
87:     x       : torch.Tensor  shape (T, J, F)
88:     seq_len : int
89: 
90:     Returns
91:     -------
92:     cropped : torch.Tensor  shape (seq_len, J, F)
93:     """
94:     T = x.shape[0]
95:     if seq_len >= T:
96:         return x
97:     start = (T - seq_len) // 2
98:     return x[start : start + seq_len]
99: 
100: 
101: # ── Augmentation ───────────────────────────────────────────────────────────────
102: 
103: 
104: def flip_sequence(
105:     x: torch.Tensor,  # (T, J, F)
106:     mirror_pairs: List[Tuple[int, int]],
107:     flip_feature_indices: Optional[List[int]] = None,
108: ) -> torch.Tensor:
109:     """
110:     Mirror the skeleton left-right by swapping paired joints and negating the
111:     x-axis features (x, vx, ax at indices 0, 3, 6 for F=9).
112: 
113:     Parameters
114:     ----------
115:     x                   : torch.Tensor  shape (T, J, F)
116:     mirror_pairs        : list of (left_joint_idx, right_joint_idx)
117:     flip_feature_indices: feature dimensions to negate; defaults to [0, 3, 6]
118: 
119:     Returns
120:     -------
121:     flipped : torch.Tensor  shape (T, J, F)
122:     """
123:     if flip_feature_indices is None:
124:         flip_feature_indices = [0, 3, 6]
125: 
126:     x = x.clone()
127: 
128:     # Swap joint pairs
129:     for l_idx, r_idx in mirror_pairs:
130:         left = x[:, l_idx, :].clone()
131:         right = x[:, r_idx, :].clone()
132:         x[:, l_idx, :] = right
133:         x[:, r_idx, :] = left
134: 
135:     # Negate lateral (x-axis) features
136:     x[:, :, flip_feature_indices] = -x[:, :, flip_feature_indices]
137: 
138:     return x
139: 
140: 
141: def joint_jitter(x: torch.Tensor, std: float) -> torch.Tensor:
142:     """
143:     Add zero-mean Gaussian noise to all joint features.
144: 
145:     Parameters
146:     ----------
147:     x   : torch.Tensor  shape (T, J, F)
148:     std : float         noise standard deviation
149: 
150:     Returns
151:     -------
152:     noisy : torch.Tensor  shape (T, J, F)
153:     """
154:     return x + torch.randn_like(x) * std
155: 
156: 
157: def speed_perturb(
158:     x: torch.Tensor,  # (T, J, F)
159:     factor_range: Tuple[float, float],
160:     target_len: int,
161: ) -> torch.Tensor:
162:     """
163:     Temporally stretch or compress a sequence by a random factor, then
164:     crop / pad back to ``target_len`` frames.
165: 
166:     Uses linear interpolation along the time axis.
167: 
168:     Parameters
169:     ----------
170:     x            : torch.Tensor  shape (T, J, F)
171:     factor_range : (min_factor, max_factor)  e.g. (0.8, 1.2)
172:     target_len   : output sequence length after the perturbation
173: 
174:     Returns
175:     -------
176:     perturbed : torch.Tensor  shape (target_len, J, F)
177:     """
178:     T, J, Fin = x.shape
179:     lo, hi = factor_range
180:     factor = lo + torch.rand(1).item() * (hi - lo)
181:     new_T = max(2, int(round(T * factor)))  # must be at least 2 for interpolate
182: 
183:     # F.interpolate expects (N, C, L) — treat joints×features as channels
184:     x_t = x.permute(1, 2, 0).reshape(1, J * Fin, T).float()
185:     x_resampled = F.interpolate(x_t, size=new_T, mode="linear", align_corners=False)
186:     x_resampled = x_resampled.reshape(J, Fin, new_T).permute(2, 0, 1)  # (new_T, J, F)
187: 
188:     # Crop or zero-pad to target_len
189:     if new_T >= target_len:
190:         start = torch.randint(0, new_T - target_len + 1, (1,)).item()
191:         x_resampled = x_resampled[start : start + target_len]
192:     else:
193:         pad_len = target_len - new_T
194:         pad = x_resampled[-1:].expand(pad_len, -1, -1)
195:         x_resampled = torch.cat([x_resampled, pad], dim=0)
196: 
197:     return x_resampled
198: 
199: 
200: # ── Quick sanity check ────────────────────────────────────────────────────────
201: 
202: if __name__ == "__main__":
203:     from module1.data.joint_config import MIRROR_JOINT_PAIRS, FLIP_FEATURE_INDICES
204: 
205:     T, J, F_IN = 60, 32, 6
206:     N = 8
207:     dummy_raw = torch.randn(N, F_IN, T, J)
208: 
209:     # Acceleration
210:     feat9 = compute_acceleration(dummy_raw)
211:     assert feat9.shape == (N, 9, T, J), feat9.shape
212: 
213:     # One sample in (T, J, F) format
214:     sample = feat9[0].permute(1, 2, 0)  # (T, J, 9)
215:     assert sample.shape == (T, J, 9)
216: 
217:     # Random crop
218:     cropped = random_crop(sample, seq_len=50)
219:     assert cropped.shape == (50, J, 9)
220: 
221:     # Centre crop
222:     centred = center_crop(sample, seq_len=50)
223:     assert centred.shape == (50, J, 9)
224: 
225:     # Flip
226:     flipped = flip_sequence(sample, MIRROR_JOINT_PAIRS, FLIP_FEATURE_INDICES)
227:     assert flipped.shape == sample.shape
228:     # After double-flip we should get back the original
229:     double_flipped = flip_sequence(flipped, MIRROR_JOINT_PAIRS, FLIP_FEATURE_INDICES)
230:     assert torch.allclose(sample, double_flipped)
231: 
232:     # Jitter
233:     jittered = joint_jitter(sample, std=0.01)
234:     assert jittered.shape == sample.shape
235: 
236:     # Speed perturb
237:     perturbed = speed_perturb(sample, factor_range=(0.8, 1.2), target_len=60)
238:     assert perturbed.shape == (60, J, 9)
239: 
240:     print("preprocess OK.")

(End of file - total 240 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/module1/data/anubis_loader.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/module1/data/anubis_loader.py</path>
<type>file</type>
<content>1: # module1/data/anubis_loader.py
2: #
3: # Stage 1 — ANUBIS Dataset Loader
4: # ─────────────────────────────────
5: # Loads preprocessed ANUBIS skeleton data from .npy files and exposes it as a
6: # PyTorch Dataset / DataLoader pair ready for model training.
7: #
8: # Expected files on disk (configured in configs/pretrain.yaml):
9: #   trn_features_cleaned.npy  —  shape (N, 6, 60, 32)
10: #                                  OR (N, 6, 60, 32, M)
11: #                                  axes: (samples, channels, frames, joints[, persons])
12: #                                  channels: [x, y, z, vx, vy, vz]
13: #                                  coordinates are pelvis-centred.
14: #   trn_labels_cleaned.npy    —  shape (N,)  integer class index in [0, 101]
15: #
16: # The loader converts 6-channel data → 9-channel (pos + vel + acc), transposes
17: # to (N, T, J, F) format, and wraps in an ANUBISDataset that applies
18: # configurable augmentation during training.
19: 
20: from __future__ import annotations
21: 
22: import os
23: from typing import Dict, Optional, Tuple
24: 
25: import numpy as np
26: import torch
27: from torch.utils.data import DataLoader, Dataset, random_split
28: 
29: from module1.data.joint_config import (
30:     FLIP_FEATURE_INDICES,
31:     MIRROR_JOINT_PAIRS,
32: )
33: from module1.data.preprocess import (
34:     center_crop,
35:     compute_acceleration,
36:     flip_sequence,
37:     joint_jitter,
38:     random_crop,
39:     speed_perturb,
40: )
41: from utils.logger import get_logger
42: 
43: log = get_logger(__name__)
44: 
45: 
46: # ── Dataset ────────────────────────────────────────────────────────────────────
47: 
48: 
49: class ANUBISDataset(Dataset):
50:     """
51:     PyTorch Dataset wrapping preprocessed ANUBIS skeleton clips.
52: 
53:     Each item is a tuple ``(features, label)`` where:
54:       features : torch.Tensor  shape (seq_len, J, F)   float32
55:       label    : torch.Tensor  shape ()                 int64
56: 
57:     Parameters
58:     ----------
59:     features : torch.Tensor  shape (N, T, J, F)
60:         Pre-loaded and pre-processed feature tensor (pos + vel + acc).
61:     labels : torch.Tensor  shape (N,)
62:         Integer class indices in [0, num_classes).
63:     seq_len : int
64:         Number of frames per sample.  A random crop is applied during training
65:         and a centre crop during validation when seq_len < T.
66:     augment : bool
67:         Whether to apply data augmentation (training split only).
68:     aug_cfg : dict, optional
69:         Augmentation hyper-parameters.  Keys:
70:           flip_prob        (float, default 0.5)
71:           joint_jitter_std (float, default 0.01)
72:           speed_perturb    (bool,  default True)
73:           speed_range      (list,  default [0.8, 1.2])
74:     """
75: 
76:     def __init__(
77:         self,
78:         features: torch.Tensor,
79:         labels: torch.Tensor,
80:         seq_len: int,
81:         augment: bool = False,
82:         aug_cfg: Optional[Dict] = None,
83:     ) -> None:
84:         self.features = features  # (N, T, J, F)
85:         self.labels = labels  # (N,)
86:         self.seq_len = seq_len
87:         self.clip_len = features.shape[1]
88:         self.augment = augment
89:         self.aug_cfg = aug_cfg or {}
90: 
91:     def __len__(self) -> int:
92:         return len(self.labels)
93: 
94:     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
95:         x = self.features[idx].clone()  # (T, J, F)
96:         y = self.labels[idx]
97: 
98:         # ── Window crop ──────────────────────────────────────────────────────
99:         if self.seq_len < self.clip_len:
100:             x = (
101:                 random_crop(x, self.seq_len)
102:                 if self.augment
103:                 else center_crop(x, self.seq_len)
104:             )
105: 
106:         # ── Speed perturbation (before other augmentations) ──────────────────
107:         if self.augment and self.aug_cfg.get("speed_perturb", True):
108:             sr = self.aug_cfg.get("speed_range", [0.8, 1.2])
109:             x = speed_perturb(x, factor_range=(sr[0], sr[1]), target_len=self.seq_len)
110: 
111:         # ── Horizontal flip ──────────────────────────────────────────────────
112:         if self.augment:
113:             prob = self.aug_cfg.get("flip_prob", 0.5)
114:             if torch.rand(1).item() < prob:
115:                 x = flip_sequence(x, MIRROR_JOINT_PAIRS, FLIP_FEATURE_INDICES)
116: 
117:         # ── Joint jitter ─────────────────────────────────────────────────────
118:         if self.augment:
119:             std = self.aug_cfg.get("joint_jitter_std", 0.01)
120:             if std > 0:
121:                 x = joint_jitter(x, std)
122: 
123:         return x, y
124: 
125: 
126: # ── Data loading helpers ───────────────────────────────────────────────────────
127: 
128: 
129: def _load_npy(path: str, desc: str) -> np.ndarray:
130:     """Load a .npy file with a descriptive error if it does not exist."""
131:     if not os.path.exists(path):
132:         raise FileNotFoundError(
133:             f"{desc} not found: {path}\n"
134:             "Copy your preprocessed ANUBIS files to the anubis_dir specified in "
135:             "configs/pretrain.yaml, then run again."
136:         )
137:     arr = np.load(path)
138:     log.info("Loaded %s  shape=%s  dtype=%s", desc, arr.shape, arr.dtype)
139:     return arr
140: 
141: 
142: def load_anubis_tensors(
143:     features_path: str,
144:     labels_path: str,
145: ) -> Tuple[torch.Tensor, torch.Tensor]:
146:     """
147:     Load raw .npy files, compute acceleration, and return (features, labels).
148: 
149:     Parameters
150:     ----------
151:     features_path : str   path to .npy file of shape (N, 6, T, J)
152:                           or (N, 6, T, J, M)
153:     labels_path   : str   path to .npy file of shape (N,)
154: 
155:     Returns
156:     -------
157:     features : torch.Tensor  shape (N, T, J, 9)   float32
158:     labels   : torch.Tensor  shape (N,)            int64
159:     """
160:     raw = torch.from_numpy(_load_npy(features_path, "features").astype(np.float32))
161:     lbls = torch.from_numpy(_load_npy(labels_path, "labels").astype(np.int64))
162: 
163:     # Support both (N, C, T, J) and (N, C, T, J, M) layouts.
164:     # If multiple persons are present, follow the preprocessing notebook and
165:     # keep only the primary person at index 0.
166:     if raw.ndim == 5:
167:         num_persons = raw.shape[-1]
168:         if num_persons < 1:
169:             raise ValueError(
170:                 "Expected person dimension M >= 1 in (N, C, T, J, M), got 0."
171:             )
172:         raw = raw[..., 0]
173:         log.info("Selected primary person index 0 from M=%d", num_persons)
174: 
175:     if raw.ndim != 4:
176:         raise ValueError(
177:             "Expected features to have 4 dimensions (N, C, T, J) or 5 dimensions "
178:             f"(N, C, T, J, M), got {raw.ndim}. Check that the file matches the "
179:             "format described in anubis_loader.py."
180:         )
181:     if raw.shape[1] != 6:
182:         raise ValueError(
183:             f"Expected 6 channels [x,y,z,vx,vy,vz], got {raw.shape[1]}. "
184:             "Acceleration will be computed here; do not pre-compute it."
185:         )
186: 
187:     # Compute acceleration: raw (N,6,T,J) → features (N,9,T,J)
188:     features_9 = compute_acceleration(raw)  # (N, 9, T, J)
189:     # Transpose to (N, T, J, 9) — time-first layout expected by the model
190:     features_9 = features_9.permute(0, 2, 3, 1).contiguous()
191: 
192:     log.info("Features after acc. computation: shape=%s", tuple(features_9.shape))
193:     return features_9, lbls
194: 
195: 
196: def build_dataloaders(
197:     cfg: Dict,
198:     num_workers: int = 4,
199:     pin_memory: bool = True,
200: ) -> Tuple[DataLoader, DataLoader]:
201:     """
202:     Build train and validation DataLoaders from config.
203: 
204:     Parameters
205:     ----------
206:     cfg : dict
207:         Full config dict loaded from configs/pretrain.yaml.
208:     num_workers : int
209:         DataLoader worker processes.
210:     pin_memory : bool
211:         Pin host memory for faster GPU transfer.
212: 
213:     Returns
214:     -------
215:     train_loader : DataLoader
216:     val_loader   : DataLoader
217:     """
218:     data_cfg = cfg["data"]
219:     aug_cfg = cfg.get("augmentation", {})
220: 
221:     base_dir = data_cfg["anubis_dir"]
222:     seq_len = data_cfg["seq_len"]
223:     batch_size = cfg["training"]["batch_size"]
224:     val_split = data_cfg.get("val_split", 0.15)
225:     has_sep_val = data_cfg.get("has_separate_val", False)
226: 
227:     # ── Load training data ───────────────────────────────────────────────────
228:     trn_feat, trn_lbl = load_anubis_tensors(
229:         features_path=os.path.join(base_dir, data_cfg["train_features"]),
230:         labels_path=os.path.join(base_dir, data_cfg["train_labels"]),
231:     )
232: 
233:     # ── Validation split ─────────────────────────────────────────────────────
234:     if has_sep_val:
235:         val_feat, val_lbl = load_anubis_tensors(
236:             features_path=os.path.join(base_dir, data_cfg["val_features"]),
237:             labels_path=os.path.join(base_dir, data_cfg["val_labels"]),
238:         )
239:     else:
240:         # Random split of the training set
241:         N = len(trn_lbl)
242:         n_val = int(N * val_split)
243:         n_trn = N - n_val
244:         indices = torch.randperm(N)
245:         trn_idx, val_idx = indices[:n_trn], indices[n_trn:]
246:         val_feat = trn_feat[val_idx]
247:         val_lbl = trn_lbl[val_idx]
248:         trn_feat = trn_feat[trn_idx]
249:         trn_lbl = trn_lbl[trn_idx]
250:         log.info("Random val split: %d train / %d val", n_trn, n_val)
251: 
252:     # ── Datasets ─────────────────────────────────────────────────────────────
253:     aug_enabled = aug_cfg.get("enabled", True)
254:     train_ds = ANUBISDataset(
255:         trn_feat,
256:         trn_lbl,
257:         seq_len=seq_len,
258:         augment=aug_enabled,
259:         aug_cfg=aug_cfg,
260:     )
261:     val_ds = ANUBISDataset(
262:         val_feat,
263:         val_lbl,
264:         seq_len=seq_len,
265:         augment=False,
266:     )
267: 
268:     log.info("Train dataset: %d samples", len(train_ds))
269:     log.info("Val   dataset: %d samples", len(val_ds))
270: 
271:     # ── DataLoaders ──────────────────────────────────────────────────────────
272:     train_loader = DataLoader(
273:         train_ds,
274:         batch_size=batch_size,
275:         shuffle=True,
276:         num_workers=num_workers,
277:         pin_memory=pin_memory,
278:         drop_last=True,
279:     )
280:     val_loader = DataLoader(
281:         val_ds,
282:         batch_size=batch_size,
283:         shuffle=False,
284:         num_workers=num_workers,
285:         pin_memory=pin_memory,
286:         drop_last=False,
287:     )
288: 
289:     return train_loader, val_loader
290: 
291: 
292: # ── Quick sanity check ────────────────────────────────────────────────────────
293: 
294: if __name__ == "__main__":
295:     import tempfile
296: 
297:     from module1.data.joint_config import NUM_JOINTS
298: 
299:     J, T, N = NUM_JOINTS, 60, 20
300:     # Create dummy raw data matching the .npy format
301:     raw = np.random.randn(N, 6, T, J).astype(np.float32)
302:     lbl = np.random.randint(0, 102, size=(N,)).astype(np.int64)
303: 
304:     with tempfile.TemporaryDirectory() as tmpdir:
305:         fp = os.path.join(tmpdir, "features.npy")
306:         lp = os.path.join(tmpdir, "labels.npy")
307:         np.save(fp, raw)
308:         np.save(lp, lbl)
309: 
310:         feats, labels = load_anubis_tensors(fp, lp)
311:         assert feats.shape == (N, T, J, 9), feats.shape
312:         assert labels.shape == (N,), labels.shape
313: 
314:         ds = ANUBISDataset(feats, labels, seq_len=50, augment=True)
315:         x, y = ds[0]
316:         assert x.shape == (50, J, 9), x.shape
317:         assert y.dtype == torch.int64
318: 
319:     print("anubis_loader OK.")

(End of file - total 319 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/module1/pretrain/train.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/module1/pretrain/train.py</path>
<type>file</type>
<content>1:                                                             # module1/pretrain/train.py
2: #
3: # Stage 1 — ANUBIS Pretraining Loop
4: # ──────────────────────────────────
5: # Trains ActionClassifier end-to-end on ANUBIS 102-class action recognition.
6: # LR schedule: linear warm-up for warmup_epochs, then cosine annealing.
7: # After training, only encoder weights are needed for fine-tuning.
8: #
9: # Usage:
10: #   python module1/pretrain/train.py --config configs/pretrain.yaml
11: #
12: # Checkpoints are saved to paths.checkpoint_dir:
13: #   pretrain_best.pt     — best validation loss
14: #   pretrain_ep{N}.pt    — periodic saves every save_every_n_epochs epochs
15: 
16: from __future__ import annotations
17: 
18: import argparse
19: import math
20: import os
21: from typing import Dict, Tuple
22: 
23: import torch
24: import torch.nn as nn
25: from torch.utils.data import DataLoader
26: from tqdm import tqdm
27: 
28: from module1.data.anubis_loader import build_dataloaders
29: from module1.data.joint_config import EDGE_INDEX
30: from module1.pretrain.model import ActionClassifier, SpatioTemporalEncoder, build_model
31: from utils.config import load_config
32: from utils.logger import get_logger
33: 
34: log = get_logger(__name__)
35: 
36: 
37: # ── LR schedule ───────────────────────────────────────────────────────────────
38: 
39: 
40: def _make_lr_lambda(warmup_epochs: int, total_epochs: int):
41:     """
42:     Return a LambdaLR multiplier function implementing linear warm-up followed
43:     by cosine annealing.
44: 
45:     Multiplier at epoch e:
46:       e < warmup_epochs  →  (e+1) / warmup_epochs   (linear ramp)
47:       e >= warmup_epochs →  0.5 * (1 + cos(π * progress))
48:     """
49: 
50:     def lr_lambda(epoch: int) -> float:
51:         if epoch < warmup_epochs:
52:             return float(epoch + 1) / float(max(1, warmup_epochs))
53:         progress = float(epoch - warmup_epochs) / float(
54:             max(1, total_epochs - warmup_epochs)
55:         )
56:         return 0.5 * (1.0 + math.cos(math.pi * progress))
57: 
58:     return lr_lambda
59: 
60: 
61: # ── Training / evaluation functions ──────────────────────────────────────────
62: 
63: 
64: def train_one_epoch(
65:     model: ActionClassifier,
66:     loader: DataLoader,
67:     optimiser: torch.optim.Optimizer,
68:     criterion: nn.CrossEntropyLoss,
69:     device: torch.device,
70:     grad_clip: float,
71: ) -> Tuple[float, float]:
72:     """
73:     Run one training epoch.
74: 
75:     Parameters
76:     ----------
77:     model     : ActionClassifier
78:     loader    : DataLoader         training data
79:     optimiser : torch.optim.Optimizer
80:     criterion : nn.CrossEntropyLoss
81:     device    : torch.device
82:     grad_clip : float              max_norm for gradient clipping
83: 
84:     Returns
85:     -------
86:     avg_loss : float   mean CE loss over all batches
87:     accuracy : float   top-1 accuracy in [0, 1]
88:     """
89:     model.train()
90:     total_loss = 0.0
91:     total_correct = 0
92:     total_samples = 0
93: 
94:     for x, y in tqdm(loader, desc="  train", leave=False, unit="batch"):
95:         x = x.to(device, non_blocking=True)  # (B, T, J, F)
96:         y = y.to(device, non_blocking=True)  # (B,)
97: 
98:         optimiser.zero_grad()
99:         logits = model(x)  # (B, num_classes)
100:         loss = criterion(logits, y)
101:         loss.backward()
102: 
103:         nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
104:         optimiser.step()
105: 
106:         bs = y.size(0)
107:         total_loss += loss.item() * bs
108:         total_correct += (logits.argmax(dim=1) == y).sum().item()
109:         total_samples += bs
110: 
111:     avg_loss = total_loss / total_samples
112:     accuracy = total_correct / total_samples
113:     return avg_loss, accuracy
114: 
115: 
116: @torch.no_grad()
117: def evaluate(
118:     model: ActionClassifier,
119:     loader: DataLoader,
120:     criterion: nn.CrossEntropyLoss,
121:     device: torch.device,
122: ) -> Tuple[float, float, float]:
123:     """
124:     Evaluate the model on a data split.
125: 
126:     Parameters
127:     ----------
128:     model     : ActionClassifier
129:     loader    : DataLoader
130:     criterion : nn.CrossEntropyLoss  (label_smoothing=0 has no effect at eval)
131:     device    : torch.device
132: 
133:     Returns
134:     -------
135:     avg_loss  : float   mean CE loss
136:     top1_acc  : float   top-1 accuracy in [0, 1]
137:     top5_acc  : float   top-5 accuracy in [0, 1]
138:     """
139:     model.eval()
140:     total_loss = 0.0
141:     total_top1 = 0
142:     total_top5 = 0
143:     total_samples = 0
144: 
145:     for x, y in tqdm(loader, desc="  val  ", leave=False, unit="batch"):
146:         x = x.to(device, non_blocking=True)
147:         y = y.to(device, non_blocking=True)
148: 
149:         logits = model(x)  # (B, num_classes)
150:         loss = criterion(logits, y)
151: 
152:         bs = y.size(0)
153:         total_loss += loss.item() * bs
154: 
155:         # Top-1
156:         total_top1 += (logits.argmax(dim=1) == y).sum().item()
157: 
158:         # Top-5
159:         top5_preds = logits.topk(k=min(5, logits.size(1)), dim=1).indices  # (B, 5)
160:         total_top5 += (top5_preds == y.unsqueeze(1)).any(dim=1).sum().item()
161: 
162:         total_samples += bs
163: 
164:     avg_loss = total_loss / total_samples
165:     top1_acc = total_top1 / total_samples
166:     top5_acc = total_top5 / total_samples
167:     return avg_loss, top1_acc, top5_acc
168: 
169: 
170: # ── Checkpoint helpers ────────────────────────────────────────────────────────
171: 
172: 
173: def _save_checkpoint(
174:     path: str,
175:     epoch: int,
176:     encoder: SpatioTemporalEncoder,
177:     classifier: ActionClassifier,
178:     optimiser: torch.optim.Optimizer,
179:     scheduler: torch.optim.lr_scheduler.LRScheduler,
180:     metrics: Dict,
181: ) -> None:
182:     os.makedirs(os.path.dirname(path), exist_ok=True)
183:     torch.save(
184:         {
185:             "epoch": epoch,
186:             "encoder": encoder.state_dict(),
187:             "classifier": classifier.state_dict(),
188:             "optimiser": optimiser.state_dict(),
189:             "scheduler": scheduler.state_dict(),
190:             **metrics,
191:         },
192:         path,
193:     )
194:     log.info("Saved checkpoint → %s", path)
195: 
196: 
197: # ── Main training entry point ─────────────────────────────────────────────────
198: 
199: 
200: def main(config_path: str) -> None:
201:     cfg = load_config(config_path)
202: 
203:     # ── Device ──────────────────────────────────────────────────────────────
204:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
205:     log.info("Device: %s", device)
206: 
207:     # ── Data ────────────────────────────────────────────────────────────────
208:     log.info("Loading ANUBIS dataset …")
209:     train_loader, val_loader = build_dataloaders(cfg)
210: 
211:     # ── Model ───────────────────────────────────────────────────────────────
212:     edge_index = EDGE_INDEX.to(device)
213:     encoder, classifier = build_model(cfg, edge_index)
214:     classifier = classifier.to(device)
215: 
216:     # ── Optimiser ───────────────────────────────────────────────────────────
217:     t_cfg = cfg["training"]
218:     optimiser = torch.optim.AdamW(
219:         classifier.parameters(),
220:         lr=t_cfg["learning_rate"],
221:         weight_decay=t_cfg["weight_decay"],
222:     )
223: 
224:     # Warm-up + cosine LR schedule
225:     total_epochs = t_cfg["epochs"]
226:     warmup_epochs = t_cfg.get("warmup_epochs", 5)
227:     scheduler = torch.optim.lr_scheduler.LambdaLR(
228:         optimiser,
229:         lr_lambda=_make_lr_lambda(warmup_epochs, total_epochs),
230:     )
231: 
232:     # ── Loss ────────────────────────────────────────────────────────────────
233:     criterion = nn.CrossEntropyLoss(label_smoothing=t_cfg.get("label_smoothing", 0.1))
234:     eval_criterion = nn.CrossEntropyLoss()  # no smoothing for val metrics
235: 
236:     # ── Training loop ────────────────────────────────────────────────────────
237:     ckpt_dir = cfg["paths"]["checkpoint_dir"]
238:     save_every = t_cfg.get("save_every_n_epochs", 10)
239:     grad_clip = t_cfg.get("grad_clip_norm", 1.0)
240: 
241:     best_val_loss = float("inf")
242:     best_epoch = 0
243: 
244:     log.info("Starting pretraining for %d epochs", total_epochs)
245: 
246:     for epoch in range(1, total_epochs + 1):
247:         current_lr = optimiser.param_groups[0]["lr"]
248:         log.info("Epoch %d/%d  lr=%.6f", epoch, total_epochs, current_lr)
249: 
250:         # Train
251:         trn_loss, trn_acc = train_one_epoch(
252:             classifier, train_loader, optimiser, criterion, device, grad_clip
253:         )
254: 
255:         # Validate
256:         val_loss, val_top1, val_top5 = evaluate(
257:             classifier, val_loader, eval_criterion, device
258:         )
259: 
260:         scheduler.step()
261: 
262:         log.info(
263:             "  trn loss=%.4f  acc=%.3f | val loss=%.4f  top1=%.3f  top5=%.3f",
264:             trn_loss,
265:             trn_acc,
266:             val_loss,
267:             val_top1,
268:             val_top5,
269:         )
270: 
271:         metrics = {
272:             "val_loss": val_loss,
273:             "val_top1": val_top1,
274:             "val_top5": val_top5,
275:             "trn_loss": trn_loss,
276:             "trn_acc": trn_acc,
277:         }
278: 
279:         # Save best checkpoint
280:         if val_loss < best_val_loss:
281:             best_val_loss = val_loss
282:             best_epoch = epoch
283:             _save_checkpoint(
284:                 os.path.join(ckpt_dir, "pretrain_best.pt"),
285:                 epoch,
286:                 encoder,
287:                 classifier,
288:                 optimiser,
289:                 scheduler,
290:                 metrics,
291:             )
292:             log.info("  *** New best at epoch %d  val_loss=%.4f ***", epoch, val_loss)
293: 
294:         # Periodic checkpoint
295:         if epoch % save_every == 0:
296:             _save_checkpoint(
297:                 os.path.join(ckpt_dir, f"pretrain_ep{epoch:04d}.pt"),
298:                 epoch,
299:                 encoder,
300:                 classifier,
301:                 optimiser,
302:                 scheduler,
303:                 metrics,
304:             )
305: 
306:     log.info(
307:         "Pretraining complete.  Best epoch: %d  val_loss=%.4f",
308:         best_epoch,
309:         best_val_loss,
310:     )
311:     log.info(
312:         "Encoder weights saved in:  %s/pretrain_best.pt  (key: 'encoder')", ckpt_dir
313:     )
314: 
315: 
316: # ── CLI entry point ───────────────────────────────────────────────────────────
317: 
318: if __name__ == "__main__":
319:     parser = argparse.ArgumentParser(
320:         description="Pretrain SpatioTemporalEncoder on ANUBIS"
321:     )
322:     parser.add_argument(
323:         "--config",
324:         default="configs/pretrain.yaml",
325:         help="Path to pretrain config YAML",
326:     )
327:     args = parser.parse_args()
328:     main(args.config)

(End of file - total 328 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/module1/pretrain/evaluate.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/module1/pretrain/evaluate.py</path>
<type>file</type>
<content>1: # module1/pretrain/evaluate.py
2: #
3: # Stage 1 — Pretraining Evaluation
4: # ──────────────────────────────────
5: # Loads a pretrained ActionClassifier checkpoint and computes:
6: #   • Top-1 and top-5 accuracy
7: #   • Per-class F1 score (macro and per-class breakdown)
8: #   • t-SNE visualisation of CLS embeddings coloured by action class
9: #
10: # Usage:
11: #   python module1/pretrain/evaluate.py \
12: #       --config  configs/pretrain.yaml \
13: #       --ckpt    outputs/checkpoints/pretrain/pretrain_best.pt \
14: #       --split   val          # or 'train'
15: #       --tsne                 # add this flag to generate the t-SNE plot
16: 
17: from __future__ import annotations
18: 
19: import argparse
20: import inspect
21: import os
22: from typing import Dict, List, Optional, Tuple
23: 
24: import numpy as np
25: import torch
26: import torch.nn as nn
27: from sklearn.metrics import classification_report, f1_score
28: from sklearn.manifold import TSNE
29: import matplotlib.pyplot as plt
30: from torch.utils.data import DataLoader
31: from tqdm import tqdm
32: 
33: from module1.data.anubis_loader import (
34:     build_dataloaders,
35:     load_anubis_tensors,
36:     ANUBISDataset,
37: )
38: from module1.data.joint_config import EDGE_INDEX
39: from module1.pretrain.model import ActionClassifier, build_model
40: from utils.config import load_config
41: from utils.logger import get_logger
42: 
43: log = get_logger(__name__)
44: 
45: 
46: # ── Metric computation ────────────────────────────────────────────────────────
47: 
48: 
49: @torch.no_grad()
50: def collect_predictions(
51:     model: ActionClassifier,
52:     loader: DataLoader,
53:     device: torch.device,
54: ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
55:     """
56:     Run the model over a DataLoader and collect logits, predictions, and labels.
57: 
58:     Returns
59:     -------
60:     embeddings : np.ndarray  shape (N, d_model)   CLS embeddings
61:     preds      : np.ndarray  shape (N,)            top-1 predicted class
62:     labels     : np.ndarray  shape (N,)            ground-truth class
63:     """
64:     model.eval()
65:     all_embeddings: List[np.ndarray] = []
66:     all_preds: List[np.ndarray] = []
67:     all_labels: List[np.ndarray] = []
68: 
69:     for x, y in tqdm(loader, desc="  eval ", leave=False, unit="batch"):
70:         x = x.to(device, non_blocking=True)
71:         y = y.to(device, non_blocking=True)
72: 
73:         # CLS embedding from encoder
74:         emb = model.encoder(x)  # (B, d_model)
75:         logits = model.head(emb)  # (B, num_classes)
76:         preds = logits.argmax(dim=1)  # (B,)
77: 
78:         all_embeddings.append(emb.cpu().numpy())
79:         all_preds.append(preds.cpu().numpy())
80:         all_labels.append(y.cpu().numpy())
81: 
82:     embeddings = np.concatenate(all_embeddings, axis=0)
83:     preds = np.concatenate(all_preds, axis=0)
84:     labels = np.concatenate(all_labels, axis=0)
85:     return embeddings, preds, labels
86: 
87: 
88: def compute_accuracy_metrics(
89:     preds: np.ndarray,
90:     labels: np.ndarray,
91:     logits_path: Optional[str] = None,
92: ) -> Dict[str, float]:
93:     """
94:     Compute top-1 accuracy and macro-averaged F1.
95: 
96:     Parameters
97:     ----------
98:     preds  : shape (N,) predicted class indices
99:     labels : shape (N,) ground-truth class indices
100: 
101:     Returns
102:     -------
103:     metrics : dict with keys 'top1_acc' and 'macro_f1'
104:     """
105:     top1 = float((preds == labels).mean())
106:     macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
107:     return {"top1_acc": top1, "macro_f1": macro_f1}
108: 
109: 
110: def compute_top5_accuracy(
111:     model: ActionClassifier,
112:     loader: DataLoader,
113:     device: torch.device,
114: ) -> float:
115:     """
116:     Compute top-5 accuracy by re-running the model.
117: 
118:     Returns
119:     -------
120:     top5_acc : float  in [0, 1]
121:     """
122:     model.eval()
123:     total_top5 = 0
124:     total = 0
125: 
126:     with torch.no_grad():
127:         for x, y in tqdm(loader, desc="  top5 ", leave=False, unit="batch"):
128:             x = x.to(device, non_blocking=True)
129:             y = y.to(device, non_blocking=True)
130:             logits = model(x)
131:             top5 = logits.topk(k=min(5, logits.size(1)), dim=1).indices
132:             total_top5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
133:             total += y.size(0)
134: 
135:     return total_top5 / total
136: 
137: 
138: # ── t-SNE visualisation ───────────────────────────────────────────────────────
139: 
140: 
141: def plot_tsne(
142:     embeddings: np.ndarray,
143:     labels: np.ndarray,
144:     out_path: str,
145:     n_samples: int = 2000,
146:     perplexity: float = 30.0,
147: ) -> None:
148:     """
149:     Compute t-SNE on a random subset of CLS embeddings and save a scatter plot.
150: 
151:     Parameters
152:     ----------
153:     embeddings : np.ndarray  shape (N, D)
154:     labels     : np.ndarray  shape (N,)   integer class indices
155:     out_path   : str         path to save the PNG
156:     n_samples  : int         max samples to subsample (t-SNE is O(n²))
157:     perplexity : float       t-SNE perplexity parameter
158:     """
159:     N = len(labels)
160:     if N > n_samples:
161:         idx = np.random.choice(N, n_samples, replace=False)
162:         embeddings = embeddings[idx]
163:         labels = labels[idx]
164: 
165:     log.info("Running t-SNE on %d samples (d=%d) …", len(labels), embeddings.shape[1])
166:     tsne_kwargs = {
167:         "n_components": 2,
168:         "perplexity": perplexity,
169:         "random_state": 42,
170:     }
171:     # scikit-learn renamed n_iter -> max_iter in newer versions.
172:     if "max_iter" in inspect.signature(TSNE.__init__).parameters:
173:         tsne_kwargs["max_iter"] = 1000
174:     else:
175:         tsne_kwargs["n_iter"] = 1000
176: 
177:     tsne = TSNE(**tsne_kwargs)
178:     coords = tsne.fit_transform(embeddings)  # (N, 2)
179: 
180:     num_classes = int(labels.max()) + 1
181:     cmap = plt.cm.get_cmap("tab20", num_classes)
182: 
183:     fig, ax = plt.subplots(figsize=(12, 10))
184:     scatter = ax.scatter(
185:         coords[:, 0],
186:         coords[:, 1],
187:         c=labels,
188:         cmap=cmap,
189:         s=6,
190:         alpha=0.7,
191:     )
192:     plt.colorbar(scatter, ax=ax, label="Action class")
193:     ax.set_title("t-SNE of SpatioTemporalEncoder CLS embeddings (ANUBIS)")
194:     ax.set_xlabel("t-SNE dim 1")
195:     ax.set_ylabel("t-SNE dim 2")
196:     ax.set_xticks([])
197:     ax.set_yticks([])
198: 
199:     os.makedirs(os.path.dirname(out_path), exist_ok=True)
200:     plt.tight_layout()
201:     plt.savefig(out_path, dpi=150)
202:     plt.close()
203:     log.info("t-SNE plot saved → %s", out_path)
204: 
205: 
206: # ── Checkpoint loading ────────────────────────────────────────────────────────
207: 
208: 
209: def load_checkpoint(
210:     ckpt_path: str,
211:     classifier: ActionClassifier,
212:     device: torch.device,
213: ) -> Dict:
214:     """
215:     Load a pretrain checkpoint into the classifier.
216: 
217:     Parameters
218:     ----------
219:     ckpt_path  : str
220:     classifier : ActionClassifier  (must be already constructed)
221:     device     : torch.device
222: 
223:     Returns
224:     -------
225:     ckpt : dict   full checkpoint dict (contains epoch, metrics, etc.)
226:     """
227:     if not os.path.exists(ckpt_path):
228:         raise FileNotFoundError(
229:             f"Checkpoint not found: {ckpt_path}\nRun module1/pretrain/train.py first."
230:         )
231:     ckpt = torch.load(ckpt_path, map_location=device)
232:     classifier.load_state_dict(ckpt["classifier"])
233:     log.info(
234:         "Loaded checkpoint from %s  (epoch=%d  val_top1=%.3f)",
235:         ckpt_path,
236:         ckpt.get("epoch", -1),
237:         ckpt.get("val_top1", float("nan")),
238:     )
239:     return ckpt
240: 
241: 
242: # ── Main evaluation entry point ───────────────────────────────────────────────
243: 
244: 
245: def main(
246:     config_path: str, ckpt_path: str, split: str, run_tsne: bool
247: ) -> Dict[str, float]:
248:     cfg = load_config(config_path)
249:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
250:     log.info("Device: %s", device)
251: 
252:     # ── Build model and load weights ─────────────────────────────────────
253:     edge_index = EDGE_INDEX.to(device)
254:     encoder, classifier = build_model(cfg, edge_index)
255:     classifier = classifier.to(device)
256:     load_checkpoint(ckpt_path, classifier, device)
257:     classifier.eval()
258: 
259:     # ── DataLoader ───────────────────────────────────────────────────────
260:     log.info("Loading data (split=%s) …", split)
261:     _, val_loader = build_dataloaders(cfg)
262:     loader = val_loader  # extend here if split='train' is needed
263: 
264:     # ── Collect predictions ──────────────────────────────────────────────
265:     embeddings, preds, labels = collect_predictions(classifier, loader, device)
266: 
267:     # ── Metrics ──────────────────────────────────────────────────────────
268:     metrics = compute_accuracy_metrics(preds, labels)
269:     top5 = compute_top5_accuracy(classifier, loader, device)
270:     metrics["top5_acc"] = top5
271: 
272:     log.info("Results:")
273:     log.info("  Top-1 accuracy : %.4f", metrics["top1_acc"])
274:     log.info("  Top-5 accuracy : %.4f", metrics["top5_acc"])
275:     log.info("  Macro F1       : %.4f", metrics["macro_f1"])
276: 
277:     # Detailed per-class report (printed to stdout)
278:     print("\nPer-class classification report (F1, precision, recall):")
279:     print(classification_report(labels, preds, zero_division=0))
280: 
281:     # ── t-SNE ────────────────────────────────────────────────────────────
282:     if run_tsne:
283:         tsne_path = os.path.join(cfg["paths"]["log_dir"], "pretrain_tsne.png")
284:         plot_tsne(embeddings, labels, out_path=tsne_path)
285: 
286:     return metrics
287: 
288: 
289: # ── CLI entry point ───────────────────────────────────────────────────────────
290: 
291: if __name__ == "__main__":
292:     parser = argparse.ArgumentParser(
293:         description="Evaluate a pretrained SpatioTemporalEncoder on ANUBIS"
294:     )
295:     parser.add_argument("--config", default="configs/pretrain.yaml")
296:     parser.add_argument(
297:         "--ckpt", default="outputs/checkpoints/pretrain/pretrain_best.pt"
298:     )
299:     parser.add_argument("--split", default="val", choices=["val", "train"])
300:     parser.add_argument(
301:         "--tsne",
302:         action="store_true",
303:         help="Generate t-SNE scatter plot of CLS embeddings",
304:     )
305:     args = parser.parse_args()
306:     main(args.config, args.ckpt, args.split, args.tsne)

(End of file - total 306 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py</path>
<type>file</type>
<content>1: # module1/pretrain/model.py
2: #
3: # Stage 1 — SpatioTemporalEncoder + ActionClassifier
4: # ────────────────────────────────────────────────────
5: # End-to-end model that encodes a sequence of skeleton graphs:
6: #
7: #   Input : (B, T, J, F)  — batch, frames, joints, features (pos+vel+acc = 9)
8: #
9: #   Spatial stage — time-distributed GAT (shared weights across all T frames):
10: #     GATConv( F → 16, heads=4, concat=True)  → 64-dim per node + BN + ELU
11: #     GATConv(64 → 16, heads=4, concat=True)  → 64-dim per node + BN + ELU
12: #     GATConv(64 → 64, heads=1, concat=False) → 64-dim per node + BN + ELU
13: #     global_mean_pool over joints             → (B*T, 64)
14: #     reshape                                  → (B, T, 64)
15: #
16: #   Temporal stage — Transformer encoder:
17: #     input_proj Linear(64 → d_model)
18: #     prepend CLS token
19: #     sinusoidal positional encoding
20: #     2 × TransformerEncoderLayer (pre-norm, nhead=4, ff=256)
21: #     LayerNorm
22: #     extract CLS output                       → (B, d_model)
23: #
24: #   ActionClassifier wraps SpatioTemporalEncoder with a linear head for
25: #   102-class ANUBIS pretraining.  After pretraining, save only the encoder
26: #   (encoder.state_dict()) and attach new fine-tuning heads.
27: 
28: from __future__ import annotations
29: 
30: import math
31: from typing import Optional, Tuple
32: 
33: import torch
34: import torch.nn as nn
35: import torch.nn.functional as F
36: from torch_geometric.nn import GATConv, global_mean_pool
37: 
38: from utils.logger import get_logger
39: 
40: log = get_logger(__name__)
41: 
42: 
43: # ── Positional Encoding ────────────────────────────────────────────────────────
44: 
45: 
46: class PositionalEncoding(nn.Module):
47:     """
48:     Sinusoidal positional encoding (Vaswani et al., 2017).
49: 
50:     Parameters
51:     ----------
52:     d_model  : int   embedding dimension
53:     max_len  : int   maximum sequence length (including CLS token)
54:     dropout  : float dropout applied after adding the encoding
55:     """
56: 
57:     def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1) -> None:
58:         super().__init__()
59:         self.dropout = nn.Dropout(p=dropout)
60: 
61:         position = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
62:         div_term = torch.exp(
63:             torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
64:         )
65:         pe = torch.zeros(1, max_len, d_model)
66:         pe[0, :, 0::2] = torch.sin(position * div_term)
67:         pe[0, :, 1::2] = torch.cos(position * div_term)
68:         self.register_buffer("pe", pe)
69: 
70:     def forward(self, x: torch.Tensor) -> torch.Tensor:
71:         """
72:         Parameters
73:         ----------
74:         x : torch.Tensor  shape (B, T, d_model)
75: 
76:         Returns
77:         -------
78:         out : torch.Tensor  shape (B, T, d_model)
79:         """
80:         x = x + self.pe[:, : x.size(1), :]
81:         return self.dropout(x)
82: 
83: 
84: # ── SpatioTemporalEncoder ─────────────────────────────────────────────────────
85: 
86: 
87: class SpatioTemporalEncoder(nn.Module):
88:     """
89:     End-to-end spatiotemporal encoder: time-distributed GAT + Transformer.
90: 
91:     The encoder is the shared backbone used across all training stages.
92:     It outputs a single CLS embedding per sequence.
93: 
94:     Parameters
95:     ----------
96:     num_joints   : int   number of skeleton joints (32 for ANUBIS)
97:     in_features  : int   feature dimension per joint (9: pos+vel+acc)
98:     edge_index   : torch.Tensor  shape (2, E)  skeleton connectivity;
99:                    registered as a buffer so it moves to device automatically
100:     gat_hidden   : int   output dim of every GAT block (default 64)
101:     gat_heads    : int   attention heads for GAT layers 1 & 2 (default 4)
102:     gat_dropout  : float dropout on GAT attention coefficients
103:     d_model      : int   transformer embedding dimension (default 128)
104:     nhead        : int   transformer attention heads
105:     num_layers   : int   number of TransformerEncoderLayer blocks
106:     dim_ff       : int   feedforward expansion inside transformer
107:     dropout      : float transformer dropout
108:     seq_len      : int   expected sequence length (used to size pos. encoding)
109:     cls_std      : float trunc_normal std for CLS-token initialisation
110:     """
111: 
112:     def __init__(
113:         self,
114:         num_joints: int,
115:         in_features: int,
116:         edge_index: torch.Tensor,
117:         gat_hidden: int = 64,
118:         gat_heads: int = 4,
119:         gat_dropout: float = 0.1,
120:         d_model: int = 128,
121:         nhead: int = 4,
122:         num_layers: int = 2,
123:         dim_ff: int = 256,
124:         dropout: float = 0.1,
125:         seq_len: int = 60,
126:         cls_std: float = 0.02,
127:     ) -> None:
128:         super().__init__()
129: 
130:         self.num_joints = num_joints
131:         self.gat_hidden = gat_hidden
132:         self.gat_dropout = gat_dropout
133:         self.d_model = d_model
134: 
135:         # Register skeleton topology as a non-learnable buffer
136:         self.register_buffer("edge_index", edge_index)  # (2, E)
137: 
138:         # ── GAT layers (shared across all time steps) ──────────────────────
139:         per_head_dim = gat_hidden // gat_heads  # 16
140:         self.gat1 = GATConv(
141:             in_features, per_head_dim, heads=gat_heads, concat=True, dropout=gat_dropout
142:         )
143:         self.gat2 = GATConv(
144:             gat_hidden, per_head_dim, heads=gat_heads, concat=True, dropout=gat_dropout
145:         )
146:         self.gat3 = GATConv(
147:             gat_hidden, gat_hidden, heads=1, concat=False, dropout=gat_dropout
148:         )
149: 
150:         self.bn1 = nn.BatchNorm1d(gat_hidden)
151:         self.bn2 = nn.BatchNorm1d(gat_hidden)
152:         self.bn3 = nn.BatchNorm1d(gat_hidden)
153: 
154:         # ── Transformer ────────────────────────────────────────────────────
155:         self.input_proj = nn.Linear(gat_hidden, d_model)
156: 
157:         self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
158:         nn.init.trunc_normal_(self.cls_token, std=cls_std)
159: 
160:         # max_len = seq_len + 1 to accommodate the CLS token
161:         self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)
162: 
163:         encoder_layer = nn.TransformerEncoderLayer(
164:             d_model=d_model,
165:             nhead=nhead,
166:             dim_feedforward=dim_ff,
167:             dropout=dropout,
168:             batch_first=True,
169:             norm_first=True,  # pre-norm variant — more stable gradients
170:             activation="gelu",
171:         )
172:         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
173:         self.norm = nn.LayerNorm(d_model)
174: 
175:     # ── Internal helpers ───────────────────────────────────────────────────────
176: 
177:     def _gat_forward(self, x: torch.Tensor) -> torch.Tensor:
178:         """
179:         Process all T frames through the GAT in a single batched pass.
180: 
181:         Each (frame, sequence) pair is treated as an independent graph.
182:         All B*T graphs share the same edge topology (self.edge_index).
183: 
184:         Parameters
185:         ----------
186:         x : torch.Tensor  shape (B, T, J, F)
187: 
188:         Returns
189:         -------
190:         spatial : torch.Tensor  shape (B, T, gat_hidden)
191:         """
192:         B, T, J, Fin = x.shape
193:         num_graphs = B * T
194: 
195:         # Flatten nodes: (B*T*J, Fin)
196:         x_nodes = x.reshape(num_graphs * J, Fin)
197: 
198:         # Build batched edge index by offsetting each graph's node indices by J
199:         # edge_index: (2, E)  →  broadcast to (2, num_graphs, E)
200:         E = self.edge_index.shape[1]
201:         offsets = torch.arange(num_graphs, device=x.device) * J  # (num_graphs,)
202:         edge_batch = self.edge_index.unsqueeze(1).expand(
203:             -1, num_graphs, -1
204:         )  # (2, G, E)
205:         edge_batch = edge_batch + offsets.view(1, -1, 1)  # (2, G, E)
206:         edge_flat = edge_batch.reshape(2, num_graphs * E)  # (2, G*E)
207: 
208:         # batch_tensor: maps each node to its graph index
209:         batch_tensor = torch.arange(num_graphs, device=x.device).repeat_interleave(J)
210: 
211:         # ── GAT layer 1 ────────────────────────────────────────────────────
212:         h = self.gat1(x_nodes, edge_flat)  # (B*T*J, gat_hidden)
213:         h = self.bn1(h)
214:         h = F.elu(h)
215:         h = F.dropout(h, p=self.gat_dropout, training=self.training)
216: 
217:         # ── GAT layer 2 ────────────────────────────────────────────────────
218:         h = self.gat2(h, edge_flat)
219:         h = self.bn2(h)
220:         h = F.elu(h)
221:         h = F.dropout(h, p=self.gat_dropout, training=self.training)
222: 
223:         # ── GAT layer 3 ────────────────────────────────────────────────────
224:         h = self.gat3(h, edge_flat)
225:         h = self.bn3(h)
226:         h = F.elu(h)
227: 
228:         # Global mean pool: (B*T*J, gat_hidden) → (B*T, gat_hidden)
229:         spatial = global_mean_pool(h, batch_tensor)
230: 
231:         return spatial.reshape(B, T, self.gat_hidden)  # (B, T, gat_hidden)
232: 
233:     # ── Forward pass ──────────────────────────────────────────────────────────
234: 
235:     def forward(self, x: torch.Tensor) -> torch.Tensor:
236:         """
237:         Encode a batch of skeleton sequences.
238: 
239:         Parameters
240:         ----------
241:         x : torch.Tensor  shape (B, T, J, F)
242:             Joint features: pos + vel + acc (F = 9).
243: 
244:         Returns
245:         -------
246:         cls_embedding : torch.Tensor  shape (B, d_model)
247:             Sequence-level representation extracted from the CLS token.
248:         """
249:         B = x.shape[0]
250: 
251:         # Stage 1: spatial encoding (time-distributed GAT)
252:         spatial = self._gat_forward(x)  # (B, T, gat_hidden)
253: 
254:         # Stage 2: temporal encoding
255:         h = self.input_proj(spatial)  # (B, T, d_model)
256: 
257:         # Prepend CLS token
258:         cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
259:         h = torch.cat([cls, h], dim=1)  # (B, T+1, d_model)
260: 
261:         # Positional encoding
262:         h = self.pos_enc(h)  # (B, T+1, d_model)
263: 
264:         # Transformer encoder
265:         h = self.transformer(h)  # (B, T+1, d_model)
266:         h = self.norm(h)
267: 
268:         # Extract CLS token (position 0)
269:         return h[:, 0, :]  # (B, d_model)
270: 
271: 
272: # ── ActionClassifier ──────────────────────────────────────────────────────────
273: 
274: 
275: class ActionClassifier(nn.Module):
276:     """
277:     Full pretraining model: SpatioTemporalEncoder + linear classification head.
278: 
279:     Used only for Stage 1 (ANUBIS action classification pretraining).
280:     After pretraining, save encoder.state_dict() and discard the head.
281: 
282:     Parameters
283:     ----------
284:     encoder     : SpatioTemporalEncoder
285:     num_classes : int   number of action classes (102 for ANUBIS)
286:     """
287: 
288:     def __init__(self, encoder: SpatioTemporalEncoder, num_classes: int) -> None:
289:         super().__init__()
290:         self.encoder = encoder
291:         self.head = nn.Linear(encoder.d_model, num_classes)
292: 
293:     def forward(self, x: torch.Tensor) -> torch.Tensor:
294:         """
295:         Parameters
296:         ----------
297:         x : torch.Tensor  shape (B, T, J, F)
298: 
299:         Returns
300:         -------
301:         logits : torch.Tensor  shape (B, num_classes)
302:         """
303:         cls_emb = self.encoder(x)  # (B, d_model)
304:         return self.head(cls_emb)  # (B, num_classes)
305: 
306: 
307: # ── Factory helper ────────────────────────────────────────────────────────────
308: 
309: 
310: def build_model(
311:     cfg: dict, edge_index: torch.Tensor
312: ) -> Tuple[SpatioTemporalEncoder, ActionClassifier]:
313:     """
314:     Construct encoder and full classifier from config dict.
315: 
316:     Parameters
317:     ----------
318:     cfg        : dict               full config loaded from configs/pretrain.yaml
319:     edge_index : torch.Tensor (2,E) skeleton edge index
320: 
321:     Returns
322:     -------
323:     encoder    : SpatioTemporalEncoder
324:     classifier : ActionClassifier
325:     """
326:     m = cfg["model"]
327:     d = cfg["data"]
328:     encoder = SpatioTemporalEncoder(
329:         num_joints=d["num_joints"],
330:         in_features=m["in_features"],
331:         edge_index=edge_index,
332:         gat_hidden=m["gat_hidden_dim"],
333:         gat_heads=m["gat_heads"],
334:         gat_dropout=m["gat_dropout"],
335:         d_model=m["d_model"],
336:         nhead=m["nhead"],
337:         num_layers=m["num_transformer_layers"],
338:         dim_ff=m["dim_feedforward"],
339:         dropout=m["transformer_dropout"],
340:         seq_len=d["seq_len"],
341:         cls_std=m["cls_std"],
342:     )
343:     classifier = ActionClassifier(encoder, num_classes=d["num_classes"])
344:     total = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
345:     log.info("Model built: %d trainable parameters", total)
346:     return encoder, classifier
347: 
348: 
349: # ── Quick sanity check ────────────────────────────────────────────────────────
350: 
351: if __name__ == "__main__":
352:     from module1.data.joint_config import EDGE_INDEX, NUM_JOINTS
353: 
354:     B, T, J, F_IN = 4, 60, NUM_JOINTS, 9
355:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
356: 
357:     encoder = SpatioTemporalEncoder(
358:         num_joints=J,
359:         in_features=F_IN,
360:         edge_index=EDGE_INDEX,
361:         seq_len=T,
362:     ).to(device)
363: 
364:     classifier = ActionClassifier(encoder, num_classes=102).to(device)
365:     classifier.eval()
366: 
367:     dummy = torch.randn(B, T, J, F_IN, device=device)
368: 
369:     with torch.no_grad():
370:         cls_emb = encoder(dummy)
371:         logits = classifier(dummy)
372: 
373:     assert cls_emb.shape == (B, 128), f"Expected (4,128), got {cls_emb.shape}"
374:     assert logits.shape == (B, 102), f"Expected (4,102), got {logits.shape}"
375:     assert not torch.isnan(cls_emb).any(), "NaN in encoder output"
376:     assert not torch.isnan(logits).any(), "NaN in classifier logits"
377: 
378:     n_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
379:     print(f"Encoder output shape    : {cls_emb.shape}")
380:     print(f"Classifier output shape : {logits.shape}")
381:     print(f"Trainable parameters    : {n_params:,}")
382:     print("model.py OK.")

(End of file - total 382 lines)
</content>
```

---

## Assistant (Build · gpt-5.3-codex · 7.9s)

_Thinking:_

**Analyzing run log for context**

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/outputs/logs/run.log",
  "offset": 612,
  "limit": 1200
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/outputs/logs/run.log</path>
<type>file</type>
<content>612: 06:36:14  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
613: 06:36:14  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
614: 06:36:14  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
615: 06:36:14  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
616: 06:36:14  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
617: 06:45:23  INFO      __main__ — Trial 0 done | best val top1=0.3015 top5=0.6297 loss=2.6767
618: 06:45:23  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
619: 06:45:23  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
620: 06:45:24  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
621: 06:45:24  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
622: 06:45:24  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
623: 06:45:24  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
624: 06:45:24  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
625: 06:54:35  INFO      __main__ — Trial 1 done | best val top1=0.3918 top5=0.7369 loss=2.2448
626: 06:54:35  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
627: 06:54:35  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
628: 06:54:36  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
629: 06:54:36  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
630: 06:54:36  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
631: 06:54:36  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
632: 06:54:36  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
633: 07:03:46  INFO      __main__ — Trial 2 done | best val top1=0.3638 top5=0.7063 loss=2.3476
634: 07:03:46  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
635: 07:03:47  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
636: 07:03:47  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
637: 07:03:47  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
638: 07:03:47  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
639: 07:03:47  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
640: 07:03:47  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
641: 07:12:56  INFO      __main__ — Trial 3 done | best val top1=0.3124 top5=0.6302 loss=2.6606
642: 07:12:57  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
643: 07:12:57  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
644: 07:12:57  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
645: 07:12:57  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
646: 07:12:57  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
647: 07:12:57  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
648: 07:12:57  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
649: 07:22:09  INFO      __main__ — Trial 4 done | best val top1=0.3071 top5=0.6468 loss=2.6257
650: 07:22:10  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
651: 07:22:10  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
652: 07:22:10  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
653: 07:22:10  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
654: 07:22:10  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
655: 07:22:10  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
656: 07:22:10  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
657: 07:31:21  INFO      __main__ — Trial 5 done | best val top1=0.3721 top5=0.7110 loss=2.3157
658: 07:31:21  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
659: 07:31:21  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
660: 07:31:22  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
661: 07:31:22  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
662: 07:31:22  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
663: 07:31:22  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
664: 07:31:22  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
665: 07:40:32  INFO      __main__ — Trial 6 done | best val top1=0.4541 top5=0.7960 loss=1.9338
666: 07:40:32  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
667: 07:40:32  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
668: 07:40:33  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
669: 07:40:33  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
670: 07:40:33  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
671: 07:40:33  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
672: 07:40:33  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
673: 07:41:39  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
674: 07:41:40  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
675: 07:41:40  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
676: 07:41:40  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
677: 07:41:40  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
678: 07:41:40  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
679: 07:41:40  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
680: 07:42:46  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
681: 07:42:46  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
682: 07:42:47  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
683: 07:42:47  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
684: 07:42:47  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
685: 07:42:47  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
686: 07:42:47  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
687: 07:46:49  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
688: 07:46:50  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
689: 07:46:50  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
690: 07:46:50  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
691: 07:46:50  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
692: 07:46:50  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
693: 07:46:50  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
694: 07:56:00  INFO      __main__ — Trial 10 done | best val top1=0.4858 top5=0.8247 loss=1.7722
695: 07:56:01  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
696: 07:56:01  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
697: 07:56:01  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
698: 07:56:01  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
699: 07:56:01  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
700: 07:56:01  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
701: 07:56:01  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
702: 08:05:12  INFO      __main__ — Trial 11 done | best val top1=0.4828 top5=0.8179 loss=1.7819
703: 08:05:12  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
704: 08:05:12  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
705: 08:05:13  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
706: 08:05:13  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
707: 08:05:13  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
708: 08:05:13  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
709: 08:05:13  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
710: 08:14:23  INFO      __main__ — Trial 12 done | best val top1=0.4989 top5=0.8279 loss=1.7416
711: 08:14:24  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
712: 08:14:24  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
713: 08:14:24  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
714: 08:14:24  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
715: 08:14:24  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
716: 08:14:24  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
717: 08:14:24  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
718: 08:23:35  INFO      __main__ — Trial 13 done | best val top1=0.4907 top5=0.8263 loss=1.7593
719: 08:23:35  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
720: 08:23:35  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
721: 08:23:36  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
722: 08:23:36  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
723: 08:23:36  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
724: 08:23:36  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
725: 08:23:36  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
726: 08:24:42  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
727: 08:24:42  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
728: 08:24:43  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
729: 08:24:43  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
730: 08:24:43  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
731: 08:24:43  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
732: 08:24:43  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
733: 08:33:51  INFO      __main__ — Trial 15 done | best val top1=0.4639 top5=0.8082 loss=1.8727
734: 08:33:52  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
735: 08:33:52  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
736: 08:33:52  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
737: 08:33:52  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
738: 08:33:52  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
739: 08:33:52  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
740: 08:33:52  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
741: 08:35:43  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
742: 08:35:43  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
743: 08:35:43  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
744: 08:35:43  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
745: 08:35:43  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
746: 08:35:43  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
747: 08:35:43  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
748: 08:44:52  INFO      __main__ — Trial 17 done | best val top1=0.4748 top5=0.8076 loss=1.8408
749: 08:44:52  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
750: 08:44:52  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
751: 08:44:53  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
752: 08:44:53  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
753: 08:44:53  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
754: 08:44:53  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
755: 08:44:53  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
756: 08:45:59  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
757: 08:45:59  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
758: 08:46:00  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
759: 08:46:00  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
760: 08:46:00  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
761: 08:46:00  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
762: 08:46:00  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
763: 08:47:06  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
764: 08:47:06  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
765: 08:47:07  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
766: 08:47:07  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
767: 08:47:07  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
768: 08:47:07  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
769: 08:47:07  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
770: 08:56:18  INFO      __main__ — Trial 20 done | best val top1=0.4651 top5=0.8121 loss=1.8858
771: 08:56:18  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
772: 08:56:18  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
773: 08:56:18  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
774: 08:56:19  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
775: 08:56:19  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
776: 08:56:19  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
777: 08:56:19  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
778: 09:05:29  INFO      __main__ — Trial 21 done | best val top1=0.4892 top5=0.8235 loss=1.7695
779: 09:05:30  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
780: 09:05:30  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
781: 09:05:30  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
782: 09:05:30  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
783: 09:05:30  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
784: 09:05:30  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
785: 09:05:30  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
786: 09:14:41  INFO      __main__ — Trial 22 done | best val top1=0.4852 top5=0.8267 loss=1.7820
787: 09:14:41  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
788: 09:14:41  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
789: 09:14:42  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
790: 09:14:42  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
791: 09:14:42  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
792: 09:14:42  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
793: 09:14:42  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
794: 09:16:54  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
795: 09:16:55  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
796: 09:16:55  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
797: 09:16:55  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
798: 09:16:55  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
799: 09:16:55  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
800: 09:16:55  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
801: 09:18:01  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
802: 09:18:01  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
803: 09:18:02  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
804: 09:18:02  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
805: 09:18:02  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
806: 09:18:02  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
807: 09:18:02  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
808: 09:19:30  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
809: 09:19:30  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
810: 09:19:31  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
811: 09:19:31  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
812: 09:19:31  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
813: 09:19:31  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
814: 09:19:31  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
815: 09:20:37  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
816: 09:20:37  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
817: 09:20:38  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
818: 09:20:38  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
819: 09:20:38  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
820: 09:20:38  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
821: 09:20:38  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
822: 09:22:06  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
823: 09:22:06  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
824: 09:22:07  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
825: 09:22:07  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
826: 09:22:07  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
827: 09:22:07  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
828: 09:22:07  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
829: 09:23:13  INFO      module1.data.anubis_loader — Loaded features  shape=(33501, 6, 60, 32)  dtype=float32
830: 09:23:13  INFO      module1.data.anubis_loader — Loaded labels  shape=(33501,)  dtype=int64
831: 09:23:14  INFO      module1.data.anubis_loader — Features after acc. computation: shape=(33501, 60, 32, 9)
832: 09:23:14  INFO      module1.data.anubis_loader — Random val split: 28476 train / 5025 val
833: 09:23:14  INFO      module1.data.anubis_loader — Train dataset: 28476 samples
834: 09:23:14  INFO      module1.data.anubis_loader — Val   dataset: 5025 samples
835: 09:23:14  INFO      module1.pretrain.model — Model built: 296550 trainable parameters
836: 09:24:20  INFO      __main__ — Best trial: 12
837: 09:24:20  INFO      __main__ — Best value (val_top1): 0.4989
838: 09:24:20  INFO      __main__ — Best params: {'learning_rate': 0.0019464623176755585, 'weight_decay': 0.00020433313066500611, 'label_smoothing': 0.040048660941805404, 'gat_dropout': 0.051065527924036666, 'transformer_dropout': 0.22439178858315353, 'flip_prob': 0.39576557171375976, 'joint_jitter_std': 0.0020494388932778985, 'speed_perturb': False}
839: 09:24:20  INFO      __main__ — Saved best params → outputs/logs/optuna_best_params.yaml
840: 09:24:20  INFO      __main__ — Saved best config → outputs/logs/pretrain_optuna_best.yaml

(End of file - total 840 lines)
</content>
```

---

## Assistant (Build · gpt-5.3-codex · 100.9s)

_Thinking:_

**Recommending immediate training improvements**

_Thinking:_

**Proposing concise no-Optuna improvement plan**

You are thinking in the right direction.  
Also, quick note: I couldn’t read that PDF here because PDF parsing isn’t supported in this environment.

What you should do now (without rerunning Optuna):

- Use Optuna’s best params as base (`outputs/logs/pretrain_optuna_best.yaml`), but this file currently uses random split (`has_separate_val: false`), which inflates scores.
- Create a new final config and set:
  - `data.has_separate_val: true`
  - keep Optuna params
  - `training.epochs: 60-80`
  - `augmentation.speed_perturb: true` and `speed_range: [0.9, 1.1]` (helps “basic motion” invariance)
- Run one clean training + eval on official val only. No Optuna needed.
- Keep the best checkpoint from this run as your pretrained backbone.

Why this is the best next step:
- Your current model is good, but tuned on random split.  
- One proper run on official val gives a much more trustworthy pretrained encoder for downstream football fine-tuning.
- Speed perturbation improves motion understanding (same action at different tempo).

If you want, I can now prepare a ready-to-run `configs/pretrain_final.yaml` with these exact settings so you only run one command.

---

## User

yes go on with the plan

---

## Assistant (Build · gpt-5.3-codex · 25.4s)

_Thinking:_

**Creating final training config**

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Add File: /home/alan/mydev/college/football-injury-ai/configs/pretrain_final.yaml\n+# configs/pretrain_final.yaml\n+#\n+# Stage 1 — Final ANUBIS Pretraining Config (No Optuna)\n+# ──────────────────────────────────────────────────────\n+# This configuration uses the best Optuna hyperparameters discovered earlier,\n+# but trains and validates on the official ANUBIS split (separate val files)\n+# for a trustworthy generalisation estimate.\n+\n+# ── Data ──────────────────────────────────────────────────────────────────────\n+data:\n+  anubis_dir:         data/anubis/\n+\n+  train_features:     trn_features_cleaned.npy\n+  train_labels:       trn_labels_cleaned.npy\n+  val_features:       val_features_cleaned.npy\n+  val_labels:         val_labels_cleaned.npy\n+\n+  has_separate_val:   true\n+  val_split:          0.15    # ignored when has_separate_val=true\n+\n+  num_classes:        102\n+  num_joints:         32\n+  clip_frames:        60\n+  seq_len:            60\n+\n+# ── Model ─────────────────────────────────────────────────────────────────────\n+model:\n+  in_features:             9\n+  gat_hidden_dim:         64\n+  gat_heads:               4\n+  gat_dropout:             0.051065527924036666\n+  d_model:               128\n+  nhead:                   4\n+  num_transformer_layers:  2\n+  dim_feedforward:       256\n+  transformer_dropout:     0.22439178858315353\n+  cls_std:                 0.02\n+\n+# ── Training ──────────────────────────────────────────────────────────────────\n+training:\n+  epochs:                80\n+  batch_size:            64\n+  learning_rate:         0.0019464623176755585\n+  weight_decay:          0.00020433313066500611\n+  warmup_epochs:         5\n+  grad_clip_norm:        1.0\n+  label_smoothing:       0.040048660941805404\n+  save_every_n_epochs:   10\n+\n+# ── Augmentation ──────────────────────────────────────────────────────────────\n+augmentation:\n+  enabled:               true\n+  flip_prob:             0.39576557171375976\n+  joint_jitter_std:      0.0020494388932778985\n+  speed_perturb:         true\n+  speed_range:           [0.9, 1.1]\n+\n+# ── Paths ─────────────────────────────────────────────────────────────────────\n+paths:\n+  checkpoint_dir:  outputs/checkpoints/pretrain/\n+  log_dir:         outputs/logs/\n+\n+# ── Optuna Hyperparameter Search ──────────────────────────────────────────────\n+optuna:\n+  enabled:          false\n+  study_name:       anubis_pretrain_hpo\n+  direction:        maximize\n+  n_trials:         30\n+  timeout_sec:      null\n+  trial_epochs:     25\n+  seed:             42\n+  num_workers:      4\n+  pin_memory:       true\n+  storage:          null\n+  load_if_exists:   true\n+\n+  output_best_params: outputs/logs/optuna_best_params.yaml\n+  output_best_config: outputs/logs/pretrain_optuna_best.yaml\n+\n+  search_space:\n+    learning_rate:\n+      low: 0.0001\n+      high: 0.002\n+      log: true\n+    weight_decay:\n+      low: 0.00001\n+      high: 0.001\n+      log: true\n+    label_smoothing:\n+      low: 0.0\n+      high: 0.15\n+    gat_dropout:\n+      low: 0.05\n+      high: 0.35\n+    transformer_dropout:\n+      low: 0.05\n+      high: 0.35\n+    flip_prob:\n+      low: 0.1\n+      high: 0.6\n+    joint_jitter_std:\n+      low: 0.001\n+      high: 0.02\n+      log: true\n+    speed_perturb:\n+      choices: [true, false]\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
A configs/pretrain_final.yaml
```

---

## Assistant (Build · gpt-5.3-codex · 5.3s)

Great — I created a final config for you at `configs/pretrain_final.yaml`.

What I changed:
- Uses Optuna best hyperparameters
- Forces official split: `data.has_separate_val: true`
- Keeps strong training length: `training.epochs: 80`
- Enables mild tempo augmentation for better motion understanding:
  - `augmentation.speed_perturb: true`
  - `augmentation.speed_range: [0.9, 1.1]`

Run this now:
```bash
python module1/pretrain/train.py --config configs/pretrain_final.yaml
```

Then evaluate:
```bash
python module1/pretrain/evaluate.py --config configs/pretrain_final.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --split val --tsne
```

If you paste the final `Top-1 / Top-5 / Macro-F1`, I’ll tell you whether this checkpoint is strong enough to freeze as your official pretrained backbone.

---

## User

please commit and push the changes so that i can train the model on different meachine

---

## Assistant (Build · minimax-m2.5-free · 11.8s)

_Thinking:_

The user wants me to commit and push the changes so they can train on a different machine. Let me check git status, git diff, and recent commit messages to follow the commit conventions.

**Tool: bash**

**Input:**
```json
{
  "command": "git status",
  "description": "Check git status"
}
```

**Output:**
```
On branch master
Your branch is up to date with 'origin/master'.

Changes not staged for commit:
  (use "git add/rm <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	deleted:    LABELING_GUIDE.md
	modified:   module1/pretrain/train.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	configs/pretrain_final.yaml
	session-ses_33bc.md

no changes added to commit (use "git add" and/or "git commit -a")

```

**Tool: bash**

**Input:**
```json
{
  "command": "git diff",
  "description": "Check git diff"
}
```

**Output:**
```
diff --git a/LABELING_GUIDE.md b/LABELING_GUIDE.md
deleted file mode 100644
index d19a49b..0000000
--- a/LABELING_GUIDE.md
+++ /dev/null
@@ -1,156 +0,0 @@
-# Football Injury Risk Labeling Guide
-
-## What Are We Doing?
-
-We are teaching an AI to recognize when a football player might be at risk of injury based on how they move. We need humans to watch short video clips and label them: "this movement looks safe" or "this movement looks risky."
-
----
-
-## Step 1: Collect Videos
-
-### What to record
-- Short clips of football movements (10-60 seconds each)
-- Good examples to include:
-  - **Running / sprinting**
-  - **Stopping quickly (deceleration)**
-  - **Changing direction (cutting)**
-  - **Jumping and landing**
-  - **Kicking**
-  - **Tackling / being tackled**
-
-### How to save
-1. Create a folder: `data/football/raw/`
-2. Name files clearly: `player_001_run.mp4`, `player_002_cut.mp4`
-3. Keep clips short (30-60 seconds max)
-
----
-
-## Step 2: Label Each Clip
-
-### Two Questions Per Clip
-
-**Question 1: Is the player at risk?**
-| Label | Meaning |
-|-------|---------|
-| `0` = Healthy | Good technique, safe movement |
-| `1` = At Risk | Poor technique, injury warning sign |
-
-**Question 2: Which body part? (optional)**
-| Label | Body Part |
-|-------|-----------|
-| `left_knee` | Left knee |
-| `right_knee` | Right knee |
-| `left_ankle` | Left ankle |
-| `right_ankle` | Right ankle |
-| `left_hamstring` | Left hamstring |
-| `right_hamstring` | Right hamstring |
-| `spine` | Back / spine |
-| `other` | Something else |
-
-### Example Labels
-
-| Video File | Risk (0/1) | Body Part | Notes |
-|------------|------------|-----------|-------|
-| player_001_run.mp4 | 0 | - | Good form |
-| player_002_cut.mp4 | 1 | left_knee | Knee caving inward |
-| player_003_jump.mp4 | 0 | - | Clean landing |
-| player_004_land.mp4 | 1 | right_ankle | Heavy landing |
-
----
-
-## Step 3: Use Label Studio (Easiest)
-
-### What is Label Studio?
-A free tool that lets you label data through a web browser. No coding needed (One Time)
-
-1. Go to.
-
-### Setup: https://labelstudio.io/
-2. Download and install Label Studio
-3. Open Label Studio in your browser (usually http://localhost:9090)
-4. Create an account
-
-### Create a Project
-
-1. Click **Create Project**
-2. Name it: "Football Injury Risk"
-3. Click **Data Import** → Upload all your video files
-4. Click **Labeling Config** → Copy and paste this:
-
-```xml
-<View>
-  <Header value="Is this player at risk of injury?"/>
-  <Choices name="risk" toName="video">
-    <Choice value="0 - Healthy" />
-    <Choice value="1 - At Risk" />
-  </Choices>
-  <Header value="Which body part is at risk?"/>
-  <Choices name="region" toName="video">
-    <Choice value="none" />
-    <Choice value="left_knee" />
-    <Choice value="right_knee" />
-    <Choice value="left_ankle" />
-    <Choice value="right_ankle" />
-    <Choice value="left_hamstring" />
-    <Choice value="right_hamstring" />
-    <Choice value="spine" />
-    <Choice value="other" />
-  </Choices>
-  <TextArea name="notes" toName="video"
-            placeholder="Add any notes about this clip..."
-            rows="3" />
-</View>
-```
-
-5. Click **Save**
-
-### Start Labeling
-
-1. Click **Label** to open a video
-2. Watch the video (play button)
-3. Select:
-   - **Risk**: 0 (Healthy) or 1 (At Risk)
-   - **Region**: Body part at risk (or "none")
-4. Click **Submit** → Next video
-5. Repeat!
-
-### Export Labels
-
-When done:
-1. Click **Export**
-2. Choose **CSV**
-3. Save the file as `football_labels.csv`
-
----
-
-## Step 4: Convert to Training Format
-
-Once labeling is done, we (the technical team) will:
-1. Convert your CSV labels to the format the AI needs
-2. Extract skeleton data from videos automatically
-3. Combine into training files
-
-You'll just give us: `football_labels.csv` + all video files
-
----
-
-## Summary
-
-| Step | What You Do | Output |
-|------|-------------|--------|
-| 1 | Record short football clips | Videos in a folder |
-| 2 | Watch each clip, label risk (0/1) + body part | Label Studio project |
-| 3 | Export as CSV | `football_labels.csv` |
-| 4 | Give us the videos + CSV | We handle the rest |
-
----
-
-## Need Help?
-
-- **Video too long?** → Trim to 30-60 seconds
-- **Unclear if at risk?** → Ask another person to label same clip
-- **Technical issues?** → Contact: [your email]
-
----
-
-**Thank you for helping!**
diff --git a/module1/pretrain/train.py b/module1/pretrain/train.py
index a074c78..b6925ba 100644
--- a/module1/pretrain/train.py
+++ b/module1/pretrain/train.py
@@ -1,4 +1,4 @@
-# module1/pretrain/train.py
+                                                            # module1/pretrain/train.py
 #
 # Stage 1 — ANUBIS Pretraining Loop
 # ──────────────────────────────────

```

**Tool: bash**

**Input:**
```json
{
  "command": "git log --oneline -5",
  "description": "Check recent commits"
}
```

**Output:**
```
d32659b Add docs folder with labeling guide and command reference
f74ea39 Add non-technical labeling workflow guide for football injury dataset
d71a314 Add Optuna-based hyperparameter tuning for pretraining
24c1d0f Increase ANUBIS pretraining epochs to 120
0fd74f9 Tune pretrain defaults for better generalization and add feature stats script

```

---

## Assistant (Build · minimax-m2.5-free · 20.9s)

_Thinking:_

I see there are some changes:
1. A new config file `configs/pretrain_final.yaml` - this is what we just created
2. `module1/pretrain/train.py` has a weird modification - it added some spaces at the beginning of line 1
3. `LABELING_GUIDE.md` was deleted

I need to add the new config file and the train.py change is unintentional (just whitespace). Let me check what's going on with the deleted file - it seems like it was moved to docs/ folder.

Let me check the docs folder:

**Tool: glob**

**Input:**
```json
{
  "pattern": "docs/**/*.md"
}
```

**Output:**
```
/home/alan/mydev/college/football-injury-ai/docs/COMMANDS.md
/home/alan/mydev/college/football-injury-ai/docs/LABELING_GUIDE.md
```

---

## Assistant (Build · minimax-m2.5-free · 5.9s)

_Thinking:_

Ah, the LABELING_GUIDE.md was moved to docs/ folder. So it's not really deleted, just moved. Let me stage the new config file and commit it.

**Tool: bash**

**Input:**
```json
{
  "command": "git add configs/pretrain_final.yaml",
  "description": "Stage new config file"
}
```

---

## Assistant (Build · minimax-m2.5-free · 6.7s)

_Thinking:_

Now let me commit with a clear message about the new config file for final pretraining.

**Tool: bash**

**Input:**
```json
{
  "command": "git commit -m \"Add pretrain_final.yaml: final pretraining config with Optuna best params and official val split\"",
  "description": "Commit new config"
}
```

**Output:**
```
[master 5349c71] Add pretrain_final.yaml: final pretraining config with Optuna best params and official val split
 1 file changed, 106 insertions(+)
 create mode 100644 configs/pretrain_final.yaml

```

---

## Assistant (Build · minimax-m2.5-free · 13.0s)

_Thinking:_

Now push to remote.

**Tool: bash**

**Input:**
```json
{
  "command": "git push",
  "description": "Push to remote"
}
```

**Output:**
```
To github.com:alanfrancis442/football-injury-ai.git
   d32659b..5349c71  master -> master

```

---

## Assistant (Build · minimax-m2.5-free · 6.4s)

_Thinking:_

Done. Let me give the user a summary.

Done. Pushed `configs/pretrain_final.yaml` to master.

On your GPU machine, run:
```bash
git pull
python module1/pretrain/train.py --config configs/pretrain_final.yaml
```

Then evaluate:
```bash
python module1/pretrain/evaluate.py --config configs/pretrain_final.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --split val --tsne
```

---

## User

so while the model is training can you create me the testing pipeline where we run the model on a live footage to get the result from the model

---

## Assistant (Plan · gpt-5.3-codex · 6.5s)

_Thinking:_

**Planning testing pipeline inspection**

**Tool: glob**

**Input:**
```json
{
  "pattern": "module1/**/*"
}
```

**Output:**
```
/home/alan/mydev/college/football-injury-ai/module1/pretrain/train.py
/home/alan/mydev/college/football-injury-ai/module1/pretrain/__pycache__/tune_optuna.cpython-314.pyc
/home/alan/mydev/college/football-injury-ai/module1/pretrain/tune_optuna.py
/home/alan/mydev/college/football-injury-ai/module1/pretrain/evaluate.py
/home/alan/mydev/college/football-injury-ai/module1/data/__pycache__/preprocess.cpython-312.pyc
/home/alan/mydev/college/football-injury-ai/module1/data/__pycache__/anubis_loader.cpython-312.pyc
/home/alan/mydev/college/football-injury-ai/module1/data/anubis_loader.py
/home/alan/mydev/college/football-injury-ai/module1/data/preprocess.py
/home/alan/mydev/college/football-injury-ai/module1/pretrain/__pycache__/model.cpython-312.pyc
/home/alan/mydev/college/football-injury-ai/module1/pretrain/__pycache__/__init__.cpython-312.pyc
/home/alan/mydev/college/football-injury-ai/module1/data/__pycache__/joint_config.cpython-312.pyc
/home/alan/mydev/college/football-injury-ai/module1/data/__pycache__/__init__.cpython-312.pyc
/home/alan/mydev/college/football-injury-ai/module1/__pycache__/__init__.cpython-312.pyc
/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py
/home/alan/mydev/college/football-injury-ai/module1/data/joint_config.py
/home/alan/mydev/college/football-injury-ai/module1/pretrain/__init__.py
/home/alan/mydev/college/football-injury-ai/module1/pose/__init__.py
/home/alan/mydev/college/football-injury-ai/module1/data/__init__.py
/home/alan/mydev/college/football-injury-ai/module1/__init__.py
```

---

## Assistant (Plan · gpt-5.3-codex · 14.3s)

_Thinking:_

**Planning live footage testing pipeline**

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/module1/data/joint_config.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/module1/data/joint_config.py</path>
<type>file</type>
<content>1: # module1/data/joint_config.py
2: #
3: # Stage 2 — ANUBIS 32-Joint Skeleton Configuration
4: # ──────────────────────────────────────────────────
5: # Defines the 32-joint skeleton used by the preprocessed ANUBIS dataset:
6: #   • Joint names and indices
7: #   • Skeleton edges for the Graph Attention Network
8: #   • Left/right mirror pairs for horizontal-flip augmentation
9: #   • Biomechanical angle triplets for diagnostics
10: #   • Body-region groupings for fine-tuning (injury localisation head)
11: #
12: # IMPORTANT — verify joint indices against your preprocessing notebook.
13: # The indices here follow a standard 32-joint motion-capture hierarchy
14: # (pelvis-rooted, pelvis coordinate already subtracted during preprocessing).
15: # If your ANUBIS preprocessing used a different joint ordering, update
16: # JOINT_NAMES and SKELETON_EDGES accordingly; all downstream code will
17: # automatically pick up the change.
18: 
19: from __future__ import annotations
20: 
21: from typing import Dict, List, Tuple
22: 
23: import torch
24: 
25: 
26: # ── Joint names (index → label) ───────────────────────────────────────────────
27: # Hierarchy:
28: #   Pelvis (0)
29: #   ├─ Spine chain:  0→1→2→3→4→5   (Pelvis→Spine1→Spine2→Spine3→Neck→Head)
30: #   ├─ Left leg:     0→6→7→8        (Pelvis→L_Hip→L_Knee→L_Ankle)
31: #   │                  8→9          (L_Ankle→L_Heel)
32: #   │                  8→10         (L_Ankle→L_FootIndex)
33: #   │                  10→28        (L_FootIndex→L_Toes)
34: #   ├─ Right leg:    0→11→12→13     (Pelvis→R_Hip→R_Knee→R_Ankle)
35: #   │                  13→14        (R_Ankle→R_Heel)
36: #   │                  13→15        (R_Ankle→R_FootIndex)
37: #   │                  15→29        (R_FootIndex→R_Toes)
38: #   ├─ Left arm:     3→16→17→18→19→20→21
39: #   │               (Spine3→L_Collar→L_Shoulder→L_Elbow→L_Wrist→L_Hand→L_Fingertip)
40: #   └─ Right arm:    3→22→23→24→25→26→27
41: #                   (Spine3→R_Collar→R_Shoulder→R_Elbow→R_Wrist→R_Hand→R_Fingertip)
42: #   Extra:           28→30 (L_Toes→L_ThumbToe), 29→31 (R_Toes→R_ThumbToe)
43: 
44: JOINT_NAMES: List[str] = [
45:     "Pelvis",  # 0  — root, always at origin after normalisation
46:     "Spine1",  # 1  — lower lumbar
47:     "Spine2",  # 2  — mid lumbar
48:     "Spine3",  # 3  — thorax / upper spine
49:     "Neck",  # 4
50:     "Head",  # 5
51:     "L_Hip",  # 6
52:     "L_Knee",  # 7
53:     "L_Ankle",  # 8
54:     "L_Heel",  # 9
55:     "L_FootIndex",  # 10 — ball of left foot
56:     "R_Hip",  # 11
57:     "R_Knee",  # 12
58:     "R_Ankle",  # 13
59:     "R_Heel",  # 14
60:     "R_FootIndex",  # 15 — ball of right foot
61:     "L_Collar",  # 16 — left clavicle
62:     "L_Shoulder",  # 17
63:     "L_Elbow",  # 18
64:     "L_Wrist",  # 19
65:     "L_Hand",  # 20
66:     "L_Fingertip",  # 21
67:     "R_Collar",  # 22 — right clavicle
68:     "R_Shoulder",  # 23
69:     "R_Elbow",  # 24
70:     "R_Wrist",  # 25
71:     "R_Hand",  # 26
72:     "R_Fingertip",  # 27
73:     "L_Toes",  # 28 — left toe base
74:     "R_Toes",  # 29 — right toe base
75:     "L_ThumbToe",  # 30 — left big-toe tip
76:     "R_ThumbToe",  # 31 — right big-toe tip
77: ]
78: 
79: NUM_JOINTS: int = len(JOINT_NAMES)  # 32
80: 
81: 
82: # ── Skeleton edges ─────────────────────────────────────────────────────────────
83: # Each tuple is (parent_index, child_index).
84: # Edges are made bidirectional inside _build_edge_index().
85: 
86: SKELETON_EDGES: List[Tuple[int, int]] = [
87:     # Spine / head chain
88:     (0, 1),
89:     (1, 2),
90:     (2, 3),
91:     (3, 4),
92:     (4, 5),
93:     # Pelvis to hips
94:     (0, 6),
95:     (0, 11),
96:     # Left leg
97:     (6, 7),  # hip → knee
98:     (7, 8),  # knee → ankle
99:     (8, 9),  # ankle → heel
100:     (8, 10),  # ankle → ball of foot
101:     (10, 28),  # ball of foot → toe base
102:     (28, 30),  # toe base → big-toe tip
103:     # Right leg
104:     (11, 12),
105:     (12, 13),
106:     (13, 14),
107:     (13, 15),
108:     (15, 29),
109:     (29, 31),
110:     # Spine3 to collars
111:     (3, 16),
112:     (3, 22),
113:     # Left arm
114:     (16, 17),  # collar → shoulder
115:     (17, 18),  # shoulder → elbow
116:     (18, 19),  # elbow → wrist
117:     (19, 20),  # wrist → hand
118:     (20, 21),  # hand → fingertip
119:     # Right arm
120:     (22, 23),
121:     (23, 24),
122:     (24, 25),
123:     (25, 26),
124:     (26, 27),
125: ]
126: 
127: NUM_EDGES: int = len(SKELETON_EDGES)  # 31 unique edges → 62 directed
128: 
129: 
130: def build_edge_index() -> torch.Tensor:
131:     """
132:     Build a bidirectional edge index tensor for the ANUBIS skeleton.
133: 
134:     Returns
135:     -------
136:     edge_index : torch.Tensor  shape (2, 2 * NUM_EDGES)
137:         Directed edge index compatible with PyTorch Geometric.
138:     """
139:     src, dst = zip(*SKELETON_EDGES)
140:     edge_src = list(src) + list(dst)  # forward + reverse
141:     edge_dst = list(dst) + list(src)
142:     return torch.tensor([edge_src, edge_dst], dtype=torch.long)
143: 
144: 
145: # Pre-built edge index — computed once at import time, CPU tensor.
146: # The model registers this as a buffer so it moves to GPU automatically.
147: EDGE_INDEX: torch.Tensor = build_edge_index()  # shape (2, 62)
148: 
149: 
150: # ── Left / right mirror pairs ─────────────────────────────────────────────────
151: # Used by the horizontal-flip augmentation: swap these joint indices AND
152: # negate the x-coordinate (feature index 0) + vx (index 3) + ax (index 6).
153: 
154: MIRROR_JOINT_PAIRS: List[Tuple[int, int]] = [
155:     (6, 11),  # L_Hip        ↔ R_Hip
156:     (7, 12),  # L_Knee       ↔ R_Knee
157:     (8, 13),  # L_Ankle      ↔ R_Ankle
158:     (9, 14),  # L_Heel       ↔ R_Heel
159:     (10, 15),  # L_FootIndex  ↔ R_FootIndex
160:     (16, 22),  # L_Collar     ↔ R_Collar
161:     (17, 23),  # L_Shoulder   ↔ R_Shoulder
162:     (18, 24),  # L_Elbow      ↔ R_Elbow
163:     (19, 25),  # L_Wrist      ↔ R_Wrist
164:     (20, 26),  # L_Hand       ↔ R_Hand
165:     (21, 27),  # L_Fingertip  ↔ R_Fingertip
166:     (28, 29),  # L_Toes       ↔ R_Toes
167:     (30, 31),  # L_ThumbToe   ↔ R_ThumbToe
168: ]
169: 
170: # Feature indices that flip sign when the skeleton is mirrored left-right.
171: # For feature layout [x, y, z, vx, vy, vz, ax, ay, az]: x→0, vx→3, ax→6.
172: FLIP_FEATURE_INDICES: List[int] = [0, 3, 6]
173: 
174: 
175: # ── Biomechanical angle triplets ──────────────────────────────────────────────
176: # (joint_idx, proximal_neighbour, distal_neighbour)
177: # Angle at joint_idx is the angle formed by the vectors:
178: #   joint → proximal   and   joint → distal
179: # Used for diagnostic visualisation and optional feature engineering.
180: 
181: ANGLE_TRIPLETS: Dict[int, Tuple[int, int]] = {
182:     7: (6, 8),  # L_Knee:   L_Hip  → L_Knee  → L_Ankle
183:     12: (11, 13),  # R_Knee:   R_Hip  → R_Knee  → R_Ankle
184:     6: (0, 7),  # L_Hip:    Pelvis → L_Hip   → L_Knee
185:     11: (0, 12),  # R_Hip:    Pelvis → R_Hip   → R_Knee
186:     8: (7, 9),  # L_Ankle:  L_Knee → L_Ankle → L_Heel
187:     13: (12, 14),  # R_Ankle:  R_Knee → R_Ankle → R_Heel
188:     18: (17, 19),  # L_Elbow:  L_Shoulder → L_Elbow → L_Wrist
189:     24: (23, 25),  # R_Elbow:  R_Shoulder → R_Elbow → R_Wrist
190:     1: (0, 2),  # Spine1:   Pelvis → Spine1 → Spine2  (trunk flexion)
191: }
192: 
193: 
194: # ── Body-region groupings (for fine-tuning injury-localisation head) ──────────
195: # Maps region name → list of joint indices most representative of that region.
196: # The transformer's CLS embedding is supervised against these labels in Stage 3.
197: 
198: BODY_REGIONS: Dict[str, List[int]] = {
199:     "left_hip": [6, 7],  # L_Hip + L_Knee (proximal hamstring)
200:     "right_hip": [11, 12],  # R_Hip + R_Knee
201:     "left_hamstring": [6, 7, 8],  # hip–knee–ankle chain
202:     "right_hamstring": [11, 12, 13],
203:     "left_knee": [7, 8],  # L_Knee + L_Ankle
204:     "right_knee": [12, 13],
205:     "left_ankle": [8, 9, 10],  # L_Ankle + L_Heel + L_FootIndex
206:     "right_ankle": [13, 14, 15],
207:     "spine": [0, 1, 2, 3],
208:     "left_shoulder": [16, 17],
209:     "right_shoulder": [22, 23],
210: }
211: 
212: REGION_NAMES: List[str] = list(BODY_REGIONS.keys())
213: NUM_REGIONS: int = len(REGION_NAMES)  # 11
214: 
215: 
216: # ── Quick sanity check ────────────────────────────────────────────────────────
217: 
218: if __name__ == "__main__":
219:     ei = build_edge_index()
220:     assert ei.shape == (2, 2 * NUM_EDGES), f"Unexpected edge_index shape: {ei.shape}"
221:     assert len(JOINT_NAMES) == NUM_JOINTS == 32
222: 
223:     # Every edge node index must be in [0, NUM_JOINTS)
224:     assert ei.max().item() < NUM_JOINTS
225:     assert ei.min().item() >= 0
226: 
227:     # Every mirror pair must reference valid joints
228:     for l, r in MIRROR_JOINT_PAIRS:
229:         assert 0 <= l < NUM_JOINTS and 0 <= r < NUM_JOINTS
230: 
231:     print(f"NUM_JOINTS  : {NUM_JOINTS}")
232:     print(f"NUM_EDGES   : {NUM_EDGES} unique ({2 * NUM_EDGES} directed)")
233:     print(f"NUM_REGIONS : {NUM_REGIONS}")
234:     print("joint_config OK.")

(End of file - total 234 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/configs/module1.yaml"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/configs/module1.yaml</path>
<type>file</type>
<content>1: # configs/module1.yaml
2: #
3: # Module 1 — Biomechanical Risk Monitor  (fine-tuning / inference config)
4: # ────────────────────────────────────────────────────────────────────────
5: # Hyperparameters for fine-tuning the pretrained SpatioTemporalEncoder on
6: # labelled football clips and for real-time inference.
7: #
8: # Pretraining hyperparameters are in configs/pretrain.yaml.
9: # Never hardcode any of these values inside Python files.
10: 
11: # ── File paths ────────────────────────────────────────────────────────────────
12: paths:
13:   raw_dir:           data/raw/
14:   processed_dir:     data/processed/
15:   samples_dir:       data/samples/
16:   checkpoint_dir:    outputs/checkpoints/
17:   log_dir:           outputs/logs/
18:   pretrained_encoder: outputs/checkpoints/pretrain/pretrain_best.pt
19: 
20: # ── Skeleton / data format ────────────────────────────────────────────────────
21: data:
22:   num_joints:   32       # ANUBIS 32-joint pelvis-rooted skeleton
23:   in_features:   9       # per-joint features: [x,y,z, vx,vy,vz, ax,ay,az]
24:   seq_len:      60       # frames per sliding window fed to the model
25:   num_classes:   2       # binary: healthy (0) / at-risk (1)
26: 
27: # ── Model (must match pretrained encoder dims) ────────────────────────────────
28: model:
29:   gat_hidden_dim:          64
30:   gat_heads:                4
31:   gat_dropout:            0.1
32:   d_model:                128
33:   nhead:                    4
34:   num_transformer_layers:   2
35:   dim_feedforward:        256
36:   transformer_dropout:    0.1
37: 
38: # ── Fine-tuning ───────────────────────────────────────────────────────────────
39: training:
40:   epochs:            30
41:   batch_size:        32
42:   learning_rate:  0.0001   # lower LR for fine-tuning
43:   weight_decay:  0.0001
44:   warmup_epochs:      2
45:   grad_clip_norm:   1.0
46:   label_smoothing:  0.05
47: 
48:   # Class imbalance — at-risk samples are rare; increase if model misses them
49:   class_weight_healthy: 1.0
50:   class_weight_at_risk: 4.0
51: 
52: # ── Body-region localisation head ────────────────────────────────────────────
53: region_head:
54:   num_regions:         11   # matches NUM_REGIONS in joint_config.py
55:   region_loss_weight: 0.3   # weight of region-classification loss
56: 
57: # ── Inference / risk thresholds ───────────────────────────────────────────────
58: inference:
59:   risk_tier_thresholds:
60:     green:  0.30
61:     amber:  0.55
62:     orange: 0.75
63:     # above 0.75 → RED
64:   sustained_window: 3    # consecutive windows before escalating alert tier

(End of file - total 64 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/README.md"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/README.md</path>
<type>file</type>
<content>1: # Football Player Injury Risk Prediction — Module 1
2: 
3: Biomechanical Risk Monitor using Pose Estimation, GNN, and Transformer.
4: 
5: ---
6: 
7: ## Project Structure
8: 
9: ```
10: football-injury-ai/
11: ├── module1/            # Biomechanical Risk Monitor (active build)
12: │   ├── pose/           # Stage 1 — keypoint extraction & player tracking
13: │   ├── data/           # Data loading & preprocessing
14: │   └── pretrain/       # Training, evaluation & tuning scripts
15: ├── module2/            # Placeholder — Acute Workload Monitor
16: ├── module3/            # Placeholder — Historical Risk Profile
17: ├── fusion/             # Placeholder — Risk Fusion Engine
18: ├── docs/               # Documentation
19: │   ├── COMMANDS.md     # All commands reference
20: │   └── LABELING_GUIDE.md  # How to label data
21: ├── data/
22: │   ├── raw/            # Original video clips / downloaded datasets
23: │   ├── processed/      # Extracted keypoints (.npy / .csv)
24: │   ├── samples/        # Short clips for fast dev iteration
25: │   └── anubis/         # ANUBIS dataset files
26: ├── notebooks/          # Jupyter experimentation sandbox
27: ├── tests/              # Unit + integration tests
28: ├── configs/            # Hyperparameters (edit here, not in code)
29: └── outputs/
30:     ├── checkpoints/    # Saved model weights (.pt)
31:     └── logs/          # Training logs
32: ```
33: 
34: ---
35: 
36: ## Quick Links
37: 
38: | Guide | Description |
39: |-------|-------------|
40: | [docs/COMMANDS.md](docs/COMMANDS.md) | All commands for training, testing, evaluation |
41: | [docs/LABELING_GUIDE.md](docs/LABELING_GUIDE.md) | How to label football clips (for non-tech friends) |
42: 
43: ---
44: 
45: ## Setup
46: 
47: ### 1. Install dependencies
48: 
49: ```bash
50: uv sync
51: ```
52: 
53: ### 2. Install PyTorch Geometric extras
54: 
55: CPU:
56: ```bash
57: uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
58: ```
59: 
60: GPU (adjust CUDA version):
61: ```bash
62: uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
63: ```
64: 
65: ---
66: 
67: ## Common Commands
68: 
69: ```bash
70: # Train ANUBIS pretraining
71: uv run module1/pretrain/train.py --config configs/pretrain.yaml
72: 
73: # Evaluate
74: uv run module1/pretrain/evaluate.py --config configs/pretrain.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt
75: 
76: # Run Optuna tuning
77: uv run module1/pretrain/tune_optuna.py --config configs/pretrain.yaml --n-trials 30
78: 
79: # Run tests
80: uv run pytest tests/
81: 
82: # Check data stats
83: uv run scripts/anubis_label_stats.py
84: uv run scripts/anubis_feature_stats.py
85: ```
86: 
87: See [docs/COMMANDS.md](docs/COMMANDS.md) for the full command list.
88: 
89: ---
90: 
91: ## Development Workflow
92: 
93: 1. **Experiment** in `notebooks/` — break things freely here
94: 2. **Port clean code** into the matching `module1/` file once it works
95: 3. **Adjust hyperparameters** in `configs/pretrain.yaml` — never hardcode them
96: 4. **Verify nothing broke** with `uv run pytest tests/`

(End of file - total 96 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/docs/COMMANDS.md"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/docs/COMMANDS.md</path>
<type>file</type>
<content>1: # Football Injury AI — Command Reference
2: 
3: This guide covers all commands you need to train, evaluate, and work with the project.
4: 
5: ---
6: 
7: ## Quick Start
8: 
9: ### 1. Install Dependencies
10: 
11: ```bash
12: # Clone the repo
13: git clone https://github.com/alanfrancis442/football-injury-ai.git
14: cd football-injury-ai
15: 
16: # Install all dependencies
17: uv sync
18: 
19: # Install PyTorch Geometric extras (CPU)
20: uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
21: 
22: # Or for GPU (adjust CUDA version)
23: uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
24: ```
25: 
26: ---
27: 
28: ## Data Setup
29: 
30: ### Download ANUBIS Dataset
31: 
32: Place these files in `data/anubis/`:
33: 
34: ```
35: data/anubis/
36: ├── trn_features_cleaned.npy   # Training features
37: ├── trn_labels_cleaned.npy     # Training labels
38: ├── val_features_cleaned.npy   # Validation features
39: └── val_labels_cleaned.npy     # Validation labels
40: ```
41: 
42: ### Check Data Stats
43: 
44: ```bash
45: # Label distribution
46: uv run scripts/anubis_label_stats.py
47: 
48: # Feature distribution
49: uv run scripts/anubis_feature_stats.py
50: ```
51: 
52: ---
53: 
54: ## Training (Stage 1: ANUBIS Pretraining)
55: 
56: ### Standard Training
57: 
58: ```bash
59: uv run module1/pretrain/train.py --config configs/pretrain.yaml
60: ```
61: 
62: ### With Custom Checkpoint Path
63: 
64: ```bash
65: uv run module1/pretrain/train.py --config configs/pretrain.yaml --checkpoint outputs/checkpoints/pretrain/my_run.pt
66: ```
67: 
68: ### Hyperparameter Tuning (Optuna)
69: 
70: ```bash
71: # Run 30 trials
72: uv run module1/pretrain/tune_optuna.py --config configs/pretrain.yaml --n-trials 30
73: 
74: # Run with custom timeout (e.g., 2 hours)
75: uv run module1/pretrain/tune_optuna.py --config configs/pretrain.yaml --timeout-sec 7200
76: ```
77: 
78: ### Train with Best Tuned Parameters
79: 
80: ```bash
81: uv run module1/pretrain/train.py --config outputs/logs/pretrain_optuna_best.yaml
82: ```
83: 
84: ---
85: 
86: ## Evaluation
87: 
88: ### Evaluate a Checkpoint
89: 
90: ```bash
91: # Basic evaluation
92: uv run module1/pretrain/evaluate.py --config configs/pretrain.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt
93: 
94: # With t-SNE visualization
95: uv run module1/pretrain/evaluate.py --config configs/pretrain.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --tsne
96: ```
97: 
98: ### Checkpoint Locations
99: 
100: | Checkpoint | Path |
101: |------------|------|
102: | Best (by val loss) | `outputs/checkpoints/pretrain/pretrain_best.pt` |
103: | Periodic (every 10 epochs) | `outputs/checkpoints/pretrain/pretrain_ep{N:04d}.pt` |
104: | Optuna best | `outputs/logs/pretrain_optuna_best.yaml` |
105: 
106: ---
107: 
108: ## Testing
109: 
110: ### Run All Tests
111: 
112: ```bash
113: uv run pytest tests/
114: ```
115: 
116: ### Run Specific Test File
117: 
118: ```bash
119: uv run pytest tests/test_model.py -v
120: uv run pytest tests/test_anubis_loader.py -v
121: uv run pytest tests/test_joint_config.py -v
122: ```
123: 
124: ### Run Single Test
125: 
126: ```bash
127: uv run pytest tests/test_model.py::TestSpatioTemporalEncoder::test_output_shape -v
128: ```
129: 
130: ---
131: 
132: ## Model Smoke Tests
133: 
134: Quick sanity checks without data:
135: 
136: ```bash
137: # Test model forward pass
138: uv run module1/pretrain/model.py
139: ```
140: 
141: ---
142: 
143: ## Project Structure
144: 
145: ```
146: football-injury-ai/
147: ├── module1/
148: │   ├── data/              # Data loading & preprocessing
149: │   ├── pretrain/          # Training & evaluation scripts
150: │   └── pose/              # Pose extraction (future)
151: ├── configs/
152: │   ├── pretrain.yaml      # ANUBIS pretraining config
153: │   └── module1.yaml       # Fine-tuning config
154: ├── scripts/               # Utility scripts
155: │   ├── anubis_label_stats.py
156: │   └── anubis_feature_stats.py
157: ├── docs/
158: │   ├── LABELING_GUIDE.md  # How to label data
159: │   └── COMMANDS.md        # This file
160: ├── outputs/
161: │   ├── checkpoints/       # Saved models
162: │   └── logs/              # Training logs
163: └── tests/                 # Unit tests
164: ```
165: 
166: ---
167: 
168: ## Common Tasks
169: 
170: ### Check GPU Availability
171: 
172: ```bash
173: python -c "import torch; print('CUDA:', torch.cuda.is_available())"
174: ```
175: 
176: ### View Training Logs
177: 
178: ```bash
179: cat outputs/logs/run.log
180: ```
181: 
182: ### Delete Old Checkpoints
183: 
184: ```bash
185: rm outputs/checkpoints/pretrain/pretrain_ep*.pt
186: ```
187: 
188: ---
189: 
190: ## Troubleshooting
191: 
192: ### "No module named 'torch'"
193: 
194: ```bash
195: uv sync
196: ```
197: 
198: ### "CUDA out of memory"
199: 
200: Reduce batch size in config:
201: ```yaml
202: training:
203:   batch_size: 32  # Try 16 or 8
204: ```
205: 
206: ### "Checkpoint not found"
207: 
208: Make sure you're running from the project root directory:
209: ```bash
210: cd football-injury-ai
211: uv run module1/pretrain/train.py --config configs/pretrain.yaml
212: ```
213: 
214: ---
215: 
216: ## Next Steps
217: 
218: After ANUBIS pretraining:
219: 
220: 1. **Collect football clips** — see `docs/LABELING_GUIDE.md`
221: 2. **Label the clips** — use Label Studio
222: 3. **Fine-tune on football data** — see future updates
223: 
224: ---
225: 
226: For questions, check the main README.md or open an issue on GitHub.

(End of file - total 226 lines)
</content>
```

---

## Assistant (Plan · gpt-5.3-codex · 10.3s)

_Thinking:_

**Planning live inference pipeline**

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/utils/config.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/utils/config.py</path>
<type>file</type>
<content>1: # utils/config.py
2: #
3: # Config loader — reads configs/module1.yaml and returns a plain dict.
4: # Import this in any script that needs hyperparameters.
5: #
6: # Usage:
7: #   from utils.config import load_config
8: #   cfg = load_config("configs/module1.yaml")
9: #   lr  = cfg["training"]["learning_rate"]
10: 
11: import os
12: import yaml
13: 
14: 
15: def load_config(config_path: str) -> dict:
16:     """
17:     Load a YAML config file and return it as a nested dict.
18: 
19:     Parameters
20:     ----------
21:     config_path : str — path to the .yaml file (relative to project root)
22: 
23:     Raises
24:     ------
25:     FileNotFoundError if the config file does not exist.
26:     """
27:     if not os.path.exists(config_path):
28:         raise FileNotFoundError(
29:             f"Config not found: {config_path}\n"
30:             "Make sure you are running commands from the project root directory."
31:         )
32: 
33:     with open(config_path, "r") as f:
34:         cfg = yaml.safe_load(f)
35: 
36:     return cfg

(End of file - total 36 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/utils/logger.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/utils/logger.py</path>
<type>file</type>
<content>1: # utils/logger.py
2: #
3: # Centralised logger — use this instead of print() in all modules.
4: # Writes to both the console and outputs/logs/run.log
5: #
6: # Usage:
7: #   from utils.logger import get_logger
8: #   log = get_logger(__name__)
9: #   log.info("Training started")
10: #   log.warning("No checkpoint found")
11: #   log.error("File not found")
12: 
13: import logging
14: import os
15: from datetime import datetime
16: 
17: 
18: LOG_DIR = "outputs/logs"
19: LOG_FILE = os.path.join(LOG_DIR, "run.log")
20: 
21: 
22: def get_logger(name: str) -> logging.Logger:
23:     """
24:     Returns a configured logger that writes to console + log file.
25: 
26:     Parameters
27:     ----------
28:     name : str — typically pass __name__ from the calling module
29:     """
30:     os.makedirs(LOG_DIR, exist_ok=True)
31: 
32:     logger = logging.getLogger(name)
33: 
34:     # Avoid adding duplicate handlers if called multiple times
35:     if logger.handlers:
36:         return logger
37: 
38:     logger.setLevel(logging.DEBUG)
39: 
40:     fmt = logging.Formatter(
41:         fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
42:         datefmt="%H:%M:%S",
43:     )
44: 
45:     # Console handler
46:     console = logging.StreamHandler()
47:     console.setLevel(logging.INFO)
48:     console.setFormatter(fmt)
49: 
50:     # File handler
51:     file_handler = logging.FileHandler(LOG_FILE)
52:     file_handler.setLevel(logging.DEBUG)
53:     file_handler.setFormatter(fmt)
54: 
55:     logger.addHandler(console)
56:     logger.addHandler(file_handler)
57: 
58:     return logger

(End of file - total 58 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/pyproject.toml"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/pyproject.toml</path>
<type>file</type>
<content>1: [project]
2: name = "football-injury-ai"
3: version = "0.1.0"
4: description = "Real-time football player injury risk prediction using pose estimation and deep learning"
5: requires-python = ">=3.12"
6: dependencies = [
7:     # Core
8:     "numpy>=1.24",
9:     "pandas>=2.0",
10:     "pyyaml>=6.0",
11: 
12:     # Computer Vision & Pose Estimation
13:     "opencv-python>=4.8",
14:     "mediapipe>=0.10",
15: 
16:     # Deep Learning
17:     "torch>=2.1",
18:     "torchvision>=0.16",
19: 
20:     # Graph Neural Networks
21:     "torch-geometric>=2.4",
22:     # torch-scatter and torch-sparse installed separately:
23:     # pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
24: 
25:     # Visualisation
26:     "matplotlib>=3.7",
27:     "seaborn>=0.13",
28:     "networkx>=3.2",
29: 
30:     # Evaluation & Utilities
31:     "scikit-learn>=1.3",
32:     "tqdm>=4.65",
33:     "optuna>=3.6",
34: 
35:     # Notebooks
36:     "jupyter>=1.0",
37:     "ipykernel>=6.0",
38: 
39:     # Testing
40:     "pytest>=7.4",
41: ]
42: 
43: [project.optional-dependencies]
44: dev = [
45:     "pytest>=7.4",
46:     "black>=23.0",
47:     "ruff>=0.1",
48: ]
49: 
50: [build-system]
51: requires = ["hatchling"]
52: build-backend = "hatchling.build"
53: 
54: [tool.pytest.ini_options]
55: testpaths = ["tests"]
56: python_files = ["test_*.py"]
57: python_classes = ["Test*"]
58: python_functions = ["test_*"]
59: 
60: [tool.ruff]
61: line-length = 88
62: target-version = "py312"
63: 
64: [tool.black]
65: line-length = 88
66: target-version = ["py312"]

(End of file - total 66 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/tests/test_model.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/tests/test_model.py</path>
<type>file</type>
<content>1: # tests/test_model.py
2: #
3: # Unit tests for module1/pretrain/model.py
4: # ──────────────────────────────────────────
5: # Covers: PositionalEncoding, SpatioTemporalEncoder (forward pass shapes,
6: #         no NaNs, edge_index buffer device movement), ActionClassifier
7: #         (forward pass, logit shape), and encoder/classifier separation.
8: #
9: # Run with: pytest tests/test_model.py -v
10: 
11: import pytest
12: import torch
13: import torch.nn as nn
14: 
15: from module1.data.joint_config import EDGE_INDEX, NUM_JOINTS
16: from module1.pretrain.model import (
17:     ActionClassifier,
18:     PositionalEncoding,
19:     SpatioTemporalEncoder,
20:     build_model,
21: )
22: 
23: 
24: # ── Constants used across tests ───────────────────────────────────────────────
25: 
26: B = 2  # batch size (small to keep tests fast)
27: T = 60  # sequence length (frames)
28: J = NUM_JOINTS  # 32
29: F_IN = 9  # features per joint
30: D_MODEL = 128
31: NUM_CLASSES = 102
32: 
33: 
34: # ── Fixtures ──────────────────────────────────────────────────────────────────
35: 
36: 
37: @pytest.fixture
38: def edge_index() -> torch.Tensor:
39:     return EDGE_INDEX.clone()
40: 
41: 
42: @pytest.fixture
43: def encoder(edge_index: torch.Tensor) -> SpatioTemporalEncoder:
44:     model = SpatioTemporalEncoder(
45:         num_joints=J,
46:         in_features=F_IN,
47:         edge_index=edge_index,
48:         gat_hidden=64,
49:         gat_heads=4,
50:         d_model=D_MODEL,
51:         nhead=4,
52:         num_layers=2,
53:         dim_ff=256,
54:         dropout=0.0,  # disable dropout so outputs are deterministic in eval
55:         seq_len=T,
56:     )
57:     model.eval()
58:     return model
59: 
60: 
61: @pytest.fixture
62: def classifier(encoder: SpatioTemporalEncoder) -> ActionClassifier:
63:     model = ActionClassifier(encoder, num_classes=NUM_CLASSES)
64:     model.eval()
65:     return model
66: 
67: 
68: @pytest.fixture
69: def dummy_input() -> torch.Tensor:
70:     """Small random batch: (B, T, J, F_IN)."""
71:     torch.manual_seed(0)
72:     return torch.randn(B, T, J, F_IN)
73: 
74: 
75: @pytest.fixture
76: def minimal_cfg() -> dict:
77:     """Minimal config dict that mirrors configs/pretrain.yaml structure."""
78:     return {
79:         "data": {
80:             "num_joints": J,
81:             "seq_len": T,
82:             "num_classes": NUM_CLASSES,
83:         },
84:         "model": {
85:             "in_features": F_IN,
86:             "gat_hidden_dim": 64,
87:             "gat_heads": 4,
88:             "gat_dropout": 0.0,
89:             "d_model": D_MODEL,
90:             "nhead": 4,
91:             "num_transformer_layers": 2,
92:             "dim_feedforward": 256,
93:             "transformer_dropout": 0.0,
94:             "cls_std": 0.02,
95:         },
96:     }
97: 
98: 
99: # ── PositionalEncoding ────────────────────────────────────────────────────────
100: 
101: 
102: class TestPositionalEncoding:
103:     def test_output_shape(self) -> None:
104:         pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
105:         x = torch.zeros(B, T, D_MODEL)
106:         out = pe(x)
107:         assert out.shape == (B, T, D_MODEL)
108: 
109:     def test_no_nan(self) -> None:
110:         pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
111:         x = torch.randn(B, T, D_MODEL)
112:         out = pe(x)
113:         assert not torch.isnan(out).any()
114: 
115:     def test_adds_positional_signal(self) -> None:
116:         # Zero input + no dropout: output should equal the PE buffer, not zero
117:         pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
118:         pe.eval()
119:         x = torch.zeros(B, T, D_MODEL)
120:         out = pe(x)
121:         assert not torch.all(out == 0)
122: 
123:     def test_different_positions_differ(self) -> None:
124:         # Adjacent time steps should (almost certainly) have different encodings
125:         pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
126:         pe.eval()
127:         x = torch.zeros(1, T, D_MODEL)
128:         out = pe(x)
129:         assert not torch.allclose(out[0, 0, :], out[0, 1, :])
130: 
131: 
132: # ── SpatioTemporalEncoder ─────────────────────────────────────────────────────
133: 
134: 
135: class TestSpatioTemporalEncoder:
136:     @torch.no_grad()
137:     def test_output_shape(
138:         self, encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
139:     ) -> None:
140:         out = encoder(dummy_input)
141:         assert out.shape == (B, D_MODEL), f"Expected ({B},{D_MODEL}), got {out.shape}"
142: 
143:     @torch.no_grad()
144:     def test_no_nan_in_output(
145:         self, encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
146:     ) -> None:
147:         out = encoder(dummy_input)
148:         assert not torch.isnan(out).any(), "NaN detected in encoder output"
149: 
150:     @torch.no_grad()
151:     def test_no_inf_in_output(
152:         self, encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
153:     ) -> None:
154:         out = encoder(dummy_input)
155:         assert not torch.isinf(out).any(), "Inf detected in encoder output"
156: 
157:     @torch.no_grad()
158:     def test_different_inputs_give_different_outputs(
159:         self, encoder: SpatioTemporalEncoder
160:     ) -> None:
161:         torch.manual_seed(1)
162:         x1 = torch.randn(B, T, J, F_IN)
163:         x2 = torch.randn(B, T, J, F_IN)
164:         out1 = encoder(x1)
165:         out2 = encoder(x2)
166:         assert not torch.allclose(out1, out2), "Encoder is constant regardless of input"
167: 
168:     @torch.no_grad()
169:     def test_batch_independence(self, encoder: SpatioTemporalEncoder) -> None:
170:         # Encoding one sample individually must equal its row in a batched pass
171:         torch.manual_seed(2)
172:         x = torch.randn(B, T, J, F_IN)
173:         batch_out = encoder(x)
174:         single_out = encoder(x[0:1])
175:         assert torch.allclose(batch_out[0], single_out[0], atol=1e-5), (
176:             "Batch output differs from single-sample output"
177:         )
178: 
179:     def test_edge_index_is_buffer(self, encoder: SpatioTemporalEncoder) -> None:
180:         # edge_index must be registered as a buffer (non-parameter)
181:         buffer_names = {name for name, _ in encoder.named_buffers()}
182:         assert "edge_index" in buffer_names
183: 
184:     def test_edge_index_shape(self, encoder: SpatioTemporalEncoder) -> None:
185:         assert encoder.edge_index.shape == EDGE_INDEX.shape
186: 
187:     def test_has_no_unused_params_in_training_mode(
188:         self, encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
189:     ) -> None:
190:         """All parameters must receive a gradient after a backward pass."""
191:         encoder.train()
192:         out = encoder(dummy_input)
193:         loss = out.sum()
194:         loss.backward()
195:         for name, param in encoder.named_parameters():
196:             if param.requires_grad:
197:                 assert param.grad is not None, f"No gradient for parameter: {name}"
198:         encoder.eval()
199: 
200: 
201: # ── ActionClassifier ──────────────────────────────────────────────────────────
202: 
203: 
204: class TestActionClassifier:
205:     @torch.no_grad()
206:     def test_output_shape(
207:         self, classifier: ActionClassifier, dummy_input: torch.Tensor
208:     ) -> None:
209:         logits = classifier(dummy_input)
210:         assert logits.shape == (B, NUM_CLASSES), (
211:             f"Expected ({B},{NUM_CLASSES}), got {logits.shape}"
212:         )
213: 
214:     @torch.no_grad()
215:     def test_no_nan_in_logits(
216:         self, classifier: ActionClassifier, dummy_input: torch.Tensor
217:     ) -> None:
218:         logits = classifier(dummy_input)
219:         assert not torch.isnan(logits).any()
220: 
221:     @torch.no_grad()
222:     def test_logits_are_unnormalised(
223:         self, classifier: ActionClassifier, dummy_input: torch.Tensor
224:     ) -> None:
225:         # The head is a plain Linear — softmax probabilities should NOT sum to 1
226:         logits = classifier(dummy_input)
227:         row_sums = logits.softmax(dim=-1).sum(dim=-1)
228:         # After softmax they should sum to ~1, but raw logits sum should not
229:         assert not torch.allclose(logits.sum(dim=-1), torch.ones(B))
230: 
231:     def test_encoder_is_accessible(self, classifier: ActionClassifier) -> None:
232:         assert isinstance(classifier.encoder, SpatioTemporalEncoder)
233: 
234:     def test_encoder_and_classifier_share_params(
235:         self, classifier: ActionClassifier
236:     ) -> None:
237:         # Parameters of classifier.encoder must be a subset of classifier's params
238:         enc_ids = {id(p) for p in classifier.encoder.parameters()}
239:         cls_ids = {id(p) for p in classifier.parameters()}
240:         assert enc_ids.issubset(cls_ids)
241: 
242:     def test_head_not_in_encoder(self, classifier: ActionClassifier) -> None:
243:         enc_ids = {id(p) for p in classifier.encoder.parameters()}
244:         head_ids = {id(p) for p in classifier.head.parameters()}
245:         assert head_ids.isdisjoint(enc_ids), "Head parameters appear in encoder"
246: 
247: 
248: # ── build_model factory ───────────────────────────────────────────────────────
249: 
250: 
251: class TestBuildModel:
252:     def test_returns_encoder_and_classifier(self, minimal_cfg: dict) -> None:
253:         encoder, classifier = build_model(minimal_cfg, EDGE_INDEX)
254:         assert isinstance(encoder, SpatioTemporalEncoder)
255:         assert isinstance(classifier, ActionClassifier)
256: 
257:     def test_classifier_wraps_same_encoder(self, minimal_cfg: dict) -> None:
258:         encoder, classifier = build_model(minimal_cfg, EDGE_INDEX)
259:         assert classifier.encoder is encoder
260: 
261:     @torch.no_grad()
262:     def test_forward_pass_after_build(self, minimal_cfg: dict) -> None:
263:         _, classifier = build_model(minimal_cfg, EDGE_INDEX)
264:         classifier.eval()
265:         dummy = torch.randn(B, T, J, F_IN)
266:         logits = classifier(dummy)
267:         assert logits.shape == (B, NUM_CLASSES)
268:         assert not torch.isnan(logits).any()

(End of file - total 268 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/module1/pose/__init__.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/module1/pose/__init__.py</path>
<type>file</type>
<content>

(End of file - total 0 lines)
</content>
```

---

## Assistant (Plan · gpt-5.3-codex · 9.9s)

_Thinking:_

**Planning live inference pipeline**

**Tool: grep**

**Input:**
```json
{
  "pattern": "mediapipe|MediaPipe|cv2.VideoCapture|Pose\\(",
  "path": "/home/alan/mydev/college/football-injury-ai"
}
```

**Output:**
```
Found 15 matches
/home/alan/mydev/college/football-injury-ai/session-ses_33bc.md:
  Line 60:   "content": "[project]\nname = \"football-injury-ai\"\nversion = \"0.1.0\"\ndescription = \"Real-time football player injury risk prediction using pose estimation and deep learning\"\nrequires-python = \">=3.12\"\ndependencies = [\n    # Core\n    \"numpy>=1.24\",\n    \"pandas>=2.0\",\n    \"pyyaml>=6.0\",\n\n    # Computer Vision & Pose Estimation\n    \"opencv-python>=4.8\",\n    \"mediapipe>=0.10\",\n\n    # Deep Learning\n    \"torch>=2.1\",\n    \"torchvision>=0.16\",\n\n    # Graph Neural Networks\n    \"torch-geometric>=2.4\",\n    # torch-scatter and torch-sparse installed separately:\n    # pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html\n\n    # Visualisation\n    \"matplotlib>=3.7\",\n    \"seaborn>=0.13\",\n    \"networkx>=3.2\",\n\n    # Evaluation & Utilities\n    \"scikit-learn>=1.3\",\n    \"tqdm>=4.65\",\n\n    # Notebooks\n    \"jupyter>=1.0\",\n    \"ipykernel>=6.0\",\n\n    # Testing\n    \"pytest>=7.4\",\n]\n\n[project.optional-dependencies]\ndev = [\n    \"pytest>=7.4\",\n    \"black>=23.0\",\n    \"ruff>=0.1\",\n]\n\n[build-system]\nrequires = [\"hatchling\"]\nbuild-backend = \"hatchling.build\"\n\n[tool.pytest.ini_options]\ntestpaths = [\"tests\"]\npython_files = [\"test_*.py\"]\npython_classes = [\"Test*\"]\npython_functions = [\"test_*\"]\n\n[tool.ruff]\nline-length = 88\ntarget-version = \"py312\"\n\n[tool.black]\nline-length = 88\ntarget-version = [\"py312\"]\n",
  Line 104: ERROR [23:8] Import "mediapipe" could not be resolved
  Line 165: ERROR [23:8] Import "mediapipe" could not be resolved
  Line 2191: 14:     "mediapipe>=0.10",

/home/alan/mydev/college/football-injury-ai/pyproject.toml:
  Line 14:     "mediapipe>=0.10",

/home/alan/mydev/college/football-injury-ai/AGENTS.md:
  Line 10: prediction. Module 1 (active) processes video → MediaPipe pose keypoints →
  Line 14: Stack: Python 3, PyTorch, PyTorch Geometric, MediaPipe, OpenCV, NumPy, pandas,
  Line 209: (e.g., `PoseExtractor` wraps a MediaPipe `Pose` object).

/home/alan/mydev/college/football-injury-ai/notebooks/01_pose_exploration.ipynb:
  Line 9:     "**Goal:** Run MediaPipe Pose on a video clip, visualise the keypoints, and understand the data format before writing any model code.\n",
  Line 15:     "2. Extract keypoints with MediaPipe\n",
  Line 32:     "import mediapipe as mp\n",
  Line 49:     "cap = cv2.VideoCapture(VIDEO_PATH)\n",
  Line 71:     "# ── Run MediaPipe Pose on the frame ──────────────────────────────────────────\n",
  Line 75:     "with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:\n",
  Line 84:     "    plt.title('MediaPipe Pose — 33 keypoints')\n",
```

**Tool: grep**

**Input:**
```json
{
  "pattern": "inference|risk_tier|sustained_window|pretrained_encoder|region_head",
  "path": "/home/alan/mydev/college/football-injury-ai"
}
```

**Output:**
```
Found 15 matches
/home/alan/mydev/college/football-injury-ai/session-ses_33bc.md:
  Line 592: 7: # Fine-tuning / inference values belong in configs/module1.yaml.
  Line 863: So “real-world check” means: video -> pose extraction -> same preprocessing -> model inference.
  Line 878:   - Run inference and check:
  Line 1527: 7: # Fine-tuning / inference values belong in configs/module1.yaml.
  Line 1923: 7: # Fine-tuning / inference values belong in configs/module1.yaml.

/home/alan/mydev/college/football-injury-ai/configs/pretrain.yaml:
  Line 7: # Fine-tuning / inference values belong in configs/module1.yaml.

/home/alan/mydev/college/football-injury-ai/configs/module1.yaml:
  Line 3: # Module 1 — Biomechanical Risk Monitor  (fine-tuning / inference config)
  Line 6: # labelled football clips and for real-time inference.
  Line 18:   pretrained_encoder: outputs/checkpoints/pretrain/pretrain_best.pt
  Line 53: region_head:
  Line 58: inference:
  Line 59:   risk_tier_thresholds:
  Line 64:   sustained_window: 3    # consecutive windows before escalating alert tier

/home/alan/mydev/college/football-injury-ai/.agents/skills/pytorch-lightning/references/distributed.md:
  Line 351: # Load for inference

/home/alan/mydev/college/football-injury-ai/AGENTS.md:
  Line 185: - Call `.eval()` before inference, `.train()` at start of training loops
```

---

## Assistant (Plan · gpt-5.3-codex · 11.0s)

_Thinking:_

**Planning live testing pipeline**

**Tool: grep**

**Input:**
```json
{
  "pattern": "class_names|label_map|labels.txt|action|102|ANUBIS",
  "path": "/home/alan/mydev/college/football-injury-ai/module1"
}
```

**Output:**
```
Found 33 matches
/home/alan/mydev/college/football-injury-ai/module1/pretrain/train.py:
  Line 3: # Stage 1 — ANUBIS Pretraining Loop
  Line 5: # Trains ActionClassifier end-to-end on ANUBIS 102-class action recognition.
  Line 208:     log.info("Loading ANUBIS dataset …")
  Line 320:         description="Pretrain SpatioTemporalEncoder on ANUBIS"

/home/alan/mydev/college/football-injury-ai/module1/pretrain/tune_optuna.py:
  Line 3: # Stage 1 — Optuna Hyperparameter Search for ANUBIS Pretraining
  Line 426:         description="Run Optuna hyperparameter search for ANUBIS pretraining"

/home/alan/mydev/college/football-injury-ai/module1/pretrain/evaluate.py:
  Line 8: #   • t-SNE visualisation of CLS embeddings coloured by action class
  Line 36:     ANUBISDataset,
  Line 193:     ax.set_title("t-SNE of SpatioTemporalEncoder CLS embeddings (ANUBIS)")
  Line 293:         description="Evaluate a pretrained SpatioTemporalEncoder on ANUBIS"
  Line 302:         action="store_true",

/home/alan/mydev/college/football-injury-ai/module1/data/anubis_loader.py:
  Line 3: # Stage 1 — ANUBIS Dataset Loader
  Line 5: # Loads preprocessed ANUBIS skeleton data from .npy files and exposes it as a
  Line 17: # to (N, T, J, F) format, and wraps in an ANUBISDataset that applies
  Line 49: class ANUBISDataset(Dataset):
  Line 51:     PyTorch Dataset wrapping preprocessed ANUBIS skeleton clips.
  Line 134:             "Copy your preprocessed ANUBIS files to the anubis_dir specified in "
  Line 254:     train_ds = ANUBISDataset(
  Line 261:     val_ds = ANUBISDataset(
  Line 302:     lbl = np.random.randint(0, 102, size=(N,)).astype(np.int64)
  Line 314:         ds = ANUBISDataset(feats, labels, seq_len=50, augment=True)

/home/alan/mydev/college/football-injury-ai/module1/data/preprocess.py:
  Line 15: # They are called inside ANUBISDataset.__getitem__ during training.
  Line 48:         the ANUBIS preprocessing notebook's boundary handling style.

/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py:
  Line 25: #   102-class ANUBIS pretraining.  After pretraining, save only the encoder
  Line 96:     num_joints   : int   number of skeleton joints (32 for ANUBIS)
  Line 279:     Used only for Stage 1 (ANUBIS action classification pretraining).
  Line 285:     num_classes : int   number of action classes (102 for ANUBIS)
  Line 364:     classifier = ActionClassifier(encoder, num_classes=102).to(device)
  Line 374:     assert logits.shape == (B, 102), f"Expected (4,102), got {logits.shape}"

/home/alan/mydev/college/football-injury-ai/module1/data/joint_config.py:
  Line 3: # Stage 2 — ANUBIS 32-Joint Skeleton Configuration
  Line 5: # Defines the 32-joint skeleton used by the preprocessed ANUBIS dataset:
  Line 15: # If your ANUBIS preprocessing used a different joint ordering, update
  Line 132:     Build a bidirectional edge index tensor for the ANUBIS skeleton.
```

**Tool: glob**

**Input:**
```json
{
  "pattern": "data/anubis/**/*"
}
```

**Output:**
```
No files found
```

---

## Assistant (Plan · gpt-5.3-codex · 11.7s)

_Thinking:_

**Planning live footage testing pipeline**

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/module1/pretrain/tune_optuna.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/module1/pretrain/tune_optuna.py</path>
<type>file</type>
<content>1: # module1/pretrain/tune_optuna.py
2: #
3: # Stage 1 — Optuna Hyperparameter Search for ANUBIS Pretraining
4: # ───────────────────────────────────────────────────────────────
5: # Runs Optuna-based hyperparameter optimisation over the Stage 1
6: # ActionClassifier training loop.
7: #
8: # Usage:
9: #   python module1/pretrain/tune_optuna.py --config configs/pretrain.yaml
10: #
11: # Notes:
12: #   - This script does NOT save per-trial model checkpoints.
13: #   - It optimises validation top-1 accuracy and stores the best parameters
14: #     and a full best-config YAML for a follow-up full training run.
15: 
16: from __future__ import annotations
17: 
18: import argparse
19: import copy
20: import os
21: import random
22: from typing import Any, Callable, Dict, Optional
23: 
24: import numpy as np
25: import torch
26: import torch.nn as nn
27: import yaml
28: 
29: from module1.data.anubis_loader import build_dataloaders
30: from module1.data.joint_config import EDGE_INDEX
31: from module1.pretrain.model import build_model
32: from module1.pretrain.train import _make_lr_lambda, evaluate, train_one_epoch
33: from utils.config import load_config
34: from utils.logger import get_logger
35: 
36: log = get_logger(__name__)
37: 
38: 
39: # ── Reproducibility helpers ───────────────────────────────────────────────────
40: 
41: 
42: def set_seed(seed: int) -> None:
43:     """
44:     Set random seeds for reproducibility.
45: 
46:     Parameters
47:     ----------
48:     seed : int
49:         Global random seed.
50:     """
51:     random.seed(seed)
52:     np.random.seed(seed)
53:     torch.manual_seed(seed)
54:     if torch.cuda.is_available():
55:         torch.cuda.manual_seed_all(seed)
56: 
57: 
58: # ── Search-space helpers ──────────────────────────────────────────────────────
59: 
60: 
61: def _get_space_spec(search_space: Dict, key: str, defaults: Dict) -> Dict:
62:     """
63:     Merge one search-space entry with defaults.
64: 
65:     Parameters
66:     ----------
67:     search_space : dict
68:     key          : str
69:     defaults     : dict
70: 
71:     Returns
72:     -------
73:     spec : dict
74:     """
75:     spec = copy.deepcopy(defaults)
76:     spec.update(search_space.get(key, {}))
77:     return spec
78: 
79: 
80: def sample_hparams(trial: Any, search_space: Dict) -> Dict:
81:     """
82:     Sample one hyperparameter set from Optuna.
83: 
84:     Parameters
85:     ----------
86:     trial        : optuna.trial.Trial
87:     search_space : dict
88: 
89:     Returns
90:     -------
91:     params : dict
92:     """
93:     lr_spec = _get_space_spec(
94:         search_space,
95:         "learning_rate",
96:         {"low": 1e-4, "high": 2e-3, "log": True},
97:     )
98:     wd_spec = _get_space_spec(
99:         search_space,
100:         "weight_decay",
101:         {"low": 1e-5, "high": 1e-3, "log": True},
102:     )
103:     ls_spec = _get_space_spec(
104:         search_space,
105:         "label_smoothing",
106:         {"low": 0.0, "high": 0.15},
107:     )
108:     gat_do_spec = _get_space_spec(
109:         search_space,
110:         "gat_dropout",
111:         {"low": 0.05, "high": 0.35},
112:     )
113:     tr_do_spec = _get_space_spec(
114:         search_space,
115:         "transformer_dropout",
116:         {"low": 0.05, "high": 0.35},
117:     )
118:     flip_spec = _get_space_spec(
119:         search_space,
120:         "flip_prob",
121:         {"low": 0.1, "high": 0.6},
122:     )
123:     jitter_spec = _get_space_spec(
124:         search_space,
125:         "joint_jitter_std",
126:         {"low": 1e-3, "high": 2e-2, "log": True},
127:     )
128:     speed_spec = _get_space_spec(
129:         search_space,
130:         "speed_perturb",
131:         {"choices": [True, False]},
132:     )
133: 
134:     params = {
135:         "learning_rate": trial.suggest_float(
136:             "learning_rate",
137:             float(lr_spec["low"]),
138:             float(lr_spec["high"]),
139:             log=bool(lr_spec.get("log", False)),
140:         ),
141:         "weight_decay": trial.suggest_float(
142:             "weight_decay",
143:             float(wd_spec["low"]),
144:             float(wd_spec["high"]),
145:             log=bool(wd_spec.get("log", False)),
146:         ),
147:         "label_smoothing": trial.suggest_float(
148:             "label_smoothing",
149:             float(ls_spec["low"]),
150:             float(ls_spec["high"]),
151:         ),
152:         "gat_dropout": trial.suggest_float(
153:             "gat_dropout",
154:             float(gat_do_spec["low"]),
155:             float(gat_do_spec["high"]),
156:         ),
157:         "transformer_dropout": trial.suggest_float(
158:             "transformer_dropout",
159:             float(tr_do_spec["low"]),
160:             float(tr_do_spec["high"]),
161:         ),
162:         "flip_prob": trial.suggest_float(
163:             "flip_prob",
164:             float(flip_spec["low"]),
165:             float(flip_spec["high"]),
166:         ),
167:         "joint_jitter_std": trial.suggest_float(
168:             "joint_jitter_std",
169:             float(jitter_spec["low"]),
170:             float(jitter_spec["high"]),
171:             log=bool(jitter_spec.get("log", False)),
172:         ),
173:         "speed_perturb": trial.suggest_categorical(
174:             "speed_perturb",
175:             list(speed_spec.get("choices", [True, False])),
176:         ),
177:     }
178:     return params
179: 
180: 
181: def apply_hparams(cfg: Dict, params: Dict) -> None:
182:     """
183:     Apply sampled hyperparameters to a config dict in-place.
184: 
185:     Parameters
186:     ----------
187:     cfg    : dict
188:     params : dict
189:     """
190:     cfg["training"]["learning_rate"] = float(params["learning_rate"])
191:     cfg["training"]["weight_decay"] = float(params["weight_decay"])
192:     cfg["training"]["label_smoothing"] = float(params["label_smoothing"])
193: 
194:     cfg["model"]["gat_dropout"] = float(params["gat_dropout"])
195:     cfg["model"]["transformer_dropout"] = float(params["transformer_dropout"])
196: 
197:     cfg["augmentation"]["flip_prob"] = float(params["flip_prob"])
198:     cfg["augmentation"]["joint_jitter_std"] = float(params["joint_jitter_std"])
199:     cfg["augmentation"]["speed_perturb"] = bool(params["speed_perturb"])
200: 
201: 
202: def save_yaml(path: str, payload: Dict) -> None:
203:     """Save a dictionary to YAML."""
204:     dir_path = os.path.dirname(path)
205:     if dir_path:
206:         os.makedirs(dir_path, exist_ok=True)
207:     with open(path, "w", encoding="utf-8") as f:
208:         yaml.safe_dump(payload, f, sort_keys=False)
209: 
210: 
211: # ── Optuna objective ──────────────────────────────────────────────────────────
212: 
213: 
214: def make_objective(base_cfg: Dict, opt_cfg: Dict, device: torch.device) -> Callable:
215:     """
216:     Build an Optuna objective closure.
217: 
218:     Parameters
219:     ----------
220:     base_cfg : dict
221:     opt_cfg  : dict
222:     device   : torch.device
223: 
224:     Returns
225:     -------
226:     objective : callable
227:     """
228:     trial_epochs = int(opt_cfg.get("trial_epochs", 25))
229:     seed = int(opt_cfg.get("seed", 42))
230:     num_workers = int(opt_cfg.get("num_workers", 4))
231:     pin_memory = bool(opt_cfg.get("pin_memory", True))
232:     search_space = opt_cfg.get("search_space", {})
233: 
234:     def objective(trial: Any) -> float:
235:         import optuna
236: 
237:         cfg = copy.deepcopy(base_cfg)
238:         cfg["training"]["epochs"] = trial_epochs
239: 
240:         sampled = sample_hparams(trial, search_space)
241:         apply_hparams(cfg, sampled)
242: 
243:         # Keep trial data split deterministic for fair trial-to-trial comparison.
244:         set_seed(seed)
245:         train_loader, val_loader = build_dataloaders(
246:             cfg,
247:             num_workers=num_workers,
248:             pin_memory=pin_memory,
249:         )
250: 
251:         edge_index = EDGE_INDEX.to(device)
252:         _, classifier = build_model(cfg, edge_index)
253:         classifier = classifier.to(device)
254: 
255:         optimiser = torch.optim.AdamW(
256:             classifier.parameters(),
257:             lr=cfg["training"]["learning_rate"],
258:             weight_decay=cfg["training"]["weight_decay"],
259:         )
260: 
261:         scheduler = torch.optim.lr_scheduler.LambdaLR(
262:             optimiser,
263:             lr_lambda=_make_lr_lambda(
264:                 int(cfg["training"].get("warmup_epochs", 5)),
265:                 int(cfg["training"]["epochs"]),
266:             ),
267:         )
268: 
269:         criterion = nn.CrossEntropyLoss(
270:             label_smoothing=float(cfg["training"].get("label_smoothing", 0.1))
271:         )
272:         eval_criterion = nn.CrossEntropyLoss()
273:         grad_clip = float(cfg["training"].get("grad_clip_norm", 1.0))
274: 
275:         best_val_top1 = 0.0
276:         best_val_top5 = 0.0
277:         best_val_loss = float("inf")
278:         trn_loss = 0.0
279:         trn_acc = 0.0
280: 
281:         for epoch in range(1, trial_epochs + 1):
282:             trn_loss, trn_acc = train_one_epoch(
283:                 classifier,
284:                 train_loader,
285:                 optimiser,
286:                 criterion,
287:                 device,
288:                 grad_clip,
289:             )
290: 
291:             val_loss, val_top1, val_top5 = evaluate(
292:                 classifier,
293:                 val_loader,
294:                 eval_criterion,
295:                 device,
296:             )
297: 
298:             scheduler.step()
299: 
300:             if val_top1 > best_val_top1:
301:                 best_val_top1 = val_top1
302:                 best_val_top5 = val_top5
303:                 best_val_loss = val_loss
304: 
305:             trial.report(best_val_top1, step=epoch)
306:             if trial.should_prune():
307:                 raise optuna.TrialPruned()
308: 
309:         trial.set_user_attr("best_val_top1", float(best_val_top1))
310:         trial.set_user_attr("best_val_top5", float(best_val_top5))
311:         trial.set_user_attr("best_val_loss", float(best_val_loss))
312:         trial.set_user_attr("final_trn_loss", float(trn_loss))
313:         trial.set_user_attr("final_trn_acc", float(trn_acc))
314: 
315:         log.info(
316:             "Trial %d done | best val top1=%.4f top5=%.4f loss=%.4f",
317:             trial.number,
318:             best_val_top1,
319:             best_val_top5,
320:             best_val_loss,
321:         )
322:         return float(best_val_top1)
323: 
324:     return objective
325: 
326: 
327: # ── Main entry point ──────────────────────────────────────────────────────────
328: 
329: 
330: def main(
331:     config_path: str,
332:     n_trials_override: Optional[int],
333:     timeout_override: Optional[int],
334: ) -> None:
335:     cfg = load_config(config_path)
336:     opt_cfg = cfg.get("optuna", {})
337: 
338:     try:
339:         import optuna
340:     except ImportError as exc:
341:         raise ImportError(
342:             "Optuna is not installed. Install it with: pip install optuna"
343:         ) from exc
344: 
345:     seed = int(opt_cfg.get("seed", 42))
346:     set_seed(seed)
347: 
348:     n_trials = (
349:         int(n_trials_override)
350:         if n_trials_override is not None
351:         else int(opt_cfg.get("n_trials", 30))
352:     )
353:     timeout_sec = timeout_override
354:     if timeout_sec is None:
355:         timeout_value = opt_cfg.get("timeout_sec", None)
356:         timeout_sec = int(timeout_value) if timeout_value is not None else None
357: 
358:     study_name = str(opt_cfg.get("study_name", "anubis_pretrain_hpo"))
359:     storage = opt_cfg.get("storage", None)
360:     load_if_exists = bool(opt_cfg.get("load_if_exists", True))
361:     direction = str(opt_cfg.get("direction", "maximize"))
362: 
363:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
364:     log.info("Device: %s", device)
365:     log.info(
366:         "Starting Optuna study '%s' | trials=%d | timeout=%s",
367:         study_name,
368:         n_trials,
369:         str(timeout_sec),
370:     )
371: 
372:     sampler = optuna.samplers.TPESampler(seed=seed)
373:     pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
374: 
375:     study = optuna.create_study(
376:         study_name=study_name,
377:         storage=storage,
378:         load_if_exists=load_if_exists,
379:         direction=direction,
380:         sampler=sampler,
381:         pruner=pruner,
382:     )
383: 
384:     objective = make_objective(cfg, opt_cfg, device)
385:     study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)
386: 
387:     try:
388:         best_trial = study.best_trial
389:     except ValueError as exc:
390:         raise RuntimeError(
391:             "No completed trial found. Check your data/configuration."
392:         ) from exc
393:     best_params = dict(best_trial.params)
394:     log.info("Best trial: %d", best_trial.number)
395:     log.info("Best value (val_top1): %.4f", best_trial.value)
396:     log.info("Best params: %s", best_params)
397: 
398:     best_cfg = copy.deepcopy(cfg)
399:     apply_hparams(best_cfg, best_params)
400: 
401:     best_params_path = str(
402:         opt_cfg.get("output_best_params", "outputs/logs/optuna_best_params.yaml")
403:     )
404:     best_cfg_path = str(
405:         opt_cfg.get("output_best_config", "outputs/logs/pretrain_optuna_best.yaml")
406:     )
407: 
408:     save_yaml(
409:         best_params_path,
410:         {
411:             "study_name": study_name,
412:             "best_trial_number": int(best_trial.number),
413:             "best_value": float(best_trial.value),
414:             "best_params": best_params,
415:             "best_user_attrs": dict(best_trial.user_attrs),
416:         },
417:     )
418:     save_yaml(best_cfg_path, best_cfg)
419: 
420:     log.info("Saved best params → %s", best_params_path)
421:     log.info("Saved best config → %s", best_cfg_path)
422: 
423: 
424: if __name__ == "__main__":
425:     parser = argparse.ArgumentParser(
426:         description="Run Optuna hyperparameter search for ANUBIS pretraining"
427:     )
428:     parser.add_argument(
429:         "--config",
430:         default="configs/pretrain.yaml",
431:         help="Path to pretrain config YAML",
432:     )
433:     parser.add_argument(
434:         "--n-trials",
435:         type=int,
436:         default=None,
437:         help="Override optuna.n_trials from config",
438:     )
439:     parser.add_argument(
440:         "--timeout-sec",
441:         type=int,
442:         default=None,
443:         help="Override optuna.timeout_sec from config",
444:     )
445:     args = parser.parse_args()
446:     main(args.config, args.n_trials, args.timeout_sec)

(End of file - total 446 lines)
</content>
```

---

## Assistant (Plan · gpt-5.3-codex · 15.8s)

_Thinking:_

**Planning testing pipeline and clarifying key questions**

**Tool: grep**

**Input:**
```json
{
  "pattern": "mapping|map|33|32|mediapipe|landmark|joint",
  "path": "/home/alan/mydev/college/football-injury-ai/notebooks",
  "include": "*.ipynb"
}
```

**Output:**
```
Found 27 matches
/home/alan/mydev/college/football-injury-ai/notebooks/02_graph_exploration.ipynb:
  Line 15:     "4. Inspect node features (position, velocity, joint angles)\n",
  Line 50:     "print(f'Loaded keypoints shape: {kp.shape}')  # (n_frames, 33, 4)\n",
  Line 53:     "seq = kp[:50]   # (50, 33, 4)"
  Line 64:     "frame_kp = seq[-1]   # (33, 4)\n",
  Line 73:     "pos = {i: (frame_kp[i, 0], -frame_kp[i, 1]) for i in range(33)}\n",
  Line 75:     "# Colour key joints differently\n",
  Line 76:     "key_joints   = [23, 24, 25, 26, 27, 28, 11, 12]  # hips, knees, ankles, shoulders\n",
  Line 77:     "node_colours = ['tomato' if i in key_joints else 'steelblue' for i in range(33)]\n",
  Line 83:     "    labels={i: KEYPOINT_NAMES[i][:6] for i in range(33)},\n",
  Line 86:     "plt.title('Skeleton graph — red = biomechanically significant joints')\n",
  Line 101:     "features = build_node_features(seq)   # (33, 10)\n",
  Line 102:     "print(f'Node features shape: {features.shape}')   # (33, 10)\n",
  Line 103:     "print('Feature columns: [x, y, z, vx, vy, vz, ax, ay, az, joint_angle, visibility]')\n",
  Line 106:     "# Print features for key joints\n",
  Line 128:     "print(f'Graph nodes     : {graph.x.shape}')          # (33, 10)\n",

/home/alan/mydev/college/football-injury-ai/notebooks/01_pose_exploration.ipynb:
  Line 32:     "import mediapipe as mp\n",
  Line 78:     "if results.pose_landmarks:\n",
  Line 81:     "    mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)\n",
  Line 84:     "    plt.title('MediaPipe Pose — 33 keypoints')\n",
  Line 98:     "if results.pose_landmarks:\n",
  Line 101:     "        for lm in results.pose_landmarks.landmark\n",
  Line 103:     "    print(f'Keypoints shape: {keypoints.shape}')  # (33, 4)\n",
  Line 104:     "    print(f'\\nFirst 5 joints (x, y, z, visibility):')\n",
  Line 108:     "    print(f'\\nKey joint names:')\n",
  Line 126:     "print(f'Extracted keypoints shape: {all_keypoints.shape}')   # (n_frames, 33, 4)\n",
  Line 136:     "# ── Plot how one joint moves across the video ─────────────────────────────────\n",
  Line 137:     "# Joint 27 = left ankle, joint 28 = right ankle\n",
```

---

## Assistant (Plan · gpt-5.3-codex · 14.4s)

_Thinking:_

**Planning live inference pipeline**

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/notebooks/01_pose_exploration.ipynb",
  "offset": 90,
  "limit": 120
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/notebooks/01_pose_exploration.ipynb</path>
<type>file</type>
<content>90:   },
91:   {
92:    "cell_type": "code",
93:    "execution_count": null,
94:    "metadata": {},
95:    "outputs": [],
96:    "source": [
97:     "# ── Inspect keypoint values ───────────────────────────────────────────────────\n",
98:     "if results.pose_landmarks:\n",
99:     "    keypoints = np.array([\n",
100:     "        [lm.x, lm.y, lm.z, lm.visibility]\n",
101:     "        for lm in results.pose_landmarks.landmark\n",
102:     "    ])\n",
103:     "    print(f'Keypoints shape: {keypoints.shape}')  # (33, 4)\n",
104:     "    print(f'\\nFirst 5 joints (x, y, z, visibility):')\n",
105:     "    print(keypoints[:5])\n",
106:     "\n",
107:     "    from module1.pose.extractor import KEYPOINT_NAMES\n",
108:     "    print(f'\\nKey joint names:')\n",
109:     "    for i in [11, 12, 23, 24, 25, 26, 27, 28]:   # shoulders, hips, knees, ankles\n",
110:     "        x, y, z, vis = keypoints[i]\n",
111:     "        print(f'  [{i:2d}] {KEYPOINT_NAMES[i]:<20} x={x:.3f}  y={y:.3f}  vis={vis:.3f}')"
112:    ]
113:   },
114:   {
115:    "cell_type": "code",
116:    "execution_count": null,
117:    "metadata": {},
118:    "outputs": [],
119:    "source": [
120:     "# ── Use the project extractor to process the full video ──────────────────────\n",
121:     "from module1.pose.extractor import PoseExtractor, save_keypoints\n",
122:     "\n",
123:     "with PoseExtractor() as extractor:\n",
124:     "    all_keypoints = extractor.extract_from_video(VIDEO_PATH)\n",
125:     "\n",
126:     "print(f'Extracted keypoints shape: {all_keypoints.shape}')   # (n_frames, 33, 4)\n",
127:     "save_keypoints(all_keypoints, '../data/processed/', 'sample_clip')"
128:    ]
129:   },
130:   {
131:    "cell_type": "code",
132:    "execution_count": null,
133:    "metadata": {},
134:    "outputs": [],
135:    "source": [
136:     "# ── Plot how one joint moves across the video ─────────────────────────────────\n",
137:     "# Joint 27 = left ankle, joint 28 = right ankle\n",
138:     "left_ankle_y  = all_keypoints[:, 27, 1]   # y-coordinate over time\n",
139:     "right_ankle_y = all_keypoints[:, 28, 1]\n",
140:     "\n",
141:     "plt.figure(figsize=(14, 4))\n",
142:     "plt.plot(left_ankle_y,  label='Left ankle (y)', color='steelblue')\n",
143:     "plt.plot(right_ankle_y, label='Right ankle (y)', color='tomato')\n",
144:     "plt.xlabel('Frame')\n",
145:     "plt.ylabel('Normalised y-position')\n",
146:     "plt.title('Ankle y-position over time — early indicator of stride asymmetry')\n",
147:     "plt.legend()\n",
148:     "plt.grid(True, alpha=0.3)\n",
149:     "plt.show()"
150:    ]
151:   }
152:  ],
153:  "metadata": {
154:   "kernelspec": {
155:    "display_name": "Python 3",
156:    "language": "python",
157:    "name": "python3"
158:   },
159:   "language_info": {
160:    "name": "python",
161:    "version": "3.10.0"
162:   }
163:  },
164:  "nbformat": 4,
165:  "nbformat_minor": 4
166: }

(End of file - total 166 lines)
</content>
```

---

## Assistant (Plan · gpt-5.3-codex · 89.4s)

_Thinking:_

**Planning live inference pipeline**

_Thinking:_

**Confirming detailed implementation plan**

Yes — great timing. I mapped out the live testing pipeline and can implement it next.

Important context first: your current checkpoint (`outputs/checkpoints/pretrain/pretrain_best.pt`) is a **102-class motion/action model**, so live output will be **action understanding** (top-1/top-5 + confidence), not injury-risk score yet.

**Plan**
- Build a real-time pose extractor in `module1/pose/extractor.py` (MediaPipe + OpenCV stream).
- Add 33→32 joint conversion to match your training skeleton (`module1/data/joint_config.py:79`), then apply pelvis-centering exactly like training.
- Add a sliding window buffer (`seq_len=60`, from `configs/module1.yaml:24`) and compute `[x,y,z,vx,vy,vz,ax,ay,az]` per joint to match `module1/pretrain/model.py`.
- Create a live inference runner `module1/pretrain/live_inference.py`:
  - loads config + checkpoint
  - reads webcam/video/RTSP source
  - runs inference every frame/window
  - overlays top-1/top-5 predictions and confidence on video
  - optional smoothing (EMA / majority vote) to reduce jitter
- Add optional label-map support (`--label-map`) so if class names exist, UI shows names; otherwise falls back to `class_<id>`.
- Add tests for preprocessing compatibility (`tests/test_live_preprocess.py`) and update `docs/COMMANDS.md` with run commands.

**What you’ll be able to run**
- `uv run module1/pretrain/live_inference.py --config configs/pretrain_final.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --source 0`
- `--source` supports webcam index, video file path, or RTSP URL.

One decision before I implement:  
Use `class_<id>` labels by default, or do you already have an ANUBIS class-name mapping file to display real names?

---

## User

implement this plan also update the command docs 

---

## Assistant (Build · gpt-5.3-codex · 15.3s)

_Thinking:_

**Resolving instruction conflicts and planning implementation**

**Tool: glob**

**Input:**
```json
{
  "pattern": "**/*"
}
```

**Output:**
```
/home/alan/mydev/college/football-injury-ai/.git/refs/remotes/origin/master
/home/alan/mydev/college/football-injury-ai/.git/refs/heads/master
/home/alan/mydev/college/football-injury-ai/.git/COMMIT_EDITMSG
/home/alan/mydev/college/football-injury-ai/.git/objects/53/49c7189ea03e4958419eeca2ba6d864150af1f
/home/alan/mydev/college/football-injury-ai/.git/objects/54/231a1546ebfd6f1cf9cc32ef937bdcede9452a
/home/alan/mydev/college/football-injury-ai/.git/objects/e7/10ebec37ee2d35780b292b8f9ec4e5098440ec
/home/alan/mydev/college/football-injury-ai/configs/pretrain_final.yaml
/home/alan/mydev/college/football-injury-ai/outputs/logs/pretrain_optuna_best.yaml
/home/alan/mydev/college/football-injury-ai/outputs/logs/optuna_best_params.yaml
/home/alan/mydev/college/football-injury-ai/outputs/logs/run.log
/home/alan/mydev/college/football-injury-ai/outputs/logs/pretrain_tsne.png
/home/alan/mydev/college/football-injury-ai/.git/objects/d3/2659b573f7cb29e07b02a0b60f032d900c7c16
/home/alan/mydev/college/football-injury-ai/.git/objects/f2/d62e8ad1d7d5e6f05cc0e738595f49f7c79fa2
/home/alan/mydev/college/football-injury-ai/.git/objects/44/44ef4677cf3bf468fb3e2f42eb0e1b10a8f2f7
/home/alan/mydev/college/football-injury-ai/.git/objects/f7/4ea3952f650b91224e5ca9d0b216690346dd34
/home/alan/mydev/college/football-injury-ai/.git/objects/60/bbd128c80bd52eabb8350475e37c13a57c27aa
/home/alan/mydev/college/football-injury-ai/session-ses_33bc.md
/home/alan/mydev/college/football-injury-ai/.git/objects/e6/ee64159386723cc01a90fa21d23866607a69cd
/home/alan/mydev/college/football-injury-ai/.git/objects/90/2dc393e6c4a78a83791e525e6a1f3aed553510
/home/alan/mydev/college/football-injury-ai/.git/objects/cd/5f4352c506590d5fcda19a604590e26a3024f4
/home/alan/mydev/college/football-injury-ai/scripts/__pycache__/anubis_feature_stats.cpython-314.pyc
/home/alan/mydev/college/football-injury-ai/scripts/__pycache__/anubis_label_stats.cpython-314.pyc
/home/alan/mydev/college/football-injury-ai/.ruff_cache/0.15.5/6267884628809027351
/home/alan/mydev/college/football-injury-ai/pyproject.toml
/home/alan/mydev/college/football-injury-ai/configs/pretrain.yaml
/home/alan/mydev/college/football-injury-ai/.git/objects/24/c1d0fa4d934ac3cdfbed7ff4c94bfe3170bec6
/home/alan/mydev/college/football-injury-ai/.git/objects/94/28a8a98cb73f136cd8bf29a6a3cec2b48de1f7
/home/alan/mydev/college/football-injury-ai/.git/objects/71/ef5359163307ad16e1e6de44352c30218f731a
/home/alan/mydev/college/football-injury-ai/.git/objects/0f/d74f987e742266b813b7466bb3b0199c253746
/home/alan/mydev/college/football-injury-ai/.git/objects/8c/e3357f1e3c0342400d897933191e7369cf247e
/home/alan/mydev/college/football-injury-ai/.git/objects/93/c1c56c6a4fd9587d930d430298e94465cfa88d
/home/alan/mydev/college/football-injury-ai/.git/objects/a0/bff2f9778033ecf3f091caf82ed3c9896aae73
/home/alan/mydev/college/football-injury-ai/.ruff_cache/0.15.5/11669862769449356671
/home/alan/mydev/college/football-injury-ai/scripts/anubis_feature_stats.py
/home/alan/mydev/college/football-injury-ai/.git/objects/c1/2e9697d9c903d22ad7512efbf42f62c1a4be56
/home/alan/mydev/college/football-injury-ai/.git/objects/f7/3024bc073c0ed89e2e1c5a745aa387cda3c9d0
/home/alan/mydev/college/football-injury-ai/.git/objects/09/8758d4501f3ad3ea635f03850b3866f1da709f
/home/alan/mydev/college/football-injury-ai/scripts/anubis_label_stats.py
/home/alan/mydev/college/football-injury-ai/.git/ORIG_HEAD
/home/alan/mydev/college/football-injury-ai/.git/FETCH_HEAD
/home/alan/mydev/college/football-injury-ai/.git/objects/46/49de2a7c3e9cf6ba8cfe600ab3b80f750de4e1
/home/alan/mydev/college/football-injury-ai/.git/objects/02/77f44f91abc53d8f4bdfa85a6b7c4bfacb3d39
/home/alan/mydev/college/football-injury-ai/.git/objects/d3/8805ecd7ae7e6dfa0e37139125243371814b1a
/home/alan/mydev/college/football-injury-ai/.git/objects/42/8ef8f458a35e7d8716d77015d7125408f74aae
/home/alan/mydev/college/football-injury-ai/.python-version
/home/alan/mydev/college/football-injury-ai/.git/config
/home/alan/mydev/college/football-injury-ai/.git/objects/36/4b5bccc859748b345c0793da803f92a64cfd49
/home/alan/mydev/college/football-injury-ai/.git/objects/7f/a85438c7fc2aa533421477aac43f4e621562b2
/home/alan/mydev/college/football-injury-ai/.git/objects/8f/dc1e6af670f39ecfe928c320de05089d7a2285
/home/alan/mydev/college/football-injury-ai/.git/objects/54/f57d448712267543c2fe526a38e6606398b567
/home/alan/mydev/college/football-injury-ai/.git/objects/37/f483c72f3ca911158ea3d4abcd856032fb293a
/home/alan/mydev/college/football-injury-ai/.git/objects/91/60a50f9a8f1a4886734ebe4b36e30103faf54f
/home/alan/mydev/college/football-injury-ai/.git/objects/a0/74c7818918ef86ba3c001f594f856dc7ab0437
/home/alan/mydev/college/football-injury-ai/.git/objects/fe/700f18d7b6bc54610e8671cbc28ea3c3e379fd
/home/alan/mydev/college/football-injury-ai/.git/objects/e6/9de29bb2d1d6434b8b29ae775ad8c2e48c5391
/home/alan/mydev/college/football-injury-ai/.git/objects/22/510421dc6e309cdf692d1d7501beb0b5af21be
/home/alan/mydev/college/football-injury-ai/.git/objects/54/6fb9f61965d1d558de00df753d81897f4d923a
/home/alan/mydev/college/football-injury-ai/.git/objects/04/5f9fec47dbd6314906e56b17ba39409ea6520a
/home/alan/mydev/college/football-injury-ai/.git/objects/c5/203634e5278bfba03489c50159eff8303fafe1
/home/alan/mydev/college/football-injury-ai/.ruff_cache/0.15.5/18392522430173918825
/home/alan/mydev/college/football-injury-ai/.ruff_cache/0.15.5/15213122632981715198
/home/alan/mydev/college/football-injury-ai/utils/__pycache__/logger.cpython-312.pyc
/home/alan/mydev/college/football-injury-ai/utils/__pycache__/__init__.cpython-312.pyc
/home/alan/mydev/college/football-injury-ai/.ruff_cache/0.15.5/16040572351761327884
/home/alan/mydev/college/football-injury-ai/configs/module1.yaml
/home/alan/mydev/college/football-injury-ai/.ruff_cache/CACHEDIR.TAG
/home/alan/mydev/college/football-injury-ai/.ruff_cache/.gitignore
/home/alan/mydev/college/football-injury-ai/.git/objects/03/7dd2b3364cca53b5beac68f4b3242b6b4a5570
/home/alan/mydev/college/football-injury-ai/.git/objects/eb/861024e2735684da8e98ade36e8f8e19746e67
/home/alan/mydev/college/football-injury-ai/.git/objects/7f/124ccfdad28357ba5bd97d51b431a9372a9df2
/home/alan/mydev/college/football-injury-ai/.git/objects/02/71a06074b91882002712f35f7f435163a351a9
/home/alan/mydev/college/football-injury-ai/.git/objects/a2/86c023ff97ed79d460247e8c6488083e3c1350
/home/alan/mydev/college/football-injury-ai/.git/objects/22/04255a22a15519cef512d777e208d7374bb6d3
/home/alan/mydev/college/football-injury-ai/.git/objects/8c/add4abcf54bc33a106fe769831d44fb3473275
/home/alan/mydev/college/football-injury-ai/.git/objects/ff/6d8c8b4242720f3ac7c6d6c9a573cc75dedc31
/home/alan/mydev/college/football-injury-ai/.git/objects/3f/5c70358a7820ed2bbff2b2753c7644cfc2fe57
/home/alan/mydev/college/football-injury-ai/.git/objects/05/b8a226ed9afc11b4a1e8a3c888075e1445afd9
/home/alan/mydev/college/football-injury-ai/.git/objects/4a/c9f0707ff5555fa7fa097e63fd7be2cde7c31d
/home/alan/mydev/college/football-injury-ai/.git/objects/fc/017d946d24ac53004cea8181e6424f79b6b8e1
/home/alan/mydev/college/football-injury-ai/.git/objects/10/ca19a6f367c2976193ff641a8252f4e67acb94
/home/alan/mydev/college/football-injury-ai/.git/objects/d8/58374922d449daae7f16da83bbebd0954d4268
/home/alan/mydev/college/football-injury-ai/.git/objects/6d/5983ef2f8d31690c6701f5e25d811f7a753a56
/home/alan/mydev/college/football-injury-ai/.git/objects/ea/57f711690040dc8cdb31ad80c1a8269a8c2b50
/home/alan/mydev/college/football-injury-ai/.git/objects/7b/5216b3c3c6605a8a62a673cbf2a53d4848308f
/home/alan/mydev/college/football-injury-ai/.git/objects/88/6b3c75ac5d1b97165f4cc7e3084f6d47f6d1cd
/home/alan/mydev/college/football-injury-ai/.git/objects/04/2facd430b317aea0c11cc99da8c4d104921e59
/home/alan/mydev/college/football-injury-ai/.git/objects/3d/65ffa2d0d03681bfcf2f257ae04207dac4fffc
/home/alan/mydev/college/football-injury-ai/.git/description
/home/alan/mydev/college/football-injury-ai/skills-lock.json
/home/alan/mydev/college/football-injury-ai/AGENTS.md
/home/alan/mydev/college/football-injury-ai/notebooks/03_model_experiments.ipynb
/home/alan/mydev/college/football-injury-ai/notebooks/02_graph_exploration.ipynb
/home/alan/mydev/college/football-injury-ai/notebooks/01_pose_exploration.ipynb
/home/alan/mydev/college/football-injury-ai/utils/logger.py
/home/alan/mydev/college/football-injury-ai/utils/config.py
/home/alan/mydev/college/football-injury-ai/module3/.gitkeep
/home/alan/mydev/college/football-injury-ai/module2/.gitkeep
/home/alan/mydev/college/football-injury-ai/outputs/logs/.gitkeep
/home/alan/mydev/college/football-injury-ai/outputs/checkpoints/.gitkeep
/home/alan/mydev/college/football-injury-ai/utils/__init__.py

(Results are truncated: showing first 100 results. Consider using a more specific path or pattern.)
```

---

## Assistant (Build · gpt-5.3-codex · 7.8s)

_Thinking:_

**Planning pose extraction pipeline**

**Tool: grep**

**Input:**
```json
{
  "pattern": "KEYPOINT_NAMES|PoseExtractor|extract_from_video|module1.pose.extractor|MIRROR_JOINT_PAIRS",
  "path": "/home/alan/mydev/college/football-injury-ai/session-ses_33bc.md"
}
```

**Output:**
```
Found 3 matches
/home/alan/mydev/college/football-injury-ai/session-ses_33bc.md:
  Line 102: <diagnostics file="/home/alan/mydev/college/football-injury-ai/module1/pose/extractor.py">
  Line 163: <diagnostics file="/home/alan/mydev/college/football-injury-ai/module1/pose/extractor.py">
  Line 3002: 70: python module1/pose/extractor.py --video data/samples/clip.mp4 --out data/processed/
```

---

## Assistant (Build · gpt-5.3-codex · 22.4s)

_Thinking:_

**Outlining live test pipeline**

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/tests/test_anubis_loader.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/tests/test_anubis_loader.py</path>
<type>file</type>
<content>1: # tests/test_anubis_loader.py
2: #
3: # Unit tests for module1/data/anubis_loader.py and module1/data/preprocess.py
4: # ─────────────────────────────────────────────────────────────────────────────
5: # Tests preprocessing functions (compute_acceleration, random/center_crop,
6: # flip_sequence, joint_jitter, speed_perturb) and the ANUBISDataset /
7: # load_anubis_tensors / build_dataloaders pipeline using in-memory dummy data.
8: #
9: # Run with: pytest tests/test_anubis_loader.py -v
10: 
11: from __future__ import annotations
12: 
13: import os
14: import tempfile
15: from typing import Tuple
16: 
17: import numpy as np
18: import pytest
19: import torch
20: 
21: from module1.data.anubis_loader import ANUBISDataset, load_anubis_tensors
22: from module1.data.joint_config import FLIP_FEATURE_INDICES, MIRROR_JOINT_PAIRS
23: from module1.data.preprocess import (
24:     center_crop,
25:     compute_acceleration,
26:     flip_sequence,
27:     joint_jitter,
28:     random_crop,
29:     speed_perturb,
30: )
31: 
32: 
33: # ── Constants ─────────────────────────────────────────────────────────────────
34: 
35: N = 20  # samples
36: T = 60  # frames
37: J = 32  # joints
38: C = 6  # raw channels (pos + vel)
39: F = 9  # full channels (pos + vel + acc)
40: 
41: 
42: # ── Fixtures ──────────────────────────────────────────────────────────────────
43: 
44: 
45: @pytest.fixture
46: def raw_batch() -> torch.Tensor:
47:     """Random raw features: (N, 6, T, J)."""
48:     torch.manual_seed(42)
49:     return torch.randn(N, C, T, J)
50: 
51: 
52: @pytest.fixture
53: def sample() -> torch.Tensor:
54:     """Single sample after acc computation, transposed: (T, J, F)."""
55:     torch.manual_seed(0)
56:     raw = torch.randn(1, C, T, J)
57:     feat9 = compute_acceleration(raw)  # (1, 9, T, J)
58:     return feat9[0].permute(1, 2, 0).clone()  # (T, J, F)
59: 
60: 
61: @pytest.fixture
62: def tensor_pair() -> Tuple[torch.Tensor, torch.Tensor]:
63:     """(features (N, T, J, F), labels (N,)) as returned by load_anubis_tensors."""
64:     torch.manual_seed(7)
65:     raw = torch.randn(N, C, T, J)
66:     feat9 = compute_acceleration(raw)
67:     features = feat9.permute(0, 2, 3, 1).contiguous()  # (N, T, J, F)
68:     labels = torch.randint(0, 102, (N,), dtype=torch.long)
69:     return features, labels
70: 
71: 
72: @pytest.fixture
73: def tmp_npy_files() -> Tuple[str, str, str]:
74:     """Write dummy .npy files to a temp dir; yield (dir, feat_path, lbl_path)."""
75:     np.random.seed(1)
76:     raw = np.random.randn(N, C, T, J).astype(np.float32)
77:     lbl = np.random.randint(0, 102, size=(N,)).astype(np.int64)
78:     with tempfile.TemporaryDirectory() as tmpdir:
79:         fp = os.path.join(tmpdir, "features.npy")
80:         lp = os.path.join(tmpdir, "labels.npy")
81:         np.save(fp, raw)
82:         np.save(lp, lbl)
83:         yield tmpdir, fp, lp
84: 
85: 
86: # ── compute_acceleration ──────────────────────────────────────────────────────
87: 
88: 
89: class TestComputeAcceleration:
90:     def test_output_shape(self, raw_batch: torch.Tensor) -> None:
91:         out = compute_acceleration(raw_batch)
92:         assert out.shape == (N, 9, T, J)
93: 
94:     def test_last_acc_frame_duplicates_previous(self, raw_batch: torch.Tensor) -> None:
95:         out = compute_acceleration(raw_batch)
96:         acc = out[:, 6:9, :, :]
97:         assert torch.allclose(acc[:, :, -1, :], acc[:, :, -2, :]), (
98:             "Last-frame acceleration must duplicate previous frame"
99:         )
100: 
101:     def test_acc_is_finite_difference_of_vel(self, raw_batch: torch.Tensor) -> None:
102:         out = compute_acceleration(raw_batch)
103:         vel = out[:, 3:6, :, :]
104:         acc = out[:, 6:9, :, :]
105:         expected_acc = vel[:, :, 1:, :] - vel[:, :, :-1, :]
106:         assert torch.allclose(acc[:, :, :-1, :], expected_acc)
107: 
108:     def test_pos_vel_channels_unchanged(self, raw_batch: torch.Tensor) -> None:
109:         out = compute_acceleration(raw_batch)
110:         assert torch.allclose(out[:, :6, :, :], raw_batch)
111: 
112:     def test_no_nan(self, raw_batch: torch.Tensor) -> None:
113:         out = compute_acceleration(raw_batch)
114:         assert not torch.isnan(out).any()
115: 
116: 
117: # ── random_crop / center_crop ─────────────────────────────────────────────────
118: 
119: 
120: class TestCroppingFunctions:
121:     def test_random_crop_output_shape(self, sample: torch.Tensor) -> None:
122:         cropped = random_crop(sample, seq_len=50)
123:         assert cropped.shape == (50, J, F)
124: 
125:     def test_center_crop_output_shape(self, sample: torch.Tensor) -> None:
126:         cropped = center_crop(sample, seq_len=50)
127:         assert cropped.shape == (50, J, F)
128: 
129:     def test_random_crop_no_crop_when_equal(self, sample: torch.Tensor) -> None:
130:         cropped = random_crop(sample, seq_len=T)
131:         assert cropped.shape == sample.shape
132: 
133:     def test_center_crop_no_crop_when_equal(self, sample: torch.Tensor) -> None:
134:         cropped = center_crop(sample, seq_len=T)
135:         assert cropped.shape == sample.shape
136: 
137:     def test_center_crop_is_centred(self, sample: torch.Tensor) -> None:
138:         seq_len = 40
139:         cropped = center_crop(sample, seq_len=seq_len)
140:         start = (T - seq_len) // 2
141:         assert torch.allclose(cropped, sample[start : start + seq_len])
142: 
143:     def test_random_crop_content_is_contiguous_window(
144:         self, sample: torch.Tensor
145:     ) -> None:
146:         seq_len = 30
147:         # Run several times — result must always be a contiguous window from sample
148:         for seed in range(5):
149:             torch.manual_seed(seed)
150:             cropped = random_crop(sample, seq_len=seq_len)
151:             found = False
152:             for start in range(T - seq_len + 1):
153:                 if torch.allclose(sample[start : start + seq_len], cropped):
154:                     found = True
155:                     break
156:             assert found, f"random_crop result (seed={seed}) is not a contiguous window"
157: 
158: 
159: # ── flip_sequence ─────────────────────────────────────────────────────────────
160: 
161: 
162: class TestFlipSequence:
163:     def test_output_shape(self, sample: torch.Tensor) -> None:
164:         flipped = flip_sequence(sample, MIRROR_JOINT_PAIRS, FLIP_FEATURE_INDICES)
165:         assert flipped.shape == sample.shape
166: 
167:     def test_double_flip_is_identity(self, sample: torch.Tensor) -> None:
168:         flipped = flip_sequence(sample, MIRROR_JOINT_PAIRS, FLIP_FEATURE_INDICES)
169:         double = flip_sequence(flipped, MIRROR_JOINT_PAIRS, FLIP_FEATURE_INDICES)
170:         assert torch.allclose(sample, double), "Double-flip must restore original"
171: 
172:     def test_does_not_mutate_input(self, sample: torch.Tensor) -> None:
173:         original = sample.clone()
174:         flip_sequence(sample, MIRROR_JOINT_PAIRS, FLIP_FEATURE_INDICES)
175:         assert torch.allclose(sample, original), "flip_sequence must not modify input"
176: 
177:     def test_x_axis_features_negated(self, sample: torch.Tensor) -> None:
178:         # Use empty mirror pairs — only x-axis negation should happen
179:         flipped = flip_sequence(sample, [], FLIP_FEATURE_INDICES)
180:         for fi in FLIP_FEATURE_INDICES:
181:             assert torch.allclose(flipped[:, :, fi], -sample[:, :, fi]), (
182:                 f"Feature index {fi} should be negated"
183:             )
184: 
185:     def test_non_flip_features_unchanged(self, sample: torch.Tensor) -> None:
186:         flipped = flip_sequence(sample, [], FLIP_FEATURE_INDICES)
187:         non_flip = [i for i in range(F) if i not in FLIP_FEATURE_INDICES]
188:         for fi in non_flip:
189:             assert torch.allclose(flipped[:, :, fi], sample[:, :, fi]), (
190:                 f"Feature index {fi} should be unchanged after flip"
191:             )
192: 
193: 
194: # ── joint_jitter ──────────────────────────────────────────────────────────────
195: 
196: 
197: class TestJointJitter:
198:     def test_output_shape(self, sample: torch.Tensor) -> None:
199:         noisy = joint_jitter(sample, std=0.01)
200:         assert noisy.shape == sample.shape
201: 
202:     def test_adds_noise(self, sample: torch.Tensor) -> None:
203:         torch.manual_seed(99)
204:         noisy = joint_jitter(sample, std=0.01)
205:         assert not torch.allclose(sample, noisy)
206: 
207:     def test_zero_std_is_identity(self, sample: torch.Tensor) -> None:
208:         noisy = joint_jitter(sample, std=0.0)
209:         assert torch.allclose(sample, noisy)
210: 
211:     def test_does_not_mutate_input(self, sample: torch.Tensor) -> None:
212:         original = sample.clone()
213:         joint_jitter(sample, std=0.1)
214:         assert torch.allclose(sample, original)
215: 
216: 
217: # ── speed_perturb ─────────────────────────────────────────────────────────────
218: 
219: 
220: class TestSpeedPerturb:
221:     def test_output_shape(self, sample: torch.Tensor) -> None:
222:         out = speed_perturb(sample, factor_range=(0.8, 1.2), target_len=T)
223:         assert out.shape == (T, J, F)
224: 
225:     def test_output_shape_different_target_len(self, sample: torch.Tensor) -> None:
226:         out = speed_perturb(sample, factor_range=(0.8, 1.2), target_len=50)
227:         assert out.shape == (50, J, F)
228: 
229:     def test_no_nan(self, sample: torch.Tensor) -> None:
230:         out = speed_perturb(sample, factor_range=(0.8, 1.2), target_len=T)
231:         assert not torch.isnan(out).any()
232: 
233:     def test_factor_1_is_near_identity(self, sample: torch.Tensor) -> None:
234:         # With factor fixed at exactly 1.0 the output should be very close
235:         out = speed_perturb(sample, factor_range=(1.0, 1.0), target_len=T)
236:         assert out.shape == (T, J, F)
237:         # Values may differ slightly due to linear interpolation rounding
238:         assert torch.allclose(out, sample, atol=1e-4)
239: 
240: 
241: # ── load_anubis_tensors ───────────────────────────────────────────────────────
242: 
243: 
244: class TestLoadAnubisTensors:
245:     def test_output_shapes(self, tmp_npy_files: Tuple[str, str, str]) -> None:
246:         _, fp, lp = tmp_npy_files
247:         feats, labels = load_anubis_tensors(fp, lp)
248:         assert feats.shape == (N, T, J, 9)
249:         assert labels.shape == (N,)
250: 
251:     def test_features_dtype(self, tmp_npy_files: Tuple[str, str, str]) -> None:
252:         _, fp, lp = tmp_npy_files
253:         feats, _ = load_anubis_tensors(fp, lp)
254:         assert feats.dtype == torch.float32
255: 
256:     def test_labels_dtype(self, tmp_npy_files: Tuple[str, str, str]) -> None:
257:         _, fp, lp = tmp_npy_files
258:         _, labels = load_anubis_tensors(fp, lp)
259:         assert labels.dtype == torch.int64
260: 
261:     def test_missing_features_raises(self, tmp_npy_files: Tuple[str, str, str]) -> None:
262:         _, _, lp = tmp_npy_files
263:         with pytest.raises(FileNotFoundError):
264:             load_anubis_tensors("/nonexistent/features.npy", lp)
265: 
266:     def test_missing_labels_raises(self, tmp_npy_files: Tuple[str, str, str]) -> None:
267:         _, fp, _ = tmp_npy_files
268:         with pytest.raises(FileNotFoundError):
269:             load_anubis_tensors(fp, "/nonexistent/labels.npy")
270: 
271:     def test_wrong_channel_count_raises(
272:         self, tmp_npy_files: Tuple[str, str, str]
273:     ) -> None:
274:         tmpdir, _, lp = tmp_npy_files
275:         bad = np.random.randn(N, 9, T, J).astype(np.float32)  # 9 channels, not 6
276:         bad_fp = os.path.join(tmpdir, "bad_features.npy")
277:         np.save(bad_fp, bad)
278:         with pytest.raises(ValueError, match="6 channels"):
279:             load_anubis_tensors(bad_fp, lp)
280: 
281:     def test_wrong_ndim_raises(self, tmp_npy_files: Tuple[str, str, str]) -> None:
282:         tmpdir, _, lp = tmp_npy_files
283:         bad = np.random.randn(N, T * J * C).astype(np.float32)  # 2D
284:         bad_fp = os.path.join(tmpdir, "bad_flat.npy")
285:         np.save(bad_fp, bad)
286:         with pytest.raises(ValueError, match="4 dimensions"):
287:             load_anubis_tensors(bad_fp, lp)
288: 
289:     def test_5d_input_selects_primary_person(
290:         self, tmp_npy_files: Tuple[str, str, str]
291:     ) -> None:
292:         tmpdir, _, lp = tmp_npy_files
293:         m = 2
294:         # Build deterministic data where person-0 is zeros and person-1 is ones.
295:         # Loader should select person-0 and thus return all zeros before acc concat.
296:         raw5d = np.ones((N, C, T, J, m), dtype=np.float32)
297:         raw5d[..., 0] = 0.0
298:         fp_5d = os.path.join(tmpdir, "features_5d.npy")
299:         np.save(fp_5d, raw5d)
300: 
301:         feats, _ = load_anubis_tensors(fp_5d, lp)
302:         # All channels should be zeros because person-0 contains zeros only.
303:         assert torch.allclose(feats, torch.zeros_like(feats))
304: 
305: 
306: # ── ANUBISDataset ─────────────────────────────────────────────────────────────
307: 
308: 
309: class TestANUBISDataset:
310:     def test_len(self, tensor_pair: Tuple[torch.Tensor, torch.Tensor]) -> None:
311:         features, labels = tensor_pair
312:         ds = ANUBISDataset(features, labels, seq_len=T)
313:         assert len(ds) == N
314: 
315:     def test_item_shapes_no_crop(
316:         self, tensor_pair: Tuple[torch.Tensor, torch.Tensor]
317:     ) -> None:
318:         features, labels = tensor_pair
319:         ds = ANUBISDataset(features, labels, seq_len=T, augment=False)
320:         x, y = ds[0]
321:         assert x.shape == (T, J, F)
322:         assert y.shape == ()
323: 
324:     def test_item_shapes_with_crop(
325:         self, tensor_pair: Tuple[torch.Tensor, torch.Tensor]
326:     ) -> None:
327:         features, labels = tensor_pair
328:         ds = ANUBISDataset(features, labels, seq_len=50, augment=False)
329:         x, y = ds[0]
330:         assert x.shape == (50, J, F), f"Expected (50,{J},{F}), got {x.shape}"
331: 
332:     def test_label_dtype(self, tensor_pair: Tuple[torch.Tensor, torch.Tensor]) -> None:
333:         features, labels = tensor_pair
334:         ds = ANUBISDataset(features, labels, seq_len=T)
335:         _, y = ds[0]
336:         assert y.dtype == torch.int64
337: 
338:     def test_augmentation_changes_output(
339:         self, tensor_pair: Tuple[torch.Tensor, torch.Tensor]
340:     ) -> None:
341:         features, labels = tensor_pair
342:         ds_aug = ANUBISDataset(
343:             features,
344:             labels,
345:             seq_len=T,
346:             augment=True,
347:             aug_cfg={"flip_prob": 1.0, "joint_jitter_std": 0.1, "speed_perturb": False},
348:         )
349:         ds_no_aug = ANUBISDataset(features, labels, seq_len=T, augment=False)
350:         # With flip_prob=1 and jitter the augmented output must differ from clean
351:         torch.manual_seed(0)
352:         x_aug, _ = ds_aug[0]
353:         x_clean, _ = ds_no_aug[0]
354:         assert not torch.allclose(x_aug, x_clean), (
355:             "Augmented sample must differ from clean sample"
356:         )
357: 
358:     def test_getitem_does_not_mutate_stored_tensor(
359:         self, tensor_pair: Tuple[torch.Tensor, torch.Tensor]
360:     ) -> None:
361:         features, labels = tensor_pair
362:         original = features[0].clone()
363:         ds = ANUBISDataset(
364:             features,
365:             labels,
366:             seq_len=T,
367:             augment=True,
368:             aug_cfg={"flip_prob": 1.0, "joint_jitter_std": 0.5, "speed_perturb": False},
369:         )
370:         _ = ds[0]
371:         assert torch.allclose(features[0], original), (
372:             "__getitem__ must not mutate the stored feature tensor"
373:         )

(End of file - total 373 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/tests/test_joint_config.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/tests/test_joint_config.py</path>
<type>file</type>
<content>1: # tests/test_joint_config.py
2: #
3: # Unit tests for module1/data/joint_config.py
4: # ─────────────────────────────────────────────
5: # Covers: joint count, edge validity, bidirectionality, mirror pairs,
6: #         flip feature indices, angle triplets, and body-region mappings.
7: #
8: # Run with: pytest tests/test_joint_config.py -v
9: 
10: import pytest
11: import torch
12: 
13: from module1.data.joint_config import (
14:     ANGLE_TRIPLETS,
15:     BODY_REGIONS,
16:     EDGE_INDEX,
17:     FLIP_FEATURE_INDICES,
18:     JOINT_NAMES,
19:     MIRROR_JOINT_PAIRS,
20:     NUM_EDGES,
21:     NUM_JOINTS,
22:     NUM_REGIONS,
23:     REGION_NAMES,
24:     SKELETON_EDGES,
25:     build_edge_index,
26: )
27: 
28: 
29: # ── Fixtures ──────────────────────────────────────────────────────────────────
30: 
31: 
32: @pytest.fixture
33: def edge_index() -> torch.Tensor:
34:     """Freshly built edge index tensor."""
35:     return build_edge_index()
36: 
37: 
38: # ── Joint names ───────────────────────────────────────────────────────────────
39: 
40: 
41: class TestJointNames:
42:     def test_count(self) -> None:
43:         assert len(JOINT_NAMES) == 32
44: 
45:     def test_num_joints_constant(self) -> None:
46:         assert NUM_JOINTS == 32
47: 
48:     def test_all_strings(self) -> None:
49:         for name in JOINT_NAMES:
50:             assert isinstance(name, str) and name
51: 
52:     def test_unique_names(self) -> None:
53:         assert len(JOINT_NAMES) == len(set(JOINT_NAMES)), "Duplicate joint names"
54: 
55:     def test_pelvis_is_root(self) -> None:
56:         assert JOINT_NAMES[0] == "Pelvis"
57: 
58: 
59: # ── Skeleton edges ────────────────────────────────────────────────────────────
60: 
61: 
62: class TestSkeletonEdges:
63:     def test_num_edges_constant_matches_list(self) -> None:
64:         assert NUM_EDGES == len(SKELETON_EDGES)
65: 
66:     def test_expected_edge_count(self) -> None:
67:         # 31 unique undirected edges (as defined in joint_config.py)
68:         assert NUM_EDGES == 31
69: 
70:     def test_no_self_loops(self) -> None:
71:         for src, dst in SKELETON_EDGES:
72:             assert src != dst, f"Self-loop at joint {src}"
73: 
74:     def test_node_indices_in_range(self) -> None:
75:         for src, dst in SKELETON_EDGES:
76:             assert 0 <= src < NUM_JOINTS, f"src {src} out of range"
77:             assert 0 <= dst < NUM_JOINTS, f"dst {dst} out of range"
78: 
79:     def test_no_duplicate_edges(self) -> None:
80:         # Count both (a,b) and (b,a) as the same undirected edge
81:         undirected = {frozenset((s, d)) for s, d in SKELETON_EDGES}
82:         assert len(undirected) == len(SKELETON_EDGES), "Duplicate edges detected"
83: 
84: 
85: # ── build_edge_index ──────────────────────────────────────────────────────────
86: 
87: 
88: class TestBuildEdgeIndex:
89:     def test_shape(self, edge_index: torch.Tensor) -> None:
90:         assert edge_index.shape == (2, 2 * NUM_EDGES)
91: 
92:     def test_dtype(self, edge_index: torch.Tensor) -> None:
93:         assert edge_index.dtype == torch.long
94: 
95:     def test_node_range(self, edge_index: torch.Tensor) -> None:
96:         assert int(edge_index.min()) >= 0
97:         assert int(edge_index.max()) < NUM_JOINTS
98: 
99:     def test_bidirectional(self, edge_index: torch.Tensor) -> None:
100:         # For every forward edge (s→d) there must be a reverse edge (d→s)
101:         edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
102:         for s, d in SKELETON_EDGES:
103:             assert (s, d) in edges, f"Missing forward edge ({s}→{d})"
104:             assert (d, s) in edges, f"Missing reverse edge ({d}→{s})"
105: 
106:     def test_prebuilt_constant_matches(self, edge_index: torch.Tensor) -> None:
107:         assert torch.equal(edge_index, EDGE_INDEX)
108: 
109: 
110: # ── Mirror pairs ──────────────────────────────────────────────────────────────
111: 
112: 
113: class TestMirrorPairs:
114:     def test_indices_in_range(self) -> None:
115:         for l, r in MIRROR_JOINT_PAIRS:
116:             assert 0 <= l < NUM_JOINTS, f"Left mirror index {l} out of range"
117:             assert 0 <= r < NUM_JOINTS, f"Right mirror index {r} out of range"
118: 
119:     def test_no_self_pairs(self) -> None:
120:         for l, r in MIRROR_JOINT_PAIRS:
121:             assert l != r, f"Self-pair at joint {l}"
122: 
123:     def test_no_duplicate_joint_appearances(self) -> None:
124:         # Each joint should appear at most once across all pairs
125:         flat = [j for pair in MIRROR_JOINT_PAIRS for j in pair]
126:         assert len(flat) == len(set(flat)), "A joint appears in multiple mirror pairs"
127: 
128:     def test_flip_feature_indices(self) -> None:
129:         # Must include x (0), vx (3), ax (6)
130:         assert set(FLIP_FEATURE_INDICES) == {0, 3, 6}
131: 
132: 
133: # ── Angle triplets ────────────────────────────────────────────────────────────
134: 
135: 
136: class TestAngleTriplets:
137:     def test_joint_indices_in_range(self) -> None:
138:         for joint_idx, (prox, dist) in ANGLE_TRIPLETS.items():
139:             assert 0 <= joint_idx < NUM_JOINTS
140:             assert 0 <= prox < NUM_JOINTS
141:             assert 0 <= dist < NUM_JOINTS
142: 
143:     def test_no_degenerate_triplets(self) -> None:
144:         # joint_idx must differ from both neighbours
145:         for joint_idx, (prox, dist) in ANGLE_TRIPLETS.items():
146:             assert joint_idx != prox
147:             assert joint_idx != dist
148: 
149: 
150: # ── Body regions ──────────────────────────────────────────────────────────────
151: 
152: 
153: class TestBodyRegions:
154:     def test_num_regions_constant(self) -> None:
155:         assert NUM_REGIONS == len(BODY_REGIONS) == len(REGION_NAMES)
156: 
157:     def test_expected_region_count(self) -> None:
158:         assert NUM_REGIONS == 11
159: 
160:     def test_all_joint_indices_in_range(self) -> None:
161:         for region, joints in BODY_REGIONS.items():
162:             for j in joints:
163:                 assert 0 <= j < NUM_JOINTS, (
164:                     f"Region '{region}' contains out-of-range joint {j}"
165:                 )
166: 
167:     def test_no_empty_regions(self) -> None:
168:         for region, joints in BODY_REGIONS.items():
169:             assert len(joints) > 0, f"Region '{region}' is empty"
170: 
171:     def test_region_names_list_matches_dict(self) -> None:
172:         assert REGION_NAMES == list(BODY_REGIONS.keys())

(End of file - total 172 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/conftest.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/conftest.py</path>
<type>file</type>
<content>1: # conftest.py
2: #
3: # Pytest root configuration — adds the project root to sys.path so that
4: # 'module1', 'module2', 'utils', etc. are importable without installation.
5: import sys
6: import os
7: 
8: sys.path.insert(0, os.path.dirname(__file__))

(End of file - total 8 lines)
</content>
```

---

## Assistant (Build · gpt-5.3-codex · 142.7s)

_Thinking:_

**Designing pose extraction pipeline**

_Thinking:_

**Implementing live inference script and tests**

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Add File: /home/alan/mydev/college/football-injury-ai/module1/pretrain/live_utils.py\n+# module1/pretrain/live_utils.py\n+#\n+# Stage 1 — Live Inference Utilities\n+# ───────────────────────────────────\n+# Helper functions shared by the live ANUBIS inference pipeline:\n+#   • class-label loading\n+#   • pelvis-centred normalisation\n+#   • velocity / acceleration feature construction\n+#   • probability smoothing\n+\n+from __future__ import annotations\n+\n+import json\n+import os\n+from typing import Dict, List, Optional, Union\n+\n+import numpy as np\n+import torch\n+import yaml\n+\n+\n+def parse_video_source(source: str) -> Union[int, str]:\n+    \"\"\"\n+    Convert a CLI source value to OpenCV's expected type.\n+\n+    Parameters\n+    ----------\n+    source : str\n+        Webcam index (e.g. \"0\") or path/URL.\n+\n+    Returns\n+    -------\n+    parsed_source : int | str\n+        Integer webcam index if input is numeric, otherwise the original string.\n+    \"\"\"\n+    stripped = source.strip()\n+    if stripped.isdigit():\n+        return int(stripped)\n+    return stripped\n+\n+\n+def normalise_pelvis_center(\n+    positions: np.ndarray,\n+    pelvis_idx: int = 0,\n+) -> np.ndarray:\n+    \"\"\"\n+    Pelvis-centre a window of joint coordinates.\n+\n+    Parameters\n+    ----------\n+    positions : np.ndarray  shape (T, J, 3)\n+        Raw joint coordinates in normalised image space.\n+    pelvis_idx : int\n+        Pelvis joint index in the skeleton definition.\n+\n+    Returns\n+    -------\n+    centered : np.ndarray  shape (T, J, 3)\n+        Pelvis-centred coordinates.\n+    \"\"\"\n+    if positions.ndim != 3 or positions.shape[2] != 3:\n+        raise ValueError(\n+            \"Expected positions shape (T, J, 3), \"\n+            f\"got {tuple(positions.shape)}\"\n+        )\n+    if not (0 <= pelvis_idx < positions.shape[1]):\n+        raise ValueError(\n+            f\"Invalid pelvis_idx={pelvis_idx} for J={positions.shape[1]} joints.\"\n+        )\n+\n+    centered = positions.astype(np.float32, copy=True)\n+    centered -= centered[:, pelvis_idx : pelvis_idx + 1, :]\n+    return centered\n+\n+\n+def compute_velocity(positions: np.ndarray) -> np.ndarray:\n+    \"\"\"\n+    Compute velocity via forward finite difference along time.\n+\n+    Boundary handling follows the ANUBIS preprocessing style:\n+    last velocity frame duplicates the previous one.\n+\n+    Parameters\n+    ----------\n+    positions : np.ndarray  shape (T, J, 3)\n+\n+    Returns\n+    -------\n+    velocity : np.ndarray  shape (T, J, 3)\n+    \"\"\"\n+    vel = np.zeros_like(positions, dtype=np.float32)\n+    if positions.shape[0] > 1:\n+        vel[:-1] = positions[1:] - positions[:-1]\n+        vel[-1] = vel[-2]\n+    return vel\n+\n+\n+def compute_acceleration(velocity: np.ndarray) -> np.ndarray:\n+    \"\"\"\n+    Compute acceleration via forward finite difference of velocity.\n+\n+    Boundary handling follows the ANUBIS preprocessing style:\n+    last acceleration frame duplicates the previous one.\n+\n+    Parameters\n+    ----------\n+    velocity : np.ndarray  shape (T, J, 3)\n+\n+    Returns\n+    -------\n+    acceleration : np.ndarray  shape (T, J, 3)\n+    \"\"\"\n+    acc = np.zeros_like(velocity, dtype=np.float32)\n+    if velocity.shape[0] > 1:\n+        acc[:-1] = velocity[1:] - velocity[:-1]\n+        acc[-1] = acc[-2]\n+    return acc\n+\n+\n+def build_feature_window(\n+    positions: np.ndarray,\n+    pelvis_idx: int = 0,\n+) -> torch.Tensor:\n+    \"\"\"\n+    Build model-ready per-joint features from a position window.\n+\n+    Output feature order per joint matches training:\n+      [x, y, z, vx, vy, vz, ax, ay, az]\n+\n+    Parameters\n+    ----------\n+    positions : np.ndarray  shape (T, J, 3)\n+        Raw 3D joint coordinates for one temporal window.\n+    pelvis_idx : int\n+        Pelvis index for centring.\n+\n+    Returns\n+    -------\n+    features : torch.Tensor  shape (T, J, 9)\n+        Float32 tensor ready to be batched as (B, T, J, 9).\n+    \"\"\"\n+    centered = normalise_pelvis_center(positions, pelvis_idx=pelvis_idx)\n+    vel = compute_velocity(centered)\n+    acc = compute_acceleration(vel)\n+    feat = np.concatenate([centered, vel, acc], axis=2).astype(np.float32, copy=False)\n+    return torch.from_numpy(feat)\n+\n+\n+def _from_mapping_dict(mapping: Dict, num_classes: int) -> List[str]:\n+    names = [f\"class_{i}\" for i in range(num_classes)]\n+    for key, value in mapping.items():\n+        idx = int(key)\n+        if 0 <= idx < num_classes:\n+            names[idx] = str(value)\n+    return names\n+\n+\n+def load_class_names(\n+    num_classes: int,\n+    label_map_path: Optional[str] = None,\n+) -> List[str]:\n+    \"\"\"\n+    Load class names for display from text / YAML / JSON mapping.\n+\n+    Supported formats\n+    -----------------\n+    - .txt  : one class name per line\n+    - .yaml : list of names OR dict {class_id: class_name}\n+    - .json : list of names OR dict {class_id: class_name}\n+\n+    Parameters\n+    ----------\n+    num_classes : int\n+        Number of model output classes.\n+    label_map_path : str, optional\n+        Optional file path containing class labels.\n+\n+    Returns\n+    -------\n+    class_names : list[str]\n+        Length == num_classes.\n+    \"\"\"\n+    fallback = [f\"class_{i}\" for i in range(num_classes)]\n+    if label_map_path is None:\n+        return fallback\n+\n+    if not os.path.exists(label_map_path):\n+        raise FileNotFoundError(\n+            f\"Label map not found: {label_map_path}\\n\"\n+            \"Provide a valid --label-map path or omit this flag.\"\n+        )\n+\n+    ext = os.path.splitext(label_map_path)[1].lower()\n+\n+    if ext == \".txt\":\n+        with open(label_map_path, \"r\", encoding=\"utf-8\") as f:\n+            lines = [line.strip() for line in f.readlines()]\n+        names = [line for line in lines if line]\n+        merged = fallback[:]\n+        limit = min(num_classes, len(names))\n+        merged[:limit] = names[:limit]\n+        return merged\n+\n+    if ext in (\".yaml\", \".yml\"):\n+        with open(label_map_path, \"r\", encoding=\"utf-8\") as f:\n+            payload = yaml.safe_load(f)\n+    elif ext == \".json\":\n+        with open(label_map_path, \"r\", encoding=\"utf-8\") as f:\n+            payload = json.load(f)\n+    else:\n+        raise ValueError(\n+            f\"Unsupported label-map extension: {ext}. \"\n+            \"Use .txt, .yaml/.yml, or .json.\"\n+        )\n+\n+    if isinstance(payload, list):\n+        merged = fallback[:]\n+        limit = min(num_classes, len(payload))\n+        for idx in range(limit):\n+            merged[idx] = str(payload[idx])\n+        return merged\n+\n+    if isinstance(payload, dict):\n+        return _from_mapping_dict(payload, num_classes)\n+\n+    raise ValueError(\n+        \"Label map must be a list or dict in YAML/JSON format, \"\n+        f\"got type={type(payload).__name__}.\"\n+    )\n+\n+\n+def smooth_probs(\n+    previous: Optional[torch.Tensor],\n+    current: torch.Tensor,\n+    alpha: float,\n+) -> torch.Tensor:\n+    \"\"\"\n+    Exponential moving average over class probabilities.\n+\n+    Parameters\n+    ----------\n+    previous : torch.Tensor, optional  shape (C,)\n+        Previous smoothed probability vector.\n+    current : torch.Tensor  shape (C,)\n+        Current probability vector.\n+    alpha : float\n+        EMA factor in [0, 1). Higher values produce smoother but slower output.\n+\n+    Returns\n+    -------\n+    smoothed : torch.Tensor  shape (C,)\n+    \"\"\"\n+    clipped_alpha = float(min(max(alpha, 0.0), 0.999))\n+    if previous is None:\n+        return current\n+    return (previous * clipped_alpha) + (current * (1.0 - clipped_alpha))\n+\n+\n+if __name__ == \"__main__\":\n+    demo = np.random.randn(60, 32, 3).astype(np.float32)\n+    feat = build_feature_window(demo)\n+    assert feat.shape == (60, 32, 9)\n+    print(\"live_utils OK.\")\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
A module1/pretrain/live_utils.py
```

---

## Assistant (Build · gpt-5.3-codex · 115.5s)

_Thinking:_

**Designing pose extractor and live inference**

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Add File: /home/alan/mydev/college/football-injury-ai/module1/pose/extractor.py\n+# module1/pose/extractor.py\n+#\n+# Stage 1 — Live Pose Extraction (MediaPipe → ANUBIS-32)\n+# ────────────────────────────────────────────────────────\n+# Converts MediaPipe Pose (33 landmarks) into the repository's 32-joint\n+# skeleton format used by the pretraining model.\n+\n+from __future__ import annotations\n+\n+from typing import Any, Optional\n+\n+import cv2\n+import numpy as np\n+\n+from utils.logger import get_logger\n+\n+log = get_logger(__name__)\n+\n+\n+NUM_MEDIAPIPE_JOINTS = 33\n+NUM_ANUBIS_JOINTS = 32\n+\n+\n+def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:\n+    return (a + b) * 0.5\n+\n+\n+def mediapipe33_to_anubis32(keypoints_33: np.ndarray) -> np.ndarray:\n+    \"\"\"\n+    Convert MediaPipe's 33-joint layout to the ANUBIS 32-joint layout.\n+\n+    Parameters\n+    ----------\n+    keypoints_33 : np.ndarray  shape (33, 3) or (33, 4)\n+        MediaPipe landmark array. If a 4th column exists (visibility), it is\n+        ignored in the output conversion.\n+\n+    Returns\n+    -------\n+    keypoints_32 : np.ndarray  shape (32, 3)\n+        Joint coordinates in ANUBIS order expected by module1/data/joint_config.py\n+    \"\"\"\n+    if keypoints_33.ndim != 2 or keypoints_33.shape[0] != NUM_MEDIAPIPE_JOINTS:\n+        raise ValueError(\n+            \"Expected keypoints shape (33, 3|4), \"\n+            f\"got {tuple(keypoints_33.shape)}\"\n+        )\n+    if keypoints_33.shape[1] < 3:\n+        raise ValueError(\n+            \"Expected at least 3 coordinates per keypoint, \"\n+            f\"got {keypoints_33.shape[1]}\"\n+        )\n+\n+    mp = keypoints_33[:, :3].astype(np.float32, copy=False)\n+    out = np.zeros((NUM_ANUBIS_JOINTS, 3), dtype=np.float32)\n+\n+    # MediaPipe references\n+    l_shoulder = mp[11]\n+    r_shoulder = mp[12]\n+    l_elbow = mp[13]\n+    r_elbow = mp[14]\n+    l_wrist = mp[15]\n+    r_wrist = mp[16]\n+    l_pinky = mp[17]\n+    r_pinky = mp[18]\n+    l_index = mp[19]\n+    r_index = mp[20]\n+    l_thumb = mp[21]\n+    r_thumb = mp[22]\n+    l_hip = mp[23]\n+    r_hip = mp[24]\n+    l_knee = mp[25]\n+    r_knee = mp[26]\n+    l_ankle = mp[27]\n+    r_ankle = mp[28]\n+    l_heel = mp[29]\n+    r_heel = mp[30]\n+    l_foot_index = mp[31]\n+    r_foot_index = mp[32]\n+\n+    pelvis = _midpoint(l_hip, r_hip)\n+    neck = _midpoint(l_shoulder, r_shoulder)\n+    head = (mp[0] + mp[7] + mp[8]) / 3.0\n+\n+    # Spine interpolation between pelvis and neck.\n+    spine1 = pelvis * 0.75 + neck * 0.25\n+    spine2 = pelvis * 0.50 + neck * 0.50\n+    spine3 = pelvis * 0.25 + neck * 0.75\n+\n+    # Left/right collar approximated between neck and shoulder.\n+    l_collar = _midpoint(neck, l_shoulder)\n+    r_collar = _midpoint(neck, r_shoulder)\n+\n+    # Hand centroids and fingertips.\n+    l_hand = (l_pinky + l_index + l_thumb) / 3.0\n+    r_hand = (r_pinky + r_index + r_thumb) / 3.0\n+    l_fingertip = l_index\n+    r_fingertip = r_index\n+\n+    # Fill ANUBIS-32 output layout.\n+    out[0] = pelvis\n+    out[1] = spine1\n+    out[2] = spine2\n+    out[3] = spine3\n+    out[4] = neck\n+    out[5] = head\n+\n+    out[6] = l_hip\n+    out[7] = l_knee\n+    out[8] = l_ankle\n+    out[9] = l_heel\n+    out[10] = l_foot_index\n+\n+    out[11] = r_hip\n+    out[12] = r_knee\n+    out[13] = r_ankle\n+    out[14] = r_heel\n+    out[15] = r_foot_index\n+\n+    out[16] = l_collar\n+    out[17] = l_shoulder\n+    out[18] = l_elbow\n+    out[19] = l_wrist\n+    out[20] = l_hand\n+    out[21] = l_fingertip\n+\n+    out[22] = r_collar\n+    out[23] = r_shoulder\n+    out[24] = r_elbow\n+    out[25] = r_wrist\n+    out[26] = r_hand\n+    out[27] = r_fingertip\n+\n+    # ANUBIS has toe-base and thumb-toe nodes; MediaPipe has one foot-index point.\n+    out[28] = l_foot_index\n+    out[29] = r_foot_index\n+    out[30] = l_foot_index\n+    out[31] = r_foot_index\n+\n+    return out\n+\n+\n+class PoseExtractor:\n+    \"\"\"\n+    Lightweight MediaPipe Pose wrapper for frame-wise extraction.\n+\n+    Usage\n+    -----\n+    with PoseExtractor() as extractor:\n+        joints_32 = extractor.extract_from_frame(frame_bgr)\n+    \"\"\"\n+\n+    def __init__(\n+        self,\n+        min_detection_confidence: float = 0.5,\n+        min_tracking_confidence: float = 0.5,\n+        model_complexity: int = 1,\n+    ) -> None:\n+        self.min_detection_confidence = float(min_detection_confidence)\n+        self.min_tracking_confidence = float(min_tracking_confidence)\n+        self.model_complexity = int(model_complexity)\n+\n+        self._mp_pose: Optional[Any] = None\n+        self._pose: Optional[Any] = None\n+\n+    def __enter__(self) -> \"PoseExtractor\":\n+        try:\n+            import mediapipe as mp\n+        except ImportError as exc:\n+            raise ImportError(\n+                \"mediapipe is not installed. Install dependencies and retry.\"\n+            ) from exc\n+\n+        self._mp_pose = mp.solutions.pose\n+        self._pose = self._mp_pose.Pose(\n+            static_image_mode=False,\n+            model_complexity=self.model_complexity,\n+            smooth_landmarks=True,\n+            min_detection_confidence=self.min_detection_confidence,\n+            min_tracking_confidence=self.min_tracking_confidence,\n+        )\n+        return self\n+\n+    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:\n+        if self._pose is not None:\n+            self._pose.close()\n+        self._pose = None\n+        self._mp_pose = None\n+\n+    def extract_from_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:\n+        \"\"\"\n+        Extract one ANUBIS-32 skeleton from a BGR frame.\n+\n+        Parameters\n+        ----------\n+        frame_bgr : np.ndarray  shape (H, W, 3)\n+\n+        Returns\n+        -------\n+        joints_32 : np.ndarray  shape (32, 3), optional\n+            None if no pose is detected in this frame.\n+        \"\"\"\n+        if self._pose is None:\n+            raise RuntimeError(\n+                \"PoseExtractor is not initialised. Use it as a context manager.\"\n+            )\n+\n+        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)\n+        results = self._pose.process(frame_rgb)\n+        if results.pose_landmarks is None:\n+            return None\n+\n+        keypoints_33 = np.array(\n+            [\n+                [lm.x, lm.y, lm.z, lm.visibility]\n+                for lm in results.pose_landmarks.landmark\n+            ],\n+            dtype=np.float32,\n+        )\n+        return mediapipe33_to_anubis32(keypoints_33)\n+\n+\n+if __name__ == \"__main__\":\n+    cap = cv2.VideoCapture(0)\n+    if not cap.isOpened():\n+        print(\"Could not open webcam (index 0).\")\n+    else:\n+        with PoseExtractor() as extractor:\n+            ok, frame = cap.read()\n+            if ok:\n+                joints = extractor.extract_from_frame(frame)\n+                if joints is not None:\n+                    print(f\"Extracted shape: {joints.shape}\")\n+                else:\n+                    print(\"No pose detected on first frame.\")\n+    cap.release()\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
A module1/pose/extractor.py
```

---

## Assistant (Build · gpt-5.3-codex · 40.5s)

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Add File: /home/alan/mydev/college/football-injury-ai/module1/pretrain/live_inference.py\n+# module1/pretrain/live_inference.py\n+#\n+# Stage 1 — Live ANUBIS Motion Inference\n+# ──────────────────────────────────────\n+# Runs webcam/video/RTSP live inference using the pretrained ActionClassifier\n+# backbone. This is an action-understanding test pipeline for the pretrained\n+# model (102 classes), not the football injury-risk head.\n+\n+from __future__ import annotations\n+\n+import argparse\n+import collections\n+import os\n+import time\n+from typing import Deque, List, Optional, Tuple\n+\n+import cv2\n+import numpy as np\n+import torch\n+import torch.nn.functional as F\n+\n+from module1.data.joint_config import EDGE_INDEX\n+from module1.pose.extractor import PoseExtractor\n+from module1.pretrain.live_utils import (\n+    build_feature_window,\n+    load_class_names,\n+    parse_video_source,\n+    smooth_probs,\n+)\n+from module1.pretrain.model import ActionClassifier, build_model\n+from utils.config import load_config\n+from utils.logger import get_logger\n+\n+log = get_logger(__name__)\n+\n+\n+def _draw_overlay(\n+    frame: np.ndarray,\n+    class_names: List[str],\n+    probs: torch.Tensor,\n+    topk: int,\n+    fps: float,\n+    frames_ready: int,\n+    seq_len: int,\n+) -> np.ndarray:\n+    \"\"\"Draw prediction text overlay onto a frame.\"\"\"\n+    canvas = frame.copy()\n+    H, W = canvas.shape[:2]\n+\n+    # Background panel\n+    panel_h = 30 + (topk + 3) * 24\n+    cv2.rectangle(canvas, (10, 10), (min(W - 10, 680), 10 + panel_h), (0, 0, 0), -1)\n+    cv2.rectangle(canvas, (10, 10), (min(W - 10, 680), 10 + panel_h), (70, 70, 70), 1)\n+\n+    y = 34\n+    cv2.putText(\n+        canvas,\n+        \"Live ANUBIS Motion Inference (Pretraining Model)\",\n+        (20, y),\n+        cv2.FONT_HERSHEY_SIMPLEX,\n+        0.6,\n+        (230, 230, 230),\n+        1,\n+        cv2.LINE_AA,\n+    )\n+\n+    y += 26\n+    cv2.putText(\n+        canvas,\n+        f\"FPS: {fps:.1f} | Window: {frames_ready}/{seq_len}\",\n+        (20, y),\n+        cv2.FONT_HERSHEY_SIMPLEX,\n+        0.55,\n+        (170, 255, 170),\n+        1,\n+        cv2.LINE_AA,\n+    )\n+\n+    topv, topi = torch.topk(probs, k=min(topk, probs.numel()))\n+    for rank, (score, idx) in enumerate(zip(topv.tolist(), topi.tolist()), start=1):\n+        y += 24\n+        label = class_names[idx] if idx < len(class_names) else f\"class_{idx}\"\n+        text = f\"{rank}. [{idx:03d}] {label}  prob={score:.3f}\"\n+        cv2.putText(\n+            canvas,\n+            text,\n+            (20, y),\n+            cv2.FONT_HERSHEY_SIMPLEX,\n+            0.55,\n+            (255, 255, 255),\n+            1,\n+            cv2.LINE_AA,\n+        )\n+\n+    return canvas\n+\n+\n+def _load_checkpoint(\n+    ckpt_path: str,\n+    classifier: ActionClassifier,\n+    device: torch.device,\n+) -> None:\n+    if not os.path.exists(ckpt_path):\n+        raise FileNotFoundError(\n+            f\"Checkpoint not found: {ckpt_path}\\n\"\n+            \"Run pretraining first or provide --ckpt path.\"\n+        )\n+\n+    ckpt = torch.load(ckpt_path, map_location=device)\n+    if \"classifier\" in ckpt:\n+        classifier.load_state_dict(ckpt[\"classifier\"])\n+    else:\n+        # Fallback if a raw state_dict is provided.\n+        classifier.load_state_dict(ckpt)\n+\n+    log.info(\n+        \"Loaded checkpoint from %s (epoch=%s val_top1=%s)\",\n+        ckpt_path,\n+        str(ckpt.get(\"epoch\", \"n/a\")) if isinstance(ckpt, dict) else \"n/a\",\n+        (\n+            f\"{ckpt.get('val_top1', float('nan')):.3f}\"\n+            if isinstance(ckpt, dict) and \"val_top1\" in ckpt\n+            else \"n/a\"\n+        ),\n+    )\n+\n+\n+def run_live(\n+    config_path: str,\n+    ckpt_path: str,\n+    source: str,\n+    topk: int,\n+    smooth_alpha: float,\n+    label_map: Optional[str],\n+    show: bool,\n+) -> None:\n+    \"\"\"\n+    Run live action inference from a camera/video stream.\n+\n+    Parameters\n+    ----------\n+    config_path : str\n+    ckpt_path : str\n+    source : str\n+        Camera index (e.g. \"0\"), video path, or RTSP URL.\n+    topk : int\n+        Number of top predictions shown in the overlay.\n+    smooth_alpha : float\n+        EMA alpha for probability smoothing (0=no smoothing, close to 1=strong).\n+    label_map : str, optional\n+        Optional class-name map file (.txt/.yaml/.json).\n+    show : bool\n+        Whether to display OpenCV window. Disable for headless testing.\n+    \"\"\"\n+    cfg = load_config(config_path)\n+    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n+    log.info(\"Device: %s\", device)\n+\n+    edge_index = EDGE_INDEX.to(device)\n+    _, classifier = build_model(cfg, edge_index)\n+    classifier = classifier.to(device)\n+    classifier.eval()\n+    _load_checkpoint(ckpt_path, classifier, device)\n+\n+    num_classes = int(cfg[\"data\"][\"num_classes\"])\n+    seq_len = int(cfg[\"data\"][\"seq_len\"])\n+    class_names = load_class_names(num_classes, label_map)\n+\n+    parsed_source = parse_video_source(source)\n+    cap = cv2.VideoCapture(parsed_source)\n+    if not cap.isOpened():\n+        raise RuntimeError(\n+            f\"Could not open source: {source}\\n\"\n+            \"Check camera index/path/URL and retry.\"\n+        )\n+\n+    log.info(\n+        \"Starting live inference from source=%s | seq_len=%d | topk=%d\",\n+        str(source),\n+        seq_len,\n+        topk,\n+    )\n+\n+    frame_buffer: Deque[np.ndarray] = collections.deque(maxlen=seq_len)\n+    smoothed_probs: Optional[torch.Tensor] = None\n+    fps = 0.0\n+    prev_time = time.time()\n+\n+    with PoseExtractor() as extractor:\n+        try:\n+            while True:\n+                ok, frame = cap.read()\n+                if not ok:\n+                    log.info(\"Stream ended or frame read failed.\")\n+                    break\n+\n+                joints32 = extractor.extract_from_frame(frame)\n+                if joints32 is not None:\n+                    frame_buffer.append(joints32)\n+\n+                # Default distribution before warm-up.\n+                probs = torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float32)\n+\n+                if len(frame_buffer) == seq_len:\n+                    window = np.stack(frame_buffer, axis=0)  # (T, J, 3)\n+                    features = build_feature_window(window, pelvis_idx=0)  # (T, J, 9)\n+\n+                    with torch.no_grad():\n+                        x = features.unsqueeze(0).to(device)  # (1, T, J, 9)\n+                        logits = classifier(x).squeeze(0)\n+                        current_probs = F.softmax(logits, dim=0).cpu()\n+\n+                    probs = smooth_probs(smoothed_probs, current_probs, alpha=smooth_alpha)\n+                    smoothed_probs = probs\n+\n+                # FPS update\n+                now = time.time()\n+                delta = max(now - prev_time, 1e-6)\n+                fps = 0.9 * fps + 0.1 * (1.0 / delta)\n+                prev_time = now\n+\n+                if show:\n+                    overlay = _draw_overlay(\n+                        frame=frame,\n+                        class_names=class_names,\n+                        probs=probs,\n+                        topk=topk,\n+                        fps=fps,\n+                        frames_ready=len(frame_buffer),\n+                        seq_len=seq_len,\n+                    )\n+                    cv2.imshow(\"Football Injury AI — Live Motion Test\", overlay)\n+                    key = cv2.waitKey(1) & 0xFF\n+                    if key in (27, ord(\"q\")):\n+                        break\n+\n+        finally:\n+            cap.release()\n+            if show:\n+                cv2.destroyAllWindows()\n+\n+\n+def main() -> None:\n+    parser = argparse.ArgumentParser(\n+        description=\"Run live ANUBIS action inference from webcam/video stream\"\n+    )\n+    parser.add_argument(\n+        \"--config\",\n+        default=\"configs/pretrain_final.yaml\",\n+        help=\"Path to pretrain config YAML\",\n+    )\n+    parser.add_argument(\n+        \"--ckpt\",\n+        default=\"outputs/checkpoints/pretrain/pretrain_best.pt\",\n+        help=\"Path to classifier checkpoint\",\n+    )\n+    parser.add_argument(\n+        \"--source\",\n+        default=\"0\",\n+        help=\"Video source: webcam index, file path, or RTSP URL\",\n+    )\n+    parser.add_argument(\n+        \"--topk\",\n+        type=int,\n+        default=5,\n+        help=\"Number of top classes shown\",\n+    )\n+    parser.add_argument(\n+        \"--smooth-alpha\",\n+        type=float,\n+        default=0.7,\n+        help=\"EMA alpha for probability smoothing in [0,1)\",\n+    )\n+    parser.add_argument(\n+        \"--label-map\",\n+        default=None,\n+        help=\"Optional class-name map (.txt/.yaml/.json)\",\n+    )\n+    parser.add_argument(\n+        \"--no-show\",\n+        action=\"store_true\",\n+        help=\"Disable OpenCV preview window (headless)\",\n+    )\n+    args = parser.parse_args()\n+\n+    run_live(\n+        config_path=args.config,\n+        ckpt_path=args.ckpt,\n+        source=args.source,\n+        topk=max(1, args.topk),\n+        smooth_alpha=args.smooth_alpha,\n+        label_map=args.label_map,\n+        show=not args.no_show,\n+    )\n+\n+\n+if __name__ == \"__main__\":\n+    main()\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
A module1/pretrain/live_inference.py
```

---

## Assistant (Build · gpt-5.3-codex · 52.9s)

_Thinking:_

**Creating comprehensive live utils tests**

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Add File: /home/alan/mydev/college/football-injury-ai/tests/test_live_pipeline.py\n+# tests/test_live_pipeline.py\n+#\n+# Unit tests for module1/pretrain/live_utils.py and module1/pose/extractor.py\n+# ─────────────────────────────────────────────────────────────────────────────\n+# Covers: source parsing, feature-window construction (pos+vel+acc), label-map\n+# loading, EMA smoothing, and MediaPipe33->ANUBIS32 joint conversion.\n+#\n+# Run with: pytest tests/test_live_pipeline.py -v\n+\n+from __future__ import annotations\n+\n+import json\n+from typing import List\n+\n+import numpy as np\n+import pytest\n+import torch\n+import yaml\n+\n+from module1.pose.extractor import mediapipe33_to_anubis32\n+from module1.pretrain.live_utils import (\n+    build_feature_window,\n+    compute_acceleration,\n+    compute_velocity,\n+    load_class_names,\n+    normalise_pelvis_center,\n+    parse_video_source,\n+    smooth_probs,\n+)\n+\n+\n+class TestParseVideoSource:\n+    def test_numeric_source_returns_int(self) -> None:\n+        assert parse_video_source(\"0\") == 0\n+        assert parse_video_source(\"12\") == 12\n+\n+    def test_path_or_url_returns_str(self) -> None:\n+        assert parse_video_source(\"data/samples/clip.mp4\") == \"data/samples/clip.mp4\"\n+        assert parse_video_source(\"rtsp://cam/live\") == \"rtsp://cam/live\"\n+\n+\n+class TestFeatureConstruction:\n+    def test_normalise_pelvis_center(self) -> None:\n+        x = np.zeros((3, 4, 3), dtype=np.float32)\n+        x[:, 0, :] = np.array([2.0, -1.0, 0.5], dtype=np.float32)\n+        x[:, 1, :] = np.array([3.0, 2.0, 0.5], dtype=np.float32)\n+\n+        out = normalise_pelvis_center(x, pelvis_idx=0)\n+        assert np.allclose(out[:, 0, :], 0.0)\n+        assert np.allclose(out[:, 1, :], np.array([1.0, 3.0, 0.0], dtype=np.float32))\n+\n+    def test_velocity_and_acceleration_boundary_handling(self) -> None:\n+        pos = np.zeros((4, 1, 3), dtype=np.float32)\n+        pos[:, 0, 0] = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32)\n+\n+        vel = compute_velocity(pos)\n+        acc = compute_acceleration(vel)\n+\n+        # Forward differences on x channel\n+        assert np.allclose(vel[:, 0, 0], np.array([1.0, 2.0, 3.0, 3.0], dtype=np.float32))\n+        assert np.allclose(acc[:, 0, 0], np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))\n+\n+    def test_build_feature_window_shape_and_order(self) -> None:\n+        T = 5\n+        J = 3\n+        pos = np.zeros((T, J, 3), dtype=np.float32)\n+        # Simple motion for joint 1 on x axis; pelvis is joint 0.\n+        pos[:, 0, :] = np.array([10.0, 0.0, 0.0], dtype=np.float32)\n+        pos[:, 1, 0] = np.arange(T, dtype=np.float32)\n+\n+        feat = build_feature_window(pos, pelvis_idx=0)\n+        assert feat.shape == (T, J, 9)\n+\n+        # Channels 0:3 are centred positions. Pelvis must be zeros.\n+        assert torch.allclose(feat[:, 0, 0:3], torch.zeros((T, 3), dtype=torch.float32))\n+        # Channels 3:6 are velocity. Joint-1 x velocity is 1 except last duplicated.\n+        assert torch.allclose(feat[:, 1, 3], torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32))\n+\n+\n+class TestClassNames:\n+    def test_default_fallback(self) -> None:\n+        names = load_class_names(num_classes=4, label_map_path=None)\n+        assert names == [\"class_0\", \"class_1\", \"class_2\", \"class_3\"]\n+\n+    def test_load_from_txt(self, tmp_path: pytest.TempPathFactory) -> None:\n+        p = tmp_path / \"labels.txt\"\n+        p.write_text(\"walk\\nrun\\njump\\n\", encoding=\"utf-8\")\n+        names = load_class_names(num_classes=5, label_map_path=str(p))\n+        assert names[:3] == [\"walk\", \"run\", \"jump\"]\n+        assert names[4] == \"class_4\"\n+\n+    def test_load_from_yaml_dict(self, tmp_path: pytest.TempPathFactory) -> None:\n+        p = tmp_path / \"labels.yaml\"\n+        payload = {0: \"walk\", 2: \"kick\"}\n+        p.write_text(yaml.safe_dump(payload), encoding=\"utf-8\")\n+        names = load_class_names(num_classes=4, label_map_path=str(p))\n+        assert names == [\"walk\", \"class_1\", \"kick\", \"class_3\"]\n+\n+    def test_load_from_json_list(self, tmp_path: pytest.TempPathFactory) -> None:\n+        p = tmp_path / \"labels.json\"\n+        payload: List[str] = [\"a\", \"b\"]\n+        p.write_text(json.dumps(payload), encoding=\"utf-8\")\n+        names = load_class_names(num_classes=3, label_map_path=str(p))\n+        assert names == [\"a\", \"b\", \"class_2\"]\n+\n+\n+class TestSmoothProbs:\n+    def test_first_step_returns_current(self) -> None:\n+        cur = torch.tensor([0.2, 0.8])\n+        out = smooth_probs(None, cur, alpha=0.7)\n+        assert torch.allclose(out, cur)\n+\n+    def test_ema(self) -> None:\n+        prev = torch.tensor([0.5, 0.5])\n+        cur = torch.tensor([0.1, 0.9])\n+        out = smooth_probs(prev, cur, alpha=0.5)\n+        assert torch.allclose(out, torch.tensor([0.3, 0.7]))\n+\n+\n+class TestMediapipeConversion:\n+    def test_output_shape(self) -> None:\n+        mp = np.zeros((33, 4), dtype=np.float32)\n+        out = mediapipe33_to_anubis32(mp)\n+        assert out.shape == (32, 3)\n+\n+    def test_pelvis_is_midpoint_of_hips(self) -> None:\n+        mp = np.zeros((33, 4), dtype=np.float32)\n+        mp[23, :3] = np.array([2.0, 0.0, 0.0], dtype=np.float32)  # left hip\n+        mp[24, :3] = np.array([4.0, 2.0, 0.0], dtype=np.float32)  # right hip\n+        out = mediapipe33_to_anubis32(mp)\n+        assert np.allclose(out[0], np.array([3.0, 1.0, 0.0], dtype=np.float32))\n+\n+    def test_direct_joint_mappings(self) -> None:\n+        mp = np.zeros((33, 4), dtype=np.float32)\n+        mp[23, :3] = np.array([1, 2, 3], dtype=np.float32)  # L_Hip\n+        mp[24, :3] = np.array([4, 5, 6], dtype=np.float32)  # R_Hip\n+        mp[25, :3] = np.array([7, 8, 9], dtype=np.float32)  # L_Knee\n+        mp[26, :3] = np.array([10, 11, 12], dtype=np.float32)  # R_Knee\n+        out = mediapipe33_to_anubis32(mp)\n+        assert np.allclose(out[6], mp[23, :3])\n+        assert np.allclose(out[11], mp[24, :3])\n+        assert np.allclose(out[7], mp[25, :3])\n+        assert np.allclose(out[12], mp[26, :3])\n+\n+    def test_invalid_shape_raises(self) -> None:\n+        bad = np.zeros((32, 3), dtype=np.float32)\n+        with pytest.raises(ValueError):\n+            mediapipe33_to_anubis32(bad)\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
A tests/test_live_pipeline.py
```

---

## Assistant (Build · gpt-5.3-codex · 29.8s)

_Thinking:_

**Reviewing type hints and docs updates**

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/module1/pretrain/live_utils.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/module1/pretrain/live_utils.py</path>
<type>file</type>
<content>1: # module1/pretrain/live_utils.py
2: #
3: # Stage 1 — Live Inference Utilities
4: # ───────────────────────────────────
5: # Helper functions shared by the live ANUBIS inference pipeline:
6: #   • class-label loading
7: #   • pelvis-centred normalisation
8: #   • velocity / acceleration feature construction
9: #   • probability smoothing
10: 
11: from __future__ import annotations
12: 
13: import json
14: import os
15: from typing import Dict, List, Optional, Union
16: 
17: import numpy as np
18: import torch
19: import yaml
20: 
21: 
22: def parse_video_source(source: str) -> Union[int, str]:
23:     """
24:     Convert a CLI source value to OpenCV's expected type.
25: 
26:     Parameters
27:     ----------
28:     source : str
29:         Webcam index (e.g. "0") or path/URL.
30: 
31:     Returns
32:     -------
33:     parsed_source : int | str
34:         Integer webcam index if input is numeric, otherwise the original string.
35:     """
36:     stripped = source.strip()
37:     if stripped.isdigit():
38:         return int(stripped)
39:     return stripped
40: 
41: 
42: def normalise_pelvis_center(
43:     positions: np.ndarray,
44:     pelvis_idx: int = 0,
45: ) -> np.ndarray:
46:     """
47:     Pelvis-centre a window of joint coordinates.
48: 
49:     Parameters
50:     ----------
51:     positions : np.ndarray  shape (T, J, 3)
52:         Raw joint coordinates in normalised image space.
53:     pelvis_idx : int
54:         Pelvis joint index in the skeleton definition.
55: 
56:     Returns
57:     -------
58:     centered : np.ndarray  shape (T, J, 3)
59:         Pelvis-centred coordinates.
60:     """
61:     if positions.ndim != 3 or positions.shape[2] != 3:
62:         raise ValueError(
63:             f"Expected positions shape (T, J, 3), got {tuple(positions.shape)}"
64:         )
65:     if not (0 <= pelvis_idx < positions.shape[1]):
66:         raise ValueError(
67:             f"Invalid pelvis_idx={pelvis_idx} for J={positions.shape[1]} joints."
68:         )
69: 
70:     centered = positions.astype(np.float32, copy=True)
71:     centered -= centered[:, pelvis_idx : pelvis_idx + 1, :]
72:     return centered
73: 
74: 
75: def compute_velocity(positions: np.ndarray) -> np.ndarray:
76:     """
77:     Compute velocity via forward finite difference along time.
78: 
79:     Boundary handling follows the ANUBIS preprocessing style:
80:     last velocity frame duplicates the previous one.
81: 
82:     Parameters
83:     ----------
84:     positions : np.ndarray  shape (T, J, 3)
85: 
86:     Returns
87:     -------
88:     velocity : np.ndarray  shape (T, J, 3)
89:     """
90:     vel = np.zeros_like(positions, dtype=np.float32)
91:     if positions.shape[0] > 1:
92:         vel[:-1] = positions[1:] - positions[:-1]
93:         vel[-1] = vel[-2]
94:     return vel
95: 
96: 
97: def compute_acceleration(velocity: np.ndarray) -> np.ndarray:
98:     """
99:     Compute acceleration via forward finite difference of velocity.
100: 
101:     Boundary handling follows the ANUBIS preprocessing style:
102:     last acceleration frame duplicates the previous one.
103: 
104:     Parameters
105:     ----------
106:     velocity : np.ndarray  shape (T, J, 3)
107: 
108:     Returns
109:     -------
110:     acceleration : np.ndarray  shape (T, J, 3)
111:     """
112:     acc = np.zeros_like(velocity, dtype=np.float32)
113:     if velocity.shape[0] > 1:
114:         acc[:-1] = velocity[1:] - velocity[:-1]
115:         acc[-1] = acc[-2]
116:     return acc
117: 
118: 
119: def build_feature_window(
120:     positions: np.ndarray,
121:     pelvis_idx: int = 0,
122: ) -> torch.Tensor:
123:     """
124:     Build model-ready per-joint features from a position window.
125: 
126:     Output feature order per joint matches training:
127:       [x, y, z, vx, vy, vz, ax, ay, az]
128: 
129:     Parameters
130:     ----------
131:     positions : np.ndarray  shape (T, J, 3)
132:         Raw 3D joint coordinates for one temporal window.
133:     pelvis_idx : int
134:         Pelvis index for centring.
135: 
136:     Returns
137:     -------
138:     features : torch.Tensor  shape (T, J, 9)
139:         Float32 tensor ready to be batched as (B, T, J, 9).
140:     """
141:     centered = normalise_pelvis_center(positions, pelvis_idx=pelvis_idx)
142:     vel = compute_velocity(centered)
143:     acc = compute_acceleration(vel)
144:     feat = np.concatenate([centered, vel, acc], axis=2).astype(np.float32, copy=False)
145:     return torch.from_numpy(feat)
146: 
147: 
148: def _from_mapping_dict(mapping: Dict, num_classes: int) -> List[str]:
149:     names = [f"class_{i}" for i in range(num_classes)]
150:     for key, value in mapping.items():
151:         idx = int(key)
152:         if 0 <= idx < num_classes:
153:             names[idx] = str(value)
154:     return names
155: 
156: 
157: def load_class_names(
158:     num_classes: int,
159:     label_map_path: Optional[str] = None,
160: ) -> List[str]:
161:     """
162:     Load class names for display from text / YAML / JSON mapping.
163: 
164:     Supported formats
165:     -----------------
166:     - .txt  : one class name per line
167:     - .yaml : list of names OR dict {class_id: class_name}
168:     - .json : list of names OR dict {class_id: class_name}
169: 
170:     Parameters
171:     ----------
172:     num_classes : int
173:         Number of model output classes.
174:     label_map_path : str, optional
175:         Optional file path containing class labels.
176: 
177:     Returns
178:     -------
179:     class_names : list[str]
180:         Length == num_classes.
181:     """
182:     fallback = [f"class_{i}" for i in range(num_classes)]
183:     if label_map_path is None:
184:         return fallback
185: 
186:     if not os.path.exists(label_map_path):
187:         raise FileNotFoundError(
188:             f"Label map not found: {label_map_path}\n"
189:             "Provide a valid --label-map path or omit this flag."
190:         )
191: 
192:     ext = os.path.splitext(label_map_path)[1].lower()
193: 
194:     if ext == ".txt":
195:         with open(label_map_path, "r", encoding="utf-8") as f:
196:             lines = [line.strip() for line in f.readlines()]
197:         names = [line for line in lines if line]
198:         merged = fallback[:]
199:         limit = min(num_classes, len(names))
200:         merged[:limit] = names[:limit]
201:         return merged
202: 
203:     if ext in (".yaml", ".yml"):
204:         with open(label_map_path, "r", encoding="utf-8") as f:
205:             payload = yaml.safe_load(f)
206:     elif ext == ".json":
207:         with open(label_map_path, "r", encoding="utf-8") as f:
208:             payload = json.load(f)
209:     else:
210:         raise ValueError(
211:             f"Unsupported label-map extension: {ext}. Use .txt, .yaml/.yml, or .json."
212:         )
213: 
214:     if isinstance(payload, list):
215:         merged = fallback[:]
216:         limit = min(num_classes, len(payload))
217:         for idx in range(limit):
218:             merged[idx] = str(payload[idx])
219:         return merged
220: 
221:     if isinstance(payload, dict):
222:         return _from_mapping_dict(payload, num_classes)
223: 
224:     raise ValueError(
225:         "Label map must be a list or dict in YAML/JSON format, "
226:         f"got type={type(payload).__name__}."
227:     )
228: 
229: 
230: def smooth_probs(
231:     previous: Optional[torch.Tensor],
232:     current: torch.Tensor,
233:     alpha: float,
234: ) -> torch.Tensor:
235:     """
236:     Exponential moving average over class probabilities.
237: 
238:     Parameters
239:     ----------
240:     previous : torch.Tensor, optional  shape (C,)
241:         Previous smoothed probability vector.
242:     current : torch.Tensor  shape (C,)
243:         Current probability vector.
244:     alpha : float
245:         EMA factor in [0, 1). Higher values produce smoother but slower output.
246: 
247:     Returns
248:     -------
249:     smoothed : torch.Tensor  shape (C,)
250:     """
251:     clipped_alpha = float(min(max(alpha, 0.0), 0.999))
252:     if previous is None:
253:         return current
254:     return (previous * clipped_alpha) + (current * (1.0 - clipped_alpha))
255: 
256: 
257: if __name__ == "__main__":
258:     demo = np.random.randn(60, 32, 3).astype(np.float32)
259:     feat = build_feature_window(demo)
260:     assert feat.shape == (60, 32, 9)
261:     print("live_utils OK.")

(End of file - total 261 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/module1/pose/extractor.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/module1/pose/extractor.py</path>
<type>file</type>
<content>1: # module1/pose/extractor.py
2: #
3: # Stage 1 — Live Pose Extraction (MediaPipe → ANUBIS-32)
4: # ────────────────────────────────────────────────────────
5: # Converts MediaPipe Pose (33 landmarks) into the repository's 32-joint
6: # skeleton format used by the pretraining model.
7: 
8: from __future__ import annotations
9: 
10: from typing import Any, Optional
11: 
12: import cv2
13: import numpy as np
14: 
15: from utils.logger import get_logger
16: 
17: log = get_logger(__name__)
18: 
19: 
20: NUM_MEDIAPIPE_JOINTS = 33
21: NUM_ANUBIS_JOINTS = 32
22: 
23: 
24: def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
25:     return (a + b) * 0.5
26: 
27: 
28: def mediapipe33_to_anubis32(keypoints_33: np.ndarray) -> np.ndarray:
29:     """
30:     Convert MediaPipe's 33-joint layout to the ANUBIS 32-joint layout.
31: 
32:     Parameters
33:     ----------
34:     keypoints_33 : np.ndarray  shape (33, 3) or (33, 4)
35:         MediaPipe landmark array. If a 4th column exists (visibility), it is
36:         ignored in the output conversion.
37: 
38:     Returns
39:     -------
40:     keypoints_32 : np.ndarray  shape (32, 3)
41:         Joint coordinates in ANUBIS order expected by module1/data/joint_config.py
42:     """
43:     if keypoints_33.ndim != 2 or keypoints_33.shape[0] != NUM_MEDIAPIPE_JOINTS:
44:         raise ValueError(
45:             f"Expected keypoints shape (33, 3|4), got {tuple(keypoints_33.shape)}"
46:         )
47:     if keypoints_33.shape[1] < 3:
48:         raise ValueError(
49:             f"Expected at least 3 coordinates per keypoint, got {keypoints_33.shape[1]}"
50:         )
51: 
52:     mp = keypoints_33[:, :3].astype(np.float32, copy=False)
53:     out = np.zeros((NUM_ANUBIS_JOINTS, 3), dtype=np.float32)
54: 
55:     # MediaPipe references
56:     l_shoulder = mp[11]
57:     r_shoulder = mp[12]
58:     l_elbow = mp[13]
59:     r_elbow = mp[14]
60:     l_wrist = mp[15]
61:     r_wrist = mp[16]
62:     l_pinky = mp[17]
63:     r_pinky = mp[18]
64:     l_index = mp[19]
65:     r_index = mp[20]
66:     l_thumb = mp[21]
67:     r_thumb = mp[22]
68:     l_hip = mp[23]
69:     r_hip = mp[24]
70:     l_knee = mp[25]
71:     r_knee = mp[26]
72:     l_ankle = mp[27]
73:     r_ankle = mp[28]
74:     l_heel = mp[29]
75:     r_heel = mp[30]
76:     l_foot_index = mp[31]
77:     r_foot_index = mp[32]
78: 
79:     pelvis = _midpoint(l_hip, r_hip)
80:     neck = _midpoint(l_shoulder, r_shoulder)
81:     head = (mp[0] + mp[7] + mp[8]) / 3.0
82: 
83:     # Spine interpolation between pelvis and neck.
84:     spine1 = pelvis * 0.75 + neck * 0.25
85:     spine2 = pelvis * 0.50 + neck * 0.50
86:     spine3 = pelvis * 0.25 + neck * 0.75
87: 
88:     # Left/right collar approximated between neck and shoulder.
89:     l_collar = _midpoint(neck, l_shoulder)
90:     r_collar = _midpoint(neck, r_shoulder)
91: 
92:     # Hand centroids and fingertips.
93:     l_hand = (l_pinky + l_index + l_thumb) / 3.0
94:     r_hand = (r_pinky + r_index + r_thumb) / 3.0
95:     l_fingertip = l_index
96:     r_fingertip = r_index
97: 
98:     # Fill ANUBIS-32 output layout.
99:     out[0] = pelvis
100:     out[1] = spine1
101:     out[2] = spine2
102:     out[3] = spine3
103:     out[4] = neck
104:     out[5] = head
105: 
106:     out[6] = l_hip
107:     out[7] = l_knee
108:     out[8] = l_ankle
109:     out[9] = l_heel
110:     out[10] = l_foot_index
111: 
112:     out[11] = r_hip
113:     out[12] = r_knee
114:     out[13] = r_ankle
115:     out[14] = r_heel
116:     out[15] = r_foot_index
117: 
118:     out[16] = l_collar
119:     out[17] = l_shoulder
120:     out[18] = l_elbow
121:     out[19] = l_wrist
122:     out[20] = l_hand
123:     out[21] = l_fingertip
124: 
125:     out[22] = r_collar
126:     out[23] = r_shoulder
127:     out[24] = r_elbow
128:     out[25] = r_wrist
129:     out[26] = r_hand
130:     out[27] = r_fingertip
131: 
132:     # ANUBIS has toe-base and thumb-toe nodes; MediaPipe has one foot-index point.
133:     out[28] = l_foot_index
134:     out[29] = r_foot_index
135:     out[30] = l_foot_index
136:     out[31] = r_foot_index
137: 
138:     return out
139: 
140: 
141: class PoseExtractor:
142:     """
143:     Lightweight MediaPipe Pose wrapper for frame-wise extraction.
144: 
145:     Usage
146:     -----
147:     with PoseExtractor() as extractor:
148:         joints_32 = extractor.extract_from_frame(frame_bgr)
149:     """
150: 
151:     def __init__(
152:         self,
153:         min_detection_confidence: float = 0.5,
154:         min_tracking_confidence: float = 0.5,
155:         model_complexity: int = 1,
156:     ) -> None:
157:         self.min_detection_confidence = float(min_detection_confidence)
158:         self.min_tracking_confidence = float(min_tracking_confidence)
159:         self.model_complexity = int(model_complexity)
160: 
161:         self._mp_pose: Optional[Any] = None
162:         self._pose: Optional[Any] = None
163: 
164:     def __enter__(self) -> "PoseExtractor":
165:         try:
166:             import mediapipe as mp
167:         except ImportError as exc:
168:             raise ImportError(
169:                 "mediapipe is not installed. Install dependencies and retry."
170:             ) from exc
171: 
172:         self._mp_pose = mp.solutions.pose
173:         self._pose = self._mp_pose.Pose(
174:             static_image_mode=False,
175:             model_complexity=self.model_complexity,
176:             smooth_landmarks=True,
177:             min_detection_confidence=self.min_detection_confidence,
178:             min_tracking_confidence=self.min_tracking_confidence,
179:         )
180:         return self
181: 
182:     def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
183:         if self._pose is not None:
184:             self._pose.close()
185:         self._pose = None
186:         self._mp_pose = None
187: 
188:     def extract_from_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
189:         """
190:         Extract one ANUBIS-32 skeleton from a BGR frame.
191: 
192:         Parameters
193:         ----------
194:         frame_bgr : np.ndarray  shape (H, W, 3)
195: 
196:         Returns
197:         -------
198:         joints_32 : np.ndarray  shape (32, 3), optional
199:             None if no pose is detected in this frame.
200:         """
201:         if self._pose is None:
202:             raise RuntimeError(
203:                 "PoseExtractor is not initialised. Use it as a context manager."
204:             )
205: 
206:         frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
207:         results = self._pose.process(frame_rgb)
208:         if results.pose_landmarks is None:
209:             return None
210: 
211:         keypoints_33 = np.array(
212:             [
213:                 [lm.x, lm.y, lm.z, lm.visibility]
214:                 for lm in results.pose_landmarks.landmark
215:             ],
216:             dtype=np.float32,
217:         )
218:         return mediapipe33_to_anubis32(keypoints_33)
219: 
220: 
221: if __name__ == "__main__":
222:     cap = cv2.VideoCapture(0)
223:     if not cap.isOpened():
224:         print("Could not open webcam (index 0).")
225:     else:
226:         with PoseExtractor() as extractor:
227:             ok, frame = cap.read()
228:             if ok:
229:                 joints = extractor.extract_from_frame(frame)
230:                 if joints is not None:
231:                     print(f"Extracted shape: {joints.shape}")
232:                 else:
233:                     print("No pose detected on first frame.")
234:     cap.release()

(End of file - total 234 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/module1/pretrain/live_inference.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/module1/pretrain/live_inference.py</path>
<type>file</type>
<content>1: # module1/pretrain/live_inference.py
2: #
3: # Stage 1 — Live ANUBIS Motion Inference
4: # ──────────────────────────────────────
5: # Runs webcam/video/RTSP live inference using the pretrained ActionClassifier
6: # backbone. This is an action-understanding test pipeline for the pretrained
7: # model (102 classes), not the football injury-risk head.
8: 
9: from __future__ import annotations
10: 
11: import argparse
12: import collections
13: import os
14: import time
15: from typing import Deque, List, Optional, Tuple
16: 
17: import cv2
18: import numpy as np
19: import torch
20: import torch.nn.functional as F
21: 
22: from module1.data.joint_config import EDGE_INDEX
23: from module1.pose.extractor import PoseExtractor
24: from module1.pretrain.live_utils import (
25:     build_feature_window,
26:     load_class_names,
27:     parse_video_source,
28:     smooth_probs,
29: )
30: from module1.pretrain.model import ActionClassifier, build_model
31: from utils.config import load_config
32: from utils.logger import get_logger
33: 
34: log = get_logger(__name__)
35: 
36: 
37: def _draw_overlay(
38:     frame: np.ndarray,
39:     class_names: List[str],
40:     probs: torch.Tensor,
41:     topk: int,
42:     fps: float,
43:     frames_ready: int,
44:     seq_len: int,
45: ) -> np.ndarray:
46:     """Draw prediction text overlay onto a frame."""
47:     canvas = frame.copy()
48:     H, W = canvas.shape[:2]
49: 
50:     # Background panel
51:     panel_h = 30 + (topk + 3) * 24
52:     cv2.rectangle(canvas, (10, 10), (min(W - 10, 680), 10 + panel_h), (0, 0, 0), -1)
53:     cv2.rectangle(canvas, (10, 10), (min(W - 10, 680), 10 + panel_h), (70, 70, 70), 1)
54: 
55:     y = 34
56:     cv2.putText(
57:         canvas,
58:         "Live ANUBIS Motion Inference (Pretraining Model)",
59:         (20, y),
60:         cv2.FONT_HERSHEY_SIMPLEX,
61:         0.6,
62:         (230, 230, 230),
63:         1,
64:         cv2.LINE_AA,
65:     )
66: 
67:     y += 26
68:     cv2.putText(
69:         canvas,
70:         f"FPS: {fps:.1f} | Window: {frames_ready}/{seq_len}",
71:         (20, y),
72:         cv2.FONT_HERSHEY_SIMPLEX,
73:         0.55,
74:         (170, 255, 170),
75:         1,
76:         cv2.LINE_AA,
77:     )
78: 
79:     topv, topi = torch.topk(probs, k=min(topk, probs.numel()))
80:     for rank, (score, idx) in enumerate(zip(topv.tolist(), topi.tolist()), start=1):
81:         y += 24
82:         label = class_names[idx] if idx < len(class_names) else f"class_{idx}"
83:         text = f"{rank}. [{idx:03d}] {label}  prob={score:.3f}"
84:         cv2.putText(
85:             canvas,
86:             text,
87:             (20, y),
88:             cv2.FONT_HERSHEY_SIMPLEX,
89:             0.55,
90:             (255, 255, 255),
91:             1,
92:             cv2.LINE_AA,
93:         )
94: 
95:     return canvas
96: 
97: 
98: def _load_checkpoint(
99:     ckpt_path: str,
100:     classifier: ActionClassifier,
101:     device: torch.device,
102: ) -> None:
103:     if not os.path.exists(ckpt_path):
104:         raise FileNotFoundError(
105:             f"Checkpoint not found: {ckpt_path}\n"
106:             "Run pretraining first or provide --ckpt path."
107:         )
108: 
109:     ckpt = torch.load(ckpt_path, map_location=device)
110:     if "classifier" in ckpt:
111:         classifier.load_state_dict(ckpt["classifier"])
112:     else:
113:         # Fallback if a raw state_dict is provided.
114:         classifier.load_state_dict(ckpt)
115: 
116:     log.info(
117:         "Loaded checkpoint from %s (epoch=%s val_top1=%s)",
118:         ckpt_path,
119:         str(ckpt.get("epoch", "n/a")) if isinstance(ckpt, dict) else "n/a",
120:         (
121:             f"{ckpt.get('val_top1', float('nan')):.3f}"
122:             if isinstance(ckpt, dict) and "val_top1" in ckpt
123:             else "n/a"
124:         ),
125:     )
126: 
127: 
128: def run_live(
129:     config_path: str,
130:     ckpt_path: str,
131:     source: str,
132:     topk: int,
133:     smooth_alpha: float,
134:     label_map: Optional[str],
135:     show: bool,
136: ) -> None:
137:     """
138:     Run live action inference from a camera/video stream.
139: 
140:     Parameters
141:     ----------
142:     config_path : str
143:     ckpt_path : str
144:     source : str
145:         Camera index (e.g. "0"), video path, or RTSP URL.
146:     topk : int
147:         Number of top predictions shown in the overlay.
148:     smooth_alpha : float
149:         EMA alpha for probability smoothing (0=no smoothing, close to 1=strong).
150:     label_map : str, optional
151:         Optional class-name map file (.txt/.yaml/.json).
152:     show : bool
153:         Whether to display OpenCV window. Disable for headless testing.
154:     """
155:     cfg = load_config(config_path)
156:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
157:     log.info("Device: %s", device)
158: 
159:     edge_index = EDGE_INDEX.to(device)
160:     _, classifier = build_model(cfg, edge_index)
161:     classifier = classifier.to(device)
162:     classifier.eval()
163:     _load_checkpoint(ckpt_path, classifier, device)
164: 
165:     num_classes = int(cfg["data"]["num_classes"])
166:     seq_len = int(cfg["data"]["seq_len"])
167:     class_names = load_class_names(num_classes, label_map)
168: 
169:     parsed_source = parse_video_source(source)
170:     cap = cv2.VideoCapture(parsed_source)
171:     if not cap.isOpened():
172:         raise RuntimeError(
173:             f"Could not open source: {source}\nCheck camera index/path/URL and retry."
174:         )
175: 
176:     log.info(
177:         "Starting live inference from source=%s | seq_len=%d | topk=%d",
178:         str(source),
179:         seq_len,
180:         topk,
181:     )
182: 
183:     frame_buffer: Deque[np.ndarray] = collections.deque(maxlen=seq_len)
184:     smoothed_probs: Optional[torch.Tensor] = None
185:     fps = 0.0
186:     prev_time = time.time()
187: 
188:     with PoseExtractor() as extractor:
189:         try:
190:             while True:
191:                 ok, frame = cap.read()
192:                 if not ok:
193:                     log.info("Stream ended or frame read failed.")
194:                     break
195: 
196:                 joints32 = extractor.extract_from_frame(frame)
197:                 if joints32 is not None:
198:                     frame_buffer.append(joints32)
199: 
200:                 # Default distribution before warm-up.
201:                 probs = torch.full(
202:                     (num_classes,), 1.0 / num_classes, dtype=torch.float32
203:                 )
204: 
205:                 if len(frame_buffer) == seq_len:
206:                     window = np.stack(frame_buffer, axis=0)  # (T, J, 3)
207:                     features = build_feature_window(window, pelvis_idx=0)  # (T, J, 9)
208: 
209:                     with torch.no_grad():
210:                         x = features.unsqueeze(0).to(device)  # (1, T, J, 9)
211:                         logits = classifier(x).squeeze(0)
212:                         current_probs = F.softmax(logits, dim=0).cpu()
213: 
214:                     probs = smooth_probs(
215:                         smoothed_probs, current_probs, alpha=smooth_alpha
216:                     )
217:                     smoothed_probs = probs
218: 
219:                 # FPS update
220:                 now = time.time()
221:                 delta = max(now - prev_time, 1e-6)
222:                 fps = 0.9 * fps + 0.1 * (1.0 / delta)
223:                 prev_time = now
224: 
225:                 if show:
226:                     overlay = _draw_overlay(
227:                         frame=frame,
228:                         class_names=class_names,
229:                         probs=probs,
230:                         topk=topk,
231:                         fps=fps,
232:                         frames_ready=len(frame_buffer),
233:                         seq_len=seq_len,
234:                     )
235:                     cv2.imshow("Football Injury AI — Live Motion Test", overlay)
236:                     key = cv2.waitKey(1) & 0xFF
237:                     if key in (27, ord("q")):
238:                         break
239: 
240:         finally:
241:             cap.release()
242:             if show:
243:                 cv2.destroyAllWindows()
244: 
245: 
246: def main() -> None:
247:     parser = argparse.ArgumentParser(
248:         description="Run live ANUBIS action inference from webcam/video stream"
249:     )
250:     parser.add_argument(
251:         "--config",
252:         default="configs/pretrain_final.yaml",
253:         help="Path to pretrain config YAML",
254:     )
255:     parser.add_argument(
256:         "--ckpt",
257:         default="outputs/checkpoints/pretrain/pretrain_best.pt",
258:         help="Path to classifier checkpoint",
259:     )
260:     parser.add_argument(
261:         "--source",
262:         default="0",
263:         help="Video source: webcam index, file path, or RTSP URL",
264:     )
265:     parser.add_argument(
266:         "--topk",
267:         type=int,
268:         default=5,
269:         help="Number of top classes shown",
270:     )
271:     parser.add_argument(
272:         "--smooth-alpha",
273:         type=float,
274:         default=0.7,
275:         help="EMA alpha for probability smoothing in [0,1)",
276:     )
277:     parser.add_argument(
278:         "--label-map",
279:         default=None,
280:         help="Optional class-name map (.txt/.yaml/.json)",
281:     )
282:     parser.add_argument(
283:         "--no-show",
284:         action="store_true",
285:         help="Disable OpenCV preview window (headless)",
286:     )
287:     args = parser.parse_args()
288: 
289:     run_live(
290:         config_path=args.config,
291:         ckpt_path=args.ckpt,
292:         source=args.source,
293:         topk=max(1, args.topk),
294:         smooth_alpha=args.smooth_alpha,
295:         label_map=args.label_map,
296:         show=not args.no_show,
297:     )
298: 
299: 
300: if __name__ == "__main__":
301:     main()

(End of file - total 301 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/tests/test_live_pipeline.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/tests/test_live_pipeline.py</path>
<type>file</type>
<content>1: # tests/test_live_pipeline.py
2: #
3: # Unit tests for module1/pretrain/live_utils.py and module1/pose/extractor.py
4: # ─────────────────────────────────────────────────────────────────────────────
5: # Covers: source parsing, feature-window construction (pos+vel+acc), label-map
6: # loading, EMA smoothing, and MediaPipe33->ANUBIS32 joint conversion.
7: #
8: # Run with: pytest tests/test_live_pipeline.py -v
9: 
10: from __future__ import annotations
11: 
12: import json
13: from typing import List
14: 
15: import numpy as np
16: import pytest
17: import torch
18: import yaml
19: 
20: from module1.pose.extractor import mediapipe33_to_anubis32
21: from module1.pretrain.live_utils import (
22:     build_feature_window,
23:     compute_acceleration,
24:     compute_velocity,
25:     load_class_names,
26:     normalise_pelvis_center,
27:     parse_video_source,
28:     smooth_probs,
29: )
30: 
31: 
32: class TestParseVideoSource:
33:     def test_numeric_source_returns_int(self) -> None:
34:         assert parse_video_source("0") == 0
35:         assert parse_video_source("12") == 12
36: 
37:     def test_path_or_url_returns_str(self) -> None:
38:         assert parse_video_source("data/samples/clip.mp4") == "data/samples/clip.mp4"
39:         assert parse_video_source("rtsp://cam/live") == "rtsp://cam/live"
40: 
41: 
42: class TestFeatureConstruction:
43:     def test_normalise_pelvis_center(self) -> None:
44:         x = np.zeros((3, 4, 3), dtype=np.float32)
45:         x[:, 0, :] = np.array([2.0, -1.0, 0.5], dtype=np.float32)
46:         x[:, 1, :] = np.array([3.0, 2.0, 0.5], dtype=np.float32)
47: 
48:         out = normalise_pelvis_center(x, pelvis_idx=0)
49:         assert np.allclose(out[:, 0, :], 0.0)
50:         assert np.allclose(out[:, 1, :], np.array([1.0, 3.0, 0.0], dtype=np.float32))
51: 
52:     def test_velocity_and_acceleration_boundary_handling(self) -> None:
53:         pos = np.zeros((4, 1, 3), dtype=np.float32)
54:         pos[:, 0, 0] = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32)
55: 
56:         vel = compute_velocity(pos)
57:         acc = compute_acceleration(vel)
58: 
59:         # Forward differences on x channel
60:         assert np.allclose(
61:             vel[:, 0, 0], np.array([1.0, 2.0, 3.0, 3.0], dtype=np.float32)
62:         )
63:         assert np.allclose(
64:             acc[:, 0, 0], np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
65:         )
66: 
67:     def test_build_feature_window_shape_and_order(self) -> None:
68:         T = 5
69:         J = 3
70:         pos = np.zeros((T, J, 3), dtype=np.float32)
71:         # Simple motion for joint 1 on x axis; pelvis is joint 0.
72:         pos[:, 0, :] = np.array([10.0, 0.0, 0.0], dtype=np.float32)
73:         pos[:, 1, 0] = np.arange(T, dtype=np.float32)
74: 
75:         feat = build_feature_window(pos, pelvis_idx=0)
76:         assert feat.shape == (T, J, 9)
77: 
78:         # Channels 0:3 are centred positions. Pelvis must be zeros.
79:         assert torch.allclose(feat[:, 0, 0:3], torch.zeros((T, 3), dtype=torch.float32))
80:         # Channels 3:6 are velocity. Joint-1 x velocity is 1 except last duplicated.
81:         assert torch.allclose(
82:             feat[:, 1, 3], torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32)
83:         )
84: 
85: 
86: class TestClassNames:
87:     def test_default_fallback(self) -> None:
88:         names = load_class_names(num_classes=4, label_map_path=None)
89:         assert names == ["class_0", "class_1", "class_2", "class_3"]
90: 
91:     def test_load_from_txt(self, tmp_path: pytest.TempPathFactory) -> None:
92:         p = tmp_path / "labels.txt"
93:         p.write_text("walk\nrun\njump\n", encoding="utf-8")
94:         names = load_class_names(num_classes=5, label_map_path=str(p))
95:         assert names[:3] == ["walk", "run", "jump"]
96:         assert names[4] == "class_4"
97: 
98:     def test_load_from_yaml_dict(self, tmp_path: pytest.TempPathFactory) -> None:
99:         p = tmp_path / "labels.yaml"
100:         payload = {0: "walk", 2: "kick"}
101:         p.write_text(yaml.safe_dump(payload), encoding="utf-8")
102:         names = load_class_names(num_classes=4, label_map_path=str(p))
103:         assert names == ["walk", "class_1", "kick", "class_3"]
104: 
105:     def test_load_from_json_list(self, tmp_path: pytest.TempPathFactory) -> None:
106:         p = tmp_path / "labels.json"
107:         payload: List[str] = ["a", "b"]
108:         p.write_text(json.dumps(payload), encoding="utf-8")
109:         names = load_class_names(num_classes=3, label_map_path=str(p))
110:         assert names == ["a", "b", "class_2"]
111: 
112: 
113: class TestSmoothProbs:
114:     def test_first_step_returns_current(self) -> None:
115:         cur = torch.tensor([0.2, 0.8])
116:         out = smooth_probs(None, cur, alpha=0.7)
117:         assert torch.allclose(out, cur)
118: 
119:     def test_ema(self) -> None:
120:         prev = torch.tensor([0.5, 0.5])
121:         cur = torch.tensor([0.1, 0.9])
122:         out = smooth_probs(prev, cur, alpha=0.5)
123:         assert torch.allclose(out, torch.tensor([0.3, 0.7]))
124: 
125: 
126: class TestMediapipeConversion:
127:     def test_output_shape(self) -> None:
128:         mp = np.zeros((33, 4), dtype=np.float32)
129:         out = mediapipe33_to_anubis32(mp)
130:         assert out.shape == (32, 3)
131: 
132:     def test_pelvis_is_midpoint_of_hips(self) -> None:
133:         mp = np.zeros((33, 4), dtype=np.float32)
134:         mp[23, :3] = np.array([2.0, 0.0, 0.0], dtype=np.float32)  # left hip
135:         mp[24, :3] = np.array([4.0, 2.0, 0.0], dtype=np.float32)  # right hip
136:         out = mediapipe33_to_anubis32(mp)
137:         assert np.allclose(out[0], np.array([3.0, 1.0, 0.0], dtype=np.float32))
138: 
139:     def test_direct_joint_mappings(self) -> None:
140:         mp = np.zeros((33, 4), dtype=np.float32)
141:         mp[23, :3] = np.array([1, 2, 3], dtype=np.float32)  # L_Hip
142:         mp[24, :3] = np.array([4, 5, 6], dtype=np.float32)  # R_Hip
143:         mp[25, :3] = np.array([7, 8, 9], dtype=np.float32)  # L_Knee
144:         mp[26, :3] = np.array([10, 11, 12], dtype=np.float32)  # R_Knee
145:         out = mediapipe33_to_anubis32(mp)
146:         assert np.allclose(out[6], mp[23, :3])
147:         assert np.allclose(out[11], mp[24, :3])
148:         assert np.allclose(out[7], mp[25, :3])
149:         assert np.allclose(out[12], mp[26, :3])
150: 
151:     def test_invalid_shape_raises(self) -> None:
152:         bad = np.zeros((32, 3), dtype=np.float32)
153:         with pytest.raises(ValueError):
154:             mediapipe33_to_anubis32(bad)

(End of file - total 154 lines)
</content>
```

---

## Assistant (Build · gpt-5.3-codex · 21.7s)

_Thinking:_

**Enforcing typing style and ASCII compliance**

**Tool: bash**

**Input:**
```json
{
  "command": "python - <<'PY'\nfrom pathlib import Path\nfiles=[\n'./module1/pretrain/live_utils.py',\n'./module1/pose/extractor.py',\n'./module1/pretrain/live_inference.py',\n'./tests/test_live_pipeline.py',\n]\nfor f in files:\n    text=Path(f).read_text(encoding='utf-8')\n    bad=[(i+1,ch) for i,ch in enumerate(text) if ord(ch)>127]\n    if bad:\n        print(f, 'non-ascii count', len(bad))\n    else:\n        print(f, 'ascii-only')\nPY",
  "description": "Check new files for non-ASCII characters"
}
```

**Output:**
```
./module1/pretrain/live_utils.py non-ascii count 40
./module1/pose/extractor.py non-ascii count 58
./module1/pretrain/live_inference.py non-ascii count 40
./tests/test_live_pipeline.py non-ascii count 77

```

---

## Assistant (Build · gpt-5.3-codex · 13.7s)

_Thinking:_

**Planning Unicode to ASCII cleanup**

**Tool: bash**

**Input:**
```json
{
  "command": "python - <<'PY'\nfrom pathlib import Path\nfiles=[\n'/home/alan/mydev/college/football-injury-ai/module1/pretrain/live_utils.py',\n'/home/alan/mydev/college/football-injury-ai/module1/pose/extractor.py',\n'/home/alan/mydev/college/football-injury-ai/module1/pretrain/live_inference.py',\n'/home/alan/mydev/college/football-injury-ai/tests/test_live_pipeline.py',\n]\nfor f in files:\n    text=Path(f).read_text(encoding='utf-8')\n    chars=sorted({ch for ch in text if ord(ch)>127})\n    print('\\n',f)\n    print('unique:',[(ch,ord(ch)) for ch in chars])\nPY",
  "description": "List unique non-ASCII characters in files"
}
```

**Output:**
```

 /home/alan/mydev/college/football-injury-ai/module1/pretrain/live_utils.py
unique: [('—', 8212), ('•', 8226), ('─', 9472)]

 /home/alan/mydev/college/football-injury-ai/module1/pose/extractor.py
unique: [('—', 8212), ('→', 8594), ('─', 9472)]

 /home/alan/mydev/college/football-injury-ai/module1/pretrain/live_inference.py
unique: [('—', 8212), ('─', 9472)]

 /home/alan/mydev/college/football-injury-ai/tests/test_live_pipeline.py
unique: [('─', 9472)]

```

---

## Assistant (Build · gpt-5.3-codex · 19.7s)

_Thinking:_

**Replacing non-ASCII characters in comments**

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Update File: /home/alan/mydev/college/football-injury-ai/module1/pretrain/live_utils.py\n@@\n-# Stage 1 — Live Inference Utilities\n-# ───────────────────────────────────\n+# Stage 1 - Live Inference Utilities\n+# ----------------------------------\n # Helper functions shared by the live ANUBIS inference pipeline:\n-#   • class-label loading\n-#   • pelvis-centred normalisation\n-#   • velocity / acceleration feature construction\n-#   • probability smoothing\n+#   - class-label loading\n+#   - pelvis-centred normalisation\n+#   - velocity / acceleration feature construction\n+#   - probability smoothing\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
M module1/pretrain/live_utils.py

LSP errors detected in module1/pretrain/live_utils.py, please fix:
<diagnostics file="/home/alan/mydev/college/football-injury-ai/module1/pretrain/live_utils.py">
ERROR [18:8] Import "torch" could not be resolved
</diagnostics>
```

---

## Assistant (Build · gpt-5.3-codex · 12.2s)

_Thinking:_

**Acknowledging environment LSP issue**

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Update File: /home/alan/mydev/college/football-injury-ai/module1/pose/extractor.py\n@@\n-# Stage 1 — Live Pose Extraction (MediaPipe → ANUBIS-32)\n-# ────────────────────────────────────────────────────────\n+# Stage 1 - Live Pose Extraction (MediaPipe -> ANUBIS-32)\n+# --------------------------------------------------------\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
M module1/pose/extractor.py

LSP errors detected in module1/pose/extractor.py, please fix:
<diagnostics file="/home/alan/mydev/college/football-injury-ai/module1/pose/extractor.py">
ERROR [12:8] Import "cv2" could not be resolved
ERROR [166:20] Import "mediapipe" could not be resolved
ERROR [173:36] "Pose" is not a known attribute of "None"
</diagnostics>
```

---

## Assistant (Build · gpt-5.3-codex · 16.0s)

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Update File: /home/alan/mydev/college/football-injury-ai/module1/pretrain/live_inference.py\n@@\n-# Stage 1 — Live ANUBIS Motion Inference\n-# ──────────────────────────────────────\n+# Stage 1 - Live ANUBIS Motion Inference\n+# --------------------------------------\n@@\n-                    cv2.imshow(\"Football Injury AI — Live Motion Test\", overlay)\n+                    cv2.imshow(\"Football Injury AI - Live Motion Test\", overlay)\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
M module1/pretrain/live_inference.py

LSP errors detected in module1/pretrain/live_inference.py, please fix:
<diagnostics file="/home/alan/mydev/college/football-injury-ai/module1/pretrain/live_inference.py">
ERROR [17:8] Import "cv2" could not be resolved
ERROR [19:8] Import "torch" could not be resolved
ERROR [20:8] Import "torch.nn.functional" could not be resolved
</diagnostics>
```

---

## Assistant (Build · gpt-5.3-codex · 11.8s)

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Update File: /home/alan/mydev/college/football-injury-ai/tests/test_live_pipeline.py\n@@\n-# ─────────────────────────────────────────────────────────────────────────────\n+# -----------------------------------------------------------------------------\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
M tests/test_live_pipeline.py

LSP errors detected in tests/test_live_pipeline.py, please fix:
<diagnostics file="/home/alan/mydev/college/football-injury-ai/tests/test_live_pipeline.py">
ERROR [16:8] Import "pytest" could not be resolved
ERROR [17:8] Import "torch" could not be resolved
</diagnostics>
```

---

## Assistant (Build · gpt-5.3-codex · 9.4s)

_Thinking:_

**Checking ASCII compliance in docstrings**

**Tool: bash**

**Input:**
```json
{
  "command": "python - <<'PY'\nfrom pathlib import Path\nfiles=[\n'./module1/pretrain/live_utils.py',\n'./module1/pose/extractor.py',\n'./module1/pretrain/live_inference.py',\n'./tests/test_live_pipeline.py',\n]\nfor f in files:\n    text=Path(f).read_text(encoding='utf-8')\n    bad=sorted({ch for ch in text if ord(ch)>127})\n    print(f, 'non-ascii:' ,[(ch,ord(ch)) for ch in bad])\nPY",
  "description": "Recheck files for non-ASCII characters"
}
```

**Output:**
```
./module1/pretrain/live_utils.py non-ascii: []
./module1/pose/extractor.py non-ascii: []
./module1/pretrain/live_inference.py non-ascii: []
./tests/test_live_pipeline.py non-ascii: []

```

---

## Assistant (Build · gpt-5.3-codex · 24.9s)

_Thinking:_

**Updating command docs**

**Tool: bash**

**Input:**
```json
{
  "command": "pytest tests/test_live_pipeline.py -v",
  "description": "Run live pipeline unit tests"
}
```

**Output:**
```
zsh:1: command not found: pytest

```

---

## Assistant (Build · gpt-5.3-codex · 21.9s)

_Thinking:_

**Running tests with pytest**

**Tool: bash**

**Input:**
```json
{
  "command": "uv run pytest tests/test_live_pipeline.py -v",
  "description": "Run live pipeline tests via uv"
}
```

**Output:**
```
Using CPython 3.12.10
Creating virtual environment at: .venv
   Building football-injury-ai @ file:///home/alan/mydev/college/football-injury-ai
Downloading sympy (6.0MiB)
Downloading pygments (1.2MiB)
Downloading nvidia-cufile-cu12 (1.1MiB)
Downloading nvidia-cuda-cupti-cu12 (9.8MiB)
Downloading opencv-python (69.5MiB)
Downloading kiwisolver (1.4MiB)
Downloading notebook (13.9MiB)
Downloading networkx (2.0MiB)
Downloading nvidia-cusparse-cu12 (274.9MiB)
Downloading nvidia-cublas-cu12 (566.8MiB)
Downloading matplotlib (8.3MiB)
Downloading torch (873.3MiB)
Downloading nvidia-nccl-cu12 (307.4MiB)
Downloading nvidia-cudnn-cu12 (674.0MiB)
Downloading mediapipe (9.9MiB)
Downloading pandas (10.4MiB)
Downloading nvidia-cuda-nvrtc-cu12 (84.0MiB)
Downloading torch-geometric (1.2MiB)
Downloading fonttools (4.7MiB)
Downloading jupyterlab (11.9MiB)
Downloading debugpy (4.1MiB)
Downloading sqlalchemy (3.2MiB)
Downloading scikit-learn (8.5MiB)
Downloading triton (179.5MiB)
Downloading widgetsnbextension (2.1MiB)
Downloading jedi (1.5MiB)
Downloading nvidia-curand-cu12 (60.7MiB)
Downloading cuda-bindings (11.6MiB)
Downloading aiohttp (1.7MiB)
Downloading nvidia-nvjitlink-cu12 (37.4MiB)
Downloading nvidia-nvshmem-cu12 (132.7MiB)
Downloading babel (9.7MiB)
Downloading scipy (33.6MiB)
Downloading nvidia-cusolver-cu12 (255.1MiB)
Downloading opencv-contrib-python (75.5MiB)
Downloading nvidia-cufft-cu12 (184.2MiB)
Downloading numpy (15.8MiB)
Downloading torchvision (7.7MiB)
Downloading nvidia-cusparselt-cu12 (273.9MiB)
Downloading pillow (6.7MiB)
 Downloaded pygments
 Downloaded nvidia-cufile-cu12
 Downloaded torch-geometric
 Downloaded kiwisolver
 Downloaded jedi
 Downloaded aiohttp
 Downloaded networkx
 Downloaded widgetsnbextension
 Downloaded sqlalchemy
 Downloaded debugpy
 Downloaded fonttools
 Downloaded sympy
 Downloaded pillow
 Downloaded torchvision
  × Failed to build `football-injury-ai @
  │ file:///home/alan/mydev/college/football-injury-ai`
  ├─▶ The build backend returned an error
  ╰─▶ Call to `hatchling.build.build_editable` failed (exit status: 1)

      [stderr]
      Traceback (most recent call last):
        File "<string>", line 11, in <module>
        File
      "/home/alan/.cache/uv/builds-v0/.tmp10lcBj/lib/python3.12/site-packages/hatchling/build.py",
      line 83, in build_editable
          return os.path.basename(next(builder.build(directory=wheel_directory,
      versions=["editable"])))
      
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File
      "/home/alan/.cache/uv/builds-v0/.tmp10lcBj/lib/python3.12/site-packages/hatchling/builders/plugin/interface.py",
      line 157, in build
          artifact = version_api[version](directory, **build_data)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File
      "/home/alan/.cache/uv/builds-v0/.tmp10lcBj/lib/python3.12/site-packages/hatchling/builders/wheel.py",
      line 524, in build_editable
          return self.build_editable_detection(directory, **build_data)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File
      "/home/alan/.cache/uv/builds-v0/.tmp10lcBj/lib/python3.12/site-packages/hatchling/builders/wheel.py",
      line 536, in build_editable_detection
          for included_file in self.recurse_selected_project_files():
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File
      "/home/alan/.cache/uv/builds-v0/.tmp10lcBj/lib/python3.12/site-packages/hatchling/builders/plugin/interface.py",
      line 182, in recurse_selected_project_files
          if self.config.only_include:
             ^^^^^^^^^^^^^^^^^^^^^^^^
        File
      "/home/alan/.local/share/uv/python/cpython-3.12.10-linux-x86_64-gnu/lib/python3.12/functools.py",
      line 998, in __get__
          val = self.func(instance)
                ^^^^^^^^^^^^^^^^^^^
        File
      "/home/alan/.cache/uv/builds-v0/.tmp10lcBj/lib/python3.12/site-packages/hatchling/builders/config.py",
      line 715, in only_include
          only_include = only_include_config.get("only-include",
      self.default_only_include()) or self.packages
      
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File
      "/home/alan/.cache/uv/builds-v0/.tmp10lcBj/lib/python3.12/site-packages/hatchling/builders/wheel.py",
      line 270, in default_only_include
          return self.default_file_selection_options.only_include
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File
      "/home/alan/.local/share/uv/python/cpython-3.12.10-linux-x86_64-gnu/lib/python3.12/functools.py",
      line 998, in __get__
          val = self.func(instance)
                ^^^^^^^^^^^^^^^^^^^
        File
      "/home/alan/.cache/uv/builds-v0/.tmp10lcBj/lib/python3.12/site-packages/hatchling/builders/wheel.py",
      line 258, in default_file_selection_options
          raise ValueError(message)
      ValueError: Unable to determine which files to ship
      inside the wheel using the following heuristics:
      https://hatch.pypa.io/latest/plugins/builder/wheel/#default-file-selection

      The most likely cause of this is that there is no directory that matches
      the name of your project (football_injury_ai).

      At least one file selection option must be defined
      in the `tool.hatch.build.targets.wheel` table, see:
      https://hatch.pypa.io/latest/config/build/

      As an example, if you intend to ship a directory named `foo` that
      resides within a `src` directory located at the root of your project,
      you can define the following:

      [tool.hatch.build.targets.wheel]
      packages = ["src/foo"]

      hint: This usually indicates a problem with the package or the build
      environment.

```

---

## Assistant (Build · gpt-5.3-codex · 7.7s)

_Thinking:_

**Exploring test run strategies despite uv build failure**

**Tool: bash**

**Input:**
```json
{
  "command": "python - <<'PY'\nimport importlib\nmods=['torch','cv2','numpy','yaml']\nfor m in mods:\n    try:\n        importlib.import_module(m)\n        print(m,'ok')\n    except Exception as e:\n        print(m,'err',type(e).__name__,e)\nPY",
  "description": "Check required Python modules availability"
}
```

**Output:**
```
torch err ModuleNotFoundError No module named 'torch'
cv2 err ModuleNotFoundError No module named 'cv2'
numpy ok
yaml ok

```

---

## Assistant (Build · gpt-5.3-codex · 25.7s)

_Thinking:_

**Planning docs updates and style fixes**

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Update File: /home/alan/mydev/college/football-injury-ai/tests/test_live_pipeline.py\n@@\n import json\n+from pathlib import Path\n from typing import List\n@@\n-    def test_load_from_txt(self, tmp_path: pytest.TempPathFactory) -> None:\n+    def test_load_from_txt(self, tmp_path: Path) -> None:\n@@\n-    def test_load_from_yaml_dict(self, tmp_path: pytest.TempPathFactory) -> None:\n+    def test_load_from_yaml_dict(self, tmp_path: Path) -> None:\n@@\n-    def test_load_from_json_list(self, tmp_path: pytest.TempPathFactory) -> None:\n+    def test_load_from_json_list(self, tmp_path: Path) -> None:\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
M tests/test_live_pipeline.py

LSP errors detected in tests/test_live_pipeline.py, please fix:
<diagnostics file="/home/alan/mydev/college/football-injury-ai/tests/test_live_pipeline.py">
ERROR [17:8] Import "pytest" could not be resolved
ERROR [18:8] Import "torch" could not be resolved
</diagnostics>
```

---

## Assistant (Build · gpt-5.3-codex · 34.2s)

_Thinking:_

**Planning robust checkpoint loading and docs update**

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Update File: /home/alan/mydev/college/football-injury-ai/module1/pretrain/live_inference.py\n@@\n import argparse\n import collections\n import os\n import time\n-from typing import Deque, List, Optional, Tuple\n+from typing import Deque, List, Optional\n@@\n ) -> np.ndarray:\n     \"\"\"Draw prediction text overlay onto a frame.\"\"\"\n     canvas = frame.copy()\n-    H, W = canvas.shape[:2]\n+    _, W = canvas.shape[:2]\n@@\n def _load_checkpoint(\n     ckpt_path: str,\n     classifier: ActionClassifier,\n     device: torch.device,\n ) -> None:\n@@\n-    ckpt = torch.load(ckpt_path, map_location=device)\n-    if \"classifier\" in ckpt:\n-        classifier.load_state_dict(ckpt[\"classifier\"])\n-    else:\n-        # Fallback if a raw state_dict is provided.\n-        classifier.load_state_dict(ckpt)\n+    ckpt = torch.load(ckpt_path, map_location=device)\n+\n+    state_dict = None\n+    if isinstance(ckpt, dict):\n+        if \"classifier\" in ckpt:\n+            state_dict = ckpt[\"classifier\"]\n+        elif \"state_dict\" in ckpt:\n+            state_dict = ckpt[\"state_dict\"]\n+        else:\n+            values = list(ckpt.values())\n+            if values and all(torch.is_tensor(v) for v in values):\n+                state_dict = ckpt\n+    elif hasattr(ckpt, \"keys\"):\n+        state_dict = ckpt\n+\n+    if state_dict is None:\n+        raise ValueError(\n+            \"Checkpoint format not recognised. Expected either a full training \"\n+            \"checkpoint with key 'classifier' or a classifier state_dict.\"\n+        )\n+\n+    classifier.load_state_dict(state_dict)\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
M module1/pretrain/live_inference.py

LSP errors detected in module1/pretrain/live_inference.py, please fix:
<diagnostics file="/home/alan/mydev/college/football-injury-ai/module1/pretrain/live_inference.py">
ERROR [17:8] Import "cv2" could not be resolved
ERROR [19:8] Import "torch" could not be resolved
ERROR [20:8] Import "torch.nn.functional" could not be resolved
</diagnostics>
```

---

## Assistant (Build · gpt-5.3-codex · 12.1s)

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Update File: /home/alan/mydev/college/football-injury-ai/docs/COMMANDS.md\n@@\n ## Evaluation\n@@\n ```bash\n # Basic evaluation\n uv run module1/pretrain/evaluate.py --config configs/pretrain.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt\n@@\n uv run module1/pretrain/evaluate.py --config configs/pretrain.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --tsne\n ```\n+\n+### Evaluate with Final Pretraining Config\n+\n+```bash\n+uv run module1/pretrain/evaluate.py --config configs/pretrain_final.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --split val --tsne\n+```\n@@\n ## Testing\n+\n+### Live Inference (Webcam / Video / RTSP)\n+\n+This tests the pretrained ANUBIS action model on live footage.\n+\n+```bash\n+# Webcam (index 0)\n+uv run module1/pretrain/live_inference.py --config configs/pretrain_final.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --source 0\n+\n+# Video file\n+uv run module1/pretrain/live_inference.py --config configs/pretrain_final.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --source data/samples/clip.mp4\n+\n+# RTSP stream\n+uv run module1/pretrain/live_inference.py --config configs/pretrain_final.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --source \"rtsp://<user>:<pass>@<ip>:554/stream\"\n+```\n+\n+Optional flags:\n+\n+```bash\n+# Show top-3 classes only\n+uv run module1/pretrain/live_inference.py --source 0 --topk 3\n+\n+# Stronger prediction smoothing\n+uv run module1/pretrain/live_inference.py --source 0 --smooth-alpha 0.85\n+\n+# Use custom class names (.txt/.yaml/.json)\n+uv run module1/pretrain/live_inference.py --source 0 --label-map outputs/logs/anubis_class_names.txt\n+\n+# Headless mode (no OpenCV window)\n+uv run module1/pretrain/live_inference.py --source data/samples/clip.mp4 --no-show\n+```\n+\n+Exit live mode with `q` or `Esc`.\n@@\n ```bash\n uv run pytest tests/test_model.py -v\n uv run pytest tests/test_anubis_loader.py -v\n uv run pytest tests/test_joint_config.py -v\n+uv run pytest tests/test_live_pipeline.py -v\n ```\n@@\n ├── module1/\n │   ├── data/              # Data loading & preprocessing\n+│   ├── pose/              # Live pose extraction (MediaPipe -> ANUBIS-32)\n │   ├── pretrain/          # Training & evaluation scripts\n-│   └── pose/              # Pose extraction (future)\n+│   │   ├── live_inference.py\n+│   │   └── live_utils.py\n+│   └── ...\n@@\n ### \"No module named 'torch'\"\n@@\n uv sync\n ```\n+\n+### \"No module named 'cv2'\" or \"No module named 'mediapipe'\"\n+\n+```bash\n+uv sync\n+```\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
M docs/COMMANDS.md
```

---

## Assistant (Build · gpt-5.3-codex · 6.4s)

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Update File: /home/alan/mydev/college/football-injury-ai/README.md\n@@\n ## Common Commands\n \n ```bash\n # Train ANUBIS pretraining\n uv run module1/pretrain/train.py --config configs/pretrain.yaml\n+\n+# Train final pretraining config (official val split)\n+uv run module1/pretrain/train.py --config configs/pretrain_final.yaml\n \n # Evaluate\n uv run module1/pretrain/evaluate.py --config configs/pretrain.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt\n+\n+# Live webcam/video action test\n+uv run module1/pretrain/live_inference.py --config configs/pretrain_final.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --source 0\n \n # Run Optuna tuning\n uv run module1/pretrain/tune_optuna.py --config configs/pretrain.yaml --n-trials 30\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
M README.md
```

---

## Assistant (Build · gpt-5.3-codex · 5.8s)

_Thinking:_

**Preparing README and syntax checks**

**Tool: bash**

**Input:**
```json
{
  "command": "python -m py_compile module1/pretrain/live_utils.py module1/pose/extractor.py module1/pretrain/live_inference.py tests/test_live_pipeline.py",
  "description": "Check syntax of new Python files"
}
```

---

## Assistant (Build · gpt-5.3-codex · 2.7s)

**Tool: bash**

**Input:**
```json
{
  "command": "git status",
  "description": "Check working tree status"
}
```

**Output:**
```
On branch master
Your branch is up to date with 'origin/master'.

Changes not staged for commit:
  (use "git add/rm <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	deleted:    LABELING_GUIDE.md
	modified:   README.md
	modified:   docs/COMMANDS.md
	modified:   module1/pretrain/train.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	module1/pose/extractor.py
	module1/pretrain/live_inference.py
	module1/pretrain/live_utils.py
	session-ses_33bc.md
	tests/test_live_pipeline.py
	uv.lock

no changes added to commit (use "git add" and/or "git commit -a")

```

---

## Assistant (Build · gpt-5.3-codex · 39.2s)

_Thinking:_

**Summarizing implementation and docs update**

**Tool: bash**

**Input:**
```json
{
  "command": "git diff -- module1/pretrain/live_utils.py module1/pose/extractor.py module1/pretrain/live_inference.py tests/test_live_pipeline.py docs/COMMANDS.md README.md",
  "description": "Show diffs for implemented pipeline files"
}
```

**Output:**
```
diff --git a/README.md b/README.md
index 1205703..61c35e4 100644
--- a/README.md
+++ b/README.md
@@ -70,9 +70,15 @@ uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.
 # Train ANUBIS pretraining
 uv run module1/pretrain/train.py --config configs/pretrain.yaml
 
+# Train final pretraining config (official val split)
+uv run module1/pretrain/train.py --config configs/pretrain_final.yaml
+
 # Evaluate
 uv run module1/pretrain/evaluate.py --config configs/pretrain.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt
 
+# Live webcam/video action test
+uv run module1/pretrain/live_inference.py --config configs/pretrain_final.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --source 0
+
 # Run Optuna tuning
 uv run module1/pretrain/tune_optuna.py --config configs/pretrain.yaml --n-trials 30
 
diff --git a/docs/COMMANDS.md b/docs/COMMANDS.md
index 4444ef4..4bbb522 100644
--- a/docs/COMMANDS.md
+++ b/docs/COMMANDS.md
@@ -95,6 +95,12 @@ uv run module1/pretrain/evaluate.py --config configs/pretrain.yaml --ckpt output
 uv run module1/pretrain/evaluate.py --config configs/pretrain.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --tsne
 ```
 
+### Evaluate with Final Pretraining Config
+
+```bash
+uv run module1/pretrain/evaluate.py --config configs/pretrain_final.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --split val --tsne
+```
+
 ### Checkpoint Locations
 
 | Checkpoint | Path |
@@ -107,6 +113,39 @@ uv run module1/pretrain/evaluate.py --config configs/pretrain.yaml --ckpt output
 
 ## Testing
 
+### Live Inference (Webcam / Video / RTSP)
+
+This tests the pretrained ANUBIS action model on live footage.
+
+```bash
+# Webcam (index 0)
+uv run module1/pretrain/live_inference.py --config configs/pretrain_final.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --source 0
+
+# Video file
+uv run module1/pretrain/live_inference.py --config configs/pretrain_final.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --source data/samples/clip.mp4
+
+# RTSP stream
+uv run module1/pretrain/live_inference.py --config configs/pretrain_final.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --source "rtsp://<user>:<pass>@<ip>:554/stream"
+```
+
+Optional flags:
+
+```bash
+# Show top-3 classes only
+uv run module1/pretrain/live_inference.py --source 0 --topk 3
+
+# Stronger prediction smoothing
+uv run module1/pretrain/live_inference.py --source 0 --smooth-alpha 0.85
+
+# Use custom class names (.txt/.yaml/.json)
+uv run module1/pretrain/live_inference.py --source 0 --label-map outputs/logs/anubis_class_names.txt
+
+# Headless mode (no OpenCV window)
+uv run module1/pretrain/live_inference.py --source data/samples/clip.mp4 --no-show
+```
+
+Exit live mode with `q` or `Esc`.
+
 ### Run All Tests
 
 ```bash
@@ -119,6 +158,7 @@ uv run pytest tests/
 uv run pytest tests/test_model.py -v
 uv run pytest tests/test_anubis_loader.py -v
 uv run pytest tests/test_joint_config.py -v
+uv run pytest tests/test_live_pipeline.py -v
 ```
 
 ### Run Single Test
@@ -146,8 +186,11 @@ uv run module1/pretrain/model.py
 football-injury-ai/
 ├── module1/
 │   ├── data/              # Data loading & preprocessing
+│   ├── pose/              # Live pose extraction (MediaPipe -> ANUBIS-32)
 │   ├── pretrain/          # Training & evaluation scripts
-│   └── pose/              # Pose extraction (future)
+│   │   ├── live_inference.py
+│   │   └── live_utils.py
+│   └── ...
 ├── configs/
 │   ├── pretrain.yaml      # ANUBIS pretraining config
 │   └── module1.yaml       # Fine-tuning config
@@ -195,6 +238,12 @@ rm outputs/checkpoints/pretrain/pretrain_ep*.pt
 uv sync
 ```
 
+### "No module named 'cv2'" or "No module named 'mediapipe'"
+
+```bash
+uv sync
+```
+
 ### "CUDA out of memory"
 
 Reduce batch size in config:

```

**Tool: bash**

**Input:**
```json
{
  "command": "git status --short",
  "description": "Show concise git status"
}
```

**Output:**
```
 D LABELING_GUIDE.md
 M README.md
 M docs/COMMANDS.md
 M module1/pretrain/train.py
?? module1/pose/extractor.py
?? module1/pretrain/live_inference.py
?? module1/pretrain/live_utils.py
?? session-ses_33bc.md
?? tests/test_live_pipeline.py
?? uv.lock

```

---

## Assistant (Build · gpt-5.3-codex · 37.8s)

_Thinking:_

**Cleaning untracked lock file**

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Delete File: /home/alan/mydev/college/football-injury-ai/uv.lock\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
D uv.lock
```

---

## Assistant (Build · gpt-5.3-codex · 40.0s)

_Thinking:_

**Validating repository cleanup and code style**

**Tool: bash**

**Input:**
```json
{
  "command": "git status --short",
  "description": "Check concise status after implementing pipeline"
}
```

**Output:**
```
 D LABELING_GUIDE.md
 M README.md
 M docs/COMMANDS.md
 M module1/pretrain/train.py
?? module1/pose/extractor.py
?? module1/pretrain/live_inference.py
?? module1/pretrain/live_utils.py
?? session-ses_33bc.md
?? tests/test_live_pipeline.py

```

---

## Assistant (Build · gpt-5.3-codex · 22.9s)

_Thinking:_

**Delivering implementation and docs update**

Implemented code and tests per plan:  
- **Pose extraction:** Added context manager and keypoint mapping in `module1/pose/extractor.py`.  
- **Live utilities:** Parsing, feature building, smoothing functions in `module1/pretrain/live_utils.py`.  
- **Live inference pipeline:** Streaming, checkpoint/config loading, sliding window, top-k labels overlay in `module1/pretrain/live_inference.py`.  
- **Tests:** Added in `tests/test_live_pipeline.py`.  
- **Documentation:** Updated `docs/COMMANDS.md` and `README.md` with commands and usage instructions.

Pipeline runs a pretrained ANUBIS action model (not injury risk). Dependencies include torch, cv2, and mediapipe.

Local environment lacks these deps, so full tests could not run; `py_compile` verified no syntax errors. The command to try running (with required environment) is:  
```
uv run python module1/pretrain/live_inference.py --help
```

No modifications were made to unrelated dirty files such as `module1/pretrain/train.py` or `LABELING_GUIDE.md`. The untracked `uv.lock` was deleted but this has no effect.

If desired, next steps can include committing and pushing these changes.

Please advise if committing/pushing is requested or if further actions are needed.

Implemented — the live testing pipeline is now in place and docs are updated.

- Added `module1/pose/extractor.py` with a `PoseExtractor` context manager and `mediapipe33_to_anubis32()` conversion so live MediaPipe output matches your 32-joint model layout.
- Added `module1/pretrain/live_utils.py` for shared preprocessing: source parsing, pelvis centering, velocity/acceleration construction, class-name loading (`.txt/.yaml/.json`), and EMA probability smoothing.
- Added `module1/pretrain/live_inference.py` to run webcam/video/RTSP inference with sliding-window features (`(T,J,9)`), checkpoint loading, top-k overlay, FPS display, smoothing, and `q`/`Esc` exit.
- Added tests in `tests/test_live_pipeline.py` for conversion + preprocessing + label-map utilities.
- Updated docs in `docs/COMMANDS.md` and `README.md` with live inference and final-pretrain commands.

Run commands:
```bash
uv run module1/pretrain/live_inference.py --config configs/pretrain_final.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --source 0
```

```bash
uv run module1/pretrain/live_inference.py --config configs/pretrain_final.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --source data/samples/clip.mp4
```

```bash
uv run module1/pretrain/live_inference.py --source 0 --topk 3 --smooth-alpha 0.85
```

Notes:
- This is an **action-understanding live test** for the pretrained ANUBIS model, not injury-risk scoring yet.
- I validated syntax with `python -m py_compile` for all new files.
- I could not run full pytest in this environment because `torch/cv2/mediapipe` are not installed locally.

If you want, I can now commit and push only these new live-pipeline/doc changes (excluding unrelated dirty files already in your working tree).

---

## User

now the accuracy droped to 0.650 also got the top-1: 0.2722,top-5: 0.5859,macro f1:0.2530, this is the result I got after the pretraining with the new config, keep in mind that the training was done on a different and more powerfull computer and not this one

---

## Assistant (Build · gpt-5.3-codex · 49.8s)

_Thinking:_

**Interpreting metrics and advising next steps**

This is actually a **solid result**, not a real regression.

- Your old official-val baseline was around `Top-1 0.2686`, `Top-5 0.5553`, `Macro-F1 0.2569`.
- New run is `Top-1 0.2722`, `Top-5 0.5859`, `Macro-F1 0.2530`.
- So on the metric that matters (official val split), **Top-1 and Top-5 improved**; Macro-F1 is basically flat.

About the `0.650` drop: that is likely train accuracy. With stronger augmentation (`speed_perturb` on), lower train acc is expected and often healthier (less memorization).

For your goal (“understand basic human motions”): this checkpoint is good enough to keep as your pretrained backbone.

**Do this now**
1. Save this checkpoint as a fixed baseline:
   - `outputs/checkpoints/pretrain/pretrain_best_realval_v1.pt`
2. Use it in the live pipeline and validate qualitatively on real clips.
3. If you want one cheap extra boost (no Optuna), run one short ablation:
   - same config, but `speed_perturb: false`, 40-50 epochs, compare Top-1 only.

If you share that one ablation result, I can tell you which checkpoint to lock as final.

---

## User

in the base paper that we are considering they have used gan for noise reduction and it is turned off during the inference, are we using that, what is current architecture that we are following

---

## Assistant (Build · gpt-5.3-codex · 7.9s)

_Thinking:_

**Confirming GAN absence**

**Tool: grep**

**Input:**
```json
{
  "pattern": "GAN|gan|generator|discriminator",
  "path": "/home/alan/mydev/college/football-injury-ai/module1"
}
```

**Output:**
```
No files found
```

**Tool: grep**

**Input:**
```json
{
  "pattern": "SpatioTemporalEncoder|GATConv|TransformerEncoderLayer|ActionClassifier|build_feature_window",
  "path": "/home/alan/mydev/college/football-injury-ai/module1"
}
```

**Output:**
```
Found 51 matches
/home/alan/mydev/college/football-injury-ai/module1/pretrain/live_inference.py:
  Line 5: # Runs webcam/video/RTSP live inference using the pretrained ActionClassifier
  Line 25:     build_feature_window,
  Line 30: from module1.pretrain.model import ActionClassifier, build_model
  Line 100:     classifier: ActionClassifier,
  Line 223:                     features = build_feature_window(window, pelvis_idx=0)  # (T, J, 9)

/home/alan/mydev/college/football-injury-ai/module1/pretrain/live_utils.py:
  Line 119: def build_feature_window(
  Line 259:     feat = build_feature_window(demo)

/home/alan/mydev/college/football-injury-ai/module1/pretrain/train.py:
  Line 5: # Trains ActionClassifier end-to-end on ANUBIS 102-class action recognition.
  Line 30: from module1.pretrain.model import ActionClassifier, SpatioTemporalEncoder, build_model
  Line 65:     model: ActionClassifier,
  Line 77:     model     : ActionClassifier
  Line 118:     model: ActionClassifier,
  Line 128:     model     : ActionClassifier
  Line 176:     encoder: SpatioTemporalEncoder,
  Line 177:     classifier: ActionClassifier,
  Line 320:         description="Pretrain SpatioTemporalEncoder on ANUBIS"

/home/alan/mydev/college/football-injury-ai/module1/pretrain/tune_optuna.py:
  Line 6: # ActionClassifier training loop.

/home/alan/mydev/college/football-injury-ai/module1/pretrain/evaluate.py:
  Line 5: # Loads a pretrained ActionClassifier checkpoint and computes:
  Line 39: from module1.pretrain.model import ActionClassifier, build_model
  Line 51:     model: ActionClassifier,
  Line 111:     model: ActionClassifier,
  Line 193:     ax.set_title("t-SNE of SpatioTemporalEncoder CLS embeddings (ANUBIS)")
  Line 211:     classifier: ActionClassifier,
  Line 220:     classifier : ActionClassifier  (must be already constructed)
  Line 293:         description="Evaluate a pretrained SpatioTemporalEncoder on ANUBIS"

/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py:
  Line 3: # Stage 1 — SpatioTemporalEncoder + ActionClassifier
  Line 10: #     GATConv( F → 16, heads=4, concat=True)  → 64-dim per node + BN + ELU
  Line 11: #     GATConv(64 → 16, heads=4, concat=True)  → 64-dim per node + BN + ELU
  Line 12: #     GATConv(64 → 64, heads=1, concat=False) → 64-dim per node + BN + ELU
  Line 20: #     2 × TransformerEncoderLayer (pre-norm, nhead=4, ff=256)
  Line 24: #   ActionClassifier wraps SpatioTemporalEncoder with a linear head for
  Line 36: from torch_geometric.nn import GATConv, global_mean_pool
  Line 84: # ── SpatioTemporalEncoder ─────────────────────────────────────────────────────
  Line 87: class SpatioTemporalEncoder(nn.Module):
  Line 105:     num_layers   : int   number of TransformerEncoderLayer blocks
  Line 140:         self.gat1 = GATConv(
  Line 143:         self.gat2 = GATConv(
  Line 146:         self.gat3 = GATConv(
  Line 163:         encoder_layer = nn.TransformerEncoderLayer(
  Line 272: # ── ActionClassifier ──────────────────────────────────────────────────────────
  Line 275: class ActionClassifier(nn.Module):
  Line 277:     Full pretraining model: SpatioTemporalEncoder + linear classification head.
  Line 284:     encoder     : SpatioTemporalEncoder
  Line 288:     def __init__(self, encoder: SpatioTemporalEncoder, num_classes: int) -> None:
  Line 312: ) -> Tuple[SpatioTemporalEncoder, ActionClassifier]:
  Line 323:     encoder    : SpatioTemporalEncoder
  Line 324:     classifier : ActionClassifier
  Line 328:     encoder = SpatioTemporalEncoder(
  Line 343:     classifier = ActionClassifier(encoder, num_classes=d["num_classes"])
  Line 357:     encoder = SpatioTemporalEncoder(
  Line 364:     classifier = ActionClassifier(encoder, num_classes=102).to(device)
```

---

## Assistant (Build · gpt-5.3-codex · 17.4s)

_Thinking:_

**Confirming no GAN usage**

Short answer: **No, we are not using a GAN denoiser right now**.

- I checked the codebase and there is no `GAN`/generator/discriminator module in `module1/` (no adversarial training path).
- Your current pipeline is a **direct spatiotemporal classifier**.

Current architecture we are following:

- **Input features**: `(T, J, 9)` per sample, where 9 = `[x,y,z,vx,vy,vz,ax,ay,az]` built in preprocessing (`module1/data/anubis_loader.py:187`, `module1/pretrain/live_utils.py:119`).
- **Spatial encoder**: 3 GAT layers + BN + ELU + dropout over skeleton graphs (`module1/pretrain/model.py:140`).
- **Temporal encoder**: Transformer encoder (pre-norm) with CLS token + sinusoidal positional encoding (`module1/pretrain/model.py:157`, `module1/pretrain/model.py:163`).
- **Head**: Linear classification head for ANUBIS 102 classes (`module1/pretrain/model.py:291`).
- **Training-time robustness**: data augmentation (flip, jitter, optional speed perturb), not GAN denoising (`module1/data/anubis_loader.py:107`, `configs/pretrain_final.yaml`).
- **Inference**: model in eval mode, no train augmentations; live pipeline only does optional probability smoothing (EMA), not denoising GAN (`module1/pretrain/live_inference.py:214`).

So compared to the base paper: we matched the spatiotemporal modeling idea, but **not** the GAN noise-reduction block.

---

## User

can we swap out sinusodial with rotary encoding, also replacing normal atention mechanis with latent attention and adding a GAN for denoising what is your openion

---

## Assistant (Plan · gpt-5.3-codex · 25.9s)

_Thinking:_

**Providing architecture upgrade recommendations**

- **Verdict:** Swapping to rotary embeddings, latent attention, and GAN denoising is technically feasible but not advised all at once during current pretraining due to complexity and risk of confounding effects.

- **Component assessments:**  
  1) Rotary Positional Embeddings: Moderate complexity; potential benefit for longer sequences but sequence length 60 is small, so gains may be limited.  
  2) Latent Attention: Ambiguous term; requires custom architecture; highest complexity and implementation risk. Clarification recommended before proceeding.  
  3) GAN Denoising: High complexity and training instability; needs paired noisy/clean data or reliable synthetic noise; better handled post-pretraining with deterministic filters as first step.

- **Recommended phased plan:**  
  1) Keep current architecture baseline.  
  2) Introduce simple denoising preprocessing (e.g., temporal smoothing, outlier clipping).  
  3) Run ablation study adding rotary embeddings.  
  4) Explore latent attention if sequence length substantially increases (>120) and term clarification provided.  
  5) Consider GAN denoising only with sufficient noisy data for adversarial training.

- **Experiment design:** Use small-scale tests with clear stop criteria based on improvement over baseline (e.g., top-1 accuracy gain from 27%). Avoid simultaneous multiple architecture changes during pretraining.

- **Clarifying question:** Which specific latent attention mechanism is intended (e.g., linear attention, Perceiver-style bottleneck, or DeepSeek MLA)? This impacts feasibility and approach.

Great question. My opinion: **yes, all three are possible, but don’t do them together now**.

- **RoPE instead of sinusoidal**: reasonable to try; low-to-medium risk, but likely modest gain at `seq_len=60`.
- **Latent attention**: possible, but high ambiguity/high risk (could mean Perceiver latents, linear attention, MLA, etc.); bigger refactor than RoPE.
- **GAN denoising**: highest risk/complexity; adversarial training is unstable and needs a clear noisy→clean supervision strategy (or very careful synthetic-noise setup).

For your current stage (ANUBIS pretraining), I’d prioritize like this:

1. **Keep core model stable as baseline** (your current checkpoint is already valid).
2. **Try RoPE ablation first** (single change, compare Top-1/Top-5/Macro-F1).
3. **Before GAN, add deterministic denoising in live pipeline** (Kalman/OneEuro/Savitzky-Golay + confidence gating). This usually gives better ROI for noisy pose streams.
4. **Only then evaluate latent attention** if you plan longer clips or lower latency.
5. **GAN last**, and only if real live-noise is the bottleneck after step 3.

Why: if you add all 3 now, you won’t know what helped or hurt, and training/debug cost explodes.

If you want, next I can give you a clean experiment plan (A/B matrix) with exact run order and stop criteria.  
One clarification I’ll need for implementation later: when you say **latent attention**, do you mean **Perceiver-style latent bottlenecks** (recommended default) or another specific variant?

---

## User

yes I was reffering to Perceiver-style latent bottlenecks can you implement this and also implement rotary encoding

---

## Assistant (Plan · gpt-5.3-codex · 7.9s)

_Thinking:_

**Planning integration strategy**

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py</path>
<type>file</type>
<content>1: # module1/pretrain/model.py
2: #
3: # Stage 1 — SpatioTemporalEncoder + ActionClassifier
4: # ────────────────────────────────────────────────────
5: # End-to-end model that encodes a sequence of skeleton graphs:
6: #
7: #   Input : (B, T, J, F)  — batch, frames, joints, features (pos+vel+acc = 9)
8: #
9: #   Spatial stage — time-distributed GAT (shared weights across all T frames):
10: #     GATConv( F → 16, heads=4, concat=True)  → 64-dim per node + BN + ELU
11: #     GATConv(64 → 16, heads=4, concat=True)  → 64-dim per node + BN + ELU
12: #     GATConv(64 → 64, heads=1, concat=False) → 64-dim per node + BN + ELU
13: #     global_mean_pool over joints             → (B*T, 64)
14: #     reshape                                  → (B, T, 64)
15: #
16: #   Temporal stage — Transformer encoder:
17: #     input_proj Linear(64 → d_model)
18: #     prepend CLS token
19: #     sinusoidal positional encoding
20: #     2 × TransformerEncoderLayer (pre-norm, nhead=4, ff=256)
21: #     LayerNorm
22: #     extract CLS output                       → (B, d_model)
23: #
24: #   ActionClassifier wraps SpatioTemporalEncoder with a linear head for
25: #   102-class ANUBIS pretraining.  After pretraining, save only the encoder
26: #   (encoder.state_dict()) and attach new fine-tuning heads.
27: 
28: from __future__ import annotations
29: 
30: import math
31: from typing import Optional, Tuple
32: 
33: import torch
34: import torch.nn as nn
35: import torch.nn.functional as F
36: from torch_geometric.nn import GATConv, global_mean_pool
37: 
38: from utils.logger import get_logger
39: 
40: log = get_logger(__name__)
41: 
42: 
43: # ── Positional Encoding ────────────────────────────────────────────────────────
44: 
45: 
46: class PositionalEncoding(nn.Module):
47:     """
48:     Sinusoidal positional encoding (Vaswani et al., 2017).
49: 
50:     Parameters
51:     ----------
52:     d_model  : int   embedding dimension
53:     max_len  : int   maximum sequence length (including CLS token)
54:     dropout  : float dropout applied after adding the encoding
55:     """
56: 
57:     def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1) -> None:
58:         super().__init__()
59:         self.dropout = nn.Dropout(p=dropout)
60: 
61:         position = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
62:         div_term = torch.exp(
63:             torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
64:         )
65:         pe = torch.zeros(1, max_len, d_model)
66:         pe[0, :, 0::2] = torch.sin(position * div_term)
67:         pe[0, :, 1::2] = torch.cos(position * div_term)
68:         self.register_buffer("pe", pe)
69: 
70:     def forward(self, x: torch.Tensor) -> torch.Tensor:
71:         """
72:         Parameters
73:         ----------
74:         x : torch.Tensor  shape (B, T, d_model)
75: 
76:         Returns
77:         -------
78:         out : torch.Tensor  shape (B, T, d_model)
79:         """
80:         x = x + self.pe[:, : x.size(1), :]
81:         return self.dropout(x)
82: 
83: 
84: # ── SpatioTemporalEncoder ─────────────────────────────────────────────────────
85: 
86: 
87: class SpatioTemporalEncoder(nn.Module):
88:     """
89:     End-to-end spatiotemporal encoder: time-distributed GAT + Transformer.
90: 
91:     The encoder is the shared backbone used across all training stages.
92:     It outputs a single CLS embedding per sequence.
93: 
94:     Parameters
95:     ----------
96:     num_joints   : int   number of skeleton joints (32 for ANUBIS)
97:     in_features  : int   feature dimension per joint (9: pos+vel+acc)
98:     edge_index   : torch.Tensor  shape (2, E)  skeleton connectivity;
99:                    registered as a buffer so it moves to device automatically
100:     gat_hidden   : int   output dim of every GAT block (default 64)
101:     gat_heads    : int   attention heads for GAT layers 1 & 2 (default 4)
102:     gat_dropout  : float dropout on GAT attention coefficients
103:     d_model      : int   transformer embedding dimension (default 128)
104:     nhead        : int   transformer attention heads
105:     num_layers   : int   number of TransformerEncoderLayer blocks
106:     dim_ff       : int   feedforward expansion inside transformer
107:     dropout      : float transformer dropout
108:     seq_len      : int   expected sequence length (used to size pos. encoding)
109:     cls_std      : float trunc_normal std for CLS-token initialisation
110:     """
111: 
112:     def __init__(
113:         self,
114:         num_joints: int,
115:         in_features: int,
116:         edge_index: torch.Tensor,
117:         gat_hidden: int = 64,
118:         gat_heads: int = 4,
119:         gat_dropout: float = 0.1,
120:         d_model: int = 128,
121:         nhead: int = 4,
122:         num_layers: int = 2,
123:         dim_ff: int = 256,
124:         dropout: float = 0.1,
125:         seq_len: int = 60,
126:         cls_std: float = 0.02,
127:     ) -> None:
128:         super().__init__()
129: 
130:         self.num_joints = num_joints
131:         self.gat_hidden = gat_hidden
132:         self.gat_dropout = gat_dropout
133:         self.d_model = d_model
134: 
135:         # Register skeleton topology as a non-learnable buffer
136:         self.register_buffer("edge_index", edge_index)  # (2, E)
137: 
138:         # ── GAT layers (shared across all time steps) ──────────────────────
139:         per_head_dim = gat_hidden // gat_heads  # 16
140:         self.gat1 = GATConv(
141:             in_features, per_head_dim, heads=gat_heads, concat=True, dropout=gat_dropout
142:         )
143:         self.gat2 = GATConv(
144:             gat_hidden, per_head_dim, heads=gat_heads, concat=True, dropout=gat_dropout
145:         )
146:         self.gat3 = GATConv(
147:             gat_hidden, gat_hidden, heads=1, concat=False, dropout=gat_dropout
148:         )
149: 
150:         self.bn1 = nn.BatchNorm1d(gat_hidden)
151:         self.bn2 = nn.BatchNorm1d(gat_hidden)
152:         self.bn3 = nn.BatchNorm1d(gat_hidden)
153: 
154:         # ── Transformer ────────────────────────────────────────────────────
155:         self.input_proj = nn.Linear(gat_hidden, d_model)
156: 
157:         self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
158:         nn.init.trunc_normal_(self.cls_token, std=cls_std)
159: 
160:         # max_len = seq_len + 1 to accommodate the CLS token
161:         self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)
162: 
163:         encoder_layer = nn.TransformerEncoderLayer(
164:             d_model=d_model,
165:             nhead=nhead,
166:             dim_feedforward=dim_ff,
167:             dropout=dropout,
168:             batch_first=True,
169:             norm_first=True,  # pre-norm variant — more stable gradients
170:             activation="gelu",
171:         )
172:         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
173:         self.norm = nn.LayerNorm(d_model)
174: 
175:     # ── Internal helpers ───────────────────────────────────────────────────────
176: 
177:     def _gat_forward(self, x: torch.Tensor) -> torch.Tensor:
178:         """
179:         Process all T frames through the GAT in a single batched pass.
180: 
181:         Each (frame, sequence) pair is treated as an independent graph.
182:         All B*T graphs share the same edge topology (self.edge_index).
183: 
184:         Parameters
185:         ----------
186:         x : torch.Tensor  shape (B, T, J, F)
187: 
188:         Returns
189:         -------
190:         spatial : torch.Tensor  shape (B, T, gat_hidden)
191:         """
192:         B, T, J, Fin = x.shape
193:         num_graphs = B * T
194: 
195:         # Flatten nodes: (B*T*J, Fin)
196:         x_nodes = x.reshape(num_graphs * J, Fin)
197: 
198:         # Build batched edge index by offsetting each graph's node indices by J
199:         # edge_index: (2, E)  →  broadcast to (2, num_graphs, E)
200:         E = self.edge_index.shape[1]
201:         offsets = torch.arange(num_graphs, device=x.device) * J  # (num_graphs,)
202:         edge_batch = self.edge_index.unsqueeze(1).expand(
203:             -1, num_graphs, -1
204:         )  # (2, G, E)
205:         edge_batch = edge_batch + offsets.view(1, -1, 1)  # (2, G, E)
206:         edge_flat = edge_batch.reshape(2, num_graphs * E)  # (2, G*E)
207: 
208:         # batch_tensor: maps each node to its graph index
209:         batch_tensor = torch.arange(num_graphs, device=x.device).repeat_interleave(J)
210: 
211:         # ── GAT layer 1 ────────────────────────────────────────────────────
212:         h = self.gat1(x_nodes, edge_flat)  # (B*T*J, gat_hidden)
213:         h = self.bn1(h)
214:         h = F.elu(h)
215:         h = F.dropout(h, p=self.gat_dropout, training=self.training)
216: 
217:         # ── GAT layer 2 ────────────────────────────────────────────────────
218:         h = self.gat2(h, edge_flat)
219:         h = self.bn2(h)
220:         h = F.elu(h)
221:         h = F.dropout(h, p=self.gat_dropout, training=self.training)
222: 
223:         # ── GAT layer 3 ────────────────────────────────────────────────────
224:         h = self.gat3(h, edge_flat)
225:         h = self.bn3(h)
226:         h = F.elu(h)
227: 
228:         # Global mean pool: (B*T*J, gat_hidden) → (B*T, gat_hidden)
229:         spatial = global_mean_pool(h, batch_tensor)
230: 
231:         return spatial.reshape(B, T, self.gat_hidden)  # (B, T, gat_hidden)
232: 
233:     # ── Forward pass ──────────────────────────────────────────────────────────
234: 
235:     def forward(self, x: torch.Tensor) -> torch.Tensor:
236:         """
237:         Encode a batch of skeleton sequences.
238: 
239:         Parameters
240:         ----------
241:         x : torch.Tensor  shape (B, T, J, F)
242:             Joint features: pos + vel + acc (F = 9).
243: 
244:         Returns
245:         -------
246:         cls_embedding : torch.Tensor  shape (B, d_model)
247:             Sequence-level representation extracted from the CLS token.
248:         """
249:         B = x.shape[0]
250: 
251:         # Stage 1: spatial encoding (time-distributed GAT)
252:         spatial = self._gat_forward(x)  # (B, T, gat_hidden)
253: 
254:         # Stage 2: temporal encoding
255:         h = self.input_proj(spatial)  # (B, T, d_model)
256: 
257:         # Prepend CLS token
258:         cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
259:         h = torch.cat([cls, h], dim=1)  # (B, T+1, d_model)
260: 
261:         # Positional encoding
262:         h = self.pos_enc(h)  # (B, T+1, d_model)
263: 
264:         # Transformer encoder
265:         h = self.transformer(h)  # (B, T+1, d_model)
266:         h = self.norm(h)
267: 
268:         # Extract CLS token (position 0)
269:         return h[:, 0, :]  # (B, d_model)
270: 
271: 
272: # ── ActionClassifier ──────────────────────────────────────────────────────────
273: 
274: 
275: class ActionClassifier(nn.Module):
276:     """
277:     Full pretraining model: SpatioTemporalEncoder + linear classification head.
278: 
279:     Used only for Stage 1 (ANUBIS action classification pretraining).
280:     After pretraining, save encoder.state_dict() and discard the head.
281: 
282:     Parameters
283:     ----------
284:     encoder     : SpatioTemporalEncoder
285:     num_classes : int   number of action classes (102 for ANUBIS)
286:     """
287: 
288:     def __init__(self, encoder: SpatioTemporalEncoder, num_classes: int) -> None:
289:         super().__init__()
290:         self.encoder = encoder
291:         self.head = nn.Linear(encoder.d_model, num_classes)
292: 
293:     def forward(self, x: torch.Tensor) -> torch.Tensor:
294:         """
295:         Parameters
296:         ----------
297:         x : torch.Tensor  shape (B, T, J, F)
298: 
299:         Returns
300:         -------
301:         logits : torch.Tensor  shape (B, num_classes)
302:         """
303:         cls_emb = self.encoder(x)  # (B, d_model)
304:         return self.head(cls_emb)  # (B, num_classes)
305: 
306: 
307: # ── Factory helper ────────────────────────────────────────────────────────────
308: 
309: 
310: def build_model(
311:     cfg: dict, edge_index: torch.Tensor
312: ) -> Tuple[SpatioTemporalEncoder, ActionClassifier]:
313:     """
314:     Construct encoder and full classifier from config dict.
315: 
316:     Parameters
317:     ----------
318:     cfg        : dict               full config loaded from configs/pretrain.yaml
319:     edge_index : torch.Tensor (2,E) skeleton edge index
320: 
321:     Returns
322:     -------
323:     encoder    : SpatioTemporalEncoder
324:     classifier : ActionClassifier
325:     """
326:     m = cfg["model"]
327:     d = cfg["data"]
328:     encoder = SpatioTemporalEncoder(
329:         num_joints=d["num_joints"],
330:         in_features=m["in_features"],
331:         edge_index=edge_index,
332:         gat_hidden=m["gat_hidden_dim"],
333:         gat_heads=m["gat_heads"],
334:         gat_dropout=m["gat_dropout"],
335:         d_model=m["d_model"],
336:         nhead=m["nhead"],
337:         num_layers=m["num_transformer_layers"],
338:         dim_ff=m["dim_feedforward"],
339:         dropout=m["transformer_dropout"],
340:         seq_len=d["seq_len"],
341:         cls_std=m["cls_std"],
342:     )
343:     classifier = ActionClassifier(encoder, num_classes=d["num_classes"])
344:     total = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
345:     log.info("Model built: %d trainable parameters", total)
346:     return encoder, classifier
347: 
348: 
349: # ── Quick sanity check ────────────────────────────────────────────────────────
350: 
351: if __name__ == "__main__":
352:     from module1.data.joint_config import EDGE_INDEX, NUM_JOINTS
353: 
354:     B, T, J, F_IN = 4, 60, NUM_JOINTS, 9
355:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
356: 
357:     encoder = SpatioTemporalEncoder(
358:         num_joints=J,
359:         in_features=F_IN,
360:         edge_index=EDGE_INDEX,
361:         seq_len=T,
362:     ).to(device)
363: 
364:     classifier = ActionClassifier(encoder, num_classes=102).to(device)
365:     classifier.eval()
366: 
367:     dummy = torch.randn(B, T, J, F_IN, device=device)
368: 
369:     with torch.no_grad():
370:         cls_emb = encoder(dummy)
371:         logits = classifier(dummy)
372: 
373:     assert cls_emb.shape == (B, 128), f"Expected (4,128), got {cls_emb.shape}"
374:     assert logits.shape == (B, 102), f"Expected (4,102), got {logits.shape}"
375:     assert not torch.isnan(cls_emb).any(), "NaN in encoder output"
376:     assert not torch.isnan(logits).any(), "NaN in classifier logits"
377: 
378:     n_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
379:     print(f"Encoder output shape    : {cls_emb.shape}")
380:     print(f"Classifier output shape : {logits.shape}")
381:     print(f"Trainable parameters    : {n_params:,}")
382:     print("model.py OK.")

(End of file - total 382 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/configs/pretrain.yaml"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/configs/pretrain.yaml</path>
<type>file</type>
<content>1: # configs/pretrain.yaml
2: #
3: # Stage 1 — ANUBIS Pretraining Hyperparameters
4: # ─────────────────────────────────────────────
5: # All hyperparameters for pretraining the SpatioTemporalEncoder on the ANUBIS
6: # action-recognition dataset (102 classes, 32 joints, pelvis-centred).
7: # Fine-tuning / inference values belong in configs/module1.yaml.
8: # Never hardcode any of these values inside Python files.
9: 
10: # ── Data ──────────────────────────────────────────────────────────────────────
11: data:
12:   anubis_dir:         data/anubis/
13: 
14:   # File names inside anubis_dir (produced by your preprocessing notebook).
15:   # Shape convention: features (N, 6, 60, 32) — (samples, channels, frames, joints)
16:   #                   channels = [x, y, z, vx, vy, vz] (pelvis-centred, vel pre-computed)
17:   #                   labels   (N,) — integer class index in [0, 101]
18:   train_features:     trn_features_cleaned.npy
19:   train_labels:       trn_labels_cleaned.npy
20:   val_features:       val_features_cleaned.npy   # only used when has_separate_val=true
21:   val_labels:         val_labels_cleaned.npy
22: 
23:   has_separate_val:   true    # using separate val files
24:   val_split:          0.15    # fraction of training set used for val when has_separate_val=false
25: 
26:   num_classes:        102     # ANUBIS action categories
27:   num_joints:         32
28:   clip_frames:        60      # T stored in the .npy file
29:   seq_len:            60      # frames fed to the model; must be <= clip_frames
30:                               # if seq_len < clip_frames a random window is cropped
31: 
32: # ── Model ─────────────────────────────────────────────────────────────────────
33: model:
34:   in_features:           9    # pos(3) + vel(3) + acc(3) per joint
35:   gat_hidden_dim:       64    # output width of every GAT block (heads × per_head_dim)
36:   gat_heads:             4    # multi-head attention (first two GAT layers)
37:   gat_dropout:         0.2
38:   d_model:             128    # transformer embedding width
39:   nhead:                 4    # transformer attention heads
40:   num_transformer_layers: 2
41:   dim_feedforward:     256
42:   transformer_dropout: 0.2
43:   cls_std:            0.02    # trunc_normal std for CLS-token initialisation
44: 
45: # ── Training ──────────────────────────────────────────────────────────────────
46: training:
47:   epochs:              120
48:   batch_size:           64
49:   learning_rate:      0.0005
50:   weight_decay:      0.0005
51:   warmup_epochs:         5    # linear LR warm-up before cosine decay
52:   grad_clip_norm:      1.0
53:   label_smoothing:    0.05    # applied to CrossEntropyLoss
54:   save_every_n_epochs:  10
55: 
56: # ── Augmentation ──────────────────────────────────────────────────────────────
57: augmentation:
58:   enabled:            true
59:   flip_prob:          0.3     # probability of left-right joint flip per sample
60:   joint_jitter_std:  0.005    # Gaussian noise σ added to all coordinates
61:   speed_perturb:     false    # temporal speed perturbation
62:   speed_range:        [0.8, 1.2]   # [min_factor, max_factor] for temporal stretch
63: 
64: # ── Paths ─────────────────────────────────────────────────────────────────────
65: paths:
66:   checkpoint_dir:  outputs/checkpoints/pretrain/
67:   log_dir:         outputs/logs/
68: 
69: # ── Optuna Hyperparameter Search ──────────────────────────────────────────────
70: optuna:
71:   enabled:          false
72:   study_name:       anubis_pretrain_hpo
73:   direction:        maximize
74:   n_trials:         30
75:   timeout_sec:      null
76:   trial_epochs:     25
77:   seed:             42
78:   num_workers:      4
79:   pin_memory:       true
80:   storage:          null
81:   load_if_exists:   true
82: 
83:   output_best_params: outputs/logs/optuna_best_params.yaml
84:   output_best_config: outputs/logs/pretrain_optuna_best.yaml
85: 
86:   # Optional overrides for sampling ranges
87:   search_space:
88:     learning_rate:
89:       low: 0.0001
90:       high: 0.002
91:       log: true
92:     weight_decay:
93:       low: 0.00001
94:       high: 0.001
95:       log: true
96:     label_smoothing:
97:       low: 0.0
98:       high: 0.15
99:     gat_dropout:
100:       low: 0.05
101:       high: 0.35
102:     transformer_dropout:
103:       low: 0.05
104:       high: 0.35
105:     flip_prob:
106:       low: 0.1
107:       high: 0.6
108:     joint_jitter_std:
109:       low: 0.001
110:       high: 0.02
111:       log: true
112:     speed_perturb:
113:       choices: [true, false]

(End of file - total 113 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/configs/pretrain_final.yaml"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/configs/pretrain_final.yaml</path>
<type>file</type>
<content>1: # configs/pretrain_final.yaml
2: #
3: # Stage 1 — Final ANUBIS Pretraining Config (No Optuna)
4: # ──────────────────────────────────────────────────────
5: # This configuration uses the best Optuna hyperparameters discovered earlier,
6: # but trains and validates on the official ANUBIS split (separate val files)
7: # for a trustworthy generalisation estimate.
8: 
9: # ── Data ──────────────────────────────────────────────────────────────────────
10: data:
11:   anubis_dir:         data/anubis/
12: 
13:   train_features:     trn_features_cleaned.npy
14:   train_labels:       trn_labels_cleaned.npy
15:   val_features:       val_features_cleaned.npy
16:   val_labels:         val_labels_cleaned.npy
17: 
18:   has_separate_val:   true
19:   val_split:          0.15    # ignored when has_separate_val=true
20: 
21:   num_classes:        102
22:   num_joints:         32
23:   clip_frames:        60
24:   seq_len:            60
25: 
26: # ── Model ─────────────────────────────────────────────────────────────────────
27: model:
28:   in_features:             9
29:   gat_hidden_dim:         64
30:   gat_heads:               4
31:   gat_dropout:             0.051065527924036666
32:   d_model:               128
33:   nhead:                   4
34:   num_transformer_layers:  2
35:   dim_feedforward:       256
36:   transformer_dropout:     0.22439178858315353
37:   cls_std:                 0.02
38: 
39: # ── Training ──────────────────────────────────────────────────────────────────
40: training:
41:   epochs:                80
42:   batch_size:            64
43:   learning_rate:         0.0019464623176755585
44:   weight_decay:          0.00020433313066500611
45:   warmup_epochs:         5
46:   grad_clip_norm:        1.0
47:   label_smoothing:       0.040048660941805404
48:   save_every_n_epochs:   10
49: 
50: # ── Augmentation ──────────────────────────────────────────────────────────────
51: augmentation:
52:   enabled:               true
53:   flip_prob:             0.39576557171375976
54:   joint_jitter_std:      0.0020494388932778985
55:   speed_perturb:         true
56:   speed_range:           [0.9, 1.1]
57: 
58: # ── Paths ─────────────────────────────────────────────────────────────────────
59: paths:
60:   checkpoint_dir:  outputs/checkpoints/pretrain/
61:   log_dir:         outputs/logs/
62: 
63: # ── Optuna Hyperparameter Search ──────────────────────────────────────────────
64: optuna:
65:   enabled:          false
66:   study_name:       anubis_pretrain_hpo
67:   direction:        maximize
68:   n_trials:         30
69:   timeout_sec:      null
70:   trial_epochs:     25
71:   seed:             42
72:   num_workers:      4
73:   pin_memory:       true
74:   storage:          null
75:   load_if_exists:   true
76: 
77:   output_best_params: outputs/logs/optuna_best_params.yaml
78:   output_best_config: outputs/logs/pretrain_optuna_best.yaml
79: 
80:   search_space:
81:     learning_rate:
82:       low: 0.0001
83:       high: 0.002
84:       log: true
85:     weight_decay:
86:       low: 0.00001
87:       high: 0.001
88:       log: true
89:     label_smoothing:
90:       low: 0.0
91:       high: 0.15
92:     gat_dropout:
93:       low: 0.05
94:       high: 0.35
95:     transformer_dropout:
96:       low: 0.05
97:       high: 0.35
98:     flip_prob:
99:       low: 0.1
100:       high: 0.6
101:     joint_jitter_std:
102:       low: 0.001
103:       high: 0.02
104:       log: true
105:     speed_perturb:
106:       choices: [true, false]

(End of file - total 106 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/tests/test_model.py"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/tests/test_model.py</path>
<type>file</type>
<content>1: # tests/test_model.py
2: #
3: # Unit tests for module1/pretrain/model.py
4: # ──────────────────────────────────────────
5: # Covers: PositionalEncoding, SpatioTemporalEncoder (forward pass shapes,
6: #         no NaNs, edge_index buffer device movement), ActionClassifier
7: #         (forward pass, logit shape), and encoder/classifier separation.
8: #
9: # Run with: pytest tests/test_model.py -v
10: 
11: import pytest
12: import torch
13: import torch.nn as nn
14: 
15: from module1.data.joint_config import EDGE_INDEX, NUM_JOINTS
16: from module1.pretrain.model import (
17:     ActionClassifier,
18:     PositionalEncoding,
19:     SpatioTemporalEncoder,
20:     build_model,
21: )
22: 
23: 
24: # ── Constants used across tests ───────────────────────────────────────────────
25: 
26: B = 2  # batch size (small to keep tests fast)
27: T = 60  # sequence length (frames)
28: J = NUM_JOINTS  # 32
29: F_IN = 9  # features per joint
30: D_MODEL = 128
31: NUM_CLASSES = 102
32: 
33: 
34: # ── Fixtures ──────────────────────────────────────────────────────────────────
35: 
36: 
37: @pytest.fixture
38: def edge_index() -> torch.Tensor:
39:     return EDGE_INDEX.clone()
40: 
41: 
42: @pytest.fixture
43: def encoder(edge_index: torch.Tensor) -> SpatioTemporalEncoder:
44:     model = SpatioTemporalEncoder(
45:         num_joints=J,
46:         in_features=F_IN,
47:         edge_index=edge_index,
48:         gat_hidden=64,
49:         gat_heads=4,
50:         d_model=D_MODEL,
51:         nhead=4,
52:         num_layers=2,
53:         dim_ff=256,
54:         dropout=0.0,  # disable dropout so outputs are deterministic in eval
55:         seq_len=T,
56:     )
57:     model.eval()
58:     return model
59: 
60: 
61: @pytest.fixture
62: def classifier(encoder: SpatioTemporalEncoder) -> ActionClassifier:
63:     model = ActionClassifier(encoder, num_classes=NUM_CLASSES)
64:     model.eval()
65:     return model
66: 
67: 
68: @pytest.fixture
69: def dummy_input() -> torch.Tensor:
70:     """Small random batch: (B, T, J, F_IN)."""
71:     torch.manual_seed(0)
72:     return torch.randn(B, T, J, F_IN)
73: 
74: 
75: @pytest.fixture
76: def minimal_cfg() -> dict:
77:     """Minimal config dict that mirrors configs/pretrain.yaml structure."""
78:     return {
79:         "data": {
80:             "num_joints": J,
81:             "seq_len": T,
82:             "num_classes": NUM_CLASSES,
83:         },
84:         "model": {
85:             "in_features": F_IN,
86:             "gat_hidden_dim": 64,
87:             "gat_heads": 4,
88:             "gat_dropout": 0.0,
89:             "d_model": D_MODEL,
90:             "nhead": 4,
91:             "num_transformer_layers": 2,
92:             "dim_feedforward": 256,
93:             "transformer_dropout": 0.0,
94:             "cls_std": 0.02,
95:         },
96:     }
97: 
98: 
99: # ── PositionalEncoding ────────────────────────────────────────────────────────
100: 
101: 
102: class TestPositionalEncoding:
103:     def test_output_shape(self) -> None:
104:         pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
105:         x = torch.zeros(B, T, D_MODEL)
106:         out = pe(x)
107:         assert out.shape == (B, T, D_MODEL)
108: 
109:     def test_no_nan(self) -> None:
110:         pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
111:         x = torch.randn(B, T, D_MODEL)
112:         out = pe(x)
113:         assert not torch.isnan(out).any()
114: 
115:     def test_adds_positional_signal(self) -> None:
116:         # Zero input + no dropout: output should equal the PE buffer, not zero
117:         pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
118:         pe.eval()
119:         x = torch.zeros(B, T, D_MODEL)
120:         out = pe(x)
121:         assert not torch.all(out == 0)
122: 
123:     def test_different_positions_differ(self) -> None:
124:         # Adjacent time steps should (almost certainly) have different encodings
125:         pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
126:         pe.eval()
127:         x = torch.zeros(1, T, D_MODEL)
128:         out = pe(x)
129:         assert not torch.allclose(out[0, 0, :], out[0, 1, :])
130: 
131: 
132: # ── SpatioTemporalEncoder ─────────────────────────────────────────────────────
133: 
134: 
135: class TestSpatioTemporalEncoder:
136:     @torch.no_grad()
137:     def test_output_shape(
138:         self, encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
139:     ) -> None:
140:         out = encoder(dummy_input)
141:         assert out.shape == (B, D_MODEL), f"Expected ({B},{D_MODEL}), got {out.shape}"
142: 
143:     @torch.no_grad()
144:     def test_no_nan_in_output(
145:         self, encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
146:     ) -> None:
147:         out = encoder(dummy_input)
148:         assert not torch.isnan(out).any(), "NaN detected in encoder output"
149: 
150:     @torch.no_grad()
151:     def test_no_inf_in_output(
152:         self, encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
153:     ) -> None:
154:         out = encoder(dummy_input)
155:         assert not torch.isinf(out).any(), "Inf detected in encoder output"
156: 
157:     @torch.no_grad()
158:     def test_different_inputs_give_different_outputs(
159:         self, encoder: SpatioTemporalEncoder
160:     ) -> None:
161:         torch.manual_seed(1)
162:         x1 = torch.randn(B, T, J, F_IN)
163:         x2 = torch.randn(B, T, J, F_IN)
164:         out1 = encoder(x1)
165:         out2 = encoder(x2)
166:         assert not torch.allclose(out1, out2), "Encoder is constant regardless of input"
167: 
168:     @torch.no_grad()
169:     def test_batch_independence(self, encoder: SpatioTemporalEncoder) -> None:
170:         # Encoding one sample individually must equal its row in a batched pass
171:         torch.manual_seed(2)
172:         x = torch.randn(B, T, J, F_IN)
173:         batch_out = encoder(x)
174:         single_out = encoder(x[0:1])
175:         assert torch.allclose(batch_out[0], single_out[0], atol=1e-5), (
176:             "Batch output differs from single-sample output"
177:         )
178: 
179:     def test_edge_index_is_buffer(self, encoder: SpatioTemporalEncoder) -> None:
180:         # edge_index must be registered as a buffer (non-parameter)
181:         buffer_names = {name for name, _ in encoder.named_buffers()}
182:         assert "edge_index" in buffer_names
183: 
184:     def test_edge_index_shape(self, encoder: SpatioTemporalEncoder) -> None:
185:         assert encoder.edge_index.shape == EDGE_INDEX.shape
186: 
187:     def test_has_no_unused_params_in_training_mode(
188:         self, encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
189:     ) -> None:
190:         """All parameters must receive a gradient after a backward pass."""
191:         encoder.train()
192:         out = encoder(dummy_input)
193:         loss = out.sum()
194:         loss.backward()
195:         for name, param in encoder.named_parameters():
196:             if param.requires_grad:
197:                 assert param.grad is not None, f"No gradient for parameter: {name}"
198:         encoder.eval()
199: 
200: 
201: # ── ActionClassifier ──────────────────────────────────────────────────────────
202: 
203: 
204: class TestActionClassifier:
205:     @torch.no_grad()
206:     def test_output_shape(
207:         self, classifier: ActionClassifier, dummy_input: torch.Tensor
208:     ) -> None:
209:         logits = classifier(dummy_input)
210:         assert logits.shape == (B, NUM_CLASSES), (
211:             f"Expected ({B},{NUM_CLASSES}), got {logits.shape}"
212:         )
213: 
214:     @torch.no_grad()
215:     def test_no_nan_in_logits(
216:         self, classifier: ActionClassifier, dummy_input: torch.Tensor
217:     ) -> None:
218:         logits = classifier(dummy_input)
219:         assert not torch.isnan(logits).any()
220: 
221:     @torch.no_grad()
222:     def test_logits_are_unnormalised(
223:         self, classifier: ActionClassifier, dummy_input: torch.Tensor
224:     ) -> None:
225:         # The head is a plain Linear — softmax probabilities should NOT sum to 1
226:         logits = classifier(dummy_input)
227:         row_sums = logits.softmax(dim=-1).sum(dim=-1)
228:         # After softmax they should sum to ~1, but raw logits sum should not
229:         assert not torch.allclose(logits.sum(dim=-1), torch.ones(B))
230: 
231:     def test_encoder_is_accessible(self, classifier: ActionClassifier) -> None:
232:         assert isinstance(classifier.encoder, SpatioTemporalEncoder)
233: 
234:     def test_encoder_and_classifier_share_params(
235:         self, classifier: ActionClassifier
236:     ) -> None:
237:         # Parameters of classifier.encoder must be a subset of classifier's params
238:         enc_ids = {id(p) for p in classifier.encoder.parameters()}
239:         cls_ids = {id(p) for p in classifier.parameters()}
240:         assert enc_ids.issubset(cls_ids)
241: 
242:     def test_head_not_in_encoder(self, classifier: ActionClassifier) -> None:
243:         enc_ids = {id(p) for p in classifier.encoder.parameters()}
244:         head_ids = {id(p) for p in classifier.head.parameters()}
245:         assert head_ids.isdisjoint(enc_ids), "Head parameters appear in encoder"
246: 
247: 
248: # ── build_model factory ───────────────────────────────────────────────────────
249: 
250: 
251: class TestBuildModel:
252:     def test_returns_encoder_and_classifier(self, minimal_cfg: dict) -> None:
253:         encoder, classifier = build_model(minimal_cfg, EDGE_INDEX)
254:         assert isinstance(encoder, SpatioTemporalEncoder)
255:         assert isinstance(classifier, ActionClassifier)
256: 
257:     def test_classifier_wraps_same_encoder(self, minimal_cfg: dict) -> None:
258:         encoder, classifier = build_model(minimal_cfg, EDGE_INDEX)
259:         assert classifier.encoder is encoder
260: 
261:     @torch.no_grad()
262:     def test_forward_pass_after_build(self, minimal_cfg: dict) -> None:
263:         _, classifier = build_model(minimal_cfg, EDGE_INDEX)
264:         classifier.eval()
265:         dummy = torch.randn(B, T, J, F_IN)
266:         logits = classifier(dummy)
267:         assert logits.shape == (B, NUM_CLASSES)
268:         assert not torch.isnan(logits).any()

(End of file - total 268 lines)
</content>
```

---

## Assistant (Plan · gpt-5.3-codex · 36.7s)

_Thinking:_

**Planning Perceiver and Rotary integration**

Perfect — yes, I can implement both **Perceiver-style latent bottlenecks** and **RoPE**.

I’m currently in read-only plan mode, so here is the exact implementation plan I’ll execute next.

**Plan**
- Add a config-driven temporal backbone in `module1/pretrain/model.py`:
  - keep existing `transformer` path (current baseline),
  - add new `perceiver` path (latent bottleneck).
- Add RoPE support for attention layers:
  - implement rotary embedding helper,
  - apply RoPE to Q/K in Perceiver cross-attention and latent self-attention.
- Keep output contract unchanged: encoder still returns `(B, d_model)`, so `ActionClassifier`, `train.py`, `evaluate.py`, and live inference continue to work without interface changes.

**Model Changes (Detailed)**
- In `SpatioTemporalEncoder`:
  - Keep spatial GAT stage unchanged.
  - Replace temporal stage with switch:
    - `temporal_backbone: transformer` -> current implementation.
    - `temporal_backbone: perceiver` -> new latent stack:
      - learnable latent array `(num_latents, d_model)`,
      - repeated blocks: cross-attn (latents attend input tokens) + latent self-attn + FFN (pre-norm residual),
      - readout via latent[0] (CLS-like) -> `(B, d_model)`.
- Add RoPE module:
  - sequence-length-aware cos/sin cache,
  - rotate half-dim for Q/K,
  - configurable `rope_base`.

**Config Changes**
- Add new keys under `model` (with safe defaults):
  - `temporal_backbone: transformer|perceiver`
  - `positional_encoding: sinusoidal|rope`
  - `num_latents`, `num_perceiver_layers`
  - `cross_attn_heads`, `self_attn_heads`
  - `perceiver_dropout`, `perceiver_dim_feedforward`
  - `rope_base`
- Create a dedicated experiment config (recommended):
  - `configs/pretrain_perceiver_rope.yaml`
  - based on your best current config, only architecture keys changed.

**Tests**
- Update `tests/test_model.py` to include:
  - perceiver+rope forward shape test,
  - no-NaN/no-Inf checks,
  - gradient flow test for new path,
  - build-model factory test with new config keys.

**Docs**
- Update `docs/COMMANDS.md` with:
  - training command for new config,
  - brief note that old checkpoints are architecture-specific.
- Optionally add one line in `README.md` pointing to Perceiver+RoPE experiment config.

**Execution Strategy**
- Keep baseline unchanged.
- Add new architecture as opt-in so you can run A/B cleanly:
  1) baseline (`pretrain_final.yaml`),  
  2) perceiver+rope (`pretrain_perceiver_rope.yaml`).

One decision before I execute:  
1. **Use separate experimental config (Recommended)**  
2. Replace `pretrain_final.yaml` defaults immediately

Reply with `1` or `2`, and I’ll implement.

---

## User

1

---

## Assistant (Plan · gpt-5.3-codex · 20.6s)

_Thinking:_

**Presenting comprehensive implementation plan**

Great — we’ll go with **Option 1 (separate experimental config)**.

Here’s the implementation plan I’ll execute next (no edits yet, per plan mode).

**Plan**
- Add a temporal-backbone switch in `module1/pretrain/model.py`:
  - keep existing path as `transformer` (current baseline behavior),
  - add new `perceiver` path with latent bottlenecks.
- Implement RoPE in the new Perceiver attention stack:
  - rotary helper for Q/K rotation,
  - apply in cross-attention and latent self-attention,
  - configurable `rope_base` (default `10000`).
- Keep interfaces stable:
  - `SpatioTemporalEncoder.forward(x)` still returns `(B, d_model)`,
  - `ActionClassifier`, `train.py`, `evaluate.py`, and `live_inference.py` require no API changes.
- Add config keys (optional, backward compatible) used by `build_model()`:
  - `temporal_backbone`, `positional_encoding`,
  - `num_latents`, `num_perceiver_layers`,
  - `cross_attn_heads`, `self_attn_heads`,
  - `perceiver_dim_feedforward`, `perceiver_dropout`,
  - `rope_base`.
- Create new experiment config `configs/pretrain_perceiver_rope.yaml`:
  - based on `configs/pretrain_final.yaml`,
  - set `temporal_backbone: perceiver`, `positional_encoding: rope`,
  - recommended defaults: `num_latents: 64`, `num_perceiver_layers: 2`.
- Update tests in `tests/test_model.py`:
  - add perceiver+rope forward shape and NaN checks,
  - gradient flow on new path,
  - factory/build coverage with new config keys.
- Update docs in `docs/COMMANDS.md`:
  - add train/eval commands for `pretrain_perceiver_rope.yaml`,
  - note checkpoint compatibility (old transformer checkpoints won’t load into perceiver model).

**Validation run (after implementation)**
- `uv run module1/pretrain/train.py --config configs/pretrain_perceiver_rope.yaml`
- `uv run module1/pretrain/evaluate.py --config configs/pretrain_perceiver_rope.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --split val --tsne`

One quick preference before I implement: keep `num_latents=64` (recommended), or use `32` to reduce GPU memory/compute?

---

## User

now build the architecture

---

## Assistant (Build · gpt-5.3-codex · 167.5s)

_Thinking:_

**Planning major model.py overhaul**

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Update File: /home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py\n@@\n-# module1/pretrain/model.py\n-#\n-# Stage 1 — SpatioTemporalEncoder + ActionClassifier\n-# ────────────────────────────────────────────────────\n-# End-to-end model that encodes a sequence of skeleton graphs:\n-#\n-#   Input : (B, T, J, F)  — batch, frames, joints, features (pos+vel+acc = 9)\n-#\n-#   Spatial stage — time-distributed GAT (shared weights across all T frames):\n-#     GATConv( F → 16, heads=4, concat=True)  → 64-dim per node + BN + ELU\n-#     GATConv(64 → 16, heads=4, concat=True)  → 64-dim per node + BN + ELU\n-#     GATConv(64 → 64, heads=1, concat=False) → 64-dim per node + BN + ELU\n-#     global_mean_pool over joints             → (B*T, 64)\n-#     reshape                                  → (B, T, 64)\n-#\n-#   Temporal stage — Transformer encoder:\n-#     input_proj Linear(64 → d_model)\n-#     prepend CLS token\n-#     sinusoidal positional encoding\n-#     2 × TransformerEncoderLayer (pre-norm, nhead=4, ff=256)\n-#     LayerNorm\n-#     extract CLS output                       → (B, d_model)\n-#\n-#   ActionClassifier wraps SpatioTemporalEncoder with a linear head for\n-#   102-class ANUBIS pretraining.  After pretraining, save only the encoder\n-#   (encoder.state_dict()) and attach new fine-tuning heads.\n-\n-from __future__ import annotations\n-\n-import math\n-from typing import Optional, Tuple\n-\n-import torch\n-import torch.nn as nn\n-import torch.nn.functional as F\n-from torch_geometric.nn import GATConv, global_mean_pool\n-\n-from utils.logger import get_logger\n-\n-log = get_logger(__name__)\n-\n-\n-# ── Positional Encoding ────────────────────────────────────────────────────────\n-\n-\n-class PositionalEncoding(nn.Module):\n-    \"\"\"\n-    Sinusoidal positional encoding (Vaswani et al., 2017).\n-\n-    Parameters\n-    ----------\n-    d_model  : int   embedding dimension\n-    max_len  : int   maximum sequence length (including CLS token)\n-    dropout  : float dropout applied after adding the encoding\n-    \"\"\"\n-\n-    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1) -> None:\n-        super().__init__()\n-        self.dropout = nn.Dropout(p=dropout)\n-\n-        position = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)\n-        div_term = torch.exp(\n-            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)\n-        )\n-        pe = torch.zeros(1, max_len, d_model)\n-        pe[0, :, 0::2] = torch.sin(position * div_term)\n-        pe[0, :, 1::2] = torch.cos(position * div_term)\n-        self.register_buffer(\"pe\", pe)\n-\n-    def forward(self, x: torch.Tensor) -> torch.Tensor:\n-        \"\"\"\n-        Parameters\n-        ----------\n-        x : torch.Tensor  shape (B, T, d_model)\n-\n-        Returns\n-        -------\n-        out : torch.Tensor  shape (B, T, d_model)\n-        \"\"\"\n-        x = x + self.pe[:, : x.size(1), :]\n-        return self.dropout(x)\n-\n-\n-# ── SpatioTemporalEncoder ─────────────────────────────────────────────────────\n-\n-\n-class SpatioTemporalEncoder(nn.Module):\n-    \"\"\"\n-    End-to-end spatiotemporal encoder: time-distributed GAT + Transformer.\n-\n-    The encoder is the shared backbone used across all training stages.\n-    It outputs a single CLS embedding per sequence.\n-\n-    Parameters\n-    ----------\n-    num_joints   : int   number of skeleton joints (32 for ANUBIS)\n-    in_features  : int   feature dimension per joint (9: pos+vel+acc)\n-    edge_index   : torch.Tensor  shape (2, E)  skeleton connectivity;\n-                   registered as a buffer so it moves to device automatically\n-    gat_hidden   : int   output dim of every GAT block (default 64)\n-    gat_heads    : int   attention heads for GAT layers 1 & 2 (default 4)\n-    gat_dropout  : float dropout on GAT attention coefficients\n-    d_model      : int   transformer embedding dimension (default 128)\n-    nhead        : int   transformer attention heads\n-    num_layers   : int   number of TransformerEncoderLayer blocks\n-    dim_ff       : int   feedforward expansion inside transformer\n-    dropout      : float transformer dropout\n-    seq_len      : int   expected sequence length (used to size pos. encoding)\n-    cls_std      : float trunc_normal std for CLS-token initialisation\n-    \"\"\"\n-\n-    def __init__(\n-        self,\n-        num_joints: int,\n-        in_features: int,\n-        edge_index: torch.Tensor,\n-        gat_hidden: int = 64,\n-        gat_heads: int = 4,\n-        gat_dropout: float = 0.1,\n-        d_model: int = 128,\n-        nhead: int = 4,\n-        num_layers: int = 2,\n-        dim_ff: int = 256,\n-        dropout: float = 0.1,\n-        seq_len: int = 60,\n-        cls_std: float = 0.02,\n-    ) -> None:\n-        super().__init__()\n-\n-        self.num_joints = num_joints\n-        self.gat_hidden = gat_hidden\n-        self.gat_dropout = gat_dropout\n-        self.d_model = d_model\n-\n-        # Register skeleton topology as a non-learnable buffer\n-        self.register_buffer(\"edge_index\", edge_index)  # (2, E)\n-\n-        # ── GAT layers (shared across all time steps) ──────────────────────\n-        per_head_dim = gat_hidden // gat_heads  # 16\n-        self.gat1 = GATConv(\n-            in_features, per_head_dim, heads=gat_heads, concat=True, dropout=gat_dropout\n-        )\n-        self.gat2 = GATConv(\n-            gat_hidden, per_head_dim, heads=gat_heads, concat=True, dropout=gat_dropout\n-        )\n-        self.gat3 = GATConv(\n-            gat_hidden, gat_hidden, heads=1, concat=False, dropout=gat_dropout\n-        )\n-\n-        self.bn1 = nn.BatchNorm1d(gat_hidden)\n-        self.bn2 = nn.BatchNorm1d(gat_hidden)\n-        self.bn3 = nn.BatchNorm1d(gat_hidden)\n-\n-        # ── Transformer ────────────────────────────────────────────────────\n-        self.input_proj = nn.Linear(gat_hidden, d_model)\n-\n-        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))\n-        nn.init.trunc_normal_(self.cls_token, std=cls_std)\n-\n-        # max_len = seq_len + 1 to accommodate the CLS token\n-        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)\n-\n-        encoder_layer = nn.TransformerEncoderLayer(\n-            d_model=d_model,\n-            nhead=nhead,\n-            dim_feedforward=dim_ff,\n-            dropout=dropout,\n-            batch_first=True,\n-            norm_first=True,  # pre-norm variant — more stable gradients\n-            activation=\"gelu\",\n-        )\n-        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)\n-        self.norm = nn.LayerNorm(d_model)\n-\n-    # ── Internal helpers ───────────────────────────────────────────────────────\n-\n-    def _gat_forward(self, x: torch.Tensor) -> torch.Tensor:\n-        \"\"\"\n-        Process all T frames through the GAT in a single batched pass.\n-\n-        Each (frame, sequence) pair is treated as an independent graph.\n-        All B*T graphs share the same edge topology (self.edge_index).\n-\n-        Parameters\n-        ----------\n-        x : torch.Tensor  shape (B, T, J, F)\n-\n-        Returns\n-        -------\n-        spatial : torch.Tensor  shape (B, T, gat_hidden)\n-        \"\"\"\n-        B, T, J, Fin = x.shape\n-        num_graphs = B * T\n-\n-        # Flatten nodes: (B*T*J, Fin)\n-        x_nodes = x.reshape(num_graphs * J, Fin)\n-\n-        # Build batched edge index by offsetting each graph's node indices by J\n-        # edge_index: (2, E)  →  broadcast to (2, num_graphs, E)\n-        E = self.edge_index.shape[1]\n-        offsets = torch.arange(num_graphs, device=x.device) * J  # (num_graphs,)\n-        edge_batch = self.edge_index.unsqueeze(1).expand(\n-            -1, num_graphs, -1\n-        )  # (2, G, E)\n-        edge_batch = edge_batch + offsets.view(1, -1, 1)  # (2, G, E)\n-        edge_flat = edge_batch.reshape(2, num_graphs * E)  # (2, G*E)\n-\n-        # batch_tensor: maps each node to its graph index\n-        batch_tensor = torch.arange(num_graphs, device=x.device).repeat_interleave(J)\n-\n-        # ── GAT layer 1 ────────────────────────────────────────────────────\n-        h = self.gat1(x_nodes, edge_flat)  # (B*T*J, gat_hidden)\n-        h = self.bn1(h)\n-        h = F.elu(h)\n-        h = F.dropout(h, p=self.gat_dropout, training=self.training)\n-\n-        # ── GAT layer 2 ────────────────────────────────────────────────────\n-        h = self.gat2(h, edge_flat)\n-        h = self.bn2(h)\n-        h = F.elu(h)\n-        h = F.dropout(h, p=self.gat_dropout, training=self.training)\n-\n-        # ── GAT layer 3 ────────────────────────────────────────────────────\n-        h = self.gat3(h, edge_flat)\n-        h = self.bn3(h)\n-        h = F.elu(h)\n-\n-        # Global mean pool: (B*T*J, gat_hidden) → (B*T, gat_hidden)\n-        spatial = global_mean_pool(h, batch_tensor)\n-\n-        return spatial.reshape(B, T, self.gat_hidden)  # (B, T, gat_hidden)\n-\n-    # ── Forward pass ──────────────────────────────────────────────────────────\n-\n-    def forward(self, x: torch.Tensor) -> torch.Tensor:\n-        \"\"\"\n-        Encode a batch of skeleton sequences.\n-\n-        Parameters\n-        ----------\n-        x : torch.Tensor  shape (B, T, J, F)\n-            Joint features: pos + vel + acc (F = 9).\n-\n-        Returns\n-        -------\n-        cls_embedding : torch.Tensor  shape (B, d_model)\n-            Sequence-level representation extracted from the CLS token.\n-        \"\"\"\n-        B = x.shape[0]\n-\n-        # Stage 1: spatial encoding (time-distributed GAT)\n-        spatial = self._gat_forward(x)  # (B, T, gat_hidden)\n-\n-        # Stage 2: temporal encoding\n-        h = self.input_proj(spatial)  # (B, T, d_model)\n-\n-        # Prepend CLS token\n-        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)\n-        h = torch.cat([cls, h], dim=1)  # (B, T+1, d_model)\n-\n-        # Positional encoding\n-        h = self.pos_enc(h)  # (B, T+1, d_model)\n-\n-        # Transformer encoder\n-        h = self.transformer(h)  # (B, T+1, d_model)\n-        h = self.norm(h)\n-\n-        # Extract CLS token (position 0)\n-        return h[:, 0, :]  # (B, d_model)\n-\n-\n-# ── ActionClassifier ──────────────────────────────────────────────────────────\n-\n-\n-class ActionClassifier(nn.Module):\n-    \"\"\"\n-    Full pretraining model: SpatioTemporalEncoder + linear classification head.\n-\n-    Used only for Stage 1 (ANUBIS action classification pretraining).\n-    After pretraining, save encoder.state_dict() and discard the head.\n-\n-    Parameters\n-    ----------\n-    encoder     : SpatioTemporalEncoder\n-    num_classes : int   number of action classes (102 for ANUBIS)\n-    \"\"\"\n-\n-    def __init__(self, encoder: SpatioTemporalEncoder, num_classes: int) -> None:\n-        super().__init__()\n-        self.encoder = encoder\n-        self.head = nn.Linear(encoder.d_model, num_classes)\n-\n-    def forward(self, x: torch.Tensor) -> torch.Tensor:\n-        \"\"\"\n-        Parameters\n-        ----------\n-        x : torch.Tensor  shape (B, T, J, F)\n-\n-        Returns\n-        -------\n-        logits : torch.Tensor  shape (B, num_classes)\n-        \"\"\"\n-        cls_emb = self.encoder(x)  # (B, d_model)\n-        return self.head(cls_emb)  # (B, num_classes)\n-\n-\n-# ── Factory helper ────────────────────────────────────────────────────────────\n-\n-\n-def build_model(\n-    cfg: dict, edge_index: torch.Tensor\n-) -> Tuple[SpatioTemporalEncoder, ActionClassifier]:\n-    \"\"\"\n-    Construct encoder and full classifier from config dict.\n-\n-    Parameters\n-    ----------\n-    cfg        : dict               full config loaded from configs/pretrain.yaml\n-    edge_index : torch.Tensor (2,E) skeleton edge index\n-\n-    Returns\n-    -------\n-    encoder    : SpatioTemporalEncoder\n-    classifier : ActionClassifier\n-    \"\"\"\n-    m = cfg[\"model\"]\n-    d = cfg[\"data\"]\n-    encoder = SpatioTemporalEncoder(\n-        num_joints=d[\"num_joints\"],\n-        in_features=m[\"in_features\"],\n-        edge_index=edge_index,\n-        gat_hidden=m[\"gat_hidden_dim\"],\n-        gat_heads=m[\"gat_heads\"],\n-        gat_dropout=m[\"gat_dropout\"],\n-        d_model=m[\"d_model\"],\n-        nhead=m[\"nhead\"],\n-        num_layers=m[\"num_transformer_layers\"],\n-        dim_ff=m[\"dim_feedforward\"],\n-        dropout=m[\"transformer_dropout\"],\n-        seq_len=d[\"seq_len\"],\n-        cls_std=m[\"cls_std\"],\n-    )\n-    classifier = ActionClassifier(encoder, num_classes=d[\"num_classes\"])\n-    total = sum(p.numel() for p in classifier.parameters() if p.requires_grad)\n-    log.info(\"Model built: %d trainable parameters\", total)\n-    return encoder, classifier\n-\n-\n-# ── Quick sanity check ────────────────────────────────────────────────────────\n-\n-if __name__ == \"__main__\":\n-    from module1.data.joint_config import EDGE_INDEX, NUM_JOINTS\n-\n-    B, T, J, F_IN = 4, 60, NUM_JOINTS, 9\n-    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n-\n-    encoder = SpatioTemporalEncoder(\n-        num_joints=J,\n-        in_features=F_IN,\n-        edge_index=EDGE_INDEX,\n-        seq_len=T,\n-    ).to(device)\n-\n-    classifier = ActionClassifier(encoder, num_classes=102).to(device)\n-    classifier.eval()\n-\n-    dummy = torch.randn(B, T, J, F_IN, device=device)\n-\n-    with torch.no_grad():\n-        cls_emb = encoder(dummy)\n-        logits = classifier(dummy)\n-\n-    assert cls_emb.shape == (B, 128), f\"Expected (4,128), got {cls_emb.shape}\"\n-    assert logits.shape == (B, 102), f\"Expected (4,102), got {logits.shape}\"\n-    assert not torch.isnan(cls_emb).any(), \"NaN in encoder output\"\n-    assert not torch.isnan(logits).any(), \"NaN in classifier logits\"\n-\n-    n_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)\n-    print(f\"Encoder output shape    : {cls_emb.shape}\")\n-    print(f\"Classifier output shape : {logits.shape}\")\n-    print(f\"Trainable parameters    : {n_params:,}\")\n-    print(\"model.py OK.\")\n+# module1/pretrain/model.py\n+#\n+# Stage 1 - SpatioTemporalEncoder + ActionClassifier\n+# ---------------------------------------------------\n+# End-to-end model that encodes a sequence of skeleton graphs.\n+#\n+# Input: (B, T, J, F) where F=9 [x,y,z,vx,vy,vz,ax,ay,az].\n+#\n+# Spatial stage (shared across all T frames):\n+#   GATConv -> BN -> ELU -> Dropout\n+#   GATConv -> BN -> ELU -> Dropout\n+#   GATConv -> BN -> ELU\n+#   global_mean_pool over joints -> (B, T, gat_hidden)\n+#\n+# Temporal stage (configurable):\n+#   1) Transformer (baseline): CLS + sinusoidal position + TransformerEncoder\n+#   2) Perceiver latent bottleneck: cross-attn + latent self-attn (+ optional RoPE)\n+\n+from __future__ import annotations\n+\n+import math\n+from typing import Optional, Tuple\n+\n+import torch\n+import torch.nn as nn\n+import torch.nn.functional as F\n+from torch_geometric.nn import GATConv, global_mean_pool\n+\n+from utils.logger import get_logger\n+\n+log = get_logger(__name__)\n+\n+\n+# -- Positional encoding -------------------------------------------------------\n+\n+\n+class PositionalEncoding(nn.Module):\n+    \"\"\"\n+    Sinusoidal positional encoding (Vaswani et al., 2017).\n+\n+    Parameters\n+    ----------\n+    d_model : int\n+        Embedding dimension.\n+    max_len : int\n+        Maximum sequence length.\n+    dropout : float\n+        Dropout applied after adding positional encoding.\n+    \"\"\"\n+\n+    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1) -> None:\n+        super().__init__()\n+        self.dropout = nn.Dropout(p=dropout)\n+\n+        position = torch.arange(max_len).unsqueeze(1).float()\n+        div_term = torch.exp(\n+            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)\n+        )\n+        pe = torch.zeros(1, max_len, d_model)\n+        pe[0, :, 0::2] = torch.sin(position * div_term)\n+        pe[0, :, 1::2] = torch.cos(position * div_term)\n+        self.register_buffer(\"pe\", pe)\n+\n+    def forward(self, x: torch.Tensor) -> torch.Tensor:\n+        x = x + self.pe[:, : x.size(1), :]\n+        return self.dropout(x)\n+\n+\n+def _rotate_half(x: torch.Tensor) -> torch.Tensor:\n+    \"\"\"Rotate even/odd features for rotary embedding.\"\"\"\n+    x_even = x[..., ::2]\n+    x_odd = x[..., 1::2]\n+    x_rot = torch.stack((-x_odd, x_even), dim=-1)\n+    return x_rot.flatten(-2)\n+\n+\n+class RotaryEmbedding(nn.Module):\n+    \"\"\"\n+    Rotary positional embedding helper for attention Q/K tensors.\n+\n+    Parameters\n+    ----------\n+    dim : int\n+        Attention head dimension (must be even).\n+    base : float\n+        RoPE base frequency.\n+    \"\"\"\n+\n+    def __init__(self, dim: int, base: float = 10000.0) -> None:\n+        super().__init__()\n+        if dim % 2 != 0:\n+            raise ValueError(f\"RoPE requires even head dimension, got dim={dim}\")\n+        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))\n+        self.register_buffer(\"inv_freq\", inv_freq)\n+\n+    def get_cos_sin(\n+        self,\n+        seq_len: int,\n+        device: torch.device,\n+        dtype: torch.dtype,\n+    ) -> Tuple[torch.Tensor, torch.Tensor]:\n+        \"\"\"Return cosine and sine tensors shaped for (B, H, T, D).\"\"\"\n+        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)\n+        freqs = torch.outer(t, self.inv_freq)  # (T, D/2)\n+        emb = torch.cat([freqs, freqs], dim=-1)  # (T, D)\n+        cos = emb.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)\n+        sin = emb.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)\n+        return cos, sin\n+\n+\n+# -- Attention blocks ----------------------------------------------------------\n+\n+\n+class MultiHeadAttention(nn.Module):\n+    \"\"\"\n+    Multi-head attention supporting self-attention and cross-attention.\n+\n+    Parameters\n+    ----------\n+    d_model : int\n+        Embedding dimension.\n+    nhead : int\n+        Number of attention heads.\n+    dropout : float\n+        Dropout on attention probabilities and output projection.\n+    cross_attention : bool\n+        If True, query comes from x and key/value comes from context.\n+    use_rope : bool\n+        If True, apply RoPE to q/k tensors.\n+    rope_base : float\n+        RoPE base frequency.\n+    \"\"\"\n+\n+    def __init__(\n+        self,\n+        d_model: int,\n+        nhead: int,\n+        dropout: float = 0.1,\n+        cross_attention: bool = False,\n+        use_rope: bool = False,\n+        rope_base: float = 10000.0,\n+    ) -> None:\n+        super().__init__()\n+        if d_model % nhead != 0:\n+            raise ValueError(\n+                f\"d_model={d_model} must be divisible by nhead={nhead}\"\n+            )\n+\n+        self.d_model = d_model\n+        self.nhead = nhead\n+        self.head_dim = d_model // nhead\n+        self.scale = self.head_dim ** -0.5\n+        self.cross_attention = cross_attention\n+        self.use_rope = use_rope\n+\n+        if cross_attention:\n+            self.q_proj = nn.Linear(d_model, d_model)\n+            self.kv_proj = nn.Linear(d_model, 2 * d_model)\n+        else:\n+            self.qkv_proj = nn.Linear(d_model, 3 * d_model)\n+\n+        self.out_proj = nn.Linear(d_model, d_model)\n+        self.attn_dropout = nn.Dropout(dropout)\n+        self.out_dropout = nn.Dropout(dropout)\n+\n+        if use_rope:\n+            self.rope = RotaryEmbedding(self.head_dim, base=rope_base)\n+        else:\n+            self.rope = None\n+\n+    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:\n+        \"\"\"(B, T, D) -> (B, H, T, Dh)\"\"\"\n+        B, T, _ = x.shape\n+        x = x.view(B, T, self.nhead, self.head_dim)\n+        return x.permute(0, 2, 1, 3)\n+\n+    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:\n+        \"\"\"(B, H, T, Dh) -> (B, T, D)\"\"\"\n+        B, H, T, Dh = x.shape\n+        return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * Dh)\n+\n+    def forward(\n+        self,\n+        x: torch.Tensor,\n+        context: Optional[torch.Tensor] = None,\n+    ) -> torch.Tensor:\n+        \"\"\"\n+        Parameters\n+        ----------\n+        x : torch.Tensor\n+            Query tensor, shape (B, Tq, D).\n+        context : torch.Tensor, optional\n+            Key/value source, shape (B, Tk, D). Required for cross-attention.\n+\n+        Returns\n+        -------\n+        out : torch.Tensor\n+            Attention output, shape (B, Tq, D).\n+        \"\"\"\n+        if self.cross_attention:\n+            if context is None:\n+                raise ValueError(\"context is required for cross-attention\")\n+            q = self.q_proj(x)\n+            kv = self.kv_proj(context)\n+            k, v = torch.chunk(kv, 2, dim=-1)\n+        else:\n+            qkv = self.qkv_proj(x)\n+            q, k, v = torch.chunk(qkv, 3, dim=-1)\n+\n+        q = self._reshape_heads(q)\n+        k = self._reshape_heads(k)\n+        v = self._reshape_heads(v)\n+\n+        if self.rope is not None:\n+            cos_q, sin_q = self.rope.get_cos_sin(q.shape[2], q.device, q.dtype)\n+            cos_k, sin_k = self.rope.get_cos_sin(k.shape[2], k.device, k.dtype)\n+            q = (q * cos_q) + (_rotate_half(q) * sin_q)\n+            k = (k * cos_k) + (_rotate_half(k) * sin_k)\n+\n+        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n+        attn_probs = torch.softmax(attn_scores, dim=-1)\n+        attn_probs = self.attn_dropout(attn_probs)\n+\n+        out = torch.matmul(attn_probs, v)\n+        out = self._merge_heads(out)\n+        out = self.out_proj(out)\n+        out = self.out_dropout(out)\n+        return out\n+\n+\n+class FeedForward(nn.Module):\n+    \"\"\"Transformer-style feedforward block.\"\"\"\n+\n+    def __init__(self, d_model: int, dim_ff: int, dropout: float) -> None:\n+        super().__init__()\n+        self.net = nn.Sequential(\n+            nn.Linear(d_model, dim_ff),\n+            nn.GELU(),\n+            nn.Dropout(dropout),\n+            nn.Linear(dim_ff, d_model),\n+            nn.Dropout(dropout),\n+        )\n+\n+    def forward(self, x: torch.Tensor) -> torch.Tensor:\n+        return self.net(x)\n+\n+\n+class PerceiverLayer(nn.Module):\n+    \"\"\"\n+    One Perceiver layer: cross-attention + latent self-attention + FFN.\n+    \"\"\"\n+\n+    def __init__(\n+        self,\n+        d_model: int,\n+        cross_attn_heads: int,\n+        self_attn_heads: int,\n+        dim_ff: int,\n+        dropout: float,\n+        use_rope: bool,\n+        rope_base: float,\n+    ) -> None:\n+        super().__init__()\n+        self.norm_latent_cross = nn.LayerNorm(d_model)\n+        self.norm_input_cross = nn.LayerNorm(d_model)\n+        self.norm_latent_self = nn.LayerNorm(d_model)\n+        self.norm_ff = nn.LayerNorm(d_model)\n+\n+        self.cross_attn = MultiHeadAttention(\n+            d_model=d_model,\n+            nhead=cross_attn_heads,\n+            dropout=dropout,\n+            cross_attention=True,\n+            use_rope=use_rope,\n+            rope_base=rope_base,\n+        )\n+        self.self_attn = MultiHeadAttention(\n+            d_model=d_model,\n+            nhead=self_attn_heads,\n+            dropout=dropout,\n+            cross_attention=False,\n+            use_rope=use_rope,\n+            rope_base=rope_base,\n+        )\n+        self.ff = FeedForward(d_model=d_model, dim_ff=dim_ff, dropout=dropout)\n+\n+    def forward(self, latents: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:\n+        latents = latents + self.cross_attn(\n+            x=self.norm_latent_cross(latents),\n+            context=self.norm_input_cross(tokens),\n+        )\n+        latents = latents + self.self_attn(self.norm_latent_self(latents))\n+        latents = latents + self.ff(self.norm_ff(latents))\n+        return latents\n+\n+\n+# -- SpatioTemporalEncoder -----------------------------------------------------\n+\n+\n+class SpatioTemporalEncoder(nn.Module):\n+    \"\"\"\n+    End-to-end spatiotemporal encoder: GAT + temporal backbone.\n+\n+    Temporal backbones:\n+    - \"transformer\": baseline with CLS token and sinusoidal encoding.\n+    - \"perceiver\": learnable latent bottleneck; supports sinusoidal or RoPE.\n+    \"\"\"\n+\n+    def __init__(\n+        self,\n+        num_joints: int,\n+        in_features: int,\n+        edge_index: torch.Tensor,\n+        gat_hidden: int = 64,\n+        gat_heads: int = 4,\n+        gat_dropout: float = 0.1,\n+        d_model: int = 128,\n+        nhead: int = 4,\n+        num_layers: int = 2,\n+        dim_ff: int = 256,\n+        dropout: float = 0.1,\n+        seq_len: int = 60,\n+        cls_std: float = 0.02,\n+        temporal_backbone: str = \"transformer\",\n+        positional_encoding: str = \"sinusoidal\",\n+        num_latents: int = 64,\n+        num_perceiver_layers: int = 2,\n+        cross_attn_heads: int = 4,\n+        self_attn_heads: int = 4,\n+        perceiver_dim_ff: int = 256,\n+        perceiver_dropout: float = 0.1,\n+        rope_base: float = 10000.0,\n+    ) -> None:\n+        super().__init__()\n+\n+        self.num_joints = num_joints\n+        self.gat_hidden = gat_hidden\n+        self.gat_dropout = gat_dropout\n+        self.d_model = d_model\n+        self.temporal_backbone = temporal_backbone.lower()\n+        self.positional_encoding = positional_encoding.lower()\n+\n+        self.register_buffer(\"edge_index\", edge_index)\n+\n+        # Spatial GAT stack\n+        per_head_dim = gat_hidden // gat_heads\n+        self.gat1 = GATConv(\n+            in_features,\n+            per_head_dim,\n+            heads=gat_heads,\n+            concat=True,\n+            dropout=gat_dropout,\n+        )\n+        self.gat2 = GATConv(\n+            gat_hidden,\n+            per_head_dim,\n+            heads=gat_heads,\n+            concat=True,\n+            dropout=gat_dropout,\n+        )\n+        self.gat3 = GATConv(\n+            gat_hidden,\n+            gat_hidden,\n+            heads=1,\n+            concat=False,\n+            dropout=gat_dropout,\n+        )\n+\n+        self.bn1 = nn.BatchNorm1d(gat_hidden)\n+        self.bn2 = nn.BatchNorm1d(gat_hidden)\n+        self.bn3 = nn.BatchNorm1d(gat_hidden)\n+\n+        self.input_proj = nn.Linear(gat_hidden, d_model)\n+\n+        # Temporal backbone selection\n+        if self.temporal_backbone == \"transformer\":\n+            if self.positional_encoding != \"sinusoidal\":\n+                raise ValueError(\n+                    \"Transformer backbone currently supports only \"\n+                    \"positional_encoding='sinusoidal'.\"\n+                )\n+\n+            self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))\n+            nn.init.trunc_normal_(self.cls_token, std=cls_std)\n+\n+            self.pos_enc = PositionalEncoding(\n+                d_model=d_model,\n+                max_len=seq_len + 1,\n+                dropout=dropout,\n+            )\n+\n+            encoder_layer = nn.TransformerEncoderLayer(\n+                d_model=d_model,\n+                nhead=nhead,\n+                dim_feedforward=dim_ff,\n+                dropout=dropout,\n+                batch_first=True,\n+                norm_first=True,\n+                activation=\"gelu\",\n+            )\n+            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)\n+            self.norm = nn.LayerNorm(d_model)\n+\n+        elif self.temporal_backbone == \"perceiver\":\n+            if self.positional_encoding not in (\"sinusoidal\", \"rope\"):\n+                raise ValueError(\n+                    \"Perceiver backbone supports positional_encoding='sinusoidal' \"\n+                    \"or 'rope'.\"\n+                )\n+\n+            self.latents = nn.Parameter(torch.empty(1, num_latents, d_model))\n+            nn.init.trunc_normal_(self.latents, std=cls_std)\n+\n+            self.pos_enc = None\n+            if self.positional_encoding == \"sinusoidal\":\n+                self.pos_enc = PositionalEncoding(\n+                    d_model=d_model,\n+                    max_len=seq_len,\n+                    dropout=perceiver_dropout,\n+                )\n+\n+            use_rope = self.positional_encoding == \"rope\"\n+            self.perceiver_layers = nn.ModuleList(\n+                [\n+                    PerceiverLayer(\n+                        d_model=d_model,\n+                        cross_attn_heads=cross_attn_heads,\n+                        self_attn_heads=self_attn_heads,\n+                        dim_ff=perceiver_dim_ff,\n+                        dropout=perceiver_dropout,\n+                        use_rope=use_rope,\n+                        rope_base=rope_base,\n+                    )\n+                    for _ in range(num_perceiver_layers)\n+                ]\n+            )\n+            self.norm = nn.LayerNorm(d_model)\n+\n+        else:\n+            raise ValueError(\n+                f\"Unsupported temporal_backbone='{temporal_backbone}'. \"\n+                \"Use 'transformer' or 'perceiver'.\"\n+            )\n+\n+    def _gat_forward(self, x: torch.Tensor) -> torch.Tensor:\n+        \"\"\"Run all B*T graphs through GAT in one batched pass.\"\"\"\n+        B, T, J, Fin = x.shape\n+        num_graphs = B * T\n+\n+        x_nodes = x.reshape(num_graphs * J, Fin)\n+\n+        E = self.edge_index.shape[1]\n+        offsets = torch.arange(num_graphs, device=x.device) * J\n+        edge_batch = self.edge_index.unsqueeze(1).expand(-1, num_graphs, -1)\n+        edge_batch = edge_batch + offsets.view(1, -1, 1)\n+        edge_flat = edge_batch.reshape(2, num_graphs * E)\n+\n+        batch_tensor = torch.arange(num_graphs, device=x.device).repeat_interleave(J)\n+\n+        h = self.gat1(x_nodes, edge_flat)\n+        h = self.bn1(h)\n+        h = F.elu(h)\n+        h = F.dropout(h, p=self.gat_dropout, training=self.training)\n+\n+        h = self.gat2(h, edge_flat)\n+        h = self.bn2(h)\n+        h = F.elu(h)\n+        h = F.dropout(h, p=self.gat_dropout, training=self.training)\n+\n+        h = self.gat3(h, edge_flat)\n+        h = self.bn3(h)\n+        h = F.elu(h)\n+\n+        spatial = global_mean_pool(h, batch_tensor)\n+        return spatial.reshape(B, T, self.gat_hidden)\n+\n+    def _forward_transformer(self, h: torch.Tensor) -> torch.Tensor:\n+        \"\"\"Temporal forward for baseline transformer path.\"\"\"\n+        B = h.shape[0]\n+        cls = self.cls_token.expand(B, -1, -1)\n+        h = torch.cat([cls, h], dim=1)\n+        h = self.pos_enc(h)\n+        h = self.transformer(h)\n+        h = self.norm(h)\n+        return h[:, 0, :]\n+\n+    def _forward_perceiver(self, h: torch.Tensor) -> torch.Tensor:\n+        \"\"\"Temporal forward for Perceiver latent bottleneck path.\"\"\"\n+        if self.pos_enc is not None:\n+            h = self.pos_enc(h)\n+\n+        B = h.shape[0]\n+        latents = self.latents.expand(B, -1, -1)\n+        for layer in self.perceiver_layers:\n+            latents = layer(latents, h)\n+\n+        latents = self.norm(latents)\n+        return latents[:, 0, :]\n+\n+    def forward(self, x: torch.Tensor) -> torch.Tensor:\n+        \"\"\"\n+        Encode a batch of skeleton sequences.\n+\n+        Parameters\n+        ----------\n+        x : torch.Tensor\n+            Shape (B, T, J, F).\n+\n+        Returns\n+        -------\n+        cls_embedding : torch.Tensor\n+            Shape (B, d_model).\n+        \"\"\"\n+        spatial = self._gat_forward(x)\n+        h = self.input_proj(spatial)\n+\n+        if self.temporal_backbone == \"transformer\":\n+            return self._forward_transformer(h)\n+        return self._forward_perceiver(h)\n+\n+\n+# -- ActionClassifier ----------------------------------------------------------\n+\n+\n+class ActionClassifier(nn.Module):\n+    \"\"\"SpatioTemporalEncoder + linear classification head.\"\"\"\n+\n+    def __init__(self, encoder: SpatioTemporalEncoder, num_classes: int) -> None:\n+        super().__init__()\n+        self.encoder = encoder\n+        self.head = nn.Linear(encoder.d_model, num_classes)\n+\n+    def forward(self, x: torch.Tensor) -> torch.Tensor:\n+        cls_emb = self.encoder(x)\n+        return self.head(cls_emb)\n+\n+\n+# -- Factory helper ------------------------------------------------------------\n+\n+\n+def build_model(\n+    cfg: dict,\n+    edge_index: torch.Tensor,\n+) -> Tuple[SpatioTemporalEncoder, ActionClassifier]:\n+    \"\"\"Construct encoder and classifier from config.\"\"\"\n+    m = cfg[\"model\"]\n+    d = cfg[\"data\"]\n+\n+    encoder = SpatioTemporalEncoder(\n+        num_joints=d[\"num_joints\"],\n+        in_features=m[\"in_features\"],\n+        edge_index=edge_index,\n+        gat_hidden=m[\"gat_hidden_dim\"],\n+        gat_heads=m[\"gat_heads\"],\n+        gat_dropout=m[\"gat_dropout\"],\n+        d_model=m[\"d_model\"],\n+        nhead=m[\"nhead\"],\n+        num_layers=m[\"num_transformer_layers\"],\n+        dim_ff=m[\"dim_feedforward\"],\n+        dropout=m[\"transformer_dropout\"],\n+        seq_len=d[\"seq_len\"],\n+        cls_std=m[\"cls_std\"],\n+        temporal_backbone=m.get(\"temporal_backbone\", \"transformer\"),\n+        positional_encoding=m.get(\"positional_encoding\", \"sinusoidal\"),\n+        num_latents=m.get(\"num_latents\", 64),\n+        num_perceiver_layers=m.get(\"num_perceiver_layers\", 2),\n+        cross_attn_heads=m.get(\"cross_attn_heads\", m[\"nhead\"]),\n+        self_attn_heads=m.get(\"self_attn_heads\", m[\"nhead\"]),\n+        perceiver_dim_ff=m.get(\"perceiver_dim_feedforward\", m[\"dim_feedforward\"]),\n+        perceiver_dropout=m.get(\"perceiver_dropout\", m[\"transformer_dropout\"]),\n+        rope_base=m.get(\"rope_base\", 10000.0),\n+    )\n+    classifier = ActionClassifier(encoder, num_classes=d[\"num_classes\"])\n+    total = sum(p.numel() for p in classifier.parameters() if p.requires_grad)\n+    log.info(\"Model built: %d trainable parameters\", total)\n+    return encoder, classifier\n+\n+\n+# -- Quick sanity check --------------------------------------------------------\n+\n+\n+if __name__ == \"__main__\":\n+    from module1.data.joint_config import EDGE_INDEX, NUM_JOINTS\n+\n+    B, T, J, F_IN = 4, 60, NUM_JOINTS, 9\n+    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n+\n+    encoder = SpatioTemporalEncoder(\n+        num_joints=J,\n+        in_features=F_IN,\n+        edge_index=EDGE_INDEX,\n+        seq_len=T,\n+    ).to(device)\n+    classifier = ActionClassifier(encoder, num_classes=102).to(device)\n+    classifier.eval()\n+\n+    dummy = torch.randn(B, T, J, F_IN, device=device)\n+    with torch.no_grad():\n+        cls_emb = encoder(dummy)\n+        logits = classifier(dummy)\n+\n+    assert cls_emb.shape == (B, 128), f\"Expected (4,128), got {cls_emb.shape}\"\n+    assert logits.shape == (B, 102), f\"Expected (4,102), got {logits.shape}\"\n+    assert not torch.isnan(cls_emb).any(), \"NaN in encoder output\"\n+    assert not torch.isnan(logits).any(), \"NaN in classifier logits\"\n+\n+    n_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)\n+    print(f\"Encoder output shape    : {cls_emb.shape}\")\n+    print(f\"Classifier output shape : {logits.shape}\")\n+    print(f\"Trainable parameters    : {n_params:,}\")\n+    print(\"model.py OK.\")\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
M module1/pretrain/model.py

LSP errors detected in module1/pretrain/model.py, please fix:
<diagnostics file="/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py">
ERROR [24:8] Import "torch" could not be resolved
ERROR [25:8] Import "torch.nn" could not be resolved
ERROR [26:8] Import "torch.nn.functional" could not be resolved
ERROR [27:6] Import "torch_geometric.nn" could not be resolved
ERROR [482:13] Object of type "None" cannot be called
</diagnostics>
```

---

## Assistant (Build · gpt-5.3-codex · 8.6s)

_Thinking:_

**Identifying LSP error cause**

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py",
  "offset": 430,
  "limit": 120
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py</path>
<type>file</type>
<content>430:                         dropout=perceiver_dropout,
431:                         use_rope=use_rope,
432:                         rope_base=rope_base,
433:                     )
434:                     for _ in range(num_perceiver_layers)
435:                 ]
436:             )
437:             self.norm = nn.LayerNorm(d_model)
438: 
439:         else:
440:             raise ValueError(
441:                 f"Unsupported temporal_backbone='{temporal_backbone}'. "
442:                 "Use 'transformer' or 'perceiver'."
443:             )
444: 
445:     def _gat_forward(self, x: torch.Tensor) -> torch.Tensor:
446:         """Run all B*T graphs through GAT in one batched pass."""
447:         B, T, J, Fin = x.shape
448:         num_graphs = B * T
449: 
450:         x_nodes = x.reshape(num_graphs * J, Fin)
451: 
452:         E = self.edge_index.shape[1]
453:         offsets = torch.arange(num_graphs, device=x.device) * J
454:         edge_batch = self.edge_index.unsqueeze(1).expand(-1, num_graphs, -1)
455:         edge_batch = edge_batch + offsets.view(1, -1, 1)
456:         edge_flat = edge_batch.reshape(2, num_graphs * E)
457: 
458:         batch_tensor = torch.arange(num_graphs, device=x.device).repeat_interleave(J)
459: 
460:         h = self.gat1(x_nodes, edge_flat)
461:         h = self.bn1(h)
462:         h = F.elu(h)
463:         h = F.dropout(h, p=self.gat_dropout, training=self.training)
464: 
465:         h = self.gat2(h, edge_flat)
466:         h = self.bn2(h)
467:         h = F.elu(h)
468:         h = F.dropout(h, p=self.gat_dropout, training=self.training)
469: 
470:         h = self.gat3(h, edge_flat)
471:         h = self.bn3(h)
472:         h = F.elu(h)
473: 
474:         spatial = global_mean_pool(h, batch_tensor)
475:         return spatial.reshape(B, T, self.gat_hidden)
476: 
477:     def _forward_transformer(self, h: torch.Tensor) -> torch.Tensor:
478:         """Temporal forward for baseline transformer path."""
479:         B = h.shape[0]
480:         cls = self.cls_token.expand(B, -1, -1)
481:         h = torch.cat([cls, h], dim=1)
482:         h = self.pos_enc(h)
483:         h = self.transformer(h)
484:         h = self.norm(h)
485:         return h[:, 0, :]
486: 
487:     def _forward_perceiver(self, h: torch.Tensor) -> torch.Tensor:
488:         """Temporal forward for Perceiver latent bottleneck path."""
489:         if self.pos_enc is not None:
490:             h = self.pos_enc(h)
491: 
492:         B = h.shape[0]
493:         latents = self.latents.expand(B, -1, -1)
494:         for layer in self.perceiver_layers:
495:             latents = layer(latents, h)
496: 
497:         latents = self.norm(latents)
498:         return latents[:, 0, :]
499: 
500:     def forward(self, x: torch.Tensor) -> torch.Tensor:
501:         """
502:         Encode a batch of skeleton sequences.
503: 
504:         Parameters
505:         ----------
506:         x : torch.Tensor
507:             Shape (B, T, J, F).
508: 
509:         Returns
510:         -------
511:         cls_embedding : torch.Tensor
512:             Shape (B, d_model).
513:         """
514:         spatial = self._gat_forward(x)
515:         h = self.input_proj(spatial)
516: 
517:         if self.temporal_backbone == "transformer":
518:             return self._forward_transformer(h)
519:         return self._forward_perceiver(h)
520: 
521: 
522: # -- ActionClassifier ----------------------------------------------------------
523: 
524: 
525: class ActionClassifier(nn.Module):
526:     """SpatioTemporalEncoder + linear classification head."""
527: 
528:     def __init__(self, encoder: SpatioTemporalEncoder, num_classes: int) -> None:
529:         super().__init__()
530:         self.encoder = encoder
531:         self.head = nn.Linear(encoder.d_model, num_classes)
532: 
533:     def forward(self, x: torch.Tensor) -> torch.Tensor:
534:         cls_emb = self.encoder(x)
535:         return self.head(cls_emb)
536: 
537: 
538: # -- Factory helper ------------------------------------------------------------
539: 
540: 
541: def build_model(
542:     cfg: dict,
543:     edge_index: torch.Tensor,
544: ) -> Tuple[SpatioTemporalEncoder, ActionClassifier]:
545:     """Construct encoder and classifier from config."""
546:     m = cfg["model"]
547:     d = cfg["data"]
548: 
549:     encoder = SpatioTemporalEncoder(

(Showing lines 430-549 of 611. Use offset=550 to continue.)
</content>
```

---

## Assistant (Build · gpt-5.3-codex · 20.6s)

_Thinking:_

**Patching optional positional encoding error**

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Update File: /home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py\n@@\n         self.input_proj = nn.Linear(gat_hidden, d_model)\n+\n+        # Optional temporal modules are initialised per-backbone below.\n+        self.pos_enc: Optional[nn.Module] = None\n+        self.transformer: Optional[nn.Module] = None\n+        self.latents: Optional[nn.Parameter] = None\n+        self.perceiver_layers: Optional[nn.ModuleList] = None\n@@\n             self.pos_enc = PositionalEncoding(\n                 d_model=d_model,\n                 max_len=seq_len + 1,\n                 dropout=dropout,\n             )\n@@\n-            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)\n+            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)\n             self.norm = nn.LayerNorm(d_model)\n@@\n-            self.latents = nn.Parameter(torch.empty(1, num_latents, d_model))\n+            self.latents = nn.Parameter(torch.empty(1, num_latents, d_model))\n             nn.init.trunc_normal_(self.latents, std=cls_std)\n@@\n-            self.perceiver_layers = nn.ModuleList(\n+            self.perceiver_layers = nn.ModuleList(\n                 [\n                     PerceiverLayer(\n@@\n     def _forward_transformer(self, h: torch.Tensor) -> torch.Tensor:\n         \"\"\"Temporal forward for baseline transformer path.\"\"\"\n+        if self.pos_enc is None or self.transformer is None:\n+            raise RuntimeError(\"Transformer temporal modules are not initialised.\")\n         B = h.shape[0]\n         cls = self.cls_token.expand(B, -1, -1)\n         h = torch.cat([cls, h], dim=1)\n         h = self.pos_enc(h)\n         h = self.transformer(h)\n@@\n     def _forward_perceiver(self, h: torch.Tensor) -> torch.Tensor:\n         \"\"\"Temporal forward for Perceiver latent bottleneck path.\"\"\"\n+        if self.latents is None or self.perceiver_layers is None:\n+            raise RuntimeError(\"Perceiver temporal modules are not initialised.\")\n         if self.pos_enc is not None:\n             h = self.pos_enc(h)\n*** End Patch"
}
```

**Error:**
```
Error: apply_patch verification failed: Error: Failed to find expected lines in /home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py:
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
```

---

## Assistant (Build · gpt-5.3-codex · 6.1s)

_Thinking:_

**Investigating patch failure cause**

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py",
  "offset": 330,
  "limit": 140
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py</path>
<type>file</type>
<content>330:         rope_base: float = 10000.0,
331:     ) -> None:
332:         super().__init__()
333: 
334:         self.num_joints = num_joints
335:         self.gat_hidden = gat_hidden
336:         self.gat_dropout = gat_dropout
337:         self.d_model = d_model
338:         self.temporal_backbone = temporal_backbone.lower()
339:         self.positional_encoding = positional_encoding.lower()
340: 
341:         self.register_buffer("edge_index", edge_index)
342: 
343:         # Spatial GAT stack
344:         per_head_dim = gat_hidden // gat_heads
345:         self.gat1 = GATConv(
346:             in_features,
347:             per_head_dim,
348:             heads=gat_heads,
349:             concat=True,
350:             dropout=gat_dropout,
351:         )
352:         self.gat2 = GATConv(
353:             gat_hidden,
354:             per_head_dim,
355:             heads=gat_heads,
356:             concat=True,
357:             dropout=gat_dropout,
358:         )
359:         self.gat3 = GATConv(
360:             gat_hidden,
361:             gat_hidden,
362:             heads=1,
363:             concat=False,
364:             dropout=gat_dropout,
365:         )
366: 
367:         self.bn1 = nn.BatchNorm1d(gat_hidden)
368:         self.bn2 = nn.BatchNorm1d(gat_hidden)
369:         self.bn3 = nn.BatchNorm1d(gat_hidden)
370: 
371:         self.input_proj = nn.Linear(gat_hidden, d_model)
372: 
373:         # Temporal backbone selection
374:         if self.temporal_backbone == "transformer":
375:             if self.positional_encoding != "sinusoidal":
376:                 raise ValueError(
377:                     "Transformer backbone currently supports only "
378:                     "positional_encoding='sinusoidal'."
379:                 )
380: 
381:             self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
382:             nn.init.trunc_normal_(self.cls_token, std=cls_std)
383: 
384:             self.pos_enc = PositionalEncoding(
385:                 d_model=d_model,
386:                 max_len=seq_len + 1,
387:                 dropout=dropout,
388:             )
389: 
390:             encoder_layer = nn.TransformerEncoderLayer(
391:                 d_model=d_model,
392:                 nhead=nhead,
393:                 dim_feedforward=dim_ff,
394:                 dropout=dropout,
395:                 batch_first=True,
396:                 norm_first=True,
397:                 activation="gelu",
398:             )
399:             self.transformer = nn.TransformerEncoder(
400:                 encoder_layer, num_layers=num_layers
401:             )
402:             self.norm = nn.LayerNorm(d_model)
403: 
404:         elif self.temporal_backbone == "perceiver":
405:             if self.positional_encoding not in ("sinusoidal", "rope"):
406:                 raise ValueError(
407:                     "Perceiver backbone supports positional_encoding='sinusoidal' "
408:                     "or 'rope'."
409:                 )
410: 
411:             self.latents = nn.Parameter(torch.empty(1, num_latents, d_model))
412:             nn.init.trunc_normal_(self.latents, std=cls_std)
413: 
414:             self.pos_enc = None
415:             if self.positional_encoding == "sinusoidal":
416:                 self.pos_enc = PositionalEncoding(
417:                     d_model=d_model,
418:                     max_len=seq_len,
419:                     dropout=perceiver_dropout,
420:                 )
421: 
422:             use_rope = self.positional_encoding == "rope"
423:             self.perceiver_layers = nn.ModuleList(
424:                 [
425:                     PerceiverLayer(
426:                         d_model=d_model,
427:                         cross_attn_heads=cross_attn_heads,
428:                         self_attn_heads=self_attn_heads,
429:                         dim_ff=perceiver_dim_ff,
430:                         dropout=perceiver_dropout,
431:                         use_rope=use_rope,
432:                         rope_base=rope_base,
433:                     )
434:                     for _ in range(num_perceiver_layers)
435:                 ]
436:             )
437:             self.norm = nn.LayerNorm(d_model)
438: 
439:         else:
440:             raise ValueError(
441:                 f"Unsupported temporal_backbone='{temporal_backbone}'. "
442:                 "Use 'transformer' or 'perceiver'."
443:             )
444: 
445:     def _gat_forward(self, x: torch.Tensor) -> torch.Tensor:
446:         """Run all B*T graphs through GAT in one batched pass."""
447:         B, T, J, Fin = x.shape
448:         num_graphs = B * T
449: 
450:         x_nodes = x.reshape(num_graphs * J, Fin)
451: 
452:         E = self.edge_index.shape[1]
453:         offsets = torch.arange(num_graphs, device=x.device) * J
454:         edge_batch = self.edge_index.unsqueeze(1).expand(-1, num_graphs, -1)
455:         edge_batch = edge_batch + offsets.view(1, -1, 1)
456:         edge_flat = edge_batch.reshape(2, num_graphs * E)
457: 
458:         batch_tensor = torch.arange(num_graphs, device=x.device).repeat_interleave(J)
459: 
460:         h = self.gat1(x_nodes, edge_flat)
461:         h = self.bn1(h)
462:         h = F.elu(h)
463:         h = F.dropout(h, p=self.gat_dropout, training=self.training)
464: 
465:         h = self.gat2(h, edge_flat)
466:         h = self.bn2(h)
467:         h = F.elu(h)
468:         h = F.dropout(h, p=self.gat_dropout, training=self.training)
469: 

(Showing lines 330-469 of 611. Use offset=470 to continue.)
</content>
```

---

## Assistant (Build · gpt-5.3-codex · 12.7s)

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Update File: /home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py\n@@\n         self.bn3 = nn.BatchNorm1d(gat_hidden)\n \n         self.input_proj = nn.Linear(gat_hidden, d_model)\n+\n+        # Optional temporal modules initialised per-backbone.\n+        self.pos_enc: Optional[nn.Module] = None\n+        self.transformer: Optional[nn.Module] = None\n+        self.latents: Optional[nn.Parameter] = None\n+        self.perceiver_layers: Optional[nn.ModuleList] = None\n@@\n     def _forward_transformer(self, h: torch.Tensor) -> torch.Tensor:\n         \"\"\"Temporal forward for baseline transformer path.\"\"\"\n+        if self.pos_enc is None or self.transformer is None:\n+            raise RuntimeError(\"Transformer temporal modules are not initialised.\")\n         B = h.shape[0]\n         cls = self.cls_token.expand(B, -1, -1)\n         h = torch.cat([cls, h], dim=1)\n         h = self.pos_enc(h)\n         h = self.transformer(h)\n@@\n     def _forward_perceiver(self, h: torch.Tensor) -> torch.Tensor:\n         \"\"\"Temporal forward for Perceiver latent bottleneck path.\"\"\"\n+        if self.latents is None or self.perceiver_layers is None:\n+            raise RuntimeError(\"Perceiver temporal modules are not initialised.\")\n         if self.pos_enc is not None:\n             h = self.pos_enc(h)\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
M module1/pretrain/model.py

LSP errors detected in module1/pretrain/model.py, please fix:
<diagnostics file="/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py">
ERROR [24:8] Import "torch" could not be resolved
ERROR [25:8] Import "torch.nn" could not be resolved
ERROR [26:8] Import "torch.nn.functional" could not be resolved
ERROR [27:6] Import "torch_geometric.nn" could not be resolved
</diagnostics>
```

---

## Assistant (Build · gpt-5.3-codex · 14.3s)

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Add File: /home/alan/mydev/college/football-injury-ai/configs/pretrain_perceiver_rope.yaml\n+# configs/pretrain_perceiver_rope.yaml\n+#\n+# Stage 1 - ANUBIS Pretraining (Perceiver + RoPE)\n+# ------------------------------------------------\n+# Experimental architecture config:\n+#   - same data/training defaults as pretrain_final.yaml\n+#   - temporal backbone switched from Transformer to Perceiver latent bottleneck\n+#   - positional encoding switched to RoPE\n+\n+# -- Data ----------------------------------------------------------------------\n+data:\n+  anubis_dir:         data/anubis/\n+\n+  train_features:     trn_features_cleaned.npy\n+  train_labels:       trn_labels_cleaned.npy\n+  val_features:       val_features_cleaned.npy\n+  val_labels:         val_labels_cleaned.npy\n+\n+  has_separate_val:   true\n+  val_split:          0.15\n+\n+  num_classes:        102\n+  num_joints:         32\n+  clip_frames:        60\n+  seq_len:            60\n+\n+# -- Model ---------------------------------------------------------------------\n+model:\n+  in_features:                 9\n+  gat_hidden_dim:             64\n+  gat_heads:                   4\n+  gat_dropout:                 0.051065527924036666\n+  d_model:                   128\n+  nhead:                       4\n+  num_transformer_layers:      2\n+  dim_feedforward:           256\n+  transformer_dropout:         0.22439178858315353\n+  cls_std:                     0.02\n+\n+  temporal_backbone:         perceiver\n+  positional_encoding:       rope\n+  num_latents:               64\n+  num_perceiver_layers:       2\n+  cross_attn_heads:           4\n+  self_attn_heads:            4\n+  perceiver_dim_feedforward: 256\n+  perceiver_dropout:          0.22439178858315353\n+  rope_base:               10000.0\n+\n+# -- Training ------------------------------------------------------------------\n+training:\n+  epochs:                80\n+  batch_size:            64\n+  learning_rate:         0.0019464623176755585\n+  weight_decay:          0.00020433313066500611\n+  warmup_epochs:         5\n+  grad_clip_norm:        1.0\n+  label_smoothing:       0.040048660941805404\n+  save_every_n_epochs:   10\n+\n+# -- Augmentation --------------------------------------------------------------\n+augmentation:\n+  enabled:               true\n+  flip_prob:             0.39576557171375976\n+  joint_jitter_std:      0.0020494388932778985\n+  speed_perturb:         true\n+  speed_range:           [0.9, 1.1]\n+\n+# -- Paths ---------------------------------------------------------------------\n+paths:\n+  checkpoint_dir:  outputs/checkpoints/pretrain/\n+  log_dir:         outputs/logs/\n+\n+# -- Optuna (disabled for this fixed experiment config) ------------------------\n+optuna:\n+  enabled:          false\n+  study_name:       anubis_pretrain_hpo\n+  direction:        maximize\n+  n_trials:         30\n+  timeout_sec:      null\n+  trial_epochs:     25\n+  seed:             42\n+  num_workers:      4\n+  pin_memory:       true\n+  storage:          null\n+  load_if_exists:   true\n+\n+  output_best_params: outputs/logs/optuna_best_params.yaml\n+  output_best_config: outputs/logs/pretrain_optuna_best.yaml\n+\n+  search_space:\n+    learning_rate:\n+      low: 0.0001\n+      high: 0.002\n+      log: true\n+    weight_decay:\n+      low: 0.00001\n+      high: 0.001\n+      log: true\n+    label_smoothing:\n+      low: 0.0\n+      high: 0.15\n+    gat_dropout:\n+      low: 0.05\n+      high: 0.35\n+    transformer_dropout:\n+      low: 0.05\n+      high: 0.35\n+    flip_prob:\n+      low: 0.1\n+      high: 0.6\n+    joint_jitter_std:\n+      low: 0.001\n+      high: 0.02\n+      log: true\n+    speed_perturb:\n+      choices: [true, false]\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
A configs/pretrain_perceiver_rope.yaml
```

---

## Assistant (Build · gpt-5.3-codex · 20.1s)

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Update File: /home/alan/mydev/college/football-injury-ai/tests/test_model.py\n@@\n from module1.pretrain.model import (\n     ActionClassifier,\n+    MultiHeadAttention,\n     PositionalEncoding,\n+    RotaryEmbedding,\n     SpatioTemporalEncoder,\n     build_model,\n )\n@@\n class TestPositionalEncoding:\n@@\n         out = pe(x)\n         assert not torch.allclose(out[0, 0, :], out[0, 1, :])\n+\n+\n+class TestRotaryEmbedding:\n+    def test_cos_sin_shape(self) -> None:\n+        rope = RotaryEmbedding(dim=32, base=10000.0)\n+        cos, sin = rope.get_cos_sin(\n+            seq_len=10,\n+            device=torch.device(\"cpu\"),\n+            dtype=torch.float32,\n+        )\n+        assert cos.shape == (1, 1, 10, 32)\n+        assert sin.shape == (1, 1, 10, 32)\n+\n+    def test_rejects_odd_dimension(self) -> None:\n+        with pytest.raises(ValueError):\n+            _ = RotaryEmbedding(dim=31)\n+\n+\n+class TestMultiHeadAttention:\n+    @torch.no_grad()\n+    def test_self_attention_shape(self) -> None:\n+        attn = MultiHeadAttention(d_model=D_MODEL, nhead=4, dropout=0.0)\n+        x = torch.randn(B, T, D_MODEL)\n+        out = attn(x)\n+        assert out.shape == (B, T, D_MODEL)\n+\n+    @torch.no_grad()\n+    def test_cross_attention_shape(self) -> None:\n+        attn = MultiHeadAttention(\n+            d_model=D_MODEL,\n+            nhead=4,\n+            dropout=0.0,\n+            cross_attention=True,\n+            use_rope=True,\n+        )\n+        q = torch.randn(B, 8, D_MODEL)\n+        ctx = torch.randn(B, T, D_MODEL)\n+        out = attn(q, context=ctx)\n+        assert out.shape == (B, 8, D_MODEL)\n+\n+    def test_cross_attention_requires_context(self) -> None:\n+        attn = MultiHeadAttention(\n+            d_model=D_MODEL,\n+            nhead=4,\n+            dropout=0.0,\n+            cross_attention=True,\n+        )\n+        x = torch.randn(B, 8, D_MODEL)\n+        with pytest.raises(ValueError):\n+            _ = attn(x)\n@@\n class TestBuildModel:\n@@\n     def test_forward_pass_after_build(self, minimal_cfg: dict) -> None:\n         _, classifier = build_model(minimal_cfg, EDGE_INDEX)\n         classifier.eval()\n         dummy = torch.randn(B, T, J, F_IN)\n         logits = classifier(dummy)\n         assert logits.shape == (B, NUM_CLASSES)\n         assert not torch.isnan(logits).any()\n+\n+    @torch.no_grad()\n+    def test_perceiver_rope_forward_after_build(self, minimal_cfg: dict) -> None:\n+        cfg = {\n+            \"data\": {\n+                \"num_joints\": J,\n+                \"seq_len\": T,\n+                \"num_classes\": NUM_CLASSES,\n+            },\n+            \"model\": {\n+                \"in_features\": F_IN,\n+                \"gat_hidden_dim\": 64,\n+                \"gat_heads\": 4,\n+                \"gat_dropout\": 0.0,\n+                \"d_model\": D_MODEL,\n+                \"nhead\": 4,\n+                \"num_transformer_layers\": 2,\n+                \"dim_feedforward\": 256,\n+                \"transformer_dropout\": 0.0,\n+                \"cls_std\": 0.02,\n+                \"temporal_backbone\": \"perceiver\",\n+                \"positional_encoding\": \"rope\",\n+                \"num_latents\": 16,\n+                \"num_perceiver_layers\": 2,\n+                \"cross_attn_heads\": 4,\n+                \"self_attn_heads\": 4,\n+                \"perceiver_dim_feedforward\": 256,\n+                \"perceiver_dropout\": 0.0,\n+                \"rope_base\": 10000.0,\n+            },\n+        }\n+        _, classifier = build_model(cfg, EDGE_INDEX)\n+        classifier.eval()\n+        dummy = torch.randn(B, T, J, F_IN)\n+        logits = classifier(dummy)\n+        assert logits.shape == (B, NUM_CLASSES)\n+        assert not torch.isnan(logits).any()\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
M tests/test_model.py

LSP errors detected in tests/test_model.py, please fix:
<diagnostics file="/home/alan/mydev/college/football-injury-ai/tests/test_model.py">
ERROR [11:8] Import "pytest" could not be resolved
ERROR [12:8] Import "torch" could not be resolved
ERROR [13:8] Import "torch.nn" could not be resolved
</diagnostics>
```

---

## Assistant (Build · gpt-5.3-codex · 9.0s)

_Thinking:_

**Reviewing code changes and docs updates**

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py",
  "offset": 1,
  "limit": 260
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py</path>
<type>file</type>
<content>1: # module1/pretrain/model.py
2: #
3: # Stage 1 - SpatioTemporalEncoder + ActionClassifier
4: # ---------------------------------------------------
5: # End-to-end model that encodes a sequence of skeleton graphs.
6: #
7: # Input: (B, T, J, F) where F=9 [x,y,z,vx,vy,vz,ax,ay,az].
8: #
9: # Spatial stage (shared across all T frames):
10: #   GATConv -> BN -> ELU -> Dropout
11: #   GATConv -> BN -> ELU -> Dropout
12: #   GATConv -> BN -> ELU
13: #   global_mean_pool over joints -> (B, T, gat_hidden)
14: #
15: # Temporal stage (configurable):
16: #   1) Transformer (baseline): CLS + sinusoidal position + TransformerEncoder
17: #   2) Perceiver latent bottleneck: cross-attn + latent self-attn (+ optional RoPE)
18: 
19: from __future__ import annotations
20: 
21: import math
22: from typing import Optional, Tuple
23: 
24: import torch
25: import torch.nn as nn
26: import torch.nn.functional as F
27: from torch_geometric.nn import GATConv, global_mean_pool
28: 
29: from utils.logger import get_logger
30: 
31: log = get_logger(__name__)
32: 
33: 
34: # -- Positional encoding -------------------------------------------------------
35: 
36: 
37: class PositionalEncoding(nn.Module):
38:     """
39:     Sinusoidal positional encoding (Vaswani et al., 2017).
40: 
41:     Parameters
42:     ----------
43:     d_model : int
44:         Embedding dimension.
45:     max_len : int
46:         Maximum sequence length.
47:     dropout : float
48:         Dropout applied after adding positional encoding.
49:     """
50: 
51:     def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1) -> None:
52:         super().__init__()
53:         self.dropout = nn.Dropout(p=dropout)
54: 
55:         position = torch.arange(max_len).unsqueeze(1).float()
56:         div_term = torch.exp(
57:             torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
58:         )
59:         pe = torch.zeros(1, max_len, d_model)
60:         pe[0, :, 0::2] = torch.sin(position * div_term)
61:         pe[0, :, 1::2] = torch.cos(position * div_term)
62:         self.register_buffer("pe", pe)
63: 
64:     def forward(self, x: torch.Tensor) -> torch.Tensor:
65:         x = x + self.pe[:, : x.size(1), :]
66:         return self.dropout(x)
67: 
68: 
69: def _rotate_half(x: torch.Tensor) -> torch.Tensor:
70:     """Rotate even/odd features for rotary embedding."""
71:     x_even = x[..., ::2]
72:     x_odd = x[..., 1::2]
73:     x_rot = torch.stack((-x_odd, x_even), dim=-1)
74:     return x_rot.flatten(-2)
75: 
76: 
77: class RotaryEmbedding(nn.Module):
78:     """
79:     Rotary positional embedding helper for attention Q/K tensors.
80: 
81:     Parameters
82:     ----------
83:     dim : int
84:         Attention head dimension (must be even).
85:     base : float
86:         RoPE base frequency.
87:     """
88: 
89:     def __init__(self, dim: int, base: float = 10000.0) -> None:
90:         super().__init__()
91:         if dim % 2 != 0:
92:             raise ValueError(f"RoPE requires even head dimension, got dim={dim}")
93:         inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
94:         self.register_buffer("inv_freq", inv_freq)
95: 
96:     def get_cos_sin(
97:         self,
98:         seq_len: int,
99:         device: torch.device,
100:         dtype: torch.dtype,
101:     ) -> Tuple[torch.Tensor, torch.Tensor]:
102:         """Return cosine and sine tensors shaped for (B, H, T, D)."""
103:         t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
104:         freqs = torch.outer(t, self.inv_freq)  # (T, D/2)
105:         emb = torch.cat([freqs, freqs], dim=-1)  # (T, D)
106:         cos = emb.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
107:         sin = emb.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
108:         return cos, sin
109: 
110: 
111: # -- Attention blocks ----------------------------------------------------------
112: 
113: 
114: class MultiHeadAttention(nn.Module):
115:     """
116:     Multi-head attention supporting self-attention and cross-attention.
117: 
118:     Parameters
119:     ----------
120:     d_model : int
121:         Embedding dimension.
122:     nhead : int
123:         Number of attention heads.
124:     dropout : float
125:         Dropout on attention probabilities and output projection.
126:     cross_attention : bool
127:         If True, query comes from x and key/value comes from context.
128:     use_rope : bool
129:         If True, apply RoPE to q/k tensors.
130:     rope_base : float
131:         RoPE base frequency.
132:     """
133: 
134:     def __init__(
135:         self,
136:         d_model: int,
137:         nhead: int,
138:         dropout: float = 0.1,
139:         cross_attention: bool = False,
140:         use_rope: bool = False,
141:         rope_base: float = 10000.0,
142:     ) -> None:
143:         super().__init__()
144:         if d_model % nhead != 0:
145:             raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
146: 
147:         self.d_model = d_model
148:         self.nhead = nhead
149:         self.head_dim = d_model // nhead
150:         self.scale = self.head_dim**-0.5
151:         self.cross_attention = cross_attention
152:         self.use_rope = use_rope
153: 
154:         if cross_attention:
155:             self.q_proj = nn.Linear(d_model, d_model)
156:             self.kv_proj = nn.Linear(d_model, 2 * d_model)
157:         else:
158:             self.qkv_proj = nn.Linear(d_model, 3 * d_model)
159: 
160:         self.out_proj = nn.Linear(d_model, d_model)
161:         self.attn_dropout = nn.Dropout(dropout)
162:         self.out_dropout = nn.Dropout(dropout)
163: 
164:         if use_rope:
165:             self.rope = RotaryEmbedding(self.head_dim, base=rope_base)
166:         else:
167:             self.rope = None
168: 
169:     def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
170:         """(B, T, D) -> (B, H, T, Dh)"""
171:         B, T, _ = x.shape
172:         x = x.view(B, T, self.nhead, self.head_dim)
173:         return x.permute(0, 2, 1, 3)
174: 
175:     def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
176:         """(B, H, T, Dh) -> (B, T, D)"""
177:         B, H, T, Dh = x.shape
178:         return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * Dh)
179: 
180:     def forward(
181:         self,
182:         x: torch.Tensor,
183:         context: Optional[torch.Tensor] = None,
184:     ) -> torch.Tensor:
185:         """
186:         Parameters
187:         ----------
188:         x : torch.Tensor
189:             Query tensor, shape (B, Tq, D).
190:         context : torch.Tensor, optional
191:             Key/value source, shape (B, Tk, D). Required for cross-attention.
192: 
193:         Returns
194:         -------
195:         out : torch.Tensor
196:             Attention output, shape (B, Tq, D).
197:         """
198:         if self.cross_attention:
199:             if context is None:
200:                 raise ValueError("context is required for cross-attention")
201:             q = self.q_proj(x)
202:             kv = self.kv_proj(context)
203:             k, v = torch.chunk(kv, 2, dim=-1)
204:         else:
205:             qkv = self.qkv_proj(x)
206:             q, k, v = torch.chunk(qkv, 3, dim=-1)
207: 
208:         q = self._reshape_heads(q)
209:         k = self._reshape_heads(k)
210:         v = self._reshape_heads(v)
211: 
212:         if self.rope is not None:
213:             cos_q, sin_q = self.rope.get_cos_sin(q.shape[2], q.device, q.dtype)
214:             cos_k, sin_k = self.rope.get_cos_sin(k.shape[2], k.device, k.dtype)
215:             q = (q * cos_q) + (_rotate_half(q) * sin_q)
216:             k = (k * cos_k) + (_rotate_half(k) * sin_k)
217: 
218:         attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
219:         attn_probs = torch.softmax(attn_scores, dim=-1)
220:         attn_probs = self.attn_dropout(attn_probs)
221: 
222:         out = torch.matmul(attn_probs, v)
223:         out = self._merge_heads(out)
224:         out = self.out_proj(out)
225:         out = self.out_dropout(out)
226:         return out
227: 
228: 
229: class FeedForward(nn.Module):
230:     """Transformer-style feedforward block."""
231: 
232:     def __init__(self, d_model: int, dim_ff: int, dropout: float) -> None:
233:         super().__init__()
234:         self.net = nn.Sequential(
235:             nn.Linear(d_model, dim_ff),
236:             nn.GELU(),
237:             nn.Dropout(dropout),
238:             nn.Linear(dim_ff, d_model),
239:             nn.Dropout(dropout),
240:         )
241: 
242:     def forward(self, x: torch.Tensor) -> torch.Tensor:
243:         return self.net(x)
244: 
245: 
246: class PerceiverLayer(nn.Module):
247:     """
248:     One Perceiver layer: cross-attention + latent self-attention + FFN.
249:     """
250: 
251:     def __init__(
252:         self,
253:         d_model: int,
254:         cross_attn_heads: int,
255:         self_attn_heads: int,
256:         dim_ff: int,
257:         dropout: float,
258:         use_rope: bool,
259:         rope_base: float,
260:     ) -> None:

(Showing lines 1-260 of 621. Use offset=261 to continue.)
</content>
```

---

## Assistant (Build · gpt-5.3-codex · 6.0s)

_Thinking:_



**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py",
  "offset": 261,
  "limit": 380
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py</path>
<type>file</type>
<content>261:         super().__init__()
262:         self.norm_latent_cross = nn.LayerNorm(d_model)
263:         self.norm_input_cross = nn.LayerNorm(d_model)
264:         self.norm_latent_self = nn.LayerNorm(d_model)
265:         self.norm_ff = nn.LayerNorm(d_model)
266: 
267:         self.cross_attn = MultiHeadAttention(
268:             d_model=d_model,
269:             nhead=cross_attn_heads,
270:             dropout=dropout,
271:             cross_attention=True,
272:             use_rope=use_rope,
273:             rope_base=rope_base,
274:         )
275:         self.self_attn = MultiHeadAttention(
276:             d_model=d_model,
277:             nhead=self_attn_heads,
278:             dropout=dropout,
279:             cross_attention=False,
280:             use_rope=use_rope,
281:             rope_base=rope_base,
282:         )
283:         self.ff = FeedForward(d_model=d_model, dim_ff=dim_ff, dropout=dropout)
284: 
285:     def forward(self, latents: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
286:         latents = latents + self.cross_attn(
287:             x=self.norm_latent_cross(latents),
288:             context=self.norm_input_cross(tokens),
289:         )
290:         latents = latents + self.self_attn(self.norm_latent_self(latents))
291:         latents = latents + self.ff(self.norm_ff(latents))
292:         return latents
293: 
294: 
295: # -- SpatioTemporalEncoder -----------------------------------------------------
296: 
297: 
298: class SpatioTemporalEncoder(nn.Module):
299:     """
300:     End-to-end spatiotemporal encoder: GAT + temporal backbone.
301: 
302:     Temporal backbones:
303:     - "transformer": baseline with CLS token and sinusoidal encoding.
304:     - "perceiver": learnable latent bottleneck; supports sinusoidal or RoPE.
305:     """
306: 
307:     def __init__(
308:         self,
309:         num_joints: int,
310:         in_features: int,
311:         edge_index: torch.Tensor,
312:         gat_hidden: int = 64,
313:         gat_heads: int = 4,
314:         gat_dropout: float = 0.1,
315:         d_model: int = 128,
316:         nhead: int = 4,
317:         num_layers: int = 2,
318:         dim_ff: int = 256,
319:         dropout: float = 0.1,
320:         seq_len: int = 60,
321:         cls_std: float = 0.02,
322:         temporal_backbone: str = "transformer",
323:         positional_encoding: str = "sinusoidal",
324:         num_latents: int = 64,
325:         num_perceiver_layers: int = 2,
326:         cross_attn_heads: int = 4,
327:         self_attn_heads: int = 4,
328:         perceiver_dim_ff: int = 256,
329:         perceiver_dropout: float = 0.1,
330:         rope_base: float = 10000.0,
331:     ) -> None:
332:         super().__init__()
333: 
334:         self.num_joints = num_joints
335:         self.gat_hidden = gat_hidden
336:         self.gat_dropout = gat_dropout
337:         self.d_model = d_model
338:         self.temporal_backbone = temporal_backbone.lower()
339:         self.positional_encoding = positional_encoding.lower()
340: 
341:         self.register_buffer("edge_index", edge_index)
342: 
343:         # Spatial GAT stack
344:         per_head_dim = gat_hidden // gat_heads
345:         self.gat1 = GATConv(
346:             in_features,
347:             per_head_dim,
348:             heads=gat_heads,
349:             concat=True,
350:             dropout=gat_dropout,
351:         )
352:         self.gat2 = GATConv(
353:             gat_hidden,
354:             per_head_dim,
355:             heads=gat_heads,
356:             concat=True,
357:             dropout=gat_dropout,
358:         )
359:         self.gat3 = GATConv(
360:             gat_hidden,
361:             gat_hidden,
362:             heads=1,
363:             concat=False,
364:             dropout=gat_dropout,
365:         )
366: 
367:         self.bn1 = nn.BatchNorm1d(gat_hidden)
368:         self.bn2 = nn.BatchNorm1d(gat_hidden)
369:         self.bn3 = nn.BatchNorm1d(gat_hidden)
370: 
371:         self.input_proj = nn.Linear(gat_hidden, d_model)
372: 
373:         # Optional temporal modules initialised per-backbone.
374:         self.pos_enc: Optional[nn.Module] = None
375:         self.transformer: Optional[nn.Module] = None
376:         self.latents: Optional[nn.Parameter] = None
377:         self.perceiver_layers: Optional[nn.ModuleList] = None
378: 
379:         # Temporal backbone selection
380:         if self.temporal_backbone == "transformer":
381:             if self.positional_encoding != "sinusoidal":
382:                 raise ValueError(
383:                     "Transformer backbone currently supports only "
384:                     "positional_encoding='sinusoidal'."
385:                 )
386: 
387:             self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
388:             nn.init.trunc_normal_(self.cls_token, std=cls_std)
389: 
390:             self.pos_enc = PositionalEncoding(
391:                 d_model=d_model,
392:                 max_len=seq_len + 1,
393:                 dropout=dropout,
394:             )
395: 
396:             encoder_layer = nn.TransformerEncoderLayer(
397:                 d_model=d_model,
398:                 nhead=nhead,
399:                 dim_feedforward=dim_ff,
400:                 dropout=dropout,
401:                 batch_first=True,
402:                 norm_first=True,
403:                 activation="gelu",
404:             )
405:             self.transformer = nn.TransformerEncoder(
406:                 encoder_layer, num_layers=num_layers
407:             )
408:             self.norm = nn.LayerNorm(d_model)
409: 
410:         elif self.temporal_backbone == "perceiver":
411:             if self.positional_encoding not in ("sinusoidal", "rope"):
412:                 raise ValueError(
413:                     "Perceiver backbone supports positional_encoding='sinusoidal' "
414:                     "or 'rope'."
415:                 )
416: 
417:             self.latents = nn.Parameter(torch.empty(1, num_latents, d_model))
418:             nn.init.trunc_normal_(self.latents, std=cls_std)
419: 
420:             self.pos_enc = None
421:             if self.positional_encoding == "sinusoidal":
422:                 self.pos_enc = PositionalEncoding(
423:                     d_model=d_model,
424:                     max_len=seq_len,
425:                     dropout=perceiver_dropout,
426:                 )
427: 
428:             use_rope = self.positional_encoding == "rope"
429:             self.perceiver_layers = nn.ModuleList(
430:                 [
431:                     PerceiverLayer(
432:                         d_model=d_model,
433:                         cross_attn_heads=cross_attn_heads,
434:                         self_attn_heads=self_attn_heads,
435:                         dim_ff=perceiver_dim_ff,
436:                         dropout=perceiver_dropout,
437:                         use_rope=use_rope,
438:                         rope_base=rope_base,
439:                     )
440:                     for _ in range(num_perceiver_layers)
441:                 ]
442:             )
443:             self.norm = nn.LayerNorm(d_model)
444: 
445:         else:
446:             raise ValueError(
447:                 f"Unsupported temporal_backbone='{temporal_backbone}'. "
448:                 "Use 'transformer' or 'perceiver'."
449:             )
450: 
451:     def _gat_forward(self, x: torch.Tensor) -> torch.Tensor:
452:         """Run all B*T graphs through GAT in one batched pass."""
453:         B, T, J, Fin = x.shape
454:         num_graphs = B * T
455: 
456:         x_nodes = x.reshape(num_graphs * J, Fin)
457: 
458:         E = self.edge_index.shape[1]
459:         offsets = torch.arange(num_graphs, device=x.device) * J
460:         edge_batch = self.edge_index.unsqueeze(1).expand(-1, num_graphs, -1)
461:         edge_batch = edge_batch + offsets.view(1, -1, 1)
462:         edge_flat = edge_batch.reshape(2, num_graphs * E)
463: 
464:         batch_tensor = torch.arange(num_graphs, device=x.device).repeat_interleave(J)
465: 
466:         h = self.gat1(x_nodes, edge_flat)
467:         h = self.bn1(h)
468:         h = F.elu(h)
469:         h = F.dropout(h, p=self.gat_dropout, training=self.training)
470: 
471:         h = self.gat2(h, edge_flat)
472:         h = self.bn2(h)
473:         h = F.elu(h)
474:         h = F.dropout(h, p=self.gat_dropout, training=self.training)
475: 
476:         h = self.gat3(h, edge_flat)
477:         h = self.bn3(h)
478:         h = F.elu(h)
479: 
480:         spatial = global_mean_pool(h, batch_tensor)
481:         return spatial.reshape(B, T, self.gat_hidden)
482: 
483:     def _forward_transformer(self, h: torch.Tensor) -> torch.Tensor:
484:         """Temporal forward for baseline transformer path."""
485:         if self.pos_enc is None or self.transformer is None:
486:             raise RuntimeError("Transformer temporal modules are not initialised.")
487:         B = h.shape[0]
488:         cls = self.cls_token.expand(B, -1, -1)
489:         h = torch.cat([cls, h], dim=1)
490:         h = self.pos_enc(h)
491:         h = self.transformer(h)
492:         h = self.norm(h)
493:         return h[:, 0, :]
494: 
495:     def _forward_perceiver(self, h: torch.Tensor) -> torch.Tensor:
496:         """Temporal forward for Perceiver latent bottleneck path."""
497:         if self.latents is None or self.perceiver_layers is None:
498:             raise RuntimeError("Perceiver temporal modules are not initialised.")
499:         if self.pos_enc is not None:
500:             h = self.pos_enc(h)
501: 
502:         B = h.shape[0]
503:         latents = self.latents.expand(B, -1, -1)
504:         for layer in self.perceiver_layers:
505:             latents = layer(latents, h)
506: 
507:         latents = self.norm(latents)
508:         return latents[:, 0, :]
509: 
510:     def forward(self, x: torch.Tensor) -> torch.Tensor:
511:         """
512:         Encode a batch of skeleton sequences.
513: 
514:         Parameters
515:         ----------
516:         x : torch.Tensor
517:             Shape (B, T, J, F).
518: 
519:         Returns
520:         -------
521:         cls_embedding : torch.Tensor
522:             Shape (B, d_model).
523:         """
524:         spatial = self._gat_forward(x)
525:         h = self.input_proj(spatial)
526: 
527:         if self.temporal_backbone == "transformer":
528:             return self._forward_transformer(h)
529:         return self._forward_perceiver(h)
530: 
531: 
532: # -- ActionClassifier ----------------------------------------------------------
533: 
534: 
535: class ActionClassifier(nn.Module):
536:     """SpatioTemporalEncoder + linear classification head."""
537: 
538:     def __init__(self, encoder: SpatioTemporalEncoder, num_classes: int) -> None:
539:         super().__init__()
540:         self.encoder = encoder
541:         self.head = nn.Linear(encoder.d_model, num_classes)
542: 
543:     def forward(self, x: torch.Tensor) -> torch.Tensor:
544:         cls_emb = self.encoder(x)
545:         return self.head(cls_emb)
546: 
547: 
548: # -- Factory helper ------------------------------------------------------------
549: 
550: 
551: def build_model(
552:     cfg: dict,
553:     edge_index: torch.Tensor,
554: ) -> Tuple[SpatioTemporalEncoder, ActionClassifier]:
555:     """Construct encoder and classifier from config."""
556:     m = cfg["model"]
557:     d = cfg["data"]
558: 
559:     encoder = SpatioTemporalEncoder(
560:         num_joints=d["num_joints"],
561:         in_features=m["in_features"],
562:         edge_index=edge_index,
563:         gat_hidden=m["gat_hidden_dim"],
564:         gat_heads=m["gat_heads"],
565:         gat_dropout=m["gat_dropout"],
566:         d_model=m["d_model"],
567:         nhead=m["nhead"],
568:         num_layers=m["num_transformer_layers"],
569:         dim_ff=m["dim_feedforward"],
570:         dropout=m["transformer_dropout"],
571:         seq_len=d["seq_len"],
572:         cls_std=m["cls_std"],
573:         temporal_backbone=m.get("temporal_backbone", "transformer"),
574:         positional_encoding=m.get("positional_encoding", "sinusoidal"),
575:         num_latents=m.get("num_latents", 64),
576:         num_perceiver_layers=m.get("num_perceiver_layers", 2),
577:         cross_attn_heads=m.get("cross_attn_heads", m["nhead"]),
578:         self_attn_heads=m.get("self_attn_heads", m["nhead"]),
579:         perceiver_dim_ff=m.get("perceiver_dim_feedforward", m["dim_feedforward"]),
580:         perceiver_dropout=m.get("perceiver_dropout", m["transformer_dropout"]),
581:         rope_base=m.get("rope_base", 10000.0),
582:     )
583:     classifier = ActionClassifier(encoder, num_classes=d["num_classes"])
584:     total = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
585:     log.info("Model built: %d trainable parameters", total)
586:     return encoder, classifier
587: 
588: 
589: # -- Quick sanity check --------------------------------------------------------
590: 
591: 
592: if __name__ == "__main__":
593:     from module1.data.joint_config import EDGE_INDEX, NUM_JOINTS
594: 
595:     B, T, J, F_IN = 4, 60, NUM_JOINTS, 9
596:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
597: 
598:     encoder = SpatioTemporalEncoder(
599:         num_joints=J,
600:         in_features=F_IN,
601:         edge_index=EDGE_INDEX,
602:         seq_len=T,
603:     ).to(device)
604:     classifier = ActionClassifier(encoder, num_classes=102).to(device)
605:     classifier.eval()
606: 
607:     dummy = torch.randn(B, T, J, F_IN, device=device)
608:     with torch.no_grad():
609:         cls_emb = encoder(dummy)
610:         logits = classifier(dummy)
611: 
612:     assert cls_emb.shape == (B, 128), f"Expected (4,128), got {cls_emb.shape}"
613:     assert logits.shape == (B, 102), f"Expected (4,102), got {logits.shape}"
614:     assert not torch.isnan(cls_emb).any(), "NaN in encoder output"
615:     assert not torch.isnan(logits).any(), "NaN in classifier logits"
616: 
617:     n_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
618:     print(f"Encoder output shape    : {cls_emb.shape}")
619:     print(f"Classifier output shape : {logits.shape}")
620:     print(f"Trainable parameters    : {n_params:,}")
621:     print("model.py OK.")

(End of file - total 621 lines)
</content>
```

---

## Assistant (Build · gpt-5.3-codex · 12.1s)

_Thinking:_

**Finalizing imports and configs**

**Tool: bash**

**Input:**
```json
{
  "command": "python -m py_compile module1/pretrain/model.py tests/test_model.py module1/pretrain/live_inference.py module1/pretrain/live_utils.py module1/pose/extractor.py",
  "description": "Compile modified Python files for syntax"
}
```

**Tool: bash**

**Input:**
```json
{
  "command": "python - <<'PY'\nfrom pathlib import Path\nfiles=[\n'./module1/pretrain/model.py',\n'./tests/test_model.py',\n'./configs/pretrain_perceiver_rope.yaml',\n]\nfor f in files:\n    text=Path(f).read_text(encoding='utf-8')\n    bad=sorted({ch for ch in text if ord(ch)>127})\n    print(f, 'non-ascii', bad)\nPY",
  "description": "Check changed files for non-ASCII chars"
}
```

**Output:**
```
./module1/pretrain/model.py non-ascii []
./tests/test_model.py non-ascii ['—', '─']
./configs/pretrain_perceiver_rope.yaml non-ascii []

```

---

## Assistant (Build · gpt-5.3-codex · 7.5s)

_Thinking:_

**Assessing non-ASCII use and doc updates**

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/tests/test_model.py",
  "offset": 1,
  "limit": 360
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/tests/test_model.py</path>
<type>file</type>
<content>1: # tests/test_model.py
2: #
3: # Unit tests for module1/pretrain/model.py
4: # ──────────────────────────────────────────
5: # Covers: PositionalEncoding, SpatioTemporalEncoder (forward pass shapes,
6: #         no NaNs, edge_index buffer device movement), ActionClassifier
7: #         (forward pass, logit shape), and encoder/classifier separation.
8: #
9: # Run with: pytest tests/test_model.py -v
10: 
11: import pytest
12: import torch
13: import torch.nn as nn
14: 
15: from module1.data.joint_config import EDGE_INDEX, NUM_JOINTS
16: from module1.pretrain.model import (
17:     ActionClassifier,
18:     MultiHeadAttention,
19:     PositionalEncoding,
20:     RotaryEmbedding,
21:     SpatioTemporalEncoder,
22:     build_model,
23: )
24: 
25: 
26: # ── Constants used across tests ───────────────────────────────────────────────
27: 
28: B = 2  # batch size (small to keep tests fast)
29: T = 60  # sequence length (frames)
30: J = NUM_JOINTS  # 32
31: F_IN = 9  # features per joint
32: D_MODEL = 128
33: NUM_CLASSES = 102
34: 
35: 
36: # ── Fixtures ──────────────────────────────────────────────────────────────────
37: 
38: 
39: @pytest.fixture
40: def edge_index() -> torch.Tensor:
41:     return EDGE_INDEX.clone()
42: 
43: 
44: @pytest.fixture
45: def encoder(edge_index: torch.Tensor) -> SpatioTemporalEncoder:
46:     model = SpatioTemporalEncoder(
47:         num_joints=J,
48:         in_features=F_IN,
49:         edge_index=edge_index,
50:         gat_hidden=64,
51:         gat_heads=4,
52:         d_model=D_MODEL,
53:         nhead=4,
54:         num_layers=2,
55:         dim_ff=256,
56:         dropout=0.0,  # disable dropout so outputs are deterministic in eval
57:         seq_len=T,
58:     )
59:     model.eval()
60:     return model
61: 
62: 
63: @pytest.fixture
64: def classifier(encoder: SpatioTemporalEncoder) -> ActionClassifier:
65:     model = ActionClassifier(encoder, num_classes=NUM_CLASSES)
66:     model.eval()
67:     return model
68: 
69: 
70: @pytest.fixture
71: def dummy_input() -> torch.Tensor:
72:     """Small random batch: (B, T, J, F_IN)."""
73:     torch.manual_seed(0)
74:     return torch.randn(B, T, J, F_IN)
75: 
76: 
77: @pytest.fixture
78: def minimal_cfg() -> dict:
79:     """Minimal config dict that mirrors configs/pretrain.yaml structure."""
80:     return {
81:         "data": {
82:             "num_joints": J,
83:             "seq_len": T,
84:             "num_classes": NUM_CLASSES,
85:         },
86:         "model": {
87:             "in_features": F_IN,
88:             "gat_hidden_dim": 64,
89:             "gat_heads": 4,
90:             "gat_dropout": 0.0,
91:             "d_model": D_MODEL,
92:             "nhead": 4,
93:             "num_transformer_layers": 2,
94:             "dim_feedforward": 256,
95:             "transformer_dropout": 0.0,
96:             "cls_std": 0.02,
97:         },
98:     }
99: 
100: 
101: # ── PositionalEncoding ────────────────────────────────────────────────────────
102: 
103: 
104: class TestPositionalEncoding:
105:     def test_output_shape(self) -> None:
106:         pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
107:         x = torch.zeros(B, T, D_MODEL)
108:         out = pe(x)
109:         assert out.shape == (B, T, D_MODEL)
110: 
111:     def test_no_nan(self) -> None:
112:         pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
113:         x = torch.randn(B, T, D_MODEL)
114:         out = pe(x)
115:         assert not torch.isnan(out).any()
116: 
117:     def test_adds_positional_signal(self) -> None:
118:         # Zero input + no dropout: output should equal the PE buffer, not zero
119:         pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
120:         pe.eval()
121:         x = torch.zeros(B, T, D_MODEL)
122:         out = pe(x)
123:         assert not torch.all(out == 0)
124: 
125:     def test_different_positions_differ(self) -> None:
126:         # Adjacent time steps should (almost certainly) have different encodings
127:         pe = PositionalEncoding(d_model=D_MODEL, max_len=T + 1, dropout=0.0)
128:         pe.eval()
129:         x = torch.zeros(1, T, D_MODEL)
130:         out = pe(x)
131:         assert not torch.allclose(out[0, 0, :], out[0, 1, :])
132: 
133: 
134: class TestRotaryEmbedding:
135:     def test_cos_sin_shape(self) -> None:
136:         rope = RotaryEmbedding(dim=32, base=10000.0)
137:         cos, sin = rope.get_cos_sin(
138:             seq_len=10,
139:             device=torch.device("cpu"),
140:             dtype=torch.float32,
141:         )
142:         assert cos.shape == (1, 1, 10, 32)
143:         assert sin.shape == (1, 1, 10, 32)
144: 
145:     def test_rejects_odd_dimension(self) -> None:
146:         with pytest.raises(ValueError):
147:             _ = RotaryEmbedding(dim=31)
148: 
149: 
150: class TestMultiHeadAttention:
151:     @torch.no_grad()
152:     def test_self_attention_shape(self) -> None:
153:         attn = MultiHeadAttention(d_model=D_MODEL, nhead=4, dropout=0.0)
154:         x = torch.randn(B, T, D_MODEL)
155:         out = attn(x)
156:         assert out.shape == (B, T, D_MODEL)
157: 
158:     @torch.no_grad()
159:     def test_cross_attention_shape(self) -> None:
160:         attn = MultiHeadAttention(
161:             d_model=D_MODEL,
162:             nhead=4,
163:             dropout=0.0,
164:             cross_attention=True,
165:             use_rope=True,
166:         )
167:         q = torch.randn(B, 8, D_MODEL)
168:         ctx = torch.randn(B, T, D_MODEL)
169:         out = attn(q, context=ctx)
170:         assert out.shape == (B, 8, D_MODEL)
171: 
172:     def test_cross_attention_requires_context(self) -> None:
173:         attn = MultiHeadAttention(
174:             d_model=D_MODEL,
175:             nhead=4,
176:             dropout=0.0,
177:             cross_attention=True,
178:         )
179:         x = torch.randn(B, 8, D_MODEL)
180:         with pytest.raises(ValueError):
181:             _ = attn(x)
182: 
183: 
184: # ── SpatioTemporalEncoder ─────────────────────────────────────────────────────
185: 
186: 
187: class TestSpatioTemporalEncoder:
188:     @torch.no_grad()
189:     def test_output_shape(
190:         self, encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
191:     ) -> None:
192:         out = encoder(dummy_input)
193:         assert out.shape == (B, D_MODEL), f"Expected ({B},{D_MODEL}), got {out.shape}"
194: 
195:     @torch.no_grad()
196:     def test_no_nan_in_output(
197:         self, encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
198:     ) -> None:
199:         out = encoder(dummy_input)
200:         assert not torch.isnan(out).any(), "NaN detected in encoder output"
201: 
202:     @torch.no_grad()
203:     def test_no_inf_in_output(
204:         self, encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
205:     ) -> None:
206:         out = encoder(dummy_input)
207:         assert not torch.isinf(out).any(), "Inf detected in encoder output"
208: 
209:     @torch.no_grad()
210:     def test_different_inputs_give_different_outputs(
211:         self, encoder: SpatioTemporalEncoder
212:     ) -> None:
213:         torch.manual_seed(1)
214:         x1 = torch.randn(B, T, J, F_IN)
215:         x2 = torch.randn(B, T, J, F_IN)
216:         out1 = encoder(x1)
217:         out2 = encoder(x2)
218:         assert not torch.allclose(out1, out2), "Encoder is constant regardless of input"
219: 
220:     @torch.no_grad()
221:     def test_batch_independence(self, encoder: SpatioTemporalEncoder) -> None:
222:         # Encoding one sample individually must equal its row in a batched pass
223:         torch.manual_seed(2)
224:         x = torch.randn(B, T, J, F_IN)
225:         batch_out = encoder(x)
226:         single_out = encoder(x[0:1])
227:         assert torch.allclose(batch_out[0], single_out[0], atol=1e-5), (
228:             "Batch output differs from single-sample output"
229:         )
230: 
231:     def test_edge_index_is_buffer(self, encoder: SpatioTemporalEncoder) -> None:
232:         # edge_index must be registered as a buffer (non-parameter)
233:         buffer_names = {name for name, _ in encoder.named_buffers()}
234:         assert "edge_index" in buffer_names
235: 
236:     def test_edge_index_shape(self, encoder: SpatioTemporalEncoder) -> None:
237:         assert encoder.edge_index.shape == EDGE_INDEX.shape
238: 
239:     def test_has_no_unused_params_in_training_mode(
240:         self, encoder: SpatioTemporalEncoder, dummy_input: torch.Tensor
241:     ) -> None:
242:         """All parameters must receive a gradient after a backward pass."""
243:         encoder.train()
244:         out = encoder(dummy_input)
245:         loss = out.sum()
246:         loss.backward()
247:         for name, param in encoder.named_parameters():
248:             if param.requires_grad:
249:                 assert param.grad is not None, f"No gradient for parameter: {name}"
250:         encoder.eval()
251: 
252: 
253: # ── ActionClassifier ──────────────────────────────────────────────────────────
254: 
255: 
256: class TestActionClassifier:
257:     @torch.no_grad()
258:     def test_output_shape(
259:         self, classifier: ActionClassifier, dummy_input: torch.Tensor
260:     ) -> None:
261:         logits = classifier(dummy_input)
262:         assert logits.shape == (B, NUM_CLASSES), (
263:             f"Expected ({B},{NUM_CLASSES}), got {logits.shape}"
264:         )
265: 
266:     @torch.no_grad()
267:     def test_no_nan_in_logits(
268:         self, classifier: ActionClassifier, dummy_input: torch.Tensor
269:     ) -> None:
270:         logits = classifier(dummy_input)
271:         assert not torch.isnan(logits).any()
272: 
273:     @torch.no_grad()
274:     def test_logits_are_unnormalised(
275:         self, classifier: ActionClassifier, dummy_input: torch.Tensor
276:     ) -> None:
277:         # The head is a plain Linear — softmax probabilities should NOT sum to 1
278:         logits = classifier(dummy_input)
279:         row_sums = logits.softmax(dim=-1).sum(dim=-1)
280:         # After softmax they should sum to ~1, but raw logits sum should not
281:         assert not torch.allclose(logits.sum(dim=-1), torch.ones(B))
282: 
283:     def test_encoder_is_accessible(self, classifier: ActionClassifier) -> None:
284:         assert isinstance(classifier.encoder, SpatioTemporalEncoder)
285: 
286:     def test_encoder_and_classifier_share_params(
287:         self, classifier: ActionClassifier
288:     ) -> None:
289:         # Parameters of classifier.encoder must be a subset of classifier's params
290:         enc_ids = {id(p) for p in classifier.encoder.parameters()}
291:         cls_ids = {id(p) for p in classifier.parameters()}
292:         assert enc_ids.issubset(cls_ids)
293: 
294:     def test_head_not_in_encoder(self, classifier: ActionClassifier) -> None:
295:         enc_ids = {id(p) for p in classifier.encoder.parameters()}
296:         head_ids = {id(p) for p in classifier.head.parameters()}
297:         assert head_ids.isdisjoint(enc_ids), "Head parameters appear in encoder"
298: 
299: 
300: # ── build_model factory ───────────────────────────────────────────────────────
301: 
302: 
303: class TestBuildModel:
304:     def test_returns_encoder_and_classifier(self, minimal_cfg: dict) -> None:
305:         encoder, classifier = build_model(minimal_cfg, EDGE_INDEX)
306:         assert isinstance(encoder, SpatioTemporalEncoder)
307:         assert isinstance(classifier, ActionClassifier)
308: 
309:     def test_classifier_wraps_same_encoder(self, minimal_cfg: dict) -> None:
310:         encoder, classifier = build_model(minimal_cfg, EDGE_INDEX)
311:         assert classifier.encoder is encoder
312: 
313:     @torch.no_grad()
314:     def test_forward_pass_after_build(self, minimal_cfg: dict) -> None:
315:         _, classifier = build_model(minimal_cfg, EDGE_INDEX)
316:         classifier.eval()
317:         dummy = torch.randn(B, T, J, F_IN)
318:         logits = classifier(dummy)
319:         assert logits.shape == (B, NUM_CLASSES)
320:         assert not torch.isnan(logits).any()
321: 
322:     @torch.no_grad()
323:     def test_perceiver_rope_forward_after_build(self, minimal_cfg: dict) -> None:
324:         cfg = {
325:             "data": {
326:                 "num_joints": J,
327:                 "seq_len": T,
328:                 "num_classes": NUM_CLASSES,
329:             },
330:             "model": {
331:                 "in_features": F_IN,
332:                 "gat_hidden_dim": 64,
333:                 "gat_heads": 4,
334:                 "gat_dropout": 0.0,
335:                 "d_model": D_MODEL,
336:                 "nhead": 4,
337:                 "num_transformer_layers": 2,
338:                 "dim_feedforward": 256,
339:                 "transformer_dropout": 0.0,
340:                 "cls_std": 0.02,
341:                 "temporal_backbone": "perceiver",
342:                 "positional_encoding": "rope",
343:                 "num_latents": 16,
344:                 "num_perceiver_layers": 2,
345:                 "cross_attn_heads": 4,
346:                 "self_attn_heads": 4,
347:                 "perceiver_dim_feedforward": 256,
348:                 "perceiver_dropout": 0.0,
349:                 "rope_base": 10000.0,
350:             },
351:         }
352:         _, classifier = build_model(cfg, EDGE_INDEX)
353:         classifier.eval()
354:         dummy = torch.randn(B, T, J, F_IN)
355:         logits = classifier(dummy)
356:         assert logits.shape == (B, NUM_CLASSES)
357:         assert not torch.isnan(logits).any()

(End of file - total 357 lines)
</content>
```

---

## Assistant (Build · gpt-5.3-codex · 12.7s)

_Thinking:_

**Planning ASCII cleanup in tests**

**Tool: bash**

**Input:**
```json
{
  "command": "python - <<'PY'\nfrom pathlib import Path\np=Path('/home/alan/mydev/college/football-injury-ai/tests/test_model.py')\nfor i,l in enumerate(p.read_text(encoding='utf-8').splitlines(),1):\n    if any(ord(ch)>127 for ch in l):\n        print(i, l)\nPY",
  "description": "Locate non-ASCII lines in test file"
}
```

**Output:**
```
4 # ──────────────────────────────────────────
26 # ── Constants used across tests ───────────────────────────────────────────────
36 # ── Fixtures ──────────────────────────────────────────────────────────────────
101 # ── PositionalEncoding ────────────────────────────────────────────────────────
184 # ── SpatioTemporalEncoder ─────────────────────────────────────────────────────
253 # ── ActionClassifier ──────────────────────────────────────────────────────────
277         # The head is a plain Linear — softmax probabilities should NOT sum to 1
300 # ── build_model factory ───────────────────────────────────────────────────────

```

---

## Assistant (Build · gpt-5.3-codex · 16.0s)

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Update File: /home/alan/mydev/college/football-injury-ai/tests/test_model.py\n@@\n-# ──────────────────────────────────────────\n+# ------------------------------------------\n@@\n-# ── Constants used across tests ───────────────────────────────────────────────\n+# -- Constants used across tests ------------------------------------------------\n@@\n-# ── Fixtures ──────────────────────────────────────────────────────────────────\n+# -- Fixtures -------------------------------------------------------------------\n@@\n-# ── PositionalEncoding ────────────────────────────────────────────────────────\n+# -- PositionalEncoding ---------------------------------------------------------\n@@\n-# ── SpatioTemporalEncoder ─────────────────────────────────────────────────────\n+# -- SpatioTemporalEncoder ------------------------------------------------------\n@@\n-# ── ActionClassifier ──────────────────────────────────────────────────────────\n+# -- ActionClassifier -----------------------------------------------------------\n@@\n-        # The head is a plain Linear — softmax probabilities should NOT sum to 1\n+        # The head is a plain Linear - softmax probabilities should NOT sum to 1\n@@\n-# ── build_model factory ───────────────────────────────────────────────────────\n+# -- build_model factory --------------------------------------------------------\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
M tests/test_model.py

LSP errors detected in tests/test_model.py, please fix:
<diagnostics file="/home/alan/mydev/college/football-injury-ai/tests/test_model.py">
ERROR [11:8] Import "pytest" could not be resolved
ERROR [12:8] Import "torch" could not be resolved
ERROR [13:8] Import "torch.nn" could not be resolved
</diagnostics>
```

---

## Assistant (Build · gpt-5.3-codex · 4.6s)

**Tool: bash**

**Input:**
```json
{
  "command": "python - <<'PY'\nfrom pathlib import Path\nfiles=[\n'./tests/test_model.py',\n'./module1/pretrain/model.py',\n'./configs/pretrain_perceiver_rope.yaml',\n]\nfor f in files:\n    text=Path(f).read_text(encoding='utf-8')\n    bad=sorted({ch for ch in text if ord(ch)>127})\n    print(f, bad)\nPY",
  "description": "Recheck key files for non-ASCII"
}
```

**Output:**
```
./tests/test_model.py []
./module1/pretrain/model.py []
./configs/pretrain_perceiver_rope.yaml []

```

---

## Assistant (Build · gpt-5.3-codex · 4.8s)

_Thinking:_

**Updating tests and docs**

**Tool: bash**

**Input:**
```json
{
  "command": "python -m py_compile module1/pretrain/model.py tests/test_model.py",
  "description": "Compile updated model and tests"
}
```

---

## Assistant (Build · gpt-5.3-codex · 6.1s)

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Update File: /home/alan/mydev/college/football-injury-ai/docs/COMMANDS.md\n@@\n ### Train with Best Tuned Parameters\n \n ```bash\n uv run module1/pretrain/train.py --config outputs/logs/pretrain_optuna_best.yaml\n ```\n+\n+### Train Perceiver + RoPE Experiment\n+\n+```bash\n+uv run module1/pretrain/train.py --config configs/pretrain_perceiver_rope.yaml\n+```\n+\n+Note: checkpoints from Transformer and Perceiver variants are not interchangeable.\n@@\n ### Evaluate a Checkpoint\n \n ```bash\n # Basic evaluation\n uv run module1/pretrain/evaluate.py --config configs/pretrain.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt\n@@\n # With t-SNE visualization\n uv run module1/pretrain/evaluate.py --config configs/pretrain.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --tsne\n+\n+# Perceiver + RoPE checkpoint evaluation\n+uv run module1/pretrain/evaluate.py --config configs/pretrain_perceiver_rope.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --split val --tsne\n ```\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
M docs/COMMANDS.md
```

---

## Assistant (Build · gpt-5.3-codex · 4.9s)

**Tool: apply_patch**

**Input:**
```json
{
  "patchText": "*** Begin Patch\n*** Update File: /home/alan/mydev/college/football-injury-ai/README.md\n@@\n # Train final pretraining config (official val split)\n uv run module1/pretrain/train.py --config configs/pretrain_final.yaml\n+\n+# Train Perceiver + RoPE experiment\n+uv run module1/pretrain/train.py --config configs/pretrain_perceiver_rope.yaml\n*** End Patch"
}
```

**Output:**
```
Success. Updated the following files:
M README.md
```

---

## Assistant (Build · gpt-5.3-codex · 8.0s)

**Tool: bash**

**Input:**
```json
{
  "command": "git status --short",
  "description": "Check changed files after architecture implementation"
}
```

**Output:**
```
 M README.md
 M docs/COMMANDS.md
 M module1/pretrain/model.py
 M tests/test_model.py
?? configs/pretrain_perceiver_rope.yaml

```

---

## Assistant (Build · gpt-5.3-codex · 12.7s)

_Thinking:_

**Summarizing architecture changes**

Implemented. The architecture now supports **Perceiver latent bottlenecks + RoPE** as an opt-in experiment, while keeping your baseline Transformer path intact.

- Added a configurable temporal backbone in `module1/pretrain/model.py`:
  - `temporal_backbone: transformer` (existing behavior)
  - `temporal_backbone: perceiver` (new)
- Added RoPE support in the new attention stack:
  - `RotaryEmbedding`
  - custom `MultiHeadAttention` with optional RoPE on Q/K
  - `PerceiverLayer` (cross-attn + latent self-attn + FFN)
- Kept output/API unchanged:
  - `SpatioTemporalEncoder` still returns `(B, d_model)`
  - `ActionClassifier`, train/eval/live scripts continue to work with same interfaces
- Added experiment config: `configs/pretrain_perceiver_rope.yaml`
- Updated tests in `tests/test_model.py`:
  - RoPE shape/validation tests
  - self/cross attention tests
  - Perceiver+RoPE `build_model` forward pass test
- Updated docs:
  - `docs/COMMANDS.md` (train/eval commands for Perceiver+RoPE)
  - `README.md` (new training command)

Use this to train the new architecture:
```bash
uv run module1/pretrain/train.py --config configs/pretrain_perceiver_rope.yaml
```

Then evaluate:
```bash
uv run module1/pretrain/evaluate.py --config configs/pretrain_perceiver_rope.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --split val --tsne
```

I validated syntax with `py_compile`. I couldn’t run full `pytest` here because this machine currently doesn’t have runtime deps (`torch/cv2/mediapipe`) installed.

---

## User

I have installed the dependencies, run the test, deep check is everything is correct, if yes commit and push the code

---

## Assistant (Build · gpt-5.3-codex · 17.4s)

_Thinking:_

**Running git status, diff, and log**

**Tool: bash**

**Input:**
```json
{
  "command": "git status",
  "description": "Show repository status"
}
```

**Output:**
```
On branch master
Your branch is up to date with 'origin/master'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   README.md
	modified:   docs/COMMANDS.md
	modified:   module1/pretrain/model.py
	modified:   pyproject.toml
	modified:   tests/test_model.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	configs/pretrain_perceiver_rope.yaml
	uv.lock

no changes added to commit (use "git add" and/or "git commit -a")

```

**Tool: bash**

**Input:**
```json
{
  "command": "git diff",
  "description": "Show unstaged and staged diffs"
}
```

**Output:**
```
diff --git a/README.md b/README.md
index 61c35e4..5909e0b 100644
--- a/README.md
+++ b/README.md
@@ -73,6 +73,9 @@ uv run module1/pretrain/train.py --config configs/pretrain.yaml
 # Train final pretraining config (official val split)
 uv run module1/pretrain/train.py --config configs/pretrain_final.yaml
 
+# Train Perceiver + RoPE experiment
+uv run module1/pretrain/train.py --config configs/pretrain_perceiver_rope.yaml
+
 # Evaluate
 uv run module1/pretrain/evaluate.py --config configs/pretrain.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt
 
diff --git a/docs/COMMANDS.md b/docs/COMMANDS.md
index 4bbb522..83d13d8 100644
--- a/docs/COMMANDS.md
+++ b/docs/COMMANDS.md
@@ -81,6 +81,14 @@ uv run module1/pretrain/tune_optuna.py --config configs/pretrain.yaml --timeout-
 uv run module1/pretrain/train.py --config outputs/logs/pretrain_optuna_best.yaml
 ```
 
+### Train Perceiver + RoPE Experiment
+
+```bash
+uv run module1/pretrain/train.py --config configs/pretrain_perceiver_rope.yaml
+```
+
+Note: checkpoints from Transformer and Perceiver variants are not interchangeable.
+
 ---
 
 ## Evaluation
@@ -93,6 +101,9 @@ uv run module1/pretrain/evaluate.py --config configs/pretrain.yaml --ckpt output
 
 # With t-SNE visualization
 uv run module1/pretrain/evaluate.py --config configs/pretrain.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --tsne
+
+# Perceiver + RoPE checkpoint evaluation
+uv run module1/pretrain/evaluate.py --config configs/pretrain_perceiver_rope.yaml --ckpt outputs/checkpoints/pretrain/pretrain_best.pt --split val --tsne
 ```
 
 ### Evaluate with Final Pretraining Config
diff --git a/module1/pretrain/model.py b/module1/pretrain/model.py
index 2251042..2315404 100644
--- a/module1/pretrain/model.py
+++ b/module1/pretrain/model.py
@@ -1,29 +1,20 @@
 # module1/pretrain/model.py
 #
-# Stage 1 — SpatioTemporalEncoder + ActionClassifier
-# ────────────────────────────────────────────────────
-# End-to-end model that encodes a sequence of skeleton graphs:
+# Stage 1 - SpatioTemporalEncoder + ActionClassifier
+# ---------------------------------------------------
+# End-to-end model that encodes a sequence of skeleton graphs.
 #
-#   Input : (B, T, J, F)  — batch, frames, joints, features (pos+vel+acc = 9)
+# Input: (B, T, J, F) where F=9 [x,y,z,vx,vy,vz,ax,ay,az].
 #
-#   Spatial stage — time-distributed GAT (shared weights across all T frames):
-#     GATConv( F → 16, heads=4, concat=True)  → 64-dim per node + BN + ELU
-#     GATConv(64 → 16, heads=4, concat=True)  → 64-dim per node + BN + ELU
-#     GATConv(64 → 64, heads=1, concat=False) → 64-dim per node + BN + ELU
-#     global_mean_pool over joints             → (B*T, 64)
-#     reshape                                  → (B, T, 64)
+# Spatial stage (shared across all T frames):
+#   GATConv -> BN -> ELU -> Dropout
+#   GATConv -> BN -> ELU -> Dropout
+#   GATConv -> BN -> ELU
+#   global_mean_pool over joints -> (B, T, gat_hidden)
 #
-#   Temporal stage — Transformer encoder:
-#     input_proj Linear(64 → d_model)
-#     prepend CLS token
-#     sinusoidal positional encoding
-#     2 × TransformerEncoderLayer (pre-norm, nhead=4, ff=256)
-#     LayerNorm
-#     extract CLS output                       → (B, d_model)
-#
-#   ActionClassifier wraps SpatioTemporalEncoder with a linear head for
-#   102-class ANUBIS pretraining.  After pretraining, save only the encoder
-#   (encoder.state_dict()) and attach new fine-tuning heads.
+# Temporal stage (configurable):
+#   1) Transformer (baseline): CLS + sinusoidal position + TransformerEncoder
+#   2) Perceiver latent bottleneck: cross-attn + latent self-attn (+ optional RoPE)
 
 from __future__ import annotations
 
@@ -40,7 +31,7 @@ from utils.logger import get_logger
 log = get_logger(__name__)
 
 
-# ── Positional Encoding ────────────────────────────────────────────────────────
+# -- Positional encoding -------------------------------------------------------
 
 
 class PositionalEncoding(nn.Module):
@@ -49,16 +40,19 @@ class PositionalEncoding(nn.Module):
 
     Parameters
     ----------
-    d_model  : int   embedding dimension
-    max_len  : int   maximum sequence length (including CLS token)
-    dropout  : float dropout applied after adding the encoding
+    d_model : int
+        Embedding dimension.
+    max_len : int
+        Maximum sequence length.
+    dropout : float
+        Dropout applied after adding positional encoding.
     """
 
     def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1) -> None:
         super().__init__()
         self.dropout = nn.Dropout(p=dropout)
 
-        position = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
+        position = torch.arange(max_len).unsqueeze(1).float()
         div_term = torch.exp(
             torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
         )
@@ -68,45 +62,246 @@ class PositionalEncoding(nn.Module):
         self.register_buffer("pe", pe)
 
     def forward(self, x: torch.Tensor) -> torch.Tensor:
+        x = x + self.pe[:, : x.size(1), :]
+        return self.dropout(x)
+
+
+def _rotate_half(x: torch.Tensor) -> torch.Tensor:
+    """Rotate even/odd features for rotary embedding."""
+    x_even = x[..., ::2]
+    x_odd = x[..., 1::2]
+    x_rot = torch.stack((-x_odd, x_even), dim=-1)
+    return x_rot.flatten(-2)
+
+
+class RotaryEmbedding(nn.Module):
+    """
+    Rotary positional embedding helper for attention Q/K tensors.
+
+    Parameters
+    ----------
+    dim : int
+        Attention head dimension (must be even).
+    base : float
+        RoPE base frequency.
+    """
+
+    def __init__(self, dim: int, base: float = 10000.0) -> None:
+        super().__init__()
+        if dim % 2 != 0:
+            raise ValueError(f"RoPE requires even head dimension, got dim={dim}")
+        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
+        self.register_buffer("inv_freq", inv_freq)
+
+    def get_cos_sin(
+        self,
+        seq_len: int,
+        device: torch.device,
+        dtype: torch.dtype,
+    ) -> Tuple[torch.Tensor, torch.Tensor]:
+        """Return cosine and sine tensors shaped for (B, H, T, D)."""
+        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
+        freqs = torch.outer(t, self.inv_freq)  # (T, D/2)
+        emb = torch.cat([freqs, freqs], dim=-1)  # (T, D)
+        cos = emb.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
+        sin = emb.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
+        return cos, sin
+
+
+# -- Attention blocks ----------------------------------------------------------
+
+
+class MultiHeadAttention(nn.Module):
+    """
+    Multi-head attention supporting self-attention and cross-attention.
+
+    Parameters
+    ----------
+    d_model : int
+        Embedding dimension.
+    nhead : int
+        Number of attention heads.
+    dropout : float
+        Dropout on attention probabilities and output projection.
+    cross_attention : bool
+        If True, query comes from x and key/value comes from context.
+    use_rope : bool
+        If True, apply RoPE to q/k tensors.
+    rope_base : float
+        RoPE base frequency.
+    """
+
+    def __init__(
+        self,
+        d_model: int,
+        nhead: int,
+        dropout: float = 0.1,
+        cross_attention: bool = False,
+        use_rope: bool = False,
+        rope_base: float = 10000.0,
+    ) -> None:
+        super().__init__()
+        if d_model % nhead != 0:
+            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
+
+        self.d_model = d_model
+        self.nhead = nhead
+        self.head_dim = d_model // nhead
+        self.scale = self.head_dim**-0.5
+        self.cross_attention = cross_attention
+        self.use_rope = use_rope
+
+        if cross_attention:
+            self.q_proj = nn.Linear(d_model, d_model)
+            self.kv_proj = nn.Linear(d_model, 2 * d_model)
+        else:
+            self.qkv_proj = nn.Linear(d_model, 3 * d_model)
+
+        self.out_proj = nn.Linear(d_model, d_model)
+        self.attn_dropout = nn.Dropout(dropout)
+        self.out_dropout = nn.Dropout(dropout)
+
+        if use_rope:
+            self.rope = RotaryEmbedding(self.head_dim, base=rope_base)
+        else:
+            self.rope = None
+
+    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
+        """(B, T, D) -> (B, H, T, Dh)"""
+        B, T, _ = x.shape
+        x = x.view(B, T, self.nhead, self.head_dim)
+        return x.permute(0, 2, 1, 3)
+
+    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
+        """(B, H, T, Dh) -> (B, T, D)"""
+        B, H, T, Dh = x.shape
+        return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * Dh)
+
+    def forward(
+        self,
+        x: torch.Tensor,
+        context: Optional[torch.Tensor] = None,
+    ) -> torch.Tensor:
         """
         Parameters
         ----------
-        x : torch.Tensor  shape (B, T, d_model)
+        x : torch.Tensor
+            Query tensor, shape (B, Tq, D).
+        context : torch.Tensor, optional
+            Key/value source, shape (B, Tk, D). Required for cross-attention.
 
         Returns
         -------
-        out : torch.Tensor  shape (B, T, d_model)
+        out : torch.Tensor
+            Attention output, shape (B, Tq, D).
         """
-        x = x + self.pe[:, : x.size(1), :]
-        return self.dropout(x)
+        if self.cross_attention:
+            if context is None:
+                raise ValueError("context is required for cross-attention")
+            q = self.q_proj(x)
+            kv = self.kv_proj(context)
+            k, v = torch.chunk(kv, 2, dim=-1)
+        else:
+            qkv = self.qkv_proj(x)
+            q, k, v = torch.chunk(qkv, 3, dim=-1)
+
+        q = self._reshape_heads(q)
+        k = self._reshape_heads(k)
+        v = self._reshape_heads(v)
+
+        if self.rope is not None:
+            cos_q, sin_q = self.rope.get_cos_sin(q.shape[2], q.device, q.dtype)
+            cos_k, sin_k = self.rope.get_cos_sin(k.shape[2], k.device, k.dtype)
+            q = (q * cos_q) + (_rotate_half(q) * sin_q)
+            k = (k * cos_k) + (_rotate_half(k) * sin_k)
+
+        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
+        attn_probs = torch.softmax(attn_scores, dim=-1)
+        attn_probs = self.attn_dropout(attn_probs)
+
+        out = torch.matmul(attn_probs, v)
+        out = self._merge_heads(out)
+        out = self.out_proj(out)
+        out = self.out_dropout(out)
+        return out
+
+
+class FeedForward(nn.Module):
+    """Transformer-style feedforward block."""
+
+    def __init__(self, d_model: int, dim_ff: int, dropout: float) -> None:
+        super().__init__()
+        self.net = nn.Sequential(
+            nn.Linear(d_model, dim_ff),
+            nn.GELU(),
+            nn.Dropout(dropout),
+            nn.Linear(dim_ff, d_model),
+            nn.Dropout(dropout),
+        )
+
+    def forward(self, x: torch.Tensor) -> torch.Tensor:
+        return self.net(x)
+
+
+class PerceiverLayer(nn.Module):
+    """
+    One Perceiver layer: cross-attention + latent self-attention + FFN.
+    """
+
+    def __init__(
+        self,
+        d_model: int,
+        cross_attn_heads: int,
+        self_attn_heads: int,
+        dim_ff: int,
+        dropout: float,
+        use_rope: bool,
+        rope_base: float,
+    ) -> None:
+        super().__init__()
+        self.norm_latent_cross = nn.LayerNorm(d_model)
+        self.norm_input_cross = nn.LayerNorm(d_model)
+        self.norm_latent_self = nn.LayerNorm(d_model)
+        self.norm_ff = nn.LayerNorm(d_model)
+
+        self.cross_attn = MultiHeadAttention(
+            d_model=d_model,
+            nhead=cross_attn_heads,
+            dropout=dropout,
+            cross_attention=True,
+            use_rope=use_rope,
+            rope_base=rope_base,
+        )
+        self.self_attn = MultiHeadAttention(
+            d_model=d_model,
+            nhead=self_attn_heads,
+            dropout=dropout,
+            cross_attention=False,
+            use_rope=use_rope,
+            rope_base=rope_base,
+        )
+        self.ff = FeedForward(d_model=d_model, dim_ff=dim_ff, dropout=dropout)
+
+    def forward(self, latents: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
+        latents = latents + self.cross_attn(
+            x=self.norm_latent_cross(latents),
+            context=self.norm_input_cross(tokens),
+        )
+        latents = latents + self.self_attn(self.norm_latent_self(latents))
+        latents = latents + self.ff(self.norm_ff(latents))
+        return latents
 
 
-# ── SpatioTemporalEncoder ─────────────────────────────────────────────────────
+# -- SpatioTemporalEncoder -----------------------------------------------------
 
 
 class SpatioTemporalEncoder(nn.Module):
     """
-    End-to-end spatiotemporal encoder: time-distributed GAT + Transformer.
+    End-to-end spatiotemporal encoder: GAT + temporal backbone.
 
-    The encoder is the shared backbone used across all training stages.
-    It outputs a single CLS embedding per sequence.
-
-    Parameters
-    ----------
-    num_joints   : int   number of skeleton joints (32 for ANUBIS)
-    in_features  : int   feature dimension per joint (9: pos+vel+acc)
-    edge_index   : torch.Tensor  shape (2, E)  skeleton connectivity;
-                   registered as a buffer so it moves to device automatically
-    gat_hidden   : int   output dim of every GAT block (default 64)
-    gat_heads    : int   attention heads for GAT layers 1 & 2 (default 4)
-    gat_dropout  : float dropout on GAT attention coefficients
-    d_model      : int   transformer embedding dimension (default 128)
-    nhead        : int   transformer attention heads
-    num_layers   : int   number of TransformerEncoderLayer blocks
-    dim_ff       : int   feedforward expansion inside transformer
-    dropout      : float transformer dropout
-    seq_len      : int   expected sequence length (used to size pos. encoding)
-    cls_std      : float trunc_normal std for CLS-token initialisation
+    Temporal backbones:
+    - "transformer": baseline with CLS token and sinusoidal encoding.
+    - "perceiver": learnable latent bottleneck; supports sinusoidal or RoPE.
     """
 
     def __init__(
@@ -124,6 +319,15 @@ class SpatioTemporalEncoder(nn.Module):
         dropout: float = 0.1,
         seq_len: int = 60,
         cls_std: float = 0.02,
+        temporal_backbone: str = "transformer",
+        positional_encoding: str = "sinusoidal",
+        num_latents: int = 64,
+        num_perceiver_layers: int = 2,
+        cross_attn_heads: int = 4,
+        self_attn_heads: int = 4,
+        perceiver_dim_ff: int = 256,
+        perceiver_dropout: float = 0.1,
+        rope_base: float = 10000.0,
     ) -> None:
         super().__init__()
 
@@ -131,106 +335,177 @@ class SpatioTemporalEncoder(nn.Module):
         self.gat_hidden = gat_hidden
         self.gat_dropout = gat_dropout
         self.d_model = d_model
+        self.temporal_backbone = temporal_backbone.lower()
+        self.positional_encoding = positional_encoding.lower()
 
-        # Register skeleton topology as a non-learnable buffer
-        self.register_buffer("edge_index", edge_index)  # (2, E)
+        self.register_buffer("edge_index", edge_index)
 
-        # ── GAT layers (shared across all time steps) ──────────────────────
-        per_head_dim = gat_hidden // gat_heads  # 16
+        # Spatial GAT stack
+        per_head_dim = gat_hidden // gat_heads
         self.gat1 = GATConv(
-            in_features, per_head_dim, heads=gat_heads, concat=True, dropout=gat_dropout
+            in_features,
+            per_head_dim,
+            heads=gat_heads,
+            concat=True,
+            dropout=gat_dropout,
         )
         self.gat2 = GATConv(
-            gat_hidden, per_head_dim, heads=gat_heads, concat=True, dropout=gat_dropout
+            gat_hidden,
+            per_head_dim,
+            heads=gat_heads,
+            concat=True,
+            dropout=gat_dropout,
         )
         self.gat3 = GATConv(
-            gat_hidden, gat_hidden, heads=1, concat=False, dropout=gat_dropout
+            gat_hidden,
+            gat_hidden,
+            heads=1,
+            concat=False,
+            dropout=gat_dropout,
         )
 
         self.bn1 = nn.BatchNorm1d(gat_hidden)
         self.bn2 = nn.BatchNorm1d(gat_hidden)
         self.bn3 = nn.BatchNorm1d(gat_hidden)
 
-        # ── Transformer ────────────────────────────────────────────────────
         self.input_proj = nn.Linear(gat_hidden, d_model)
 
-        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
-        nn.init.trunc_normal_(self.cls_token, std=cls_std)
-
-        # max_len = seq_len + 1 to accommodate the CLS token
-        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)
-
-        encoder_layer = nn.TransformerEncoderLayer(
-            d_model=d_model,
-            nhead=nhead,
-            dim_feedforward=dim_ff,
-            dropout=dropout,
-            batch_first=True,
-            norm_first=True,  # pre-norm variant — more stable gradients
-            activation="gelu",
-        )
-        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
-        self.norm = nn.LayerNorm(d_model)
-
-    # ── Internal helpers ───────────────────────────────────────────────────────
+        # Optional temporal modules initialised per-backbone.
+        self.pos_enc: Optional[nn.Module] = None
+        self.transformer: Optional[nn.Module] = None
+        self.latents: Optional[nn.Parameter] = None
+        self.perceiver_layers: Optional[nn.ModuleList] = None
+
+        # Temporal backbone selection
+        if self.temporal_backbone == "transformer":
+            if self.positional_encoding != "sinusoidal":
+                raise ValueError(
+                    "Transformer backbone currently supports only "
+                    "positional_encoding='sinusoidal'."
+                )
+
+            self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
+            nn.init.trunc_normal_(self.cls_token, std=cls_std)
+
+            self.pos_enc = PositionalEncoding(
+                d_model=d_model,
+                max_len=seq_len + 1,
+                dropout=dropout,
+            )
+
+            encoder_layer = nn.TransformerEncoderLayer(
+                d_model=d_model,
+                nhead=nhead,
+                dim_feedforward=dim_ff,
+                dropout=dropout,
+                batch_first=True,
+                norm_first=True,
+                activation="gelu",
+            )
+            self.transformer = nn.TransformerEncoder(
+                encoder_layer, num_layers=num_layers
+            )
+            self.norm = nn.LayerNorm(d_model)
+
+        elif self.temporal_backbone == "perceiver":
+            if self.positional_encoding not in ("sinusoidal", "rope"):
+                raise ValueError(
+                    "Perceiver backbone supports positional_encoding='sinusoidal' "
+                    "or 'rope'."
+                )
+
+            self.latents = nn.Parameter(torch.empty(1, num_latents, d_model))
+            nn.init.trunc_normal_(self.latents, std=cls_std)
+
+            self.pos_enc = None
+            if self.positional_encoding == "sinusoidal":
+                self.pos_enc = PositionalEncoding(
+                    d_model=d_model,
+                    max_len=seq_len,
+                    dropout=perceiver_dropout,
+                )
+
+            use_rope = self.positional_encoding == "rope"
+            self.perceiver_layers = nn.ModuleList(
+                [
+                    PerceiverLayer(
+                        d_model=d_model,
+                        cross_attn_heads=cross_attn_heads,
+                        self_attn_heads=self_attn_heads,
+                        dim_ff=perceiver_dim_ff,
+                        dropout=perceiver_dropout,
+                        use_rope=use_rope,
+                        rope_base=rope_base,
+                    )
+                    for _ in range(num_perceiver_layers)
+                ]
+            )
+            self.norm = nn.LayerNorm(d_model)
+
+        else:
+            raise ValueError(
+                f"Unsupported temporal_backbone='{temporal_backbone}'. "
+                "Use 'transformer' or 'perceiver'."
+            )
 
     def _gat_forward(self, x: torch.Tensor) -> torch.Tensor:
-        """
-        Process all T frames through the GAT in a single batched pass.
-
-        Each (frame, sequence) pair is treated as an independent graph.
-        All B*T graphs share the same edge topology (self.edge_index).
-
-        Parameters
-        ----------
-        x : torch.Tensor  shape (B, T, J, F)
-
-        Returns
-        -------
-        spatial : torch.Tensor  shape (B, T, gat_hidden)
-        """
+        """Run all B*T graphs through GAT in one batched pass."""
         B, T, J, Fin = x.shape
         num_graphs = B * T
 
-        # Flatten nodes: (B*T*J, Fin)
         x_nodes = x.reshape(num_graphs * J, Fin)
 
-        # Build batched edge index by offsetting each graph's node indices by J
-        # edge_index: (2, E)  →  broadcast to (2, num_graphs, E)
         E = self.edge_index.shape[1]
-        offsets = torch.arange(num_graphs, device=x.device) * J  # (num_graphs,)
-        edge_batch = self.edge_index.unsqueeze(1).expand(
-            -1, num_graphs, -1
-        )  # (2, G, E)
-        edge_batch = edge_batch + offsets.view(1, -1, 1)  # (2, G, E)
-        edge_flat = edge_batch.reshape(2, num_graphs * E)  # (2, G*E)
-
-        # batch_tensor: maps each node to its graph index
+        offsets = torch.arange(num_graphs, device=x.device) * J
+        edge_batch = self.edge_index.unsqueeze(1).expand(-1, num_graphs, -1)
+        edge_batch = edge_batch + offsets.view(1, -1, 1)
+        edge_flat = edge_batch.reshape(2, num_graphs * E)
+
         batch_tensor = torch.arange(num_graphs, device=x.device).repeat_interleave(J)
 
-        # ── GAT layer 1 ────────────────────────────────────────────────────
-        h = self.gat1(x_nodes, edge_flat)  # (B*T*J, gat_hidden)
+        h = self.gat1(x_nodes, edge_flat)
         h = self.bn1(h)
         h = F.elu(h)
         h = F.dropout(h, p=self.gat_dropout, training=self.training)
 
-        # ── GAT layer 2 ────────────────────────────────────────────────────
         h = self.gat2(h, edge_flat)
         h = self.bn2(h)
         h = F.elu(h)
         h = F.dropout(h, p=self.gat_dropout, training=self.training)
 
-        # ── GAT layer 3 ────────────────────────────────────────────────────
         h = self.gat3(h, edge_flat)
         h = self.bn3(h)
         h = F.elu(h)
 
-        # Global mean pool: (B*T*J, gat_hidden) → (B*T, gat_hidden)
         spatial = global_mean_pool(h, batch_tensor)
+        return spatial.reshape(B, T, self.gat_hidden)
+
+    def _forward_transformer(self, h: torch.Tensor) -> torch.Tensor:
+        """Temporal forward for baseline transformer path."""
+        if self.pos_enc is None or self.transformer is None:
+            raise RuntimeError("Transformer temporal modules are not initialised.")
+        B = h.shape[0]
+        cls = self.cls_token.expand(B, -1, -1)
+        h = torch.cat([cls, h], dim=1)
+        h = self.pos_enc(h)
+        h = self.transformer(h)
+        h = self.norm(h)
+        return h[:, 0, :]
 
-        return spatial.reshape(B, T, self.gat_hidden)  # (B, T, gat_hidden)
+    def _forward_perceiver(self, h: torch.Tensor) -> torch.Tensor:
+        """Temporal forward for Perceiver latent bottleneck path."""
+        if self.latents is None or self.perceiver_layers is None:
+            raise RuntimeError("Perceiver temporal modules are not initialised.")
+        if self.pos_enc is not None:
+            h = self.pos_enc(h)
 
-    # ── Forward pass ──────────────────────────────────────────────────────────
+        B = h.shape[0]
+        latents = self.latents.expand(B, -1, -1)
+        for layer in self.perceiver_layers:
+            latents = layer(latents, h)
+
+        latents = self.norm(latents)
+        return latents[:, 0, :]
 
     def forward(self, x: torch.Tensor) -> torch.Tensor:
         """
@@ -238,52 +513,27 @@ class SpatioTemporalEncoder(nn.Module):
 
         Parameters
         ----------
-        x : torch.Tensor  shape (B, T, J, F)
-            Joint features: pos + vel + acc (F = 9).
+        x : torch.Tensor
+            Shape (B, T, J, F).
 
         Returns
         -------
-        cls_embedding : torch.Tensor  shape (B, d_model)
-            Sequence-level representation extracted from the CLS token.
+        cls_embedding : torch.Tensor
+            Shape (B, d_model).
         """
-        B = x.shape[0]
-
-        # Stage 1: spatial encoding (time-distributed GAT)
-        spatial = self._gat_forward(x)  # (B, T, gat_hidden)
-
-        # Stage 2: temporal encoding
-        h = self.input_proj(spatial)  # (B, T, d_model)
+        spatial = self._gat_forward(x)
+        h = self.input_proj(spatial)
 
-        # Prepend CLS token
-        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
-        h = torch.cat([cls, h], dim=1)  # (B, T+1, d_model)
+        if self.temporal_backbone == "transformer":
+            return self._forward_transformer(h)
+        return self._forward_perceiver(h)
 
-        # Positional encoding
-        h = self.pos_enc(h)  # (B, T+1, d_model)
 
-        # Transformer encoder
-        h = self.transformer(h)  # (B, T+1, d_model)
-        h = self.norm(h)
-
-        # Extract CLS token (position 0)
-        return h[:, 0, :]  # (B, d_model)
-
-
-# ── ActionClassifier ──────────────────────────────────────────────────────────
+# -- ActionClassifier ----------------------------------------------------------
 
 
 class ActionClassifier(nn.Module):
-    """
-    Full pretraining model: SpatioTemporalEncoder + linear classification head.
-
-    Used only for Stage 1 (ANUBIS action classification pretraining).
-    After pretraining, save encoder.state_dict() and discard the head.
-
-    Parameters
-    ----------
-    encoder     : SpatioTemporalEncoder
-    num_classes : int   number of action classes (102 for ANUBIS)
-    """
+    """SpatioTemporalEncoder + linear classification head."""
 
     def __init__(self, encoder: SpatioTemporalEncoder, num_classes: int) -> None:
         super().__init__()
@@ -291,40 +541,21 @@ class ActionClassifier(nn.Module):
         self.head = nn.Linear(encoder.d_model, num_classes)
 
     def forward(self, x: torch.Tensor) -> torch.Tensor:
-        """
-        Parameters
-        ----------
-        x : torch.Tensor  shape (B, T, J, F)
+        cls_emb = self.encoder(x)
+        return self.head(cls_emb)
 
-        Returns
-        -------
-        logits : torch.Tensor  shape (B, num_classes)
-        """
-        cls_emb = self.encoder(x)  # (B, d_model)
-        return self.head(cls_emb)  # (B, num_classes)
 
-
-# ── Factory helper ────────────────────────────────────────────────────────────
+# -- Factory helper ------------------------------------------------------------
 
 
 def build_model(
-    cfg: dict, edge_index: torch.Tensor
+    cfg: dict,
+    edge_index: torch.Tensor,
 ) -> Tuple[SpatioTemporalEncoder, ActionClassifier]:
-    """
-    Construct encoder and full classifier from config dict.
-
-    Parameters
-    ----------
-    cfg        : dict               full config loaded from configs/pretrain.yaml
-    edge_index : torch.Tensor (2,E) skeleton edge index
-
-    Returns
-    -------
-    encoder    : SpatioTemporalEncoder
-    classifier : ActionClassifier
-    """
+    """Construct encoder and classifier from config."""
     m = cfg["model"]
     d = cfg["data"]
+
     encoder = SpatioTemporalEncoder(
         num_joints=d["num_joints"],
         in_features=m["in_features"],
@@ -339,6 +570,15 @@ def build_model(
         dropout=m["transformer_dropout"],
         seq_len=d["seq_len"],
         cls_std=m["cls_std"],
+        temporal_backbone=m.get("temporal_backbone", "transformer"),
+        positional_encoding=m.get("positional_encoding", "sinusoidal"),
+        num_latents=m.get("num_latents", 64),
+        num_perceiver_layers=m.get("num_perceiver_layers", 2),
+        cross_attn_heads=m.get("cross_attn_heads", m["nhead"]),
+        self_attn_heads=m.get("self_attn_heads", m["nhead"]),
+        perceiver_dim_ff=m.get("perceiver_dim_feedforward", m["dim_feedforward"]),
+        perceiver_dropout=m.get("perceiver_dropout", m["transformer_dropout"]),
+        rope_base=m.get("rope_base", 10000.0),
     )
     classifier = ActionClassifier(encoder, num_classes=d["num_classes"])
     total = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
@@ -346,7 +586,8 @@ def build_model(
     return encoder, classifier
 
 
-# ── Quick sanity check ────────────────────────────────────────────────────────
+# -- Quick sanity check --------------------------------------------------------
+
 
 if __name__ == "__main__":
     from module1.data.joint_config import EDGE_INDEX, NUM_JOINTS
@@ -360,12 +601,10 @@ if __name__ == "__main__":
         edge_index=EDGE_INDEX,
         seq_len=T,
     ).to(device)
-
     classifier = ActionClassifier(encoder, num_classes=102).to(device)
     classifier.eval()
 
     dummy = torch.randn(B, T, J, F_IN, device=device)
-
     with torch.no_grad():
         cls_emb = encoder(dummy)
         logits = classifier(dummy)
diff --git a/pyproject.toml b/pyproject.toml
index cd5f435..0748065 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -51,6 +51,9 @@ dev = [
 requires = ["hatchling"]
 build-backend = "hatchling.build"
 
+[tool.hatch.build.targets.wheel]
+packages = ["module1", "module2", "module3", "fusion", "utils"]
+
 [tool.pytest.ini_options]
 testpaths = ["tests"]
 python_files = ["test_*.py"]
diff --git a/tests/test_model.py b/tests/test_model.py
index 66912d2..f99f566 100644
--- a/tests/test_model.py
+++ b/tests/test_model.py
@@ -1,7 +1,7 @@
 # tests/test_model.py
 #
 # Unit tests for module1/pretrain/model.py
-# ──────────────────────────────────────────
+# ------------------------------------------
 # Covers: PositionalEncoding, SpatioTemporalEncoder (forward pass shapes,
 #         no NaNs, edge_index buffer device movement), ActionClassifier
 #         (forward pass, logit shape), and encoder/classifier separation.
@@ -15,13 +15,15 @@ import torch.nn as nn
 from module1.data.joint_config import EDGE_INDEX, NUM_JOINTS
 from module1.pretrain.model import (
     ActionClassifier,
+    MultiHeadAttention,
     PositionalEncoding,
+    RotaryEmbedding,
     SpatioTemporalEncoder,
     build_model,
 )
 
 
-# ── Constants used across tests ───────────────────────────────────────────────
+# -- Constants used across tests ------------------------------------------------
 
 B = 2  # batch size (small to keep tests fast)
 T = 60  # sequence length (frames)
@@ -31,7 +33,7 @@ D_MODEL = 128
 NUM_CLASSES = 102
 
 
-# ── Fixtures ──────────────────────────────────────────────────────────────────
+# -- Fixtures -------------------------------------------------------------------
 
 
 @pytest.fixture
@@ -96,7 +98,7 @@ def minimal_cfg() -> dict:
     }
 
 
-# ── PositionalEncoding ────────────────────────────────────────────────────────
+# -- PositionalEncoding ---------------------------------------------------------
 
 
 class TestPositionalEncoding:
@@ -129,7 +131,57 @@ class TestPositionalEncoding:
         assert not torch.allclose(out[0, 0, :], out[0, 1, :])
 
 
-# ── SpatioTemporalEncoder ─────────────────────────────────────────────────────
+class TestRotaryEmbedding:
+    def test_cos_sin_shape(self) -> None:
+        rope = RotaryEmbedding(dim=32, base=10000.0)
+        cos, sin = rope.get_cos_sin(
+            seq_len=10,
+            device=torch.device("cpu"),
+            dtype=torch.float32,
+        )
+        assert cos.shape == (1, 1, 10, 32)
+        assert sin.shape == (1, 1, 10, 32)
+
+    def test_rejects_odd_dimension(self) -> None:
+        with pytest.raises(ValueError):
+            _ = RotaryEmbedding(dim=31)
+
+
+class TestMultiHeadAttention:
+    @torch.no_grad()
+    def test_self_attention_shape(self) -> None:
+        attn = MultiHeadAttention(d_model=D_MODEL, nhead=4, dropout=0.0)
+        x = torch.randn(B, T, D_MODEL)
+        out = attn(x)
+        assert out.shape == (B, T, D_MODEL)
+
+    @torch.no_grad()
+    def test_cross_attention_shape(self) -> None:
+        attn = MultiHeadAttention(
+            d_model=D_MODEL,
+            nhead=4,
+            dropout=0.0,
+            cross_attention=True,
+            use_rope=True,
+        )
+        q = torch.randn(B, 8, D_MODEL)
+        ctx = torch.randn(B, T, D_MODEL)
+        out = attn(q, context=ctx)
+        assert out.shape == (B, 8, D_MODEL)
+
+    def test_cross_attention_requires_context(self) -> None:
+        attn = MultiHeadAttention(
+            d_model=D_MODEL,
+            nhead=4,
+            dropout=0.0,
+            cross_attention=True,
+        )
+        x = torch.randn(B, 8, D_MODEL)
+        with pytest.raises(ValueError):
+            _ = attn(x)
+
+
+# -- SpatioTemporalEncoder ------------------------------------------------------
 
 
 class TestSpatioTemporalEncoder:
@@ -198,7 +250,7 @@ class TestSpatioTemporalEncoder:
         encoder.eval()
 
 
-# ── ActionClassifier ──────────────────────────────────────────────────────────
+# -- ActionClassifier -----------------------------------------------------------
 
 
 class TestActionClassifier:
@@ -222,7 +274,7 @@ class TestActionClassifier:
     def test_logits_are_unnormalised(
         self, classifier: ActionClassifier, dummy_input: torch.Tensor
     ) -> None:
-        # The head is a plain Linear — softmax probabilities should NOT sum to 1
+        # The head is a plain Linear - softmax probabilities should NOT sum to 1
         logits = classifier(dummy_input)
         row_sums = logits.softmax(dim=-1).sum(dim=-1)
         # After softmax they should sum to ~1, but raw logits sum should not
@@ -245,7 +297,7 @@ class TestActionClassifier:
         assert head_ids.isdisjoint(enc_ids), "Head parameters appear in encoder"
 
 
-# ── build_model factory ───────────────────────────────────────────────────────
+# -- build_model factory --------------------------------------------------------
 
 
 class TestBuildModel:
@@ -266,3 +318,40 @@ class TestBuildModel:
         logits = classifier(dummy)
         assert logits.shape == (B, NUM_CLASSES)
         assert not torch.isnan(logits).any()
+
+    @torch.no_grad()
+    def test_perceiver_rope_forward_after_build(self, minimal_cfg: dict) -> None:
+        cfg = {
+            "data": {
+                "num_joints": J,
+                "seq_len": T,
+                "num_classes": NUM_CLASSES,
+            },
+            "model": {
+                "in_features": F_IN,
+                "gat_hidden_dim": 64,
+                "gat_heads": 4,
+                "gat_dropout": 0.0,
+                "d_model": D_MODEL,
+                "nhead": 4,
+                "num_transformer_layers": 2,
+                "dim_feedforward": 256,
+                "transformer_dropout": 0.0,
+                "cls_std": 0.02,
+                "temporal_backbone": "perceiver",
+                "positional_encoding": "rope",
+                "num_latents": 16,
+                "num_perceiver_layers": 2,
+                "cross_attn_heads": 4,
+                "self_attn_heads": 4,
+                "perceiver_dim_feedforward": 256,
+                "perceiver_dropout": 0.0,
+                "rope_base": 10000.0,
+            },
+        }
+        _, classifier = build_model(cfg, EDGE_INDEX)
+        classifier.eval()
+        dummy = torch.randn(B, T, J, F_IN)
+        logits = classifier(dummy)
+        assert logits.shape == (B, NUM_CLASSES)
+        assert not torch.isnan(logits).any()

```

**Tool: bash**

**Input:**
```json
{
  "command": "git log --oneline -5",
  "description": "Show recent commit messages"
}
```

**Output:**
```
00a597b Remove LABELING_GUIDE.md and add live inference functionality with supporting utilities for real-time action recognition. Update README and COMMANDS.md to reflect new training and evaluation commands. Introduce tests for live pipeline components.
5349c71 Add pretrain_final.yaml: final pretraining config with Optuna best params and official val split
d32659b Add docs folder with labeling guide and command reference
f74ea39 Add non-technical labeling workflow guide for football injury dataset
d71a314 Add Optuna-based hyperparameter tuning for pretraining

```

---

## Assistant (Build · gpt-5.3-codex · 28.6s)

_Thinking:_

**Running full tests before commit**

**Tool: bash**

**Input:**
```json
{
  "command": "pytest tests/ -v",
  "description": "Run full test suite with verbosity"
}
```

**Output:**
```
zsh:1: command not found: pytest

```

---

## Assistant (Build · gpt-5.3-codex · 17.0s)

_Thinking:_

**Running tests with uv**

**Tool: bash**

**Input:**
```json
{
  "command": "uv run pytest tests/ -v",
  "description": "Run full test suite via uv"
}
```

**Output:**
```
============================= test session starts ==============================
platform linux -- Python 3.12.10, pytest-9.0.2, pluggy-1.6.0 -- /home/alan/mydev/college/football-injury-ai/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /home/alan/mydev/college/football-injury-ai
configfile: pyproject.toml
plugins: anyio-4.12.1
collecting ... collected 106 items

tests/test_anubis_loader.py::TestComputeAcceleration::test_output_shape PASSED [  0%]
tests/test_anubis_loader.py::TestComputeAcceleration::test_last_acc_frame_duplicates_previous PASSED [  1%]
tests/test_anubis_loader.py::TestComputeAcceleration::test_acc_is_finite_difference_of_vel PASSED [  2%]
tests/test_anubis_loader.py::TestComputeAcceleration::test_pos_vel_channels_unchanged PASSED [  3%]
tests/test_anubis_loader.py::TestComputeAcceleration::test_no_nan PASSED [  4%]
tests/test_anubis_loader.py::TestCroppingFunctions::test_random_crop_output_shape PASSED [  5%]
tests/test_anubis_loader.py::TestCroppingFunctions::test_center_crop_output_shape PASSED [  6%]
tests/test_anubis_loader.py::TestCroppingFunctions::test_random_crop_no_crop_when_equal PASSED [  7%]
tests/test_anubis_loader.py::TestCroppingFunctions::test_center_crop_no_crop_when_equal PASSED [  8%]
tests/test_anubis_loader.py::TestCroppingFunctions::test_center_crop_is_centred PASSED [  9%]
tests/test_anubis_loader.py::TestCroppingFunctions::test_random_crop_content_is_contiguous_window PASSED [ 10%]
tests/test_anubis_loader.py::TestFlipSequence::test_output_shape PASSED  [ 11%]
tests/test_anubis_loader.py::TestFlipSequence::test_double_flip_is_identity PASSED [ 12%]
tests/test_anubis_loader.py::TestFlipSequence::test_does_not_mutate_input PASSED [ 13%]
tests/test_anubis_loader.py::TestFlipSequence::test_x_axis_features_negated PASSED [ 14%]
tests/test_anubis_loader.py::TestFlipSequence::test_non_flip_features_unchanged PASSED [ 15%]
tests/test_anubis_loader.py::TestJointJitter::test_output_shape PASSED   [ 16%]
tests/test_anubis_loader.py::TestJointJitter::test_adds_noise PASSED     [ 16%]
tests/test_anubis_loader.py::TestJointJitter::test_zero_std_is_identity PASSED [ 17%]
tests/test_anubis_loader.py::TestJointJitter::test_does_not_mutate_input PASSED [ 18%]
tests/test_anubis_loader.py::TestSpeedPerturb::test_output_shape PASSED  [ 19%]
tests/test_anubis_loader.py::TestSpeedPerturb::test_output_shape_different_target_len PASSED [ 20%]
tests/test_anubis_loader.py::TestSpeedPerturb::test_no_nan PASSED        [ 21%]
tests/test_anubis_loader.py::TestSpeedPerturb::test_factor_1_is_near_identity PASSED [ 22%]
tests/test_anubis_loader.py::TestLoadAnubisTensors::test_output_shapes PASSED [ 23%]
tests/test_anubis_loader.py::TestLoadAnubisTensors::test_features_dtype PASSED [ 24%]
tests/test_anubis_loader.py::TestLoadAnubisTensors::test_labels_dtype PASSED [ 25%]
tests/test_anubis_loader.py::TestLoadAnubisTensors::test_missing_features_raises PASSED [ 26%]
tests/test_anubis_loader.py::TestLoadAnubisTensors::test_missing_labels_raises PASSED [ 27%]
tests/test_anubis_loader.py::TestLoadAnubisTensors::test_wrong_channel_count_raises PASSED [ 28%]
tests/test_anubis_loader.py::TestLoadAnubisTensors::test_wrong_ndim_raises PASSED [ 29%]
tests/test_anubis_loader.py::TestLoadAnubisTensors::test_5d_input_selects_primary_person PASSED [ 30%]
tests/test_anubis_loader.py::TestANUBISDataset::test_len PASSED          [ 31%]
tests/test_anubis_loader.py::TestANUBISDataset::test_item_shapes_no_crop PASSED [ 32%]
tests/test_anubis_loader.py::TestANUBISDataset::test_item_shapes_with_crop PASSED [ 33%]
tests/test_anubis_loader.py::TestANUBISDataset::test_label_dtype PASSED  [ 33%]
tests/test_anubis_loader.py::TestANUBISDataset::test_augmentation_changes_output PASSED [ 34%]
tests/test_anubis_loader.py::TestANUBISDataset::test_getitem_does_not_mutate_stored_tensor PASSED [ 35%]
tests/test_joint_config.py::TestJointNames::test_count PASSED            [ 36%]
tests/test_joint_config.py::TestJointNames::test_num_joints_constant PASSED [ 37%]
tests/test_joint_config.py::TestJointNames::test_all_strings PASSED      [ 38%]
tests/test_joint_config.py::TestJointNames::test_unique_names PASSED     [ 39%]
tests/test_joint_config.py::TestJointNames::test_pelvis_is_root PASSED   [ 40%]
tests/test_joint_config.py::TestSkeletonEdges::test_num_edges_constant_matches_list PASSED [ 41%]
tests/test_joint_config.py::TestSkeletonEdges::test_expected_edge_count PASSED [ 42%]
tests/test_joint_config.py::TestSkeletonEdges::test_no_self_loops PASSED [ 43%]
tests/test_joint_config.py::TestSkeletonEdges::test_node_indices_in_range PASSED [ 44%]
tests/test_joint_config.py::TestSkeletonEdges::test_no_duplicate_edges PASSED [ 45%]
tests/test_joint_config.py::TestBuildEdgeIndex::test_shape PASSED        [ 46%]
tests/test_joint_config.py::TestBuildEdgeIndex::test_dtype PASSED        [ 47%]
tests/test_joint_config.py::TestBuildEdgeIndex::test_node_range PASSED   [ 48%]
tests/test_joint_config.py::TestBuildEdgeIndex::test_bidirectional PASSED [ 49%]
tests/test_joint_config.py::TestBuildEdgeIndex::test_prebuilt_constant_matches PASSED [ 50%]
tests/test_joint_config.py::TestMirrorPairs::test_indices_in_range PASSED [ 50%]
tests/test_joint_config.py::TestMirrorPairs::test_no_self_pairs PASSED   [ 51%]
tests/test_joint_config.py::TestMirrorPairs::test_no_duplicate_joint_appearances PASSED [ 52%]
tests/test_joint_config.py::TestMirrorPairs::test_flip_feature_indices PASSED [ 53%]
tests/test_joint_config.py::TestAngleTriplets::test_joint_indices_in_range PASSED [ 54%]
tests/test_joint_config.py::TestAngleTriplets::test_no_degenerate_triplets PASSED [ 55%]
tests/test_joint_config.py::TestBodyRegions::test_num_regions_constant PASSED [ 56%]
tests/test_joint_config.py::TestBodyRegions::test_expected_region_count PASSED [ 57%]
tests/test_joint_config.py::TestBodyRegions::test_all_joint_indices_in_range PASSED [ 58%]
tests/test_joint_config.py::TestBodyRegions::test_no_empty_regions PASSED [ 59%]
tests/test_joint_config.py::TestBodyRegions::test_region_names_list_matches_dict PASSED [ 60%]
tests/test_live_pipeline.py::TestParseVideoSource::test_numeric_source_returns_int PASSED [ 61%]
tests/test_live_pipeline.py::TestParseVideoSource::test_path_or_url_returns_str PASSED [ 62%]
tests/test_live_pipeline.py::TestFeatureConstruction::test_normalise_pelvis_center PASSED [ 63%]
tests/test_live_pipeline.py::TestFeatureConstruction::test_velocity_and_acceleration_boundary_handling PASSED [ 64%]
tests/test_live_pipeline.py::TestFeatureConstruction::test_build_feature_window_shape_and_order PASSED [ 65%]
tests/test_live_pipeline.py::TestClassNames::test_default_fallback PASSED [ 66%]
tests/test_live_pipeline.py::TestClassNames::test_load_from_txt PASSED   [ 66%]
tests/test_live_pipeline.py::TestClassNames::test_load_from_yaml_dict PASSED [ 67%]
tests/test_live_pipeline.py::TestClassNames::test_load_from_json_list PASSED [ 68%]
tests/test_live_pipeline.py::TestSmoothProbs::test_first_step_returns_current PASSED [ 69%]
tests/test_live_pipeline.py::TestSmoothProbs::test_ema PASSED            [ 70%]
tests/test_live_pipeline.py::TestMediapipeConversion::test_output_shape PASSED [ 71%]
tests/test_live_pipeline.py::TestMediapipeConversion::test_pelvis_is_midpoint_of_hips PASSED [ 72%]
tests/test_live_pipeline.py::TestMediapipeConversion::test_direct_joint_mappings PASSED [ 73%]
tests/test_live_pipeline.py::TestMediapipeConversion::test_invalid_shape_raises PASSED [ 74%]
tests/test_model.py::TestPositionalEncoding::test_output_shape PASSED    [ 75%]
tests/test_model.py::TestPositionalEncoding::test_no_nan PASSED          [ 76%]
tests/test_model.py::TestPositionalEncoding::test_adds_positional_signal PASSED [ 77%]
tests/test_model.py::TestPositionalEncoding::test_different_positions_differ PASSED [ 78%]
tests/test_model.py::TestRotaryEmbedding::test_cos_sin_shape PASSED      [ 79%]
tests/test_model.py::TestRotaryEmbedding::test_rejects_odd_dimension PASSED [ 80%]
tests/test_model.py::TestMultiHeadAttention::test_self_attention_shape PASSED [ 81%]
tests/test_model.py::TestMultiHeadAttention::test_cross_attention_shape PASSED [ 82%]
tests/test_model.py::TestMultiHeadAttention::test_cross_attention_requires_context PASSED [ 83%]
tests/test_model.py::TestSpatioTemporalEncoder::test_output_shape PASSED [ 83%]
tests/test_model.py::TestSpatioTemporalEncoder::test_no_nan_in_output PASSED [ 84%]
tests/test_model.py::TestSpatioTemporalEncoder::test_no_inf_in_output PASSED [ 85%]
tests/test_model.py::TestSpatioTemporalEncoder::test_different_inputs_give_different_outputs PASSED [ 86%]
tests/test_model.py::TestSpatioTemporalEncoder::test_batch_independence PASSED [ 87%]
tests/test_model.py::TestSpatioTemporalEncoder::test_edge_index_is_buffer PASSED [ 88%]
tests/test_model.py::TestSpatioTemporalEncoder::test_edge_index_shape PASSED [ 89%]
tests/test_model.py::TestSpatioTemporalEncoder::test_has_no_unused_params_in_training_mode PASSED [ 90%]
tests/test_model.py::TestActionClassifier::test_output_shape PASSED      [ 91%]
tests/test_model.py::TestActionClassifier::test_no_nan_in_logits PASSED  [ 92%]
tests/test_model.py::TestActionClassifier::test_logits_are_unnormalised PASSED [ 93%]
tests/test_model.py::TestActionClassifier::test_encoder_is_accessible PASSED [ 94%]
tests/test_model.py::TestActionClassifier::test_encoder_and_classifier_share_params PASSED [ 95%]
tests/test_model.py::TestActionClassifier::test_head_not_in_encoder PASSED [ 96%]
tests/test_model.py::TestBuildModel::test_returns_encoder_and_classifier PASSED [ 97%]
tests/test_model.py::TestBuildModel::test_classifier_wraps_same_encoder PASSED [ 98%]
tests/test_model.py::TestBuildModel::test_forward_pass_after_build PASSED [ 99%]
tests/test_model.py::TestBuildModel::test_perceiver_rope_forward_after_build PASSED [100%]

=============================== warnings summary ===============================
.venv/lib/python3.12/site-packages/torch_geometric/llm/utils/backend_utils.py:26
  /home/alan/mydev/college/football-injury-ai/.venv/lib/python3.12/site-packages/torch_geometric/llm/utils/backend_utils.py:26: DeprecationWarning: `torch_geometric.distributed` has been deprecated since 2.7.0 and will no longer be maintained. For distributed training, refer to our tutorials on distributed training at https://pytorch-geometric.readthedocs.io/en/latest/tutorial/distributed.html or cuGraph examples at https://github.com/rapidsai/cugraph-gnn/tree/main/python/cugraph-pyg/cugraph_pyg/examples
    from torch_geometric.distributed import (

.venv/lib/python3.12/site-packages/torch/jit/_script.py:1480
.venv/lib/python3.12/site-packages/torch/jit/_script.py:1480
  /home/alan/mydev/college/football-injury-ai/.venv/lib/python3.12/site-packages/torch/jit/_script.py:1480: DeprecationWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

tests/test_model.py: 17 warnings
  /home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py:405: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
    self.transformer = nn.TransformerEncoder(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 106 passed, 20 warnings in 7.64s =======================

```

---

## Assistant (Build · gpt-5.3-codex · 18.3s)

_Thinking:_

**Validating architecture with smoke tests**

**Tool: bash**

**Input:**
```json
{
  "command": "uv run python module1/pretrain/model.py",
  "description": "Run baseline model smoke test"
}
```

**Output:**
```
/home/alan/mydev/college/football-injury-ai/module1/pretrain/model.py:405: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.transformer = nn.TransformerEncoder(
Encoder output shape    : torch.Size([4, 128])
Classifier output shape : torch.Size([4, 102])
Trainable parameters    : 296,550
model.py OK.

```

**Tool: bash**

**Input:**
```json
{
  "command": "uv run python - <<'PY'\nimport torch\nfrom utils.config import load_config\nfrom module1.data.joint_config import EDGE_INDEX, NUM_JOINTS\nfrom module1.pretrain.model import build_model\ncfg = load_config('configs/pretrain_perceiver_rope.yaml')\n_, clf = build_model(cfg, EDGE_INDEX)\nclf.eval()\nx = torch.randn(2, cfg['data']['seq_len'], NUM_JOINTS, cfg['model']['in_features'])\nwith torch.no_grad():\n    y = clf(x)\nprint('perceiver logits shape', tuple(y.shape))\nprint('nan', torch.isnan(y).any().item())\nPY",
  "description": "Run perceiver-rope forward smoke test"
}
```

**Output:**
```
23:23:54  INFO      module1.pretrain.model — Model built: 437734 trainable parameters
perceiver logits shape (2, 102)
nan False

```

---

## Assistant (Build · gpt-5.3-codex · 24.7s)

_Thinking:_

**Staging selective files excluding pyproject**

**Tool: bash**

**Input:**
```json
{
  "command": "git status --short",
  "description": "Confirm files before staging commit"
}
```

**Output:**
```
 M README.md
 M docs/COMMANDS.md
 M module1/pretrain/model.py
 M pyproject.toml
 M tests/test_model.py
?? configs/pretrain_perceiver_rope.yaml
?? uv.lock

```

---

## Assistant (Build · gpt-5.3-codex · 6.8s)

_Thinking:_

**Reviewing pyproject changes**

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/alan/mydev/college/football-injury-ai/pyproject.toml"
}
```

**Output:**
```
<path>/home/alan/mydev/college/football-injury-ai/pyproject.toml</path>
<type>file</type>
<content>1: [project]
2: name = "football-injury-ai"
3: version = "0.1.0"
4: description = "Real-time football player injury risk prediction using pose estimation and deep learning"
5: requires-python = ">=3.12"
6: dependencies = [
7:     # Core
8:     "numpy>=1.24",
9:     "pandas>=2.0",
10:     "pyyaml>=6.0",
11: 
12:     # Computer Vision & Pose Estimation
13:     "opencv-python>=4.8",
14:     "mediapipe>=0.10",
15: 
16:     # Deep Learning
17:     "torch>=2.1",
18:     "torchvision>=0.16",
19: 
20:     # Graph Neural Networks
21:     "torch-geometric>=2.4",
22:     # torch-scatter and torch-sparse installed separately:
23:     # pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
24: 
25:     # Visualisation
26:     "matplotlib>=3.7",
27:     "seaborn>=0.13",
28:     "networkx>=3.2",
29: 
30:     # Evaluation & Utilities
31:     "scikit-learn>=1.3",
32:     "tqdm>=4.65",
33:     "optuna>=3.6",
34: 
35:     # Notebooks
36:     "jupyter>=1.0",
37:     "ipykernel>=6.0",
38: 
39:     # Testing
40:     "pytest>=7.4",
41: ]
42: 
43: [project.optional-dependencies]
44: dev = [
45:     "pytest>=7.4",
46:     "black>=23.0",
47:     "ruff>=0.1",
48: ]
49: 
50: [build-system]
51: requires = ["hatchling"]
52: build-backend = "hatchling.build"
53: 
54: [tool.hatch.build.targets.wheel]
55: packages = ["module1", "module2", "module3", "fusion", "utils"]
56: 
57: [tool.pytest.ini_options]
58: testpaths = ["tests"]
59: python_files = ["test_*.py"]
60: python_classes = ["Test*"]
61: python_functions = ["test_*"]
62: 
63: [tool.ruff]
64: line-length = 88
65: target-version = "py312"
66: 
67: [tool.black]
68: line-length = 88
69: target-version = ["py312"]

(End of file - total 69 lines)
</content>
```

---

## Assistant (Build · gpt-5.3-codex · 18.6s)

_Thinking:_

**Deciding commit inclusion of lock and config**

**Tool: bash**

**Input:**
```json
{
  "command": "git add README.md docs/COMMANDS.md module1/pretrain/model.py tests/test_model.py configs/pretrain_perceiver_rope.yaml pyproject.toml",
  "description": "Stage perceiver and RoPE architecture changes"
}
```

---

## Assistant (Build · gpt-5.3-codex · 4.3s)

**Tool: bash**

**Input:**
```json
{
  "command": "git commit -m \"Add Perceiver+RoPE temporal backbone for pretraining\"",
  "description": "Commit architecture and config updates"
}
```

**Output:**
```
[master 36891c8] Add Perceiver+RoPE temporal backbone for pretraining
 6 files changed, 645 insertions(+), 183 deletions(-)
 create mode 100644 configs/pretrain_perceiver_rope.yaml

```

---

## Assistant (Build · gpt-5.3-codex · 9.4s)

_Thinking:_

**Verifying committed files count**

**Tool: bash**

**Input:**
```json
{
  "command": "git status --short",
  "description": "Check remaining changes after commit"
}
```

**Output:**
```
?? uv.lock

```

---

## Assistant (Build · gpt-5.3-codex · 15.4s)

_Thinking:_

**Confirming staged files before push**

**Tool: bash**

**Input:**
```json
{
  "command": "git push",
  "description": "Push committed changes to remote"
}
```

**Output:**
```
To github.com:alanfrancis442/football-injury-ai.git
   00a597b..36891c8  master -> master

```

---

## Assistant (Build · gpt-5.3-codex · 6.6s)

All good — I ran a deep check, and everything passed.

What I verified:
- Full test suite: `uv run pytest tests/ -v`  
  - Result: **106 passed**
- Baseline smoke test: `uv run python module1/pretrain/model.py`  
  - Result: forward pass OK, no NaNs
- Perceiver+RoPE smoke test from config:
  - Loaded `configs/pretrain_perceiver_rope.yaml`
  - Built model and ran forward
  - Result: logits shape `(2, 102)`, no NaNs

I then committed and pushed the architecture changes.

Commit:
- `36891c8` — Add Perceiver+RoPE temporal backbone for pretraining

Pushed to:
- `origin/master`

Included in commit:
- `module1/pretrain/model.py`
- `tests/test_model.py`
- `configs/pretrain_perceiver_rope.yaml`
- `docs/COMMANDS.md`
- `README.md`
- `pyproject.toml` (wheel package selection fix)

Not committed:
- `uv.lock` (left untracked)

You can now train the new architecture on the other machine with:
```bash
uv run module1/pretrain/train.py --config configs/pretrain_perceiver_rope.yaml
```

---

