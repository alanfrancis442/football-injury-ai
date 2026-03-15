# Module 1 Research Notes

Last updated: 2026-03-15

## Purpose

This file records the research findings, old log review, and architecture reasoning used to restart Module 1.

## Sources Reviewed

### System Design Document

- `/home/alan/Downloads/football_injury_ai_system.md`

### Papers

- `/home/alan/Downloads/research_papers/mini_project/fnins-18-1353257.pdf`
- `/home/alan/Downloads/research_papers/mini_project/s41598-025-21613-2.pdf`
- `/home/alan/Downloads/research_papers/mini_project/fphys-14-1174525.pdf`

### Old Training Artifacts

- `outputs/logs/run.log`
- `outputs/logs/pretrain_optuna_best.yaml`
- `outputs/logs/optuna_best_params.yaml`

## What the System Design Document Says

The full project is a multi-module football injury AI system. Module 1 is the biomechanical video component.

### Module Roles

- Module 1: biomechanical risk monitoring from video and pose
- Module 2: workload / training-log injury risk
- Module 3: historical / chronic risk profile
- Fusion engine: combines all risk sources later

### Main Implication for the Restart

Module 1 should not try to solve the whole injury problem alone. It should become a strong biomechanical signal generator that detects:

- current movement
- risky movement patterns
- likely body region under stress
- short-term biomechanical warning signs

### Key Football-Specific Challenges

The document makes clear that football is hard because of:

- multiple players in view
- camera motion
- occlusion
- contact events
- nonlinear risk accumulation
- highly individualized responses

### Important Precursor Ideas Mentioned

Module 1 is especially motivated to detect patterns such as:

- hip drop
- knee valgus
- stride asymmetry
- trunk lean
- ankle instability
- fatigue-related compensation

## Paper 1: Zhu et al. (2024)

File:
`/home/alan/Downloads/research_papers/mini_project/fnins-18-1353257.pdf`

### Why It Matters

This is the strongest direct inspiration for Module 1.

### Main Takeaways

- Transformer models global temporal dependencies in skeleton sequences.
- GNN models local anatomical and joint relationships.
- GAN-based augmentation is proposed to improve robustness.
- The paper reports strong motion classification performance and argues that combining temporal and spatial modeling helps injury-prevention-related analysis.

### What We Should Use

Useful idea:
- combine temporal reasoning and joint-graph reasoning for skeleton motion

Most relevant architecture takeaway:
- `GNN + Transformer` is a valid Module 1 research direction

### Limitation

This is mainly evidence for skeletal motion analysis and classification, not a direct proof of football injury prediction in messy broadcast footage.

## Paper 2: Ye et al. (2023)

File:
`/home/alan/Downloads/research_papers/mini_project/fphys-14-1174525.pdf`

### Why It Matters

This paper is more relevant to workload-based injury prediction than to video biomechanics, but its conceptual lessons are important.

### Main Takeaways

- injury risk relationships are nonlinear
- time-series image encodings such as `GASF`, `GADF`, `MTF`, and `RP` can improve representation
- `GASF + DCAE + DNN` performed best in their study
- the paper emphasizes better generalization than simpler regression-style methods

### What We Should Use

Useful idea:
- do not trust simple linear or shallow baselines just because they are easy
- generalization matters more than optimistic internal metrics

### Limitation

This paper is based on training-load style data and is more useful for Module 2 than Module 1.

## Paper 3: Fang and Chen (2025)

File:
`/home/alan/Downloads/research_papers/mini_project/s41598-025-21613-2.pdf`

### Why It Matters

This paper supports graph-based injury modeling and transfer learning when target sport data is limited.

### Main Takeaways

- temporal graph encoding helps represent structured time-series risk data
- GNN can model interaction-aware risk patterns
- transfer learning helps when target-domain data is small
- small-sample performance can improve with cross-domain knowledge transfer

### What We Should Use

Useful idea:
- transfer learning and pretraining matter when football-specific labels are scarce

### Limitation

This is not a direct single-player football broadcast biomechanics solution. It is more of a graph-based injury-prediction framework than a direct Module 1 deployment recipe.

## What the Old Logs Suggest

We reviewed:

- `outputs/logs/run.log`
- `outputs/logs/pretrain_optuna_best.yaml`
- `outputs/logs/optuna_best_params.yaml`

### Finding 1: The Old Validation Setup Was Likely Too Optimistic

The saved best config still used:

- `has_separate_val: false`

from:

- `outputs/logs/pretrain_optuna_best.yaml`

This suggests the strongest reported result was based on a random split setup, not a truly separate validation scenario.

### Finding 2: Performance Dropped Hard on More Realistic Evaluation

On the easier setup, evaluation reached about:

- `Top-1 accuracy: 0.7180`
- `Macro F1: 0.7167`

recorded in:

- `outputs/logs/run.log`

On the more realistic separate validation setup, evaluation dropped to about:

- `Top-1 accuracy: 0.2686`
- `Macro F1: 0.2569`

also recorded in:

- `outputs/logs/run.log`

Later runs stayed in a similar weak range, roughly `0.25-0.29` top-1 on separate validation, despite improved training metrics.

### Finding 3: The Main Failure Looks Like Generalization

Working interpretation:

- the old system likely learned benchmark-friendly patterns
- it did not transfer well to truly separate data
- domain mismatch was probably a bigger issue than pure model size

### Finding 4: Speed Was Probably Not the Core Problem

The older model was not very large in some runs:

- about `296550` trainable parameters according to `outputs/logs/run.log`

This suggests the central failure was not simply latency. The harder problem was reliable transfer to real football conditions.

## What the Research Means for the Restart

### Conclusion 1

Do not optimize for benchmark accuracy alone.

### Conclusion 2

Build around real football footage from the beginning.

### Conclusion 3

Make action understanding a first-class task, not just a side product.

### Conclusion 4

Favor low false positives over aggressive alerting.

### Conclusion 5

Compare one stable baseline and one higher-capacity model instead of betting everything on one architecture immediately.

## Why We Are Testing Two Model Families

### `GNN/ST-GCN + TCN`

Reason to test:
- simpler and usually more stable
- often easier to calibrate on small or noisy data
- strong baseline for low-false-positive behavior

Expected role:
- safer first benchmark
- good control model for real football footage

### `GNN + lightweight Transformer`

Reason to test:
- strongest fit to the Zhu paper
- better global temporal reasoning
- more expressive if the pose and labels are good enough

Expected role:
- higher-upside comparison model
- useful if longer-range motion context really matters

## Current Working Hypothesis

The best Module 1 model will be the one that best balances:

- action understanding
- pattern localization
- body-region localization
- low false positives
- stable behavior on unseen real-footage clips

It may or may not be the Transformer-based model.

## User Decisions Captured So Far

The following product choices are already decided:

- restart from scratch
- use actual football match clips for testing
- use both broadcast and tactical footage
- run development on local `RTX 4060`
- prefer fewer false positives
- make the model understand what the player is doing on the field
- begin with one selected player per clip
- compare both model families:
  1. `GNN/ST-GCN + TCN`
  2. `GNN + lightweight Transformer`

## Risks to Watch During Implementation

- pose instability in broadcast footage
- occlusion and player switching
- mismatch between proxy labels and true injury events
- poor data splitting leading to leakage
- overfitting because football-specific labeled data is limited

## Practical Research Consequences

Based on all reviewed material, the best restart strategy is:

- start offline
- use one player per clip
- build football-specific data early
- evaluate on untouched real-match clips
- compare the two model families under the same pipeline
- select the winner by real behavior, not just accuracy

## Short Summary

The research supports using graph-based skeleton modeling for Module 1, strongly justifies testing a `GNN + Transformer` branch, and also strongly warns against trusting optimistic validation or benchmark-style accuracy. The restart should therefore emphasize football-specific data, clean split design, action understanding, and low-false-positive evaluation while comparing both a stable baseline and a higher-capacity temporal model.
