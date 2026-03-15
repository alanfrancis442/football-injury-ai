# Module 1 Restart Plan

Last updated: 2026-03-15

## Status

We are restarting Module 1 from scratch on `main`. The previous work has been preserved on a separate branch.

## Why We Are Restarting

The previous model showed acceptable benchmark-style results but did not behave reliably on real football footage. The new version should prioritize:

- fewer false positives
- stronger real-world generalization
- clear understanding of what the player is currently doing
- fast enough inference for practical use on a local `RTX 4060`

## Confirmed Decisions

- Build only Module 1 for now.
- Start from scratch.
- Do not reuse old pretrained weights.
- Use football footage from YouTube for early data collection.
- Support both broadcast and tactical footage.
- Start with one selected player per clip.
- Start offline first, then consider real-time later.
- Compare two model families:
  1. `GNN/ST-GCN + TCN`
  2. `GNN + lightweight Transformer`
- Optimize for low false positives, not just raw accuracy.

## Module 1 v1 Goal

Given a short football clip focused on one player, predict:

- current action
- biomechanical risk pattern
- likely body region
- confidence

Direct injury probability is deferred to a later stage.

## v1 Non-Goals

The first version should not try to do all of the following:

- full multi-player match-wide analysis
- fully automatic broadcast-wide tracking
- full fusion with Module 2 and Module 3
- clinically final injury probability prediction

## Working Principle

The first model must understand the player's current movement before it can make a useful risk judgment.

This means the pipeline should learn:

1. what the player is doing
2. whether the movement shows a risky biomechanical pattern
3. which body region is most likely involved

## Data Strategy

### Sources

- YouTube broadcast clips
- YouTube tactical / analysis clips

### Clip Policy

Collect short windows centered on football actions such as:

- run / sprint
- deceleration
- cut / change of direction
- jump / landing
- kick
- tackle / post-contact recovery

### Player Policy

- Focus on one target player per clip.
- For v1, manual player selection is acceptable.
- Keep source metadata for each clip so train/val/test splits stay clean.

### Split Policy

Do not split randomly by windows from the same source.

Instead, split by:

- source video
- match
- player identity when possible
- camera context

Keep a small untouched demo set from day one.

## Label Schema

### Action Labels

- idle
- walk_jog
- run_sprint
- decelerate
- cut_change_direction
- jump
- land
- kick
- tackle_contact
- recover_turn

### Risk Pattern Labels

- none
- knee_valgus
- hip_drop
- stride_asymmetry
- stiff_landing
- ankle_collapse
- trunk_lean
- post_contact_instability
- fatigue_compensation
- uncertain_occluded

### Body Region Labels

- none
- left_knee
- right_knee
- left_ankle
- right_ankle
- left_hamstring
- right_hamstring
- left_hip_groin
- right_hip_groin
- spine
- other

## Modeling Plan

### Approach A: `GNN/ST-GCN + TCN`

Role:
- stable baseline
- simpler and easier to calibrate
- good candidate for lower false positives

Expected strengths:
- easier training on limited or noisy football data
- faster iteration
- strong first benchmark for real-match clips

Expected weaknesses:
- may miss some long-range temporal dependencies

### Approach B: `GNN + lightweight Transformer`

Role:
- higher-capacity comparison model
- literature-aligned temporal model for skeleton sequences

Expected strengths:
- better global temporal reasoning
- may capture longer movement context
- matches the strongest Module 1 paper direction

Expected weaknesses:
- more sensitive to noisy pose/tracking
- may overfit more easily with limited football-specific labels
- may require more calibration to avoid false alerts

## Pretraining Strategy

Because we are starting from scratch:

- do not reuse the old ANUBIS checkpoint
- pretrain first on motion understanding rather than direct injury prediction
- fine-tune on football-specific action + pattern + region labels

Current preference:
- move toward football-specific data as early as possible
- prioritize domain match over benchmark-only performance

## Evaluation Strategy

### Primary Metrics

- false positives per minute
- precision of risky detections
- action F1
- risk-pattern F1
- body-region F1

### Secondary Metrics

- inference speed on `RTX 4060`
- calibration
- consistency across broadcast vs tactical footage

### Qualitative Evaluation

Keep several untouched real-match clips for final qualitative comparison.

## Selection Rule Between the Two Models

The final v1 model should not be chosen by accuracy alone.

Select the winner based on:

- lower false positives
- better action understanding
- better region localization
- more stable behavior on unseen football clips
- acceptable speed on local GPU

## Key Risks

- unstable pose estimation in broadcast footage
- occlusion and player switching
- mismatch between proxy biomechanical labels and true injury outcomes
- data leakage from poor split design
- overfitting due to limited football-specific labels

## Practical Scope for v1

What v1 should do:

- process offline football clips
- focus on one selected player
- identify current action
- identify risky biomechanical patterns
- identify likely body region
- keep false positives low

What v1 should not do yet:

- analyze every player in the frame automatically
- provide final medical-grade injury prediction
- replace the full multi-module system design

## Immediate Next Steps

1. Define a clean folder structure for football clips and labels.
2. Collect the first batch of YouTube football clips.
3. Benchmark pose quality on broadcast and tactical footage.
4. Finalize the initial labeling template.
5. Build the preprocessing pipeline for one-player clips.
6. Train the `GNN/ST-GCN + TCN` baseline.
7. Train the `GNN + lightweight Transformer` comparison model.
8. Compare both models using real football evaluation criteria.
9. Choose the better v1 model based on low false positives and behavior on unseen clips.

## One-Sentence Summary

We are rebuilding Module 1 as a football-specific, low-false-positive biomechanical video model that first understands what the player is doing, then predicts risk pattern and body region, while comparing both a safer `GNN + TCN` baseline and a more expressive `GNN + Transformer` model.
