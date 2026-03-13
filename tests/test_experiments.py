# tests/test_experiments.py
#
# Unit tests for module1/pretrain/experiments.py
# -----------------------------------------------
# Covers experiment scoping, slug generation, and output directory resolution.
#
# Run with: pytest tests/test_experiments.py -v

from __future__ import annotations

from module1.pretrain.experiments import (
    prepare_experiment_config,
    resolve_artifact_path,
)


class TestExperimentMetadata:
    def test_prepare_config_scopes_paths(self) -> None:
        cfg = {
            "paths": {
                "checkpoint_dir": "outputs/checkpoints",
                "log_dir": "outputs/logs",
            },
            "experiment": {
                "group": "module1",
                "name": "Baseline Transformer",
            },
        }
        scoped_cfg, meta = prepare_experiment_config(
            cfg,
            config_path="configs/experiments/module1/baseline_transformer.yaml",
        )
        assert meta.group == "module1"
        assert meta.slug == "baseline_transformer"
        assert scoped_cfg["paths"]["checkpoint_dir"].endswith(
            "module1/baseline_transformer"
        )
        assert scoped_cfg["paths"]["log_dir"].endswith("module1/baseline_transformer")

    def test_resolve_artifact_path_joins_relative_name(self) -> None:
        path = resolve_artifact_path(
            "outputs/logs/module1/baseline_transformer", "metrics.json"
        )
        assert path.endswith("outputs/logs/module1/baseline_transformer/metrics.json")

    def test_config_path_stem_used_when_name_missing(self) -> None:
        cfg = {
            "paths": {
                "checkpoint_dir": "outputs/checkpoints",
                "log_dir": "outputs/logs",
            }
        }
        _, meta = prepare_experiment_config(
            cfg,
            config_path="configs/experiments/module1/tcn_ablation.yaml",
        )
        assert meta.slug == "tcn_ablation"
