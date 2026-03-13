# module1/pretrain/tune_optuna.py
#
# Stage 1 — Optuna Hyperparameter Search for ANUBIS Pretraining
# ───────────────────────────────────────────────────────────────
# Runs Optuna-based hyperparameter optimisation over the Stage 1
# ActionClassifier training loop.
#
# Usage:
#   python module1/pretrain/tune_optuna.py \
#       --config configs/experiments/module1/baseline_transformer.yaml
#
# Notes:
#   - This script does NOT save per-trial model checkpoints.
#   - It optimises validation top-1 accuracy and stores the best parameters
#     and a full best-config YAML for a follow-up full training run.

from __future__ import annotations

import argparse
import copy
import os
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import yaml

from module1.data.anubis_loader import build_dataloaders
from module1.data.joint_config import EDGE_INDEX
from module1.pretrain.experiments import (
    ensure_experiment_dirs,
    prepare_experiment_config,
    resolve_artifact_path,
    set_seed,
)
from module1.pretrain.model import build_model
from module1.pretrain.train import _make_lr_lambda, evaluate, train_one_epoch
from utils.config import load_config
from utils.logger import get_logger

log = get_logger(__name__)


# ── Search-space helpers ──────────────────────────────────────────────────────


def _get_space_spec(search_space: Dict, key: str, defaults: Dict) -> Dict:
    """
    Merge one search-space entry with defaults.

    Parameters
    ----------
    search_space : dict
    key          : str
    defaults     : dict

    Returns
    -------
    spec : dict
    """
    spec = copy.deepcopy(defaults)
    spec.update(search_space.get(key, {}))
    return spec


def sample_hparams(trial: Any, search_space: Dict) -> Dict:
    """
    Sample one hyperparameter set from Optuna.

    Parameters
    ----------
    trial        : optuna.trial.Trial
    search_space : dict

    Returns
    -------
    params : dict
    """
    lr_spec = _get_space_spec(
        search_space,
        "learning_rate",
        {"low": 1e-4, "high": 2e-3, "log": True},
    )
    wd_spec = _get_space_spec(
        search_space,
        "weight_decay",
        {"low": 1e-5, "high": 1e-3, "log": True},
    )
    ls_spec = _get_space_spec(
        search_space,
        "label_smoothing",
        {"low": 0.0, "high": 0.15},
    )
    gat_do_spec = _get_space_spec(
        search_space,
        "gat_dropout",
        {"low": 0.05, "high": 0.35},
    )
    tr_do_spec = _get_space_spec(
        search_space,
        "transformer_dropout",
        {"low": 0.05, "high": 0.35},
    )
    flip_spec = _get_space_spec(
        search_space,
        "flip_prob",
        {"low": 0.1, "high": 0.6},
    )
    jitter_spec = _get_space_spec(
        search_space,
        "joint_jitter_std",
        {"low": 1e-3, "high": 2e-2, "log": True},
    )
    speed_spec = _get_space_spec(
        search_space,
        "speed_perturb",
        {"choices": [True, False]},
    )

    params = {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            float(lr_spec["low"]),
            float(lr_spec["high"]),
            log=bool(lr_spec.get("log", False)),
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            float(wd_spec["low"]),
            float(wd_spec["high"]),
            log=bool(wd_spec.get("log", False)),
        ),
        "label_smoothing": trial.suggest_float(
            "label_smoothing",
            float(ls_spec["low"]),
            float(ls_spec["high"]),
        ),
        "gat_dropout": trial.suggest_float(
            "gat_dropout",
            float(gat_do_spec["low"]),
            float(gat_do_spec["high"]),
        ),
        "transformer_dropout": trial.suggest_float(
            "transformer_dropout",
            float(tr_do_spec["low"]),
            float(tr_do_spec["high"]),
        ),
        "flip_prob": trial.suggest_float(
            "flip_prob",
            float(flip_spec["low"]),
            float(flip_spec["high"]),
        ),
        "joint_jitter_std": trial.suggest_float(
            "joint_jitter_std",
            float(jitter_spec["low"]),
            float(jitter_spec["high"]),
            log=bool(jitter_spec.get("log", False)),
        ),
        "speed_perturb": trial.suggest_categorical(
            "speed_perturb",
            list(speed_spec.get("choices", [True, False])),
        ),
    }
    return params


def apply_hparams(cfg: Dict, params: Dict) -> None:
    """
    Apply sampled hyperparameters to a config dict in-place.

    Parameters
    ----------
    cfg    : dict
    params : dict
    """
    cfg["training"]["learning_rate"] = float(params["learning_rate"])
    cfg["training"]["weight_decay"] = float(params["weight_decay"])
    cfg["training"]["label_smoothing"] = float(params["label_smoothing"])

    cfg["model"]["gat_dropout"] = float(params["gat_dropout"])
    cfg["model"]["transformer_dropout"] = float(params["transformer_dropout"])

    cfg["augmentation"]["flip_prob"] = float(params["flip_prob"])
    cfg["augmentation"]["joint_jitter_std"] = float(params["joint_jitter_std"])
    cfg["augmentation"]["speed_perturb"] = bool(params["speed_perturb"])


def save_yaml(path: str, payload: Dict) -> None:
    """Save a dictionary to YAML."""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


# ── Optuna objective ──────────────────────────────────────────────────────────


def make_objective(base_cfg: Dict, opt_cfg: Dict, device: torch.device) -> Callable:
    """
    Build an Optuna objective closure.

    Parameters
    ----------
    base_cfg : dict
    opt_cfg  : dict
    device   : torch.device

    Returns
    -------
    objective : callable
    """
    trial_epochs = int(opt_cfg.get("trial_epochs", 25))
    seed = int(opt_cfg.get("seed", 42))
    num_workers = int(opt_cfg.get("num_workers", 4))
    pin_memory = bool(opt_cfg.get("pin_memory", True))
    search_space = opt_cfg.get("search_space", {})

    def objective(trial: Any) -> float:
        import optuna

        cfg = copy.deepcopy(base_cfg)
        cfg["training"]["epochs"] = trial_epochs

        sampled = sample_hparams(trial, search_space)
        apply_hparams(cfg, sampled)

        # Keep trial data split deterministic for fair trial-to-trial comparison.
        set_seed(seed)
        train_loader, val_loader = build_dataloaders(
            cfg,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        edge_index = EDGE_INDEX.to(device)
        _, classifier = build_model(cfg, edge_index)
        classifier = classifier.to(device)

        optimiser = torch.optim.AdamW(
            classifier.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimiser,
            lr_lambda=_make_lr_lambda(
                int(cfg["training"].get("warmup_epochs", 5)),
                int(cfg["training"]["epochs"]),
            ),
        )

        criterion = nn.CrossEntropyLoss(
            label_smoothing=float(cfg["training"].get("label_smoothing", 0.1))
        )
        eval_criterion = nn.CrossEntropyLoss()
        grad_clip = float(cfg["training"].get("grad_clip_norm", 1.0))

        best_val_top1 = 0.0
        best_val_top5 = 0.0
        best_val_macro_f1 = 0.0
        best_val_loss = float("inf")
        trn_loss = 0.0
        trn_acc = 0.0

        for epoch in range(1, trial_epochs + 1):
            trn_loss, trn_acc = train_one_epoch(
                classifier,
                train_loader,
                optimiser,
                criterion,
                device,
                grad_clip,
            )

            val_metrics = evaluate(
                classifier,
                val_loader,
                eval_criterion,
                device,
            )

            scheduler.step()

            if val_metrics["val_top1"] > best_val_top1:
                best_val_top1 = float(val_metrics["val_top1"])
                best_val_top5 = float(val_metrics["val_top5"])
                best_val_macro_f1 = float(val_metrics["val_macro_f1"])
                best_val_loss = float(val_metrics["val_loss"])

            trial.report(best_val_top1, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("best_val_top1", float(best_val_top1))
        trial.set_user_attr("best_val_top5", float(best_val_top5))
        trial.set_user_attr("best_val_macro_f1", float(best_val_macro_f1))
        trial.set_user_attr("best_val_loss", float(best_val_loss))
        trial.set_user_attr("final_trn_loss", float(trn_loss))
        trial.set_user_attr("final_trn_acc", float(trn_acc))

        log.info(
            "Trial %d done | best val top1=%.4f top5=%.4f macro_f1=%.4f loss=%.4f",
            trial.number,
            best_val_top1,
            best_val_top5,
            best_val_macro_f1,
            best_val_loss,
        )
        return float(best_val_top1)

    return objective


# ── Main entry point ──────────────────────────────────────────────────────────


def main(
    config_path: str,
    n_trials_override: Optional[int],
    timeout_override: Optional[int],
) -> None:
    raw_cfg = load_config(config_path)
    cfg, experiment = prepare_experiment_config(raw_cfg, config_path=config_path)
    ensure_experiment_dirs(experiment)
    opt_cfg = cfg.get("optuna", {})

    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is not installed. Install it with: pip install optuna"
        ) from exc

    seed = int(opt_cfg.get("seed", 42))
    set_seed(seed)

    n_trials = (
        int(n_trials_override)
        if n_trials_override is not None
        else int(opt_cfg.get("n_trials", 30))
    )
    timeout_sec = timeout_override
    if timeout_sec is None:
        timeout_value = opt_cfg.get("timeout_sec", None)
        timeout_sec = int(timeout_value) if timeout_value is not None else None

    study_name = str(opt_cfg.get("study_name", "anubis_pretrain_hpo"))
    storage = opt_cfg.get("storage", None)
    load_if_exists = bool(opt_cfg.get("load_if_exists", True))
    direction = str(opt_cfg.get("direction", "maximize"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    log.info(
        "Experiment: group=%s name=%s slug=%s",
        experiment.group,
        experiment.name,
        experiment.slug,
    )
    log.info(
        "Starting Optuna study '%s' | trials=%d | timeout=%s",
        study_name,
        n_trials,
        str(timeout_sec),
    )

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_objective(cfg, opt_cfg, device)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)

    try:
        best_trial = study.best_trial
    except ValueError as exc:
        raise RuntimeError(
            "No completed trial found. Check your data/configuration."
        ) from exc
    best_params = dict(best_trial.params)
    best_value = (
        float(best_trial.value) if best_trial.value is not None else float("nan")
    )
    log.info("Best trial: %d", best_trial.number)
    log.info("Best value (val_top1): %.4f", best_value)
    log.info("Best params: %s", best_params)

    best_cfg = copy.deepcopy(cfg)
    apply_hparams(best_cfg, best_params)
    best_cfg.setdefault("experiment", {})["name"] = (
        f"{best_cfg['experiment'].get('name', 'baseline_transformer')}_optuna_best"
    )

    best_params_path = resolve_artifact_path(
        experiment.log_dir,
        str(opt_cfg.get("output_best_params", "optuna_best_params.yaml")),
    )
    best_cfg_path = resolve_artifact_path(
        experiment.log_dir,
        str(opt_cfg.get("output_best_config", "pretrain_optuna_best.yaml")),
    )

    save_yaml(
        best_params_path,
        {
            "study_name": study_name,
            "best_trial_number": int(best_trial.number),
            "best_value": best_value,
            "best_params": best_params,
            "best_user_attrs": dict(best_trial.user_attrs),
        },
    )
    save_yaml(best_cfg_path, best_cfg)

    log.info("Saved best params → %s", best_params_path)
    log.info("Saved best config → %s", best_cfg_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter search for ANUBIS pretraining"
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/module1/baseline_transformer.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override optuna.n_trials from config",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=None,
        help="Override optuna.timeout_sec from config",
    )
    args = parser.parse_args()
    main(args.config, args.n_trials, args.timeout_sec)
