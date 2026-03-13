# module1/pretrain/experiments.py
#
# Stage 1 — Experiment Metadata Helpers
# ─────────────────────────────────────
# Normalises experiment names, scopes checkpoints/logs by experiment group, sets
# random seeds, and writes small JSON artifacts for repeatable comparisons.

from __future__ import annotations

import copy
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


DEFAULT_EXPERIMENT_GROUP = "module1"
DEFAULT_EXPERIMENT_NAME = "default"


@dataclass(frozen=True)
class ExperimentMetadata:
    """Resolved experiment metadata and output locations."""

    group: str  # logical experiment collection name
    name: str  # user-facing experiment name
    slug: str  # filesystem-safe experiment identifier
    checkpoint_dir: str  # scoped checkpoint directory
    log_dir: str  # scoped log directory


def _slugify(value: str) -> str:
    """Convert a free-form experiment name into a filesystem-safe slug."""
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered)
    slug = slug.strip("_")
    return slug or DEFAULT_EXPERIMENT_NAME


def _default_name_from_config_path(config_path: Optional[str]) -> str:
    """Fallback experiment name when the config omits one."""
    if config_path is None:
        return DEFAULT_EXPERIMENT_NAME
    stem = os.path.splitext(os.path.basename(config_path))[0]
    return stem or DEFAULT_EXPERIMENT_NAME


def resolve_experiment_metadata(
    cfg: Dict,
    config_path: Optional[str] = None,
) -> ExperimentMetadata:
    """Resolve experiment metadata and scoped output directories."""
    exp_cfg = cfg.get("experiment", {})
    paths_cfg = cfg.get("paths", {})

    group = str(exp_cfg.get("group", DEFAULT_EXPERIMENT_GROUP)).strip()
    if not group:
        group = DEFAULT_EXPERIMENT_GROUP

    name = str(exp_cfg.get("name", _default_name_from_config_path(config_path))).strip()
    if not name:
        name = _default_name_from_config_path(config_path)

    slug = _slugify(name)

    base_checkpoint_dir = str(paths_cfg.get("checkpoint_dir", "outputs/checkpoints"))
    base_log_dir = str(paths_cfg.get("log_dir", "outputs/logs"))

    checkpoint_dir = os.path.join(os.path.normpath(base_checkpoint_dir), group, slug)
    log_dir = os.path.join(os.path.normpath(base_log_dir), group, slug)

    return ExperimentMetadata(
        group=group,
        name=name,
        slug=slug,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )


def prepare_experiment_config(
    cfg: Dict,
    config_path: Optional[str] = None,
) -> Tuple[Dict, ExperimentMetadata]:
    """Copy a config and inject resolved experiment metadata/paths."""
    scoped_cfg = copy.deepcopy(cfg)
    metadata = resolve_experiment_metadata(scoped_cfg, config_path=config_path)

    exp_cfg = scoped_cfg.setdefault("experiment", {})
    exp_cfg["group"] = metadata.group
    exp_cfg["name"] = metadata.name
    exp_cfg["slug"] = metadata.slug

    paths_cfg = scoped_cfg.setdefault("paths", {})
    paths_cfg["checkpoint_dir"] = metadata.checkpoint_dir
    paths_cfg["log_dir"] = metadata.log_dir

    return scoped_cfg, metadata


def ensure_experiment_dirs(metadata: ExperimentMetadata) -> None:
    """Create the experiment checkpoint and log directories."""
    os.makedirs(metadata.checkpoint_dir, exist_ok=True)
    os.makedirs(metadata.log_dir, exist_ok=True)


def resolve_artifact_path(base_dir: str, path_or_name: str) -> str:
    """Resolve a filename relative to an experiment directory."""
    if os.path.isabs(path_or_name):
        return path_or_name
    if os.path.dirname(path_or_name):
        return path_or_name
    return os.path.join(base_dir, path_or_name)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    """Write a JSON artifact with stable formatting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def set_seed(seed: int) -> None:
    """Set global random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
