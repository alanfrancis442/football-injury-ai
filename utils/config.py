# utils/config.py
#
# Config loader — reads configs/module1.yaml and returns a plain dict.
# Import this in any script that needs hyperparameters.
#
# Usage:
#   from utils.config import load_config
#   cfg = load_config("configs/module1.yaml")
#   lr  = cfg["training"]["learning_rate"]

import os
import yaml


def load_config(config_path: str) -> dict:
    """
    Load a YAML config file and return it as a nested dict.

    Parameters
    ----------
    config_path : str — path to the .yaml file (relative to project root)

    Raises
    ------
    FileNotFoundError if the config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            "Make sure you are running commands from the project root directory."
        )

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg
