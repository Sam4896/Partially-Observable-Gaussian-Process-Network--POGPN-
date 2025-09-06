import os
from typing import Dict, Any

import yaml


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config_to_yaml(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
