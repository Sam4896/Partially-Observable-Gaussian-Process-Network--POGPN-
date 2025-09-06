from __future__ import annotations

from typing import Dict, Any, TYPE_CHECKING, Tuple

import importlib
from botorch.utils.transforms import normalize

if TYPE_CHECKING:
    from botorch.models.transforms.outcome import OutcomeTransform
    import torch


def setup_node_transforms(
    exp_config: Dict[str, Any], data_dict: Dict[str, torch.Tensor]
):
    """Set up the node transforms."""
    node_transforms = {}
    outcome_transform_module = importlib.import_module("botorch.models.transforms")
    for node_name, transform_name in exp_config["simulation"][
        "node_transforms"
    ].items():
        transform = getattr(outcome_transform_module, transform_name)
        node_transforms[node_name] = transform(m=data_dict[node_name].shape[-1])
    return node_transforms


def transform_data_dict(
    exp_config: Dict[str, Any],
    sim_env_bounds: torch.Tensor,
    data_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, OutcomeTransform]]:
    """Transform the data dictionary."""
    node_transforms = setup_node_transforms(exp_config, data_dict)
    for node_name, transform in node_transforms.items():
        data_dict[node_name], _ = transform(data_dict[node_name])
        transform.eval()
    data_dict["inputs"] = normalize(data_dict["inputs"], sim_env_bounds)
    return data_dict, node_transforms
