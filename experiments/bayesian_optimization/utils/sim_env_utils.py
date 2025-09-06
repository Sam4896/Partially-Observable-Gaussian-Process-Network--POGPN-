import importlib
from typing import Dict, Any, Optional

import torch
from torch import Tensor

from botorch.utils.transforms import unnormalize
from src.synthetic_test_function.base.dag_experiment_base import (
    DAGSyntheticTestFunction,
)


def get_initial_training_data(
    input_dim: int,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: Optional[int] = None,
) -> Tensor:
    """Get initial training data for the test function using Sobol sampling."""
    # Use context manager to avoid affecting global state
    sobol = torch.quasirandom.SobolEngine(dimension=input_dim, scramble=True, seed=seed)
    train_x = sobol.draw(num_samples).to(device=device, dtype=dtype)

    return train_x


def get_best_observed(sim_env, output_dict) -> Tensor:
    """Get the best observed value from the output dictionary."""
    objective = output_dict[sim_env.objective_node_name]
    if objective.shape[-1] > 1:
        return objective[..., -1].max()
    else:
        return objective.max()


def evaluate_output_from_sim_env(
    sim_env: DAGSyntheticTestFunction, train_x: Tensor
) -> Dict[str, Tensor]:
    """Evaluate the output of the simulation environment."""
    sim_env.to(train_x.device, train_x.dtype)
    output_dict = sim_env(unnormalize(train_x, sim_env.bounds))
    return output_dict


def load_sim_env_from_config(
    sim_env_config: Dict[str, Any],
) -> DAGSyntheticTestFunction:
    """Load simulation environment from configuration.

    Args:
        sim_env_config: Configuration dictionary containing simulation parameters

    Returns:
        Initialized simulation environment

    Example config:
        simulation:
            name: Ackley
            dim: 6
            observation_noise_std: 0.01
            process_stochasticity_std: 0.01

    """
    sim_name = sim_env_config["name"]

    # Dynamically import the simulation class
    module = importlib.import_module("src.synthetic_test_function")
    sim_class = getattr(module, sim_name)

    # Create and return the simulation environment
    return sim_class(
        dim=sim_env_config["dim"],
        observation_noise_std=sim_env_config["observation_noise_std"],
        process_stochasticity_std=sim_env_config["process_stochasticity_std"],
    )
