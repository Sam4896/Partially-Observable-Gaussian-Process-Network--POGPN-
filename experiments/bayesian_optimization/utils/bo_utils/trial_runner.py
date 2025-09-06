from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import time
import numpy as np
import torch


from experiments.bayesian_optimization.utils import (
    get_initial_training_data,
    evaluate_output_from_sim_env,
    get_best_observed,
)
from experiments.bayesian_optimization.utils.bo_utils.transforms import (
    transform_data_dict,
)

if TYPE_CHECKING:
    import logging
    from botorch.acquisition import AcquisitionFunction
    from src.synthetic_test_function.base.dag_experiment_base import (
        DAGSyntheticTestFunction,
    )
    from botorch.models.transforms.outcome import OutcomeTransform


class BestTrialTracker:
    """Tracks the best trial value, model, and data across the experiment."""

    def __init__(self) -> None:
        """Initialize the best trial tracker."""
        self.best_value: float | None = None
        self.model: Any | None = None

    def update(self, value: float, model: Any) -> None:
        """Update the best trial tracker."""
        if (self.best_value is None) or (value > self.best_value):
            self.best_value = value
            self.model = model


class TrialRunner:
    """Runs a single trial over BO iterations by delegating to provided callbacks."""

    def __init__(
        self,
        sim_env: DAGSyntheticTestFunction,
        exp_config: Dict[str, Any],
        acqf_params: Dict[str, Any],
        logger: logging.Logger,
        optimize_fn: Callable[[AcquisitionFunction], Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.sim_env = sim_env
        self.exp_config = exp_config
        self.acqf_params = acqf_params
        self.logger = logger
        self.optimize_fn = optimize_fn
        self.device = device
        self.dtype = dtype

    def run_trial(
        self,
        trial: int,
        setup_model_fn: Callable[
            [Dict[str, torch.Tensor], Optional[Any], Dict[str, OutcomeTransform]], Any
        ],
        setup_acqf_fn: Callable[[Any, torch.Tensor], AcquisitionFunction],
        compute_trial_metrics_fn: Optional[
            Callable[
                [Dict[str, torch.Tensor], List[float], Any, AcquisitionFunction, int],
                Dict[str, float],
            ]
        ] = None,
    ) -> Tuple[
        List[float],
        Dict[str, float],
        Dict[str, torch.Tensor],
        torch.Tensor,
        Any,
    ]:
        """Run a single trial of the Bayesian optimization experiment."""
        # Initial data
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(trial)
        train_x = get_initial_training_data(
            input_dim=self.sim_env.dim,
            seed=trial,
            num_samples=self.acqf_params["num_initial_data_samples"],
            device=self.device,
            dtype=self.dtype,
        )
        untransformed_data_dict = evaluate_output_from_sim_env(self.sim_env, train_x)
        self.logger.info(
            f"Objective output of initial data: {untransformed_data_dict[self.sim_env.objective_node_name]}"
        )
        torch.random.set_rng_state(rng_state)

        untransformed_best_value = get_best_observed(
            self.sim_env, untransformed_data_dict
        )
        history: List[float] = [np.round(untransformed_best_value.item(), 4)]

        model: Any = None

        for iteration in range(1, self.acqf_params["bo_budget"] + 1):
            self.logger.info(
                f"\nTrial {trial:>2} of {self.acqf_params['num_trials']}, Iteration {iteration:>3} of {self.acqf_params['bo_budget']}"
            )
            t0 = time.monotonic()

            transformed_data_dict = {
                k: v.clone() for k, v in untransformed_data_dict.items()
            }
            transformed_data_dict, node_transforms = transform_data_dict(
                self.exp_config, self.sim_env.bounds, transformed_data_dict
            )

            model = setup_model_fn(transformed_data_dict, model, node_transforms)
            acqf = setup_acqf_fn(
                model, get_best_observed(self.sim_env, transformed_data_dict)
            )

            new_input_point, acqf_value = self.optimize_fn(acqf)
            untransformed_new_data_dict = evaluate_output_from_sim_env(
                self.sim_env, new_input_point
            )
            for k in untransformed_data_dict.keys():
                untransformed_data_dict[k] = torch.cat(
                    [untransformed_data_dict[k], untransformed_new_data_dict[k]], dim=-2
                )
            untransformed_best_value = get_best_observed(
                self.sim_env, untransformed_data_dict
            )
            history.append(np.round(untransformed_best_value.item(), 4))

            t1 = time.monotonic()
            self.logger.info(
                f"Trial {trial:>2}, Iteration {iteration:>3}: "
                f"New point = {new_input_point.tolist()}, Acq value = {acqf_value.item():.4f}, "
                f"Best value = {untransformed_best_value.item():.4f}, Time = {t1 - t0:.2f}s"
            )
        self.logger.info(
            f"Completed trial {trial:>2} of {self.acqf_params['num_trials']}"
        )

        if compute_trial_metrics_fn is not None:
            trial_metrics = compute_trial_metrics_fn(
                untransformed_data_dict,
                history,
                model,
                acqf,
                trial,
            )

        return (
            history,
            trial_metrics,
            untransformed_data_dict,
            untransformed_best_value,
            model,
        )
