from typing import Dict, Optional

import torch
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.objective import GenericMCObjective

from src.other_models.gp_network import GaussianProcessNetwork, GPNetworkDAG

from experiments.bayesian_optimization.utils.base_bo_experiment import BaseBOExperiment
from experiments.bayesian_optimization.utils.bo_utils.data_logging import save_script
from botorch.models.transforms.outcome import OutcomeTransform
import os
from src.pogpn_botorch.likelihood_prior import DEFAULT_TARGET_DENSITY_AT_MEAN


class GPNetworkCommon(BaseBOExperiment):
    """MLflow-enabled experiment for GP network Bayesian optimization (EIFN) - Raul's implementation.

    Not implementing the extra baseline candidates using posterior mean so that the
    comparison between methods is fair.
    """

    def setup_model(
        self,
        data_dict: Dict[str, torch.Tensor],
        previous_iter_model: Optional[GaussianProcessNetwork] = None,
        node_transforms: Optional[Dict[str, OutcomeTransform]] = None,
    ) -> GaussianProcessNetwork:
        """Set up and train the GP network model."""
        input_dim = self.sim_env.dim

        parents_nodes = [[], [], [], [0, 1, 2]]
        active_input_indices = [
            [0],
            list(range(input_dim - 1)),
            [input_dim - 1],
            [],
        ]
        dag = GPNetworkDAG(parents_nodes)

        train_x = data_dict["inputs"]
        train_y = self.sim_env.get_output_tensor(
            data_dict, node_order=["y1", "y2", "y3", "y4"]
        )

        likelihood_type = self.model_config["model_type"]["params"]["likelihood"]
        if likelihood_type == "noise_prior":
            self.logger.info("Using noise prior for likelihood.")
            node_observation_noise = self.sim_env.observation_noise_std
        else:
            node_observation_noise = None

        model = GaussianProcessNetwork(
            train_X=train_x,
            train_Y=train_y,
            dag=dag,
            active_input_indices=active_input_indices,
            objective_output_index=-1,
            optimizer_kwargs=self.mll_optimizer_kwargs,
            node_observation_noise=node_observation_noise,
        )

        return model

    def setup_acquisition_function(self, model, best_value):
        """Set up the marginalized Expected Improvement acquisition function."""

        def network_to_objective_transform(samples, X=None):
            return samples[..., -1]

        return self.acquisition_function(
            model=model,
            best_f=best_value,
            sampler=SobolQMCNormalSampler(
                sample_shape=torch.Size(
                    [self.acqf_optim_params.qmc_sampler_sample_shape]
                ),
            ),
            objective=GenericMCObjective(network_to_objective_transform),
        )


def run_gp_network_noise_prior(exp_config_path: str):
    """Run the GP network noise prior experiment."""
    exp = GPNetworkCommon(
        exp_config_path=exp_config_path,
        model_config_path=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs",
            "model_configs",
            "gp_network_noise_prior.yaml",
        ),
        mlflow_run_name_prefix="gp_network_noise_prior",
        log_mlflow_to_gitlab=True,
        run_note=f"Using noise prior for likelihood. Target density at mean: {DEFAULT_TARGET_DENSITY_AT_MEAN}.",
    )
    save_script(exp.mlflow_manager, __file__)
    exp.run_experiment()
    print("End of the script.")


def run_gp_network(exp_config_path: str):
    """Run the GP network without experiment."""
    exp = GPNetworkCommon(
        exp_config_path=exp_config_path,
        model_config_path=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs",
            "model_configs",
            "gp_network.yaml",
        ),
        mlflow_run_name_prefix="gp_network",
        log_mlflow_to_gitlab=True,
        run_note="Using default likelihood.",
    )
    save_script(exp.mlflow_manager, __file__)
    exp.run_experiment()
    print("End of the script.")
