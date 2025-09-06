from typing import Dict, Optional

import torch
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.objective import GenericMCObjective

from experiments.bayesian_optimization.utils.base_bo_experiment import BaseBOExperiment
import gpytorch
from src.pogpn_botorch.dag import DAG, RootNode, RegressionNode
from botorch.models.utils.inducing_point_allocators import GreedyImprovementReduction
from botorch.models.transforms.outcome import OutcomeTransform
import os
from experiments.bayesian_optimization.utils.bo_utils.data_logging import save_script
from src.pogpn_botorch.pogpn_pathwise import POGPNPathwise
from gpytorch.kernels import ScaleKernel, RBFKernel
from torch import Tensor
import logging
from typing import Optional, Tuple, Dict
from botorch.models.model import Model
from gpytorch.mlls import MarginalLogLikelihood
from src.pogpn_botorch.pogpn_mll import (
    VariationalELBOCustom,
    PredictiveLogLikelihoodCustom,
    VariationalELBOCustomWithNaN,
)
from botorch.models.utils.inducing_point_allocators import InducingPointAllocator
from src.pogpn_botorch.custom_approximate_gp import BoTorchVariationalGP
from src.pogpn_botorch.likelihood_prior import get_lognormal_likelihood_prior
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.constraints import GreaterThan
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL

logger = logging.getLogger("POGPN PATHWISE")


# TODO: Add the minibatching to the custom scipy and torch fit because with a small batch size,
# the intermediate layer MC samples will be better space filling for the given MC sample shape.


class POGPNPathwiseCustom(POGPNPathwise):
    """POGPNPathwise with node-wise conditional training of the nodes."""

    def _get_node_model_and_mll(
        self,
        node_name: str,
        train_X_node: Tensor,  # noqa: N803
        train_Y_node: Tensor,  # noqa: N803
        inducing_point_ratio: float,
        mll_beta: float,
        mll_type: str = "PLL",
        inducing_point_allocator: Optional[InducingPointAllocator] = None,
        learn_inducing_points: bool = True,
        node_observation_noise: Optional[float] = None,
        node_transform: Optional[OutcomeTransform] = None,
    ) -> Tuple[Model, MarginalLogLikelihood]:
        """Get the node model and marginal log likelihood for the given node.

        Args:
            node_name: Name of the node.
                Can be used in case you want to use different models or mlls for different nodes
            train_X_node: Training data for the node.
            train_Y_node: Training labels for the node.
            inducing_point_ratio: The ratio of inducing points to the number of observations.
                This is used to determine the number of inducing points for each node.
            mll_beta: The beta parameter for the ApproximateMarginalLogLikelihood.
            mll_type: The type of marginal log likelihood to use.
                Can be "ELBO" for Variational ELBO or "PLL" for Predictive Log Likelihood.
            inducing_point_allocator: The inducing point allocator for the node.
                This has been provided to be able to use different inducing point allocators for different nodes like GreedyImprovementReduction.
            learn_inducing_points: Whether to learn the inducing points.
            node_observation_noise: The noise level of the node observation.
            node_transform: The outcome transform for the node. This can be used to get likelihood prior using noise and the transform of the node.

        Returns:
            Tuple[Model, MarginalLogLikelihood]: The node model and marginal log likelihood.

        """
        if node_observation_noise is not None:
            likelihood_prior = get_lognormal_likelihood_prior(
                node_observation_noise=node_observation_noise,
                node_transform=node_transform,
            )
            if train_Y_node.shape[-1] == 1:
                likelihood = GaussianLikelihood(
                    noise_prior=likelihood_prior,
                    noise_constraint=GreaterThan(
                        MIN_INFERRED_NOISE_LEVEL,
                        transform=None,  # type: ignore
                        initial_value=likelihood_prior.mode,
                    ),
                )
            elif train_Y_node.shape[-1] > 1:
                likelihood = MultitaskGaussianLikelihood(
                    num_tasks=train_Y_node.shape[-1],
                    noise_prior=likelihood_prior,
                    noise_constraint=GreaterThan(
                        MIN_INFERRED_NOISE_LEVEL,
                        transform=None,  # type: ignore
                        initial_value=likelihood_prior.mode,
                    ),
                )
        else:
            likelihood = None

        model = BoTorchVariationalGP(
            train_X=train_X_node,
            train_Y=train_Y_node,
            inducing_points=int(inducing_point_ratio * train_X_node.shape[-2]),
            num_outputs=train_Y_node.shape[-1],
            inducing_point_allocator=inducing_point_allocator,
            learn_inducing_points=learn_inducing_points,
            likelihood=likelihood,
            covar_module=ScaleKernel(
                RBFKernel(ard_num_dims=train_X_node.shape[-1]),
                ard_num_dims=train_X_node.shape[-1],
            ),
        )
        if mll_type.upper() == "ELBO":
            if self.masks_dict is not None and node_name in self.masks_dict:
                mll = VariationalELBOCustomWithNaN(
                    likelihood=model.likelihood,
                    model=model.model,
                    num_data=self.num_observations,
                    beta=mll_beta,
                )
                # Attach row mask for this node
                mll.row_mask = self.masks_dict[node_name]
            else:
                mll = VariationalELBOCustom(
                    likelihood=model.likelihood,
                    model=model.model,
                    num_data=self.num_observations,
                    beta=mll_beta,
                )
        elif mll_type.upper() == "PLL":
            mll = PredictiveLogLikelihoodCustom(
                likelihood=model.likelihood,
                model=model.model,
                num_data=self.num_observations,
                beta=mll_beta,
            )
        return model, mll


class POGPNCustom(BaseBOExperiment):
    """BO experiment for POGPN Nodewise model."""

    def setup_model(
        self,
        data_dict: Dict[str, torch.Tensor],
        previous_iter_model: Optional[POGPNPathwiseCustom] = None,
        node_transforms: Optional[Dict[str, OutcomeTransform]] = None,
    ) -> POGPNPathwiseCustom:
        """Set up and train the GP network model."""
        num_mc_samples = self.model_config["model_type"]["params"]["num_mc_samples"]
        inducing_point_ratio = self.model_config["model_type"]["params"][
            "inducing_point_ratio"
        ]
        mll_beta = self.model_config["model_type"]["params"]["mll_beta"]
        mll_type = self.model_config["model_type"]["params"]["mll_type"]
        inducing_point_allocator_name = self.model_config["model_type"]["params"][
            "inducing_point_allocator"
        ]
        likelihood_type = self.model_config["model_type"]["params"]["likelihood"]

        if inducing_point_allocator_name == "GreedyImprovementReduction":
            if previous_iter_model is None:
                self.logger.warning(
                    "Previous iteration model is None. GreedyImprovementReduction is not used."
                )
                objective_inducing_point_allocator = None
                objective_learn_inducing_points = True
            else:
                objective_inducing_point_allocator = GreedyImprovementReduction(
                    previous_iter_model.model.node_models_dict[
                        self.sim_env.objective_node_name
                    ],
                    maximize=True,
                )
                objective_learn_inducing_points = False
        else:
            objective_inducing_point_allocator = None
            objective_learn_inducing_points = True
            self.logger.info(
                "Using default GreedyVarianceReduction inducing point allocator."
            )

        self.mlflow_manager.log_params(
            {"objective_learn_inducing_points": objective_learn_inducing_points}
        )

        if likelihood_type == "noise_prior":
            self.logger.info("Using noise prior for likelihood.")
            node_observation_noise = self.sim_env.observation_noise_std
        else:
            node_observation_noise = None

        x_node = RootNode(
            name="x",
            parents=[],
            node_output_dim=self.sim_env.dim,
        )
        y1_node = RegressionNode(
            name="y1",
            parents=[x_node],
            node_output_dim=1,
            node_observation_noise=node_observation_noise,
            node_transform=node_transforms["y1"] if node_transforms else None,
        )
        y2_node = RegressionNode(
            name="y2",
            parents=[x_node],
            node_output_dim=1,
            node_observation_noise=node_observation_noise,
            node_transform=node_transforms["y2"] if node_transforms else None,
        )
        y3_node = RegressionNode(
            name="y3",
            parents=[y1_node, y2_node],
            node_output_dim=1,
            node_observation_noise=node_observation_noise,
            node_transform=node_transforms["y3"] if node_transforms else None,
            inducing_point_allocator=objective_inducing_point_allocator,
            learn_inducing_points=objective_learn_inducing_points,
        )
        dag = DAG(dag_nodes=[x_node, y1_node, y2_node, y3_node])

        with gpytorch.settings.num_likelihood_samples(num_mc_samples):
            model = POGPNPathwiseCustom(
                dag=dag,
                data_dict=data_dict,
                root_node_indices_dict={"x": list(range(self.sim_env.dim))},
                objective_node_name="y3",
                inducing_point_ratio=inducing_point_ratio,
                mll_beta=mll_beta,
                mll_type=mll_type,
            )

            model.fit(
                data_dict=data_dict,
                optimizer=self.model_config["mll_optimizer_kwargs"]["optimizer"],
                lr=float(self.model_config["mll_optimizer_kwargs"]["lr"]),
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


def run_elbo_experiment(exp_config_path: str):
    """Run the ELBO experiment with GreedyImprovementReduction inducing point allocator."""
    name = "pogpn_pathwise_1.0beta_ELBO_1.0IPRatio_GIR_cov_no_prior"
    exp = POGPNCustom(
        exp_config_path=exp_config_path,
        model_config_path=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs",
            "model_configs",
            f"{name}.yaml",
        ),
        mlflow_run_name_prefix=name,
        log_mlflow_to_gitlab=True,
        run_note="Using GreedyImprovementReduction inducing point allocator and no covariance prior.",
    )
    save_script(exp.mlflow_manager, __file__)
    exp.run_experiment()
    print("ELBO experiment completed.")


def run_pll_experiment(exp_config_path: str):
    """Run the PLL experiment with GreedyImprovementReduction inducing point allocator."""
    name = "pogpn_pathwise_1.0beta_PLL_1.0IPRatio_GIR_cov_no_prior"
    exp = POGPNCustom(
        exp_config_path=exp_config_path,
        model_config_path=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs",
            "model_configs",
            f"{name}.yaml",
        ),
        mlflow_run_name_prefix=name,
        log_mlflow_to_gitlab=True,
        run_note="Using GreedyImprovementReduction inducing point allocator and no covariance prior.",
    )
    save_script(exp.mlflow_manager, __file__)
    exp.run_experiment()
    print("PLL experiment completed.")


if __name__ == "__main__":
    exp_config_path = os.path.join(
        "experiments",
        "bayesian_optimization",
        "rosenbrock",
        "configs",
        "exp_configs",
        "rosenbrock_200D.yaml",
    )
    run_elbo_experiment(exp_config_path)
    run_pll_experiment(exp_config_path)
    print("All experiments completed.")
