from typing import Dict, Optional

import torch
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.objective import GenericMCObjective

from experiments.bayesian_optimization.utils.base_bo_experiment import BaseBOExperiment
import gpytorch
from src.pogpn_botorch.pogpn_nodewise import POGPNNodewise
from src.pogpn_botorch.pogpn_pathwise import POGPNPathwise
from src.pogpn_botorch.dag import DAG, RootNode, RegressionNode
from botorch.models.utils.inducing_point_allocators import GreedyImprovementReduction
from botorch.models.transforms.outcome import OutcomeTransform
import os
from experiments.bayesian_optimization.utils.bo_utils.data_logging import save_script


class POGPNCommon(BaseBOExperiment):
    """BO experiment for POGPN Nodewise model."""

    def setup_model(
        self,
        data_dict: Dict[str, torch.Tensor],
        previous_iter_model: Optional[POGPNNodewise | POGPNPathwise] = None,
        node_transforms: Optional[Dict[str, OutcomeTransform]] = None,
    ) -> POGPNNodewise | POGPNPathwise:
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

        if self.model_config["model_type"]["name"] == "POGPN_Nodewise":
            model_class = POGPNNodewise
        elif self.model_config["model_type"]["name"] == "POGPN_Pathwise":
            model_class = POGPNPathwise
        else:
            raise ValueError(
                f"Invalid model type: {self.model_config['model_type']['name']}"
            )

        with gpytorch.settings.num_likelihood_samples(num_mc_samples):
            model = model_class(
                dag=dag,
                data_dict=data_dict,
                root_node_indices_dict={"x": list(range(self.sim_env.dim))},
                objective_node_name="y3",
                inducing_point_ratio=inducing_point_ratio,
                mll_beta=mll_beta,
                mll_type=mll_type,
            )

            if isinstance(model, POGPNPathwise):
                model.fit(
                    data_dict=data_dict,
                    optimizer=self.model_config["mll_optimizer_kwargs"]["optimizer"],
                    lr=float(self.model_config["mll_optimizer_kwargs"]["lr"]),
                )
            else:
                model.fit(
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

    print("End of the script.")


def run_pogpn_pathwise_elbo_gir(exp_config_path: str):
    """Run the POGPN pathwise ELBO with GreedyImprovementReduction."""
    exp = POGPNCommon(
        exp_config_path=exp_config_path,
        model_config_path=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs",
            "model_configs",
            "pogpn_pathwise_1.0beta_ELBO_1.0IPRatio_GIR.yaml",
        ),
        mlflow_run_name_prefix="pogpn_pathwise_1.0beta_ELBO_1.0IPRatio_GIR",
        log_mlflow_to_gitlab=True,
        run_note="Using GreedyImprovementReduction inducing point allocator.",
    )
    save_script(exp.mlflow_manager, __file__)
    exp.run_experiment()
    print("End of the script.")


def run_pogpn_pathwise_pll_gir(exp_config_path: str):
    """Run the POGPN pathwise ELBO with GreedyImprovementReduction."""
    exp = POGPNCommon(
        exp_config_path=exp_config_path,
        model_config_path=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs",
            "model_configs",
            "pogpn_pathwise_1.0beta_PLL_1.0IPRatio_GIR.yaml",
        ),
        mlflow_run_name_prefix="pogpn_pathwise_1.0beta_PLL_1.0IPRatio_GIR",
        log_mlflow_to_gitlab=True,
        run_note="Using GreedyImprovementReduction inducing point allocator.",
    )
    save_script(exp.mlflow_manager, __file__)
    exp.run_experiment()
    print("End of the script.")
