import os
import torch
from botorch.acquisition import IdentityMCObjective
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from typing import Dict, Optional
from experiments.bayesian_optimization.utils.base_bo_experiment import BaseBOExperiment
from experiments.bayesian_optimization.utils.bo_utils.data_logging import save_script
from botorch.models.transforms.outcome import OutcomeTransform


class STGPBOExperiment(BaseBOExperiment):
    """MLflow-enabled Bayesian optimization experiment for the STGP model."""

    def setup_model(
        self,
        data_dict: Dict[str, torch.Tensor],
        previous_iter_model: Optional[SingleTaskGP] = None,
        node_transforms: Optional[Dict[str, OutcomeTransform]] = None,
    ) -> SingleTaskGP:
        """Set up and train the STGP model."""
        train_x = data_dict["inputs"]
        train_y = data_dict[self.sim_env.objective_node_name]

        params = self.model_config["model_type"].get("params", {})

        self.mlflow_manager.log_params(params)

        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        model.to(self.device)

        fit_gpytorch_mll(mll, optimizer_kwargs=self.mll_optimizer_kwargs)

        return model

    def setup_acquisition_function(self, model, best_value):
        """Set up the acquisition function."""
        return self.acquisition_function(
            model=model,
            sampler=SobolQMCNormalSampler(
                sample_shape=torch.Size(
                    [self.acqf_optim_params.qmc_sampler_sample_shape]
                ),
            ),
            objective=IdentityMCObjective(),
            best_f=best_value,
        )


def run_stgp(exp_config_path: str):
    """Run the STGP without noise prior."""
    exp = STGPBOExperiment(
        exp_config_path=exp_config_path,
        model_config_path=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs",
            "model_configs",
            "stgp.yaml",
        ),
        mlflow_run_name_prefix="stgp",
        log_mlflow_to_gitlab=True,
        run_note="Using default likelihood.",
    )
    save_script(exp.mlflow_manager, __file__)
    exp.run_experiment()
    print("End of the script.")
