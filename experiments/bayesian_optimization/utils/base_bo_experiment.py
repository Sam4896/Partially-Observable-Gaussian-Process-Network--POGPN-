import importlib
import logging
import os
from abc import abstractmethod
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dataclasses import dataclass, asdict

from botorch.models.model import Model
import pandas as pd
import torch
from botorch.acquisition import AcquisitionFunction

from experiments.bayesian_optimization.utils import (
    load_sim_env_from_config,
)
from experiments.utils.logging_utils import (
    setup_logging,
    format_dict_as_yaml,
)
from experiments.utils.device_utils import setup_device
from experiments.utils.yaml_utils import load_config_from_yaml
from experiments.utils.mlflow_manager import setup_mlflow_tracking
from botorch.optim import optimize_acqf

from experiments.bayesian_optimization.utils.bo_utils import (
    create_pogpn_run_name_prefix,
    build_experiment_name,
    plot_results,
    save_data_dict,
    save_log_file_to_mlflow,
    save_script,
    setup_gitlab_mlflow_env,
    setup_mlflow_logging_dir,
    _is_pogpn,
    create_trajectory,
    aggregate_metrics,
    _is_gp_network,
    create_gp_network_run_name_prefix,
    _is_stgp,
    create_stgp_run_name_prefix,
)
from experiments.bayesian_optimization.utils.bo_utils.tags import (
    build_run_description,
)
from experiments.bayesian_optimization.utils.bo_utils import (
    BestTrialTracker,
    MetricsAggregator,
    TrialRunner,
)
from botorch.models.transforms.outcome import OutcomeTransform


# Import BO metrics (optional)
try:
    from experiments.bayesian_optimization.utils.bo_metrics import compute_bo_metrics

    BO_METRICS_AVAILABLE = True
except ImportError:
    BO_METRICS_AVAILABLE = False


class BaseBOExperiment:
    """Enhanced base class for Bayesian optimization experiments with MLflow tracking."""

    def __init__(
        self,
        exp_config_path: str,
        model_config_path: str,
        mlflow_run_name_prefix: str,
        run_note: Optional[str] = None,
        log_mlflow_to_gitlab: bool = False,
        mlflow_tracking_uri: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize the enhanced Bayesian optimization experiment.

        Args:
            exp_config_path: Path to the experiment configuration file
            model_config_path: Path to the model configuration file
            mlflow_run_name_prefix: Prefix for the MLflow run name
                ex: POGPN_Nodewise_Torch_GIR
            mlflow_tracking_uri: Optional MLflow tracking URI
            log_mlflow_to_gitlab: Whether to log to GitLab MLflow
            device: Optional device to use for the experiment

        """
        # Store config paths
        self.exp_config_path = exp_config_path
        self.model_config_path = model_config_path
        self.mlflow_run_name_prefix = mlflow_run_name_prefix

        # Load configurations
        self.exp_config = load_config_from_yaml(exp_config_path)
        self.model_config = load_config_from_yaml(model_config_path)

        # Setup model and acquisition function
        self._setup_model_and_acquisition_config()

        self._create_folder_and_other_names()

        self._create_mlflow_run_tags_and_description()

        self.log_mlflow_to_gitlab = log_mlflow_to_gitlab

        # Set up GitLab MLflow if requested
        if log_mlflow_to_gitlab:
            if mlflow_tracking_uri is not None:
                raise ValueError(
                    "mlflow_tracking_uri is not supported when log_mlflow_to_gitlab is True. Read GitLab MLflow documentation for more information."
                )
            self._setup_gitlab_mlflow()

        self._setup_mlflow_tracking(mlflow_tracking_uri)

        if run_note is not None:
            self.mlflow_manager.set_run_note(run_note)
        self.mlflow_manager.set_run_description(self.run_description)

        self.mlflow_manager.log_configs(
            local_config_paths=[self.exp_config_path, self.model_config_path]
        )
        self.mlflow_manager.log_params(self.acqf_params)
        self.mlflow_manager.log_params(asdict(self.acqf_optim_params))
        if self.model_config["model_type"].get("params", None):
            self.mlflow_manager.log_params(self.model_config["model_type"]["params"])
        if self.mll_optimizer_kwargs is not None:
            self.mlflow_manager.log_params(self.mll_optimizer_kwargs)

        if not self.log_mlflow_to_gitlab:
            self._setup_mlflow_logging()
        else:
            # No MLflow: Use local results folder
            os.makedirs(self.run_folder, exist_ok=True)
            setup_logging(self.run_folder)

        self.logger = logging.getLogger("BaseBOExperiment")
        if self.log_mlflow_to_gitlab:
            self.logger.info(
                "Using local logging with remote upload (GitLab MLflow detected). Will upload logs to GitLab at the end of the experiment."
            )
        else:
            self.logger.info("Using local logging (local MLflow detected)")

        # Log experiment setup
        self._log_experiment_setup()

        # Setup device
        if device is None:
            self.device = setup_device()
        else:
            self.device = device
        self.dtype = torch.double

        self.sim_env = load_sim_env_from_config(self.exp_config["simulation"]).to(
            self.device, self.dtype
        )

        # Initialize BO metrics (optional)
        self.enable_bo_metrics = BO_METRICS_AVAILABLE

        try:
            if self.sim_env.optimal_value is not None:
                self.mlflow_manager.log_metrics(
                    {"sim_env_optimal_value": self.sim_env.optimal_value}
                )
        except NotImplementedError:
            self.logger.warning(
                "sim_env does not have an optimal_value attribute. Skipping logging optimal value."
            )

        self.logger.info(
            "BO metrics enabled"
            if self.enable_bo_metrics
            else "BO metrics not available (import failed)"
        )

        self.run_once_from_base_file()

        self._trial_runner = TrialRunner(
            sim_env=self.sim_env,
            exp_config=self.exp_config,
            acqf_params=self.acqf_params,
            logger=self.logger,
            optimize_fn=self.optimize_acquisition_function,
            device=self.device,
            dtype=self.dtype,
        )
        self._metrics_manager = MetricsAggregator(
            self.mlflow_manager, self.logger, self.sim_env
        )

    def run_once_from_base_file(self, **kwargs):
        """Helper function to run a particular code once from the base file."""  # noqa: D401
        pass

    def _create_mlflow_run_tags_and_description(self):
        self.run_description = build_run_description(
            model_config=self.model_config,
            exp_config=self.exp_config,
            acqf_params=self.acqf_params,
            acqf_optim_params=self.acqf_optim_params,
            sim_env_name=self.sim_env_name,
            acqf_suffix=self.acqf_suffix,
        )

    def _create_folder_and_other_names(self):
        """Create folder names for the experiment."""
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get the simulation environment folder path
        self.results_folder = os.path.join(
            "experiments",
            "bayesian_optimization",
            f"{self.exp_config['simulation']['name']}".lower(),
            f"{self.exp_config['simulation']['dim']}D",
            "results",
        )

        # Create enhanced MLflow run name prefix for POGPN experiments
        if _is_pogpn(self.model_config):
            self.mlflow_run_name_prefix = create_pogpn_run_name_prefix(
                self.model_config
            )
        elif _is_gp_network(self.model_config):
            self.mlflow_run_name_prefix = create_gp_network_run_name_prefix(
                self.model_config
            )
        elif _is_stgp(self.model_config):
            self.mlflow_run_name_prefix = create_stgp_run_name_prefix(self.model_config)

        # Build experiment name (same as MLflow experiment name)
        self.experiment_name = build_experiment_name(
            exp_config=self.exp_config,
            acqf_optim_q=self.acqf_optim_params.q,
            acqf_suffix=self.acqf_suffix,
            acqf_params=self.acqf_params,
        )

        # Build run name
        self.run_name = f"{self.mlflow_run_name_prefix}_{self.stamp}"

        # Create run folder with timestamp
        self.run_folder = os.path.join(
            self.results_folder,
            self.experiment_name,
            self.mlflow_run_name_prefix,
            self.stamp,
        )

    def _setup_gitlab_mlflow(self):
        """Set up GitLab MLflow environment variables from config file."""
        import yaml

        # Find project root by looking for gitlab_mlflow.yaml
        # Start from current directory and walk up to find the project root
        current_dir = os.path.abspath(os.curdir)
        project_root = None

        # Walk up the directory tree to find gitlab_mlflow.yaml
        while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
            config_path = os.path.join(current_dir, "gitlab_mlflow.yaml")
            if os.path.exists(config_path):
                project_root = current_dir
                break
            current_dir = os.path.dirname(current_dir)

        if project_root is None:
            raise FileNotFoundError(
                "GitLab MLflow config file 'gitlab_mlflow.yaml' not found in any parent directory. "
                "Create it with your GitLab credentials in the project root."
            )

        config_path = os.path.join(project_root, "gitlab_mlflow.yaml")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Set environment variables
        mlflow_tracking_uri = setup_gitlab_mlflow_env(config)

        print("✅ GitLab MLflow environment variables set:")
        print(f"   URI: {mlflow_tracking_uri}")
        print(
            f"   Token: {'*' * (len(config['access_token']) - 4) + config['access_token'][-4:] if len(config['access_token']) > 4 else '****'}"
        )

    def _setup_mlflow_tracking(self, mlflow_tracking_uri: Optional[str] = None):
        """Set up MLflow tracking for the experiment."""
        # Use the *model name* as run_name so that repeated experiments are grouped.
        self.mlflow_manager = setup_mlflow_tracking(
            experiment_name=self.experiment_name,
            folder_name=self.run_folder,
            run_name=self.run_name,
            is_tracking_gitlab=self.log_mlflow_to_gitlab,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )

    def _setup_mlflow_logging(self):
        """Set up logging with fallback for remote MLflow.

        This method handles both local and remote MLflow tracking:

        - **Local MLflow**: Logs are written directly to the MLflow artifacts directory
          for real-time access in the MLflow UI.

        - **Remote MLflow**: Logs are written to a local file and uploaded to MLflow
          at the end of the experiment to avoid network latency and file system issues.


        """
        logs_dir = setup_mlflow_logging_dir(
            self.mlflow_manager, run_logs_dirname="logs"
        )
        setup_logging(logs_dir, "logs.log")

    def _setup_model_and_acquisition_config(self):
        """Set up model name and acquisition function."""
        # Get acquisition function parameters
        self.acqf_params = self.exp_config["acqf_params"]

        # -----------------------------
        # Dataclass for acqf_optim_params
        # -----------------------------

        @dataclass
        class AcqfOptimParams:
            q: int
            num_restarts: int
            raw_samples: int
            acqf_batch_limit: int
            qmc_sampler_sample_shape: int

        self.acqf_optim_params = AcqfOptimParams(**self.exp_config["acqf_optim_params"])

        # Import and store acquisition function class
        acqf_module = importlib.import_module("botorch.acquisition")
        self.acquisition_function = getattr(
            acqf_module, self.acqf_params["acquisition_function_name"]
        )

        self.acqf_suffix = self._add_acqf_suffix_to_model_name()

        # Setup model name and state path
        self.model_name = self.model_config["model_type"]["name"]
        self.sim_env_name = f"{self.exp_config['simulation']['name']}_{self.exp_config['simulation']['dim']}D"

        self.mll_optimizer_kwargs = self.model_config.get("mll_optimizer_kwargs", None)

    def _add_acqf_suffix_to_model_name(self) -> str:
        """Add suffix to model name based on acquisition function."""
        if self.acquisition_function.__name__ == "qExpectedImprovement":
            suffix = "qEI"
        elif self.acquisition_function.__name__ == "qNoisyExpectedImprovement":
            suffix = "qNEI"
        elif self.acquisition_function.__name__ == "qLogExpectedImprovement":
            suffix = "qLogEI"
        elif self.acquisition_function.__name__ == "qNoisyLogExpectedImprovement":
            suffix = "qNLogEI"
        elif self.acquisition_function.__name__ == "qProbabilityOfImprovement":
            suffix = "qPI"
        elif self.acquisition_function.__name__ == "qNoisyProbabilityOfImprovement":
            suffix = "qNPI"
        else:
            suffix = self.acquisition_function.__name__

        return suffix

    def _log_experiment_setup(self):
        """Log the experiment setup details."""
        self.logger.info("Starting Bayesian optimization experiment")
        self.logger.info("Simulation config:")
        self.logger.info(format_dict_as_yaml(self.exp_config["simulation"]))
        self.logger.info("Model config:")
        self.logger.info(format_dict_as_yaml(self.model_config))
        self.logger.info("Bayesian optimization config:")
        self.logger.info(format_dict_as_yaml(self.acqf_params))
        if self.mll_optimizer_kwargs is not None:
            self.logger.info("MLL optimizer kwargs:")
            self.logger.info(format_dict_as_yaml(self.mll_optimizer_kwargs))

        self.logger.info(f"Experiment id: {self.mlflow_manager.experiment_id}")
        self.logger.info(
            f"Experiment artifact location: {self.mlflow_manager.get_artifact_uri()}"
        )
        self.logger.info(f"Experiment run id: {self.mlflow_manager.run_id}")

    @abstractmethod
    def setup_model(
        self,
        data_dict: Dict[str, torch.Tensor],
        previous_iter_model: Optional[Model] = None,
        node_transforms: Optional[Dict[str, OutcomeTransform]] = None,
    ) -> Any:
        """Set up and train the model. To be implemented by subclasses.

        Args:
            data_dict: Dictionary of training data. The keys are the node names and the values are the training data.
            previous_iter_model: Model from the previous iteration.
            node_transforms: Dictionary of node transforms. The keys are the output node names and the values are the node transforms.

        Returns:
            Trained model

        """
        raise NotImplementedError

    @abstractmethod
    def setup_acquisition_function(
        self, model: Any, best_value: torch.Tensor
    ) -> AcquisitionFunction:
        """Set up the acquisition function. To be implemented by subclasses.

        Args:
            model: Trained model
            best_value: Best observed value so far

        Returns:
            Acquisition function

        """
        raise NotImplementedError

    @contextmanager
    def acqf_optimization_context(self):
        """Context manager for acquisition function optimization.

        Can be overridden by subclasses to add model-specific context.

        Yields:
            None

        """
        yield

    def optimize_acquisition_function(
        self, acqf: AcquisitionFunction
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimize the acquisition function.

        Args:
            acqf: The acquisition function to optimize

        Returns:
            Tuple of (new_input_point, acqf_value)

        """
        bounds = torch.stack(
            [
                torch.zeros(self.sim_env.dim, dtype=self.dtype, device=self.device),
                torch.ones(self.sim_env.dim, dtype=self.dtype, device=self.device),
            ]
        )
        with self.acqf_optimization_context():
            return optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=self.acqf_optim_params.q,
                num_restarts=self.acqf_optim_params.num_restarts,
                raw_samples=self.acqf_optim_params.raw_samples,
                options={"batch_limit": self.acqf_optim_params.acqf_batch_limit},
            )

    def _log_model_summary(self, model, iteration: int, trial: int):
        """Log model summary to MLflow.

        Args:
            model: The model to log
            iteration: The iteration number
            trial: The trial number

        """
        model_name = f"{self.mlflow_run_name_prefix}_trial{trial}_iter{iteration}"
        self.mlflow_manager.log_model_summary(model, model_name)

    def _calculate_bo_metrics_for_trial(
        self,
        data_dict: Dict[str, torch.Tensor],
        best_observed_history: List[float],
        model: Any,
        acqf: AcquisitionFunction,
        trial_number: int,
    ) -> Dict[str, float]:
        """Calculate BO metrics for a single trial without logging them individually.

        Returns:
            Dictionary containing the metrics for the trial

        """
        try:
            trial_metrics = compute_bo_metrics(
                sim_env=self.sim_env,
                device=self.device,
                dtype=self.dtype,
                data_dict=data_dict,
                best_observed_history=best_observed_history,
                model=model,
                acqf=acqf,
            )

            self.logger.info(f"Trial {trial_number} metrics: {trial_metrics}")
            return trial_metrics
        except Exception as e:
            warning_msg = f"Failed to compute metrics for trial {trial_number}: {e}"
            self.logger.warning(warning_msg)

            self.mlflow_manager.mark_warning(warning_msg)

            return {}

    def _log_experiment_results(
        self,
        best_observed_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        all_trials_metrics: List[Dict[str, float]],
        aggregated_metrics: Dict[str, float],
    ) -> None:
        """Log all experiment results to MLflow.

        Args:
            best_observed_df: DataFrame containing the best observed values
            stats_df: DataFrame containing the statistics
            all_trials_metrics: List of dictionaries containing the metrics for each trial
            aggregated_metrics: Dictionary containing the aggregated metrics

        """
        # Log trajectory data
        self.mlflow_manager.log_dataframe(best_observed_df, "best_observed_values")
        self.mlflow_manager.log_dataframe(stats_df, "optimization_statistics")

        # Log individual trial metrics if available
        if all_trials_metrics:
            all_trials_df = pd.DataFrame(all_trials_metrics)
            all_trials_df.index.name = "Trial"
            self.mlflow_manager.log_dataframe(all_trials_df, "all_trials_metrics")

        # Log aggregated metrics
        if aggregated_metrics:
            self.mlflow_manager.log_metrics(aggregated_metrics)

        self.logger.info(
            f"Logged results: {len(all_trials_metrics)} trials, "
            f"{len(aggregated_metrics)} aggregated metrics"
        )

    def _log_aggregated_bo_metrics(
        self,
        all_trials_metrics: List[Dict[str, float]],
        best_observed_all_trials: List[List[float]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate and log aggregated BO metrics and trajectory data.

        Returns:
            Tuple of (best_observed_df, stats_df) containing the results

        """
        # Create trajectory DataFrames
        best_observed_df, stats_df = create_trajectory(best_observed_all_trials)

        # Calculate aggregated statistics
        aggregated_metrics = aggregate_metrics(all_trials_metrics)

        # Log everything to MLflow
        self._log_experiment_results(
            best_observed_df, stats_df, all_trials_metrics, aggregated_metrics
        )

        return best_observed_df, stats_df

    def _create_fallback_bo_optimization_results(
        self, best_observed_all_trials: List[List[float]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create trajectory data when BO metrics are disabled.

        Returns:
            Tuple of (best_observed_df, stats_df) containing the results

        """
        best_observed_df, stats_df = create_trajectory(best_observed_all_trials)

        self.mlflow_manager.log_dataframe(best_observed_df, "best_observed_values")
        self.mlflow_manager.log_dataframe(stats_df, "optimization_statistics")

        return best_observed_df, stats_df

    def _initialize_experiment(self) -> tuple[list[Any], list[Any]]:
        """Initialize trackers for a new experiment run."""
        save_script(self.mlflow_manager, __file__)
        save_script(
            self.mlflow_manager,
            os.path.join(os.path.dirname(__file__), "bo_metrics.py"),
        )
        self.best_tracker = BestTrialTracker()

        return [], []

    def _finalize_and_log_results(
        self,
        all_trials_metrics,
        untransformed_best_observed_all_trials,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Finalize the experiment by logging metrics, plots, and artifacts."""
        if self.enable_bo_metrics and all_trials_metrics:
            best_observed_df, stats_df = self._log_aggregated_bo_metrics(
                all_trials_metrics, untransformed_best_observed_all_trials
            )
        else:
            best_observed_df, stats_df = self._create_fallback_bo_optimization_results(
                untransformed_best_observed_all_trials
            )

        if (
            self.enable_bo_metrics
            and len(all_trials_metrics) < self.acqf_params["num_trials"]
        ):
            msg = f"BO metrics computed for {len(all_trials_metrics)} of {self.acqf_params['num_trials']} trials"
            self.mlflow_manager.mark_partial_failure(msg)

        try:
            optimal_value = self.sim_env.optimal_value
        except NotImplementedError:
            optimal_value = None

        fig = plot_results(
            stats_df=stats_df,
            optimal_value=optimal_value,
            model_name=self.run_name,
            run_folder=self.run_folder,
        )
        self.mlflow_manager.log_plot(fig, "optimization_performance")

        if self.log_mlflow_to_gitlab:
            save_log_file_to_mlflow(self.logger, self.mlflow_manager)

        #########################################################
        # saving data locally as well as fallback
        best_observed_df.to_csv(
            os.path.join(self.run_folder, "best_observed_values.csv")
        )
        stats_df.to_csv(os.path.join(self.run_folder, "optimization_statistics.csv"))
        #########################################################

        return best_observed_df, stats_df

    def run_experiment(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the Bayesian optimization experiment.

        Returns:
            Tuple of (best_observed_df, stats_df) containing the results

        """
        try:
            (
                untransformed_best_observed_all_trials,
                all_trials_metrics,
            ) = self._initialize_experiment()

            for trial in range(1, self.acqf_params["num_trials"] + 1):
                (
                    history,
                    trial_metrics,
                    untransformed_data,
                    best_value,
                    model,
                ) = self._trial_runner.run_trial(
                    trial,
                    setup_model_fn=lambda data_dict,
                    previous_iter_model,
                    node_transforms: self.setup_model(
                        data_dict,
                        previous_iter_model=previous_iter_model,
                        node_transforms=node_transforms,
                    ),
                    setup_acqf_fn=lambda model,
                    best_value: self.setup_acquisition_function(model, best_value),
                    compute_trial_metrics_fn=(
                        self._calculate_bo_metrics_for_trial
                        if self.enable_bo_metrics
                        else None
                    ),
                )
                untransformed_best_observed_all_trials.append(history)
                self.best_tracker.update(best_value.item(), model)
                save_data_dict(
                    self.mlflow_manager,
                    untransformed_data,
                    name=f"trial{trial}",
                    run_folder=self.run_folder,
                )
                if trial_metrics:
                    all_trials_metrics.append(trial_metrics)

            best_observed_df, stats_df = self._finalize_and_log_results(
                all_trials_metrics,
                untransformed_best_observed_all_trials,
            )

            self.mlflow_manager.end_run(status="FINISHED")

            return best_observed_df, stats_df

        except KeyboardInterrupt:
            self.logger.error("Experiment interrupted by user (KeyboardInterrupt)")
            save_log_file_to_mlflow(self.logger, self.mlflow_manager)
            self.mlflow_manager.mark_error("KeyboardInterrupt")
            self.mlflow_manager.end_run(status="FAILED")
            raise
        except Exception as e:
            import traceback

            self.logger.error(f"Experiment failed: {e}\n{traceback.format_exc()}")
            save_log_file_to_mlflow(self.logger, self.mlflow_manager)
            self.mlflow_manager.mark_error(str(e))
            self.mlflow_manager.end_run(status="FAILED")
            raise
