import os
import mlflow
from typing import Dict, Any, Optional
import mlflow.pytorch  # type: ignore
import pandas as pd
from torchinfo import summary
import tempfile
from mlflow.client import MlflowClient
import logging

logger = logging.getLogger("MLFLOW MANAGER")


class MLflowExperimentManager:
    """Simplified MLflow experiment manager optimized for GitLab MLflow."""

    def __init__(self, is_tracking_gitlab: bool):
        """Initialize the MLflow experiment manager."""
        self.experiment_name = None
        self.run_id = None
        self.is_tracking_gitlab = is_tracking_gitlab

    def get_or_create_experiment(
        self, experiment_name: str, artifact_location: Optional[str] = None
    ) -> str:
        """Get or create an MLflow experiment with proper artifact location.

        Args:
            experiment_name: Name of the experiment
            artifact_location: Optional path to the artifacts directory. Used for local tracking.

        Returns:
            Experiment ID

        """
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            if self.is_tracking_gitlab:
                experiment_id = client.create_experiment(experiment_name)
            else:
                # Create experiment with proper artifact location for local setups
                experiment_id = client.create_experiment(
                    experiment_name, artifact_location=artifact_location
                )
        else:
            experiment_id = experiment.experiment_id

        return experiment_id

    def start_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ) -> str:
        """Start an MLflow run for a specific model.

        Args:
            experiment_name: Name of the experiment
            model_name: Name of the model
            run_name: Optional custom run name
            artifact_location: Optional path to the artifacts directory. Used for local tracking.

        Returns:
            Run ID

        """
        if self.run_id is not None:
            self.run_id = None

        # Use get_or_create_experiment for proper artifact location handling
        self.experiment_id = self.get_or_create_experiment(
            experiment_name, artifact_location
        )
        mlflow.set_experiment(experiment_id=self.experiment_id)

        # Start run without context manager to keep it active
        run = mlflow.start_run(run_name=run_name)
        self.run_id = run.info.run_id
        self.experiment_name = experiment_name

        return self.run_id

    def log_configs(self, local_config_paths):
        """Log configuration files as artifacts.

        Args:
            local_config_paths: List of paths to original config files

        """
        for path in local_config_paths:
            if os.path.exists(path):
                mlflow.log_artifact(path, "configs")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number

        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_artifacts(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifacts to MLflow.

        Args:
            local_path: Path to the artifact file
            artifact_path: Optional path within the run's artifact directory

        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        mlflow.log_params(params)

    def log_model_summary(self, model, model_name: str):
        """Log model summary to MLflow."""
        try:
            model_summary = str(
                summary(
                    model,
                    depth=5,
                    col_names=["num_params", "trainable"],
                    row_settings=[
                        "var_names",
                    ],
                )
            )
            mlflow.log_text(model_summary, f"{model_name}_summary.txt")
        except Exception as e:
            logger.info(f"Warning: Could not log model summary for {model_name}: {e}")

    def log_dataframe(self, df: pd.DataFrame, name: str):
        """Log a pandas DataFrame as an artifact.

        Args:
            df: DataFrame to log
            name: Name for the artifact

        """
        if not self.is_tracking_gitlab:
            mlflow.log_table(df, f"results/{name}.json")
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_file = os.path.join(tmp_dir, f"{name}.csv")
                df.to_csv(tmp_file, index=False)
                mlflow.log_artifact(tmp_file, "results")

    def log_plot(self, fig, name: str):
        """Log a matplotlib figure as an artifact.

        Args:
            fig: Matplotlib figure
            name: Name for the artifact

        """
        mlflow.log_figure(fig, f"plots/{name}.png")

    def get_artifact_uri(self) -> str:
        """Get the artifact URI for the current MLflow run.

        Returns:
            Path to the MLflow artifacts directory

        """
        if self.run_id:
            return mlflow.get_artifact_uri()
        else:
            raise RuntimeError("No active MLflow run. Call start_run() first.")

    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run with optional status.

        Args:
            status: Run status ('FINISHED', 'FAILED', 'KILLED')

        """
        if self.run_id:
            mlflow.end_run(status=status)

    def mark_error(self, error_message: str):
        """Mark run with error tag.

        Args:
            error_message: Error description

        """
        if self.run_id:
            mlflow.set_tag("error", "true")
            mlflow.set_tag("error_message", error_message)

    def mark_warning(self, warning_message: str):
        """Mark run with warning tag.

        Args:
            warning_message: Warning description

        """
        if self.run_id:
            mlflow.set_tag("warning", "true")
            mlflow.set_tag("warning_message", warning_message)

    def mark_partial_failure(self, error_message: str):
        """Mark run as partially failed (completed with warnings/errors)."""
        if self.run_id:
            mlflow.set_tag("partial_failure", "true")
            mlflow.set_tag("partial_failure_message", error_message)

    # ---------------------------
    # Tag helpers
    # ---------------------------

    def set_run_tags(self, tags: Dict[str, str]):
        """Attach arbitrary tags to the *current* run. Only non-empty values are set."""
        for k, v in tags.items():
            if v is not None and str(v).strip() != "":
                mlflow.set_tag(k, str(v))

    def set_run_description(self, description: str):
        """Set the MLflow run description (note)."""
        if description and str(description).strip() != "":
            mlflow.set_tag("mlflow.description", str(description))

    def set_run_note(self, note: str):
        """Set the MLflow run note."""
        if note and str(note).strip() != "":
            mlflow.set_tag("mlflow.note.content", str(note))


def setup_mlflow_tracking(
    experiment_name: str,
    folder_name: str,
    is_tracking_gitlab: bool,
    mlflow_tracking_uri: Optional[str] = None,
    run_name: Optional[str] = None,
) -> MLflowExperimentManager:
    """Set up MLflow tracking for a Bayesian optimization experiment.

    Args:
        experiment_name: Name of the experiment
        model_name: Name of the model
        folder_name: Name of the folder to store the results
        is_tracking_gitlab: Whether to track the experiment on GitLab
        mlflow_tracking_uri: Optional MLflow tracking URI (uses environment variable if not provided)
        run_name: Optional custom run name (if None, auto-generated)

    Returns:
        Configured MLflowExperimentManager

    """
    artifact_location = None

    if not is_tracking_gitlab:
        if mlflow_tracking_uri is None:
            logger.info("Using fallback local SQLite for backward compatibility")
            # Fallback to local SQLite for backward compatibility
            sqlite_db = os.path.join(folder_name, "mlflow.db")
            mlflow_tracking_uri = f"sqlite:///{sqlite_db}"
            artifact_location = os.path.join(folder_name, "mlruns")
            os.makedirs(artifact_location, exist_ok=True)

        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)

    else:
        artifact_location = None

    # Create manager and start run
    manager = MLflowExperimentManager(is_tracking_gitlab)
    manager.start_run(
        experiment_name=experiment_name,
        run_name=run_name,
        artifact_location=artifact_location,
    )

    return manager
