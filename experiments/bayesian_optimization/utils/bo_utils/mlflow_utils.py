from __future__ import annotations

import os
from urllib.parse import urlparse, unquote


def setup_gitlab_mlflow_env(config):
    """Set up GitLab MLflow environment variables from config file."""
    mlflow_tracking_uri = (
        f"{config['gitlab_endpoint']}/api/v4/projects/{config['project_id']}/ml/mlflow"
    )
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    os.environ["MLFLOW_TRACKING_TOKEN"] = config["access_token"]
    return mlflow_tracking_uri


def setup_mlflow_logging_dir(mlflow_manager, run_logs_dirname: str = "logs") -> str:
    """Set up the MLflow logging directory."""
    artifact_uri = mlflow_manager.get_artifact_uri()
    parsed_uri = urlparse(artifact_uri)
    if parsed_uri.scheme == "file":
        mlflow_artifacts_dir = (
            parsed_uri.path[1:] if parsed_uri.path.startswith("/") else parsed_uri.path
        )
        mlflow_artifacts_dir = unquote(mlflow_artifacts_dir)
    else:
        mlflow_artifacts_dir = artifact_uri
    logs_dir = os.path.join(mlflow_artifacts_dir, run_logs_dirname)
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir
