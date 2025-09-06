from __future__ import annotations

import os
import logging
import pathlib
import tempfile
from typing import Dict, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from experiments.utils.mlflow_manager import MLflowExperimentManager


def save_data_dict(
    mlflow_manager: MLflowExperimentManager,
    data_dict: Dict[str, torch.Tensor],
    name: str,
    run_folder: str,
    artifact_path: str = "data_points_explored",
) -> None:
    """Save the data dictionary to a file and log it to MLflow."""
    save_dict = {key: tensor.cpu().numpy() for key, tensor in data_dict.items()}
    file_path = os.path.join(run_folder, f"{name}.npz")
    np.savez(file_path, **save_dict)
    mlflow_manager.log_artifacts(local_path=file_path, artifact_path=artifact_path)


def save_script(mlflow_manager: MLflowExperimentManager, script_path: str):
    """Save the current script to a file and log it to MLflow."""
    file_name = pathlib.Path(script_path).stem
    temp_dir = tempfile.mkdtemp()
    temp_script_path = os.path.join(temp_dir, f"{file_name}.txt")

    try:
        with open(script_path, "r") as f:
            script_content = f.read()
        with open(temp_script_path, "w") as f:
            f.write(script_content)
        mlflow_manager.log_artifacts(temp_script_path, "scripts")
    finally:
        if os.path.exists(temp_script_path):
            os.unlink(temp_script_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


def save_log_file_to_mlflow(
    logger: logging.Logger, mlflow_manager: MLflowExperimentManager
):
    """Save the log file to MLflow."""
    log_file_path = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):  # type: ignore[name-defined]
            log_file_path = handler.baseFilename
    for handler in logging.root.handlers:  # type: ignore[name-defined]
        if isinstance(handler, logging.FileHandler):  # type: ignore[name-defined]
            log_file_path = handler.baseFilename
    if log_file_path and os.path.exists(log_file_path):
        mlflow_manager.log_artifacts(log_file_path, "logs")
        logger.info(f"Uploaded log file from {log_file_path} to MLflow")
