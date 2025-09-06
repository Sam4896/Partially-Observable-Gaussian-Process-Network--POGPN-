import logging
import os
import torch
import yaml
from io import StringIO


def format_dict_as_yaml(d: dict) -> str:
    """Format a dictionary as a YAML string.

    Args:
        d (dict): Dictionary to format.

    Returns:
        str: YAML formatted string.

    """
    stream = StringIO()
    yaml.dump(d, stream, default_flow_style=False, sort_keys=False)
    return stream.getvalue()


def setup_logging(exp_folder: str | None = None, log_file: str = "logs.log"):
    """Set up the logging configuration.

    Args:
        exp_folder (str): Directory where the log file will be stored.
        log_file (str, optional): Name of the log file. Defaults to "logs.log".

    Returns:
        logging.Logger: Configured logger instance.

    """
    if exp_folder is not None:
        log_path = os.path.join(exp_folder, log_file)
    else:
        log_path = log_file

    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Simple format: timestamp | level | message
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


def load_model_state(model: torch.nn.Module, file_name: str) -> torch.nn.Module:
    """Load the state dictionary from a file into a model.

    Args:
        model (torch.nn.Module): The model to load the state into.
        file_name (str): The file path from which to load the state.

    """
    try:
        model.load_state_dict(torch.load(file_name, weights_only=True))
        logging.info(f"[Model Train/Eval] Model state loaded from {file_name}.")
        return model
    except Exception as e:
        logging.error(
            f"[Model Train/Eval] Failed to load model state from {file_name}: {e}"
        )
        raise
