import logging
import os
import platform
from typing import Optional, Tuple

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import torch

# Set visible GPUs before torch import
VISIBLE_GPUS = "0, 1, 2, 3"  # Example: restrict to GPUs 0, 1, 2, and 3

logger = logging.getLogger("DEVICE UTILS")


def is_cluster_environment() -> bool:
    """Detect if running in a cluster environment.

    This function checks various system characteristics to determine if the code
    is running in a cluster environment. It looks for:
    1. Multiple GPUs (typical in cluster environments)
    2. Specific hostname patterns
    3. Environment variables commonly set in cluster environments

    Returns:
        bool: True if running in a cluster environment, False otherwise

    """
    # Check for multiple GPUs
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return True

    # Check hostname for common cluster patterns
    hostname = platform.node().lower()
    cluster_patterns = ["node", "gpu", "compute", "cluster"]
    if any(pattern in hostname for pattern in cluster_patterns):
        return True

    # Check for common cluster environment variables
    cluster_env_vars = ["SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID"]
    if any(var in os.environ for var in cluster_env_vars):
        return True

    return False


def check_cuda() -> Tuple[int, list]:
    """Check CUDA availability and select the GPU with most free memory among the visible devices.

    Returns:
        Tuple[int, list]: (selected local GPU index, list of all local indices sorted by free mem)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPUS
    print(f"Restricted to GPUs: {VISIBLE_GPUS}")

    # Parse the mapping from local index to global index
    local_to_global = [int(x) for x in VISIBLE_GPUS.split(",")]

    # After setting CUDA_VISIBLE_DEVICES, torch sees only the local indices
    torch_gpu_count = torch.cuda.device_count()
    print(f"Available GPUs (local indices): {torch_gpu_count}")
    if torch_gpu_count != len(local_to_global):
        print(
            f"WARNING: torch sees {torch_gpu_count} GPUs, but VISIBLE_GPUS has {len(local_to_global)} entries. Check CUDA_VISIBLE_DEVICES and GPU availability."
        )

    for i in range(torch_gpu_count):
        print(
            f"Local Device {i} (Global {local_to_global[i]}): {torch.cuda.get_device_name(i)}"
        )

    nvmlInit()
    free_mems = []
    # Only check the global indices in local_to_global!
    for local_idx, global_idx in enumerate(local_to_global):
        handle = nvmlDeviceGetHandleByIndex(global_idx)
        info = nvmlDeviceGetMemoryInfo(handle)
        free_mems.append((info.free, local_idx))  # local_idx is the index in torch

    free_mems.sort(reverse=True)
    best_local_idx = free_mems[0][1]
    best_global_idx = local_to_global[best_local_idx]
    print(
        f"Using local CUDA device {best_local_idx} (Global {best_global_idx}): {torch.cuda.get_device_name(best_local_idx)}"
    )
    return best_global_idx, [idx for _, idx in free_mems]


def setup_device(use_cluster: Optional[bool] = None) -> torch.device:
    """Set up the appropriate device (CPU/GPU) for training.

    This function automatically detects the environment and sets up the device
    accordingly. If use_cluster is not specified, it will auto-detect the environment.

    Args:
        use_cluster (Optional[bool]): Force cluster mode. If None, auto-detect.
            Defaults to None.

    Returns:
        torch.device: The device to use for training

    """
    # Auto-detect environment if not specified
    if use_cluster is None:
        use_cluster = is_cluster_environment()
        logger.info(
            f"Auto-detected {'cluster' if use_cluster else 'local'} environment"
        )

    if use_cluster:
        try:
            gpu_index, gpu_indices = check_cuda()
            device = torch.device(f"cuda:{gpu_index}")
            logger.info(f"Using cluster GPU setup with device: {device}")
            return device
        except Exception as e:
            logger.warning(
                f"Cluster GPU setup failed: {e}. Falling back to local setup"
            )
            use_cluster = False

    # Local setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using local GPU setup")
    else:
        device = torch.device("cpu")
        logger.info("No GPU available, using CPU")

    return device
