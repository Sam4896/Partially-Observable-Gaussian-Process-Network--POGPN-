import importlib
from typing import Any, Dict

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import NaN
import torch

from experiments.bayesian_optimization.utils import sim_env_utils
from botorch.utils.transforms import unnormalize
from experiments.utils.device_utils import setup_device
from src.synthetic_test_function.base.dag_experiment_base import (
    DAGSyntheticTestFunction,
)


def draw_synthetic_test_function_1d(test_function: DAGSyntheticTestFunction):
    pass


def draw_synthetic_test_function_2d(
    test_function_name: str, best_value: float = torch.nan
):
    device = setup_device()
    dtype = torch.float
    n = 1000

    # construct function
    module = importlib.import_module("src.synthetic_test_function")
    sim_class = getattr(module, test_function_name)
    test_function: DAGSyntheticTestFunction = sim_class(dim=2).to(
        device=device, dtype=dtype
    )

    # create input
    values = torch.linspace(0, 1, n, device=device, dtype=dtype)
    x, y = torch.meshgrid(
        unnormalize(values, torch.tensor(test_function._bounds[0])),
        unnormalize(values, torch.tensor(test_function._bounds[1])),
    )
    inputs = torch.cartesian_prod(values, values)

    # evaluate function
    output = sim_env_utils.evaluate_output_from_sim_env(test_function, inputs)

    # show function
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.plot_surface(x.cpu(), y.cpu(), output["y3"].reshape([n, n]).cpu(), cmap=cm.Blues)
    if best_value is not torch.nan:
        ax.plot_surface(
            x.cpu(),
            y.cpu(),
            best_value * torch.ones([n, n]),
            cmap=cm.Greens,
        )

    plt.show()


if __name__ == "__main__":
    draw_synthetic_test_function_2d("Michalewicz", -1.8013)
