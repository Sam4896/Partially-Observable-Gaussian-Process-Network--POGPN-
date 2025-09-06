from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize, OutcomeTransform
from botorch.sampling import SobolQMCNormalSampler
from experiments.bayesian_optimization.utils.sim_env_utils import (
    get_initial_training_data,
    evaluate_output_from_sim_env,
)
from experiments.utils.device_utils import setup_device

from experiments.bayesian_optimization.ackley.analysis.util_func import plot_3d_surface
from botorch.utils.transforms import normalize

import yaml

from src.synthetic_test_function.base.dag_experiment_base import (
    DAGSyntheticTestFunction,
)

QMC_SAMPLER_SAMPLE_SHAPE = 32


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class BaseAnalysis(ABC):
    """Base class for model analysis with common functionality abstracted."""

    def __init__(
        self,
        sim_env: DAGSyntheticTestFunction,
        surface_n_points: int = 21,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize base analysis parameters."""
        # Load configurations
        self.sim_env = sim_env

        self.device = setup_device() if device is None else device
        self.dtype = torch.double if dtype is None else dtype

        self.surface_n_points = surface_n_points

        if seed is None:
            self.seed = int(torch.randint(0, 1000000, (1,)).item())
        else:
            self.seed = seed

        # Initialize components
        self.data_dict = None
        self.X_grid = None
        self.X1 = None
        self.X2 = None
        self.objective_surface = None
        self.sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([QMC_SAMPLER_SAMPLE_SHAPE]),
            seed=seed,
        )

    def setup_evaluation_grid(self) -> None:
        """Create evaluation grid and get surface values."""
        if self.sim_env is None:
            raise ValueError("sim_env must be set up before creating evaluation grid")

        x1 = torch.linspace(0, 1, self.surface_n_points).to(
            self.device, dtype=self.dtype
        )
        x2 = torch.linspace(0, 1, self.surface_n_points).to(
            self.device, dtype=self.dtype
        )
        self.X1, self.X2 = torch.meshgrid(x1, x2, indexing="ij")
        self.X_grid = torch.stack([self.X1.flatten(), self.X2.flatten()], dim=1)

        # Get function values for the grid points
        self.objective_surface = evaluate_output_from_sim_env(self.sim_env, self.X_grid)
        for key, transform in self.transforms.items():
            if isinstance(transform, OutcomeTransform):
                self.objective_surface[key] = transform(self.objective_surface[key])[0]
        self.objective_surface["inputs"] = normalize(
            self.objective_surface["inputs"], bounds=self.sim_env.bounds
        )

    @abstractmethod
    def setup_model(self, data_dict: Dict[str, torch.Tensor]) -> Any:
        """Set up and train the model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_posterior_predictions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior predictions on the evaluation grid.

        Returns:
            Tuple of (mean, std) tensors

        """
        pass

    @abstractmethod
    def plot_model_specific(self, fig, axes) -> None:
        """Plot model-specific visualizations. Must be implemented by subclasses."""
        pass

    def run_analysis(
        self,
        num_data_points: int,
        data_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Run the complete analysis pipeline."""
        # Setup
        if data_dict is None:
            train_x = get_initial_training_data(
                input_dim=self.sim_env.dim,
                num_samples=num_data_points,
                device=self.device,
                dtype=self.dtype,
                seed=self.seed,
            )
            data_dict = evaluate_output_from_sim_env(self.sim_env, train_x)

        self.data_dict = data_dict

        self.transforms = {
            "y1": Standardize(m=1),
            "y2": Standardize(m=1),
            "y3": Standardize(m=1),
        }

        # Apply transforms to data_dict
        if self.data_dict is not None:
            self.data_dict["inputs"] = normalize(
                self.data_dict["inputs"], bounds=self.sim_env.bounds
            )
            self.data_dict["y1"] = self.transforms["y1"](self.data_dict["y1"])[0]
            self.data_dict["y2"] = self.transforms["y2"](self.data_dict["y2"])[0]
            self.data_dict["y3"] = self.transforms["y3"](self.data_dict["y3"])[0]

            for key, data in self.data_dict.items():
                self.data_dict[key] = data.detach()

        for transform in self.transforms.values():
            transform.eval()

        self.setup_evaluation_grid()

        # Train model
        if self.data_dict is not None:
            self.model = self.setup_model(data_dict=self.data_dict)

    def plot_surface_with_uncertainty(
        self,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
        X_grid: Optional[np.ndarray] = None,  # noqa: N803
        Y_grid: Optional[np.ndarray] = None,  # noqa: N803
        mean_grid: Optional[np.ndarray] = None,
        std_grid: Optional[np.ndarray] = None,
        ax=None,
        colorbar_label: str = "Posterior Mean",
        scatter_points: Optional[Dict[str, np.ndarray]] = None,
        scatter_points_color: Optional[str] = None,
        scatter_points_label: Optional[str] = None,
    ) -> None:
        """Plot surface with uncertainty bands."""
        if self.X1 is None or self.X2 is None:
            raise ValueError("Evaluation grid must be set up before plotting")

        # Plot the posterior mean surface
        _ = plot_3d_surface(
            x=x,
            y=y,
            z=z,
            X_grid=X_grid,
            Y_grid=Y_grid,
            Z_grid=mean_grid,
            colorbar_label=colorbar_label,
            cmap="Reds",
            alpha=0.5,
            ax=ax,
            plot_contours=False,
            plot_colorbar=True,
            scatter_points=scatter_points,
            scatter_points_color=scatter_points_color,
            scatter_points_label=scatter_points_label,
        )

        if mean_grid is not None and std_grid is not None:
            # Plot the confidence region
            plot_3d_surface(
                x=x,
                y=y,
                z=z,
                X_grid=X_grid,
                Y_grid=Y_grid,
                Z_grid=(mean_grid + 2 * std_grid),
                cmap="Greens",
                alpha=0.3,
                ax=ax,
                plot_contours=False,
                plot_colorbar=False,
            )

            plot_3d_surface(
                x=x,
                y=y,
                z=z,
                X_grid=X_grid,
                Y_grid=Y_grid,
                Z_grid=(mean_grid - 2 * std_grid),
                cmap="Greens",
                alpha=0.1,
                linewidth=0,
                ax=ax,
                plot_contours=False,
                plot_colorbar=False,
            )
