"""Analysis utilities for POGPN Pathwise on the Ackley function.

This module trains a POGPN Pathwise model and provides plotting utilities,
including a combined 1x4 loss figure (total + per-node losses).
"""

import gpytorch
import torch
from typing import Dict, Tuple
import matplotlib.pyplot as plt

from experiments.utils.device_utils import setup_device
from src.pogpn_botorch.pogpn_pathwise import POGPNPathwise
from src.pogpn_botorch.dag import DAG, RootNode, RegressionNode, ClassificationNode
from experiments.bayesian_optimization.utils.base_analysis import BaseAnalysis
from experiments.bayesian_optimization.ackley.analysis.util_func import (
    plot_3d_surface,
    plot_all_surfaces,
)
from src.synthetic_test_function import Rosenbrock

mll_type = "PLL"
mll_beta = 0.5

NUM_MC_SAMPLES = 64


class POGPNPathwiseAnalysis(BaseAnalysis):
    """POGPN Pathwise Network-specific analysis implementation."""

    def setup_model(self, data_dict: Dict[str, torch.Tensor]) -> POGPNPathwise:
        """Set up and train the POGPN Pathwise network model."""
        x_node = RootNode(
            name="x",
            parents=[],
            node_output_dim=2,
        )
        y1_node = RegressionNode(
            name="y1",
            parents=[x_node],
            node_output_dim=1,
        )
        y2_node = RegressionNode(
            name="y2",
            parents=[x_node],
            node_output_dim=1,
        )
        y3_node = RegressionNode(
            name="y3",
            parents=[y1_node, y2_node],
            node_output_dim=1,
        )
        dag = DAG(dag_nodes=[x_node, y1_node, y2_node, y3_node])

        loss_history = []

        with gpytorch.settings.num_likelihood_samples(NUM_MC_SAMPLES):
            model = POGPNPathwise(
                dag=dag,
                data_dict=data_dict,
                root_node_indices_dict={"x": list(range(self.sim_env.dim))},
                objective_node_name="y3",
                inducing_point_ratio=1.0,
                mll_beta=mll_beta,
                mll_type=mll_type,
            )
            model.fit(
                data_dict=data_dict,
                optimizer="torch",
                loss_history=loss_history,
                lr=2e-2,
            )
            # model.fit_gpytorch_mll_custom_torch(loss_history=loss_history)

        self.plot_combined_loss_histories(model=model, total_loss_history=loss_history)

        return model

    def plot_combined_loss_histories(self, model: POGPNPathwise, total_loss_history):
        """Plot total loss and per-node losses in a single 1x4 figure and show once."""
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))

        # Total loss
        axes[0].plot(total_loss_history, label="Total training loss")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Total training loss")
        axes[0].legend()

        # Assume three non-root nodes (y1, y2, y3)
        for i, node_name in enumerate(model.non_root_nodes[:3]):
            node = model.dag_nodes[node_name]
            ax = axes[i + 1]
            if isinstance(node, (RegressionNode, ClassificationNode)) and hasattr(
                node, "node_mll_loss_history"
            ):
                for key in node.node_mll_loss_history.keys():
                    ax.plot(node.node_mll_loss_history[key], label=key)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.set_title(f"{node_name} loss history")
            ax.legend()
            ax.set_ylim(-3, 3)

        plt.tight_layout()
        # plt.savefig(os.path.join(path, "combined_loss_history.png"))
        plt.show()

    def get_posterior_predictions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior predictions on the evaluation grid."""
        if self.model is None or self.X_grid is None or self.sampler is None:
            raise ValueError(
                "Model, grid, and sampler must be set up before getting predictions"
            )

        with torch.no_grad():
            posterior = self.model.posterior(self.X_grid)
            samples = self.sampler(posterior)
            samples_mean = samples.mean(dim=0)
            samples_std = samples.std(dim=0)

        return samples_mean, samples_std

    def plot_model_specific(self, fig, axes) -> None:
        """Plot GP Network-specific visualizations."""
        samples_mean, samples_std = self.get_posterior_predictions()

        # Set custom view angles for each subplot
        # Format: (elevation, azimuth, roll)
        view_angles = [
            (12, -83, 0),  # upper left: Y1 vs X1, X2
            (14, -113, 0),  # upper right: Y2 vs X1, X2
            (1, -81, 0),  # lower left: Y3 vs X1, X2
            (3, 3, 0),  # lower right: Y3 vs Y1, Y2
        ]

        # Plot for outputs y1, y2, y3 vs x
        for i, node_name in enumerate(self.model.non_root_nodes):
            plot_mean = samples_mean[:, i].unsqueeze(-1)
            plot_std = samples_std[:, i].unsqueeze(-1)

            # Reshape predictions to match the grid
            mean_grid = plot_mean.reshape(self.surface_n_points, self.surface_n_points)
            std_grid = plot_std.reshape(self.surface_n_points, self.surface_n_points)

            # Plot surface with uncertainty for each output
            if i < 2:
                inducing_points = self.model.dag_nodes[
                    node_name
                ].node_model.model.variational_strategy.inducing_points.detach()
                if (
                    hasattr(
                        self.model.dag_nodes[node_name].node_model, "input_transform"
                    )
                    and self.model.dag_nodes[node_name].node_model.input_transform
                    is not None
                ):
                    inducing_points = self.model.dag_nodes[
                        node_name
                    ].node_model.input_transform.untransform(inducing_points)
                scatter_points = {
                    "x": inducing_points[:, 0],
                    "y": inducing_points[:, 1],
                    "z": (
                        torch.zeros_like(inducing_points[:, 0])  # type: ignore
                        + torch.max(self.objective_surface[f"{node_name}"])  # type: ignore
                    ),
                }
                scatter_points_color = "red"
                scatter_points_label = "Inducing points"
            else:
                scatter_points = None
                scatter_points_color = None
                scatter_points_label = None

            self.plot_surface_with_uncertainty(
                X_grid=self.X1.cpu().numpy(),  # type: ignore
                Y_grid=self.X2.cpu().numpy(),  # type: ignore
                mean_grid=mean_grid.cpu().numpy(),
                std_grid=std_grid.cpu().numpy(),
                ax=axes[i],
                colorbar_label=f"Posterior Mean Y{i + 1}",
                scatter_points=scatter_points,
                scatter_points_color=scatter_points_color,
                scatter_points_label=scatter_points_label,
            )

            # Set custom view angle for this subplot
            if i < len(view_angles):
                elev, azim, roll = view_angles[i]
                axes[i].view_init(elev=elev, azim=azim, roll=roll)

        # Plot y3 vs y1, y2 (4th subplot)
        final_node = "y3"
        inducing_points = self.model.dag_nodes[
            final_node
        ].node_model.model.variational_strategy.inducing_points.detach()
        if (
            hasattr(self.model.dag_nodes[final_node].node_model, "input_transform")
            and self.model.dag_nodes[final_node].node_model.input_transform is not None
        ):
            inducing_points = self.model.dag_nodes[
                final_node
            ].node_model.input_transform.untransform(inducing_points)
        plot_3d_surface(
            x=samples_mean[:, 0].unsqueeze(-1).cpu().numpy(),
            y=samples_mean[:, 1].unsqueeze(-1).cpu().numpy(),
            z=samples_mean[:, 2].unsqueeze(-1).cpu().numpy(),
            colorbar_label="Posterior Mean Y3",
            cmap="Reds",
            alpha=0.5,
            ax=axes[3],
            method="nearest",
            plot_contours=False,
            plot_colorbar=True,
            scatter_points={
                "x": inducing_points[:, 0],
                "y": inducing_points[:, 1],
                "z": (
                    torch.zeros_like(inducing_points[:, 0])  # type: ignore
                    + torch.max(self.objective_surface[f"{final_node}"])  # type: ignore
                ),
            },
            scatter_points_color="red",
            scatter_points_label="Inducing points",
        )

        # Set custom view angle for the 4th subplot (Y3 vs Y1, Y2)
        elev, azim, roll = view_angles[3]
        axes[3].view_init(elev=elev, azim=azim, roll=roll)

        # Plot the confidence region
        plot_3d_surface(
            x=samples_mean[:, 0].unsqueeze(-1).cpu().numpy(),
            y=samples_mean[:, 1].unsqueeze(-1).cpu().numpy(),
            z=(samples_mean[:, 2].unsqueeze(-1) + 2 * samples_std[:, 2].unsqueeze(-1))
            .cpu()
            .numpy(),
            cmap="Greens",
            alpha=0.3,
            ax=axes[3],
            method="nearest",
            plot_contours=False,
            plot_colorbar=False,
        )

        plot_3d_surface(
            x=samples_mean[:, 0].unsqueeze(-1).cpu().numpy(),
            y=samples_mean[:, 1].unsqueeze(-1).cpu().numpy(),
            z=(samples_mean[:, 2].unsqueeze(-1) - 2 * samples_std[:, 2].unsqueeze(-1))
            .cpu()
            .numpy(),
            cmap="Greens",
            alpha=0.1,
            linewidth=0,
            ax=axes[3],
            method="nearest",
            plot_contours=False,
            plot_colorbar=False,
        )


if __name__ == "__main__":
    device = setup_device()
    dtype = torch.double

    sim_env = Rosenbrock(
        dim=2,
        process_stochasticity_std=0.1,
        observation_noise_std=0.1,
    ).to(device, dtype)

    analysis = POGPNPathwiseAnalysis(
        sim_env=sim_env,
        surface_n_points=31,
        device=device,
        dtype=dtype,
        seed=42,
    )

    num_data_points = 15

    analysis.run_analysis(num_data_points=num_data_points)

    model_display_name = f"POGPN Pathwise: Rosenbrock(2D) ({num_data_points} data points) ({mll_type}), beta={mll_beta}"

    # Create base plots
    fig, axes = plot_all_surfaces(
        surface=analysis.objective_surface,
        data_dict=analysis.data_dict,
        title_prefix=f"{model_display_name}: ",
    )

    # Model-specific plotting
    analysis.plot_model_specific(fig, axes)

    plt.show()
