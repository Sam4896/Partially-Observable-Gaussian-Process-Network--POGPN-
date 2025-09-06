import os
import gpytorch
import torch
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import time

from experiments.utils.device_utils import setup_device
from src.pogpn_botorch.pogpn_nodewise import POGPNNodewise
from src.pogpn_botorch.dag import DAG, RootNode, RegressionNode
from experiments.bayesian_optimization.utils.base_analysis import BaseAnalysis
from experiments.bayesian_optimization.ackley.analysis.util_func import (
    plot_3d_surface,
    plot_all_surfaces,
)
from src.synthetic_test_function import Ackley

NUM_MC_SAMPLES = 32


class POGPNNodewiseAnalysis(BaseAnalysis):
    """POGPN Nodewise Network-specific analysis implementation."""

    def setup_model(self, data_dict: Dict[str, torch.Tensor]) -> POGPNNodewise:
        """Set up and train the POGPN Nodewise network model."""
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

        with gpytorch.settings.num_likelihood_samples(NUM_MC_SAMPLES):
            model = POGPNNodewise(
                dag=dag,
                data_dict=data_dict,
                root_node_indices_dict={"x": list(range(self.sim_env.dim))},
                objective_node_name="y3",
                inducing_point_ratio=1.0,
                mll_beta=1.0,
                # mll_type="PLL",
            )
            # model.fit(optimizer="scipy", maxiter=2500)
            model.fit(optimizer="torch", lr=1e-2)

        return model

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
    time_str = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(
        os.path.dirname(__file__), "results", f"pogpn_nodewise_{time_str}"
    )
    os.makedirs(path, exist_ok=True)

    device = setup_device()
    dtype = torch.double

    sim_env = Ackley(
        dim=2,
        process_stochasticity_std=0.1,
        observation_noise_std=0.1,
    ).to(device, dtype)

    analysis = POGPNNodewiseAnalysis(
        sim_env=sim_env,
        surface_n_points=31,
        device=device,
        dtype=dtype,
        seed=42,
    )

    num_data_points = 50

    analysis.run_analysis(num_data_points=num_data_points)

    model_display_name = f"POGPN Nodewise: {num_data_points} data points"

    # Create base plots
    fig, axes = plot_all_surfaces(
        surface=analysis.objective_surface,
        data_dict=analysis.data_dict,
        title_prefix=f"{model_display_name}: ",
    )

    # Model-specific plotting
    analysis.plot_model_specific(fig, axes)

    plt.savefig(os.path.join(path, "pogpn_nodewise_analysis.png"))

    plt.show()
