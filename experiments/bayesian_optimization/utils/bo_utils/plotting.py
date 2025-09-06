from __future__ import annotations

import os
from typing import Any
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def plot_results(
    stats_df: "pd.DataFrame",
    optimal_value: float | None,
    run_folder: str,
    model_name: str,
    color: str = "blue",
    alpha: float = 0.2,
) -> Any:
    """Plot BO trajectories with mean and min-max bands."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    x = stats_df.index.to_numpy()
    mean = stats_df["mean"].to_numpy()
    min_val = stats_df["min"].to_numpy()
    max_val = stats_df["max"].to_numpy()

    if optimal_value is not None:
        ax.plot(
            x,
            optimal_value * np.ones_like(x),
            color="red",
            linewidth=2,
            label="True Optimum",
        )
    ax.plot(x, mean, color=color, linewidth=2, label="Mean Best Value")
    # Cast to float arrays to appease type checkers
    ax.fill_between(
        x,
        min_val.astype(float),  # type: ignore[arg-type]
        max_val.astype(float),  # type: ignore[arg-type]
        color=color,
        alpha=alpha,
        label="Min-Max",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value")
    ax.set_title(f"Bayesian Optimization Performance - {model_name}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    # Save the figure
    plot_path = os.path.join(run_folder, "optimization_performance.png")
    plt.tight_layout()
    plt.savefig(plot_path)

    return fig
