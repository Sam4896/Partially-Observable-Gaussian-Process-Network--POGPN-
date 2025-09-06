from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def create_trajectory(
    best_observed_all_trials: List[List[float]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a trajectory dataframe from the best observed values for all trials."""
    best_observed_df = pd.DataFrame(best_observed_all_trials).T
    best_observed_df.columns = [
        f"Trial_{i}" for i in range(len(best_observed_all_trials))
    ]
    best_observed_df.index.name = "Iteration"

    stats_df = pd.DataFrame(
        {
            "mean": best_observed_df.mean(axis=1),
            "std": best_observed_df.std(axis=1),
            "min": best_observed_df.min(axis=1),
            "max": best_observed_df.max(axis=1),
        }
    )
    return best_observed_df, stats_df


def aggregate_metrics(all_trials_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics from all trials."""
    if not all_trials_metrics:
        return {}

    optimization_metrics = {
        "best_found_value": ["mean", "std"],
    }
    iteration_metrics = {"iteration_of_best": ["mean", "std"]}
    # model_performance_metrics = {"model_normalized_rmse": ["mean"]}
    exploration_metrics = {
        "acquisition_entropy": ["mean", "std"],
    }

    all_metric_categories = {
        **optimization_metrics,
        **iteration_metrics,
        # **model_performance_metrics,
        **exploration_metrics,
    }

    aggregated: Dict[str, float] = {}
    for metric_name, aggregations in all_metric_categories.items():
        values = [
            trial[metric_name] for trial in all_trials_metrics if metric_name in trial
        ]
        if not values:
            continue
        for agg in aggregations:
            if agg == "mean":
                aggregated[f"{metric_name}_mean"] = float(np.mean(values))
            elif agg == "std":
                aggregated[f"{metric_name}_std"] = float(np.std(values))
    return aggregated
