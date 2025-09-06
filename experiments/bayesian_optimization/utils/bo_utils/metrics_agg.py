from __future__ import annotations

from typing import Any, Dict, List, Tuple

import logging
import pandas as pd

from .metrics import create_trajectory, aggregate_metrics


class MetricsAggregator:
    """Creates trajectory data, aggregates metrics, and logs them to MLflow."""

    def __init__(
        self, mlflow_manager: Any, logger: logging.Logger, sim_env: Any
    ) -> None:
        self.mlflow_manager = mlflow_manager
        self.logger = logger
        self.sim_env = sim_env

    def create_trajectory(
        self, best_observed_all_trials: List[List[float]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return create_trajectory(best_observed_all_trials)

    def aggregate(self, all_trials_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        return aggregate_metrics(all_trials_metrics)

    def log_all(
        self,
        best_observed_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        all_trials_metrics: List[Dict[str, float]],
        aggregated_metrics: Dict[str, float],
    ) -> None:
        self.mlflow_manager.log_dataframe(best_observed_df, "best_observed_values")
        self.mlflow_manager.log_dataframe(stats_df, "optimization_statistics")
        if all_trials_metrics:
            all_trials_df = pd.DataFrame(all_trials_metrics)
            all_trials_df.index.name = "Trial"
            self.mlflow_manager.log_dataframe(all_trials_df, "all_trials_metrics")
        if aggregated_metrics:
            self.mlflow_manager.log_metrics(aggregated_metrics)
        self.logger.info(
            f"Logged results: {len(all_trials_metrics)} trials, {len(aggregated_metrics)} aggregated metrics"
        )
