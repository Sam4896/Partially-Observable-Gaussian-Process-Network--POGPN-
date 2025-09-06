import numpy as np
import torch
from typing import Dict, List, Any, Optional
from botorch.acquisition import AcquisitionFunction
from src.other_models.gp_network.gp_network import MultivariateNormalNetwork
from src.pogpn_botorch.pogpn_posterior import POGPNPosterior


class BOMetricsCalculator:
    """Lightweight calculator for Bayesian Optimization metrics.

    This class computes various BO performance metrics from the final data_dict
    and can be used as an optional add-on to any BO experiment.

    Mathematical Descriptions of Implemented Metrics:

    ## Basic Performance Metrics

    **Total Improvement:**
    I_total = |f(x_T) - f(x_0)|
    where f(x_T) is the final best value and f(x_0) is the initial value.

    ## Exploration Metrics

    **Exploration Efficiency:**
    E_eff = (1/n) * Σ_{i=1}^n min_{j≠i} ||x_i - x_j||_2 / E[d_random]
    where n is the number of points, ||·||_2 is the L2 norm, and E[d_random] ≈ 0.5√d is the expected
    random distance in d-dimensional unit hypercube.

    **Coverage Ratio (Grid-based for d ≤ 3):**
    C_grid = |{g ∈ G : min_{x ∈ X} ||g - x||_2 < r}| / |G|
    where G is a uniform grid, X is the set of evaluated points, and r is the radius threshold.

    **Coverage Ratio (Distance-based for d > 3):**
    C_dist = min(n * (d̄_min)^d, 1.0)
    where d̄_min is the average minimum distance between points.

    **Dispersion:**
    D = min_{i,j: i≠j} ||x_i - x_j||_2
    The minimum distance between any two evaluated points.

    ## Acquisition Function Metrics

    **Acquisition Diversity:**
    σ_acq = √[(1/m) * Σ_{i=1}^m (α(x_i) - μ_acq)²]
    where α(x_i) are acquisition values at m random points and μ_acq is their mean.

    **Acquisition Entropy:**
    H_acq = -Σ_{i=1}^m p_i * log(p_i)
    where p_i = α_norm(x_i) / Σ_j α_norm(x_j) and α_norm are normalized acquisition values.

    ## Model Fit Quality Metrics

    **Normalized RMSE:**
    NRMSE = RMSE / σ_y
    where σ_y is the standard deviation of the target values.

    ## Improvement Rate Metrics

    **Overall Improvement Rate:**
    R_overall = |f(x_best) - f(x_0)| / T
    where T is the total number of iterations.

    **Recent Improvement Rate:**
    R_recent = |f(x_T) - f(x_{T-w})| / (w-1)
    where w is the window size (default: 5).

    **Exponential Improvement Rate:**
    R_exp = -slope of linear fit to log(|f(x_t) - f*|) vs. t
    This measures the exponential decay rate of the gap from the optimal value.

    """

    def __init__(self, sim_env, device: torch.device, dtype: torch.dtype):
        """Initialize the metrics calculator.

        Args:
            sim_env: Simulation environment with dim, minimize, and objective_node_name
            device: PyTorch device
            dtype: PyTorch data type

        """
        self.sim_env = sim_env
        self.device = device
        self.dtype = dtype

    def _is_minimization(self) -> bool:
        """Safely check if the optimization task is minimization.

        Returns:
            True if minimization, False if maximization (default)

        """
        return getattr(self.sim_env, "minimize", False)

    def compute_all_metrics(
        self,
        data_dict: Dict[str, torch.Tensor],
        best_observed_history: List[float],
        model: Any,
        acqf: AcquisitionFunction,
    ) -> Dict[str, float]:
        """Compute all BO metrics from the final experiment data.

        Args:
            data_dict: Final data dictionary with all explored points
            best_observed_history: List of best observed values per iteration
            model: Final trained model
            acqf: Final acquisition function

        Returns:
            Dictionary of all computed metrics

        """
        metrics = {}

        # Basic performance metrics
        basic_metrics = self._compute_basic_metrics(best_observed_history)
        metrics.update(basic_metrics)

        # Acquisition function metrics - commented out as not needed for current use case
        acq_metrics = self._compute_acquisition_metrics(acqf)
        metrics.update(acq_metrics)

        # Model fit quality metrics
        # fit_metrics = self._compute_model_fit_metrics(model, data_dict)
        # metrics.update(fit_metrics)

        # Round all metrics to reasonable precision for cleaner output
        rounded_metrics = {}
        for key, value in metrics.items():
            # Convert to float if it's a numeric type, then round appropriately
            if isinstance(value, (int, float, np.integer, np.floating)):
                if key == "iteration_of_best":
                    rounded_metrics[key] = int(value)  # Keep as integer
                else:
                    # Convert to float and round to 4 decimal places
                    rounded_metrics[key] = round(float(value), 4)
            else:
                rounded_metrics[key] = value

        return rounded_metrics

    def _compute_basic_metrics(
        self, best_observed_history: List[float]
    ) -> Dict[str, float]:
        """Compute basic BO performance metrics.

        Returns:
            Dictionary containing:
            - best_found_value: Best objective value found
            - iteration_of_best: Iteration where best value was found

        """
        if not best_observed_history:
            return {}

        best_values = np.array(best_observed_history)

        # Find best value and its iteration
        if self._is_minimization():
            best_idx = np.argmin(best_values)
            best_value = best_values[best_idx]
        else:
            best_idx = np.argmax(best_values)
            best_value = best_values[best_idx]

        return {
            "best_found_value": float(best_value),  # Best objective value found
            "iteration_of_best": int(best_idx),  # Iteration where best was found
        }

    def _compute_acquisition_metrics(
        self, acqf: AcquisitionFunction
    ) -> Dict[str, float]:
        """Compute acquisition function diversity and behavior metrics.

        Mathematical formulations:
        - Acquisition Entropy: H_acq = -Σ p_i * log(p_i)
        where p_i = α_norm(x_i) / Σ_j α_norm(x_j)

        Returns:
            Dictionary containing:
            - acquisition_entropy: Entropy of normalized acquisition values

        """
        try:
            # Sample random points to evaluate acquisition function
            n_samples = min(1000, 100 * self.sim_env.dim)  # Adaptive sampling

            random_points = torch.rand(
                n_samples, self.sim_env.dim, dtype=self.dtype, device=self.device
            )

            # Evaluate acquisition function
            with torch.no_grad():
                # Try different input shapes for acquisition function
                try:
                    acq_values = acqf(random_points.unsqueeze(1))  # Add batch dimension
                except Exception:
                    # Try without batch dimension
                    acq_values = acqf(random_points)

                # Ensure we have the right shape
                if acq_values.dim() > 1:
                    acq_values = acq_values.squeeze()

            # Compute entropy-based diversity
            if len(acq_values) > 1:
                # Normalize and compute entropy
                acq_normalized = acq_values - acq_values.min()
                if acq_normalized.max() > 0:
                    acq_probs = acq_normalized / acq_normalized.sum()
                    entropy = -torch.sum(acq_probs * torch.log(acq_probs + 1e-8)).item()
                else:
                    entropy = 0.0
            else:
                entropy = 0.0

            return {
                "acquisition_entropy": entropy,
            }

        except Exception as e:
            # Fallback in case of errors - log the error for debugging
            print(f"Warning: Failed to compute acquisition metrics: {e}")
            return {
                "acquisition_entropy": 0.0,
            }

    def _compute_model_fit_metrics(
        self, model: Any, data_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute model fit quality metrics.

        Mathematical formulations:
        - Normalized RMSE: RMSE / σ_y


        Returns:
            Dictionary containing:
            - model_normalized_rmse: Normalized Root Mean Square Error of model predictions

        """
        input_key = self._find_input_key(data_dict)
        if input_key is None:
            return {}

        X = data_dict[input_key].to(self.device, self.dtype)

        objective_key = getattr(self.sim_env, "objective_node_name", None)
        if objective_key is None:
            # Try to find the objective key automatically
            for key, tensor in data_dict.items():
                if tensor.dim() == 1 and key != input_key:
                    objective_key = key
                    break
            if objective_key is None:
                return {}  # Cannot find objective values

        y = data_dict[objective_key].to(self.device, self.dtype)

        if len(X) < 2:
            return {}

        # Get model predictions
        with torch.no_grad():
            posterior = model.posterior(X)

            # Handle different posterior types
            if hasattr(posterior, "mean"):
                # Standard posterior with .mean property
                pred_mean = posterior.mean.squeeze()
            else:
                # For GP Network models that don't have .mean, use sampling
                # Sample multiple times and take the mean

                num_samples = 100
                if isinstance(posterior, POGPNPosterior) or isinstance(
                    posterior, MultivariateNormalNetwork
                ):
                    samples = posterior.rsample_objective_node(
                        sample_shape=torch.Size([num_samples])
                    )
                else:
                    samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
                pred_mean = samples.mean(dim=0).squeeze()

        # Compute fit metrics
        rmse = torch.sqrt(torch.mean((pred_mean - y.squeeze()) ** 2)).item()

        # Normalized metrics
        y_std = torch.std(y).item()
        normalized_rmse = rmse / (y_std + 1e-8)

        return {
            "model_normalized_rmse": normalized_rmse,
        }

    def _find_input_key(self, data_dict: Dict[str, torch.Tensor]) -> Optional[str]:
        """Find the input key in the data dictionary."""
        # Common input key names
        possible_keys = ["X", "inputs", "input", "x"]

        for key in possible_keys:
            if key in data_dict:
                return key

        # If not found, look for the key with the right dimensionality
        for key, tensor in data_dict.items():
            if tensor.dim() == 2 and tensor.shape[1] == self.sim_env.dim:
                return key

        return None


def compute_bo_metrics(
    sim_env,
    device: torch.device,
    dtype: torch.dtype,
    data_dict: Dict[str, torch.Tensor],
    best_observed_history: List[float],
    model: Any,
    acqf: AcquisitionFunction,
) -> Dict[str, float]:
    """Compute BO metrics from experiment data.

    This function provides a convenient interface to compute all Bayesian Optimization
    metrics from the final experiment data. See BOMetricsCalculator class documentation
    for detailed mathematical descriptions of all metrics.

    Args:
        sim_env: Simulation environment with dim, minimize, and objective_node_name attributes
        device: PyTorch device
        dtype: PyTorch data type
        data_dict: Final data dictionary with all explored points
        best_observed_history: List of best observed values per iteration
        model: Final trained model
        acqf: Final acquisition function

    """
    calculator = BOMetricsCalculator(sim_env, device, dtype)
    return calculator.compute_all_metrics(
        data_dict=data_dict,
        best_observed_history=best_observed_history,
        model=model,
        acqf=acqf,
    )
