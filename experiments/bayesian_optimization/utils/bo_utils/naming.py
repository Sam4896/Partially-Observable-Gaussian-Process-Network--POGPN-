from __future__ import annotations

from typing import Dict, Any


def _is_pogpn(model_config: Dict[str, Any]) -> bool:
    return model_config.get("model_type", {}).get("name", "").startswith("POGPN")


def _is_gp_network(model_config: Dict[str, Any]) -> bool:
    return model_config.get("model_type", {}).get("name", "").startswith("GPNetwork")


def _is_stgp(model_config: Dict[str, Any]) -> bool:
    return model_config.get("model_type", {}).get("name", "").startswith("STGP")


def _extract_pogpn_fields(
    model_config: Dict[str, Any],
) -> tuple[str, str, str, str, str, str, str]:
    model_type_name = model_config.get("model_type", {}).get("name", "")
    pogpn_type = (
        "pathwise"
        if "pathwise" in model_type_name.lower()
        else ("nodewise" if "nodewise" in model_type_name.lower() else "unknown")
    )

    params = model_config.get("model_type", {}).get("params", {})
    mll_beta = str(params.get("mll_beta", ""))
    mll_type = str(params.get("mll_type", ""))
    likelihood = str(params.get("likelihood", ""))
    inducing_point_ratio = str(params.get("inducing_point_ratio", ""))
    inducing_point_allocator = str(params.get("inducing_point_allocator", ""))

    if inducing_point_allocator == "GreedyImprovementReduction":
        inducing_point_allocator = "GIR"
    elif inducing_point_allocator == "GreedyVarianceReduction":
        inducing_point_allocator = "GVR"
    else:
        inducing_point_allocator = "Unknown"

    use_rbf_kernel = (
        model_config.get("model_type", {}).get("params", {}).get("use_rbf_kernel", None)
    )
    if use_rbf_kernel is None or use_rbf_kernel:
        kernel_suffix = "RBF"
    else:
        kernel_suffix = "Matern"

    return (
        pogpn_type,
        mll_beta,
        mll_type,
        likelihood,
        inducing_point_ratio,
        inducing_point_allocator,
        kernel_suffix,
    )


def create_gp_network_run_name_prefix(model_config: Dict[str, Any]) -> str:
    """Create a run name prefix for GP network experiments."""

    use_rbf_kernel = (
        model_config.get("model_type", {}).get("params", {}).get("use_rbf_kernel", None)
    )
    if use_rbf_kernel is None or use_rbf_kernel:
        kernel_suffix = "RBF"
    else:
        kernel_suffix = "Matern"

    return f"gp_network_{kernel_suffix}"


def create_stgp_run_name_prefix(model_config: Dict[str, Any]) -> str:
    """Create a run name prefix for STGP experiments."""
    use_rbf_kernel = (
        model_config.get("model_type", {}).get("params", {}).get("use_rbf_kernel", None)
    )
    if use_rbf_kernel is None or use_rbf_kernel:
        kernel_suffix = "RBF"
    else:
        kernel_suffix = "Matern"

    return f"stgp_{kernel_suffix}"


def create_pogpn_run_name_prefix(model_config: Dict[str, Any]) -> str:
    """Create a run name prefix for POGPN experiments."""
    (
        pogpn_type,
        mll_beta,
        mll_type,
        likelihood,
        inducing_point_ratio,
        inducing_point_allocator,
        kernel_suffix,
    ) = _extract_pogpn_fields(model_config)
    return f"POGPN_{pogpn_type}_{mll_beta}beta_{mll_type}_{inducing_point_ratio}IPratio_{inducing_point_allocator}_lik_{likelihood}_{kernel_suffix}"


def build_experiment_name(
    exp_config: Dict[str, Any],
    acqf_optim_q: int,
    acqf_suffix: str,
    acqf_params: Dict[str, Any],
) -> str:
    """Build an experiment name for POGPN experiments."""
    process_noise = exp_config["simulation"]["process_stochasticity_std"]
    obs_noise = exp_config["simulation"]["observation_noise_std"]
    return (
        f"{exp_config['simulation']['name']}_{exp_config['simulation']['dim']}D_"
        f"{acqf_suffix}_{acqf_optim_q}q_"
        f"{process_noise}process_{obs_noise}obs"
    )
