from __future__ import annotations

from typing import Any, Dict


def build_run_description(
    model_config: Dict[str, Any],
    exp_config: Dict[str, Any],
    acqf_params: Dict[str, Any],
    acqf_optim_params: Any,
    sim_env_name: str,
    acqf_suffix: str,
) -> str:
    """Construct tags and description strings for MLflow runs.

    Returns:
        (run_tags, run_description, model_tags, model_description, sim_env_tags, sim_env_description)

    """
    # Model params
    lr = model_config.get("mll_optimizer_kwargs", {}).get("lr", "")
    optimizer = model_config.get("mll_optimizer_kwargs", {}).get("optimizer", "")
    params = model_config.get("model_type", {}).get("params", {})
    inducing_point_allocator = params.get("inducing_point_allocator", "")
    inducing_point_ratio = params.get("inducing_point_ratio", "")
    mll_beta = params.get("mll_beta", "")
    mll_type = params.get("mll_type", "")
    likelihood = params.get("likelihood", "")
    num_mc_samples = params.get("num_mc_samples", "")

    model_description = (
        f"Surrogate model: {model_config['model_type']['name']} for BO experiments on {sim_env_name} "
        f"with {acqf_suffix} acquisition function. Model parameters: "
        f"MLL type={mll_type}, beta={mll_beta}, inducing point ratio={inducing_point_ratio}, "
        f"allocator={inducing_point_allocator}, likelihood={likelihood}, MC samples={num_mc_samples}, "
        f"optimizer={optimizer}, learning rate={lr}."
    )
    sim_env = exp_config["simulation"]

    sim_env_description = (
        f"Simulation environment: {sim_env['name']} for BO experiments. "
        f"Dimensions: {sim_env['dim']}D. Process stochasticity std: {sim_env['process_stochasticity_std']}. "
        f"Observation noise std: {sim_env['observation_noise_std']}. "
        f"BO budget: {acqf_params['bo_budget']} iterations, {acqf_params['num_trials']} trials, q={acqf_optim_params.q}."
    )

    run_description = model_description + "\n" + sim_env_description

    return run_description
