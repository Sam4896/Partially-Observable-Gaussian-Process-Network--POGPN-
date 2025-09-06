from experiments.bayesian_optimization.griewank.base_scripts.gp_network_common import (
    run_gp_network,
)
import os

if __name__ == "__main__":
    run_gp_network(
        exp_config_path=os.path.join(
            "experiments",
            "bayesian_optimization",
            "griewank",
            "configs",
            "exp_configs",
            "griewank_25D.yaml",
        )
    )
