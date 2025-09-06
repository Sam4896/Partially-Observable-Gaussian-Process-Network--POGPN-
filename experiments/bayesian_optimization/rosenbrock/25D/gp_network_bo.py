from experiments.bayesian_optimization.rosenbrock.base_scripts.gp_network_common import (
    run_gp_network,
)
import os

if __name__ == "__main__":
    run_gp_network(
        exp_config_path=os.path.join(
            "experiments",
            "bayesian_optimization",
            "rosenbrock",
            "configs",
            "exp_configs",
            "rosenbrock_25D.yaml",
        )
    )
