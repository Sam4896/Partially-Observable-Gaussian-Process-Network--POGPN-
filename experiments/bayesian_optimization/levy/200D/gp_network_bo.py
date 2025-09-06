from experiments.bayesian_optimization.levy.base_scripts.gp_network_common import (
    run_gp_network,
)
import os

if __name__ == "__main__":
    run_gp_network(
        exp_config_path=os.path.join(
            "experiments",
            "bayesian_optimization",
            "levy",
            "configs",
            "exp_configs",
            "levy_200D.yaml",
        )
    )
