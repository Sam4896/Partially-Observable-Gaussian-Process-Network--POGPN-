from experiments.bayesian_optimization.rosenbrock.base_scripts.pogpn_common import (
    run_pogpn_pathwise_pll_gir,
)
import os

if __name__ == "__main__":
    run_pogpn_pathwise_pll_gir(
        exp_config_path=os.path.join(
            "experiments",
            "bayesian_optimization",
            "rosenbrock",
            "configs",
            "exp_configs",
            "rosenbrock_25D.yaml",
        )
    )
