import os
from experiments.bayesian_optimization.griewank.base_scripts.stgp_common import (
    run_stgp,
)


if __name__ == "__main__":
    run_stgp(
        exp_config_path=os.path.join(
            "experiments",
            "bayesian_optimization",
            "griewank",
            "configs",
            "exp_configs",
            "griewank_25D.yaml",
        )
    )
