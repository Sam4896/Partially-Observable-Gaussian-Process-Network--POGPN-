import os
from experiments.bayesian_optimization.rosenbrock.base_scripts.stgp_common import (
    run_stgp,
)


if __name__ == "__main__":
    run_stgp(
        exp_config_path=os.path.join(
            "experiments",
            "bayesian_optimization",
            "rosenbrock",
            "configs",
            "exp_configs",
            "rosenbrock_100D.yaml",
        )
    )
