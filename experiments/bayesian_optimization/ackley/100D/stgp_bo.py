import os
from experiments.bayesian_optimization.ackley.base_scripts.stgp_common import (
    run_stgp,
)


if __name__ == "__main__":
    run_stgp(
        exp_config_path=os.path.join(
            "experiments",
            "bayesian_optimization",
            "ackley",
            "configs",
            "exp_configs",
            "ackley_100D.yaml",
        )
    )
