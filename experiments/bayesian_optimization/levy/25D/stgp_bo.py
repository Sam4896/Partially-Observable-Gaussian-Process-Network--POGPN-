import os
from experiments.bayesian_optimization.levy.base_scripts.stgp_common import (
    run_stgp,
)


if __name__ == "__main__":
    run_stgp(
        exp_config_path=os.path.join(
            "experiments",
            "bayesian_optimization",
            "levy",
            "configs",
            "exp_configs",
            "levy_25D.yaml",
        )
    )
