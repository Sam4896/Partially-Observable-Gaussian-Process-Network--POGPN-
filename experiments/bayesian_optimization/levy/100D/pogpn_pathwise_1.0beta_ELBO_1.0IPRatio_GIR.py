from experiments.bayesian_optimization.levy.base_scripts.pogpn_common import (
    run_pogpn_pathwise_elbo_gir,
)
import os

if __name__ == "__main__":
    run_pogpn_pathwise_elbo_gir(
        exp_config_path=os.path.join(
            "experiments",
            "bayesian_optimization",
            "levy",
            "configs",
            "exp_configs",
            "levy_100D.yaml",
        )
    )
