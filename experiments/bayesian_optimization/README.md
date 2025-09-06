# Bayesian Optimization Experiments

This directory contains the framework and scripts for running Bayesian Optimization (BO) experiments on various synthetic test functions.

## Directory Structure

The experiments are organized by simulation environment (synthetic function), and then by dimensionality.

```
experiments/bayesian_optimization/
├── <simulation_name>/
│   ├── <dimensionality>D/
│   │   ├── <model_script_1>.py
│   │   ├── <model_script_2>.py
│   │   └── run_all_<simulation_name><dimensionality>D.py
│   ├── base_scripts/
│   │   ├── <model_1>_common.py
│   │   └── <model_2>_common.py
│   └── configs/
│       ├── exp_configs/
│       │   └── <simulation_name>_<dimensionality>D.yaml
│       └── model_configs/
│           ├── <model_config_1>.yaml
│           └── <model_config_2>.yaml
└── utils/
    ├── base_bo_experiment.py
    ├── bo_utils/
    └── ... (other utility scripts)
```

-   **`<simulation_name>`**: The name of the synthetic test function (e.g., `ackley`, `rosenbrock`).
-   **`<dimensionality>D`**: The input dimension of the test function (e.g., `25D`, `100D`).
-   **`<model_script>.py`**: A script to run a BO experiment with a specific model.
-   **`run_all_... .py`**: A script to execute all experiment scripts within that directory.
-   **`base_scripts/`**: Contains common logic for different models, shared across dimensionalities.
-   **`configs/`**: Contains configuration files.
    -   `exp_configs/`: Experiment configurations (e.g., BO budget, number of trials).
    -   `model_configs/`: Model-specific configurations.
-   **`utils/`**: Core utilities for the BO experiment framework.

## Running Experiments

You can run experiments in two ways:

### 1. Running a Single Experiment

To run a single experiment, navigate to the project's root directory and execute the desired model script. For example, to run the STGP model on 25D Ackley function:

```bash
python experiments/bayesian_optimization/ackley/25D/stgp_bo.py
```

### 2. Running All Experiments for a Simulation Environment

Each dimensionality directory contains a `run_all_... .py` script that executes all the model experiments for that setting. For example, to run all experiments for 100D Ackley:

```bash
python experiments/bayesian_optimization/ackley/100D/run_all_ackley100D.py
```

## Configuration

Experiments are configured through YAML files.

-   **Experiment Configuration (`exp_configs/*.yaml`)**: These files define parameters for the BO loop, such as:
    -   `simulation`: name, dimension, noise levels.
    -   `acqf_optim_params`: Parameters for optimizing the acquisition function (e.g., `num_restarts`, `raw_samples`).
    -   `acqf_params`: Parameters for the acquisition function and BO loop (e.g., `bo_budget`, `num_trials`).

-   **Model Configuration (`model_configs/*.yaml`)**: These files define the surrogate model's architecture and training parameters.
    -   `model_type`: name and parameters of the model.
    -   `mll_optimizer_kwargs`: Arguments for the marginal log-likelihood optimizer.

## Experiment Tracking with MLflow

This framework is tightly integrated with [MLflow](https://mlflow.org/) for experiment tracking. It automatically logs parameters, metrics, and artifacts for each experiment run.

### How Logging Works

The MLflow setup is handled within the `BaseBOExperiment` class. The tracking URI is determined with the following priority:

1.  **GitLab Integration**: If `log_mlflow_to_gitlab=True` is passed to the `BaseBOExperiment` constructor, the framework will use the configuration from `gitlab_mlflow.yaml` to connect to a GitLab-hosted MLflow server. See `GITLAB_MLFLOW_SETUP.md` for more details.
2.  **Explicit URI**: A tracking URI can be passed directly to the constructor via the `mlflow_tracking_uri` argument.
3.  **Environment Variable**: The framework will check for the `MLFLOW_TRACKING_URI` environment variable.
4.  **Fallback to Local**: If none of the above are set, it defaults to a local tracking setup. It will create an `mlruns` directory in the root of the project to store the experiment data.

### What is Logged

For each run, the following are automatically logged:
-   **Parameters**: All parameters from the experiment and model YAML configuration files.
-   **Metrics**: Aggregated metrics from all trials, such as the mean and standard deviation of the best-found value.
-   **Artifacts**:
    -   A performance plot (`optimization_performance.png`) showing the best-observed value vs. iteration.
    -   The exact script that was executed for the run.
    -   The complete log file for the run.
    -   DataFrames (as CSVs) containing the trial-by-trial results and aggregated statistics.
    -   The final trained model from the best trial.

## Experiment Utilities (`utils/`)

The `utils/` directory provides a powerful and extensible framework for running BO experiments.

### `base_bo_experiment.py`: The Core Orchestrator

This is the heart of the framework. The `BaseBOExperiment` class manages the entire lifecycle of a BO experiment. Its key responsibilities include:
-   Loading and parsing experiment and model configuration files.
-   Setting up the MLflow tracking and creating a new experiment run.
-   Instantiating the simulation environment.
-   Calling the `TrialRunner` to execute multiple independent trials of the BO algorithm.
-   Aggregating the results from all trials.
-   Logging all parameters, aggregated metrics, and artifacts to MLflow.

The class requires subclasses to implement two key methods:
-   `setup_model(...)`: Defines how the surrogate model (e.g., POGPN, STGP) is initialized.
-   `setup_acquisition_function(...)`: Defines how the acquisition function is initialized.

### `bo_utils/` Sub-package: Specialized Tools

This package contains modules with specific responsibilities that support the main experiment loop.

-   **`trial_runner.py`**: The `TrialRunner` class is responsible for executing a single, full BO trial from start to finish. It handles the initial data generation and the main optimization loop for a given number of iterations (`bo_budget`). At each step, it calls the user-defined `setup_model_fn` and `setup_acqf_fn` callbacks and collects the results.

-   **`naming.py` & `tags.py`**: These modules are essential for keeping MLflow experiments organized. They automatically generate consistent and descriptive names for MLflow experiments and runs, as well as detailed run descriptions, based on the contents of the configuration files. This eliminates the need for manual naming and makes it easy to search and compare runs on the MLflow UI.

-   **`metrics_agg.py`**: The `MetricsAggregator` class takes the raw results (e.g., history of best-found values) from all independent trials, computes statistics (mean, standard deviation, min, max), and creates a summary DataFrame. These aggregated statistics are what get logged as metrics to MLflow.

-   **`plotting.py`**: This module contains the `plot_results` function, which takes the aggregated statistics from `MetricsAggregator` and generates the final performance plot that is saved as an artifact to MLflow.

-   **`data_logging.py`**: This module provides helper functions to log key artifacts to MLflow. `save_script` logs the source code of the running experiment script, and `save_data_dict` logs the final set of explored data points.

-   **`transforms.py`**: This module handles data preprocessing. It sets up and applies BoTorch `OutcomeTransform`s (like `Standardize`) to the outputs of the simulation environment nodes before they are passed to the model.

-   **`mlflow_utils.py`**: Contains helper functions specifically for setting up the environment for logging to a GitLab-hosted MLflow server.

-   **`bo_metrics.py`**: The `BOMetricsCalculator` computes various performance metrics for a single BO run, such as the best value found and acquisition function entropy.

-   **`sim_env_utils.py`**: A set of helper functions for interacting with the synthetic test function environments, such as generating initial data and evaluating the function.

## Adding a New Experiment

To add a new experiment (e.g., for a new synthetic function or a new model):

1.  **Create a new simulation environment class** inheriting from `DAGSyntheticTestFunction`.
2.  **Create a new directory structure** for the new simulation environment under `experiments/bayesian_optimization/`.
3.  **Add experiment and model configuration files** in the corresponding `configs` directory.
4.  **Create a `_common.py` script** in `base_scripts/` that implements the `setup_model` and `setup_acquisition_function` methods from `BaseBOExperiment`.
5.  **Create experiment scripts** in the dimensionality-specific directories that call the functions from the common script.
6.  (Optional) Add a `run_all_... .py` script.
