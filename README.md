# Partially Observable Gaussian Process Networks (POGPN)

This repository contains the implementation of [Partially Observable Gaussian Process Networks (POGPN)]((https://arxiv.org/abs/2502.13905)) using BoTorch, along with a comprehensive framework for running Bayesian Optimization (BO) experiments.

## Getting Started

### Prerequisites

This project uses [Poetry](https://python-poetry.org/) for dependency management. Please ensure you have Poetry installed on your system.

### Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone <repository_url>
    cd POGPN
    ```

2.  Install the project dependencies using Poetry:
    ```bash
    poetry install
    ```

This command will create a virtual environment and install all the necessary packages listed in `pyproject.toml`.

## Project Structure

The project is organized into two main directories:

-   `src/`: Contains the core implementation of the POGPN model.
-   `experiments/`: Contains scripts and configurations for running Bayesian Optimization experiments.

```
POGPN/
├── experiments/
│   └── bayesian_optimization/
│       ├── ackley/
│       ├── griewank/
│       │   ...
│       ├── utils/
│       └── README.md  <-- Guide for BO experiments
├── src/
│   └── pogpn_botorch/
│       └── README.md  <-- Guide for POGPN implementation
├── pyproject.toml     <-- Project dependencies
└── README.md          <-- You are here
```

## Running Bayesian Optimization Experiments

The `experiments/bayesian_optimization` directory is structured to facilitate running BO experiments on various synthetic test functions (e.g., Ackley, Rosenbrock) with different models (POGPN, STGP, etc.).

For detailed instructions on how to run experiments, configure them, and understand the utility framework, please refer to the dedicated README file:

**[➡️ experiments/bayesian_optimization/README.md](./experiments/bayesian_optimization/README.md)**

## POGPN Implementation Details

The core logic for the POGPN model is located in the `src/pogpn_botorch` directory. This includes the implementation of the DAG structure, the base POGPN class, nodewise and pathwise training strategies, and the custom posterior for inference.

For a deep dive into the architecture and functionality of each module, please see the detailed README:

**[➡️ src/pogpn_botorch/README.md](./src/pogpn_botorch/README.md)**

## Experiment Tracking with MLflow

This project is integrated with MLflow for tracking experiment runs, parameters, metrics, and artifacts. The framework supports logging to a local MLflow server or a remote server hosted on GitLab.

For instructions on how to set up your environment to log experiments to a GitLab-hosted MLflow instance, please refer to:

**[➡️ GITLAB_MLFLOW_SETUP.md](./GITLAB_MLFLOW_SETUP.md)**

## Citation

For a detailed understanding of the mathematical foundations of the Partially Observable Gaussian Process Network (POGPN), please refer to the following paper:

Kiroriwal, S., Pfrommer, J., & Beyerer, J. (2025). *Partially Observable Gaussian Process Network and Doubly Stochastic Variational Inference*. arXiv preprint arXiv:2502.13905.
[https://arxiv.org/abs/2502.13905](https://arxiv.org/abs/2502.13905)
