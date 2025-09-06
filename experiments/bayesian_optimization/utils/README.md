# 📚 Bayesian Optimization Utilities – Comprehensive Guide

**Location:** `experiments/bayesian_optimization/utils/`

This document is meant to be self-contained. It explains following contents.
1. Set up GitLab or local MLflow tracking.
2. Understand every major class / function in the utilities code.
3. Configure and run a Bayesian-optimization (BO) experiment.
4. Inspect logs, metrics, and registered surrogate-model versions.
5. Extend the framework with new models or simulators.

> All file-paths in examples are given **relative to the project root**.

---
## Table of Contents
1. [Directory Overview](#dir)
2. [Installation & Environment](#env)
3. [Tracking Back-Ends](#tracking)
   * 3.1 [GitLab MLflow](#gitlab)
   * 3.2 [Local SQLite MLflow](#local)
4. [Configuration Files](#config)
   * 4.1 [Experiment YAML (simulation+acquisition)](#exp-yaml)
   * 4.2 [Model YAML (surrogate settings)](#model-yaml)
5. [Class Map](#classes)
   * 5.1 [`MLflowExperimentManager`](#manager)
   * 5.2 [`BaseBOExperiment`](#basebo)
6. [Execution Flow – Step by Step](#flow)
7. [Logged Artifacts & Timeline](#artifacts)
8. [Model Versioning & Registry Logic](#registry)
9. [BO Metrics – List & Formulae](#metrics)
10. [Querying, Loading & Reproducing](#query)
11. [Extending the Framework](#extend)

---
<a name="dir"></a>
## 1 · Directory Overview

```
experiments/
└─ bayesian_optimization/
   ├─ utils/
   │  ├─ base_bo_experiment.py      ← 🔑 main abstract BO class
   │  ├─ bo_metrics.py              ← optional, heavy-weight metrics
   │  ├─ __init__.py
   │  └─ README.md                  ← (this file)
   ├─ ackley/                       ← example problem (6-D Ackley)
   │  ├─ stgp_bo.py, svgp_network_bo.py …
   │  └─ configs/                   ← YAML configs used by the scripts
   └─ utils/ (legacy helpers)
```

Outside this directory you will often touch:
* `experiments/utils/mlflow_manager.py` – central MLflow helper (imported by `base_bo_experiment`).
* `experiments/utils/device_utils.py` – chooses CPU / GPU.
* `experiments/bayesian_optimization/ackley/*.py` – concrete subclasses of `BaseBOExperiment` for each model (GP, SVGP, …).

---
<a name="env"></a>
## 2 · Installation & Environment

```bash
# create conda env (example)
conda create -n bo python=3.10
conda activate bo

# project dependencies
pip install -r requirements.txt   # contains botorch, gpytorch, mlflow …

# optional: PyTorch w/ GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> 🔍 **Tip:** GPU is only used for surrogate-model training; small simulations run fine on CPU.

---
<a name="tracking"></a>
## 3 · Tracking Back-Ends

<a name="gitlab"></a>
### 3.1 GitLab MLflow (recommended for team work)

1. Create `gitlab_mlflow.yaml` **at repo root**:

```yaml
project_id: 123456                     # GitLab project numeric ID
access_token: "glpat-xxxxxxxxxxxxxxxx" # Personal access token (API scope)
gitlab_endpoint: "https://gitlab.com"  # GitLab URL or your self-hosted instance
```

2. Set `log_mlflow_to_gitlab=True` when instantiating your experiment **or** in the driver script.
   * `BaseBOExperiment` reads the YAML and sets two env-vars:
     * `MLFLOW_TRACKING_URI`
     * `MLFLOW_TRACKING_TOKEN`

3. Runs, metrics & models now appear under *Project → Model Registry* in GitLab.

<a name="local"></a>
### 3.2 Local SQLite MLflow (offline work)

*Leave `log_mlflow_to_gitlab=False`* (default) **or** delete/rename `gitlab_mlflow.yaml`.

During `setup_mlflow_tracking()`:
1. `mlflow.db` (SQLite) and `mlruns/` folder are created under the current simulation’s `results/` path.
2. Start a local UI:
   ```bash
   python experiments/bayesian_optimization/ackley/start_mlflow_local_server.py
   # visits http://127.0.0.1:5000
   ```

> All code is agnostic – switching from local → GitLab needs zero modifications besides the YAML described above.

---
<a name="config"></a>
## 4 · Configuration Files

<a name="exp-yaml"></a>
### 4.1 Experiment YAML (`exp_config.yaml`)
Defines **problem & BO settings**.

Important keys (excerpt):
| key | type | description |
|-----|------|-------------|
| `simulation` | dict | Name + params of synthetic/real env. See `experiments/simulation_environments_pano` |
| `acqf_params.acquisition_function_name` | str | e.g. `qExpectedImprovement` |
| `acqf_params.num_trials` | int | BO repetitions for Monte-Carlo stats |
| `acqf_optim_params` | dict | optimiser hyper-params for `optimize_acqf` |

Full schema lives in comments at top of each example YAML.

<a name="model-yaml"></a>
### 4.2 Model YAML (`model_config.yaml`)
Controls **surrogate architecture** **and** registry metadata.

```yaml
model_type:
  name: STGP              # will become STGP_qEI etc.
...
registry:
  description: "Sparse TP surrogate on Ackley-6D (qEI)"
  tags:
    sim_env:      "ackley_6d"
    acq_function: "qEI"
    trials:       "20"
    bo_budget:    "150"
```

* `description` is shown in GitLab UI.
* `tags` are attached to **both** the run and the model-version (searchable).

---
<a name="classes"></a>
## 5 · Class Map

<a name="manager"></a>
### 5.1 `MLflowExperimentManager`
Centralised wrapper around `mlflow.*` API.

| Method | Purpose |
|--------|---------|
| `start_run` | opens an MLflow run (handles experiment creation) |
| `log_configs / log_params / log_metrics / log_artifacts / log_dataframe / log_plot` | thin wrappers, but hide CSV/JSON differences between remote (GitLab) and local back-ends |
| `create_registered_model_if_not_exists` | idempotent registry bootstrap |
| `register_model_version` | creates a **new version** and tags the run with that version ID |
| `set_run_tags`, `tag_model_version` | convenience helpers so experiment code stays short |

<a name="basebo"></a>
### 5.2 `BaseBOExperiment`
Abstract class – concrete subclasses implement only two hooks:
* `setup_model(data_dict)` – build & train surrogate.
* `setup_acquisition_function(model, best_value)` – return any BoTorch acqf.

Everything else (Sobol init, optimisation loop, logging, plotting, metrics, versioning) lives in the base class.

Key private helpers:
| helper | job |
|---------|-----|
| `_setup_mlflow_tracking` | chooses run name = `model_name`, starts manager |
| `_transform_data_dict` | normalises inputs, applies outcome transforms |
| `_save_data_dict` | stores `.npz` snapshot each iteration (artifact) |
| `_calculate_bo_metrics_for_trial` | calls external `bo_metrics.py` (if importable) |
| `_update_best_trial` | keeps the best surrogate + data for registration |

> The class also uses a **dataclass** `AcqfOptimParams` to enforce type-safe access to optimiser settings.

---
<a name="flow"></a>
## 6 · Execution Flow – Step by Step

```text
┌──────────────┐ 1. Driver script (e.g. stgp_bo.py) instantiates subclass
│  User code   │    with exp & model YAML paths.
└──────┬───────┘
       │
       │ 2. BaseBOExperiment.__init__
       ├─ loads YAMLs, prepares device, sim env
       ├─ sets up MLflow (local or GitLab)
       └─ writes params & configs to run
       │
       │ 3. run_experiment()
       ├─ for trial in 1..num_trials
       │   ├─ Sobol init → evaluate sim_env
       │   ├─ iterate 1..bo_budget
       │   │   ├─ train surrogate
       │   │   ├─ optimise acquisition
       │   │   ├─ evaluate new point
       │   │   └─ log per-iter info (file only)
       │   └─ compute optional per-trial metrics
       ├─ store trajectory DataFrames & plots
       ├─ find best trial surrogate → log Pytorch model
       ├─ register new model version + tags
       └─ end MLflow run
```

---
<a name="artifacts"></a>
## 7 · Logged Artifacts & Timeline

| Time | Artifact / Metric | Location (MLflow UI) |
|------|-------------------|-----------------------|
| run start | YAML configs | `artifacts/configs/` |
| every iteration | script log lines | `artifacts/logs/logs.log` (remote) or live in MLflow (local) |
| key iters | `model_summaries/*.txt` | human readable summary |
| per-trial end | `results/best_observed_values.json`, etc. |
| run end | `plots/optimization_performance.png` |
| run end | `data_points_explored/*.npz` |
| run end | `best_trial_data/best_trial_data.npz` + **registered model** |

---
<a name="registry"></a>
## 8 · Model Versioning & Registry Logic

1. **Registered-Model** container created once per `model_name` (e.g. `STGP_qEI`).
2. After BO completes, the surrogate from the best trial is logged via `mlflow.pytorch.log_model()`.
3. That URI (`runs:/<run_id>/model`) is passed to `mlflow.register_model()` → **version N**.
4. Run & version are both tagged using the `registry.tags` block from YAML.

Retrieving older versions is guaranteed because GitLab links model-version → run ID → artifacts.

---
<a name="metrics"></a>
## 9 · BO Metrics – Definitions

> Implemented in `bo_metrics.py` (import optional – experiment proceeds even if file missing).

| Metric | Formula / Description |
|--------|-----------------------|
| `best_found_value` | min objective across explored points |
| `total_improvement` | `initial_value - best_found_value` |
| `improvement_rate` | total_improvement / `bo_budget` |
| `recent_improvement_rate` | slope over last 10 iters |
| `exponential_improvement_rate` | EWMA with α = 0.1 |
| `iteration_of_best` | iteration index of global best |
| `model_normalized_rmse` | RMSE(model, sim_env) / (max - min) |
| `exploration_efficiency` | unique points / total points |
| `coverage_ratio` | convex-hull volume / design-space vol |
| `dispersion` | mean pairwise distance between points |
| `acquisition_diversity` | Shannon entropy of acqf-selected points |
| `acquisition_entropy` | entropy of acqf values |
| `mean_acquisition_value` | mean of acqf(x) over grid |
| `max_acquisition_value` | max of acqf(x) over grid |

Aggregated at run-level as **mean** and **std** across trials.

---
<a name="query"></a>
## 10 · Querying, Loading & Reproducing

### 10.1 Search runs
```python
from mlflow import MlflowClient
c = MlflowClient()
runs = c.search_runs(["Ackley_qEI"], "tags.sim_env='ackley_6d' and metrics.best_found_value_mean < 0.5")
```

### 10.2 Load a registered surrogate
```python
mv = c.get_model_version("STGP_qEI", "4")   # version 4
surrogate = mlflow.pytorch.load_model(mv.source)
```

### 10.3 Download artifacts programmatically
```python
path = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="best_trial_data/best_trial_data.npz")
```

### 10.4 CLI helpers
* `mlflow ui` – local UI.
* GitLab – **Model Registry** & **Experiments** tabs.

---
<a name="extend"></a>
## 11 · Extending the Framework

1. **Add a new surrogate model**
   * Create `my_model_bo.py` under the problem directory.
   * Subclass `BaseBOExperiment` and implement `setup_model` and `setup_acquisition_function`.
   * Provide a YAML in `configs/` with `model_type.name="MyModel"`.
2. **Add a new synthetic function / simulator**
   * Implement a `SimulationEnvironment` under `src/simulation_environments_pano` (see existing examples).
   * Update its config under `simulation` key of `exp_config.yaml`.
3. **Custom BO metrics**
   * Add functions to `bo_metrics.py`; include them in `compute_bo_metrics()` and they will auto-appear in logs.

---
**Happy Optimising & Learning!** 