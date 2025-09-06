# GitLab MLflow Setup Guide

## Prerequisites

1. **GitLab Premium** (required for MLflow feature)
2. **GitLab Access Token** with `api` scope
3. **Python packages**: `mlflow`, `pyyaml`

## Step 1: Enable MLflow in GitLab

1. Go to your GitLab project
2. Navigate to **Settings > General**
3. Note your **Project ID** (you'll need this)

## Step 2: Create GitLab Access Token

1. Go to **User Settings > Access Tokens**
2. Create a new token with `api` scope
3. Copy the token (you won't see it again)

## Step 3: Create GitLab Configuration File

1. Create a file named `gitlab_mlflow.yaml` in your project root
2. Add the following content (replace with your values):

```yaml
# GitLab MLflow Configuration
# This file is gitignored for security - each user sets their own credentials

gitlab_endpoint: "https://gitlab.cc-asp.fraunhofer.de"
project_id: "YOUR_PROJECT_ID"  # Replace with your project ID
access_token: "YOUR_ACCESS_TOKEN"  # Replace with your token
```

**Note:** This file is automatically gitignored for security. Each team member creates their own copy.

## Step 4: Test the Setup

```bash
# Test the GitLab MLflow integration
python test_gitlab_integration.py
```

## Step 5: Use in Your Experiments

Your existing experiments will automatically work with GitLab MLflow. Simply add the `log_mlflow_to_gitlab=True` parameter:

```python
# In your experiment script
from experiments.bayesian_optimization.utils.base_bo_experiment import BaseBOExperiment

class YourExperiment(BaseBOExperiment):
    def __init__(self, exp_config_path, model_config_path):
        super().__init__(
            exp_config_path=exp_config_path,
            model_config_path=model_config_path,
            log_mlflow_to_gitlab=True  # Enable GitLab MLflow logging
        )
```

### Key Benefits

- **No local `.db` files** - everything stored in GitLab
- **Automatic artifact cleanup** - deleting runs/experiments removes artifacts
- **Team collaboration** - all team members see the same experiments
- **Secure credentials** - each user manages their own `gitlab_mlflow.yaml`
- **Simple setup** - just one boolean parameter

### Experiment Naming Convention

Experiments are automatically named using the pattern: `{sim_env_name}_{acqf_name}`

Examples:
- `ackley_qExpectedImprovement`
- `hartmann_qLogExpectedImprovement`
- `frigola_qNoisyExpectedImprovement`

## Usage Examples

### Running Experiments with GitLab MLflow

```python
# Your existing code works unchanged
from experiments.utils.mlflow_utils import setup_mlflow_tracking

# For direct MLflow usage
manager = setup_mlflow_tracking(
    experiment_name="ackley_qEI",
    model_name="GP_Network",
    # No tracking_uri needed - uses environment variable automatically
)

# For experiment classes
experiment = YourExperimentClass(
    exp_config_path="config.yaml",
    model_config_path="model_config.yaml",
    log_mlflow_to_gitlab=True  # Enable GitLab logging
)
```

### What Happens Automatically

When `log_mlflow_to_gitlab=True`:

1. **Environment variables set** from `gitlab_mlflow.yaml`
2. **No `mlflow.set_tracking_uri()` called** (GitLab best practice)
3. **Experiment created** in GitLab MLflow
4. **Runs logged** to GitLab with automatic artifact cleanup

## Security Benefits

✅ **No hardcoded tokens** - Credentials in gitignored file  
✅ **Per-user credentials** - Each team member uses their own config  
✅ **Token rotation** - Change tokens without code changes  
✅ **GitLab compliant** - Follows official GitLab documentation  
✅ **No Git conflicts** - MLflow data not in Git  

## Team Setup

### For Each Team Member

1. **Get project ID** from GitLab project page
2. **Create personal access token** with `api` scope
3. **Create `gitlab_mlflow.yaml`** with your credentials
4. **Test setup**: `python test_gitlab_integration.py`
5. **Use in experiments**: Set `log_mlflow_to_gitlab=True`

### Example `gitlab_mlflow.yaml`

```yaml
# gitlab_mlflow.yaml (create your own copy)
gitlab_endpoint: "https://gitlab.cc-asp.fraunhofer.de"
project_id: "68554"  # Your project ID
access_token: "glpat-your-token-here"  # Your access token
```

## Troubleshooting

### Configuration Issues
```bash
# Check if gitlab_mlflow.yaml exists
ls gitlab_mlflow.yaml

# Test the integration
python test_gitlab_integration.py
```

### Permission Issues
- Ensure your access token has `api` scope
- Check that MLflow is enabled in your GitLab project
- Verify your project ID is correct

### GitLab Connection Issues
```bash
# Test GitLab API directly
curl -H "Authorization: Bearer YOUR_TOKEN" \
     "https://gitlab.cc-asp.fraunhofer.de/api/v4/projects/YOUR_PROJECT_ID"
```

## Migration from Local MLflow

```bash
# Export local experiments (if needed)
mlflow export --experiment-name "your_experiment" --output-dir ./export

# Import to GitLab (if needed)
mlflow import --input-dir ./export --experiment-name "your_experiment"
```

## How It Works

1. **Environment Setup**: `log_mlflow_to_gitlab=True` reads `gitlab_mlflow.yaml` and sets environment variables
2. **GitLab Compliance**: No `mlflow.set_tracking_uri()` called when environment variables are set
3. **Automatic Logging**: All MLflow operations go directly to GitLab
4. **Cleanup**: GitLab handles artifact deletion automatically when runs/experiments are deleted 