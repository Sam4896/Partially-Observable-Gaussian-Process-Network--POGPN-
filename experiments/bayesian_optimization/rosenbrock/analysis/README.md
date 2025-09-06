# Analysis Framework

This directory contains a refactored analysis framework that abstracts common functionality for analyzing different GP models on the Ackley function.

## Architecture

### Base Class: `BaseAnalysis`

The `BaseAnalysis` class handles all common functionality:

- **Setup**: Device/dtype configuration, matplotlib backend
- **Data Management**: Loading and preprocessing data from npz files
- **Grid Creation**: Setting up evaluation grids for surface plotting
- **Sampling**: MC sampling configuration
- **Basic Plotting**: Common plotting utilities with uncertainty bands

### Model-Specific Implementations

Each model type inherits from `BaseAnalysis` and implements:

1. `setup_model()`: Model-specific training logic
2. `get_posterior_predictions()`: Model-specific prediction logic  
3. `plot_model_specific()`: Model-specific visualization logic

#### `STGPAnalysis`
- Handles Single Task GP models
- Plots single output (y3) with uncertainty

#### `GPNetworkAnalysis`  
- Handles GP Network models with multi-output
- Plots all outputs (y1, y2, y3) and output space relationships

## Usage

### Simple Usage
```python
from stgp_analysis_refactored import STGPAnalysis

analysis = STGPAnalysis()
analysis.run_analysis(
    parent_dir="path/to/results",
    model_name="stgp",
    model_display_name="STGP"
)
```

### Custom Configuration
```python
analysis = STGPAnalysis(
    num_data_points=20,
    num_mc_samples=256,
    surface_n_points=50,
    seed=123
)
analysis.run_analysis(parent_dir, model_name)
```

### Step-by-Step Control
```python
analysis = STGPAnalysis()
analysis.setup_all(parent_dir, model_name)
model = analysis.setup_model()
mean, std = analysis.get_posterior_predictions()
# Custom analysis with predictions...
```

## Benefits

1. **Code Reuse**: Common functionality is shared across all models
2. **Easy Extension**: New models only need to implement 3 methods
3. **Consistent Interface**: All models use the same API
4. **Flexible Configuration**: Easy to customize parameters
5. **Simple Debugging**: Clear separation of concerns

## Adding New Models

To add a new model:

1. Create a new class inheriting from `BaseAnalysis`
2. Implement the 3 abstract methods:
   - `setup_model()`
   - `get_posterior_predictions()`  
   - `plot_model_specific()`
3. Optionally override other methods for custom behavior

Example:
```python
class MyModelAnalysis(BaseAnalysis):
    def setup_model(self):
        # Your model setup logic
        return trained_model
    
    def get_posterior_predictions(self):
        # Your prediction logic
        return mean, std
    
    def plot_model_specific(self, fig, axes):
        # Your plotting logic
        pass
```

## Files

- `base_analysis.py`: Base class with common functionality
- `stgp_analysis_refactored.py`: STGP-specific implementation
- `gp_network_analysis_refactored.py`: GP Network-specific implementation
- `run_analysis_example.py`: Usage examples
- Legacy files:
  - `stgp_analysis.py`: Original STGP analysis (for reference)
  - `gp_network_analysis.py`: Original GP Network analysis (for reference) 