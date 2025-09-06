# BO Utils

Modular utilities to keep `BaseBOExperiment` thin and focused on orchestration.

## Structure
- naming.py: Experiment/run/folder naming helpers (POGPN-aware)
- tags.py: MLflow run tags and description builders
- metrics.py: Trajectory creation and aggregated metrics
- plotting.py: Plot results
- transforms.py: Node transforms and data normalization
- acqf_utils.py: Acquisition function optimization with bounds
- data_logging.py: Save artifacts, scripts, logs to MLflow
- mlflow_utils.py: MLflow env and logging directory helpers
- types.py: Protocols for interfaces
- factory.py: Factories returning concrete implementations

## Extension points
- Implement your own module conforming to `Namer`, `TagsBuilder`, or `MetricsComputer` and
  swap it in the factory to change behavior without touching orchestration.

## Testing
- Each module is small and unit-testable. See tests under `tests/bo_utils/`.
