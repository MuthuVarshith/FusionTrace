"""Offline MLOps tooling for FusionTrace: evaluation metrics, detector selection,
ONNX export/quantisation helpers and optional MLflow tracking/registry.

Nothing here is imported by the web app. Every helper runs locally; MLflow is
optional and only imported when a ``--mlflow`` flag (or an explicit call) asks
for it. See ``docs/mlops.md`` for the command-line workflow.
"""
