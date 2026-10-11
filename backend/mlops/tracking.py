"""Optional MLflow tracking against a local store. Nothing here runs unless asked.

The tracking URI is ``MLFLOW_TRACKING_URI`` when set, else a file store at
``<repo>/.mlruns`` (``C:/FusionTrace/.mlruns`` in the main checkout). The
registry URI follows ``MLFLOW_REGISTRY_URI`` or the tracking URI. No remote or
hosted service is contacted unless you point those variables at one yourself.

MLflow 3.x keeps the file store in maintenance mode and refuses it unless
``MLFLOW_ALLOW_FILE_STORE=true``; this module sets that default for ``file:``
URIs. Use ``MLFLOW_TRACKING_URI=sqlite:///<path>/mlflow.db`` for the database
backend instead (needs the full ``mlflow`` package or SQLAlchemy + Alembic).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = REPO_ROOT / ".mlruns"
INSTALL_HINT = "pip install mlflow-skinny==3.17.0 (see requirements/extra-agent-e.txt), or run without --mlflow"
PARAM_LIMIT = 500  # conservative across MLflow versions and backends


class MlflowUnavailableError(RuntimeError):
    """MLflow is not installed or the store cannot be used."""


def tracking_uri(uri: str | None = None) -> str:
    raw = uri or os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_STORE.as_uri()
    if "://" not in raw and not raw.startswith("file:"):
        raw = Path(raw).resolve().as_uri()  # a plain directory path
    return raw


def _file_store_dir(uri: str) -> Path | None:
    if not uri.startswith("file:"):
        return None
    from urllib.parse import unquote, urlparse

    parsed = urlparse(uri)
    path = unquote(parsed.path)
    if os.name == "nt" and path.startswith("/") and len(path) > 2 and path[2] == ":":
        path = path[1:]  # file:///C:/x -> C:/x
    return Path(path)


def import_mlflow(uri: str | None = None) -> Any:
    """Import and configure mlflow for the resolved local store; raise a readable error if missing."""
    try:
        import mlflow
    except ImportError as exc:
        raise MlflowUnavailableError(f"MLflow is not installed: {INSTALL_HINT}") from exc
    resolved = tracking_uri(uri)
    store = _file_store_dir(resolved)
    if store is not None:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(resolved)
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI") or resolved)
    return mlflow


def clean_params(params: Mapping[str, Any]) -> dict[str, str]:
    """Stringify params (JSON for containers) and truncate to a size every backend accepts."""
    cleaned: dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            continue
        text = json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list, tuple)) else str(value)
        cleaned[key] = text if len(text) <= PARAM_LIMIT else text[: PARAM_LIMIT - 3] + "..."
    return cleaned


def clean_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Keep finite numbers only (MLflow rejects None; NaN would hide a missing value)."""
    cleaned: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            cleaned[key] = number
    return cleaned


@contextmanager
def run(experiment: str, run_name: str | None = None, tags: Mapping[str, Any] | None = None, uri: str | None = None) -> Iterator[tuple[Any, Any]]:
    """``with run("fusiontrace-evaluation") as (mlflow, active):`` -> logs into the local store."""
    mlflow = import_mlflow(uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags={key: str(value) for key, value in (tags or {}).items()}) as active:
        yield mlflow, active


def log_evaluation(
    section: Mapping[str, Any],
    *,
    dataset: Mapping[str, Any],
    report_path: Path | None,
    markdown_path: Path | None,
    experiment: str = "fusiontrace-evaluation",
    uri: str | None = None,
) -> str:
    """One run per evaluated detector: params, metrics, report JSON/markdown and confusion matrix."""
    detector = section["detector"]
    at_detector = section.get("at_detector_threshold") or {}
    params = {
        "detector.name": detector.get("name"),
        "detector.kind": detector.get("kind"),
        "detector.version": detector.get("version"),
        "detector.source": detector.get("source"),
        "detector.class": detector.get("class"),
        "detector.options": detector.get("options") or {},
        "threshold": (section.get("at_threshold") or {}).get("threshold"),
        "dataset.path": dataset.get("path"),
        "dataset.sha256": dataset.get("sha256"),
        "dataset.files": dataset.get("files"),
    }
    counts = section.get("counts") or {}
    metrics = {
        "roc_auc": section.get("roc_auc"),
        **{key: at_detector.get(key) for key in ("accuracy", "balanced_accuracy", "fake_precision", "fake_recall", "fake_f1", "real_recall")},
        **{f"count.{key}": value for key, value in counts.items()},
    }
    at_threshold = section.get("at_threshold") or {}
    metrics.update({f"at_threshold.{key}": at_threshold.get(key) for key in ("accuracy", "fake_precision", "fake_recall", "fake_f1", "real_recall")})
    tags = {"fusiontrace.tool": "evaluate_models", "fusiontrace.kind": detector.get("kind"), "fusiontrace.detector": detector.get("name")}
    with run(experiment, run_name=f"eval-{detector.get('kind')}-{detector.get('name')}", tags=tags, uri=uri) as (mlflow, active):
        mlflow.log_params(clean_params(params))
        mlflow.log_metrics(clean_metrics(metrics))
        mlflow.log_dict(at_detector.get("confusion_matrix") or {}, "confusion_matrix.json")
        if at_threshold:
            mlflow.log_dict(at_threshold.get("confusion_matrix") or {}, "confusion_matrix_at_threshold.json")
        mlflow.log_dict(dict(detector), "detector.json")
        for path in (report_path, markdown_path):
            if path is not None and path.exists():
                mlflow.log_artifact(str(path), artifact_path="report")
        return active.info.run_id


def log_benchmark(
    report: Mapping[str, Any], *, json_path: Path, markdown_path: Path, experiment: str = "fusiontrace-benchmarks", uri: str | None = None
) -> str:
    detector = report["detector"]
    params = {"detector.name": detector.get("name"), "detector.kind": detector.get("kind"), "detector.version": detector.get("version"),
              **{key: report.get(key) for key in ("runs", "warmup", "threads", "isolated")}, "platform": report["environment"].get("platform")}
    metrics: dict[str, Any] = {}
    for row in report["results"]:
        if not row.get("available"):
            continue
        variant = row["variant"]
        latency = row.get("latency_ms") or {}
        metrics.update({f"{variant}.p50_ms": latency.get("p50"), f"{variant}.p95_ms": latency.get("p95"),
                        f"{variant}.size_mb": row.get("size_mb"), f"{variant}.peak_rss_delta_mb": row.get("peak_rss_delta_mb")})
    tags = {"fusiontrace.tool": "benchmark_models", "fusiontrace.kind": detector.get("kind"), "fusiontrace.detector": detector.get("name")}
    with run(experiment, run_name=f"bench-{detector.get('name')}", tags=tags, uri=uri) as (mlflow, active):
        mlflow.log_params(clean_params(params))
        mlflow.log_metrics(clean_metrics(metrics))
        for path in (json_path, markdown_path):
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="benchmark")
        return active.info.run_id
