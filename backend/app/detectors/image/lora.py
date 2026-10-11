"""Optional LoRA adapter for the Community Forensics image detector.

Off by default. Enable in ``config.yaml`` under ``detectors.image.options.lora``::

    lora:
      enabled: true
      source: mlflow            # mlflow: resolve the alias below | path: a local run folder
      model_name: fusiontrace-image-lora
      alias: candidate          # never "production" unless you have reviewed the run
      path: null                # when source is "path": the run folder (adapter/ + threshold.json)

With an adapter, the LoRA weights are merged into the ViT at load time, so inference and GradCAM
run exactly as for the base model. The decision threshold comes from the adapter's run (chosen on
its validation split), unless ``detectors.image.options.threshold`` is set explicitly.

The adapter is loaded from MLflow only by alias, so the app never follows an unreviewed version.
Every loaded adapter is reported in the scan's details (name, version, data version, threshold).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..base import ModelUnavailableError

logger = logging.getLogger("fusiontrace")

DEFAULT_MODEL_NAME = "fusiontrace-image-lora"


@dataclass(frozen=True)
class LoraSettings:
    enabled: bool
    source: str
    model_name: str
    alias: str
    path: str | None

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> LoraSettings:
        raw = dict(options.get("lora") or {})
        source = str(raw.get("source", "mlflow"))
        if source not in ("mlflow", "path"):
            raise ModelUnavailableError(f"lora.source must be 'mlflow' or 'path', got {source!r}")
        return cls(
            enabled=bool(raw.get("enabled", False)),
            source=source,
            model_name=str(raw.get("model_name", DEFAULT_MODEL_NAME)),
            alias=str(raw.get("alias", "candidate")),
            path=raw.get("path"),
        )


@dataclass(frozen=True)
class ResolvedAdapter:
    adapter_dir: Path
    threshold: float
    provenance: dict[str, Any]


def resolve(settings: LoraSettings, cache_dir: Path) -> ResolvedAdapter:
    if settings.source == "path":
        if not settings.path:
            raise ModelUnavailableError("lora.source is 'path' but lora.path is not set")
        return _from_run_folder(Path(settings.path))
    return _from_mlflow(settings, cache_dir)


def _from_run_folder(run: Path) -> ResolvedAdapter:
    adapter_dir = run / "adapter"
    threshold_file = run / "threshold.json"
    if not (adapter_dir / "adapter_config.json").is_file() or not threshold_file.is_file():
        raise ModelUnavailableError(f"{run} is not a fine-tune run folder (needs adapter/ and threshold.json)")
    stored = json.loads(threshold_file.read_text(encoding="utf-8"))
    return ResolvedAdapter(
        adapter_dir=adapter_dir,
        threshold=float(stored["threshold"]),
        provenance={"source": "path", "run": str(run), "data_version": stored.get("data_version"), "threshold": float(stored["threshold"])},
    )


def _from_mlflow(settings: LoraSettings, cache_dir: Path) -> ResolvedAdapter:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise ModelUnavailableError("LoRA from MLflow needs the 'finetune' extra (mlflow-skinny)") from exc
    try:
        version = MlflowClient().get_model_version_by_alias(settings.model_name, settings.alias)
    except Exception as exc:  # no such model or alias; the exception type varies by store
        raise ModelUnavailableError(f"no MLflow version of {settings.model_name!r} has alias {settings.alias!r}") from exc
    tags = version.tags or {}
    if "threshold" not in tags:
        raise ModelUnavailableError(f"{settings.model_name} v{version.version} has no threshold tag; refusing to guess one")
    target = cache_dir / f"{settings.model_name}-v{version.version}"
    if not (target / "adapter_config.json").is_file():
        target.mkdir(parents=True, exist_ok=True)
        local = mlflow.artifacts.download_artifacts(artifact_uri=version.source, dst_path=str(target))
        logger.info("Downloaded LoRA adapter %s v%s to %s", settings.model_name, version.version, local)
    adapter_dir = target if (target / "adapter_config.json").is_file() else target / "adapter"
    return ResolvedAdapter(
        adapter_dir=adapter_dir,
        threshold=float(tags["threshold"]),
        provenance={
            "source": "mlflow", "model": settings.model_name, "alias": settings.alias, "version": str(version.version),
            "data_version": tags.get("data_version"), "threshold": float(tags["threshold"]),
        },
    )


def merged_module(base_module, adapter_dir: Path):
    """Base timm ViT with the LoRA weights and trained head merged in (plain module again)."""
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(base_module, str(adapter_dir))
    return peft_model.merge_and_unload().eval()
