"""Register exported ONNX detectors in the local MLflow model registry and manage aliases.

Flow: export (and optionally quantise) -> ``register`` with alias ``candidate`` ->
evaluate the candidate -> ``promote --from candidate --to production``.
Aliases are labels only: the web app does not read the registry; deploying a
version means copying its files (``list`` prints the artefact location).

    python -m mlops.model_registry register --model <onnx dir>/tiny-torch/model.onnx --alias candidate
    python -m mlops.model_registry list --name fusiontrace-image-tiny-torch
    python -m mlops.model_registry promote --name fusiontrace-image-tiny-torch --from candidate --to production
    python -m mlops.model_registry alias --name fusiontrace-image-tiny-torch --version 1 --alias production

Run from ``backend/``. Requires MLflow (see ``mlops.tracking``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import onnx_tools, tracking

EXPERIMENT = "fusiontrace-models"


@dataclass(frozen=True)
class RegisteredVersion:
    name: str
    version: str
    run_id: str
    source: str
    aliases: tuple[str, ...]


def default_name(metadata: Mapping[str, Any]) -> str:
    detector = metadata.get("detector") or {}
    return f"fusiontrace-{detector.get('kind', 'model')}-{detector.get('name', 'unknown')}"


def _tags(metadata: Mapping[str, Any], extra: Mapping[str, str] | None) -> dict[str, str]:
    detector = metadata.get("detector") or {}
    parity = metadata.get("parity") or {}
    tags = {
        "fusiontrace.detector": detector.get("name"),
        "fusiontrace.kind": detector.get("kind"),
        "fusiontrace.detector_version": detector.get("version") or "",
        "fusiontrace.source": detector.get("source"),
        "fusiontrace.precision": metadata.get("precision"),
        "fusiontrace.sha256": metadata.get("sha256"),
        "fusiontrace.opset": metadata.get("opset"),
        "fusiontrace.parity_passed": parity.get("passed"),
        "fusiontrace.file": metadata.get("file"),
    }
    tags.update(extra or {})
    return {key: str(value) for key, value in tags.items() if value is not None}


def _client(uri: str | None = None) -> Any:
    mlflow = tracking.import_mlflow(uri)
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=mlflow.get_tracking_uri(), registry_uri=mlflow.get_registry_uri())


def register_onnx_model(
    model_path: Path,
    *,
    name: str | None = None,
    aliases: Sequence[str] = (),
    tags: Mapping[str, str] | None = None,
    allow_failed_parity: bool = False,
    uri: str | None = None,
) -> RegisteredVersion:
    """Log the .onnx (+ metadata) as run artefacts and create a registered model version."""
    metadata = onnx_tools.read_metadata(model_path)
    if metadata is None:
        raise tracking.MlflowUnavailableError(f"{model_path} has no metadata JSON; export it with scripts/export_onnx.py")
    if metadata.get("sha256") and metadata["sha256"] != onnx_tools.sha256_file(model_path):
        raise tracking.MlflowUnavailableError(f"{model_path} does not match the sha256 in its metadata; re-export it")
    if not (metadata.get("parity") or {}).get("passed", False) and not allow_failed_parity:
        raise tracking.MlflowUnavailableError(f"{model_path.name} failed its parity check; pass --allow-failed-parity to register it anyway")
    model_name = name or default_name(metadata)
    version_tags = _tags(metadata, tags)
    with tracking.run(EXPERIMENT, run_name=f"register-{model_name}-{metadata.get('precision')}", tags={"fusiontrace.tool": "model_registry"}, uri=uri) as (mlflow, active):
        mlflow.log_params(tracking.clean_params({"model.name": model_name, "model.file": model_path.name, "opset": metadata.get("opset")}))
        mlflow.log_metrics(tracking.clean_metrics({"size_bytes": metadata.get("size_bytes"), **{
            f"parity.{key}": value for key, value in (metadata.get("parity") or {}).items() if key != "files"
        }}))
        for path in onnx_tools.model_files(model_path):
            mlflow.log_artifact(str(path), artifact_path="onnx")
        mlflow.log_artifact(str(onnx_tools.metadata_path(model_path)), artifact_path="onnx")
        run_id, artifact_uri = active.info.run_id, active.info.artifact_uri
    client = _client(uri)
    try:
        client.get_registered_model(model_name)
    except Exception:  # noqa: BLE001  # RESOURCE_DOES_NOT_EXIST; the exception type differs across stores
        detector = metadata.get("detector") or {}
        client.create_registered_model(model_name, tags={"fusiontrace.kind": str(detector.get("kind")), "fusiontrace.detector": str(detector.get("name"))},
                                       description=f"ONNX exports of the FusionTrace {detector.get('display_name', detector.get('name'))} detector.")
    source = f"{artifact_uri}/onnx"
    version = client.create_model_version(model_name, source=source, run_id=run_id, tags=version_tags)
    for alias in aliases:
        client.set_registered_model_alias(model_name, alias, version.version)
    return RegisteredVersion(model_name, str(version.version), run_id, source, tuple(aliases))


def set_alias(name: str, alias: str, version: str, uri: str | None = None) -> None:
    _client(uri).set_registered_model_alias(name, alias, str(version))


def promote(name: str, from_alias: str, to_alias: str, uri: str | None = None) -> str:
    """Point ``to_alias`` at the version ``from_alias`` currently names; return that version."""
    client = _client(uri)
    version = client.get_model_version_by_alias(name, from_alias).version
    client.set_registered_model_alias(name, to_alias, version)
    return str(version)


def list_versions(name: str | None = None, uri: str | None = None) -> list[dict[str, Any]]:
    client = _client(uri)
    rows: list[dict[str, Any]] = []
    models = [client.get_registered_model(name)] if name else client.search_registered_models()
    for model in models:
        aliases_by_version: dict[str, list[str]] = {}
        for alias, version in (model.aliases or {}).items():
            aliases_by_version.setdefault(str(version), []).append(alias)
        for version in client.search_model_versions(f"name='{model.name}'"):
            rows.append({"name": model.name, "version": str(version.version), "aliases": sorted(aliases_by_version.get(str(version.version), [])),
                         "source": version.source, "run_id": version.run_id, "tags": dict(version.tags or {})})
    return sorted(rows, key=lambda row: (row["name"], int(row["version"])))


def log_and_maybe_register(model_path: Path, *, register: str | None, aliases: Sequence[str], tool: str) -> None:
    """Hook for the export/quantise scripts' ``--mlflow`` / ``--register`` flags. Prints, never raises for MLflow problems."""
    try:
        if register is None:
            metadata = onnx_tools.read_metadata(model_path) or {}
            with tracking.run(EXPERIMENT, run_name=f"{tool}-{metadata.get('precision')}", tags={"fusiontrace.tool": tool}) as (mlflow, active):
                mlflow.log_params(tracking.clean_params({"file": model_path.name, "opset": metadata.get("opset"), "precision": metadata.get("precision")}))
                mlflow.log_artifact(str(onnx_tools.metadata_path(model_path)), artifact_path="onnx")
                print(f"mlflow: logged run {active.info.run_id} ({tracking.tracking_uri()})")
            return
        registered = register_onnx_model(model_path, name=register or None, aliases=aliases)
        print(f"mlflow: registered {registered.name} version {registered.version} aliases={list(registered.aliases)} ({tracking.tracking_uri()})")
    except tracking.MlflowUnavailableError as exc:
        print(f"mlflow: skipped: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mlops.model_registry", description="Local MLflow model registry for ONNX detectors")
    parser.add_argument("--tracking-uri", help="default: MLFLOW_TRACKING_URI or <repo>/.mlruns")
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register", help="register an exported .onnx (its metadata JSON must sit next to it)")
    reg.add_argument("--model", type=Path, required=True)
    reg.add_argument("--name", help="registered model name (default fusiontrace-<kind>-<detector>)")
    reg.add_argument("--alias", action="append", default=[], help="e.g. candidate (repeatable)")
    reg.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE")
    reg.add_argument("--allow-failed-parity", action="store_true")
    ali = sub.add_parser("alias", help="point an alias at a version")
    ali.add_argument("--name", required=True)
    ali.add_argument("--version", required=True)
    ali.add_argument("--alias", required=True)
    pro = sub.add_parser("promote", help="move an alias to the version another alias names (candidate -> production)")
    pro.add_argument("--name", required=True)
    pro.add_argument("--from", dest="from_alias", default="candidate")
    pro.add_argument("--to", dest="to_alias", default="production")
    lst = sub.add_parser("list", help="list registered versions, aliases and tags")
    lst.add_argument("--name")
    args = parser.parse_args(argv)

    try:
        if args.command == "register":
            tags = dict(pair.split("=", 1) for pair in args.tag)
            result: Any = asdict(register_onnx_model(args.model, name=args.name, aliases=args.alias, tags=tags,
                                                     allow_failed_parity=args.allow_failed_parity, uri=args.tracking_uri))
        elif args.command == "alias":
            set_alias(args.name, args.alias, args.version, uri=args.tracking_uri)
            result = {"name": args.name, "alias": args.alias, "version": args.version}
        elif args.command == "promote":
            version = promote(args.name, args.from_alias, args.to_alias, uri=args.tracking_uri)
            result = {"name": args.name, "alias": args.to_alias, "version": version, "from": args.from_alias}
        else:
            result = list_versions(args.name, uri=args.tracking_uri)
    except tracking.MlflowUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
