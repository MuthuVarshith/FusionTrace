"""Export a detector's torch module to ONNX and verify it against torch with onnxruntime.

The detector is chosen through the registry (``--kind`` for the configured one,
``--detector`` for a registry key or a ``module:Class`` import path) and must
implement the export hooks (``exportable = True``). Writes, by default under
``<FUSIONTRACE_MODEL_DIR>/onnx/<detector-name>/``:

    model.onnx   the exported graph (opset 17+, dynamic batch axis where the model allows it)
    model.json   detector name/version/source, opset, I/O names and shapes, sha256,
                 tolerance and the measured torch-vs-onnxruntime difference

Exit status: 0 on success, 1 when the parity check fails (files are still written
and flagged), 2 when the detector cannot be exported.

    python scripts/export_onnx.py --kind image
    python scripts/export_onnx.py --detector app.detectors.testing:TinyTorchDetector --sample test/real/00007.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("FUSIONTRACE_DEVICE", "cpu")  # export and parity run on CPU

from mlops import onnx_tools  # noqa: E402
from mlops.selection import (  # noqa: E402
    SelectionError,
    detector_spec,
    parse_options,
    require_exportable,
    resolve_detector,
    resolve_sample,
)

logger = logging.getLogger("fusiontrace.export")
MIN_OPSET = 17


def _flat_module(module: Any) -> Any:
    """Wrap ``module`` so its forward returns a flat tuple of tensors (stable ONNX output order)."""
    import torch

    class FlatOutputs(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return tuple(onnx_tools.flatten_outputs(self.inner(*args)))

    return FlatOutputs(module).eval()


def _torch_export(module: Any, inputs: tuple[Any, ...], path: Path, opset: int, input_names: list[str], output_names: list[str], dynamic: bool) -> None:
    import torch

    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names} if dynamic else None
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            module,
            inputs,
            str(path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,  # TorchScript exporter: no onnxscript dependency, handles HF models
        )


def _check_dynamic_batch(path: Path, inputs: tuple[Any, ...]) -> str | None:
    """Run the exported model with batch 2; return an error string if that fails."""
    import numpy as np

    try:
        session = onnx_tools.ort_session(path)
        doubled = [np.concatenate([onnx_tools.to_numpy(x)] * 2, axis=0) for x in inputs]
        outputs = onnx_tools.ort_run(session, doubled)
        if any(output.ndim == 0 or output.shape[0] != 2 for output in outputs):
            return f"batch-2 run returned shapes {[tuple(output.shape) for output in outputs]}"
    except Exception as exc:  # onnxruntime raises its own exception types
        return f"{type(exc).__name__}: {exc}"
    return None


def _onnx_check(path: Path) -> str:
    try:
        import onnx
    except ImportError:
        return "skipped (onnx not installed)"
    onnx.checker.check_model(str(path))
    return "passed"


def export(
    detector: Any,
    sample: Path,
    output_dir: Path,
    *,
    opset: int = MIN_OPSET,
    dynamic_batch: bool = True,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """Export ``detector`` and return the metadata dict (also written next to the model)."""
    import torch

    if opset < MIN_OPSET:
        raise SelectionError(f"opset must be >= {MIN_OPSET}, got {opset}")
    require_exportable(detector)
    detector.load()
    module = _flat_module(detector.export_module().to("cpu").eval())
    inputs = tuple(tensor.to("cpu") if hasattr(tensor, "to") else tensor for tensor in detector.prepare_inputs(sample))
    with torch.no_grad():
        reference = list(module(*inputs))
    if not reference:
        raise SelectionError("export_module() returned no tensors")

    input_names = onnx_tools.io_names("input", len(inputs))
    output_names = onnx_tools.io_names("output", len(reference))
    batchable = all(getattr(tensor, "ndim", 0) >= 1 and tensor.shape[0] == 1 for tensor in inputs)
    dynamic = dynamic_batch and batchable
    dynamic_note = None if batchable or not dynamic_batch else "inputs have no leading batch dimension of size 1"

    path = output_dir / onnx_tools.FP32_NAME
    _torch_export(module, inputs, path, opset, input_names, output_names, dynamic)
    if dynamic:
        problem = _check_dynamic_batch(path, inputs)
        if problem:
            logger.warning("Dynamic batch export does not run with batch 2 (%s); re-exporting with a static batch.", problem)
            dynamic, dynamic_note = False, f"static batch: dynamic export failed at batch 2 ({problem})"
            _torch_export(module, inputs, path, opset, input_names, output_names, dynamic=False)

    checker = _onnx_check(path)
    session = onnx_tools.ort_session(path)
    candidate = onnx_tools.ort_run(session, inputs)
    parity = onnx_tools.compare_outputs(reference, candidate, atol=atol, rtol=rtol)
    risk_torch = float(detector.risk_from_outputs(reference))
    risk_onnx = float(detector.risk_from_outputs(candidate))
    parity.update({"risk_torch": risk_torch, "risk_onnx": risk_onnx, "risk_abs_diff": abs(risk_torch - risk_onnx)})

    metadata = {
        "format_version": onnx_tools.METADATA_FORMAT,
        "created_at": onnx_tools.utc_now(),
        "precision": "fp32",
        "file": path.name,
        "size_bytes": onnx_tools.model_size_bytes(path),
        "sha256": onnx_tools.sha256_file(path),
        "detector": onnx_tools.detector_metadata(detector),
        "opset": opset,
        "exporter": "torch.onnx.export (TorchScript)",
        "dynamic_batch": dynamic,
        "dynamic_batch_note": dynamic_note,
        "inputs": onnx_tools.describe_io(session.get_inputs()),
        "outputs": onnx_tools.describe_io(session.get_outputs()),
        "sample": {"name": sample.name, "sha256": onnx_tools.sha256_file(sample)},
        "onnx_checker": checker,
        "parity": parity,
        "environment": onnx_tools.environment(),
    }
    onnx_tools.write_json(onnx_tools.metadata_path(path), metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kind", choices=("image", "audio", "video"), help="media kind; without --detector, uses the configured detector")
    parser.add_argument("--detector", help="registry key or module:Class import path")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE", help="detector option override (repeatable)")
    parser.add_argument("--sample", type=Path, help="file passed to prepare_inputs() (default: first bundled test file of that kind)")
    parser.add_argument("--output-dir", type=Path, help="default: <FUSIONTRACE_MODEL_DIR>/onnx/<detector-name>")
    parser.add_argument("--opset", type=int, default=MIN_OPSET)
    parser.add_argument("--static-batch", action="store_true", help="do not mark the batch axis as dynamic")
    parser.add_argument("--atol", type=float, default=1e-4, help="absolute tolerance for torch vs onnxruntime outputs")
    parser.add_argument("--rtol", type=float, default=1e-3, help="relative tolerance for torch vs onnxruntime outputs")
    parser.add_argument("--mlflow", action="store_true", help="log the export to MLflow (needs mlflow installed)")
    parser.add_argument(
        "--register", nargs="?", const="", metavar="NAME", help="with --mlflow: register the model (default name fusiontrace-<kind>-<detector>)",
    )
    parser.add_argument("--alias", action="append", default=[], help="with --register: alias to set on the new version, e.g. candidate (repeatable)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        detector = resolve_detector(args.kind, args.detector, parse_options(args.option))
        require_exportable(detector)
        sample = resolve_sample(detector, args.sample)
        output_dir = args.output_dir or onnx_tools.artifact_dir(detector.info.name)
        metadata = export(detector, sample, output_dir, opset=args.opset, dynamic_batch=not args.static_batch, atol=args.atol, rtol=args.rtol)
    except SelectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    model_path = output_dir / onnx_tools.FP32_NAME
    summary = {k: metadata[k] for k in ("sha256", "size_bytes", "opset", "dynamic_batch", "parity")}
    print(json.dumps({"model": str(model_path), "detector_class": detector_spec(detector), **summary}, indent=2))
    if args.mlflow:
        from mlops import model_registry

        model_registry.log_and_maybe_register(model_path, register=args.register, aliases=args.alias, tool="export_onnx")
    if not metadata["parity"]["passed"]:
        print("error: onnxruntime outputs differ from torch beyond the tolerance; see parity in the metadata", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
