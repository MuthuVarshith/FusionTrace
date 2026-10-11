"""Shared helpers for ONNX export, quantisation and benchmarking.

Layout of exported artefacts (default root ``<FUSIONTRACE_MODEL_DIR>/onnx``)::

    <root>/<detector-name>/model.onnx              FP32 export
    <root>/<detector-name>/model.json              metadata (written next to the model)
    <root>/<detector-name>/model.int8.onnx         dynamic INT8
    <root>/<detector-name>/model.int8-static.onnx  static INT8 (QDQ, calibrated)
    <root>/<detector-name>/benchmark.{json,md}     benchmark results
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import onnxruntime as ort

METADATA_FORMAT = 1
FP32_NAME = "model.onnx"
VARIANT_FILES = {
    "onnx-fp32": FP32_NAME,
    "onnx-int8": "model.int8.onnx",
    "onnx-int8-static": "model.int8-static.onnx",
}


def default_onnx_root() -> Path:
    """``FUSIONTRACE_ONNX_DIR`` if set, else ``<model_dir>/onnx`` from the registry."""
    override = os.getenv("FUSIONTRACE_ONNX_DIR")
    if override:
        return Path(override)
    from app.detectors import registry

    return registry.model_dir() / "onnx"


def artifact_dir(detector_name: str, root: Path | None = None) -> Path:
    return (root or default_onnx_root()) / detector_name


def metadata_path(model_path: Path) -> Path:
    """``model.onnx`` -> ``model.json``; ``model.int8.onnx`` -> ``model.int8.json``."""
    return model_path.with_suffix(".json")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_files(model_path: Path) -> list[Path]:
    """The .onnx file plus an external-data sidecar (``<name>.onnx.data``) when present."""
    sidecar = model_path.with_name(model_path.name + ".data")
    return [model_path] + ([sidecar] if sidecar.exists() else [])


def model_size_bytes(model_path: Path) -> int:
    return sum(path.stat().st_size for path in model_files(model_path))


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")


def read_metadata(model_path: Path) -> dict[str, Any] | None:
    path = metadata_path(model_path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def environment() -> dict[str, Any]:
    """Library versions and host facts recorded with every artefact."""
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor() or None}
    for module in ("torch", "onnx", "onnxruntime", "numpy"):
        try:
            info[module] = __import__(module).__version__
        except ImportError:
            info[module] = None
    return info


# ── tensors and outputs ─────────────────────────────────────────────────────


def flatten_outputs(outputs: Any) -> list[Any]:
    """Flatten a forward result (tensor, tuple/list, dict or HF ``ModelOutput``) to a list of tensors.

    ``None`` entries are dropped (as ``ModelOutput.to_tuple()`` does). The ONNX
    export wraps the module so its graph outputs follow exactly this order.
    """
    if outputs is None:
        return []
    if isinstance(outputs, Mapping):
        return [tensor for value in outputs.values() for tensor in flatten_outputs(value)]
    if isinstance(outputs, (tuple, list)):
        return [tensor for value in outputs for tensor in flatten_outputs(value)]
    return [outputs]


def to_numpy(value: Any) -> np.ndarray:
    import numpy as np

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def io_names(prefix: str, count: int) -> list[str]:
    return [prefix] if count == 1 else [f"{prefix}_{index}" for index in range(count)]


def compare_outputs(reference: Sequence[Any], candidate: Sequence[Any], atol: float, rtol: float) -> dict[str, Any]:
    """Element-wise comparison: passes when ``|a - b| <= atol + rtol * |a|`` everywhere."""
    import numpy as np

    if len(reference) != len(candidate):
        return {"passed": False, "reason": f"{len(reference)} reference outputs vs {len(candidate)} candidate outputs"}
    max_abs = 0.0
    max_rel = 0.0
    passed = True
    for expected_raw, actual_raw in zip(reference, candidate, strict=True):
        expected = to_numpy(expected_raw).astype(np.float64)
        actual = to_numpy(actual_raw).astype(np.float64)
        if expected.shape != actual.shape:
            return {"passed": False, "reason": f"shape mismatch {expected.shape} vs {actual.shape}"}
        if expected.size == 0:
            continue
        diff = np.abs(expected - actual)
        max_abs = max(max_abs, float(diff.max()))
        max_rel = max(max_rel, float((diff / np.maximum(np.abs(expected), 1e-12)).max()))
        passed = passed and bool(np.all(diff <= atol + rtol * np.abs(expected)))
    return {"passed": passed, "atol": atol, "rtol": rtol, "max_abs_diff": max_abs, "max_rel_diff": max_rel}


# ── onnxruntime ─────────────────────────────────────────────────────────────


def ort_session(model_path: Path, threads: int | None = None) -> ort.InferenceSession:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def ort_feed(session: ort.InferenceSession, inputs: Sequence[Any]) -> dict[str, np.ndarray]:
    names = [item.name for item in session.get_inputs()]
    if len(names) != len(inputs):
        raise ValueError(f"model expects {len(names)} inputs {names}, got {len(inputs)}")
    return {name: to_numpy(value) for name, value in zip(names, inputs, strict=True)}


def ort_run(session: ort.InferenceSession, inputs: Sequence[Any]) -> list[np.ndarray]:
    return list(session.run(None, ort_feed(session, inputs)))


def describe_io(items: Iterable[Any]) -> list[dict[str, Any]]:
    """onnxruntime NodeArg list -> [{name, shape (ints or symbolic names), dtype}]."""
    return [{"name": item.name, "shape": list(item.shape), "dtype": item.type} for item in items]


def detector_metadata(detector: Any) -> dict[str, Any]:
    info = detector.info
    cls = type(detector)
    return {
        **info.to_report(),
        "class": f"{cls.__module__}:{cls.__qualname__}",
        "options": dict(getattr(detector, "options", {}) or {}),
    }
