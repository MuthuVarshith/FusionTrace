"""INT8-quantise an exported ONNX detector and check its risk against the FP32 model.

Modes:
    dynamic  onnxruntime ``quantize_dynamic`` (weights INT8 ahead of time, activations
             quantised at run time). Per-channel when the model allows it. CPU kernels
             for ConvInteger with INT8 weights are missing in some onnxruntime builds,
             so the script falls back (per-tensor, then without Conv) and records which
             configuration it used.
    static   ``quantize_static`` in QDQ format (U8 activations, S8 weights) calibrated
             with MinMax on a small folder of media (``--calibration-dir``). Usually the
             better choice for CNNs.

Writes ``model.int8.onnx`` / ``model.int8-static.onnx`` and a ``.json`` metadata file
next to the FP32 model. The parity check runs the detector's ``prepare_inputs`` and
``risk_from_outputs`` on the sample (and up to ``--parity-limit`` more files) through
both models and fails (exit 1) if any risk differs by more than ``--max-risk-diff``.

    python scripts/quantize_onnx.py --kind image
    python scripts/quantize_onnx.py --kind image --mode static --calibration-dir test/real
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("FUSIONTRACE_DEVICE", "cpu")

from app.detectors.base import Detector, DetectorError  # noqa: E402
from mlops import onnx_tools  # noqa: E402
from mlops.selection import SelectionError, media_files, parse_options, require_exportable, resolve_detector, resolve_sample  # noqa: E402

logger = logging.getLogger("fusiontrace.quantize")
QUANT_OPS = {
    "ConvInteger", "MatMulInteger", "DynamicQuantizeLinear", "QuantizeLinear", "DequantizeLinear",
    "QLinearConv", "QLinearMatMul", "DynamicQuantizeMatMul", "QAttention", "DynamicQuantizeLSTM",
}
EXTERNAL_DATA_THRESHOLD = 1_800_000_000  # protobuf's 2 GB cap, with margin


def _require_onnx() -> None:
    try:
        import onnx  # noqa: F401
        import onnxruntime.quantization  # noqa: F401
    except ImportError as exc:
        raise SelectionError(f"quantisation needs the onnx package ({exc}); install the versions in requirements/extra-agent-e.txt") from exc


def _quant_op_counts(path: Path) -> dict[str, int]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    counts = Counter(node.op_type for node in model.graph.node if node.op_type in QUANT_OPS)
    return dict(sorted(counts.items()))


def _preprocess(source: Path, work_dir: Path, external: bool) -> tuple[Path, str]:
    """onnxruntime's recommended pre-quantisation pass (shape inference + graph optimisation); best effort."""
    from onnxruntime.quantization.shape_inference import quant_pre_process

    target = work_dir / "preprocessed.onnx"
    try:
        quant_pre_process(str(source), str(target), save_as_external_data=external, all_tensors_to_one_file=True)
        return target, "applied"
    except Exception as exc:  # noqa: BLE001  # symbolic shape inference does not support every graph
        logger.warning("Pre-processing skipped: %s", exc)
        return source, f"skipped ({type(exc).__name__}: {exc})"


def _runs(path: Path, inputs: tuple[Any, ...]) -> str | None:
    try:
        onnx_tools.ort_run(onnx_tools.ort_session(path), inputs)
    except Exception as exc:  # noqa: BLE001  # report any onnxruntime failure as "does not run"
        return f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"
    return None


def quantize_dynamic_model(source: Path, target: Path, inputs: tuple[Any, ...], *, per_channel: bool, reduce_range: bool, external: bool) -> dict[str, Any]:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.registry import IntegerOpsRegistry

    without_conv = sorted(op for op in IntegerOpsRegistry if op != "Conv")
    candidates: list[tuple[list[str] | None, bool]] = [(None, per_channel), (None, False), (without_conv, per_channel), (without_conv, False)]
    attempts: list[dict[str, Any]] = []
    for op_types, channel in dict.fromkeys((tuple(ops) if ops else None, pc) for ops, pc in candidates):
        config = {"op_types": list(op_types) if op_types else "all supported", "per_channel": channel}
        try:
            quantize_dynamic(
                source, target, op_types_to_quantize=list(op_types) if op_types else None, per_channel=channel,
                reduce_range=reduce_range, weight_type=QuantType.QInt8, use_external_data_format=external,
            )
            problem = _runs(target, inputs)
        except Exception as exc:  # noqa: BLE001  # try the next configuration
            problem = f"{type(exc).__name__}: {exc}"
        attempts.append({**config, "error": problem})
        if problem is None:
            return {"mode": "dynamic", "weight_type": "QInt8", "reduce_range": reduce_range, **config, "attempts": attempts}
        logger.info("Dynamic quantisation config %s does not run here: %s", config, problem)
    raise SelectionError(f"no dynamic quantisation configuration produced a runnable model: {attempts}")


class _CalibrationReader:
    """Feeds ``prepare_inputs`` of each calibration file to onnxruntime's calibrator."""

    def __init__(self, detector: Detector, files: list[Path], input_names: list[str]) -> None:
        self._detector = detector
        self._files = iter(files)
        self._names = input_names
        self.used = 0
        self.skipped = 0

    def get_next(self) -> dict[str, Any] | None:
        for path in self._files:
            try:
                tensors = self._detector.prepare_inputs(path)
            except DetectorError as exc:
                logger.info("Calibration file %s skipped: %s", path.name, exc)
                self.skipped += 1
                continue
            self.used += 1
            return {name: onnx_tools.to_numpy(value) for name, value in zip(self._names, tensors, strict=True)}
        return None


def quantize_static_model(
    source: Path, target: Path, detector: Detector, files: list[Path], *, per_channel: bool, reduce_range: bool, external: bool
) -> dict[str, Any]:
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

    if not files:
        raise SelectionError("static quantisation needs --calibration-dir with at least one file of the detector's media kind")
    class Reader(_CalibrationReader, CalibrationDataReader):  # onnxruntime (and onnx) import lazily
        pass

    names = [item.name for item in onnx_tools.ort_session(source).get_inputs()]
    reader = Reader(detector, files, names)
    quantize_static(
        source, target, reader, quant_format=QuantFormat.QDQ, per_channel=per_channel, reduce_range=reduce_range,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax,
        use_external_data_format=external,
    )
    if reader.used == 0:
        raise SelectionError("no calibration file could be prepared by the detector")
    return {
        "mode": "static", "format": "QDQ", "activation_type": "QUInt8", "weight_type": "QInt8", "calibrate_method": "MinMax",
        "per_channel": per_channel, "reduce_range": reduce_range,
        "calibration": {"files_used": reader.used, "files_skipped": reader.skipped},
    }


def risk_parity(detector: Detector, fp32: Path, int8: Path, files: list[Path], max_risk_diff: float) -> dict[str, Any]:
    reference, candidate = onnx_tools.ort_session(fp32), onnx_tools.ort_session(int8)
    rows: list[dict[str, Any]] = []
    for path in files:
        try:
            inputs = detector.prepare_inputs(path)
        except DetectorError as exc:
            rows.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        risk_fp32 = float(detector.risk_from_outputs(onnx_tools.ort_run(reference, inputs)))
        risk_int8 = float(detector.risk_from_outputs(onnx_tools.ort_run(candidate, inputs)))
        rows.append({"file": path.name, "risk_fp32": risk_fp32, "risk_int8": risk_int8, "abs_diff": abs(risk_fp32 - risk_int8)})
    scored = [row for row in rows if "abs_diff" in row]
    if not scored:
        return {"passed": False, "reason": "no parity file could be prepared", "files": rows}
    diffs = [row["abs_diff"] for row in scored]
    agree = sum((row["risk_fp32"] >= 0.5) == (row["risk_int8"] >= 0.5) for row in scored)
    return {
        "passed": max(diffs) <= max_risk_diff,
        "max_risk_diff_allowed": max_risk_diff,
        "files_compared": len(scored),
        "max_abs_risk_diff": max(diffs),
        "mean_abs_risk_diff": sum(diffs) / len(diffs),
        "label_agreement_at_0_5": agree / len(scored),
        "files": rows,
    }


def quantize(
    detector: Detector,
    fp32_path: Path,
    sample: Path,
    *,
    mode: str = "dynamic",
    per_channel: bool = True,
    reduce_range: bool = False,
    calibration_files: list[Path] | None = None,
    parity_files: list[Path] | None = None,
    max_risk_diff: float = 0.05,
    preprocess: bool = True,
) -> list[dict[str, Any]]:
    """Quantise ``fp32_path`` in the requested mode(s); return the metadata of each output."""
    _require_onnx()
    require_exportable(detector)
    if not fp32_path.is_file():
        raise SelectionError(f"FP32 model {fp32_path} not found; run scripts/export_onnx.py first")
    base_meta = onnx_tools.read_metadata(fp32_path) or {}
    sample_inputs = detector.prepare_inputs(sample)
    external = onnx_tools.model_size_bytes(fp32_path) >= EXTERNAL_DATA_THRESHOLD
    modes = ("dynamic", "static") if mode == "both" else (mode,)
    outputs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="fusiontrace-quant-") as work:
        source, preprocess_note = _preprocess(fp32_path, Path(work), external) if preprocess else (fp32_path, "disabled")
        for current in modes:
            variant = "onnx-int8" if current == "dynamic" else "onnx-int8-static"
            target = fp32_path.with_name(onnx_tools.VARIANT_FILES[variant])
            if current == "dynamic":
                settings = quantize_dynamic_model(source, target, sample_inputs, per_channel=per_channel, reduce_range=reduce_range, external=external)
            else:
                settings = quantize_static_model(
                    source, target, detector, calibration_files or [], per_channel=per_channel, reduce_range=reduce_range, external=external
                )
            op_counts = _quant_op_counts(target)
            if not op_counts:
                logger.warning("%s: no operator was quantised; the model is effectively FP32", target.name)
            session = onnx_tools.ort_session(target)
            parity = risk_parity(detector, fp32_path, target, [sample, *(parity_files or [])], max_risk_diff)
            size = onnx_tools.model_size_bytes(target)
            metadata = {
                "format_version": onnx_tools.METADATA_FORMAT,
                "created_at": onnx_tools.utc_now(),
                "precision": f"int8-{current}",
                "file": target.name,
                "size_bytes": size,
                "size_ratio_vs_fp32": size / onnx_tools.model_size_bytes(fp32_path),
                "sha256": onnx_tools.sha256_file(target),
                "detector": base_meta.get("detector") or onnx_tools.detector_metadata(detector),
                "opset": base_meta.get("opset"),
                "dynamic_batch": base_meta.get("dynamic_batch"),
                "inputs": onnx_tools.describe_io(session.get_inputs()),
                "outputs": onnx_tools.describe_io(session.get_outputs()),
                "base_model": {"file": fp32_path.name, "sha256": base_meta.get("sha256") or onnx_tools.sha256_file(fp32_path)},
                "quantization": {**settings, "preprocess": preprocess_note, "quantized_op_counts": op_counts},
                "parity": parity,
                "environment": onnx_tools.environment(),
            }
            onnx_tools.write_json(onnx_tools.metadata_path(target), metadata)
            outputs.append(metadata)
    return outputs


def _detector_for(args: argparse.Namespace) -> tuple[Detector, Path]:
    options = parse_options(args.option)
    if args.model and not (args.kind or args.detector):
        meta = onnx_tools.read_metadata(args.model)
        if not meta:
            raise SelectionError(f"{args.model} has no metadata JSON next to it; pass --kind/--detector")
        detector = resolve_detector(meta["detector"]["kind"], meta["detector"]["class"], {**meta["detector"].get("options", {}), **options})
        return detector, args.model
    detector = resolve_detector(args.kind, args.detector, options)
    return detector, args.model or onnx_tools.artifact_dir(detector.info.name) / onnx_tools.FP32_NAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kind", choices=("image", "audio", "video"))
    parser.add_argument("--detector", help="registry key or module:Class import path")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--model", type=Path, help="FP32 .onnx (default: <onnx dir>/<detector>/model.onnx); its metadata names the detector")
    parser.add_argument("--mode", choices=("dynamic", "static", "both"), default="dynamic")
    parser.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reduce-range", action="store_true", help="7-bit weights; can help accuracy on CPUs without VNNI")
    parser.add_argument("--calibration-dir", type=Path, help="media folder for static calibration (searched recursively)")
    parser.add_argument("--calibration-limit", type=int, default=32)
    parser.add_argument("--sample", type=Path, help="parity sample (default: first bundled test file of that kind)")
    parser.add_argument("--parity-dir", type=Path, help="extra files for the risk parity check (default: the calibration dir)")
    parser.add_argument("--parity-limit", type=int, default=8)
    parser.add_argument("--max-risk-diff", type=float, default=0.05, help="largest allowed |risk_fp32 - risk_int8|")
    parser.add_argument("--no-preprocess", action="store_true", help="skip onnxruntime's quant_pre_process pass")
    parser.add_argument("--mlflow", action="store_true", help="log the quantised model(s) to MLflow")
    parser.add_argument("--register", nargs="?", const="", metavar="NAME", help="with --mlflow: register each quantised model")
    parser.add_argument("--alias", action="append", default=[])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        detector, fp32_path = _detector_for(args)
        sample = resolve_sample(detector, args.sample)
        kind = detector.info.kind
        calibration = media_files(args.calibration_dir, kind, args.calibration_limit) if args.calibration_dir else []
        parity_dir = args.parity_dir or args.calibration_dir
        parity = [path for path in media_files(parity_dir, kind, args.parity_limit + 1) if path != sample][: args.parity_limit] if parity_dir else []
        results = quantize(
            detector, fp32_path, sample, mode=args.mode, per_channel=args.per_channel, reduce_range=args.reduce_range,
            calibration_files=calibration, parity_files=parity, max_risk_diff=args.max_risk_diff, preprocess=not args.no_preprocess,
        )
    except SelectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = [
        {"model": str(fp32_path.with_name(meta["file"])), "precision": meta["precision"], "size_bytes": meta["size_bytes"],
         "size_ratio_vs_fp32": round(meta["size_ratio_vs_fp32"], 4), "quantized_op_counts": meta["quantization"]["quantized_op_counts"],
         "parity": {k: v for k, v in meta["parity"].items() if k != "files"}}
        for meta in results
    ]
    print(json.dumps(summary, indent=2))
    if args.mlflow:
        from mlops import model_registry

        for meta in results:
            model_registry.log_and_maybe_register(fp32_path.with_name(meta["file"]), register=args.register, aliases=args.alias, tool="quantize_onnx")
    if not all(meta["parity"]["passed"] for meta in results):
        print("error: INT8 risk differs from FP32 beyond --max-risk-diff; see parity in the metadata", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
