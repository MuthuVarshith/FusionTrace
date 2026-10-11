"""Benchmark a detector: torch vs ONNX FP32 vs ONNX INT8 on CPU.

Per variant: load time, latency over ``--runs`` timed forwards after ``--warmup``
(p50/p95/mean/min/max, ms), peak RSS delta (sampled with psutil), size on disk
(ONNX file; torch = parameter+buffer bytes). Each variant runs in a fresh
subprocess by default so memory numbers do not mix. Writes JSON and markdown
(default ``<onnx dir>/<detector>/benchmark.{json,md}``).

    python scripts/benchmark_models.py --detector app.detectors.testing:TinyTorchDetector --runs 20
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("FUSIONTRACE_DEVICE", "cpu")

from mlops import onnx_tools  # noqa: E402
from mlops.selection import SelectionError, detector_spec, parse_options, require_exportable, resolve_detector, resolve_sample  # noqa: E402

VARIANTS = ("torch", "onnx-fp32", "onnx-int8", "onnx-int8-static")
MARKER = "BENCHMARK_RESULT "


class _RssSampler:
    """Max RSS seen between start() and stop(), polled every 5 ms (None without psutil)."""

    def __init__(self) -> None:
        try:
            import psutil

            self._proc: Any = psutil.Process()
        except ImportError:
            self._proc = None
        self.baseline = self.peak = self._rss()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _rss(self) -> int | None:
        return self._proc.memory_info().rss if self._proc else None

    def _loop(self) -> None:
        while not self._stop.wait(0.005):
            current = self._rss()
            if current and self.peak is not None:
                self.peak = max(self.peak, current)

    def start(self) -> None:
        self.baseline = self.peak = self._rss()
        self._thread.start()

    def stop(self) -> float | None:
        self._stop.set()
        self._thread.join()
        current = self._rss()
        if current is None or self.baseline is None or self.peak is None:
            return None
        return (max(self.peak, current) - self.baseline) / 2**20


def _latency(fn: Any, runs: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    times.sort()

    def pct(q: float) -> float:
        k = (len(times) - 1) * q
        lo, hi = int(k), min(int(k) + 1, len(times) - 1)
        return times[lo] + (times[hi] - times[lo]) * (k - lo)

    return {"p50": pct(0.5), "p95": pct(0.95), "mean": statistics.fmean(times), "min": times[0], "max": times[-1]}


def run_variant(variant: str, detector: Any, inputs: list[Any], model_path: Path | None, runs: int, warmup: int, threads: int) -> dict[str, Any]:
    """Measure one variant in this process. ``inputs`` are numpy arrays."""
    row: dict[str, Any] = {"variant": variant, "available": True}
    sampler = _RssSampler()
    if variant == "torch":
        import torch

        torch.set_num_threads(threads)
        sampler.start()
        start = time.perf_counter()
        detector.load()
        module = detector.export_module().to("cpu").eval()
        row["load_seconds"] = time.perf_counter() - start
        tensors = [torch.from_numpy(x) for x in inputs]
        with torch.inference_mode():
            row["latency_ms"] = _latency(lambda: module(*tensors), runs, warmup)
            outputs = onnx_tools.flatten_outputs(module(*tensors))
        size = sum(t.numel() * t.element_size() for t in module.state_dict().values())
        row["size_note"] = "parameter + buffer bytes"
    else:
        assert model_path is not None
        sampler.start()
        start = time.perf_counter()
        session = onnx_tools.ort_session(model_path, threads)
        row["load_seconds"] = time.perf_counter() - start
        feed = onnx_tools.ort_feed(session, inputs)
        row["latency_ms"] = _latency(lambda: session.run(None, feed), runs, warmup)
        outputs = session.run(None, feed)
        size = onnx_tools.model_size_bytes(model_path)
    row["peak_rss_delta_mb"] = sampler.stop()
    row["size_mb"] = size / 2**20
    row["risk"] = float(detector.risk_from_outputs(outputs))
    return row


def benchmark(detector: Any, sample: Path, onnx_dir: Path, *, variants: tuple[str, ...] = VARIANTS, runs: int = 20, warmup: int = 3,
              threads: int | None = None, isolate: bool = True, detector_arg: str | None = None, kind: str | None = None,
              options: list[str] | None = None) -> dict[str, Any]:
    require_exportable(detector)
    threads = threads or int(os.getenv("OMP_NUM_THREADS") or os.cpu_count() or 1)
    inputs = [onnx_tools.to_numpy(x) for x in detector.prepare_inputs(sample)]
    results = []
    with tempfile.TemporaryDirectory(prefix="fusiontrace-bench-") as work:
        npz = Path(work) / "inputs.npz"
        import numpy as np

        np.savez(npz, *inputs)
        for variant in variants:
            model_path = None if variant == "torch" else onnx_dir / onnx_tools.VARIANT_FILES[variant]
            if model_path is not None and not model_path.is_file():
                results.append({"variant": variant, "available": False, "reason": f"{model_path.name} not found; run export_onnx.py / quantize_onnx.py"})
                continue
            if not isolate:
                results.append(run_variant(variant, detector, inputs, model_path, runs, warmup, threads))
                continue
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", variant, "--inputs", str(npz), "--runs", str(runs),
                   "--warmup", str(warmup), "--threads", str(threads), "--detector", detector_arg or detector_spec(detector)]
            cmd += (["--kind", kind] if kind else []) + (["--model", str(model_path)] if model_path else [])
            for option in options or []:
                cmd += ["--option", option]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            lines = [line for line in proc.stdout.splitlines() if line.startswith(MARKER)]
            if proc.returncode or not lines:
                results.append({"variant": variant, "available": False, "reason": f"worker failed: {proc.stderr.strip()[-500:]}"})
            else:
                results.append(json.loads(lines[-1][len(MARKER):]))
    torch_risk = next((r["risk"] for r in results if r["variant"] == "torch" and r.get("available")), None)
    for row in results:
        if row.get("available") and torch_risk is not None:
            row["risk_abs_diff_vs_torch"] = abs(row["risk"] - torch_risk)
    return {"created_at": onnx_tools.utc_now(), "detector": onnx_tools.detector_metadata(detector), "sample": sample.name,
            "runs": runs, "warmup": warmup, "threads": threads, "isolated": isolate, "environment": onnx_tools.environment(), "results": results}


def to_markdown(report: dict[str, Any]) -> str:
    def fmt(value: Any, digits: int = 2) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    det = report["detector"]
    lines = [f"# Benchmark: {det['name']} ({det['kind']})", "",
             f"{report['created_at']} - {report['runs']} runs after {report['warmup']} warm-up, {report['threads']} thread(s), "
             f"{'one subprocess per variant' if report['isolated'] else 'single process'}; {report['environment']['platform']}", "",
             "| Variant | Size (MB) | Load (s) | p50 (ms) | p95 (ms) | Mean (ms) | Peak RSS delta (MB) | Risk | Risk diff vs torch |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in report["results"]:
        if not row.get("available"):
            lines.append(f"| {row['variant']} | not run: {row.get('reason', '')[:80]} |||||||| ")
            continue
        lat = row["latency_ms"]
        lines.append(f"| {row['variant']} | {fmt(row['size_mb'], 3)} | {fmt(row['load_seconds'], 3)} | {fmt(lat['p50'], 3)} | {fmt(lat['p95'], 3)} | "
                     f"{fmt(lat['mean'], 3)} | {fmt(row['peak_rss_delta_mb'], 1)} | {fmt(row['risk'], 4)} | {fmt(row.get('risk_abs_diff_vs_torch'), 5)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kind", choices=("image", "audio", "video"))
    parser.add_argument("--detector", help="registry key or module:Class import path")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--onnx-dir", type=Path, help="default: <onnx dir>/<detector-name>")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--in-process", action="store_true", help="no subprocesses (memory numbers then overlap)")
    parser.add_argument("--output", type=Path, help="JSON path (markdown written next to it)")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--inputs", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--model", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        detector = resolve_detector(args.kind, args.detector, parse_options(args.option))
        if args.worker:
            import numpy as np

            with np.load(args.inputs) as data:
                inputs = [data[key] for key in sorted(data.files, key=lambda k: int(k.split("_")[1]))]
            row = run_variant(args.worker, detector, inputs, args.model, args.runs, args.warmup, args.threads)
            print(MARKER + json.dumps(row))
            return 0
        require_exportable(detector)
        sample = resolve_sample(detector, args.sample)
        onnx_dir = args.onnx_dir or onnx_tools.artifact_dir(detector.info.name)
        variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
        unknown = set(variants) - set(VARIANTS)
        if unknown:
            raise SelectionError(f"unknown variants {sorted(unknown)}; choose from {VARIANTS}")
        report = benchmark(detector, sample, onnx_dir, variants=variants, runs=args.runs, warmup=args.warmup, threads=args.threads,
                           isolate=not args.in_process, detector_arg=args.detector, kind=args.kind, options=args.option)
    except SelectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    json_path = args.output or onnx_dir / "benchmark.json"
    md_path = json_path.with_suffix(".md")
    onnx_tools.write_json(json_path, report)
    md_path.write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))
    print(f"wrote {json_path} and {md_path}")
    if args.mlflow:
        from mlops import tracking

        try:
            print(f"mlflow: logged run {tracking.log_benchmark(report, json_path=json_path, markdown_path=md_path)}")
        except tracking.MlflowUnavailableError as exc:
            print(f"mlflow: skipped: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
