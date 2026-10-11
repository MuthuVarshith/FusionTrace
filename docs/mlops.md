# MLOps: export, quantise, benchmark, evaluate, track

Everything runs locally on CPU. No media leaves the machine, and nothing here is
loaded by the web app. Run commands from `backend/` with the project venv.

```bash
cd backend
export FUSIONTRACE_MODEL_DIR=../.models      # weights root; ONNX goes to $FUSIONTRACE_MODEL_DIR/onnx
pip install -r ../requirements/extra-agent-e.txt   # onnx (+ optional mlflow-skinny, psutil)
```

## Choosing a detector

Every tool picks the detector through `app.detectors.registry`:

| Flag | Meaning |
|---|---|
| `--kind image` | the configured detector (`config.yaml` / `FUSIONTRACE_IMAGE_MODEL`) |
| `--detector <key>` | a registered implementation, by registry key |
| `--detector pkg.module:Class` | an explicit class, e.g. `app.detectors.testing:TinyTorchDetector` for smoke tests |
| `--option key=value` | a detector option override (repeatable, JSON values) |

Export, quantisation and benchmarks need `exportable = True` (the detector implements
`export_module`, `prepare_inputs` and `risk_from_outputs`). Otherwise they stop with
an error naming the detector. Evaluation works with any detector.

## 1. Export to ONNX

```bash
python scripts/export_onnx.py --kind image                     # sample = first bundled test file
python scripts/export_onnx.py --kind audio --sample test/real/audio1.wav --atol 1e-4 --rtol 1e-3
```

This writes `<onnx dir>/<detector>/model.onnx` and `model.json`. The defaults are opset 17
and a dynamic batch axis. The exporter checks that the model runs at batch 2 and falls
back to a static batch if it doesn't. The script compares onnxruntime against torch on
the sample and exits with status 1 if the outputs differ by more than the tolerance.
`model.json` records the detector name, version and source, opset, input/output names
and shapes, sha256, tolerance, max abs/rel diff and risk diff.

## 2. Quantise to INT8

```bash
python scripts/quantize_onnx.py --kind image                                   # dynamic INT8
python scripts/quantize_onnx.py --kind image --mode static --calibration-dir test/real
python scripts/quantize_onnx.py --model ../.models/onnx/<detector>/model.onnx --mode both --calibration-dir test
```

- **Dynamic** (`model.int8.onnx`) is per-channel when possible. If onnxruntime can't run
  the result (for example, ConvInteger with INT8 weights), the script retries
  per-tensor, then without Conv. It records which configuration it used.
- **Static** (`model.int8-static.onnx`) uses QDQ with U8 activations and S8 weights,
  MinMax-calibrated on up to `--calibration-limit` files.
- **Parity:** the script compares FP32 and INT8 risk on the sample plus up to
  `--parity-limit` files. It exits with status 1 if any difference exceeds
  `--max-risk-diff` (default 0.05).

## 3. Benchmark

```bash
python scripts/benchmark_models.py --kind image --runs 20 --warmup 3 --threads 2
```

The benchmark compares torch, ONNX FP32, ONNX INT8 and static INT8 (whichever exist).
For each, it reports load time, p50/p95/mean latency, peak RSS delta (needs psutil) and
size. Each variant runs in its own subprocess; pass `--in-process` to skip that. Output
goes to `<onnx dir>/<detector>/benchmark.{json,md}`.

## 4. Evaluate

Data layout: `<dir>/real/*` and `<dir>/fake/*` (or `deepfakes/*`), searched recursively.

```bash
python scripts/evaluate_models.py --media image --data-dir /path/to/labelled --threshold 0.7
python scripts/evaluate_models.py --media all                       # bundled backend/test set
```

For each detector, the report includes:

- ROC AUC (rank-based, ties count half)
- Fake precision, recall and F1
- Real recall and fake recall
- Accuracy
- The confusion matrix at the detector's own label, and at `--threshold` if given
- Counts, and files that raised `DetectorError` (listed separately and excluded from the metrics)
- A high-confidence error review queue

Output goes to `data/evaluation-report.{json,md}`. Video evaluation runs the temporal
detector only, without pipeline frame results. **These results apply only to the
supplied dataset; they are not a general accuracy claim.**

## 5. Tracking and registry (optional MLflow)

Add `--mlflow` to any of the four scripts. The store is `MLFLOW_TRACKING_URI` if set,
otherwise a file store at `<repo>/.mlruns`. MLflow 3.x needs `MLFLOW_ALLOW_FILE_STORE=true`
for a file store, and the helpers set it for you. To use a database instead, set
`MLFLOW_TRACKING_URI=sqlite:///...`. Evaluation runs log:

- the detector name, version, class and options
- the dataset path and content sha256
- the metrics
- the report JSON/markdown and confusion matrices

```bash
python scripts/export_onnx.py --kind image --mlflow --register --alias candidate
python -m mlops.model_registry register --model ../.models/onnx/<detector>/model.int8.onnx --alias candidate
python -m mlops.model_registry list
python -m mlops.model_registry promote --name fusiontrace-image-<detector> --from candidate --to production
python -m mlops.model_registry alias --name fusiontrace-image-<detector> --version 2 --alias production
```

Registration refuses a model whose sha256 doesn't match its metadata or whose parity
check failed (override with `--allow-failed-parity`). Aliases are labels only, because
the app does not read the registry.

## Tests

```bash
python -m pytest -q tests/test_mlops.py
```

The tests cover the metric math, evaluation with stub detectors, and export parity on
`TinyTorchDetector` (skipped when onnx isn't installed).
