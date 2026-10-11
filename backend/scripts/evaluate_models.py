"""Evaluate detectors (selected through the registry) against labelled folders.

Expected layout (searched recursively, filtered by the media kind's extensions):
    <data-dir>/real/*                  genuine media
    <data-dir>/fake/* or deepfakes/*   manipulated/synthetic media

Reports ROC AUC, Fake precision/recall/F1, real and fake recall and the confusion
matrix at the detector's own threshold (its label), optionally also at
``--threshold``. Files that raise DetectorError are counted separately and
excluded from the metrics. Writes JSON and markdown.

The report is only valid for the supplied dataset: the bundled test files may be
related to the training data, so it is not a general accuracy claim.

    python scripts/evaluate_models.py --media image --data-dir test
    python scripts/evaluate_models.py --media audio --detector <key> --threshold 0.7 --mlflow
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.config import BASE_DIR  # noqa: E402
from app.detectors.base import AnalysisContext, Detector, DetectorError, MediaKind  # noqa: E402
from mlops import metrics, onnx_tools  # noqa: E402
from mlops.selection import SelectionError, labelled_files, parse_options, resolve_detector  # noqa: E402

WARNING = "This report is only valid for the supplied dataset; do not use it as a general accuracy claim."
REVIEW_CONFIDENCE = 0.9


def dataset_fingerprint(files: list[tuple[Path, str]], root: Path) -> str:
    """sha256 over (label, relative path, file sha256) of every file, in sorted order."""
    digest = hashlib.sha256()
    for path, label in sorted(files, key=lambda item: item[0].relative_to(root).as_posix()):
        digest.update(f"{label}\t{path.relative_to(root).as_posix()}\t{onnx_tools.sha256_file(path)}\n".encode())
    return digest.hexdigest()


def evaluate_detector(detector: Detector, data_dir: Path, kind: MediaKind, threshold: float | None = None) -> dict[str, Any]:
    files = labelled_files(data_dir, kind)
    section: dict[str, Any] = {"detector": onnx_tools.detector_metadata(detector),
                               "dataset": {"path": str(data_dir), "files": len(files), "sha256": dataset_fingerprint(files, data_dir)}}
    if not files:
        return {**section, "status": "skipped", "reason": f"no labelled {kind.value} files under {data_dir}"}
    try:
        detector.load()
    except DetectorError as exc:
        return {**section, "status": "skipped", "reason": f"detector unavailable: {exc}"}
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for path, expected in files:
        rel = path.relative_to(data_dir).as_posix()
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="fusiontrace-eval-") as work:
            try:
                result = detector.analyze(path, AnalysisContext(work_dir=Path(work), options=detector.options))
            except DetectorError as exc:
                errors.append({"file": rel, "expected": expected, "error_type": type(exc).__name__, "message": str(exc)})
                continue
        row = {"file": rel, "expected": expected, "predicted": result.label, "risk": result.risk,
               "confidence": result.confidence, "seconds": round(time.perf_counter() - start, 4)}
        if "threshold" in result.details:
            row["detector_threshold"] = result.details["threshold"]
        results.append(row)
    expected = [r["expected"] for r in results]
    scored = metrics.scored_report(expected, [r["risk"] for r in results], [r["predicted"] for r in results], threshold) if results else {}
    review = [{"file": r["file"], "issue": "high-confidence false positive" if r["expected"] == "Real" else "high-confidence false negative",
               "confidence": r["confidence"], "risk": r["risk"]}
              for r in results if r["expected"] != r["predicted"] and r["confidence"] >= REVIEW_CONFIDENCE]
    counts = {"files": len(files), "scored": len(results), "real": expected.count("Real"), "fake": expected.count("Fake"), "errors": len(errors)}
    return {**section, "status": "ok", "counts": counts, **scored, "errors": errors, "calibration_review_queue": review, "results": results}


def to_markdown(report: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = ["# FusionTrace evaluation", "", f"> **{report['warning']}**", "", f"Generated {report['generated_at']} from `{report['data_dir']}`.", ""]
    for kind, section in report["evaluations"].items():
        det = section["detector"]
        lines += [f"## {kind}: {det['name']} {det.get('version') or ''}".rstrip(), "",
                  f"Dataset sha256 `{section['dataset']['sha256'][:16]}...`, {section['dataset']['files']} files.", ""]
        if section["status"] != "ok":
            lines += [f"Skipped: {section['reason']}", ""]
            continue
        c = section["counts"]
        auc = fmt(section.get("roc_auc"))
        lines += [f"Scored {c['scored']} ({c['real']} real, {c['fake']} fake); {c['errors']} file(s) raised DetectorError. ROC AUC: {auc}", "",
                  "| Threshold | Accuracy | Fake precision | Fake recall | Fake F1 | Real recall | TN | FP | FN | TP |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        user = section.get("at_threshold") or {}
        for title, block in (("detector", section.get("at_detector_threshold")), (str(user.get("threshold")), user)):
            if block:
                cm = block["confusion_matrix"]
                lines.append(f"| {title} | {fmt(block['accuracy'])} | {fmt(block['fake_precision'])} | {fmt(block['fake_recall'])} | {fmt(block['fake_f1'])} | "
                             f"{fmt(block['real_recall'])} | {cm['tn']} | {cm['fp']} | {cm['fn']} | {cm['tp']} |")
        lines.append("")
        if section["errors"]:
            lines += ["Errors:", ""] + [f"- `{e['file']}` ({e['expected']}): {e['error_type']}" for e in section["errors"][:20]] + [""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--media", "--kind", dest="media", choices=("image", "audio", "video", "all"), default="all")
    parser.add_argument("--detector", help="registry key or module:Class (requires a single --media)")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--data-dir", "--test-dir", dest="data_dir", type=Path, default=BASE_DIR / "test")
    parser.add_argument("--threshold", type=float, help="also report metrics at risk >= THRESHOLD")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "data" / "evaluation-report.json")
    parser.add_argument("--mlflow", action="store_true", help="log one MLflow run per detector")
    parser.add_argument("--experiment", default="fusiontrace-evaluation")
    args = parser.parse_args(argv)
    if args.detector and args.media == "all":
        print("error: --detector requires --media image|audio|video", file=sys.stderr)
        return 2
    kinds = list(MediaKind) if args.media == "all" else [MediaKind(args.media)]
    report: dict[str, Any] = {"warning": WARNING, "generated_at": onnx_tools.utc_now(), "data_dir": str(args.data_dir), "evaluations": {}}
    for kind in kinds:
        try:
            detector = resolve_detector(kind, args.detector, parse_options(args.option))
        except SelectionError as exc:
            if args.media != "all":
                print(f"error: {exc}", file=sys.stderr)
                return 2
            report["evaluations"][kind.value] = {"detector": {"name": None, "kind": kind.value}, "dataset": {"sha256": "", "files": 0},
                                                 "status": "skipped", "reason": str(exc)}
            continue
        report["evaluations"][kind.value] = evaluate_detector(detector, args.data_dir, kind, args.threshold)
    md_path = args.output.with_suffix(".md")
    onnx_tools.write_json(args.output, report)
    md_path.write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))
    print(f"wrote {args.output} and {md_path}")
    if args.mlflow:
        from mlops import tracking

        for section in report["evaluations"].values():
            if section["status"] == "ok":
                try:
                    run_id = tracking.log_evaluation(
                        section, dataset=section["dataset"], report_path=args.output, markdown_path=md_path, experiment=args.experiment
                    )
                    print(f"mlflow: logged run {run_id}")
                except tracking.MlflowUnavailableError as exc:
                    print(f"mlflow: skipped: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
