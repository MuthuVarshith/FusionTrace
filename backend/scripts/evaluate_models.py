"""Evaluate bundled models against a labelled folder layout.

Expected layout:
    test/real/*       genuine media
    test/deepfakes/*  manipulated/synthetic media

The generated report is intended for model validation, not for claiming a
universal accuracy figure: the test files may be related to the training data.
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.config import AUDIO_MODEL_PATH, BASE_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}


def metric_report(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(item["expected"] == item["predicted"] for item in results)
    true_fake = sum(item["expected"] == "Fake" and item["predicted"] == "Fake" for item in results)
    false_positive = sum(item["expected"] == "Real" and item["predicted"] == "Fake" for item in results)
    false_negative = sum(item["expected"] == "Fake" and item["predicted"] == "Real" for item in results)
    precision = true_fake / (true_fake + false_positive) if true_fake + false_positive else None
    recall = true_fake / (true_fake + false_negative) if true_fake + false_negative else None
    review_queue = [
        {"file": item["file"], "issue": "high-confidence false positive", "confidence": item["confidence"]}
        for item in results
        if item["expected"] == "Real" and item["predicted"] == "Fake" and float(item["confidence"].rstrip("%")) >= 90
    ]
    return {"samples": total, "accuracy": correct / total if total else None, "fake_precision": precision, "fake_recall": recall, "false_positives": false_positive, "false_negatives": false_negative, "calibration_review_queue": review_queue, "results": results}


def labelled_files(test_dir: Path, extensions: set[str]):
    for folder, label in ((test_dir / "real", "Real"), (test_dir / "deepfakes", "Fake")):
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in extensions:
                yield path, label


def evaluate_images(test_dir: Path) -> dict:
    from app.image_detection import detect_image_deepfake
    results = []
    for path, expected in labelled_files(test_dir, IMAGE_EXTENSIONS):
        output = detect_image_deepfake(str(path))
        results.append({"file": path.name, "expected": expected, "predicted": output["prediction"], "confidence": output["confidence"]})
    return metric_report(results)


def evaluate_audio(test_dir: Path) -> dict:
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
    from app.audio_detection import predict_audio
    processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL_PATH)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(AUDIO_MODEL_PATH)
    results = []
    for path, expected in labelled_files(test_dir, AUDIO_EXTENSIONS):
        prediction, confidence, _ = predict_audio(str(path), model, processor)
        results.append({"file": path.name, "expected": expected, "predicted": prediction, "confidence": confidence})
    return metric_report(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--media", choices=("image", "audio", "all"), default="all")
    parser.add_argument("--test-dir", type=Path, default=BASE_DIR / "test")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "data" / "evaluation-report.json")
    args = parser.parse_args()
    report = {"warning": "This report is only valid for the supplied dataset; do not use it as a general accuracy claim."}
    if args.media in ("image", "all"):
        report["image"] = evaluate_images(args.test_dir)
    if args.media in ("audio", "all"):
        report["audio"] = evaluate_audio(args.test_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
