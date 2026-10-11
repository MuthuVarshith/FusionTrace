"""Quick sanity check of the configured video (temporal) detector on a labelled folder.

Usage (from backend/):
    python scripts/check_video_models.py <root> [--no-image] [--max-clips N]

``<root>`` contains ``real/`` and ``fake/`` sub-folders of videos. Prints the
risk per file, the mean risk per class and the ROC AUC. With a handful of
clips this is a smoke test, not an evaluation.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.detectors import registry
from app.detectors.base import AnalysisContext, DetectorError, MediaKind

VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".avi", ".mkv"}


def auc(real: list[float], fake: list[float]) -> float | None:
    if not real or not fake:
        return None
    wins = sum((f > r) + 0.5 * (f == r) for f in fake for r in real)
    return wins / (len(real) * len(fake))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--no-image", action="store_true", help="disable the image-score jitter component")
    parser.add_argument("--max-clips", type=int, default=None)
    args = parser.parse_args()

    detector = registry.get(MediaKind.VIDEO)
    if detector is None:
        print("video detector disabled by configuration")
        return 1
    options = dict(registry.settings(MediaKind.VIDEO).options)
    if args.max_clips:
        options["max_clips"] = args.max_clips
    if args.no_image:
        options["score_jitter"] = False
    scores: dict[str, list[float]] = {"real": [], "fake": []}
    with tempfile.TemporaryDirectory() as work:
        for label in scores:
            for path in sorted((args.root / label).glob("*")):
                if path.suffix.lower() not in VIDEO_SUFFIXES:
                    continue
                try:
                    result = detector.analyze(path, AnalysisContext(work_dir=Path(work), options=options))
                except DetectorError as exc:
                    print(f"{label}\t{path.name}\tERROR {exc}")
                    continue
                flicker = result.details["components"]["flicker"].get("value")
                print(f"{label}\t{path.name}\trisk={result.risk:.3f}\tflicker={flicker}\t{result.details['elapsed_seconds']}s")
                scores[label].append(result.risk)
    for label, values in scores.items():
        if values:
            print(f"{label}: n={len(values)} mean_risk={sum(values) / len(values):.3f}")
    value = auc(scores["real"], scores["fake"])
    print(f"model={detector.info.name} AUC={'n/a' if value is None else f'{value:.3f}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
