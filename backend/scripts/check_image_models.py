"""Quick comparison of the registered image detectors (workstream A).

Two labelled sets:

* local  — ``backend/test/{real,deepfakes}`` images (small; probably related
  to the legacy model's training data, so it cannot show the generalisation gap).
* public — an out-of-distribution sample built by ``--fetch-public`` from two
  pinned Hugging Face datasets (no other source is used):
    real: HuggingFaceM4/FairFace (CC-BY-4.0), config "1.25", validation split
    fake: LucaCorvitto/DeepFake-Faces-Stimuli (MIT), Stable Diffusion
          Realistic Vision 5.1 faces
  Both classes get identical processing (centre square crop, resize to
  256x256, JPEG q=90) so format/resolution cannot separate them.

Each model runs in its own short-lived subprocess (one model in RAM at a time).
Inference is offline; nothing is uploaded.

    python scripts/check_image_models.py --fetch-public            # once
    python scripts/check_image_models.py                           # all models
    python scripts/check_image_models.py --models commfor-vit-s384,efficientnetv2-legacy
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_DATA_DIR = Path(os.getenv("FUSIONTRACE_CHECK_DATA_DIR", "C:/FusionTrace/.data/agent-a"))
REAL_DATASET = ("HuggingFaceM4/FairFace", "54d573cdb8b5af490ba8da9da2799628f6e5c496", "1.25", "validation")
FAKE_DATASET = ("LucaCorvitto/DeepFake-Faces-Stimuli", "28bd8dc650b0fa436205c93a7261855130f3009e", "generated/")
# Extra model variants evaluated alongside the registry: name -> (registry key, options)
VARIANTS: dict[str, tuple[str, dict[str, Any]]] = {
    "efficientnetv2-legacy[half-norm]": ("efficientnetv2-legacy", {"normalization": "half"}),
}


# ── data ────────────────────────────────────────────────────────────────────


def _normalise_image(image: Any, out_path: Path, size: int = 256) -> None:
    from PIL import Image

    image = image.convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="JPEG", quality=90)


def fetch_public(data_dir: Path, per_class: int, seed: int) -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image

    out = data_dir / "public"
    raw = data_dir / "raw"
    manifest: dict[str, Any] = {"per_class": per_class, "seed": seed, "processing": "centre square crop, 256x256 LANCZOS, JPEG q=90"}

    repo, revision, config, split = REAL_DATASET
    stream = load_dataset(repo, config, split=split, streaming=True, revision=revision).shuffle(seed=seed, buffer_size=1000)
    for index, row in enumerate(stream.take(per_class)):
        _normalise_image(row["image"], out / "real" / f"fairface_{index:03d}.jpg")
    manifest["real"] = {"dataset": repo, "revision": revision, "config": config, "split": split, "count": per_class,
                        "selection": f"streaming shuffle(seed={seed}, buffer_size=1000).take({per_class})"}

    repo, revision, prefix = FAKE_DATASET
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=revision) if f.startswith(prefix))
    chosen = random.Random(seed).sample(files, per_class)
    for index, name in enumerate(chosen):
        local = hf_hub_download(repo, name, repo_type="dataset", revision=revision, local_dir=str(raw / "stimuli"))
        with Image.open(local) as image:
            _normalise_image(image, out / "fake" / f"stimuli_{index:03d}.jpg")
    manifest["fake"] = {"dataset": repo, "revision": revision, "files": chosen}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"public sample written to {out}")


def labelled_sets(data_dir: Path) -> dict[str, list[tuple[str, int]]]:
    def images(folder: Path) -> list[Path]:
        return sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS) if folder.is_dir() else []

    sets = {"local": [(str(p), 0) for p in images(PROJECT_DIR / "test" / "real")]
             + [(str(p), 1) for p in images(PROJECT_DIR / "test" / "deepfakes")]}
    public = [(str(p), 0) for p in images(data_dir / "public" / "real")] + [(str(p), 1) for p in images(data_dir / "public" / "fake")]
    if public:
        sets["public"] = public
    return sets


# ── metrics ─────────────────────────────────────────────────────────────────


def auc(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney AUC with tie handling; None when a class is missing."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def metrics(scores: list[float], labels: list[int], threshold: float) -> dict[str, Any]:
    fake = [s for s, y in zip(scores, labels) if y == 1]
    real = [s for s, y in zip(scores, labels) if y == 0]
    return {
        "n_fake": len(fake),
        "n_real": len(real),
        "fake_recall": sum(s >= threshold for s in fake) / len(fake) if fake else None,
        "real_recall": sum(s < threshold for s in real) / len(real) if real else None,
        "auc": auc(scores, labels),
    }


# ── workers ─────────────────────────────────────────────────────────────────


def worker(name: str, data_dir: Path, out_path: Path) -> None:
    """Score every image with one model (runs in its own process)."""
    import time

    import torch

    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))
    from app.detectors import registry
    from app.detectors.base import AnalysisContext, DetectorError
    from app.detectors.image import DETECTORS

    key, options = VARIANTS.get(name, (name, {}))
    detector = DETECTORS[key](model_dir=registry.model_dir(), options=options)
    if not detector.is_available():
        out_path.write_text(json.dumps({"model": name, "error": "weights not available"}), encoding="utf-8")
        return
    start = time.perf_counter()
    detector.load()
    load_seconds = time.perf_counter() - start
    ctx = AnalysisContext(work_dir=data_dir)
    result: dict[str, Any] = {"model": name, "load_seconds": round(load_seconds, 1), "threshold": detector.threshold(None), "sets": {}}
    for set_name, items in labelled_sets(data_dir).items():
        scores, labels, errors = [], [], 0
        start = time.perf_counter()
        for path, label in items:
            try:
                scores.append(detector.analyze(Path(path), ctx).risk)
                labels.append(label)
            except DetectorError:
                errors += 1
        elapsed = time.perf_counter() - start
        result["sets"][set_name] = {**metrics(scores, labels, result["threshold"]), "errors": errors,
                                    "seconds_per_image": round(elapsed / max(1, len(items)), 3),
                                    "scores": [[Path(p).name, round(s, 4), y] for (p, _), s, y in zip(items, scores, labels)]}
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> None:
    from app.detectors.image import DETECTORS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--models", default=",".join([*DETECTORS, *VARIANTS]))
    parser.add_argument("--fetch-public", action="store_true", help="download the public sample, then exit")
    parser.add_argument("--per-class", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker(args.worker, args.data_dir, args.worker_out)
        return
    if args.fetch_public:
        fetch_public(args.data_dir, args.per_class, args.seed)
        return

    results_dir = args.data_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        out = results_dir / f"{name.replace('[', '_').replace(']', '')}.json"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", name, "--worker-out", str(out),
                        "--data-dir", str(args.data_dir)], check=True, cwd=PROJECT_DIR)
        summary.append(json.loads(out.read_text(encoding="utf-8")))

    print("| model | set | n fake/real | fake recall | real recall | AUC | s/img |")
    print("|---|---|---|---|---|---|---|")
    for item in summary:
        if "error" in item:
            print(f"| {item['model']} | - | - | {item['error']} | | | |")
            continue
        for set_name, m in item["sets"].items():
            print(f"| {item['model']} | {set_name} | {m['n_fake']}/{m['n_real']} | {_fmt(m['fake_recall'])} | "
                  f"{_fmt(m['real_recall'])} | {_fmt(m['auc'])} | {m['seconds_per_image']} |")
    (results_dir / "summary.json").write_text(
        json.dumps([{k: v for k, v in item.items()} for item in summary], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
