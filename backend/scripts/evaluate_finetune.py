"""Compare the base image model with a LoRA fine-tune on labelled test folders.

    python scripts/evaluate_finetune.py --adapter runs/lora-1 \\
        --set faceswap=data/test_faceswap/real:data/test_faceswap/fake \\
        --set ai_faces=data/test_public/real:data/test_public/fake

Each ``--set NAME=REAL_DIR:FAKE_DIR`` is one test set. The fine-tuned model uses the
threshold stored with the adapter (chosen on its validation split); the base model
uses 0.5, its shipped threshold. Reports AUC, fake recall and real recall per set,
as Markdown (stdout) and JSON (--out). Test sets must not be used for training or
threshold choice; this script never writes a threshold back.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from finetune import data, model  # noqa: E402
from mlops.metrics import roc_auc  # noqa: E402


def parse_set(text: str) -> tuple[str, Path, Path]:
    name, _, dirs = text.partition("=")
    real, _, fake = dirs.partition(":")
    if not (name and real and fake):
        raise argparse.ArgumentTypeError(f"expected NAME=REAL_DIR:FAKE_DIR, got {text!r}")
    return name, Path(real), Path(fake)


def rows_for(name: str, real: Path, fake: Path) -> list[data.Row]:
    rows = [data.Row(p, 0, p.name, "test") for p in data._images(real)] + [data.Row(p, 1, p.name, "test") for p in data._images(fake)]
    if not any(r.label == 1 for r in rows) or not any(r.label == 0 for r in rows):
        raise ValueError(f"test set {name!r} needs both real and fake images")
    return rows


def report(risks: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | None]:
    pred = risks >= threshold
    return {
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "threshold": threshold,
        "auc": roc_auc(labels.tolist(), risks.tolist()),
        "fake_recall": float(pred[labels == 1].mean()),
        "real_recall": float((~pred[labels == 0]).mean()),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", type=Path, required=True, help="run folder from finetune_lora.py (contains adapter/ and threshold.json)")
    parser.add_argument("--set", dest="sets", action="append", type=parse_set, required=True)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    stored = json.loads((args.adapter / "threshold.json").read_text(encoding="utf-8"))
    tuned_threshold = float(stored["threshold"])

    base = model.build_base(args.weights).to(device).eval()
    tuned = model.load_adapter(args.adapter / "adapter", args.weights).to(device).eval()

    results: dict[str, dict] = {}
    for name, real, fake in args.sets:
        rows = rows_for(name, real, fake)
        loader = DataLoader(data.ManifestDataset(rows, data.eval_transform()), batch_size=args.batch_size, num_workers=args.workers)
        base_risk, labels = model.score(base, loader, device)
        tuned_risk, _ = model.score(tuned, loader, device)
        results[name] = {
            "base": report(base_risk, labels, 0.5),
            "finetuned": report(tuned_risk, labels, tuned_threshold),
        }

    lines = ["| test set | model | threshold | AUC | fake recall | real recall | n real / fake |", "|---|---|---|---|---|---|---|"]
    for name, pair in results.items():
        for label, key in (("base (v2 default)", "base"), ("LoRA fine-tune", "finetuned")):
            r = pair[key]
            auc = "n/a" if r["auc"] is None else f"{r['auc']:.3f}"
            lines.append(f"| {name} | {label} | {r['threshold']:.4f} | {auc} | {r['fake_recall']:.3f} | {r['real_recall']:.3f} | {r['n_real']} / {r['n_fake']} |")
    table = "\n".join(lines)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"adapter": str(args.adapter), "results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
