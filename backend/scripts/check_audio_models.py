"""Quick, rerunnable check of the registered audio detectors.

    python scripts/check_audio_models.py --fetch-inthewild      # once: 50 real + 50 fake clips
    python scripts/check_audio_models.py                        # every registered model, one subprocess each
    python scripts/check_audio_models.py --models wav2vec2-legacy

Datasets: ``local`` = backend/test/{real,deepfakes}/*.wav; ``inthewild`` = the
sample written by ``--fetch-inthewild`` (In-the-Wild, Müller et al. 2022, via
SpeechAntiSpoofingBenchmarks/InTheWild at a pinned revision). Per model and
dataset it prints n, real recall, fake recall (threshold 0.5) and ROC AUC, plus
a splice test: real clips concatenated with / without one fake clip inserted,
scored under each aggregate. Results go to stdout as Markdown and to --out JSON.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

ITW_REPO = "SpeechAntiSpoofingBenchmarks/InTheWild"
ITW_REVISION = "f36fbe7e4e009e444a4985c66d76ab1d9439f901"
ITW_SHARD = "data/test-00000-of-00008.parquet"
DEFAULT_DATA = Path(os.getenv("FUSIONTRACE_AUDIO_EVAL_DIR", "C:/FusionTrace/.data/agent-b"))


def fetch_inthewild(data_dir: Path, per_class: int = 50, max_seconds: float = 8.0, seed: int = 0) -> None:
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfFileSystem

    out = data_dir / "inthewild"
    fs = HfFileSystem()
    pf = pq.ParquetFile(fs.open(f"datasets/{ITW_REPO}@{ITW_REVISION}/{ITW_SHARD}"))
    groups = list(range(pf.metadata.num_row_groups))
    random.Random(seed).shuffle(groups)
    picked: dict[int, list[dict]] = {0: [], 1: []}  # 0 = bonafide, 1 = spoof
    for group in groups:
        for row in pf.read_row_group(group).to_pylist():
            label = int(row["label"])
            if len(picked[label]) >= per_class:
                continue
            data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if len(data) / sr > max_seconds:
                continue
            notes = json.loads(row["notes"] or "{}")
            sub = out / ("fake" if label else "real")
            sub.mkdir(parents=True, exist_ok=True)
            name = f"{notes.get('utterance_id', row['path'])}.wav".replace("/", "_")
            sf.write(str(sub / name), data, sr, subtype="PCM_16")
            picked[label].append({"file": f"{sub.name}/{name}", "speaker": notes.get("speaker"), "seconds": round(len(data) / sr, 2)})
        if all(len(v) >= per_class for v in picked.values()):
            break
    manifest = {"dataset": ITW_REPO, "revision": ITW_REVISION, "shard": ITW_SHARD, "seed": seed, "max_seconds": max_seconds, "real": picked[0], "fake": picked[1]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {len(picked[0])} real + {len(picked[1])} fake clips to {out}")


def datasets(data_dir: Path) -> dict[str, list[tuple[Path, int]]]:
    result = {"local": [(p, 0) for p in sorted((BACKEND / "test" / "real").glob("*.wav"))] + [(p, 1) for p in sorted((BACKEND / "test" / "deepfakes").glob("*.wav"))]}
    itw = data_dir / "inthewild"
    if itw.exists():
        result["inthewild"] = [(p, 0) for p in sorted((itw / "real").glob("*.wav"))] + [(p, 1) for p in sorted((itw / "fake").glob("*.wav"))]
    return result


def metrics(rows: list[tuple[int, float]]) -> dict:
    real = [r for y, r in rows if y == 0]
    fake = [r for y, r in rows if y == 1]
    out = {"n_real": len(real), "n_fake": len(fake),
           "real_recall": round(sum(r < 0.5 for r in real) / len(real), 3) if real else None,
           "fake_recall": round(sum(r >= 0.5 for r in fake) / len(fake), 3) if fake else None, "auc": None}
    if real and fake:
        from sklearn.metrics import roc_auc_score

        out["auc"] = round(float(roc_auc_score([y for y, _ in rows], [r for _, r in rows])), 3)
    return out


def splice_test(detector, clips: list[tuple[Path, int]], work: Path, pairs: int = 10, seed: int = 0) -> dict:
    import numpy as np
    import soundfile as sf

    from app.detectors.audio.windows import aggregate
    from app.detectors.base import AnalysisContext

    rng = random.Random(seed)
    real = [p for p, y in clips if y == 0]
    fake = [p for p, y in clips if y == 1]
    rows: dict[str, list[tuple[int, float]]] = {"mean": [], "max": [], "top_k_mean": []}
    for i in range(pairs):
        parts = [sf.read(str(p), dtype="float32")[0] for p in rng.sample(real, 5)]
        for label in (0, 1):
            seq = list(parts)
            if label:
                seq.insert(rng.randrange(1, 5), sf.read(str(rng.choice(fake)), dtype="float32")[0])
            path = work / f"splice-{i}-{label}.wav"
            sf.write(str(path), np.concatenate(seq), 16000)
            scores = [w["risk"] for w in detector.analyze(path, AnalysisContext(work_dir=work)).details["per_window_scores"]]
            for method in rows:
                rows[method].append((label, aggregate(scores, method, 3)))
    return {method: metrics(r) for method, r in rows.items()}


def run_model(name: str, data_dir: Path) -> dict:
    import torch

    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))
    from app.detectors import registry
    from app.detectors.audio import DETECTORS
    from app.detectors.base import AnalysisContext, DetectorError

    detector = DETECTORS[name](model_dir=registry.model_dir(), device="cpu")
    if not detector.is_available():
        return {"model": name, "error": "weights not available"}
    detector.load()
    report: dict = {"model": name, "datasets": {}}
    windows = seconds = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        sets = datasets(data_dir)
        for dataset, clips in sets.items():
            rows, errors = [], 0
            for path, label in clips:
                try:
                    result = detector.analyze(path, AnalysisContext(work_dir=work))
                except DetectorError:
                    errors += 1
                    continue
                rows.append((label, result.risk))
                windows += result.details["window_count"]
                seconds += result.details["inference_seconds"]
            report["datasets"][dataset] = {**metrics(rows), "errors": errors}
        if "inthewild" in sets:
            report["splice_inthewild"] = splice_test(detector, sets["inthewild"], work)
    report["seconds_per_window"] = round(seconds / windows, 3) if windows else None
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fetch-inthewild", action="store_true")
    parser.add_argument("--models", default="", help="comma-separated registry keys (default: all registered)")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.fetch_inthewild:
        fetch_inthewild(args.data_dir)
        return
    if args.worker:
        print(json.dumps(run_model(args.worker, args.data_dir)))
        return
    from app.detectors.audio import DETECTORS

    names = [n for n in args.models.split(",") if n] or list(DETECTORS)
    reports = []
    for name in names:  # one short-lived process per model keeps peak RAM to one model
        started = time.time()
        done = subprocess.run([sys.executable, __file__, "--worker", name, "--data-dir", str(args.data_dir)], capture_output=True, text=True, cwd=BACKEND)
        lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
        reports.append(json.loads(lines[-1]) if lines else {"model": name, "error": done.stderr[-500:]})
        print(f"{name}: {time.time() - started:.0f}s", file=sys.stderr)
    print("| model | dataset | n real / fake | real recall | fake recall | AUC | errors |\n|---|---|---|---|---|---|---|")
    for rep in reports:
        for dataset, m in (rep.get("datasets") or {}).items():
            print(f"| {rep['model']} | {dataset} | {m['n_real']} / {m['n_fake']} | {m['real_recall']} | {m['fake_recall']} | {m['auc']} | {m['errors']} |")
        if "error" in rep:
            print(f"| {rep['model']} | - | error: {rep['error'][:80]} | | | | |")
    print("\nSplice test (In-the-Wild, 10 real-only vs 10 with one fake clip inserted):\n\n| model | aggregate | real recall | fake recall | AUC |\n|---|---|---|---|---|")
    for rep in reports:
        for method, m in (rep.get("splice_inthewild") or {}).items():
            print(f"| {rep['model']} | {method} | {m['real_recall']} | {m['fake_recall']} | {m['auc']} |")
    print("\nSeconds per window (CPU, OMP threads as set):", {r["model"]: r.get("seconds_per_window") for r in reports})
    if args.out:
        args.out.write_text(json.dumps(reports, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
