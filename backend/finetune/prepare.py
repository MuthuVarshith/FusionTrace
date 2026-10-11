"""Build the fine-tuning dataset: FairFace reals, bitmind face-swap fakes, near-duplicate groups.

Sources and licences (recorded in the manifest's README, docs/how-to-finetune.md):

* FairFace (``HuggingFaceM4/FairFace``, validation split, 0.25 padding): CC BY 4.0.
  Real faces. Attribution: Kärkkäinen & Joo, "FairFace: Face Attribute Dataset for
  Balanced Race, Gender, and Age", WACV 2021.
* bitmind/face-swap (train, validation, test parquet): Apache-2.0. Face-swap fakes.
  Same source and method for every split: any result on it is "same-source only".

Leakage control (reduces, does not remove, identity leakage): images are hashed with a
64-bit difference hash (dHash). Images within Hamming distance ``--dup-distance`` are
linked, and linked images form one group. Groups are split, never single images, so
near-duplicates (the same face at different crops or resolutions) cannot straddle splits.
Faces of the same person with different images are not detected by dHash; identity
leakage is only partly reduced.

Usage (from backend/)::

    python -m finetune.prepare --out data --reals 5000 --seed 13
"""

from __future__ import annotations

import argparse
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from finetune import data

FAIRFACE_REPO = "HuggingFaceM4/FairFace"
FAIRFACE_FILE = "0.25/validation-00000-of-00001-951dbd63c8724ee1.parquet"
FAIRFACE_LICENCE = "CC-BY-4.0"
FACESWAP_REPO = "bitmind/face-swap"
FACESWAP_FILES = ("data/face_train.parquet", "data/face_validation.parquet", "data/face_test.parquet")
FACESWAP_LICENCE = "Apache-2.0"
REGRESSION_FAKE_REPO = "LucaCorvitto/DeepFake-Faces-Stimuli"
REGRESSION_FAKE_REVISION = "28bd8dc650b0fa436205c93a7261855130f3009e"
REGRESSION_FAKE_PREFIX = "generated/"
REGRESSION_FAKE_COUNT = 90
REGRESSION_REAL_COUNT = 90


@dataclass(frozen=True)
class Sample:
    name: str
    label: int
    payload: bytes
    source: str


def _cache() -> str:
    return "C:/FusionTrace/.cache/huggingface/hub"


def _image_bytes(cell) -> bytes:
    if isinstance(cell, dict):
        return cell["bytes"]
    return cell


def load_fairface(count: int, seed: int) -> list[Sample]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(FAIRFACE_REPO, FAIRFACE_FILE, repo_type="dataset", cache_dir=_cache())
    table = pq.read_table(path, columns=["image"]).column("image").to_pylist()
    picked = random.Random(seed).sample(range(len(table)), min(count, len(table)))
    return [Sample(f"fairface_{index:05d}.jpg", 0, _image_bytes(table[index]), "fairface") for index in sorted(picked)]


def load_faceswap() -> list[Sample]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    samples: list[Sample] = []
    for filename in FACESWAP_FILES:
        path = hf_hub_download(FACESWAP_REPO, filename, repo_type="dataset", cache_dir=_cache())
        table = pq.read_table(path).to_pylist()
        split = Path(filename).stem.removeprefix("face_")
        for row in table:
            samples.append(Sample(f"faceswap_{split}_{Path(row['filename']).stem}.jpg", 1, row["image"], f"faceswap-{split}"))
    return samples


def dhash(payload: bytes, size: int = 8) -> int:
    """64-bit difference hash: compare horizontally adjacent grey pixels on a 9x8 grid."""
    with Image.open(io.BytesIO(payload)) as image:
        grey = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(grey, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def cluster(hashes: list[int], max_distance: int) -> list[int]:
    """Union-find over pairs within ``max_distance`` Hamming bits. Returns a group index per item."""
    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    values = np.array(hashes, dtype=np.uint64)
    chunk_size = 256  # bounds memory: chunk x n x 8 bytes of popcount temporaries
    for start in range(0, len(values), chunk_size):
        chunk = values[start:start + chunk_size]
        distance = _popcount(np.bitwise_xor(chunk[:, None], values[None, :]))
        for i_local, j in zip(*np.nonzero(distance <= max_distance), strict=True):
            i = start + int(i_local)
            if i < j:
                parent[find(i)] = find(int(j))
    roots: dict[int, int] = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(len(hashes))]


def _popcount(values: np.ndarray) -> np.ndarray:
    as_bytes = values.view(np.uint8).reshape((*values.shape, 8))
    table = np.array([bin(b).count("1") for b in range(256)], dtype=np.int16)
    return table[as_bytes].sum(axis=-1)


def assign_splits(groups: list[int], fractions: tuple[float, float, float], seed: int) -> dict[int, str]:
    unique = sorted(set(groups))
    random.Random(seed).shuffle(unique)
    train_end = round(fractions[0] * len(unique))
    val_end = train_end + round(fractions[1] * len(unique))
    return {g: ("train" if i < train_end else "val" if i < val_end else "test") for i, g in enumerate(unique)}


def build(out: Path, reals: int, seed: int, dup_distance: int, fractions: tuple[float, float, float]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    samples = load_fairface(reals, seed) + load_faceswap()
    hashes = [dhash(s.payload) for s in samples]
    groups = cluster(hashes, dup_distance)
    split_of = assign_splits(groups, fractions, seed)

    images = out / "images"
    rows: list[data.Row] = []
    for sample, group in zip(samples, groups, strict=True):
        klass = "fake" if sample.label else "real"
        target = images / klass / sample.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(sample.payload)
        rows.append(data.Row(target.resolve(), sample.label, f"g{group}", split_of[group]))
    data.check_disjoint(rows)
    data.check_splits_present(rows, splits=data.SPLITS)
    manifest = out / "manifest.csv"
    data.write_manifest(rows, manifest)

    stats = {
        "rows": {s: {"real": sum(1 for r in rows if r.split == s and r.label == 0), "fake": sum(1 for r in rows if r.split == s and r.label == 1)} for s in data.SPLITS},
        "groups": len(set(groups)),
        "near_duplicate_distance": dup_distance,
        "data_version": data.content_hash(rows),
        "seed": seed,
    }
    (out / "dataset.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_sources(out, samples)
    return stats


def load_regression_fakes() -> list[bytes]:
    """Stable Diffusion faces (MIT), not used in training. Same source and revision as check_image_models.py."""
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(
        f for f in HfApi().list_repo_files(REGRESSION_FAKE_REPO, repo_type="dataset", revision=REGRESSION_FAKE_REVISION)
        if f.startswith(REGRESSION_FAKE_PREFIX)
    )[:REGRESSION_FAKE_COUNT]
    return [Path(hf_hub_download(REGRESSION_FAKE_REPO, f, repo_type="dataset", revision=REGRESSION_FAKE_REVISION, cache_dir=_cache())).read_bytes() for f in files]


def build_eval(out: Path, seed: int) -> dict[str, int]:
    """Evaluation folders from the manifest's **test** rows only. Never used for training or thresholds."""
    import shutil

    rows = data.read_manifest(out / "manifest.csv")
    test = [r for r in rows if r.split == "test"]
    test_real = [r for r in test if r.label == 0]
    test_fake = [r for r in test if r.label == 1]
    root = out / "eval"

    def fill(folder: Path, items: list[tuple[Path, int]]) -> None:
        for sub in ("real", "fake"):
            shutil.rmtree(folder / sub, ignore_errors=True)
            (folder / sub).mkdir(parents=True, exist_ok=True)
        for index, (source, label) in enumerate(items):
            target = folder / ("real" if label == 0 else "fake") / f"{index:04d}{source.suffix}"
            shutil.copy(source, target)

    fill(root / "faceswap_same_source", [(r.path, r.label) for r in test_real + test_fake])
    picked_real = random.Random(seed).sample(test_real, min(REGRESSION_REAL_COUNT, len(test_real)))
    fakes = out / "regression_fakes"
    fakes.mkdir(parents=True, exist_ok=True)
    fake_paths = []
    for index, payload in enumerate(load_regression_fakes()):
        path = fakes / f"luca_{index:03d}.jpg"
        path.write_bytes(payload)
        fake_paths.append(path)
    fill(root / "ai_faces_regression", [(r.path, 0) for r in picked_real] + [(p, 1) for p in fake_paths])
    return {"faceswap_real": len(test_real), "faceswap_fake": len(test_fake), "regression_real": len(picked_real), "regression_fake": len(fake_paths)}


def write_sources(out: Path, samples: list[Sample]) -> None:
    text = f"""# Dataset sources and licences

| Source | Used for | Licence | Attribution |
|---|---|---|---|
| {FAIRFACE_REPO} (validation parquet, 0.25 padding) | real faces ({sum(1 for s in samples if s.source == 'fairface')} images) | {FAIRFACE_LICENCE} | Kärkkäinen & Joo, FairFace, WACV 2021 |
| {FACESWAP_REPO} (train, validation, test) | face-swap fakes ({sum(1 for s in samples if s.label)} images) | {FACESWAP_LICENCE} | bitmind |

Every face-swap result from this dataset is same-source only: no other face-swap set has been
licence-checked yet.
"""
    (out / "SOURCES.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--reals", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dup-distance", type=int, default=6, help="dHash Hamming distance that links near-duplicates")
    parser.add_argument("--fractions", default="0.8,0.1,0.1")
    parser.add_argument("--eval-only", action="store_true", help="rebuild only the eval folders from an existing manifest")
    args = parser.parse_args(argv)
    if args.eval_only:
        print(json.dumps(build_eval(args.out, args.seed), indent=2))
        return 0
    fractions = tuple(float(x) for x in args.fractions.split(","))
    stats = build(args.out, args.reals, args.seed, args.dup_distance, fractions)
    stats["eval"] = build_eval(args.out, args.seed)
    (args.out / "dataset.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
