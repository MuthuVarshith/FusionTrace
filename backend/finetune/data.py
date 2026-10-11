"""Manifests, group-disjoint splits and image datasets for fine-tuning.

A manifest is a CSV with columns::

    path,label,group,split

* ``label``: 1 = fake, 0 = real.
* ``group``: the unit that must never straddle two splits: a person (identity)
  for face swaps, or a source video when identities are unknown. Rows that
  share a group are assumed to be near-duplicates.
* ``split``: ``train``, ``val`` or ``test``.

``check_disjoint`` refuses to train when any group appears in two splits, so a
leaky split fails loudly instead of inflating the scores.
"""

from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from app.detectors.image.commfor import IMAGENET_MEAN, IMAGENET_STD

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "test")
FIELDS = ("path", "label", "group", "split")


@dataclass(frozen=True)
class Row:
    path: Path
    label: int
    group: str
    split: str


def read_manifest(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for line_no, raw in enumerate(csv.DictReader(handle), start=2):
            if raw["label"] not in ("0", "1"):
                raise ValueError(f"{path}:{line_no}: label must be 0 or 1, got {raw['label']!r}")
            if raw["split"] not in SPLITS:
                raise ValueError(f"{path}:{line_no}: split must be one of {SPLITS}, got {raw['split']!r}")
            rows.append(Row(Path(raw["path"]), int(raw["label"]), raw["group"], raw["split"]))
    return rows


def write_manifest(rows: Iterable[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for row in rows:
            writer.writerow([str(row.path), row.label, row.group, row.split])


def check_disjoint(rows: Sequence[Row]) -> None:
    """Raise if any group appears in more than one split."""
    owners: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        owners[row.group].add(row.split)
    leaked = {group: sorted(splits) for group, splits in owners.items() if len(splits) > 1}
    if leaked:
        sample = dict(list(leaked.items())[:5])
        raise ValueError(f"{len(leaked)} groups appear in more than one split (e.g. {sample}); fix the split")


def check_splits_present(rows: Sequence[Row], splits: Sequence[str] = ("train", "val")) -> None:
    """Every requested split needs both classes. Training needs train and val; test is only for evaluation."""
    counts = Counter((row.split, row.label) for row in rows)
    for split in splits:
        for label in (0, 1):
            if counts[(split, label)] == 0:
                raise ValueError(f"split {split!r} has no {'fake' if label else 'real'} images")


def content_hash(rows: Sequence[Row]) -> str:
    """Stable hash of the manifest contents: the data version recorded in MLflow."""
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda r: (str(r.path), r.split)):
        digest.update(f"{row.path}|{row.label}|{row.group}|{row.split}\n".encode())
    return digest.hexdigest()


def group_split_from_folders(real_dir: Path, fake_dir: Path, *, seed: int, fractions: tuple[float, float, float], group_of) -> list[Row]:
    """Assign rows to splits by group, so no group spans two splits.

    ``group_of(path)`` returns the group key. Groups are shuffled with ``seed`` and
    dealt into train/val/test by ``fractions`` of the groups (not of the images).
    """
    rows = [(p, 0, group_of(p)) for p in _images(real_dir)] + [(p, 1, group_of(p)) for p in _images(fake_dir)]
    groups = sorted({group for _, _, group in rows})
    random.Random(seed).shuffle(groups)
    train_end = round(fractions[0] * len(groups))
    val_end = train_end + round(fractions[1] * len(groups))
    assignment = {}
    for index, group in enumerate(groups):
        assignment[group] = "train" if index < train_end else "val" if index < val_end else "test"
    return [Row(path, label, group, assignment[group]) for path, label, group in rows]


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"not a folder: {folder}")
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def eval_transform() -> transforms.Compose:
    """Same preprocessing as the detector at inference (commfor.py)."""
    return transforms.Compose(
        [transforms.Resize(440), transforms.CenterCrop(384), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(440),
            transforms.RandomCrop(384),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ManifestDataset(Dataset):
    def __init__(self, rows: Sequence[Row], transform) -> None:
        self.rows = list(rows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with Image.open(row.path) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, torch.tensor(float(row.label))
