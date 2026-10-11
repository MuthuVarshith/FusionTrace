"""Unit tests for the fine-tuning split rules and threshold choice (no model weights needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from finetune import data  # noqa: E402
from scripts.finetune_lora import pick_threshold  # noqa: E402


def row(name: str, label: int, group: str, split: str) -> data.Row:
    return data.Row(Path(name), label, group, split)


def test_leaky_group_across_splits_is_rejected():
    rows = [row("a.jpg", 1, "person-1", "train"), row("b.jpg", 1, "person-1", "val")]
    with pytest.raises(ValueError, match="more than one split"):
        data.check_disjoint(rows)


def test_group_disjoint_splits_pass():
    rows = [row("a.jpg", 1, "p1", "train"), row("b.jpg", 0, "p2", "val"), row("c.jpg", 1, "p3", "test")]
    data.check_disjoint(rows)


def test_training_splits_need_both_classes():
    rows = [row("a.jpg", 1, "p1", "train"), row("b.jpg", 0, "p2", "train"), row("c.jpg", 1, "p3", "val")]
    with pytest.raises(ValueError, match="no real images"):
        data.check_splits_present(rows)


def test_group_split_keeps_groups_together(tmp_path: Path):
    real, fake = tmp_path / "real", tmp_path / "fake"
    for person in range(6):
        folder = real / f"id{person}"
        folder.mkdir(parents=True)
        (folder / "x.jpg").write_bytes(b"")
        (folder / "y.jpg").write_bytes(b"")
        fake_folder = fake / f"id{person}"
        fake_folder.mkdir(parents=True)
        (fake_folder / "x.jpg").write_bytes(b"")
    rows = data.group_split_from_folders(
        real, fake, seed=1, fractions=(0.5, 0.25, 0.25),
        group_of=lambda p: p.parent.name,
    )
    data.check_disjoint(rows)
    assert {r.split for r in rows} == {"train", "val", "test"}


def test_threshold_is_chosen_for_balanced_accuracy():
    # Fakes score high, reals low, with a gap: any threshold in the gap separates them.
    risks = np.array([0.05, 0.1, 0.2, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    threshold, metrics = pick_threshold(risks, labels)
    assert 0.2 < threshold <= 0.7
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)


def test_threshold_follows_the_data_not_a_fixed_0_5():
    # Scores all sit low: a fixed 0.5 would catch nothing; validation picks a lower cut.
    risks = np.array([0.0001, 0.0002, 0.0003, 0.0010, 0.0020, 0.0030])
    labels = np.array([0, 0, 0, 1, 1, 1])
    threshold, metrics = pick_threshold(risks, labels)
    assert threshold < 0.5
    assert metrics["fake_recall"] == pytest.approx(1.0)
