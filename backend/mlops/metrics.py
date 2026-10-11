"""Binary classification metrics for detector evaluation. Pure Python, no sklearn.

Convention: the positive class is ``Fake`` (label 1); ``Real`` is 0. Scores are
detector risks, P(synthetic or manipulated), so higher means "more Fake".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

FAKE = "Fake"
REAL = "Real"


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks of ``values``; tied values share the mean of their ranks."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        shared = (start + end) / 2 + 1
        for position in range(start, end + 1):
            ranks[order[position]] = shared
        start = end + 1
    return ranks


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Area under the ROC curve via the Mann-Whitney U statistic, with tie handling.

    Equals the probability that a random positive scores higher than a random
    negative, counting ties as one half. Returns None unless both classes occur.
    """
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if any(math.isnan(float(score)) for score in scores):
        raise ValueError("scores must not contain NaN")
    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = average_ranks([float(score) for score in scores])
    rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label == 1)
    u_statistic = rank_sum - positives * (positives + 1) / 2
    return u_statistic / (positives * negatives)


@dataclass(frozen=True)
class Confusion:
    """Counts with Fake as the positive class."""

    tp: int = 0  # Fake predicted Fake
    fp: int = 0  # Real predicted Fake
    tn: int = 0  # Real predicted Real
    fn: int = 0  # Fake predicted Real

    @classmethod
    def from_labels(cls, expected: Sequence[str], predicted: Sequence[str]) -> Confusion:
        if len(expected) != len(predicted):
            raise ValueError("expected and predicted must have the same length")
        counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for truth, guess in zip(expected, predicted, strict=True):
            if truth not in (FAKE, REAL) or guess not in (FAKE, REAL):
                raise ValueError(f"labels must be {FAKE!r} or {REAL!r}, got {truth!r}/{guess!r}")
            if truth == FAKE:
                counts["tp" if guess == FAKE else "fn"] += 1
            else:
                counts["fp" if guess == FAKE else "tn"] += 1
        return cls(**counts)

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def metrics(self) -> dict[str, Any]:
        """Precision/recall/F1 for the Fake class, per-class recall, accuracy.

        A metric whose denominator is zero is reported as None rather than 0, so
        an empty class never looks like a measured failure.
        """
        fake_recall = _ratio(self.tp, self.tp + self.fn)
        real_recall = _ratio(self.tn, self.tn + self.fp)
        balanced = (fake_recall + real_recall) / 2 if fake_recall is not None and real_recall is not None else None
        return {
            "samples": self.total,
            "accuracy": _ratio(self.tp + self.tn, self.total),
            "balanced_accuracy": balanced,
            "fake_precision": _ratio(self.tp, self.tp + self.fp),
            "fake_recall": fake_recall,
            "fake_f1": _ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn),
            "real_recall": real_recall,
            "false_positive_rate": _ratio(self.fp, self.fp + self.tn),
            "false_negative_rate": _ratio(self.fn, self.fn + self.tp),
            "confusion_matrix": self.as_matrix(),
        }

    def as_matrix(self) -> dict[str, Any]:
        """Rows are the true class, columns the predicted class, both ordered [Real, Fake]."""
        return {
            "labels": [REAL, FAKE],
            "matrix": [[self.tn, self.fp], [self.fn, self.tp]],
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
        }


def label_at_threshold(risk: float, threshold: float) -> str:
    """Fake when ``risk >= threshold`` (the same convention the detectors use at 0.5)."""
    return FAKE if risk >= threshold else REAL


def binary_report(expected: Sequence[str], predicted: Sequence[str]) -> dict[str, Any]:
    return Confusion.from_labels(expected, predicted).metrics()


def scored_report(expected: Sequence[str], risks: Sequence[float], predicted: Sequence[str], threshold: float | None = None) -> dict[str, Any]:
    """Metrics at the detector's own labels, optionally also at a user threshold, plus ROC AUC."""
    labels = [1 if truth == FAKE else 0 for truth in expected]
    report: dict[str, Any] = {
        "roc_auc": roc_auc(labels, risks),
        "at_detector_threshold": binary_report(expected, predicted),
    }
    if threshold is not None:
        report["at_threshold"] = {"threshold": threshold, **binary_report(expected, [label_at_threshold(r, threshold) for r in risks])}
    return report
