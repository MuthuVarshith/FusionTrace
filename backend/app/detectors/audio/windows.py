"""Window planning, options and score aggregation for chunked audio analysis.

Pure Python + NumPy (no torch) so it is cheap to import and easy to test.

Every audio detector scores a clip as a sequence of fixed-length windows
(``window_seconds`` long, ``hop_seconds`` apart, covering the whole clip) and
reduces the per-window risks to one clip risk with ``aggregate``. See
``docs/model-choice-audio.md`` for the reasoning behind the defaults.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

AGGREGATES: tuple[str, ...] = ("mean", "max", "top_k_mean")


@dataclass(frozen=True)
class ChunkOptions:
    """Detector options (config ``detectors.audio.options``) with their defaults."""

    window_seconds: float = 4.0  # the legacy model was trained on 4 s clips
    hop_seconds: float = 2.0  # 50% overlap: a 4 s event always falls mostly inside one window
    max_windows: int = 60  # cap on model forwards per clip; windows are spread evenly when exceeded
    aggregate: str = "top_k_mean"  # "mean" | "max" | "top_k_mean"; sets DetectorResult.risk
    top_k: int = 3  # windows averaged by "top_k_mean"
    threshold: float = 0.5  # risk at or above which the label is "Fake"
    silence_dbfs: float = -60.0  # windows quieter than this RMS level are skipped as silence
    batch_size: int = 1  # windows per model forward; the deadline is checked between batches

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any] | None) -> ChunkOptions:
        """Build from a config mapping, ignoring unknown keys. Raises ``ValueError`` on bad values."""
        options = options or {}
        known = {f.name: f for f in fields(cls)}
        values: dict[str, Any] = {}
        for key, field_ in known.items():
            if key not in options or options[key] is None:
                continue
            raw = options[key]
            try:
                if field_.type in ("int", int):
                    if isinstance(raw, bool) or float(raw) != int(float(raw)):
                        raise ValueError
                    values[key] = int(float(raw))
                elif field_.type in ("float", float):
                    values[key] = float(raw)
                else:
                    values[key] = str(raw).strip().lower()
            except (TypeError, ValueError):
                raise ValueError(f"audio option {key!r} has an invalid value: {raw!r}") from None
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        checks = [
            (0.5 <= self.window_seconds <= 30.0, "window_seconds must be between 0.5 and 30"),
            (0.1 <= self.hop_seconds <= 60.0, "hop_seconds must be between 0.1 and 60"),
            (1 <= self.max_windows <= 1000, "max_windows must be between 1 and 1000"),
            (self.aggregate in AGGREGATES, f"aggregate must be one of {AGGREGATES}"),
            (self.top_k >= 1, "top_k must be at least 1"),
            (0.0 < self.threshold < 1.0, "threshold must be between 0 and 1"),
            (self.silence_dbfs <= 0.0, "silence_dbfs must be <= 0"),
            (1 <= self.batch_size <= 64, "batch_size must be between 1 and 64"),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)


def plan_windows(num_samples: int, window: int, hop: int, max_windows: int) -> tuple[list[tuple[int, int]], bool]:
    """Sample spans ``(start, end)`` covering ``num_samples`` samples.

    * A clip no longer than one window yields a single span ``(0, num_samples)``;
      the caller pads it to ``window`` samples.
    * Otherwise windows start every ``hop`` samples, plus one final window
      aligned to the end of the clip so the tail is always covered. Every span is
      exactly ``window`` samples.
    * When that gives more than ``max_windows`` spans, ``max_windows`` of them
      are kept, evenly spaced across the clip (first and last always kept).

    Returns ``(spans, capped)``.
    """
    if num_samples <= 0:
        return [], False
    if window <= 0 or hop <= 0 or max_windows <= 0:
        raise ValueError("window, hop and max_windows must be positive")
    if num_samples <= window:
        return [(0, num_samples)], False
    starts = list(range(0, num_samples - window + 1, hop))
    if starts[-1] + window < num_samples:
        starts.append(num_samples - window)
    capped = len(starts) > max_windows
    if capped:
        if max_windows == 1:
            picks = [0]
        else:
            picks = sorted({round(i * (len(starts) - 1) / (max_windows - 1)) for i in range(max_windows)})
        starts = [starts[i] for i in picks]
    return [(start, start + window) for start in starts], capped


def aggregate(scores: Sequence[float], method: str, top_k: int = 3) -> float:
    """Reduce per-window risks to one clip risk."""
    if not scores:
        raise ValueError("no window scores to aggregate")
    values = np.asarray(scores, dtype=np.float64)
    if method == "mean":
        return float(values.mean())
    if method == "max":
        return float(values.max())
    if method == "top_k_mean":
        k = max(1, min(int(top_k), values.size))
        return float(np.sort(values)[-k:].mean())
    raise ValueError(f"unknown aggregate {method!r}; expected one of {AGGREGATES}")


def rms_dbfs(samples: np.ndarray) -> float:
    """RMS level of a float waveform in dB relative to full scale (1.0)."""
    if samples.size == 0:
        return -math.inf
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * math.log10(rms) if rms > 0 else -math.inf
