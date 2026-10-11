"""Frame-consistency statistics for the temporal heuristic (NumPy only, no learned weights).

Two measurements, both computed on short clips of consecutive frames:

``flicker`` (motion-compensated temporal roughness)
    Frames are converted to greyscale at ``WORK_WIDTH`` px. For every frame
    triple (t-1, t, t+1) the neighbours are aligned to frame t by a global
    translation (phase correlation), a fixed border is cropped, and the frame
    is divided into ``BLOCK`` x ``BLOCK`` blocks. For each block we take the
    block-mean luminance minus the frame-mean luminance (removes global
    exposure changes) and compare

        e1 = |r(t+1) - r(t-1)| / 2          (smooth change: motion, lighting)
        e2 = |r(t+1) - 2 r(t) + r(t-1)|     (non-smooth change: flicker, popping)

    averaged over the clip's triples. Per block, ``roughness = e2 / (e1 + TAU)``:
    ~0 for static or smoothly moving content, large when a region changes
    back and forth between consecutive frames (the classic face-swap
    "shimmer"). ``TAU`` (grey levels) keeps codec noise in static blocks from
    producing large ratios. The clip value is the 95th percentile over blocks,
    so a small flickering region (a face) is not averaged away by the
    background. Triples with a scene cut or with global motion larger than
    ``MAX_SHIFT`` px are skipped.

``score_jitter`` (frame-to-frame instability of the image detector)
    Mean absolute change in image-detector risk between consecutive frames.
    Consecutive frames are 1/fps apart and nearly identical, so a stable
    detector should give nearly identical scores on authentic footage.

Mapping to risk (uncalibrated, chosen a priori, see docs/model-choice-video.md)::

    flicker_risk = ramp(flicker,      FLICKER_RAMP)   # 0 at <= 1.0, 1 at >= 3.0
    jitter_risk  = ramp(score_jitter, JITTER_RAMP)    # 0 at <= 0.02, 1 at >= 0.20
    risk = weighted mean of the available components (COMPONENT_WEIGHTS)

where ``ramp(x, (lo, hi)) = clip((x - lo) / (hi - lo), 0, 1)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

WORK_WIDTH = 320  # px; frames are analysed at this width
BLOCK = 16  # px; block size for local statistics
MAX_SHIFT = 16  # px at WORK_WIDTH; larger global motion between frames is skipped (and is also the crop border)
TAU = 1.0  # grey levels; noise floor added to e1
CUT_THRESHOLD = 30.0  # grey levels; mean absolute difference that marks a scene cut

FLICKER_RAMP = (1.0, 3.0)
JITTER_RAMP = (0.02, 0.20)
COMPONENT_WEIGHTS = {"score_jitter": 0.6, "flicker": 0.4}


def ramp(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float(min(1.0, max(0.0, (value - lo) / (hi - lo))))


def load_gray(paths: Sequence[Path], width: int = WORK_WIDTH) -> np.ndarray:
    """Load frames as a (T, H, W) float32 greyscale stack at ``width`` px."""
    from PIL import Image

    frames: list[np.ndarray] = []
    size: tuple[int, int] | None = None
    for path in paths:
        with Image.open(path) as image:
            gray = image.convert("L")
            if size is None:
                height = max(2 * (MAX_SHIFT + BLOCK), round(gray.height * width / max(1, gray.width)))
                size = (width, height)
            frames.append(np.asarray(gray.resize(size, Image.BILINEAR), dtype=np.float32))
    if not frames:
        return np.zeros((0, 0, 0), dtype=np.float32)
    return np.stack(frames)


def _window(shape: tuple[int, int]) -> np.ndarray:
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1])).astype(np.float32)


def global_shift(reference: np.ndarray, moving: np.ndarray, max_shift: int = MAX_SHIFT) -> tuple[int, int] | None:
    """Integer (dy, dx) such that ``np.roll(moving, (dy, dx), (0, 1))`` aligns with ``reference``.

    Phase correlation. Returns None when the estimated shift exceeds ``max_shift``.
    """
    window = _window(reference.shape)
    a = np.fft.fft2((reference - reference.mean()) * window)
    b = np.fft.fft2((moving - moving.mean()) * window)
    cross = a * np.conj(b)
    cross /= np.abs(cross) + 1e-9
    correlation = np.fft.ifft2(cross).real
    peak_y, peak_x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    height, width = correlation.shape
    dy = int(peak_y if peak_y <= height // 2 else peak_y - height)
    dx = int(peak_x if peak_x <= width // 2 else peak_x - width)
    if abs(dy) > max_shift or abs(dx) > max_shift:
        return None
    return dy, dx


def _block_means(frame: np.ndarray, block: int = BLOCK) -> np.ndarray:
    rows, cols = frame.shape[0] // block, frame.shape[1] // block
    trimmed = frame[: rows * block, : cols * block]
    return trimmed.reshape(rows, block, cols, block).mean(axis=(1, 3))


def clip_flicker(frames: np.ndarray) -> dict[str, Any] | None:
    """Motion-compensated temporal roughness of one clip; None if no usable frame triple."""
    if frames.ndim != 3 or frames.shape[0] < 3:
        return None
    m = MAX_SHIFT
    e1: np.ndarray | None = None
    e2: np.ndarray | None = None
    triples = cuts = fast_motion = 0
    for t in range(1, frames.shape[0] - 1):
        current = frames[t]
        shift_prev = global_shift(current, frames[t - 1])
        shift_next = global_shift(current, frames[t + 1])
        if shift_prev is None or shift_next is None:
            fast_motion += 1
            continue
        prev = np.roll(frames[t - 1], shift_prev, axis=(0, 1))[m:-m, m:-m]
        nxt = np.roll(frames[t + 1], shift_next, axis=(0, 1))[m:-m, m:-m]
        cur = current[m:-m, m:-m]
        if np.abs(nxt - cur).mean() > CUT_THRESHOLD or np.abs(cur - prev).mean() > CUT_THRESHOLD:
            cuts += 1
            continue
        blocks = [_block_means(f) for f in (prev, cur, nxt)]
        rel_prev, rel_cur, rel_next = (b - b.mean() for b in blocks)
        first = np.abs(rel_next - rel_prev) / 2.0
        second = np.abs(rel_next - 2.0 * rel_cur + rel_prev)
        e1 = first if e1 is None else e1 + first
        e2 = second if e2 is None else e2 + second
        triples += 1
    if triples == 0 or e1 is None or e2 is None:
        return {"triples": 0, "scene_cuts": cuts, "fast_motion_skipped": fast_motion}
    e1 /= triples
    e2 /= triples
    roughness = e2 / (e1 + TAU)
    return {
        "triples": triples,
        "scene_cuts": cuts,
        "fast_motion_skipped": fast_motion,
        "roughness_p95": float(np.percentile(roughness, 95)),
        "roughness_median": float(np.median(roughness)),
        "second_diff_p95": float(np.percentile(e2, 95)),
    }


def score_jitter(risks: Sequence[float]) -> float | None:
    """Mean absolute change between consecutive image-detector risks; None for < 2 scores."""
    if len(risks) < 2:
        return None
    values = np.asarray(risks, dtype=np.float64)
    return float(np.abs(np.diff(values)).mean())


def combine(components: dict[str, float]) -> float:
    """Weighted mean of the available component risks (keys from COMPONENT_WEIGHTS)."""
    weights = {name: COMPONENT_WEIGHTS[name] for name in components}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("no components to combine")
    return float(sum(components[name] * weights[name] for name in components) / total)
