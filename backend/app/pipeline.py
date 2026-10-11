"""Media kind -> evidence signals, via the detector registry.

``analyze`` turns one uploaded file into a list of evidence items. Turning
evidence into a risk rating stays in ``service.fuse_evidence``.

Evidence item schema (consumed by service, UI, PDF, AI summary)::

    {
      "source":  "image-model" | "audio-model" | "video-frame-model"
                 | "video-audio-model" | "video-temporal-model",
      "label":   human-readable signal name,
      "model":   DetectorInfo.to_report() dict, or None if no detector ran,
      "status":  "ok" | "disabled" | "error" | "skipped",
      "risk":    float in [0, 1], or None when the signal did not run,
      "weight":  fusion weight (0 when risk is None),
      "details": JSON-serialisable dict; for status "ok" includes
                 "prediction" ("Fake"/"Real") and "confidence" ("97.12%"),
                 otherwise "reason" (a client-safe message),
    }

When the detector that produced a signal is a heuristic
(``DetectorInfo.heuristic``), " (heuristic)" is appended to ``label`` unless the
label already says so.

Routing:
    image -> image detector
    audio -> audio detector (the detector handles chunking over the whole clip)
    video -> image detector on sampled frames + audio detector on the soundtrack
             + video (temporal) detector, each independently switchable

Video scans run under one processing budget, ``pipeline.video.max_seconds``
(default ``DEFAULT_VIDEO_MAX_SECONDS``), passed to every detector as
``ctx.deadline``. Frame analysis stops early (keeping the frames already scored)
when it runs out; signals that cannot start in time are reported with status
"error" and the time-limit reason.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .detectors import registry
from .detectors.base import AnalysisContext, AnalysisTimeoutError, Detector, DetectorError, FrameResult, MediaKind

logger = logging.getLogger("fusiontrace")

DEFAULT_VIDEO_MAX_SECONDS = 300.0
FFMPEG_TIMEOUT_SECONDS = 120.0
TIME_LIMIT_REASON = AnalysisTimeoutError.public_message
TEMPORAL_LABEL = "Temporal consistency across frames"


class NoSignalError(DetectorError):
    """No detector produced a usable signal for this file."""

    public_message = "No detector could analyse this file."


def _pipeline_options(kind: str) -> dict[str, Any]:
    from .config import config

    return dict(((config.get("pipeline") or {}).get(kind)) or {})


def _signal_label(label: str, detector: Detector | None) -> str:
    if detector is not None and detector.info.heuristic and "heuristic" not in label.lower():
        return f"{label} (heuristic)"
    return label


def _subprocess_timeout(ctx: AnalysisContext) -> float:
    remaining = ctx.remaining_seconds()
    return FFMPEG_TIMEOUT_SECONDS if remaining is None else max(1.0, min(FFMPEG_TIMEOUT_SECONDS, remaining))


def _out_of_time(ctx: AnalysisContext) -> bool:
    remaining = ctx.remaining_seconds()
    return remaining is not None and remaining <= 0


def _not_run(source: str, label: str, detector: Detector | None, status: str, reason: str) -> dict[str, Any]:
    return {
        "source": source,
        "label": _signal_label(label, detector),
        "model": detector.info.to_report() if detector is not None else None,
        "status": status,
        "risk": None,
        "weight": 0,
        "details": {"reason": reason},
    }


def detector_evidence(kind: MediaKind, path: Path, ctx: AnalysisContext, *, source: str, label: str, weight: float | None = None) -> dict[str, Any]:
    """Run the configured detector for ``kind`` on ``path`` and wrap the result as evidence."""
    try:
        detector = registry.get(kind)
    except DetectorError as exc:
        logger.warning("%s detector unavailable: %s", kind.value, exc)
        return _not_run(source, label, None, "error", exc.public_message)
    if detector is None:
        return _not_run(source, label, None, "disabled", "Disabled by configuration.")
    current = registry.settings(kind)
    try:
        result = detector.analyze(path, replace(ctx, options=current.options))
    except DetectorError as exc:
        logger.warning("%s detector %s failed: %s", kind.value, detector.info.name, exc)
        return _not_run(source, label, detector, "error", exc.public_message)
    details = {"prediction": result.label, "confidence": f"{result.confidence * 100:.2f}%", **result.details}
    return {
        "source": source,
        "label": _signal_label(label, detector),
        "model": detector.info.to_report(),
        "status": "ok",
        "risk": result.risk,
        "weight": current.weight if weight is None else weight,
        "details": details,
    }


def video_deadline(start: float | None = None) -> float:
    """``time.monotonic()`` deadline for a video scan from ``pipeline.video.max_seconds``."""
    raw = _pipeline_options("video").get("max_seconds", DEFAULT_VIDEO_MAX_SECONDS)
    try:
        budget = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid pipeline.video.max_seconds %r; using %s", raw, DEFAULT_VIDEO_MAX_SECONDS)
        budget = DEFAULT_VIDEO_MAX_SECONDS
    return (time.monotonic() if start is None else start) + max(0.0, budget)


def analyze(kind: str, path: Path, work_dir: Path, deadline: float | None = None) -> list[dict[str, Any]]:
    """Evidence items for one file. Raises NoSignalError if nothing produced a risk.

    ``deadline`` is a ``time.monotonic()`` value. For video it defaults to
    ``pipeline.video.max_seconds`` from now.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    if kind == "video" and deadline is None:
        deadline = video_deadline()
    ctx = AnalysisContext(work_dir=work_dir, deadline=deadline)
    if kind == "image":
        items = [detector_evidence(MediaKind.IMAGE, path, ctx, source="image-model", label="Image manipulation model")]
    elif kind == "audio":
        items = [detector_evidence(MediaKind.AUDIO, path, ctx, source="audio-model", label="Synthetic voice model")]
    elif kind == "video":
        items = _video_evidence(path, ctx)
    else:
        raise ValueError(f"unknown media kind {kind!r}")
    if not any(item["risk"] is not None for item in items):
        raise NoSignalError(f"no usable signal for {kind} file {path.name}: {[item['details'] for item in items]}")
    return items


# ── video ──────────────────────────────────────────────────────────────────


def _video_evidence(path: Path, ctx: AnalysisContext) -> list[dict[str, Any]]:
    opts = _pipeline_options("video")
    if not shutil.which("ffmpeg"):
        reason = "FFmpeg is not installed in the runtime."
        return [
            _not_run("video-frame-model", "Sampled video-frame analysis", None, "error", reason),
            _not_run("video-audio-model", "Synthetic voice model (extracted from video)", None, "error", reason),
            _not_run("video-temporal-model", TEMPORAL_LABEL, None, "error", reason),
        ]
    frame_item, frame_results = _frame_evidence(path, ctx, opts)
    audio_item = _soundtrack_evidence(path, ctx, opts)
    # Weight comes from detectors.video.weight; enabled/model from detectors.video.
    temporal_item = detector_evidence(
        MediaKind.VIDEO, path, replace(ctx, frame_results=frame_results), source="video-temporal-model", label=TEMPORAL_LABEL
    )
    return [frame_item, audio_item, temporal_item]


def _frame_evidence(path: Path, ctx: AnalysisContext, opts: dict[str, Any]) -> tuple[dict[str, Any], list[FrameResult]]:
    source, label = "video-frame-model", "Sampled video-frame analysis"
    try:
        detector = registry.get(MediaKind.IMAGE)
    except DetectorError as exc:
        return _not_run(source, label, None, "error", exc.public_message), []
    if detector is None:
        return _not_run(source, label, None, "disabled", "Disabled by configuration."), []

    if _out_of_time(ctx):
        return _not_run(source, label, detector, "error", TIME_LIMIT_REASON), []
    interval = float(opts.get("frame_interval_seconds", 5))
    max_frames = int(opts.get("max_frames", 12))
    frames_dir = ctx.work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vf", f"fps=1/{interval:g}", "-frames:v", str(max_frames), str(frames_dir / "frame-%03d.jpg")],
            check=True, capture_output=True, timeout=_subprocess_timeout(ctx),
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("Frame extraction failed for %s: %s", path.name, exc)
        return _not_run(source, label, detector, "error", "Video frames could not be extracted."), []

    image_options = registry.settings(MediaKind.IMAGE).options
    frame_paths = sorted(frames_dir.glob("frame-*.jpg"))
    results: list[FrameResult] = []
    skipped = 0
    budget_exhausted = False
    for index, frame_path in enumerate(frame_paths):
        try:
            result = detector.analyze(frame_path, replace(ctx, options=image_options))
        except AnalysisTimeoutError:
            budget_exhausted = True
            break
        except DetectorError as exc:
            logger.info("Frame %s skipped: %s", frame_path.name, exc)
            skipped += 1
            continue
        results.append(FrameResult(timestamp_seconds=index * interval, path=frame_path, result=result))
    if not results:
        reason = TIME_LIMIT_REASON if budget_exhausted else "No analysable frames were extracted."
        return _not_run(source, label, detector, "error", reason), []

    risks = [frame.result.risk for frame in results]
    average = sum(risks) / len(risks)
    peak = max(results, key=lambda frame: frame.result.risk)
    item = {
        "source": source,
        "label": label,
        "model": detector.info.to_report(),
        "status": "ok",
        "risk": average,
        "weight": float(opts.get("frame_weight", 0.8)),
        "details": {
            "frames_analyzed": len(results),
            "frames_skipped": skipped,
            "frames_not_reached": len(frame_paths) - len(results) - skipped,
            "budget_exhausted": budget_exhausted,
            "average_risk_score": f"{average * 100:.1f}%",
            "peak_frame": {
                "timestamp_seconds": peak.timestamp_seconds,
                "risk_score": f"{peak.result.risk * 100:.1f}%",
                "prediction": peak.result.label,
            },
            "per_frame_scores": [
                {"timestamp_seconds": frame.timestamp_seconds, "risk_score": f"{frame.result.risk * 100:.1f}%"} for frame in results
            ],
        },
    }
    return item, results


def _soundtrack_evidence(path: Path, ctx: AnalysisContext, opts: dict[str, Any]) -> dict[str, Any]:
    source, label = "video-audio-model", "Synthetic voice model (extracted from video)"
    if _out_of_time(ctx):
        return _not_run(source, label, None, "error", TIME_LIMIT_REASON)
    audio_path = ctx.work_dir / "soundtrack.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vn", "-ar", "16000", "-ac", "1", str(audio_path)],
            capture_output=True, timeout=_subprocess_timeout(ctx), check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("Soundtrack extraction failed for %s: %s", path.name, exc)
    if not audio_path.exists() or audio_path.stat().st_size <= 100:
        return _not_run(source, label, None, "skipped", "No audio track found in this video.")
    return detector_evidence(MediaKind.AUDIO, audio_path, ctx, source=source, label=label, weight=float(opts.get("audio_weight", 0.9)))
