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

Routing:
    image -> image detector
    audio -> audio detector (the detector handles chunking over the whole clip)
    video -> image detector on sampled frames + audio detector on the soundtrack
             + video (temporal) detector, each independently switchable
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

from .detectors import registry
from .detectors.base import AnalysisContext, Detector, DetectorError, FrameResult, MediaKind

logger = logging.getLogger("fusiontrace")


class NoSignalError(DetectorError):
    """No detector produced a usable signal for this file."""

    public_message = "No detector could analyse this file."


def _pipeline_options(kind: str) -> dict[str, Any]:
    from .config import config

    return dict(((config.get("pipeline") or {}).get(kind)) or {})


def _not_run(source: str, label: str, detector: Detector | None, status: str, reason: str) -> dict[str, Any]:
    return {
        "source": source,
        "label": label,
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
        "label": label,
        "model": detector.info.to_report(),
        "status": "ok",
        "risk": result.risk,
        "weight": current.weight if weight is None else weight,
        "details": details,
    }


def analyze(kind: str, path: Path, work_dir: Path, deadline: float | None = None) -> list[dict[str, Any]]:
    """Evidence items for one file. Raises NoSignalError if nothing produced a risk."""
    work_dir.mkdir(parents=True, exist_ok=True)
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
            _not_run("video-temporal-model", "Temporal video model", None, "error", reason),
        ]
    frame_item, frame_results = _frame_evidence(path, ctx, opts)
    audio_item = _soundtrack_evidence(path, ctx, opts)
    temporal_item = detector_evidence(
        MediaKind.VIDEO, path, replace(ctx, frame_results=frame_results), source="video-temporal-model", label="Temporal video model"
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

    interval = float(opts.get("frame_interval_seconds", 5))
    max_frames = int(opts.get("max_frames", 12))
    frames_dir = ctx.work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vf", f"fps=1/{interval:g}", "-frames:v", str(max_frames), str(frames_dir / "frame-%03d.jpg")],
            check=True, capture_output=True, timeout=120,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("Frame extraction failed for %s: %s", path.name, exc)
        return _not_run(source, label, detector, "error", "Video frames could not be extracted."), []

    image_options = registry.settings(MediaKind.IMAGE).options
    results: list[FrameResult] = []
    skipped = 0
    for index, frame_path in enumerate(sorted(frames_dir.glob("frame-*.jpg"))):
        try:
            result = detector.analyze(frame_path, replace(ctx, options=image_options))
        except DetectorError as exc:
            logger.info("Frame %s skipped: %s", frame_path.name, exc)
            skipped += 1
            continue
        results.append(FrameResult(timestamp_seconds=index * interval, path=frame_path, result=result))
    if not results:
        return _not_run(source, label, detector, "error", "No analysable frames were extracted."), []

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
    audio_path = ctx.work_dir / "soundtrack.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vn", "-ar", "16000", "-ac", "1", str(audio_path)],
            capture_output=True, timeout=120, check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("Soundtrack extraction failed for %s: %s", path.name, exc)
    if not audio_path.exists() or audio_path.stat().st_size <= 100:
        return _not_run(source, label, None, "skipped", "No audio track found in this video.")
    return detector_evidence(MediaKind.AUDIO, audio_path, ctx, source=source, label=label, weight=float(opts.get("audio_weight", 0.9)))
