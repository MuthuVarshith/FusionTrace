"""Frame-consistency heuristic: a non-learned temporal signal for video.

No permissively licensed, CPU-friendly temporal deepfake model was found on the
Hugging Face Hub (see docs/model-choice-video.md), so the temporal signal is a
heuristic over short clips of consecutive frames. It is labelled as such in
the report (``DetectorInfo.heuristic``) and its risk mapping is uncalibrated.

Options (config ``detectors.video.options``; defaults in ``OPTION_DEFAULTS``):

``max_clips`` (4)
    Number of clips sampled evenly across the video.
``frames_per_clip`` (8)
    Consecutive frames per clip, at the native frame rate (minimum 3).
``max_seconds`` (120)
    Processing budget for this detector. When it runs out after at least one
    clip, the clips analysed so far are used and ``budget_exhausted`` is set.
    The scan-wide ``ctx.deadline`` applies as well.
``score_jitter`` (true)
    Score consecutive frames with the configured image detector and measure
    how much its risk jumps between neighbouring frames.
``score_frames_per_clip`` (4)
    Consecutive frames per clip sent to the image detector (cost driver).
``threshold`` (0.5)
    Risk at or above which ``label`` is "Fake".
"""

from __future__ import annotations

import logging
import math
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from ..base import (
    AnalysisContext,
    AnalysisTimeoutError,
    Detector,
    DetectorError,
    DetectorInfo,
    DetectorResult,
    MediaKind,
    UnsupportedInputError,
)
from . import consistency, sampling

logger = logging.getLogger("fusiontrace")


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


class FrameConsistencyHeuristic(Detector):
    info = DetectorInfo(
        name="frame-consistency-heuristic",
        kind=MediaKind.VIDEO,
        display_name="Frame-consistency heuristic (uncalibrated)",
        source="local",
        license="MIT",
        version="1",
        heuristic=True,
        description=(
            "Non-learned temporal check on short clips of consecutive frames: motion-compensated "
            "flicker of local brightness, plus frame-to-frame instability of the image detector's "
            "score. Uncalibrated; a high value is a prompt for manual review, not evidence of manipulation."
        ),
    )

    OPTION_DEFAULTS: ClassVar[dict[str, Any]] = {
        "max_clips": 4,
        "frames_per_clip": 8,
        "max_seconds": 120.0,
        "score_jitter": True,
        "score_frames_per_clip": 4,
        "threshold": 0.5,
    }

    def is_available(self) -> bool:
        return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

    def _load(self) -> None:
        sampling.require_ffmpeg()

    # ── helpers ──

    def _options(self, ctx: AnalysisContext) -> dict[str, Any]:
        merged = {**self.OPTION_DEFAULTS, **self.options, **dict(ctx.options or {})}
        return {
            "max_clips": max(1, int(merged["max_clips"])),
            "frames_per_clip": max(3, int(merged["frames_per_clip"])),
            "max_seconds": max(0.0, float(merged["max_seconds"])),
            "score_jitter": str(merged["score_jitter"]).strip().lower() not in {"0", "false", "no", "off"},
            "score_frames_per_clip": max(0, int(merged["score_frames_per_clip"])),
            "threshold": min(1.0, max(0.0, float(merged["threshold"]))),
        }

    @staticmethod
    def _image_detector() -> tuple[Detector | None, dict[str, Any], str | None]:
        """The configured image detector, its options, and a reason when unavailable."""
        from .. import registry

        try:
            detector = registry.get(MediaKind.IMAGE)
        except DetectorError as exc:
            return None, {}, exc.public_message
        if detector is None:
            return None, {}, "The image detector is disabled by configuration."
        return detector, dict(registry.settings(MediaKind.IMAGE).options), None

    # ── inference ──

    def _analyze(self, path: Path, ctx: AnalysisContext) -> DetectorResult:
        opts = self._options(ctx)
        started = time.monotonic()
        deadline = started + opts["max_seconds"]
        if ctx.deadline is not None:
            deadline = min(deadline, ctx.deadline)

        def remaining() -> float:
            return deadline - time.monotonic()

        info = sampling.probe(path, timeout=min(30.0, max(1.0, remaining())))
        plan = sampling.plan_clips(info, opts["max_clips"], opts["frames_per_clip"])

        scorer: Detector | None = None
        scorer_options: dict[str, Any] = {}
        jitter_reason: str | None = "Disabled by option."
        if opts["score_jitter"] and opts["score_frames_per_clip"] >= 2:
            scorer, scorer_options, jitter_reason = self._image_detector()

        work = ctx.work_dir / f"temporal-{uuid4().hex[:8]}"
        per_clip: list[dict[str, Any]] = []
        budget_exhausted = False
        frames_failed = 0
        try:
            for clip in plan:
                if remaining() <= 0:
                    if not per_clip:
                        ctx.check_deadline()
                        raise AnalysisTimeoutError("temporal detector budget exhausted before the first clip")
                    budget_exhausted = True
                    break
                clip_dir = work / f"clip-{clip.index:02d}"
                frames = sampling.extract_clip(path, clip, clip_dir, timeout=min(60.0, max(1.0, remaining())))
                entry: dict[str, Any] = {"start_seconds": clip.start_seconds, "frames": len(frames)}
                if len(frames) >= 3:
                    flicker = consistency.clip_flicker(consistency.load_gray(frames))
                    if flicker is not None:
                        entry.update({key: _round(value) if isinstance(value, float) else value for key, value in flicker.items()})
                if scorer is not None and len(frames) >= 2:
                    risks, failed, exhausted = self._score_frames(scorer, scorer_options, frames, opts["score_frames_per_clip"], work, deadline)
                    frames_failed += failed
                    budget_exhausted = budget_exhausted or exhausted
                    entry["frame_risks"] = [_round(risk) for risk in risks]
                    entry["score_jitter"] = _round(consistency.score_jitter(risks))
                per_clip.append(entry)
                shutil.rmtree(clip_dir, ignore_errors=True)
                if budget_exhausted:
                    break
        finally:
            shutil.rmtree(work, ignore_errors=True)

        flicker_values = [clip["roughness_p95"] for clip in per_clip if clip.get("roughness_p95") is not None]
        jitter_values = [clip["score_jitter"] for clip in per_clip if clip.get("score_jitter") is not None]
        if not flicker_values and not jitter_values:
            raise UnsupportedInputError(
                f"no analysable clips in {path.name} ({len(per_clip)} clip(s) extracted)",
                public_message="The video is too short or too unstable for the temporal check.",
            )

        components: dict[str, float] = {}
        component_details: dict[str, Any] = {}
        if flicker_values:
            flicker = float(statistics.median(flicker_values))
            components["flicker"] = consistency.ramp(flicker, consistency.FLICKER_RAMP)
            component_details["flicker"] = {
                "value": _round(flicker),
                "aggregate": "median over clips of the 95th-percentile block roughness",
                "risk": _round(components["flicker"]),
                "ramp": list(consistency.FLICKER_RAMP),
            }
        else:
            component_details["flicker"] = {"available": False, "reason": "No clip had three usable consecutive frames."}
        if jitter_values:
            jitter = float(sum(jitter_values) / len(jitter_values))
            components["score_jitter"] = consistency.ramp(jitter, consistency.JITTER_RAMP)
            component_details["score_jitter"] = {
                "value": _round(jitter),
                "aggregate": "mean over clips of mean |risk(t+1) - risk(t)|",
                "risk": _round(components["score_jitter"]),
                "ramp": list(consistency.JITTER_RAMP),
                "image_model": scorer.info.name if scorer is not None else None,
                "frames_failed": frames_failed,
            }
        else:
            component_details["score_jitter"] = {"available": False, "reason": jitter_reason or "Too few frames were scored."}
        for name, weight in consistency.COMPONENT_WEIGHTS.items():
            component_details[name]["weight"] = weight if name in components else 0

        risk = consistency.combine(components)
        if math.isnan(risk):
            raise UnsupportedInputError(f"temporal statistics were undefined for {path.name}")
        label = "Fake" if risk >= opts["threshold"] else "Real"

        details: dict[str, Any] = {
            "method": "heuristic",
            "calibrated": False,
            "risk_formula": "weighted mean of available components: 0.6*score_jitter_risk + 0.4*flicker_risk (each a linear ramp, clipped to [0, 1])",
            "threshold": opts["threshold"],
            "components": component_details,
            "clips_planned": len(plan),
            "clips_analyzed": len(per_clip),
            "frames_per_clip": opts["frames_per_clip"],
            "budget_exhausted": budget_exhausted,
            "video": {"duration_seconds": _round(info.duration, 3), "fps": _round(info.fps, 3), "width": info.width, "height": info.height},
            "per_clip": per_clip,
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
        if ctx.frame_results:
            coarse = [frame.result.risk for frame in ctx.frame_results]
            details["coarse_frame_risk_range"] = _round(max(coarse) - min(coarse))  # reported only; frames are seconds apart
        return DetectorResult(risk=risk, label=label, confidence=max(risk, 1.0 - risk), details=details)

    @staticmethod
    def _score_frames(
        scorer: Detector, scorer_options: dict[str, Any], frames: list[Path], count: int, work: Path, deadline: float
    ) -> tuple[list[float], int, bool]:
        """Image-detector risk on ``count`` consecutive frames from the middle of the clip."""
        count = min(count, len(frames))
        first = (len(frames) - count) // 2
        image_ctx = AnalysisContext(work_dir=work, options=scorer_options, deadline=deadline)
        risks: list[float] = []
        failed = 0
        for frame in frames[first : first + count]:
            if time.monotonic() >= deadline:
                return risks, failed, True
            try:
                risks.append(scorer.analyze(frame, image_ctx).risk)
            except AnalysisTimeoutError:
                return risks, failed, True
            except DetectorError as exc:
                logger.info("Temporal heuristic: image detector skipped %s: %s", frame.name, exc)
                failed += 1
        return risks, failed, False
