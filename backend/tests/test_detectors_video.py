"""Quick tests for the video (temporal) detectors. No downloads; synthetic clips via FFmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from app.detectors import registry
from app.detectors.base import (
    AnalysisContext,
    AnalysisTimeoutError,
    DetectorResult,
    MediaKind,
    UnsupportedInputError,
)
from app.detectors.testing import StubDetector
from app.detectors.video import DEFAULT, DETECTORS, consistency, sampling
from app.detectors.video.heuristic import FrameConsistencyHeuristic

BACKEND = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = BACKEND / "test" / "real" / "00007.jpg"
HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")


def make_clip(path: Path, *, flicker: bool = False, seconds: float = 2.0, audio: bool = False) -> Path:
    """A small synthetic clip: slow zoom over a sample photo (or a test pattern), optional flicker/audio."""
    frames = int(seconds * 25)
    if SAMPLE_IMAGE.exists():
        inputs = ["-loop", "1", "-i", str(SAMPLE_IMAGE)]
        video = f"[0:v]scale=320:240,zoompan=z='1+0.002*on':d={frames}:s=160x120:fps=25"
    else:
        inputs = ["-f", "lavfi", "-i", "color=c=gray:size=160x120:rate=25"]
        video = "[0:v]null"
    if flicker:
        video += ",drawbox=x=60:y=30:w=40:h=40:color=white@0.3:t=fill:enable='mod(n,2)'"
    video += ",format=yuv420p[v]"
    command = ["ffmpeg", "-y", "-v", "error", *inputs]
    maps = ["-map", "[v]"]
    if audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000"]
        maps += ["-map", "1:a", "-c:a", "aac"]
    command += ["-filter_complex", video, *maps, "-t", f"{seconds}", "-c:v", "libx264", str(path)]
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    return path


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not installed")
    root = tmp_path_factory.mktemp("video-clips")
    return {"smooth": make_clip(root / "smooth.mp4"), "flicker": make_clip(root / "flicker.mp4", flicker=True)}


class FrameIndexImageDetector(StubDetector):
    """Image stub whose risk alternates with the frame number in the file name (f-001.jpg, ...)."""

    def _analyze(self, path: Path, ctx: AnalysisContext) -> DetectorResult:
        index = int(path.stem.rsplit("-", 1)[-1])
        risk = 0.9 if index % 2 else 0.1
        return DetectorResult(risk=risk, label="Fake" if risk >= 0.5 else "Real", confidence=max(risk, 1 - risk))


def heuristic(**options: object) -> FrameConsistencyHeuristic:
    return FrameConsistencyHeuristic(model_dir=Path("."), options={"max_clips": 2, **options})


# ── registration ────────────────────────────────────────────────────────────


def test_heuristic_is_registered_as_default() -> None:
    assert DEFAULT == "frame-consistency-heuristic"
    cls = DETECTORS[DEFAULT]
    assert cls.info.kind is MediaKind.VIDEO
    assert cls.info.heuristic is True
    assert "heuristic" in cls.info.display_name.lower()
    assert cls.weights() == []
    assert cls.exportable is False


# ── statistics (pure NumPy) ─────────────────────────────────────────────────


def test_global_shift_recovers_translation() -> None:
    rng = np.random.default_rng(0)
    reference = rng.uniform(0, 255, (120, 160)).astype(np.float32)
    moving = np.roll(reference, (3, -5), axis=(0, 1))
    shift = consistency.global_shift(reference, moving)
    assert shift == (-3, 5)
    assert np.allclose(np.roll(moving, shift, axis=(0, 1)), reference)
    assert consistency.global_shift(reference, np.roll(reference, (40, 0), axis=(0, 1))) is None


def test_clip_flicker_separates_static_from_alternating() -> None:
    rng = np.random.default_rng(1)
    base = rng.uniform(40, 200, (120, 160)).astype(np.float32)
    static = np.stack([base + rng.normal(0, 1.0, base.shape).astype(np.float32) for _ in range(8)])
    alternating = static.copy()
    alternating[1::2, 40:72, 60:92] += 25.0  # a 32x32 patch that pops on/off every other frame
    calm = consistency.clip_flicker(static)
    flicker = consistency.clip_flicker(alternating)
    assert calm is not None and flicker is not None
    assert calm["triples"] == 6 and flicker["triples"] == 6
    assert calm["roughness_p95"] < consistency.FLICKER_RAMP[0]
    assert flicker["roughness_p95"] > consistency.FLICKER_RAMP[1]
    assert consistency.clip_flicker(static[:2]) is None


def test_scene_cut_triples_are_skipped() -> None:
    rng = np.random.default_rng(2)
    first = rng.uniform(0, 80, (120, 160)).astype(np.float32)
    second = first + 150.0  # same structure (aligns at zero shift), very different brightness
    stats = consistency.clip_flicker(np.stack([first, first, second, second]))
    assert stats is not None and stats["triples"] == 0 and stats["scene_cuts"] == 2


def test_jitter_ramp_and_combine() -> None:
    assert consistency.score_jitter([0.3]) is None
    assert consistency.score_jitter([0.2, 0.2, 0.2]) == 0.0
    assert consistency.score_jitter([0.1, 0.9, 0.1]) == pytest.approx(0.8)
    assert consistency.ramp(0.0, (1.0, 3.0)) == 0.0
    assert consistency.ramp(2.0, (1.0, 3.0)) == pytest.approx(0.5)
    assert consistency.ramp(9.0, (1.0, 3.0)) == 1.0
    assert consistency.combine({"flicker": 0.5}) == pytest.approx(0.5)
    assert consistency.combine({"flicker": 0.0, "score_jitter": 1.0}) == pytest.approx(0.6)


def test_plan_clips_is_even_and_capped() -> None:
    info = sampling.VideoProbe(duration=60.0, fps=25.0, width=640, height=360, frame_count=1500)
    plan = sampling.plan_clips(info, max_clips=4, frames_per_clip=8)
    starts = [clip.start_seconds for clip in plan]
    assert len(plan) == 4 and starts == sorted(starts)
    assert starts[0] > 0 and starts[-1] + 8 / 25 < 60
    short = sampling.VideoProbe(duration=0.2, fps=25.0, width=64, height=64, frame_count=5)
    assert [(c.start_seconds, c.frames) for c in sampling.plan_clips(short, 4, 8)] == [(0.0, 8)]
    assert all(clip.frames == 3 for clip in sampling.plan_clips(info, 2, 1))


# ── end-to-end on synthetic clips ───────────────────────────────────────────


@needs_ffmpeg
def test_smooth_clip_scores_low_and_is_serialisable(clips: dict[str, Path], tmp_path: Path) -> None:
    with registry.override(MediaKind.IMAGE, None):
        result = heuristic().analyze(clips["smooth"], AnalysisContext(work_dir=tmp_path))
    assert result.label == "Real" and result.risk < 0.5
    details = result.details
    json.dumps(details)
    assert details["method"] == "heuristic" and details["calibrated"] is False
    assert 1 <= details["clips_analyzed"] <= 2
    assert details["components"]["score_jitter"]["available"] is False
    assert details["components"]["flicker"]["weight"] == consistency.COMPONENT_WEIGHTS["flicker"]
    assert not any(tmp_path.glob("temporal-*")), "scratch frames must be cleaned up"


@needs_ffmpeg
def test_flicker_clip_scores_high(clips: dict[str, Path], tmp_path: Path) -> None:
    with registry.override(MediaKind.IMAGE, None):
        result = heuristic().analyze(clips["flicker"], AnalysisContext(work_dir=tmp_path))
    assert result.label == "Fake" and result.risk >= 0.5


@needs_ffmpeg
def test_score_jitter_uses_configured_image_detector(clips: dict[str, Path], tmp_path: Path) -> None:
    alternating = FrameIndexImageDetector.for_kind(MediaKind.IMAGE)
    with registry.override(MediaKind.IMAGE, alternating):
        jumpy = heuristic(score_frames_per_clip=4).analyze(clips["smooth"], AnalysisContext(work_dir=tmp_path))
    jitter = jumpy.details["components"]["score_jitter"]
    assert jitter["value"] == pytest.approx(0.8)
    assert jitter["weight"] == consistency.COMPONENT_WEIGHTS["score_jitter"]
    assert jumpy.risk >= 0.6

    steady = StubDetector.for_kind(MediaKind.IMAGE, risk=0.7)
    with registry.override(MediaKind.IMAGE, steady):
        calm = heuristic(score_frames_per_clip=3).analyze(clips["smooth"], AnalysisContext(work_dir=tmp_path))
    assert calm.details["components"]["score_jitter"]["value"] == 0.0
    assert len(steady.calls) == 3 * calm.details["clips_analyzed"]
    assert calm.risk < 0.5


@needs_ffmpeg
def test_budget_and_deadline_raise_typed_timeout(clips: dict[str, Path], tmp_path: Path) -> None:
    with registry.override(MediaKind.IMAGE, None):
        with pytest.raises(AnalysisTimeoutError):
            heuristic().analyze(clips["smooth"], AnalysisContext(work_dir=tmp_path, deadline=time.monotonic() - 1))
        with pytest.raises(AnalysisTimeoutError):
            heuristic(max_seconds=0).analyze(clips["smooth"], AnalysisContext(work_dir=tmp_path))


@needs_ffmpeg
def test_non_video_input_is_unsupported(tmp_path: Path) -> None:
    bogus = tmp_path / "not-a-video.mp4"
    bogus.write_text("hello", encoding="utf-8")
    with pytest.raises(UnsupportedInputError):
        heuristic().analyze(bogus, AnalysisContext(work_dir=tmp_path))


@needs_ffmpeg
def test_concurrent_analyze_on_one_instance(clips: dict[str, Path], tmp_path: Path) -> None:
    detector = heuristic(max_clips=1)
    results: list[float] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(detector.analyze(clips["smooth"], AnalysisContext(work_dir=tmp_path)).risk)
        except BaseException as exc:
            errors.append(exc)

    with registry.override(MediaKind.IMAGE, None):
        threads = [threading.Thread(target=run) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
    assert not errors and len(results) == 3
    assert max(results) - min(results) < 1e-9
