"""Quick tests for media routing, the evidence schema and the video budget. Stub detectors only."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from app import pipeline
from app.detectors import registry
from app.detectors.base import AnalysisContext, DetectorResult, InferenceError, MediaKind
from app.detectors.testing import StubDetector
from app.detectors.video.heuristic import FrameConsistencyHeuristic

from .test_detectors_video import HAS_FFMPEG, make_clip

BACKEND = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = BACKEND / "test" / "real" / "00007.jpg"
SAMPLE_AUDIO = BACKEND / "test" / "real" / "audio1.wav"
EVIDENCE_KEYS = {"source", "label", "model", "status", "risk", "weight", "details"}
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")


class RecordingStub(StubDetector):
    """Stub that remembers the context of its last call."""

    last_ctx: AnalysisContext | None = None

    def _analyze(self, path: Path, ctx: AnalysisContext) -> DetectorResult:
        self.last_ctx = ctx
        return super()._analyze(path, ctx)


@pytest.fixture(scope="module")
def video_with_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not installed")
    return make_clip(tmp_path_factory.mktemp("pipeline") / "clip.mp4", audio=True)


def assert_schema(item: dict) -> None:
    assert set(item) == EVIDENCE_KEYS
    assert item["status"] in {"ok", "disabled", "error", "skipped"}
    if item["status"] == "ok":
        assert 0.0 <= item["risk"] <= 1.0
        assert item["details"]["prediction"] in {"Fake", "Real"}
        assert item["details"]["confidence"].endswith("%")
    else:
        assert item["risk"] is None and item["weight"] == 0
        assert isinstance(item["details"]["reason"], str)


def test_image_routes_to_image_detector(tmp_path: Path) -> None:
    stub = StubDetector.for_kind(MediaKind.IMAGE, risk=0.8)
    with registry.override(MediaKind.IMAGE, stub):
        items = pipeline.analyze("image", SAMPLE_IMAGE, tmp_path)
    assert [item["source"] for item in items] == ["image-model"]
    assert_schema(items[0])
    assert items[0]["risk"] == pytest.approx(0.8)
    assert items[0]["weight"] == pytest.approx(registry.settings(MediaKind.IMAGE).weight)
    assert items[0]["details"]["confidence"] == "80.00%"
    assert stub.calls == [SAMPLE_IMAGE]


def test_no_signal_when_only_detector_fails_or_is_disabled(tmp_path: Path) -> None:
    with registry.override(MediaKind.AUDIO, StubDetector.for_kind(MediaKind.AUDIO, error=InferenceError("boom"))):
        with pytest.raises(pipeline.NoSignalError):
            pipeline.analyze("audio", SAMPLE_AUDIO, tmp_path)
    with registry.override(MediaKind.IMAGE, None):
        with pytest.raises(pipeline.NoSignalError):
            pipeline.analyze("image", SAMPLE_IMAGE, tmp_path)
    with pytest.raises(ValueError):
        pipeline.analyze("document", SAMPLE_IMAGE, tmp_path)


def test_video_deadline_reads_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "_pipeline_options", lambda kind: {"max_seconds": 42})
    assert pipeline.video_deadline(start=100.0) == pytest.approx(142.0)
    monkeypatch.setattr(pipeline, "_pipeline_options", lambda kind: {})
    assert pipeline.video_deadline(start=100.0) == pytest.approx(100.0 + pipeline.DEFAULT_VIDEO_MAX_SECONDS)


@needs_ffmpeg
def test_video_produces_three_signals_with_fixed_weights(video_with_audio: Path, tmp_path: Path) -> None:
    temporal = RecordingStub.for_kind(MediaKind.VIDEO, risk=0.9)
    with (
        registry.override(MediaKind.IMAGE, StubDetector.for_kind(MediaKind.IMAGE, risk=0.2)),
        registry.override(MediaKind.AUDIO, StubDetector.for_kind(MediaKind.AUDIO, risk=0.3)),
        registry.override(MediaKind.VIDEO, temporal),
    ):
        before = time.monotonic()
        items = pipeline.analyze("video", video_with_audio, tmp_path)
    assert [item["source"] for item in items] == ["video-frame-model", "video-audio-model", "video-temporal-model"]
    for item in items:
        assert_schema(item)
        assert item["status"] == "ok"
    frames, audio, temporal_item = items
    assert frames["weight"] == pytest.approx(0.8) and audio["weight"] == pytest.approx(0.9)
    assert temporal_item["weight"] == pytest.approx(registry.settings(MediaKind.VIDEO).weight)
    assert frames["details"]["frames_analyzed"] >= 1 and frames["details"]["budget_exhausted"] is False
    ctx = temporal.last_ctx
    assert ctx is not None and ctx.frame_results is not None
    assert len(ctx.frame_results) == frames["details"]["frames_analyzed"]
    assert ctx.deadline is not None and ctx.deadline > before


@needs_ffmpeg
def test_video_temporal_switch_and_heuristic_label(video_with_audio: Path, tmp_path: Path) -> None:
    image = StubDetector.for_kind(MediaKind.IMAGE, risk=0.2)
    audio = StubDetector.for_kind(MediaKind.AUDIO, risk=0.3)
    with registry.override(MediaKind.IMAGE, image), registry.override(MediaKind.AUDIO, audio):
        with registry.override(MediaKind.VIDEO, None):
            disabled = pipeline.analyze("video", video_with_audio, tmp_path / "a")[2]
        heuristic = FrameConsistencyHeuristic(model_dir=Path("."), options={"max_clips": 1})
        with registry.override(MediaKind.VIDEO, heuristic):
            ran = pipeline.analyze("video", video_with_audio, tmp_path / "b")[2]
    assert disabled["status"] == "disabled" and disabled["risk"] is None
    assert_schema(ran)
    assert ran["status"] == "ok" and ran["label"].endswith("(heuristic)")
    assert ran["model"]["heuristic"] is True and ran["model"]["kind"] == "video"


@needs_ffmpeg
def test_video_budget_exhausted_reports_time_limit(video_with_audio: Path, tmp_path: Path) -> None:
    with (
        registry.override(MediaKind.IMAGE, StubDetector.for_kind(MediaKind.IMAGE)),
        registry.override(MediaKind.AUDIO, StubDetector.for_kind(MediaKind.AUDIO)),
        registry.override(MediaKind.VIDEO, StubDetector.for_kind(MediaKind.VIDEO)),
    ):
        with pytest.raises(pipeline.NoSignalError) as caught:
            pipeline.analyze("video", video_with_audio, tmp_path, deadline=time.monotonic() - 1)
    assert pipeline.TIME_LIMIT_REASON in str(caught.value)


def test_video_without_ffmpeg_reports_three_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real_which = shutil.which
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: None if name == "ffmpeg" else real_which(name))
    with pytest.raises(pipeline.NoSignalError):
        pipeline.analyze("video", tmp_path / "missing.mp4", tmp_path)
