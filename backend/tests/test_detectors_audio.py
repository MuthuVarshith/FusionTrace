"""Fast tests for the audio detectors: window math, aggregation, decoding edge cases.

No downloads: a stub scorer exercises the chunking driver, and a tiny randomly
initialised Wav2Vec2 (built in a temp dir) exercises the transformers path and
the export hooks.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.detectors.audio import DEFAULT, DETECTORS
from app.detectors.audio.chunked import ChunkedAudioDetector
from app.detectors.audio.hf_models import Wav2Vec2LegacyDetector
from app.detectors.audio.windows import ChunkOptions, aggregate, plan_windows
from app.detectors.base import (
    AnalysisContext,
    AnalysisTimeoutError,
    DetectorInfo,
    MediaKind,
    ModelUnavailableError,
    UnsupportedInputError,
)

SR = 16000
WIN, HOP = 4 * SR, 2 * SR


class PeakStub(ChunkedAudioDetector):
    """Risk of a window = its peak amplitude, so a loud burst marks the 'synthetic' part."""

    info = DetectorInfo(name="stub-audio", kind=MediaKind.AUDIO, display_name="Stub audio", source="local", license="MIT")

    def __init__(self, options: dict | None = None, delay: float = 0.0) -> None:
        super().__init__(model_dir=Path("."), options=options)
        self.delay = delay
        self.lengths: list[int] = []
        self._lock = threading.Lock()

    def _load(self) -> None:
        pass

    def _score_batch(self, windows: list[np.ndarray], window_samples: int) -> list[float]:
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.lengths.extend(w.size for w in windows)
        return [min(1.0, float(np.abs(w).max())) for w in windows]


def _tone(seconds: float, amplitude: float = 0.1, sr: int = SR, channels: int = 1) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    mono = (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return mono if channels == 1 else np.stack([mono] * channels, axis=1)


def _wav(path: Path, samples: np.ndarray, sr: int = SR) -> Path:
    sf.write(str(path), samples, sr, subtype="PCM_16")
    return path


def _ctx(tmp_path: Path, **options) -> AnalysisContext:
    return AnalysisContext(work_dir=tmp_path / "work", options=options)


# ── window planning ─────────────────────────────────────────────────────────


def test_short_clip_is_one_unpadded_span():
    assert plan_windows(SR, WIN, HOP, 60) == ([(0, SR)], False)
    assert plan_windows(WIN, WIN, HOP, 60) == ([(0, WIN)], False)
    assert plan_windows(0, WIN, HOP, 60) == ([], False)


def test_windows_cover_the_whole_clip():
    spans, capped = plan_windows(10 * SR, WIN, HOP, 60)
    assert [s for s, _ in spans] == [0, 2 * SR, 4 * SR, 6 * SR]
    assert spans[-1][1] == 10 * SR and not capped
    spans, _ = plan_windows(11 * SR, WIN, HOP, 60)
    assert len(spans) == 5 and spans[-1] == (7 * SR, 11 * SR)  # tail window aligned to the end
    assert all(end - start == WIN for start, end in spans)


def test_window_cap_spreads_evenly_and_keeps_both_ends():
    n = 200 * SR
    spans, capped = plan_windows(n, WIN, HOP, 60)
    starts = [s for s, _ in spans]
    assert capped and len(spans) == 60
    assert starts[0] == 0 and spans[-1][1] == n
    assert starts == sorted(set(starts))
    gaps = np.diff(starts)
    assert gaps.max() - gaps.min() <= HOP  # evenly spaced up to one hop of rounding


def test_aggregates():
    scores = [0.1, 0.2, 0.9, 0.8, 0.1]
    assert aggregate(scores, "mean") == pytest.approx(0.42)
    assert aggregate(scores, "max") == pytest.approx(0.9)
    assert aggregate(scores, "top_k_mean", top_k=2) == pytest.approx(0.85)
    assert aggregate([0.3], "top_k_mean", top_k=3) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        aggregate(scores, "median")
    with pytest.raises(ValueError):
        aggregate([], "mean")


def test_options_defaults_and_validation():
    opts = ChunkOptions.from_mapping({})
    assert (opts.window_seconds, opts.hop_seconds, opts.max_windows, opts.aggregate, opts.top_k) == (4.0, 2.0, 60, "top_k_mean", 3)
    assert ChunkOptions.from_mapping({"max_windows": "30", "aggregate": "MAX", "unknown": 1}).max_windows == 30
    for bad in ({"aggregate": "median"}, {"hop_seconds": 0}, {"max_windows": 2.5}, {"top_k": True}, {"threshold": 1.5}):
        with pytest.raises(ValueError):
            ChunkOptions.from_mapping(bad)


# ── chunked analysis (stub scorer) ──────────────────────────────────────────


def test_spliced_segment_is_diluted_by_mean_but_not_by_top_k(tmp_path):
    clip = _tone(60, 0.1)
    clip[20 * SR : 24 * SR] = _tone(4, 0.9)  # a 4 s "synthetic" insert
    path = _wav(tmp_path / "spliced.wav", clip)
    detector = PeakStub()
    result = detector.analyze(path, _ctx(tmp_path))
    details = result.details
    assert details["window_count"] == 29 and details["aggregate"] == "top_k_mean"
    assert details["mean_risk"] < 0.25 < 0.85 < details["max_risk"]
    assert result.risk == pytest.approx(0.9, abs=0.01) and result.label == "Fake"
    windows = details["per_window_scores"]
    assert windows[0]["start_seconds"] == 0 and windows[-1]["end_seconds"] == 60
    assert details["peak_window"]["start_seconds"] in {18.0, 20.0, 22.0}
    mean_result = detector.analyze(path, _ctx(tmp_path, aggregate="mean"))
    assert mean_result.label == "Real" and mean_result.risk == pytest.approx(details["mean_risk"], abs=1e-3)


def test_short_clip_is_scored_once_and_flagged_padded(tmp_path):
    path = _wav(tmp_path / "short.wav", _tone(1.0, 0.3))
    detector = PeakStub()
    result = detector.analyze(path, _ctx(tmp_path))
    assert result.details["window_count"] == 1 and result.details["padded"] is True
    assert detector.lengths == [SR]  # raw samples; the model's _prepare pads to the window


def test_resampled_stereo_windows_have_exact_length(tmp_path):
    path = _wav(tmp_path / "stereo.wav", _tone(9.0, 0.2, sr=44100, channels=2), sr=44100)
    detector = PeakStub()
    result = detector.analyze(path, _ctx(tmp_path))
    assert result.details["source_sample_rate"] == 44100
    assert detector.lengths == [WIN] * result.details["window_count"]
    assert 0.18 < result.details["max_risk"] < 0.22


def test_silent_empty_and_tiny_clips_are_rejected(tmp_path):
    detector = PeakStub()
    cases = {
        "silent.wav": np.zeros(5 * SR, dtype=np.float32),
        "empty.wav": np.zeros(0, dtype=np.float32),
        "tiny.wav": _tone(0.1, 0.5),
    }
    for name, samples in cases.items():
        with pytest.raises(UnsupportedInputError):
            detector.analyze(_wav(tmp_path / name, samples), _ctx(tmp_path))


def test_silent_windows_are_skipped(tmp_path):
    clip = np.concatenate([_tone(4, 0.3), np.zeros(8 * SR, dtype=np.float32)])
    result = PeakStub().analyze(_wav(tmp_path / "gap.wav", clip), _ctx(tmp_path))
    assert result.details["windows_skipped_silent"] >= 2
    assert result.details["window_count"] + result.details["windows_skipped_silent"] == result.details["windows_planned"]


def test_undecodable_file_raises_unsupported(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"RIFF....WAVEjunk")
    with pytest.raises(UnsupportedInputError):
        PeakStub().analyze(path, _ctx(tmp_path))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_mp3_is_decoded_with_ffmpeg_inside_work_dir(tmp_path):
    wav = _wav(tmp_path / "src.wav", _tone(6.0, 0.4))
    mp3 = tmp_path / "clip.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), str(mp3)], check=True, capture_output=True)
    ctx = _ctx(tmp_path)
    result = PeakStub().analyze(mp3, ctx)
    assert result.details["decoder"] == "ffmpeg" and result.details["window_count"] in (2, 3)
    assert list(ctx.work_dir.glob("*.wav")) == []  # converted audio is cleaned up


def test_deadline_is_checked_between_windows(tmp_path):
    path = _wav(tmp_path / "long.wav", _tone(30, 0.2))
    ctx = AnalysisContext(work_dir=tmp_path / "work", deadline=time.monotonic() + 0.3)
    with pytest.raises(AnalysisTimeoutError):
        PeakStub(delay=0.1).analyze(path, ctx)


def test_invalid_options_raise_model_unavailable(tmp_path):
    path = _wav(tmp_path / "a.wav", _tone(2.0))
    with pytest.raises(ModelUnavailableError):
        PeakStub(options={"aggregate": "median"}).analyze(path, _ctx(tmp_path))


def test_concurrent_analyze_calls_agree(tmp_path):
    path = _wav(tmp_path / "c.wav", _tone(12, 0.25))
    detector = PeakStub()
    results: list[float] = []

    def run(i: int) -> None:
        results.append(detector.analyze(path, AnalysisContext(work_dir=tmp_path / f"w{i}")).risk)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 4 and max(results) - min(results) < 1e-9


# ── registry and the transformers path (tiny random model) ──────────────────


def test_registry_entries():
    assert DEFAULT in DETECTORS
    for name, cls in DETECTORS.items():
        assert cls.info.name == name and cls.info.kind is MediaKind.AUDIO
    spec = DETECTORS["wav2vec2-legacy"].weights()[0]
    assert spec.repo_id_env == "HF_REPO_ID" and spec.allow_patterns == ("audio_model/*",)
    for name in ("wav2vec2-xlsr-garystafford", "wav2vec2-base-melodymachine"):
        (spec,) = DETECTORS[name].weights()
        assert spec.revision and len(spec.revision) == 40


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory) -> Path:
    from transformers import Wav2Vec2Config, Wav2Vec2ForSequenceClassification

    directory = tmp_path_factory.mktemp("tiny") / "audio_model"
    config = Wav2Vec2Config(
        vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        conv_dim=(8, 8), conv_stride=(10, 8), conv_kernel=(20, 8), num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2, classifier_proj_size=8, num_labels=2,
    )
    import torch

    torch.manual_seed(0)
    Wav2Vec2ForSequenceClassification(config).save_pretrained(str(directory), safe_serialization=True)
    return directory


def _tiny_legacy(directory: Path) -> Wav2Vec2LegacyDetector:
    class TinyLegacy(Wav2Vec2LegacyDetector):
        def weight_dirs(self) -> list[Path]:
            return [directory]

    return TinyLegacy(model_dir=directory.parent)


def test_lfs_pointer_is_not_treated_as_weights(tmp_path):
    directory = tmp_path / "audio_model"
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    (directory / "model.safetensors").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    detector = _tiny_legacy(directory)
    assert not detector.is_available()
    with pytest.raises(ModelUnavailableError):
        detector.load()


def test_tiny_legacy_model_analyze_and_export_hooks_agree(tmp_path, tiny_model_dir):
    detector = _tiny_legacy(tiny_model_dir)
    assert detector.is_available()
    path = _wav(tmp_path / "short.wav", _tone(1.5, 0.3))
    result = detector.analyze(path, _ctx(tmp_path))
    assert 0.0 <= result.risk <= 1.0 and result.details["window_count"] == 1

    (inputs,) = detector.prepare_inputs(path)
    assert tuple(inputs.shape) == (1, WIN)  # short clips are zero-padded to one window, as in training
    assert float(inputs[0, int(1.5 * SR) :].abs().max()) == 0.0
    import torch

    with torch.no_grad():
        outputs = detector.export_module()(inputs)
    assert tuple(outputs.shape) == (1, 2)
    assert detector.risk_from_outputs([outputs]) == pytest.approx(result.risk, abs=1e-4)
    assert detector.risk_from_outputs([outputs.numpy()]) == pytest.approx(result.risk, abs=1e-4)


def test_tiny_legacy_model_batches_match_single_windows(tmp_path, tiny_model_dir):
    path = _wav(tmp_path / "long.wav", _tone(10.0, 0.3) + np.random.default_rng(0).normal(0, 0.01, 10 * SR).astype(np.float32))
    single = _tiny_legacy(tiny_model_dir).analyze(path, _ctx(tmp_path))
    batched = _tiny_legacy(tiny_model_dir).analyze(path, _ctx(tmp_path, batch_size=4))
    a = [w["risk"] for w in single.details["per_window_scores"]]
    b = [w["risk"] for w in batched.details["per_window_scores"]]
    assert a == pytest.approx(b, abs=1e-3)
