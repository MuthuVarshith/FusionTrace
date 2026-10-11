"""Shared sliding-window driver for audio detectors.

A concrete detector only has to turn a batch of equal-length 16 kHz windows
into per-window risks (``_score_batch``); decoding, window planning, silence
skipping, deadline checks, aggregation and the report details live here.
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ..base import AnalysisContext, Detector, DetectorResult, ModelUnavailableError, UnsupportedInputError
from .decoding import AudioSource, open_audio
from .windows import ChunkOptions, aggregate, plan_windows, rms_dbfs

logger = logging.getLogger("fusiontrace")

SAMPLE_RATE = 16000
MIN_CLIP_SECONDS = 0.25  # shorter clips carry no usable speech evidence (and are below the wav2vec2 receptive field)


class ChunkedAudioDetector(Detector):
    """Scores every (capped) window of a clip and aggregates the window risks."""

    sample_rate: int = SAMPLE_RATE

    # ── to implement ──

    @abstractmethod
    def _score_batch(self, windows: list[np.ndarray], window_samples: int) -> list[float]:
        """Risk in [0, 1] for each raw mono float32 window (shorter than ``window_samples`` only for short clips)."""

    # ── options ──

    def chunk_options(self, ctx: AnalysisContext | None = None) -> ChunkOptions:
        merged: dict[str, Any] = dict(self.options)
        if ctx is not None:
            merged.update(ctx.options or {})
        try:
            return ChunkOptions.from_mapping(merged)
        except ValueError as exc:
            raise ModelUnavailableError(
                f"{self.info.name}: invalid audio options: {exc}",
                public_message="The audio detector is misconfigured on this server.",
            ) from exc

    # ── analysis ──

    def _analyze(self, path: Path, ctx: AnalysisContext) -> DetectorResult:
        opts = self.chunk_options(ctx)
        with open_audio(path, ctx.work_dir, self.sample_rate, timeout=ctx.remaining_seconds()) as source:
            return self._analyze_source(source, ctx, opts)

    def _analyze_source(self, source: AudioSource, ctx: AnalysisContext, opts: ChunkOptions) -> DetectorResult:
        if source.num_samples <= 0:
            raise UnsupportedInputError("audio has no samples", public_message="The audio file is empty.")
        if source.num_samples < MIN_CLIP_SECONDS * self.sample_rate:
            raise UnsupportedInputError(
                f"audio is {source.duration_seconds:.3f} s long; minimum is {MIN_CLIP_SECONDS} s",
                public_message="The audio is too short to analyse.",
            )
        window = round(opts.window_seconds * self.sample_rate)
        hop = round(opts.hop_seconds * self.sample_rate)
        spans, capped = plan_windows(source.num_samples, window, hop, opts.max_windows)

        scored: list[tuple[int, int, float]] = []
        skipped_silent = 0
        pending: list[tuple[int, int, np.ndarray]] = []
        started = time.perf_counter()

        def flush() -> None:
            if not pending:
                return
            ctx.check_deadline()
            risks = self._score_batch([segment for _, _, segment in pending], window)
            if len(risks) != len(pending):
                raise RuntimeError(f"model returned {len(risks)} scores for {len(pending)} windows")
            scored.extend((start, end, float(risk)) for (start, end, _), risk in zip(pending, risks, strict=True))
            pending.clear()

        for start, end in spans:
            ctx.check_deadline()
            segment = source.read(start, end)
            if rms_dbfs(segment) < opts.silence_dbfs:
                skipped_silent += 1
                continue
            pending.append((start, end, segment))
            if len(pending) >= opts.batch_size:
                flush()
        flush()

        if not scored:
            raise UnsupportedInputError(
                f"all {len(spans)} windows are below {opts.silence_dbfs} dBFS",
                public_message="The audio is silent or too quiet to analyse.",
            )

        risks = [risk for _, _, risk in scored]
        mean_risk = float(np.mean(risks))
        max_risk = float(np.max(risks))
        risk = min(1.0, max(0.0, aggregate(risks, opts.aggregate, opts.top_k)))
        label = "Fake" if risk >= opts.threshold else "Real"
        peak = max(scored, key=lambda item: item[2])
        sr = self.sample_rate
        details: dict[str, Any] = {
            "duration_seconds": round(source.duration_seconds, 3),
            "window_seconds": opts.window_seconds,
            "hop_seconds": opts.hop_seconds,
            "window_count": len(scored),
            "windows_planned": len(spans),
            "windows_skipped_silent": skipped_silent,
            "windows_capped": capped,
            "padded": source.num_samples < window,
            "aggregate": opts.aggregate,
            "top_k": opts.top_k if opts.aggregate == "top_k_mean" else None,
            "aggregate_risk": round(risk, 4),
            "mean_risk": round(mean_risk, 4),
            "max_risk": round(max_risk, 4),
            "threshold": opts.threshold,
            "peak_window": {"start_seconds": round(peak[0] / sr, 3), "end_seconds": round(min(peak[1], source.num_samples) / sr, 3), "risk": round(peak[2], 4)},
            "per_window_scores": [
                {"start_seconds": round(start / sr, 3), "end_seconds": round(min(end, source.num_samples) / sr, 3), "risk": round(value, 4)}
                for start, end, value in scored
            ],
            "decoder": source.decoder,
            "source_sample_rate": source.native_rate,
            "inference_seconds": round(time.perf_counter() - started, 3),
        }
        return DetectorResult(risk=risk, label=label, confidence=risk if label == "Fake" else 1.0 - risk, details=details)

    # ── helpers for export hooks ──

    def first_window(self, path: Path) -> np.ndarray:
        """Raw samples of the first non-silent window of ``path`` (the export "unit")."""
        import tempfile

        opts = self.chunk_options()
        window = round(opts.window_seconds * self.sample_rate)
        hop = round(opts.hop_seconds * self.sample_rate)
        with tempfile.TemporaryDirectory(prefix="fusiontrace-audio-") as tmp, open_audio(Path(path), Path(tmp), self.sample_rate) as source:
            spans, _ = plan_windows(source.num_samples, window, hop, opts.max_windows)
            if not spans:
                raise UnsupportedInputError("audio has no samples", public_message="The audio file is empty.")
            for start, end in spans:
                segment = source.read(start, end)
                if rms_dbfs(segment) >= opts.silence_dbfs:
                    return segment
            return source.read(*spans[0])

