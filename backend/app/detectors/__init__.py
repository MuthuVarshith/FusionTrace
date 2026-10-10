"""Pluggable evidence detectors. See ``base.py`` for the contract and ``registry.py`` for selection."""

from .base import (
    AnalysisContext,
    AnalysisTimeoutError,
    Detector,
    DetectorError,
    DetectorInfo,
    DetectorResult,
    FrameResult,
    InferenceError,
    MediaKind,
    ModelUnavailableError,
    UnsupportedInputError,
    WeightsSpec,
)

__all__ = [
    "AnalysisContext",
    "AnalysisTimeoutError",
    "Detector",
    "DetectorError",
    "DetectorInfo",
    "DetectorResult",
    "FrameResult",
    "InferenceError",
    "MediaKind",
    "ModelUnavailableError",
    "UnsupportedInputError",
    "WeightsSpec",
]
