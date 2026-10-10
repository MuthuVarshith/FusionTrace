"""Stub detectors for tests and tooling smoke checks. Never registered for production use.

Use with ``registry.override``::

    with registry.override(MediaKind.IMAGE, StubDetector.for_kind(MediaKind.IMAGE, risk=0.8)):
        ...
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .base import AnalysisContext, Detector, DetectorError, DetectorInfo, DetectorResult, MediaKind


class StubDetector(Detector):
    """Returns a fixed risk, or raises a given DetectorError. No weights, no torch."""

    info = DetectorInfo(name="stub", kind=MediaKind.IMAGE, display_name="Stub detector", source="local", license="MIT")

    def __init__(self, risk: float = 0.5, error: DetectorError | None = None, info: DetectorInfo | None = None) -> None:
        super().__init__(model_dir=Path("."), device="cpu")
        self.risk = risk
        self.error = error
        self.calls: list[Path] = []
        if info is not None:
            self.info = info  # type: ignore[misc]  # per-instance override is fine for a stub

    @classmethod
    def for_kind(cls, kind: MediaKind, risk: float = 0.5, error: DetectorError | None = None) -> StubDetector:
        info = DetectorInfo(name=f"stub-{kind.value}", kind=kind, display_name=f"Stub {kind.value} detector", source="local", license="MIT")
        return cls(risk=risk, error=error, info=info)

    def _load(self) -> None:
        pass

    def _analyze(self, path: Path, ctx: AnalysisContext) -> DetectorResult:
        self.calls.append(path)
        if self.error is not None:
            raise self.error
        label = "Fake" if self.risk >= 0.5 else "Real"
        return DetectorResult(risk=self.risk, label=label, confidence=max(self.risk, 1 - self.risk), details={"stub": True})


class TinyTorchDetector(Detector):
    """A real (tiny) torch model behind the full interface, including export hooks.

    For exercising ONNX export, quantisation and benchmarking code paths without
    downloading weights. Treats any input path as a 3x32x32 image of zeros.
    """

    info = DetectorInfo(name="tiny-torch", kind=MediaKind.IMAGE, display_name="Tiny torch test model", source="local", license="MIT")
    exportable = True

    def _load(self) -> None:
        import torch

        torch.manual_seed(0)
        self._module = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 1),
        ).eval()

    def _analyze(self, path: Path, ctx: AnalysisContext) -> DetectorResult:
        import torch

        with torch.no_grad():
            outputs = self._module(*self.prepare_inputs(path))
        risk = self.risk_from_outputs([outputs])
        label = "Fake" if risk >= 0.5 else "Real"
        return DetectorResult(risk=risk, label=label, confidence=max(risk, 1 - risk))

    def export_module(self) -> Any:
        self.load()
        return self._module

    def prepare_inputs(self, path: Path) -> tuple[Any, ...]:
        import torch

        return (torch.zeros(1, 3, 32, 32),)

    def risk_from_outputs(self, outputs: Sequence[Any]) -> float:
        import numpy as np

        logit = float(np.asarray(outputs[0] if not hasattr(outputs[0], "detach") else outputs[0].detach().cpu()).reshape(-1)[0])
        return float(1 / (1 + np.exp(-logit)))
