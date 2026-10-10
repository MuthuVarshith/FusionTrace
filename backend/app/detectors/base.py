"""Shared detector contract for FusionTrace v2.

Every model that contributes an evidence signal (image, audio, video-temporal)
implements :class:`Detector`. The pipeline, the registry, the weight download
script, the evaluation script and the ONNX export tooling talk to models only
through this module. It is frozen for the duration of the v2 upgrade: changes
go through the lead, never through an individual workstream.

Conventions
-----------
* ``risk`` is always P(synthetic or manipulated) in [0, 1]; higher means more
  suspicious. A model whose native output is P(real) must invert it.
* Construction is cheap. Weights load lazily via :meth:`Detector.load`, which is
  thread-safe and idempotent.
* :meth:`Detector.analyze` must be safe to call concurrently from several
  threads on one instance. Anything that mutates model state (gradients,
  hooks, train mode) belongs in :meth:`Detector.explain`, which must isolate
  that state per call.
* Failures raise a :class:`DetectorError` subclass. ``str(exc)`` may contain
  internals (paths, library messages) and goes to the server log only;
  ``exc.public_message`` is the only text that may reach a client.
"""

from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:  # torch is only needed by concrete detectors and export tooling
    import torch


class MediaKind(str, Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


# ── Typed errors ────────────────────────────────────────────────────────────


class DetectorError(Exception):
    """Base class for every detector failure."""

    public_message: str = "The detector could not analyse this file."

    def __init__(self, message: str = "", *, public_message: str | None = None) -> None:
        super().__init__(message or self.public_message)
        if public_message is not None:
            self.public_message = public_message


class ModelUnavailableError(DetectorError):
    """Weights are missing, failed to load, or the configured model name is unknown."""

    public_message = "The detection model is not available on this server."


class UnsupportedInputError(DetectorError):
    """The file decodes but this detector cannot analyse it (too short, no face, ...)."""

    public_message = "This file could not be analysed by the detector."


class InferenceError(DetectorError):
    """Unexpected failure while running the model."""

    public_message = "The detector failed while analysing this file."


class AnalysisTimeoutError(DetectorError):
    """The per-scan processing budget ran out."""

    public_message = "Analysis exceeded the configured time limit."


# ── Value types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WeightsSpec:
    """One downloadable weight bundle, consumed by ``scripts/download_models.py``.

    Files land in ``<model_dir>/<local_subdir>``. Pin ``revision`` to a commit
    sha so a build is reproducible.
    """

    repo_id: str
    local_subdir: str
    revision: str | None = None
    allow_patterns: tuple[str, ...] | None = None
    repo_type: str = "model"
    repo_id_env: str | None = None  # env var that overrides repo_id, e.g. "HF_REPO_ID" for a private mirror


@dataclass(frozen=True)
class DetectorInfo:
    """Static description of a detector; rendered in the UI, report and PDF."""

    name: str  # stable registry key, kebab-case, e.g. "efficientnetv2-legacy"
    kind: MediaKind
    display_name: str  # e.g. "EfficientNetV2-L (FusionTrace legacy)"
    source: str  # HF repo id, or "local" for bundled weights
    license: str  # SPDX id where possible: "MIT", "Apache-2.0", "unknown"
    version: str = ""  # revision / commit / weights hash prefix
    heuristic: bool = False  # True for non-learned signals; the UI labels them as heuristics
    description: str = ""

    def to_report(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


@dataclass
class DetectorResult:
    """Outcome of one ``analyze`` call.

    ``details`` must be JSON-serialisable; it is stored verbatim in the report
    (per-chunk scores, per-clip scores, thresholds, timings...).
    """

    risk: float  # P(synthetic or manipulated), [0, 1]
    label: str  # "Fake" or "Real", at the detector's own threshold
    confidence: float  # confidence in ``label``, [0, 1]
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("risk", "confidence"):
            value = float(getattr(self, name))
            if math.isnan(value) or value < -1e-6 or value > 1 + 1e-6:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
            setattr(self, name, min(1.0, max(0.0, value)))
        if self.label not in ("Fake", "Real"):
            raise ValueError(f"label must be 'Fake' or 'Real', got {self.label!r}")


@dataclass(frozen=True)
class FrameResult:
    """An image-detector result for one sampled video frame."""

    timestamp_seconds: float
    path: Path
    result: DetectorResult


@dataclass
class AnalysisContext:
    """Per-call inputs that are not the media file itself."""

    work_dir: Path  # per-scan scratch directory; deleted when the scan finishes
    options: Mapping[str, Any] = field(default_factory=dict)  # config ``detectors.<kind>.options``
    deadline: float | None = None  # ``time.monotonic()`` value; None means no budget
    frame_results: Sequence[FrameResult] | None = None  # video only: per-frame image results from the pipeline

    def remaining_seconds(self) -> float | None:
        return None if self.deadline is None else self.deadline - time.monotonic()

    def check_deadline(self) -> None:
        remaining = self.remaining_seconds()
        if remaining is not None and remaining <= 0:
            raise AnalysisTimeoutError("processing budget exhausted")


# ── The interface ───────────────────────────────────────────────────────────


class Detector(ABC):
    """Base class for every model-backed (or heuristic) evidence signal.

    Subclasses set ``info`` and implement ``_load`` and ``_analyze``. Callers
    use ``load``/``analyze``/``explain``, never the underscored methods.
    """

    info: ClassVar[DetectorInfo]
    supports_explanation: ClassVar[bool] = False
    exportable: ClassVar[bool] = False  # True when the export hooks below are implemented

    def __init__(self, model_dir: Path, device: str = "cpu", options: Mapping[str, Any] | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.options: dict[str, Any] = dict(options or {})
        self._load_lock = threading.Lock()
        self._loaded = False

    # ── lifecycle ──

    @classmethod
    def weights(cls) -> list[WeightsSpec]:
        """Weight bundles this detector needs. Empty for heuristics."""
        return []

    def is_available(self) -> bool:
        """Cheap check that weights are present locally, without loading them."""
        return True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load weights once. Thread-safe; concurrent callers block until ready."""
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                self._load()
            except DetectorError:
                raise
            except Exception as exc:
                raise ModelUnavailableError(f"{self.info.name}: failed to load: {exc}") from exc
            self._loaded = True

    @abstractmethod
    def _load(self) -> None:
        """Load weights into memory. Called at most once, under the load lock."""

    # ── inference ──

    def analyze(self, path: Path, ctx: AnalysisContext) -> DetectorResult:
        """Score one file. Wraps unexpected exceptions in :class:`InferenceError`."""
        self.load()
        ctx.check_deadline()
        try:
            return self._analyze(Path(path), ctx)
        except DetectorError:
            raise
        except Exception as exc:
            raise InferenceError(f"{self.info.name} failed on {Path(path).name}: {exc}") from exc

    @abstractmethod
    def _analyze(self, path: Path, ctx: AnalysisContext) -> DetectorResult:
        """Score one file. May raise DetectorError subclasses directly."""

    def explain(self, path: Path, out_path: Path, ctx: AnalysisContext) -> Path | None:
        """Write a visual explanation (e.g. a GradCAM overlay JPEG) to ``out_path``.

        Return the written path, or None when no explanation is available. Must
        not share gradients, hooks or mutable module state with concurrent
        ``analyze``/``explain`` calls (use a private model copy or a dedicated
        lock). Only meaningful when ``supports_explanation`` is True.
        """
        return None

    # ── export hooks (ONNX / quantisation / benchmarks) ──
    # A detector sets ``exportable = True`` and implements all three. The
    # "unit" is one model forward: one image, one audio window, one video clip.

    def export_module(self) -> torch.nn.Module:
        """The loaded torch module whose forward maps ``prepare_inputs`` output to raw outputs."""
        raise NotImplementedError

    def prepare_inputs(self, path: Path) -> tuple[torch.Tensor, ...]:
        """Preprocess the first unit of ``path`` into the tensors ``export_module`` expects."""
        raise NotImplementedError

    def risk_from_outputs(self, outputs: Sequence[Any]) -> float:
        """Map raw forward outputs (torch tensors or numpy arrays) for one unit to risk in [0, 1]."""
        raise NotImplementedError
