"""Config-driven detector selection.

Each media kind has a package (``app.detectors.image``, ``.audio``, ``.video``)
that exports two names:

    DETECTORS: dict[str, type[Detector]]   # registry key -> class
    DEFAULT: str | None                    # key used when config says "default"

Which detector runs is decided by ``config.yaml`` (``detectors.<kind>``) with
environment overrides, so models can be switched or disabled without code
changes:

    FUSIONTRACE_<KIND>_ENABLED=true|false
    FUSIONTRACE_<KIND>_MODEL=<registry key>
    FUSIONTRACE_<KIND>_WEIGHT=<float>
    FUSIONTRACE_MODEL_DIR=<root for downloaded weights>
    FUSIONTRACE_DEVICE=auto|cpu|cuda

Frozen for the v2 upgrade alongside ``base.py``.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import Detector, DetectorError, MediaKind, ModelUnavailableError

logger = logging.getLogger("fusiontrace")

_PACKAGES: dict[MediaKind, str] = {
    MediaKind.IMAGE: "app.detectors.image",
    MediaKind.AUDIO: "app.detectors.audio",
    MediaKind.VIDEO: "app.detectors.video",
}

_instances: dict[tuple[MediaKind, str], Detector] = {}
_overrides: dict[MediaKind, Detector | None] = {}
_lock = threading.Lock()


@dataclass(frozen=True)
class DetectorSettings:
    kind: MediaKind
    enabled: bool
    model: str | None  # resolved registry key; None when the package declares no default
    weight: float
    options: dict[str, Any] = field(default_factory=dict)


def _app_config() -> dict[str, Any]:
    from ..config import config

    return config


def _base_dir() -> Path:
    from ..config import BASE_DIR

    return BASE_DIR


def _as_bool(raw: str | None, fallback: bool) -> bool:
    if raw is None or raw.strip() == "":
        return fallback
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def catalogue(kind: MediaKind) -> tuple[dict[str, type[Detector]], str | None]:
    """Return the detectors a package registers and its declared default."""
    module = importlib.import_module(_PACKAGES[kind])
    return dict(getattr(module, "DETECTORS", {})), getattr(module, "DEFAULT", None)


def settings(kind: MediaKind) -> DetectorSettings:
    section = (_app_config().get("detectors") or {}).get(kind.value) or {}
    prefix = f"FUSIONTRACE_{kind.value.upper()}"
    enabled = _as_bool(os.getenv(f"{prefix}_ENABLED"), bool(section.get("enabled", True)))
    model = os.getenv(f"{prefix}_MODEL") or section.get("model") or "default"
    if model == "default":
        model = catalogue(kind)[1]
    weight = float(os.getenv(f"{prefix}_WEIGHT") or section.get("weight", 1.0))
    return DetectorSettings(kind=kind, enabled=enabled, model=model, weight=weight, options=dict(section.get("options") or {}))


def model_dir() -> Path:
    raw = os.getenv("FUSIONTRACE_MODEL_DIR") or _app_config().get("model_dir") or "models"
    path = Path(raw)
    return path if path.is_absolute() else (_base_dir() / path).resolve()


def device() -> str:
    choice = (os.getenv("FUSIONTRACE_DEVICE") or "auto").strip().lower()
    if choice != "auto":
        return choice
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get(kind: MediaKind) -> Detector | None:
    """The configured detector instance for ``kind``, or None when disabled.

    Instances are process-wide singletons per (kind, model). Construction does
    not load weights; call ``load()`` or ``analyze()`` for that.
    """
    with _lock:
        if kind in _overrides:
            return _overrides[kind]
    current = settings(kind)
    if not current.enabled:
        return None
    detectors, _ = catalogue(kind)
    if current.model is None or current.model not in detectors:
        raise ModelUnavailableError(
            f"{kind.value} detector {current.model!r} is not registered; available: {sorted(detectors)}"
        )
    key = (kind, current.model)
    with _lock:
        instance = _instances.get(key)
        if instance is None:
            instance = detectors[current.model](model_dir=model_dir(), device=device(), options=current.options)
            _instances[key] = instance
        return instance


def all_detector_classes() -> list[type[Detector]]:
    """Every registered detector class across kinds (for downloads and evaluation)."""
    classes: list[type[Detector]] = []
    for kind in MediaKind:
        classes.extend(catalogue(kind)[0].values())
    return classes


def warm_up() -> None:
    """Load every enabled detector; log and continue on failure."""
    for kind in MediaKind:
        try:
            detector = get(kind)
            if detector is not None:
                detector.load()
                logger.info("Detector ready: %s (%s)", detector.info.name, kind.value)
        except DetectorError as exc:
            logger.warning("Detector warm-up failed for %s: %s", kind.value, exc)


@contextmanager
def override(kind: MediaKind, detector: Detector | None) -> Iterator[None]:
    """Test helper: make ``get(kind)`` return ``detector`` (None = disabled) inside the block."""
    with _lock:
        previous = _overrides.get(kind, ...)
        _overrides[kind] = detector
    try:
        yield
    finally:
        with _lock:
            if previous is ...:
                _overrides.pop(kind, None)
            else:
                _overrides[kind] = previous


def reset() -> None:
    """Drop cached instances (tests, or after a config change)."""
    with _lock:
        _instances.clear()
        _overrides.clear()
