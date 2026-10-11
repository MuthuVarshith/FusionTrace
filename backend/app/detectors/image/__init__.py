"""Image detectors. Owned by workstream A.

Register every implementation here; ``DEFAULT`` is used when config says ``model: default``.
Model choice and measurements: ``docs/model-choice-image.md``.
"""

from __future__ import annotations

from ..base import Detector
from .efficientnet_legacy import EfficientNetV2LegacyDetector

DETECTORS: dict[str, type[Detector]] = {
    EfficientNetV2LegacyDetector.info.name: EfficientNetV2LegacyDetector,
}
DEFAULT: str | None = EfficientNetV2LegacyDetector.info.name
