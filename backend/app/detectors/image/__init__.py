"""Image detectors. Owned by workstream A.

Register every implementation here; ``DEFAULT`` is used when config says ``model: default``.
Model choice and measurements: ``docs/model-choice-image.md``.
"""

from __future__ import annotations

from ..base import Detector
from .commfor import CommunityForensicsDetector
from .efficientnet_legacy import EfficientNetV2LegacyDetector
from .gend_clip import GenDClipDetector
from .siglip_deepfake import SiglipDeepfakeDetector

DETECTORS: dict[str, type[Detector]] = {
    cls.info.name: cls
    for cls in (CommunityForensicsDetector, GenDClipDetector, SiglipDeepfakeDetector, EfficientNetV2LegacyDetector)
}
DEFAULT: str | None = EfficientNetV2LegacyDetector.info.name
