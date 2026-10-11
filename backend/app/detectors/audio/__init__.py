"""Audio detectors. Owned by workstream B.

Register every implementation here; ``DEFAULT`` is used when config says ``model: default``.
Every detector scores the whole clip in sliding windows (see ``windows.py`` for
the options and ``docs/model-choice-audio.md`` for the model choice).
"""

from __future__ import annotations

from ..base import Detector
from .hf_models import Wav2Vec2LegacyDetector, Wav2Vec2MelodyMachineDetector, XlsrGaryStaffordDetector

DETECTORS: dict[str, type[Detector]] = {
    cls.info.name: cls for cls in (Wav2Vec2LegacyDetector, XlsrGaryStaffordDetector, Wav2Vec2MelodyMachineDetector)
}
DEFAULT: str | None = Wav2Vec2LegacyDetector.info.name

__all__ = ["DEFAULT", "DETECTORS", "Wav2Vec2LegacyDetector", "Wav2Vec2MelodyMachineDetector", "XlsrGaryStaffordDetector"]
