"""Audio detectors. Owned by workstream B.

Register every implementation here; ``DEFAULT`` is used when config says ``model: default``.
"""

from __future__ import annotations

from ..base import Detector

DETECTORS: dict[str, type[Detector]] = {}
DEFAULT: str | None = None
