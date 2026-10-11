"""Video (temporal) detectors. Owned by workstream F.

Register every implementation here; ``DEFAULT`` is used when config says ``model: default``.

``frame-consistency-heuristic`` is the default because no permissively
licensed, CPU-friendly learned temporal model passed the selection criteria;
see docs/model-choice-video.md for every candidate considered.
"""

from __future__ import annotations

from ..base import Detector
from .heuristic import FrameConsistencyHeuristic

DETECTORS: dict[str, type[Detector]] = {
    FrameConsistencyHeuristic.info.name: FrameConsistencyHeuristic,
}
DEFAULT: str | None = FrameConsistencyHeuristic.info.name
