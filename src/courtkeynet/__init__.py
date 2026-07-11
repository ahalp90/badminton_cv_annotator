"""CourtKeyNet court-corner detection.

Vendored upstream model under _vendor/ (see PROVENANCE.md), finetuned weights
under weights/. Our wrapper and validity gate live at this level.
"""

from .fallback import CourtQuad, scene_court
from .wrapper import CornerDetection, CourtKeyNetDetector, scene_corners

__all__ = ["CornerDetection", "CourtKeyNetDetector", "CourtQuad", "scene_corners", "scene_court"]
