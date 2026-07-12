"""CourtKeyNet court-corner detection.

Vendored upstream model under _vendor/ (see PROVENANCE.md), finetuned weights
under weights/. Our wrapper and validity gate live at this level.

The wrapper imports torch, so its three exports (CornerDetection,
CourtKeyNetDetector, scene_corners) load lazily through __getattr__ (PEP 562):
``from src.courtkeynet import CourtKeyNetDetector`` still works, but importing the
package, or pulling the fallback's court constants, no longer drags torch in.
That is what lets the hand-annotation tool run in an OpenCV-only environment (see
fallback.py and validation_scripts/annotate_court_corners.py).
"""

from typing import TYPE_CHECKING

from .fallback import CourtQuad, scene_court

if TYPE_CHECKING:
    # Static-analysis types for the lazily-exported wrapper names; keeps __all__ honest.
    from .wrapper import CornerDetection, CourtKeyNetDetector, scene_corners

__all__ = ["CornerDetection", "CourtKeyNetDetector", "CourtQuad", "scene_corners", "scene_court"]

# Resolved lazily by __getattr__ below (see the module docstring for why).
_LAZY_WRAPPER_EXPORTS = frozenset({"CornerDetection", "CourtKeyNetDetector", "scene_corners"})


def __getattr__(name: str) -> object:
    """Load a wrapper export on first access (PEP 562), importing torch only then."""
    if name in _LAZY_WRAPPER_EXPORTS:
        from . import wrapper

        return getattr(wrapper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
