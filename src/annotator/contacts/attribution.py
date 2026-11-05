"""Assign a contact to a court half using the player's current camera scene."""

from collections.abc import Sequence

import numpy as np

from annotator.courts.scenes import SceneCourt, court_at_frame
from annotator.outcomes.point_winner import CourtHalf
from annotator.types import StickyResult


def attribute_half(
    frame: int, track: np.ndarray, sticky: StickyResult, bboxes: np.ndarray, scene_courts: Sequence[SceneCourt],
) -> CourtHalf | None:
    """Locate the nearest tracked player's feet relative to this scene's net band.

    Invisible shuttles, missing wrist evidence and frames without accepted scene
    geometry leave the player unknown. A video-wide homography can be reused in
    every scene without changing this rule.
    """
    if track[frame, 2] != 1:
        return None
    distances = sticky.distances_per_slot[frame]
    if not np.isfinite(distances).any():
        return None
    try:
        band_lo, band_hi = court_at_frame(scene_courts, frame).net_band
    except ValueError:
        # Serve look-back candidates can precede accepted camera geometry.
        return None
    # A finite value beats sentinels; exact ties retain the far/top tracker pick.
    half = int(np.nanargmin(distances))
    slot = sticky.picks[frame, half]
    foot_y = float(bboxes[frame, slot][3])
    if foot_y < band_lo:
        return CourtHalf.TOP
    if foot_y > band_hi:
        return CourtHalf.BOT
    return None
