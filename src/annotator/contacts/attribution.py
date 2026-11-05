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

    Invisible shuttles and missing wrist evidence leave the player unknown.
    Tracking and court geometry share the same scene bounds. A video-wide
    homography can be reused in every scene without changing this rule.
    """
    if track[frame, 2] != 1:
        return None
    distances = sticky.distances_per_slot[frame]
    # Serve look-back frames before an accepted scene have no tracker distances.
    if not np.isfinite(distances).any():
        return None
    band_lo, band_hi = court_at_frame(scene_courts, frame).net_band
    # A finite value beats sentinels; exact ties retain the far/top tracker pick.
    half = int(np.nanargmin(distances))
    slot = sticky.picks[frame, half]
    foot_y = float(bboxes[frame, slot][3])
    if foot_y < band_lo:
        return CourtHalf.TOP
    if foot_y > band_hi:
        return CourtHalf.BOT
    return None


def wrist_distances_by_half(
    frame: int, track: np.ndarray, sticky: StickyResult, bboxes: np.ndarray, scene_courts: Sequence[SceneCourt],
) -> tuple[float, float] | None:
    """Return TOP/BOT wrist gaps in body heights when both players are visible.

    Compare both alternating orders on the same evidence. A missing player or
    two picks on the same court half cannot supply a pair of competing costs.
    """
    distances = sticky.distances_per_slot[frame]
    if track[frame, 2] != 1 or not np.isfinite(distances).all():
        return None
    band_lo, band_hi = court_at_frame(scene_courts, frame).net_band
    first_foot, second_foot = bboxes[frame, sticky.picks[frame], 3]
    first_distance, second_distance = map(float, distances)
    if first_foot < band_lo and second_foot > band_hi:
        return first_distance, second_distance
    if second_foot < band_lo and first_foot > band_hi:
        return second_distance, first_distance
    return None
