"""Calculate player evidence for a contact using the current camera scene."""

from collections.abc import Sequence

import numpy as np

from annotator.courts.scenes import SceneCourt, court_at_frame
from annotator.types import CourtHalf, StickyResult


def contact_player_evidence(
    frame: int, track: np.ndarray, sticky: StickyResult, bboxes: np.ndarray, scene_courts: Sequence[SceneCourt],
) -> tuple[CourtHalf | None, tuple[float, float] | None]:
    """Return the nearest player's half and optional TOP/BOT wrist gaps.

    Wrist gaps are in body-height units. A single tracked player can identify
    the hitter; comparing alternating orders needs one player on each half.
    Invisible shuttles and missing wrist evidence leave both results unknown.
    """
    if track[frame, 2] != 1:
        return None, None
    distances = sticky.distances_per_slot[frame]
    finite_distances = np.isfinite(distances)
    # Serve look-back frames before an accepted scene have no tracker distances.
    if not finite_distances.any():
        return None, None
    band_lo, band_hi = court_at_frame(scene_courts, frame).net_band

    # A finite value beats sentinels; exact ties retain the far/top tracker pick.
    nearest = int(np.nanargmin(distances))
    slot = sticky.picks[frame, nearest]
    foot_y = float(bboxes[frame, slot, 3])
    side = None
    if foot_y < band_lo:
        side = CourtHalf.TOP
    elif foot_y > band_hi:
        side = CourtHalf.BOT

    # A missing player or two picks on the same half cannot supply competing costs.
    if not finite_distances.all():
        return side, None
    first_foot, second_foot = bboxes[frame, sticky.picks[frame], 3]
    first_distance, second_distance = map(float, distances)
    if first_foot < band_lo and second_foot > band_hi:
        return side, (first_distance, second_distance)
    if second_foot < band_lo and first_foot > band_hi:
        return side, (second_distance, first_distance)
    return side, None
