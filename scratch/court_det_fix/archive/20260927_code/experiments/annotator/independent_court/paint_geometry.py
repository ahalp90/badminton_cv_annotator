"""Physical paint positions in coordinates whose corners are outside boundaries.

BWF Laws, Diagram A specifies 40 mm stripes, 420 mm between side stripes,
720 mm between rear stripes and 1.98 m from the net to the short-service paint.
The interval order matches the existing experimental template. Finite segments
extend through crossing paint to the outside boundary of the joined marking.
"""

from __future__ import annotations

import numpy as np

from courtkeynet.court_corners import COURT_LENGTH_M, COURT_WIDTH_M

STRIPE_WIDTH_M = 0.04
POSITION_OFFSETS_M = np.array([0.0, -STRIPE_WIDTH_M / 2, STRIPE_WIDTH_M / 2])
SIDE_PAINT_GAP_M = 0.42
REAR_PAINT_GAP_M = 0.72
NET_TO_SERVICE_PAINT_M = 1.98

_HALF_STRIPE = STRIPE_WIDTH_M / 2
_SINGLES_CENTRE = STRIPE_WIDTH_M + SIDE_PAINT_GAP_M + _HALF_STRIPE
_LONG_SERVICE_CENTRE = STRIPE_WIDTH_M + REAR_PAINT_GAP_M + _HALF_STRIPE
_FAR_SHORT_EDGE = COURT_LENGTH_M / 2 - NET_TO_SERVICE_PAINT_M
_NEAR_SHORT_EDGE = COURT_LENGTH_M - _FAR_SHORT_EDGE
_CENTRE_X = COURT_WIDTH_M / 2

CENTRE_SEGMENTS_M = np.array([
    [(_HALF_STRIPE, 0), (_HALF_STRIPE, COURT_LENGTH_M)],
    [(_SINGLES_CENTRE, 0), (_SINGLES_CENTRE, COURT_LENGTH_M)],
    [(_CENTRE_X, 0), (_CENTRE_X, _FAR_SHORT_EDGE)],
    [(_CENTRE_X, _NEAR_SHORT_EDGE), (_CENTRE_X, COURT_LENGTH_M)],
    [(COURT_WIDTH_M - _SINGLES_CENTRE, 0), (COURT_WIDTH_M - _SINGLES_CENTRE, COURT_LENGTH_M)],
    [(COURT_WIDTH_M - _HALF_STRIPE, 0), (COURT_WIDTH_M - _HALF_STRIPE, COURT_LENGTH_M)],
    [(0, _HALF_STRIPE), (COURT_WIDTH_M, _HALF_STRIPE)],
    [(0, _LONG_SERVICE_CENTRE), (COURT_WIDTH_M, _LONG_SERVICE_CENTRE)],
    [(0, _FAR_SHORT_EDGE - _HALF_STRIPE), (COURT_WIDTH_M, _FAR_SHORT_EDGE - _HALF_STRIPE)],
    [(0, _NEAR_SHORT_EDGE + _HALF_STRIPE), (COURT_WIDTH_M, _NEAR_SHORT_EDGE + _HALF_STRIPE)],
    [(0, COURT_LENGTH_M - _LONG_SERVICE_CENTRE), (COURT_WIDTH_M, COURT_LENGTH_M - _LONG_SERVICE_CENTRE)],
    [(0, COURT_LENGTH_M - _HALF_STRIPE), (COURT_WIDTH_M, COURT_LENGTH_M - _HALF_STRIPE)],
], dtype=np.float64)


def positioned_segments(centres: np.ndarray, intervals: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """:return: Finite centre or edge segments for each interval/position pair."""
    normals = np.zeros((len(intervals), 2))
    normals[intervals < 6, 0] = 1
    normals[intervals >= 6, 1] = 1
    return centres[intervals] + POSITION_OFFSETS_M[positions, None, None] * normals[:, None]
