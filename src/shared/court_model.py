"""Official BWF court dimensions and painted lines, in court metres.

x runs across the court from the left doubles sideline (0) to the right one
(6.10). y runs along it from the far baseline (0) to the near baseline (13.40).
On footage from behind a baseline, the image's top corners are the far baseline.
Corners are in TL, TR, BR, BL order.

This module holds NumPy constants only. It is separate from shared.court, which
imports pandas and uses a rotated (length, width) reference rectangle.
"""

from __future__ import annotations

import numpy as np

COURT_WIDTH_M = 6.10  # doubles sideline to doubles sideline
COURT_LENGTH_M = 13.40  # baseline to baseline

# (corner, xy) in TL TR BR BL order. float32, so float32 homographies project in float32.
CORNER_COURT_M = np.array(
    [[0.0, 0.0], [COURT_WIDTH_M, 0.0], [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]],
    dtype=np.float32,
)

# Where the inner painted lines sit.
SINGLES_INSET_M = 0.46  # singles sideline in from each doubles sideline
LONG_SERVICE_INSET_M = 0.76  # doubles long service line in from each baseline
SHORT_SERVICE_OFFSET_M = 1.98  # short service line either side of the net
NET_Y_M = COURT_LENGTH_M / 2.0  # the net is not painted on the floor
CENTRE_X_M = COURT_WIDTH_M / 2.0
FAR_SHORT_SERVICE_Y_M = NET_Y_M - SHORT_SERVICE_OFFSET_M  # 4.72
NEAR_SHORT_SERVICE_Y_M = NET_Y_M + SHORT_SERVICE_OFFSET_M  # 8.68

# Painted lines as float32 (start, end) endpoint pairs. The centre line stops at
# each short service line, so it is two segments. The first six are constant-x
# lines, left to right; the last six are constant-y lines, far to near. Code that
# indexes lines by position relies on this order.
PAINTED_SEGMENTS_M: tuple[tuple[np.ndarray, np.ndarray], ...] = tuple(
    (np.array(start, dtype=np.float32), np.array(end, dtype=np.float32))
    for start, end in (
        ((0.0, 0.0), (0.0, COURT_LENGTH_M)),  # left doubles sideline
        ((SINGLES_INSET_M, 0.0), (SINGLES_INSET_M, COURT_LENGTH_M)),  # left singles sideline
        ((CENTRE_X_M, 0.0), (CENTRE_X_M, FAR_SHORT_SERVICE_Y_M)),  # centre line, far half
        ((CENTRE_X_M, NEAR_SHORT_SERVICE_Y_M), (CENTRE_X_M, COURT_LENGTH_M)),  # centre line, near half
        ((COURT_WIDTH_M - SINGLES_INSET_M, 0.0), (COURT_WIDTH_M - SINGLES_INSET_M, COURT_LENGTH_M)),  # right singles
        ((COURT_WIDTH_M, 0.0), (COURT_WIDTH_M, COURT_LENGTH_M)),  # right doubles sideline
        ((0.0, 0.0), (COURT_WIDTH_M, 0.0)),  # far baseline
        ((0.0, LONG_SERVICE_INSET_M), (COURT_WIDTH_M, LONG_SERVICE_INSET_M)),  # far doubles long service
        ((0.0, FAR_SHORT_SERVICE_Y_M), (COURT_WIDTH_M, FAR_SHORT_SERVICE_Y_M)),  # far short service
        ((0.0, NEAR_SHORT_SERVICE_Y_M), (COURT_WIDTH_M, NEAR_SHORT_SERVICE_Y_M)),  # near short service
        ((0.0, COURT_LENGTH_M - LONG_SERVICE_INSET_M), (COURT_WIDTH_M, COURT_LENGTH_M - LONG_SERVICE_INSET_M)),
        ((0.0, COURT_LENGTH_M), (COURT_WIDTH_M, COURT_LENGTH_M)),  # near baseline
    )
)
