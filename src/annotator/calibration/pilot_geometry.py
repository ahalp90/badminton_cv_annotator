"""Pilot-vid-1 court geometry for the serve-start builders.

Shared by the stage-8 tests and the frozen sweep runner; calibration is its home
because the fixtures/harness package already pins per-fixture geometry. The sweep
script retires at Stage 7.

LOUD CAVEAT ON THE STAND-IN COURT REGION. These constants are PILOT-VIDEO-1-ONLY,
data-derived stand-ins for the court quad (production gets CourtKeyNet). They are
the axis-quantile foot-point box and padded pixel-height band fitted in
start_side_scoping.py from the top-2-score in-rally detections. NOT a measured
court quad; do not reuse for another video (one broadcast camera, one court
framing).
"""

from scraper.stage8_rally_segmentation import CourtBox


PILOT_STANDIN_COURT_X = (635.0, 1316.0)  # foot-point x bounds, pixels (frame is 1920 wide)
PILOT_STANDIN_COURT_Y = (254.0, 1030.0)  # foot-point y bounds, pixels (frame is 1080 tall)

# The homography alternative the --court-box flag selects. Bounding box of ShuttleSet's recorded
# court quad for pilot vid 1, mapped into the 1920x1080 working space by homography_check.py
# (verified 2026-07-10 against ShuttleSet's static, per-video homography; the four corners live in
# local_scratch/autograder_architecture/pilot_results/homography_view_rule/homography_check.csv).
# Being the ACTUAL court-outline quad's bounding box, it sits TIGHTER than the occupancy stand-ins
# above: players legitimately stand outside it (lunges, clears past the baseline), and courtside
# officials sit outside it but inside the stand-in box.
PILOT_HOMOG_COURT_X = (460.8, 1459.5)  # quad x-range, pixels (downleft .. downright corners)
PILOT_HOMOG_COURT_Y = (461.1, 1006.8)  # quad y-range, pixels (upright .. downleft corners)
# The net line's image y under the same homography (H^-1 of the court-space mid-line, spread
# under 2.2 px across the court width; homography_check.py, 2026-07-10). Perspective
# compresses the far half, so no arithmetic midpoint lands here: the stand-in half-split
# (642) sits 42 px above the real net, the quad-y midpoint (~734) 50 px below it.
PILOT_HOMOG_COURT_MID_Y = 683.9
# The homography half-split is a BAND, not a knife edge: the recorded homography binds a flat
# template to a monocular image of a 3D net (own height, minor x/y/z sag), so the floor-plane
# net line carries model error beyond its numeric spread (Ariel's ruling, 2026-07-10). The
# band is +/-0.5 m along court length at the net, through H^-1 at centre-court x; feet inside
# it count to NEITHER court half. Vid 15's band is (583.9, 626.6) for the coming
# generalisation window.
PILOT_HOMOG_COURT_MID_BAND = (664.6, 703.7)

PILOT_PLAYER_HEIGHT = (84.0, 336.0)  # court-player bbox pixel-height band
PILOT_RESOLUTION = (1920.0, 1080.0)  # (W, H) the shuttle xy and the bbox centres normalise by

# The stand-in half-split is a zero-width band at the y-midpoint: the stand-in split never had
# a buffer, so a foot below the midpoint is bottom-half and above it top-half (the audited
# anchors pin this). The homography half-split is the buffered net-line band above.
PILOT_STANDIN_COURT_MID_Y = (PILOT_STANDIN_COURT_Y[0] + PILOT_STANDIN_COURT_Y[1]) / 2.0  # 642 px
PILOT_STANDIN_COURT_MID_BAND = (PILOT_STANDIN_COURT_MID_Y, PILOT_STANDIN_COURT_MID_Y)

# The two CourtBoxes the runner hands to the segmenter's serve-start builders. The height band
# stays the stand-in under both choices (the homography fixes the court outline, not bbox
# pixel heights). --court-box selects between them; stand-in is the default, bit-identical to
# before the flag existed.
STANDIN_COURT_BOX = CourtBox(
    x_range=PILOT_STANDIN_COURT_X, y_range=PILOT_STANDIN_COURT_Y,
    height_band=PILOT_PLAYER_HEIGHT, mid_band=PILOT_STANDIN_COURT_MID_BAND,
)
HOMOGRAPHY_COURT_BOX = CourtBox(
    x_range=PILOT_HOMOG_COURT_X, y_range=PILOT_HOMOG_COURT_Y,
    height_band=PILOT_PLAYER_HEIGHT, mid_band=PILOT_HOMOG_COURT_MID_BAND,
)
