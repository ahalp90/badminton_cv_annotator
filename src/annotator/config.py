"""Annotation-chain constants, split from scraper.config at Stage 2.

Constant provenance is cited inline as "spec sN" against the section of
local_scratch/autograder_architecture/scraper_spec.md it came from.
"""
from dataclasses import dataclass
from typing import NamedTuple

from .fps_constants import FpsConstants, scale_for_fps

# ---------------------------------------------------------------------------
# Stage 8: rally segmentation and contact rules (spec s6)
# ---------------------------------------------------------------------------
# Speed means per-frame L2 displacement of (x_norm, y_norm) on visibility-1
# frames. fps_constants.py stores the base-30 table; these globals are its
# scaling to the legacy 25 fps surface (the original tuning fixtures).
_AT_25FPS = scale_for_fps(25.0)
REST_SPEED = _AT_25FPS.rest_speed  # norm-units/frame; a span is at rest below this
REST_WINDOW = _AT_25FPS.rest_window  # frames (~0.16 s at 25 fps)
START_SPEED = _AT_25FPS.start_speed  # rally start: speed above this...
START_MIN_FRAMES = _AT_25FPS.start_min_frames  # ...for this many consecutive frames out of rest
SMOOTH_WINDOW = _AT_25FPS.smooth_window  # moving-average window over (x, y) to survive TrackNetV3 jitter
MIN_DIR_CHANGE_DEG = 30  # contact: smoothed-velocity direction change beyond this
MIN_CONTACT_SPEED = 0.005  # with pre- and post-reversal speed above this
END_REST_FRAMES = _AT_25FPS.end_rest_frames  # rally end: extended rest of at least this (~3.0 s)
PROXIMITY_MAX = 0.15  # norm court units; player-proximity cross-check (guardrail column)


class Stage8Thresholds(NamedTuple):
    """The eight stage-8 trajectory-rule thresholds bundled as one value.

    One field per swept constant above, so a caller can hand ``segment_video`` a
    whole threshold set instead of leaning on the module globals. ``thresholds=None``
    reads the globals (the default path); a preset here reads its fields instead.
    One preset ships: SHIPPED_THRESHOLDS (the constants above, the block-2 sweep
    pick); BEST_CONFIG_THRESHOLDS aliases it so the CLI 'best' preset keeps
    selecting the same values. PROXIMITY_MAX is not swept, so it stays a plain
    global and is not carried here.
    """

    rest_speed: float
    rest_window: int
    end_rest_frames: int
    start_speed: float
    start_min_frames: int
    smooth_window: int
    min_dir_change_deg: float
    min_contact_speed: float
    impulse_floor_half_window_frames: int = _AT_25FPS.impulse_floor_half_window_frames
    contact_dedup_radius_frames: int = _AT_25FPS.contact_dedup_radius_frames
    contact_suppression_radius_frames: int = _AT_25FPS.contact_suppression_radius_frames


# The shipped thresholds as one value, built from the constants above so the
# numbers live in exactly one place. segment_video(thresholds=SHIPPED_THRESHOLDS)
# is equivalent to the default globals path.
SHIPPED_THRESHOLDS = Stage8Thresholds(
    rest_speed=REST_SPEED,
    rest_window=REST_WINDOW,
    end_rest_frames=END_REST_FRAMES,
    start_speed=START_SPEED,
    start_min_frames=START_MIN_FRAMES,
    smooth_window=SMOOTH_WINDOW,
    min_dir_change_deg=MIN_DIR_CHANGE_DEG,
    min_contact_speed=MIN_CONTACT_SPEED,
)

# The block-2 widened-sweep pick under the merge-penalised selection key is now
# the shipped default (the constants above). The name survives as an alias so
# the CLI 'best' preset keeps working.
BEST_CONFIG_THRESHOLDS = SHIPPED_THRESHOLDS

# ---------------------------------------------------------------------------
# Stage 9: replay and off-rally rules (spec s7)
# ---------------------------------------------------------------------------
COURT_ABSENT_WINDOW = _AT_25FPS.court_absent_window  # frames of court-present False to fire the signal
# Reprojected-corner displacement between adjacent segment homographies, as a
# fraction of frame size. Spec names the constant without a default; 0.05 is
# the build's starting value, tuned at B5.
PERSPECTIVE_SHIFT_THRESHOLD = 0.05
SLOWMO_SPEED_FRAC = 0.3  # median speed under this fraction of rally median = slow-mo

# Composition dead-mask (stage9_composition_mask), the per-segment alternative to
# the replay mask. A PySceneDetect content pass cuts the timeline; each segment is
# kept or dropped by the court-view vote. content threshold 27 with vote 0.5
# (comp_content27_v0p5) is the config the pilot scoring picked.
COMPOSITION_CONTENT_THRESHOLD = 27.0  # PySceneDetect ContentDetector default
COMPOSITION_KEEP_VOTE = 0.5  # a cut segment is live when >= this fraction of its frames vote court-view

# ---------------------------------------------------------------------------
# Doubles guard windowing (spec s8)
# ---------------------------------------------------------------------------
# A clip- or segment-level doubles flag fires only when the per-frame
# over-count (>2 in-court candidates) holds across more than half the frames
# of a rally span. Fraction only (ruled 2026-07-07): a consecutive-run leg
# would fire on any passerby crossing the court. Transient walk-throughs
# (a coach or ball-kid crossing) stay unflagged. Starting value.
DOUBLES_SPAN_FRACTION = 0.5


@dataclass(frozen=True)
class BaseAnnotatorConfig:
    """Preset carrying the non-fps knobs for an annotator run.

    The preset carries ``min_dir_change_deg`` and ``min_contact_speed`` plus
    legacy 25fps-surface values for fps-sensitive fields. Resolution overwrites
    every fps-sensitive field from the base-30 table, so only the non-fps knobs
    survive a custom preset. This is not a caller-supplied base-30 table; that
    table lives in ``fps_constants``. Strategy fields (dead-mask producer and
    serve lanes) arrive with their stages.
    """

    thresholds: Stage8Thresholds = SHIPPED_THRESHOLDS


@dataclass(frozen=True)
class ResolvedAnnotatorConfig:
    """Final per-video configuration, built once and never rescaled.

    ``thresholds`` is the run_video-ready value (run_video declares the
    already-scaled precondition). ``constants`` is deliberately unwired until
    threading.
    """

    fps: float
    constants: FpsConstants
    thresholds: Stage8Thresholds
