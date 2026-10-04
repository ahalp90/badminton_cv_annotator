"""Fixed heuristic preprocessing settings and shared annotation output paths.

The model bundle carries the preprocessing settings used during fitting.
Changing these settings changes the inputs the trees see and requires a refit.
Direct tree-scoring settings live with the contact model.
"""
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from .fps_constants import FpsConstants, scale_for_fps
from .types import DeadMaskMode, ReentryGuardVariant, SmoothingMode, SpanOpen

# ---------------------------------------------------------------------------
# Scrape-output paths
# ---------------------------------------------------------------------------
# One scrape root holds the flat CSVs plus the per-video sidecar dirs. Default
# sits under the repo's gitignored data/ tree; BADMINTON_SCRAPE_DIR overrides.
_REPO_ROOT = Path(__file__).resolve().parents[2]
SCRAPE_DIR = Path(os.environ.get('BADMINTON_SCRAPE_DIR', _REPO_ROOT / 'data' / 'scrape_output'))
MASKS_DIR = SCRAPE_DIR / 'masks'
RALLY_SPANS_CSV = SCRAPE_DIR / 'rally_spans.csv'
CONTACT_FRAMES_CSV = SCRAPE_DIR / 'contact_frames.csv'

# ---------------------------------------------------------------------------
# Rally segmentation and contact rules
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
END_REST_FRAMES = _AT_25FPS.end_rest_frames  # rally end: extended rest of at least this (~3.0 s)
PROXIMITY_MAX = 0.15  # norm court units; player-proximity cross-check (guardrail column)


class RallySegmentationThresholds(NamedTuple):
    """Trajectory thresholds used to find initial rallies and contact candidates.

    ``resolve`` scales frame-rate-sensitive values before annotation. The
    proximity cross-check stays separate because it measures court distance.
    """

    rest_speed: float
    rest_window: int
    end_rest_frames: int
    start_speed: float
    start_min_frames: int
    smooth_window: int
    impulse_floor_half_window_frames: int = _AT_25FPS.impulse_floor_half_window_frames
    contact_dedup_radius_frames: int = _AT_25FPS.contact_dedup_radius_frames
    contact_suppression_radius_frames: int = _AT_25FPS.contact_suppression_radius_frames
    contact_impulse_multiple: float = 4.0


# The shipped thresholds as one value, built from the constants above so the
# numbers live in exactly one place. segment_video(thresholds=SHIPPED_THRESHOLDS)
# is equivalent to the default globals path.
SHIPPED_THRESHOLDS = RallySegmentationThresholds(
    rest_speed=REST_SPEED,
    rest_window=REST_WINDOW,
    end_rest_frames=END_REST_FRAMES,
    start_speed=START_SPEED,
    start_min_frames=START_MIN_FRAMES,
    smooth_window=SMOOTH_WINDOW,
)

# ---------------------------------------------------------------------------
# Replay and off-rally masking rules
# ---------------------------------------------------------------------------
# Reprojected-corner displacement as a fraction of frame size.
PERSPECTIVE_SHIFT_THRESHOLD = 0.05
# A lower speed fraction avoids mistaking rally-end deceleration for slow motion.
SLOWMO_SPEED_FRAC = 0.15

# Scene-based exclusion uses camera cuts and the fraction of court-view frames.
COMPOSITION_CONTENT_THRESHOLD = 27.0  # PySceneDetect ContentDetector default
COMPOSITION_KEEP_VOTE = 0.5  # a cut segment is live when >= this fraction of its frames vote court-view

# ---------------------------------------------------------------------------
# Doubles guard windowing
# ---------------------------------------------------------------------------
# A clip- or segment-level doubles flag fires only when the per-frame
# over-count (>2 in-court candidates) holds across more than half the frames
# of a rally span. A short consecutive run could instead count a coach or
# ball-kid crossing the court as a doubles player.
DOUBLES_SPAN_FRACTION = 0.5


@dataclass(frozen=True)
class BaseAnnotatorConfig:
    """Heuristic preprocessing policy saved with the fitted model bundle.

    The preset carries legacy 25fps-surface values for fps-sensitive fields.
    Resolution overwrites every fps-sensitive field from the shipped base-30 table.
    ``overrides_base30`` may replace named rows before their final per-fps
    values are built. These are fixed model inputs, rather than routine
    prediction-time tuning options.
    """

    thresholds: RallySegmentationThresholds = SHIPPED_THRESHOLDS
    dead_mask_mode: DeadMaskMode = DeadMaskMode.REPLAY
    # Invisible coordinates must not pull a smoothed trajectory towards zero.
    smoothing_mode: SmoothingMode = SmoothingMode.IGNORE_INVISIBLE
    overrides_base30: Mapping[str, float] | None = None
    span_open: SpanOpen | None = SpanOpen.BACK_FILL
    gap_state_demotion_bound: float | None = 75.0
    reentry_guard_variant: ReentryGuardVariant | None = ReentryGuardVariant.TWO_SIDED
    reentry_guard_buffer: float | None = 0.05
    quiet_start_window: float | None = None
    # Reject contacts and landings on uncertain or fabricated shuttle tracks.
    rejected_grades: frozenset[int] = frozenset({1, 2, 3})

    def __post_init__(self) -> None:
        if not isinstance(self.rejected_grades, frozenset):
            raise ValueError('rejected_grades must be a frozenset')  # noqa: TRY004 -- invalid configuration value
        if any(
            isinstance(code, bool) or not isinstance(code, int) or code not in {1, 2, 3}
            for code in self.rejected_grades
        ):
            raise ValueError('rejected_grades must be a subset of {1, 2, 3}')
        guard_specified = self.reentry_guard_variant is not None or self.reentry_guard_buffer is not None
        if guard_specified and self.gap_state_demotion_bound is None:
            raise ValueError('reentry guard requires gap_state_demotion_bound')
        if (self.reentry_guard_variant is None) != (self.reentry_guard_buffer is None):
            raise ValueError('reentry guard needs both a variant and a buffer, or neither')


@dataclass(frozen=True)
class ResolvedAnnotatorConfig:
    """Final per-video configuration, built once and never rescaled.

    ``thresholds`` is the run_video-ready value (run_video declares the
    already-scaled precondition). ``constants`` holds the resolved per-FPS
    policy used throughout the run.
    """

    fps: float
    constants: FpsConstants
    thresholds: RallySegmentationThresholds
    dead_mask_mode: DeadMaskMode
    smoothing_mode: SmoothingMode
    span_open: SpanOpen | None = SpanOpen.BACK_FILL
    gap_state_demotion_bound: int | None = None
    reentry_guard_variant: ReentryGuardVariant | None = None
    reentry_guard_buffer: float | None = None
    quiet_start_window: int | None = None
    rejected_grades: frozenset[int] = frozenset({1, 2, 3})
