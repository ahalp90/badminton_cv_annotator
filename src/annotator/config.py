"""Annotation-chain constants separated from scraper configuration.

See ``docs/scraper_pipeline/scraper_architecture.md`` for the current public
file contracts.

SCRAPE_DIR, MASKS_DIR, RALLY_SPANS_CSV and CONTACT_FRAMES_CSV are also defined
here (annotator-owned) because the annotator package consumes them directly;
scraper.config imports them inward so its own consumers keep the same names
and values.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, NamedTuple

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
    """The ten rally-segmentation trajectory-rule thresholds bundled as one value.

    One field per swept constant above, so a caller can hand ``segment_video`` a
    whole threshold set instead of leaning on the module globals. ``thresholds=None``
    reads the globals (the default path); a preset here reads its fields instead.
    One preset ships: SHIPPED_THRESHOLDS (the constants above, selected by the
    segmentation sweep). PROXIMITY_MAX is not swept, so it stays a plain global
    and is not carried here.
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
# Reprojected-corner displacement between adjacent segment homographies, as a
# fraction of frame size. Spec names the constant without a default; 0.05 is
# the build's starting value from the mid-July 2026 amateur-footage scoping.
PERSPECTIVE_SHIFT_THRESHOLD = 0.05
# Median speed under this fraction of rally median = slow-mo. 0.15 is swept
# against the decontaminated baseline (records/decontam_frac_sweep, autograder
# docs); the old 0.3 was tuned against the pre-decontamination norm and read
# rally-tail deceleration as slow motion.
SLOWMO_SPEED_FRAC = 0.15

# Composition dead-mask (`composition_mask`), the per-segment alternative to
# the replay mask. A PySceneDetect content pass cuts the timeline; each segment is
# kept or dropped by the court-view vote. content threshold 27 with vote 0.5
# (comp_content27_v0p5) is the config the sset_01 scoring picked.
COMPOSITION_CONTENT_THRESHOLD = 27.0  # PySceneDetect ContentDetector default
COMPOSITION_KEEP_VOTE = 0.5  # a cut segment is live when >= this fraction of its frames vote court-view

# ---------------------------------------------------------------------------
# Doubles guard windowing
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

    The preset carries legacy 25fps-surface values for fps-sensitive fields.
    Resolution overwrites every fps-sensitive field from the shipped base-30 table.
    ``overrides_base30`` may replace named rows before their final per-fps
    values are built. Strategy fields (dead-mask producer, smoothing, and
    serve lanes) are carried by the same preset.
    """

    thresholds: RallySegmentationThresholds = SHIPPED_THRESHOLDS
    dead_mask_mode: DeadMaskMode = DeadMaskMode.REPLAY
    # Chosen together on 2026-07-28: ignore invisible coordinates during
    # smoothing, then classify sustained gaps with the ruled two-sided re-entry
    # guard.
    smoothing_mode: SmoothingMode = SmoothingMode.IGNORE_INVISIBLE
    overrides_base30: Mapping[str, float] | None = None
    span_open: SpanOpen | None = SpanOpen.BACK_FILL
    gap_state_demotion_bound: float | None = 75.0
    reentry_guard_variant: ReentryGuardVariant | None = ReentryGuardVariant.TWO_SIDED
    reentry_guard_buffer: float | None = 0.05
    quiet_start_window: float | None = None
    # Shipping default selected in commit 3f7621b (2026-07-22): rejecting all
    # three grades raised recorded correct landing calls from 59 to 72 of 287;
    # rejecting only proven-fabricated frames produced 46. frozenset() disables
    # event rejection entirely.
    rejected_grades: frozenset[int] = frozenset({1, 2, 3})

    def __post_init__(self) -> None:
        if not isinstance(self.rejected_grades, frozenset):
            raise ValueError('rejected_grades must be a frozenset')
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
