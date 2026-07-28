"""Annotation-chain constants, split from scraper.config at Stage 2.

Constant provenance is cited inline as "spec sN" against the section of
local_scratch/autograder_architecture/scraper_spec.md it came from.

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
# Scrape-output paths (dataset_schema.md section 2 tree)
# ---------------------------------------------------------------------------
# One scrape root holds the flat CSVs plus the per-video sidecar dirs. Default
# sits under the repo's gitignored data/ tree; BADMINTON_SCRAPE_DIR overrides.
_REPO_ROOT = Path(__file__).resolve().parents[2]
SCRAPE_DIR = Path(os.environ.get('BADMINTON_SCRAPE_DIR', _REPO_ROOT / 'data' / 'scrape_output'))
MASKS_DIR = SCRAPE_DIR / 'masks'  # schema s2 (stage 9)
RALLY_SPANS_CSV = SCRAPE_DIR / 'rally_spans.csv'  # spec s6 (stage 8)
CONTACT_FRAMES_CSV = SCRAPE_DIR / 'contact_frames.csv'  # spec s6 (stage 8)

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
END_REST_FRAMES = _AT_25FPS.end_rest_frames  # rally end: extended rest of at least this (~3.0 s)
PROXIMITY_MAX = 0.15  # norm court units; player-proximity cross-check (guardrail column)


class Stage8Thresholds(NamedTuple):
    """The ten stage-8 trajectory-rule thresholds bundled as one value.

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
    impulse_floor_half_window_frames: int = _AT_25FPS.impulse_floor_half_window_frames
    contact_dedup_radius_frames: int = _AT_25FPS.contact_dedup_radius_frames
    contact_suppression_radius_frames: int = _AT_25FPS.contact_suppression_radius_frames
    contact_impulse_multiple: float = 4.0


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
# Median speed under this fraction of rally median = slow-mo. 0.15 is swept
# against the decontaminated baseline (records/decontam_frac_sweep, autograder
# docs); the old 0.3 was tuned against the pre-decontamination norm and read
# rally-tail deceleration as slow motion.
SLOWMO_SPEED_FRAC = 0.15

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

    The preset carries legacy 25fps-surface values for fps-sensitive fields.
    Resolution overwrites every fps-sensitive field from the shipped base-30 table.
    ``overrides_base30`` may replace named rows before their final per-fps
    values are built. Strategy fields (dead-mask producer, smoothing, and
    serve lanes) arrive with their stages.
    """

    thresholds: Stage8Thresholds = SHIPPED_THRESHOLDS
    dead_mask_mode: DeadMaskMode = DeadMaskMode.REPLAY
    # Measured together in W2.9: ignore invisible coordinates during smoothing,
    # then classify sustained gaps with the ruled two-sided re-entry guard.
    smoothing_mode: SmoothingMode = SmoothingMode.IGNORE_INVISIBLE
    overrides_base30: Mapping[str, float] | None = None
    span_open: SpanOpen | None = SpanOpen.BACK_FILL
    gap_state_demotion_bound: float | None = 75.0
    reentry_guard_variant: ReentryGuardVariant | None = ReentryGuardVariant.TWO_SIDED
    reentry_guard_buffer: float | None = 0.05
    quiet_start_window: float | None = None
    # Shipping default from the three-arm remeasure (2026-07-22): rejecting all three
    # inpaint grades scored best on every fixture; record in the campaign docs at
    # records/commit12_default_pick.md. frozenset() disables event rejection entirely.
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
    already-scaled precondition). ``constants`` is deliberately unwired until
    threading.
    """

    fps: float
    constants: FpsConstants
    thresholds: Stage8Thresholds
    dead_mask_mode: DeadMaskMode
    smoothing_mode: SmoothingMode
    span_open: SpanOpen | None = SpanOpen.BACK_FILL
    gap_state_demotion_bound: int | None = None
    reentry_guard_variant: ReentryGuardVariant | None = None
    reentry_guard_buffer: float | None = None
    quiet_start_window: int | None = None
    rejected_grades: frozenset[int] = frozenset({1, 2, 3})
