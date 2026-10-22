"""Fixed heuristic preprocessing settings and shared annotation output paths.

The model bundle carries the preprocessing settings used during fitting.
Changing these settings changes the inputs the trees see and requires a refit.
Direct tree-scoring settings live with the contact model.
"""
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path

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

PROXIMITY_MAX = 0.15  # normalised court distance for the proximity guardrail

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


BASE_FPS = 30.0
REST_SPEED_BASE30 = 0.002
START_SPEED_BASE30 = 0.015


class ScalingKind(StrEnum):
    """Describe how a base-30 value scales with the video's frame rate."""

    PER_FRAME_SPEED = 'per_frame_speed'
    FRAME_COUNT = 'frame_count'
    DIMENSIONLESS = 'dimensionless'

    def scale(self, value: float, fps: float) -> float | int:
        """Scale one base-30 value, requiring a positive finite frame rate."""
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'fps must be positive and finite, got {fps!r}')
        if self is ScalingKind.PER_FRAME_SPEED:
            return value * BASE_FPS / fps
        if self is ScalingKind.FRAME_COUNT:
            return max(1, math.floor(value * fps / BASE_FPS + 0.5))
        return value


@dataclass(frozen=True)
class Thresholds:
    """Numeric thresholds for one video's frame rate.

    Impulse multiples and wrist distances do not depend on frame rate.
    """
    rest_speed: float
    rest_window: int
    start_speed: float
    start_min_frames: int
    smooth_window: int
    end_rest_frames: int
    court_absent_window: int
    # Same value as court_absent_window today; distinct concept (the minimum
    # masked run a mask consumer trusts enough to act on), so tuning one never
    # silently moves the other.
    replay_mask_min_frames: int
    impulse_floor_half_window_frames: int
    contact_dedup_radius_frames: int
    contact_suppression_radius_frames: int
    serve_start_lookback_frames: int
    serve_stillness_window_frames: int
    sustained_loss_frames: int
    min_descend_samples: int
    body_unit_half_window: int
    # Same value as court_absent_window today; distinct concept (PySceneDetect's
    # minimum scene length), so tuning one never silently moves the other.
    composition_min_scene_len: int
    blip_max_frames: int
    high_shot_oob_lookback_frames: int
    high_shot_oob_min_visible_frames: int
    high_shot_oob_extrap_frames: int
    reentry_lookahead_frames: int
    reentry_min_visible_frames: int
    contact_impulse_multiple: float
    wrist_distance_body_heights: float


_OVERRIDABLE_BASE30_ROWS = frozenset(field.name for field in fields(Thresholds))


@dataclass(frozen=True)
class BaseAnnotatorConfig:
    """Heuristic preprocessing policy saved with the fitted model bundle.

    ``overrides_base30`` replaces named numeric settings before their final per-video
    values are built. These settings define model inputs and are saved with
    each fitted bundle.
    """

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

    ``thresholds`` contains every resolved numeric value. Strategy fields retain
    the selected policy; frame-count strategies are scaled with the same rules.
    """

    fps: float
    thresholds: Thresholds
    dead_mask_mode: DeadMaskMode
    smoothing_mode: SmoothingMode
    span_open: SpanOpen | None = SpanOpen.BACK_FILL
    gap_state_demotion_bound: int | None = None
    reentry_guard_variant: ReentryGuardVariant | None = None
    reentry_guard_buffer: float | None = None
    quiet_start_window: int | None = None
    rejected_grades: frozenset[int] = frozenset({1, 2, 3})


def resolve(base: BaseAnnotatorConfig, fps: float) -> ResolvedAnnotatorConfig:
    """Convert saved settings for a video's frame rate."""
    overrides = base.overrides_base30
    if overrides is not None:
        unknown_settings = sorted(set(overrides) - _OVERRIDABLE_BASE30_ROWS)
        if unknown_settings:
            raise ValueError(f'unknown base-30 override settings: {unknown_settings}')
    thresholds = _resolve_thresholds(fps, overrides)

    def frame_count(value: float | None) -> int | None:
        return None if value is None else int(ScalingKind.FRAME_COUNT.scale(value, fps))
    return ResolvedAnnotatorConfig(
        fps=fps,
        thresholds=thresholds,
        dead_mask_mode=base.dead_mask_mode,
        smoothing_mode=base.smoothing_mode,
        span_open=base.span_open,
        gap_state_demotion_bound=frame_count(base.gap_state_demotion_bound),
        reentry_guard_variant=base.reentry_guard_variant,
        reentry_guard_buffer=base.reentry_guard_buffer,
        quiet_start_window=frame_count(base.quiet_start_window),
        rejected_grades=base.rejected_grades,
    )


def _resolve_thresholds(fps: float, overrides_base30: Mapping[str, float] | None) -> Thresholds:
    """Scale the base-30 table for a positive finite CFR frame rate."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f'fps must be positive and finite, got {fps!r}')
    base30 = {} if overrides_base30 is None else overrides_base30

    def frame_count(name: str, default: float) -> int:
        return int(ScalingKind.FRAME_COUNT.scale(base30.get(name, default), fps))

    def speed(name: str, default: float) -> float:
        return float(ScalingKind.PER_FRAME_SPEED.scale(base30.get(name, default), fps))

    return Thresholds(
        rest_speed=speed('rest_speed', REST_SPEED_BASE30),
        rest_window=frame_count('rest_window', 5.0),
        start_speed=speed('start_speed', START_SPEED_BASE30),
        start_min_frames=frame_count('start_min_frames', 3.0),
        smooth_window=frame_count('smooth_window', 3.0),
        end_rest_frames=frame_count('end_rest_frames', 90.0),
        court_absent_window=frame_count('court_absent_window', 15.0),
        replay_mask_min_frames=frame_count('replay_mask_min_frames', 15.0),
        impulse_floor_half_window_frames=frame_count('impulse_floor_half_window_frames', 12.0),
        contact_dedup_radius_frames=frame_count('contact_dedup_radius_frames', 3.0),
        contact_suppression_radius_frames=frame_count('contact_suppression_radius_frames', 9.0),
        serve_start_lookback_frames=frame_count('serve_start_lookback_frames', 25.0),
        serve_stillness_window_frames=frame_count('serve_stillness_window_frames', 15.0),
        sustained_loss_frames=frame_count('sustained_loss_frames', 10.0),
        min_descend_samples=frame_count('min_descend_samples', 3.0),
        body_unit_half_window=frame_count('body_unit_half_window', 12.0),
        composition_min_scene_len=frame_count('composition_min_scene_len', 15.0),
        blip_max_frames=frame_count('blip_max_frames', 12.0),
        high_shot_oob_lookback_frames=frame_count('high_shot_oob_lookback_frames', 6.0),
        # Consumers divide by sample count minus one, so each needs two frames.
        high_shot_oob_min_visible_frames=max(2, frame_count('high_shot_oob_min_visible_frames', 2.4)),
        high_shot_oob_extrap_frames=frame_count('high_shot_oob_extrap_frames', 12.0),
        reentry_lookahead_frames=frame_count('reentry_lookahead_frames', 6.0),
        reentry_min_visible_frames=max(2, frame_count('reentry_min_visible_frames', 2.4)),
        contact_impulse_multiple=base30.get('contact_impulse_multiple', 4.0),
        wrist_distance_body_heights=base30.get('wrist_distance_body_heights', 1.4),
    )
