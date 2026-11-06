"""Preprocessing settings saved with the models, then scaled for each video.

Base frame counts and speeds use 30 fps. ``scale_config_to_fps`` converts them to the
video's frame rate. Changing preprocessing changes the model inputs, so
fitted models must use the settings they were trained with.
"""
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields

from annotator.types import ReentryGuardVariant, SmoothingMode, SpanOpen

# ---------------------------------------------------------------------------
# Replay and off-rally masking rules
# ---------------------------------------------------------------------------
# A lower speed fraction avoids mistaking rally-end deceleration for slow motion.
SLOWMO_SPEED_FRAC = 0.15

# Scene-based exclusion uses camera cuts and the fraction of court-view frames.
COMPOSITION_CONTENT_THRESHOLD = 27.0  # PySceneDetect ContentDetector default
COMPOSITION_KEEP_VOTE = 0.5  # a cut segment is live when >= this fraction of its frames vote court-view

BASE_FPS = 30.0
REST_SPEED_BASE30 = 0.002
START_SPEED_BASE30 = 0.015


def frames_at_fps(frames_at_30_fps: float, fps: float) -> int:
    """Scale a base-30 frame count, rounding halves up and keeping at least one frame."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f'fps must be positive and finite, got {fps!r}')
    return max(1, math.floor(frames_at_30_fps * fps / BASE_FPS + 0.5))


def speed_at_fps(speed_at_30_fps: float, fps: float) -> float:
    """Convert speed per frame from 30 fps to the video's frame rate."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f'fps must be positive and finite, got {fps!r}')
    return speed_at_30_fps * BASE_FPS / fps


@dataclass(frozen=True)
class Thresholds:
    """Numeric settings after conversion to the video's frame rate.

    Windows and durations are in video frames. Speeds are normalised image
    distance per frame. Impulse ratios and distances in body heights do not
    change with frame rate.
    """
    rest_speed: float
    rest_window: int
    start_speed: float
    start_min_frames: int
    smooth_window: int
    end_rest_frames: int
    court_absent_window: int
    # Minimum excluded run to retain; independent of the court-absence window.
    replay_mask_min_frames: int
    impulse_floor_half_window_frames: int
    contact_dedup_radius_frames: int
    contact_suppression_radius_frames: int
    sustained_loss_frames: int
    min_descend_samples: int
    body_unit_half_window: int
    # Minimum camera-scene length for cut detection.
    composition_min_scene_len: int
    blip_max_frames: int
    high_shot_oob_lookback_frames: int
    high_shot_oob_min_visible_frames: int
    high_shot_oob_extrap_frames: int
    reentry_lookahead_frames: int
    reentry_min_visible_frames: int
    contact_impulse_multiple: float
    wrist_distance_body_heights: float


_OVERRIDABLE_THRESHOLD_FIELDS = frozenset(field.name for field in fields(Thresholds))


@dataclass(frozen=True)
class BaseAnnotatorConfig:
    """Preprocessing choices saved with a fitted model bundle.

    ``overrides_base30`` replaces named Thresholds defaults before scaling.
    Frame counts and speeds in those overrides use 30 fps.

    ``gap_state_demotion_bound`` limits how long a missing shuttle can be
    treated as a high shot still in play; in frames at 30 fps. None selects
    the simpler rule based on speed and the fraction of visible frames.
    ``reentry_guard_buffer`` is the top-edge distance as a fraction of image
    height. Set both re-entry settings to None to skip the return check.
    ``rejected_grades`` selects shuttle grades excluded from speed evidence
    and final-flight/landing decisions.
    """

    # Invisible coordinates must not pull a smoothed trajectory towards zero.
    smoothing_mode: SmoothingMode = SmoothingMode.IGNORE_INVISIBLE
    overrides_base30: Mapping[str, float] | None = None
    span_open: SpanOpen | None = SpanOpen.BACK_FILL
    gap_state_demotion_bound: float | None = 75.0
    reentry_guard_variant: ReentryGuardVariant | None = ReentryGuardVariant.TWO_SIDED
    reentry_guard_buffer: float | None = 0.05
    # Shuttle grades to exclude when estimating speed and final flights.
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
class FpsScaledAnnotatorConfig:
    """Preprocessing settings scaled to this video's frame rate.

    ``thresholds`` contains the converted numeric defaults/overrides.
    ``gap_state_demotion_bound`` is also converted to video frames. Other
    choices are copied from the saved base settings.
    """

    fps: float
    thresholds: Thresholds
    smoothing_mode: SmoothingMode
    span_open: SpanOpen | None
    gap_state_demotion_bound: int | None
    reentry_guard_variant: ReentryGuardVariant | None
    reentry_guard_buffer: float | None
    rejected_grades: frozenset[int]


def scale_config_to_fps(base: BaseAnnotatorConfig, fps: float) -> FpsScaledAnnotatorConfig:
    """Convert the saved preprocessing settings to the video frame rate."""
    overrides = base.overrides_base30
    if overrides is not None:
        unknown_settings = sorted(set(overrides) - _OVERRIDABLE_THRESHOLD_FIELDS)
        if unknown_settings:
            raise ValueError(f'unknown base-30 override settings: {unknown_settings}')
    thresholds = _scale_thresholds_to_fps(fps, overrides)

    def frame_count(value: float | None) -> int | None:
        return None if value is None else frames_at_fps(value, fps)
    return FpsScaledAnnotatorConfig(
        fps=fps,
        thresholds=thresholds,
        smoothing_mode=base.smoothing_mode,
        span_open=base.span_open,
        gap_state_demotion_bound=frame_count(base.gap_state_demotion_bound),
        reentry_guard_variant=base.reentry_guard_variant,
        reentry_guard_buffer=base.reentry_guard_buffer,
        rejected_grades=base.rejected_grades,
    )


def _scale_thresholds_to_fps(fps: float, overrides_base30: Mapping[str, float] | None) -> Thresholds:
    """Apply numeric overrides, then convert frame counts and speeds from 30 fps."""
    base30 = {} if overrides_base30 is None else overrides_base30

    def frame_count(name: str, default: float) -> int:
        return frames_at_fps(base30.get(name, default), fps)

    def speed(name: str, default: float) -> float:
        return speed_at_fps(base30.get(name, default), fps)

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
        sustained_loss_frames=frame_count('sustained_loss_frames', 10.0),
        min_descend_samples=frame_count('min_descend_samples', 3.0),
        body_unit_half_window=frame_count('body_unit_half_window', 12.0),
        composition_min_scene_len=frame_count('composition_min_scene_len', 15.0),
        blip_max_frames=frame_count('blip_max_frames', 12.0),
        high_shot_oob_lookback_frames=frame_count('high_shot_oob_lookback_frames', 6.0),
        # Speed calculations divide by sample count minus one; need at least two frames.
        high_shot_oob_min_visible_frames=max(2, frame_count('high_shot_oob_min_visible_frames', 2.4)),
        high_shot_oob_extrap_frames=frame_count('high_shot_oob_extrap_frames', 12.0),
        reentry_lookahead_frames=frame_count('reentry_lookahead_frames', 6.0),
        reentry_min_visible_frames=max(2, frame_count('reentry_min_visible_frames', 2.4)),
        contact_impulse_multiple=base30.get('contact_impulse_multiple', 4.0),
        wrist_distance_body_heights=base30.get('wrist_distance_body_heights', 1.4),
    )
