"""Preprocessing settings saved with the models, then scaled for each video.

Base frame counts and speeds use 30 fps. ``scale_config_to_fps`` converts them to the
video's frame rate. Changing preprocessing changes the model inputs, so
fitted models must use the settings they were trained with.
"""
from collections.abc import Mapping
from dataclasses import dataclass, fields

from annotator.excluded_frames.shuttle_hallucination import ShuttleGrade
from annotator.types import SmoothingMode
from shared.frame_timing import frames_at_fps, speed_at_fps

REST_SPEED_BASE30 = 0.002
START_SPEED_BASE30 = 0.015


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
    # Minimum excluded run to retain; independent of the court-absence window.
    replay_mask_min_frames: int
    impulse_floor_half_window_frames: int
    contact_dedup_radius_frames: int
    contact_suppression_radius_frames: int
    sustained_loss_frames: int
    min_descend_samples: int
    body_unit_half_window: int
    # Longer missing-shuttle gaps count as rest unless explained by a high shot.
    max_unexplained_shuttle_gap_frames: int
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

    ``require_fast_burst`` keeps an active region only if it contains sustained
    fast motion. Rallies always start at the active region's beginning.
    ``max_high_shot_gap_frames`` limits the missing frames allowed for a high
    shot, at 30 fps. ``reentry_guard_buffer`` is the top-edge margin as a fraction
    of image height, checked before and after a visibility gap.
    ``rejected_grades`` selects shuttle grades excluded from speed evidence
    and final-flight/landing decisions.
    """

    # Invisible coordinates must not pull a smoothed trajectory towards zero.
    smoothing_mode: SmoothingMode = SmoothingMode.IGNORE_INVISIBLE
    overrides_base30: Mapping[str, float] | None = None
    require_fast_burst: bool = True
    max_high_shot_gap_frames: float = 75.0
    reentry_guard_buffer: float = 0.05
    # Shuttle grades to exclude when estimating speed and final flights.
    rejected_grades: frozenset[int] = ShuttleGrade.rejectable()


@dataclass(frozen=True)
class FpsScaledAnnotatorConfig:
    """Preprocessing settings scaled to this video's frame rate.

    ``thresholds`` contains the converted numeric defaults/overrides.
    ``max_high_shot_gap_frames`` is also converted to video frames. Other
    choices are copied from the saved base settings.
    """

    fps: float
    thresholds: Thresholds
    smoothing_mode: SmoothingMode
    require_fast_burst: bool
    max_high_shot_gap_frames: int
    reentry_guard_buffer: float
    rejected_grades: frozenset[int]


def scale_config_to_fps(base: BaseAnnotatorConfig, fps: float) -> FpsScaledAnnotatorConfig:
    """Convert the saved preprocessing settings to the video frame rate."""
    overrides = base.overrides_base30
    if overrides is not None:
        unknown_settings = sorted(set(overrides) - _OVERRIDABLE_THRESHOLD_FIELDS)
        if unknown_settings:
            raise ValueError(f'unknown base-30 override settings: {unknown_settings}')
    thresholds = _scale_thresholds_to_fps(fps, overrides)

    return FpsScaledAnnotatorConfig(
        fps=fps,
        thresholds=thresholds,
        smoothing_mode=base.smoothing_mode,
        require_fast_burst=base.require_fast_burst,
        max_high_shot_gap_frames=frames_at_fps(base.max_high_shot_gap_frames, fps),
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
        replay_mask_min_frames=frame_count('replay_mask_min_frames', 15.0),
        impulse_floor_half_window_frames=frame_count('impulse_floor_half_window_frames', 12.0),
        contact_dedup_radius_frames=frame_count('contact_dedup_radius_frames', 3.0),
        contact_suppression_radius_frames=frame_count('contact_suppression_radius_frames', 9.0),
        sustained_loss_frames=frame_count('sustained_loss_frames', 10.0),
        min_descend_samples=frame_count('min_descend_samples', 3.0),
        body_unit_half_window=frame_count('body_unit_half_window', 12.0),
        max_unexplained_shuttle_gap_frames=frame_count('max_unexplained_shuttle_gap_frames', 12.0),
        high_shot_oob_lookback_frames=frame_count('high_shot_oob_lookback_frames', 6.0),
        # Speed calculations divide by sample count minus one; need at least two frames.
        high_shot_oob_min_visible_frames=max(2, frame_count('high_shot_oob_min_visible_frames', 2.4)),
        high_shot_oob_extrap_frames=frame_count('high_shot_oob_extrap_frames', 12.0),
        reentry_lookahead_frames=frame_count('reentry_lookahead_frames', 6.0),
        reentry_min_visible_frames=max(2, frame_count('reentry_min_visible_frames', 2.4)),
        contact_impulse_multiple=base30.get('contact_impulse_multiple', 4.0),
        wrist_distance_body_heights=base30.get('wrist_distance_body_heights', 1.4),
    )
