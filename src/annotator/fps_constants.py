"""Frame-rate-relative annotator constants.

All caller-supplied frame counts and per-frame speeds on the annotator surface
are base-30 values. They are scaled exactly once when fps context exists.
Fields of :class:`FpsConstants` are final and never rescaled. YouTube sources
are assumed CFR; :func:`probe_fps` rejects variable-frame-rate files loudly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path

from annotator.video_metadata import probe_video_fps

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
class FpsConstants:
    """Final fps-scaled values; the base-30 table is scaled exactly once."""
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


FPS_CONSTANT_FIELD_NAMES = frozenset(field.name for field in fields(FpsConstants))


def scale_for_fps(fps: float, overrides_base30: dict[str, float] | None = None) -> FpsConstants:
    """Scale the base-30 table for a positive finite CFR frame rate."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f'fps must be positive and finite, got {fps!r}')
    base30 = {} if overrides_base30 is None else overrides_base30
    def frame_count(name: str, shipped: float) -> int:
        return int(ScalingKind.FRAME_COUNT.scale(base30.get(name, shipped), fps))

    def speed(name: str, shipped: float) -> float:
        return float(ScalingKind.PER_FRAME_SPEED.scale(base30.get(name, shipped), fps))

    return FpsConstants(
        rest_speed=speed('rest_speed', REST_SPEED_BASE30),
        rest_window=frame_count('rest_window', 5.0), start_speed=speed('start_speed', START_SPEED_BASE30),
        start_min_frames=frame_count('start_min_frames', 3.0), smooth_window=frame_count('smooth_window', 3.0),
        end_rest_frames=frame_count('end_rest_frames', 90.0), court_absent_window=frame_count('court_absent_window', 15.0),
        replay_mask_min_frames=frame_count('replay_mask_min_frames', 15.0),
        impulse_floor_half_window_frames=frame_count('impulse_floor_half_window_frames', 12.0), contact_dedup_radius_frames=frame_count('contact_dedup_radius_frames', 3.0),
        contact_suppression_radius_frames=frame_count('contact_suppression_radius_frames', 9.0), serve_start_lookback_frames=frame_count('serve_start_lookback_frames', 25.0),
        serve_stillness_window_frames=frame_count('serve_stillness_window_frames', 15.0),
        sustained_loss_frames=frame_count('sustained_loss_frames', 10.0),
        min_descend_samples=frame_count('min_descend_samples', 3.0), body_unit_half_window=frame_count('body_unit_half_window', 12.0),
        composition_min_scene_len=frame_count('composition_min_scene_len', 15.0),
        blip_max_frames=frame_count('blip_max_frames', 12.0),
        high_shot_oob_lookback_frames=frame_count('high_shot_oob_lookback_frames', 6.0),
        # Consumers divide by sample count minus one, so each needs at least two frames.
        high_shot_oob_min_visible_frames=max(2, frame_count('high_shot_oob_min_visible_frames', 2.4)),
        high_shot_oob_extrap_frames=frame_count('high_shot_oob_extrap_frames', 12.0),
        reentry_lookahead_frames=frame_count('reentry_lookahead_frames', 6.0),
        reentry_min_visible_frames=max(2, frame_count('reentry_min_visible_frames', 2.4)),
    )


def probe_fps(video_path: Path) -> float:
    """Return the canonical CFR rate as a float for existing annotator callers."""
    return float(probe_video_fps(video_path))
