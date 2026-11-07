"""Annotator types and frame-rate scaling."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from annotator.config import BaseAnnotatorConfig, scale_config_to_fps
from annotator.types import PlayerRow, SmoothingMode
from shared.frame_timing import frames_at_fps, speed_at_fps
from shared.player_tracking import BOTTOM_ROW, TOP_ROW

_FIELDS = {
    'rest_speed': (0.002, speed_at_fps),
    'rest_window': (5.0, frames_at_fps),
    'start_speed': (0.015, speed_at_fps),
    'start_min_frames': (3.0, frames_at_fps),
    'smooth_window': (3.0, frames_at_fps),
    'end_rest_frames': (90.0, frames_at_fps),
    'impulse_floor_half_window_frames': (12.0, frames_at_fps),
    'contact_dedup_radius_frames': (3.0, frames_at_fps),
    'contact_suppression_radius_frames': (9.0, frames_at_fps),
    'sustained_loss_frames': (10.0, frames_at_fps),
    'min_descend_samples': (3.0, frames_at_fps),
    'body_unit_half_window': (12.0, frames_at_fps),
}


def test_scaling_kind_matches_every_fps_constant_field() -> None:
    for fps in (25.0, 30.0, 50.0, 60.0):
        constants = scale_config_to_fps(BaseAnnotatorConfig(), fps).thresholds
        for field, (base30, kind) in _FIELDS.items():
            result = kind(base30, fps)
            assert getattr(constants, field) == result
            if kind is frames_at_fps:
                assert type(result) is int


@pytest.mark.parametrize('kind', [frames_at_fps, speed_at_fps])
@pytest.mark.parametrize('fps', (0.0, -25.0, float('nan'), float('inf')))
def test_scaling_kind_rejects_invalid_fps(kind, fps: float) -> None:
    with pytest.raises(ValueError, match='fps must be positive and finite'):
        kind(1.0, fps)


def test_slot_values_pin_sticky_storage_rows() -> None:
    assert PlayerRow.TOP == TOP_ROW
    assert PlayerRow.BOTTOM == BOTTOM_ROW
    rows = ['row 0', 'row 1']
    assert rows[PlayerRow.TOP] == 'row 0'
    assert rows[PlayerRow.BOTTOM] == 'row 1'


def test_scale_preserves_smoothing_mode_without_fps_scaling() -> None:
    base = BaseAnnotatorConfig(smoothing_mode=SmoothingMode.INCLUDE_INVISIBLE)
    video_config = scale_config_to_fps(base, 50.0)
    assert video_config.smoothing_mode is SmoothingMode.INCLUDE_INVISIBLE


def test_config_dataclasses_are_frozen() -> None:
    base = BaseAnnotatorConfig()
    video_config = scale_config_to_fps(base, 30.0)
    with pytest.raises(FrozenInstanceError):
        base.smoothing_mode = SmoothingMode.INCLUDE_INVISIBLE  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        video_config.fps = 60.0  # type: ignore[misc]


def test_scaled_thresholds_are_frozen() -> None:
    video_config = scale_config_to_fps(BaseAnnotatorConfig(), 30.0)
    with pytest.raises(FrozenInstanceError):
        video_config.thresholds.rest_speed = 0.5  # type: ignore[misc]
