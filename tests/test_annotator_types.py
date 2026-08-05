"""Contracts for the annotator's declarations and composition resolver."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from annotator.config import (
    BaseAnnotatorConfig,
    ResolvedAnnotatorConfig,
    SHIPPED_THRESHOLDS,
)
from annotator.fps_constants import ScalingKind as FpsScalingKind, scale_for_fps
from annotator.rally_segmentation import scale_thresholds
from annotator.resolve import resolve
from annotator.types import DeadMaskMode, ReentryGuardVariant, ScalingKind, Slot, SmoothingMode
from src.bst_x.preparing_data.heuristics.sticky_anchor import SLOT_BOTTOM, SLOT_TOP


_FIELDS = {
    'rest_speed': (0.002, ScalingKind.PER_FRAME_SPEED),
    'rest_window': (5.0, ScalingKind.FRAME_COUNT),
    'start_speed': (0.015, ScalingKind.PER_FRAME_SPEED),
    'start_min_frames': (3.0, ScalingKind.FRAME_COUNT),
    'smooth_window': (3.0, ScalingKind.FRAME_COUNT),
    'end_rest_frames': (90.0, ScalingKind.FRAME_COUNT),
    'court_absent_window': (15.0, ScalingKind.FRAME_COUNT),
    'impulse_floor_half_window_frames': (12.0, ScalingKind.FRAME_COUNT),
    'contact_dedup_radius_frames': (3.0, ScalingKind.FRAME_COUNT),
    'contact_suppression_radius_frames': (9.0, ScalingKind.FRAME_COUNT),
    'serve_start_lookback_frames': (25.0, ScalingKind.FRAME_COUNT),
    'serve_stillness_window_frames': (15.0, ScalingKind.FRAME_COUNT),
    'sustained_loss_frames': (10.0, ScalingKind.FRAME_COUNT),
    'min_descend_samples': (3.0, ScalingKind.FRAME_COUNT),
    'body_unit_half_window': (12.0, ScalingKind.FRAME_COUNT),
    'composition_min_scene_len': (15.0, ScalingKind.FRAME_COUNT),
}


def test_scaling_kind_remains_reexported_from_types() -> None:
    assert ScalingKind is FpsScalingKind


def test_scaling_kind_matches_every_fps_constant_field() -> None:
    for fps in (25.0, 30.0, 50.0, 60.0):
        constants = scale_for_fps(fps)
        for field, (base30, kind) in _FIELDS.items():
            result = kind.scale(base30, fps)
            assert getattr(constants, field) == result
            if kind is ScalingKind.FRAME_COUNT:
                assert type(result) is int


@pytest.mark.parametrize('kind', list(ScalingKind))
@pytest.mark.parametrize('fps', (0.0, -25.0, float('nan'), float('inf')))
def test_scaling_kind_rejects_invalid_fps(kind: ScalingKind, fps: float) -> None:
    with pytest.raises(ValueError, match='fps must be positive and finite'):
        kind.scale(1.0, fps)


def test_slot_values_pin_sticky_storage_rows() -> None:
    assert Slot.TOP == SLOT_TOP
    assert Slot.BOTTOM == SLOT_BOTTOM
    rows = ['row 0', 'row 1']
    assert rows[Slot.TOP] == 'row 0'
    assert rows[Slot.BOTTOM] == 'row 1'


@pytest.mark.parametrize('fps', (25.0, 30.0, 50.0, 60.0))
def test_resolve_composes_final_constants_and_thresholds(fps: float) -> None:
    base = BaseAnnotatorConfig()
    resolved = resolve(base, fps)
    expected_constants = scale_for_fps(fps)
    expected_thresholds = scale_thresholds(SHIPPED_THRESHOLDS, fps)
    assert resolved == ResolvedAnnotatorConfig(
        fps=fps, constants=expected_constants, thresholds=expected_thresholds,
        dead_mask_mode=DeadMaskMode.REPLAY,
        smoothing_mode=SmoothingMode.IGNORE_INVISIBLE,
        gap_state_demotion_bound=ScalingKind.FRAME_COUNT.scale(75.0, fps),
        reentry_guard_variant=ReentryGuardVariant.TWO_SIDED,
        reentry_guard_buffer=0.05,
    )
    assert base.thresholds == SHIPPED_THRESHOLDS
    assert resolved.dead_mask_mode is DeadMaskMode.REPLAY
    assert resolved.smoothing_mode is SmoothingMode.IGNORE_INVISIBLE


def test_resolve_preserves_dead_mask_mode_without_fps_scaling() -> None:
    base = BaseAnnotatorConfig(dead_mask_mode=DeadMaskMode.UNION)
    resolved = resolve(base, 50.0)
    assert resolved.dead_mask_mode is DeadMaskMode.UNION


def test_resolve_preserves_smoothing_mode_without_fps_scaling() -> None:
    base = BaseAnnotatorConfig(smoothing_mode=SmoothingMode.ZERO_FILL)
    resolved = resolve(base, 50.0)
    assert resolved.smoothing_mode is SmoothingMode.ZERO_FILL


def test_config_dataclasses_are_frozen() -> None:
    base = BaseAnnotatorConfig()
    resolved = resolve(base, 30.0)
    with pytest.raises(FrozenInstanceError):
        base.thresholds = SHIPPED_THRESHOLDS  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        resolved.fps = 60.0  # type: ignore[misc]


def test_scaling_kind_invalid_guard_matches_scale_for_fps_message() -> None:
    for fps in (0.0, -25.0, float('nan'), float('inf')):
        with pytest.raises(ValueError) as scaling_error:
            ScalingKind.DIMENSIONLESS.scale(1.0, fps)
        with pytest.raises(ValueError) as table_error:
            scale_for_fps(fps)
        assert str(scaling_error.value) == str(table_error.value)
        assert math.isnan(fps) or 'fps' in str(scaling_error.value)
