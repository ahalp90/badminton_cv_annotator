"""Annotator config plumbing and opt-in span strategy tests."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from annotator.config import BaseAnnotatorConfig, resolve
from annotator.rallies.find_rallies import gap_state_rest_mask
from annotator.types import ReentryGuardVariant, SmoothingMode
from tests.annotator_config import burst_config


def test_base30_overrides_resolve_all_scaling_kinds() -> None:
    base = BaseAnnotatorConfig(overrides_base30={
        'rest_window': 10.0,
        'rest_speed': 0.003,
        'contact_impulse_multiple': 5.5,
    })
    resolved = resolve(base, 25.0)
    assert resolved.thresholds.rest_window == 8
    assert resolved.thresholds.rest_speed == pytest.approx(0.0036)
    assert resolved.thresholds.contact_impulse_multiple == 5.5


def test_shipped_tracking_strategies_resolve_at_video_fps() -> None:
    resolved25 = resolve(BaseAnnotatorConfig(), 25.0)
    resolved30 = resolve(BaseAnnotatorConfig(), 30.0)
    assert resolved25.smoothing_mode is SmoothingMode.IGNORE_INVISIBLE
    assert resolved25.gap_state_demotion_bound == 63
    assert resolved30.gap_state_demotion_bound == 75
    assert resolved30.reentry_guard_variant is ReentryGuardVariant.TWO_SIDED
    assert resolved30.reentry_guard_buffer == 0.05


def test_unknown_base30_override_fails_loudly() -> None:
    with pytest.raises(ValueError, match='unknown base-30'):
        resolve(BaseAnnotatorConfig(overrides_base30={'not_a_row': 1.0}), 30.0)


def test_removed_direction_change_override_fails_loudly() -> None:
    with pytest.raises(ValueError, match='unknown base-30'):
        resolve(BaseAnnotatorConfig(overrides_base30={'min_dir_change_deg': 30.0}), 30.0)


def test_rejected_grades_are_validated_and_copied_to_resolved_config() -> None:
    base = BaseAnnotatorConfig(rejected_grades=frozenset({1, 3}))
    assert resolve(base, 30.0).rejected_grades == frozenset({1, 3})
    with pytest.raises(ValueError, match='subset of'):
        BaseAnnotatorConfig(rejected_grades=frozenset({0}))
    with pytest.raises(ValueError, match='subset of'):
        BaseAnnotatorConfig(rejected_grades=frozenset({True}))
    with pytest.raises(ValueError, match='subset of'):
        BaseAnnotatorConfig(rejected_grades=frozenset({1, False}))
    with pytest.raises(ValueError, match='frozenset'):
        BaseAnnotatorConfig(rejected_grades={1})


@pytest.mark.parametrize('kwargs', [
    {
        'gap_state_demotion_bound': None,
        'reentry_guard_variant': ReentryGuardVariant.REENTRY_ONLY,
        'reentry_guard_buffer': None,
    },
    {
        'gap_state_demotion_bound': None,
        'reentry_guard_variant': None,
        'reentry_guard_buffer': 0.1,
    },
    {
        'gap_state_demotion_bound': 10.0,
        'reentry_guard_variant': ReentryGuardVariant.REENTRY_ONLY,
        'reentry_guard_buffer': None,
    },
    {
        'gap_state_demotion_bound': 10.0,
        'reentry_guard_variant': None,
        'reentry_guard_buffer': 0.1,
    },
])
def test_invalid_gap_guard_configurations_fail_loudly(kwargs: dict) -> None:
    with pytest.raises(ValueError, match='reentry guard'):
        BaseAnnotatorConfig(**kwargs)


def test_gap_guard_can_be_disabled_as_one_strategy() -> None:
    base = BaseAnnotatorConfig(
        gap_state_demotion_bound=None,
        reentry_guard_variant=None,
        reentry_guard_buffer=None,
    )
    resolved = resolve(base, 25.0)
    assert resolved.gap_state_demotion_bound is None
    assert resolved.reentry_guard_variant is None
    assert resolved.reentry_guard_buffer is None


def test_gap_state_with_and_without_reentry_guard() -> None:
    pre_y = np.linspace(0.5, 0.05, 6)
    y = np.concatenate([pre_y, np.zeros(12), [0.5, 0.6, 0.7]])
    track = np.column_stack([np.full(len(y), 0.5), y, np.ones(len(y))])
    track[6:18, 2] = 0
    speed = np.full(len(track), np.nan)
    thresholds = replace(burst_config().thresholds, rest_window=3, rest_speed=1.0)
    unguarded = gap_state_rest_mask(speed, track, thresholds, 75, None, None)
    guarded = gap_state_rest_mask(speed, track, thresholds, 75, ReentryGuardVariant.REENTRY_ONLY, 0.1)
    assert not unguarded[10]
    assert guarded[10]


def test_gap_fps_rows_match_frozen_25fps_and_base30_values() -> None:
    values25 = resolve(BaseAnnotatorConfig(), 25.0).thresholds
    values30 = resolve(BaseAnnotatorConfig(), 30.0).thresholds
    fields = (
        'blip_max_frames', 'high_shot_oob_lookback_frames',
        'high_shot_oob_min_visible_frames', 'high_shot_oob_extrap_frames',
        'reentry_lookahead_frames', 'reentry_min_visible_frames',
    )
    assert tuple(getattr(values25, field) for field in fields) == (10, 5, 2, 10, 5, 2)
    assert tuple(getattr(values30, field) for field in fields) == (12, 6, 2, 12, 6, 2)


def test_impulse_cell_candidates_consumes_threshold_multiple() -> None:
    from annotator.contacts.heuristics import impulse_cell_candidates

    apex = np.array([0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float) * 0.05
    track = np.column_stack([np.full(len(apex), 0.5), apex, np.ones(len(apex))])
    assert impulse_cell_candidates(
        track, 0, len(track), burst_config().thresholds, smoothing_mode=SmoothingMode.ZERO_FILL
    )
    strict = replace(burst_config().thresholds, contact_impulse_multiple=1000000000000.0)
    assert impulse_cell_candidates(track, 0, len(track), strict, smoothing_mode=SmoothingMode.ZERO_FILL) == []
