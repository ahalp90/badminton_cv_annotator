"""Annotator config plumbing and opt-in span strategy tests."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from annotator.config import BaseAnnotatorConfig, scale_config_to_fps
from annotator.rallies.find_rallies import build_rest_mask
from annotator.types import SmoothingMode
from tests.annotator_config import trajectory_config


def test_base30_overrides_scale_all_scaling_kinds() -> None:
    base = BaseAnnotatorConfig(overrides_base30={
        'rest_window': 10.0,
        'rest_speed': 0.003,
        'contact_impulse_multiple': 5.5,
    })
    video_config = scale_config_to_fps(base, 25.0)
    assert video_config.thresholds.rest_window == 8
    assert video_config.thresholds.rest_speed == pytest.approx(0.0036)
    assert video_config.thresholds.contact_impulse_multiple == 5.5


def test_shipped_tracking_strategies_scale_at_video_fps() -> None:
    video_config25 = scale_config_to_fps(BaseAnnotatorConfig(), 25.0)
    video_config30 = scale_config_to_fps(BaseAnnotatorConfig(), 30.0)
    assert video_config25.smoothing_mode is SmoothingMode.IGNORE_INVISIBLE
    assert video_config25.max_high_shot_gap_frames == 63
    assert video_config30.max_high_shot_gap_frames == 75
    assert video_config30.reentry_guard_buffer == 0.05


def test_unknown_base30_override_fails_loudly() -> None:
    with pytest.raises(ValueError, match='unknown base-30'):
        scale_config_to_fps(BaseAnnotatorConfig(overrides_base30={'not_a_row': 1.0}), 30.0)


def test_removed_direction_change_override_fails_loudly() -> None:
    with pytest.raises(ValueError, match='unknown base-30'):
        scale_config_to_fps(BaseAnnotatorConfig(overrides_base30={'min_dir_change_deg': 30.0}), 30.0)


def test_rejected_grades_are_copied_to_scaled_config() -> None:
    base = BaseAnnotatorConfig(rejected_grades=frozenset({1, 3}))
    assert scale_config_to_fps(base, 30.0).rejected_grades == frozenset({1, 3})


@pytest.mark.parametrize(('last_before', 'first_after', 'last_after', 'held_open'), [
    (0.05, 0.05, 0.15, True),
    (0.20, 0.05, 0.15, False),
    (0.05, 0.20, 0.30, False),
    (0.05, 0.05, 0.01, False),
])
def test_high_shot_gap_requires_top_edge_departure_and_downward_return(
    last_before, first_after, last_after, held_open,
) -> None:
    y = np.concatenate([np.linspace(last_before + 0.45, last_before, 6), np.zeros(12),
                        np.linspace(first_after, last_after, 3)])
    track = np.column_stack([np.full(len(y), 0.5), y, np.ones(len(y))])
    track[6:18, 2] = 0
    speed = np.full(len(track), np.nan)
    thresholds = replace(trajectory_config().thresholds, rest_window=3, rest_speed=1.0)
    at_rest = build_rest_mask(speed, track, thresholds, max_high_shot_gap_frames=75, reentry_guard_buffer=0.1)
    assert bool(at_rest[10]) is not held_open


def test_high_shot_gap_becomes_rest_after_the_allowed_duration() -> None:
    y = np.concatenate([np.linspace(0.5, 0.05, 6), np.zeros(12)])
    track = np.column_stack([np.full(len(y), 0.5), y, np.ones(len(y))])
    track[6:, 2] = 0
    at_rest = build_rest_mask(
        np.full(len(track), np.nan), track, trajectory_config().thresholds,
        max_high_shot_gap_frames=5, reentry_guard_buffer=0.1,
    )
    assert not at_rest[6:11].any()
    assert at_rest[11:].all()


def test_gap_fps_rows_match_frozen_25fps_and_base30_values() -> None:
    values25 = scale_config_to_fps(BaseAnnotatorConfig(), 25.0).thresholds
    values30 = scale_config_to_fps(BaseAnnotatorConfig(), 30.0).thresholds
    fields = (
        'max_unexplained_shuttle_gap_frames', 'high_shot_oob_lookback_frames',
        'high_shot_oob_min_visible_frames', 'high_shot_oob_extrap_frames',
        'reentry_lookahead_frames', 'reentry_min_visible_frames',
    )
    assert tuple(getattr(values25, field) for field in fields) == (10, 5, 2, 10, 5, 2)
    assert tuple(getattr(values30, field) for field in fields) == (12, 6, 2, 12, 6, 2)


def test_impulse_contact_candidates_consumes_threshold_multiple() -> None:
    from annotator.contacts.heuristics import impulse_contact_candidates

    apex = np.array([0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float) * 0.05
    track = np.column_stack([np.full(len(apex), 0.5), apex, np.ones(len(apex))])
    assert impulse_contact_candidates(
        track, 0, len(track), trajectory_config().thresholds, smoothing_mode=SmoothingMode.INCLUDE_INVISIBLE
    )
    strict = replace(trajectory_config().thresholds, contact_impulse_multiple=1000000000000.0)
    assert impulse_contact_candidates(track, 0, len(track), strict, smoothing_mode=SmoothingMode.INCLUDE_INVISIBLE) == []
