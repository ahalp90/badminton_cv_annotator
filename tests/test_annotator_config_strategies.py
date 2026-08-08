"""Annotator config plumbing and opt-in span strategy tests."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from annotator.config import BaseAnnotatorConfig, SHIPPED_THRESHOLDS
from annotator.fps_constants import scale_for_fps
from annotator.rally.spans import _find_rally_spans_quiet_start, _gap_state_rest_mask
from annotator.rally_segmentation import segment_video
import annotator.run_video as run_video_module
from annotator.resolve import resolve
from annotator.run_video import build_serve_options, run_video
from annotator.types import ReentryGuardVariant, SmoothingMode, SpanOpen


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


def test_no_base30_overrides_is_bit_identical() -> None:
    assert resolve(BaseAnnotatorConfig(), 25.0) == resolve(
        BaseAnnotatorConfig(overrides_base30=None), 25.0,
    )


def test_shipped_tracking_strategies_resolve_at_video_fps() -> None:
    resolved25 = resolve(BaseAnnotatorConfig(), 25.0)
    resolved30 = resolve(BaseAnnotatorConfig(), 30.0)
    assert resolved25.smoothing_mode is SmoothingMode.IGNORE_INVISIBLE
    assert resolved25.gap_state_demotion_bound == 63
    assert resolved30.gap_state_demotion_bound == 75
    assert resolved30.reentry_guard_variant is ReentryGuardVariant.TWO_SIDED
    assert resolved30.reentry_guard_buffer == 0.05


def test_low_level_segment_strategies_remain_opt_in() -> None:
    parameters = inspect.signature(segment_video).parameters
    assert parameters['smoothing_mode'].default is SmoothingMode.ZERO_FILL
    assert parameters['gap_state_demotion_bound'].default is None
    assert parameters['reentry_guard_variant'].default is None
    assert parameters['reentry_guard_buffer'].default is None


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


def test_span_open_default_and_close_guard_are_aware_of_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve(BaseAnnotatorConfig(), 30.0).span_open is SpanOpen.BACK_FILL
    config = type('Config', (), {'close': object(), 'threshold_bh': 0.1, 'mode': None,
                                  'stillness_threshold_bh': None})()
    with pytest.raises(ValueError, match='unsupported with BACK_FILL'):
        build_serve_options(config, None, scale_for_fps(30.0), (1.0, 1.0))
    monkeypatch.setattr(run_video_module.rally_segmentation, 'build_serve_setup_inputs', lambda *_args: None)
    options = build_serve_options(config, None, scale_for_fps(30.0), (1.0, 1.0), None)
    assert options.close is config.close


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


def test_quiet_start_and_serve_start_fail_in_run_video() -> None:
    with pytest.raises(ValueError, match='quiet_start_window'):
        run_video(
            None, None, None, None, None, fps=30.0,
            base=BaseAnnotatorConfig(quiet_start_window=10.0, span_open=None), landing_options=None,
            net_band=(0.0, 1.0), resolution=(1.0, 1.0), video_id=1,
            court_info={}, homo_df=None, gate_court_info={}, gate_resolution_table=None,
            serve_start=object(),
        )


@pytest.mark.parametrize('span_open', (SpanOpen.REGION_START, SpanOpen.BACK_FILL))
def test_quiet_start_and_span_open_fail_in_run_video(span_open: SpanOpen) -> None:
    with pytest.raises(ValueError, match='quiet_start_window cannot be combined with span_open'):
        run_video(
            None, None, None, None, None, fps=30.0,
            base=BaseAnnotatorConfig(quiet_start_window=10.0, span_open=span_open),
            court_optional=True, stop_after_segmentation=True,
        )


def test_gap_state_with_and_without_reentry_guard() -> None:
    pre_y = np.linspace(0.5, 0.05, 6)
    y = np.concatenate([pre_y, np.zeros(12), [0.5, 0.6, 0.7]])
    track = np.column_stack([np.full(len(y), 0.5), y, np.ones(len(y))])
    track[6:18, 2] = 0
    speed = np.full(len(track), np.nan)
    thresholds = SHIPPED_THRESHOLDS._replace(rest_window=3, rest_speed=1.0)
    constants = scale_for_fps(25.0)
    unguarded = _gap_state_rest_mask(speed, track, thresholds, constants, 75, None, None)
    guarded = _gap_state_rest_mask(
        speed, track, thresholds, constants, 75, ReentryGuardVariant.REENTRY_ONLY, 0.1,
    )
    assert not unguarded[10]
    assert guarded[10]


def test_quiet_start_selects_later_quiet_preceded_burst() -> None:
    speed = np.zeros(30)
    speed[3:6] = 1.0
    speed[15:18] = 1.0
    at_rest = np.zeros(30, dtype=bool)
    at_rest[10:15] = True
    thresholds = SHIPPED_THRESHOLDS._replace(
        start_speed=0.5, start_min_frames=3, end_rest_frames=100,
    )
    assert _find_rally_spans_quiet_start(speed, at_rest, thresholds, 5) == [(15, 30)]


def test_gap_fps_rows_match_frozen_25fps_and_base30_values() -> None:
    values25 = scale_for_fps(25.0)
    values30 = scale_for_fps(30.0)
    fields = (
        'blip_max_frames', 'high_shot_oob_lookback_frames',
        'high_shot_oob_min_visible_frames', 'high_shot_oob_extrap_frames',
        'reentry_lookahead_frames', 'reentry_min_visible_frames',
    )
    assert tuple(getattr(values25, field) for field in fields) == (10, 5, 2, 10, 5, 2)
    assert tuple(getattr(values30, field) for field in fields) == (12, 6, 2, 12, 6, 2)


def test_off_path_keeps_legacy_rest_mask_call_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    # The frozen sweep rebinds _rest_mask to a (speed, track) replacement; until
    # Until the legacy rest-mask path is retired, the OFF path must accept its call shapes.
    import annotator.rally_segmentation as seg
    import annotator.rally.spans as rally_spans

    def two_arg_rest_mask(speed: np.ndarray, track: np.ndarray, thresholds=None) -> np.ndarray:
        return np.zeros(len(speed), dtype=bool)

    monkeypatch.setattr(rally_spans, '_rest_mask', two_arg_rest_mask)
    track = np.column_stack([np.zeros(20), np.zeros(20), np.ones(20)])
    assert seg.find_rally_spans(track) == []
    assert seg.find_rally_spans(track, SHIPPED_THRESHOLDS) == []


def test_impulse_cell_candidates_consumes_threshold_multiple() -> None:
    from annotator.rally_segmentation import impulse_cell_candidates

    apex = np.array([0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float) * 0.05
    track = np.column_stack([np.full(len(apex), 0.5), apex, np.ones(len(apex))])
    assert impulse_cell_candidates(track, 0, len(track), SHIPPED_THRESHOLDS)
    strict = SHIPPED_THRESHOLDS._replace(contact_impulse_multiple=1e12)
    assert impulse_cell_candidates(track, 0, len(track), strict) == []
