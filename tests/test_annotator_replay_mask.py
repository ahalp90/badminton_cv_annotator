"""Replay-mask tests for isolated signals, their union, and missing inputs.

Synthetic inputs are shaped so each signal has one obvious firing region and one
obvious quiet region.
"""
import logging
import sys
import warnings

import numpy as np
import pytest

import annotator.replay_mask as replay_mask_module
from annotator.config import (
    PERSPECTIVE_SHIFT_THRESHOLD,
    SLOWMO_SPEED_FRAC,
)
from annotator.inpaint_guard import FABRICATED, code_counts, grade_track
from annotator.fps_constants import scale_for_fps
from annotator.replay_mask import (
    HOMOGRAPHY_CORNER_COLS,
    _cli_non_evidence,
    combine_mask,
    court_absence_signal,
    perspective_shift_signal,
    filter_short_exclusion_runs,
    velocity_drop_signal,
)

# Reference court corners (interleaved [corner, xy]); bbox diagonal 500.
REFERENCE_CORNERS = [100.0, 100.0, 500.0, 100.0, 100.0, 400.0, 500.0, 400.0]


def _homography_row(video_id: str, start: int, end: int, corners: list[float]) -> dict:
    row = {'video_id': video_id, 'start_frame': str(start), 'end_frame': str(end)}
    row.update({col: str(value) for col, value in zip(HOMOGRAPHY_CORNER_COLS, corners)})
    return row


def test_court_absence_fires_on_sustained_gap_not_blips():
    n_frames = 60
    court_present = np.ones(n_frames, dtype=bool)
    window = scale_for_fps(25.0).court_absent_window
    long_gap = slice(10, 10 + window + 5)   # sustained absence, fires
    short_blip = slice(45, 48)                            # 3 frames, below the window
    court_present[long_gap] = False
    court_present[short_blip] = False

    mask = court_absence_signal(court_present, n_frames, 25.0)
    assert mask[long_gap].all()
    assert not mask[short_blip].any()
    assert not mask[:10].any()


def test_perspective_fires_only_on_deviant_segment():
    shifted = [value + (200.0 if i % 2 == 0 else 0.0) for i, value in enumerate(REFERENCE_CORNERS)]
    rows = [
        _homography_row('v', 0, 100, REFERENCE_CORNERS),   # dominant view
        _homography_row('v', 100, 200, REFERENCE_CORNERS),  # dominant view
        _homography_row('v', 200, 220, shifted),            # +200 in x on every corner -> replay angle
    ]
    n_frames = 220
    mask = perspective_shift_signal(rows, n_frames)

    # 200 / 500 = 0.4 displacement, well over the threshold.
    assert (200.0 / 500.0) > PERSPECTIVE_SHIFT_THRESHOLD
    assert not mask[:200].any()
    assert mask[200:220].all()


def _speed_track(step: float, n_frames: int) -> np.ndarray:
    """A visible track whose per-frame speed is `step` (y toggles by `step`)."""
    ys = 0.4 + step * (np.arange(n_frames) % 2)
    return np.column_stack([np.full(n_frames, 0.5), ys, np.ones(n_frames)])


def test_velocity_drop_fires_on_slow_span_not_normal_play():
    n_frames = 90
    normal_step, slow_step = 0.1, 0.012
    track = _speed_track(normal_step, n_frames)
    track[40:80, 1] = 0.4 + slow_step * (np.arange(40, 80) % 2)  # slow replay span
    rally_spans = [(0, 31)]                                       # normal play defines the norm

    mask = velocity_drop_signal(track, rally_spans, n_frames, 25.0)
    # slow_step (0.012) < SLOWMO_SPEED_FRAC * normal_step (0.015); normal play stays above.
    assert slow_step < SLOWMO_SPEED_FRAC * normal_step
    assert mask[45:78].all()
    assert not mask[3:28].any()


def test_velocity_drop_ignores_genuine_rest():
    """Rest (below REST_SPEED) is the between-rallies state, not slow motion.

    Post-rally commentary starts during rest; if rest fired this signal, commentary pairing
    would hold every post-rally chunk out of pairing.
    """
    n_frames = 90
    track = _speed_track(0.1, n_frames)
    track[40:80, 1] = 0.4                                 # shuttle at rest: zero speed, visible
    rally_spans = [(0, 31)]

    mask = velocity_drop_signal(track, rally_spans, n_frames, 25.0)
    assert not mask[45:78].any()
    assert not mask[3:28].any()


def test_union_combines_signals():
    n_frames = 220
    court_present = np.ones(n_frames, dtype=bool)
    court_present[10:30] = False                         # court-absence region
    rows = [
        _homography_row('v', 0, 200, REFERENCE_CORNERS),
        _homography_row('v', 200, 220,
                        [value + (200.0 if i % 2 == 0 else 0.0)
                         for i, value in enumerate(REFERENCE_CORNERS)]),
    ]
    track = _speed_track(0.1, n_frames)
    rally_spans = [(30, 200)]

    mask = combine_mask(court_present, rows, track, rally_spans, n_frames, 25.0)
    assert mask[15:25].all()                             # court absence
    assert mask[205:215].all()                           # perspective shift


def test_missing_inputs_contribute_all_false():
    n_frames = 50
    assert not court_absence_signal(None, n_frames, 25.0).any()
    assert not perspective_shift_signal(None, n_frames).any()
    assert not velocity_drop_signal(None, [(0, 10)], n_frames, 25.0).any()
    assert not velocity_drop_signal(_speed_track(0.1, n_frames), None, n_frames, 25.0).any()
    assert not combine_mask(None, None, None, None, n_frames, 25.0).any()


def test_filter_short_exclusion_runs_preserves_whole_runs_and_is_idempotent():
    raw = np.array([False, True, True, False, True, True, True, False], dtype=bool)

    filtered = filter_short_exclusion_runs(raw, min_frames=3)

    np.testing.assert_array_equal(
        filtered, [False, False, False, False, True, True, True, False],
    )
    np.testing.assert_array_equal(
        filter_short_exclusion_runs(np.array([True, True, True], dtype=bool), 3),
        [True, True, True],
    )
    np.testing.assert_array_equal(
        filter_short_exclusion_runs(np.array([True, True], dtype=bool), 3),
        [False, False],
    )
    np.testing.assert_array_equal(
        filter_short_exclusion_runs(np.zeros(0, dtype=bool), 3), np.zeros(0, dtype=bool),
    )
    np.testing.assert_array_equal(
        filter_short_exclusion_runs(np.zeros(4, dtype=bool), 3), np.zeros(4, dtype=bool),
    )
    np.testing.assert_array_equal(
        filter_short_exclusion_runs(np.ones(4, dtype=bool), 3), [True, True, True, True],
    )
    np.testing.assert_array_equal(filter_short_exclusion_runs(filtered, 3), filtered)


@pytest.mark.parametrize(
    'mask', [np.zeros((2, 2), dtype=bool), np.zeros(3, dtype=np.uint8), [True, False]],
)
def test_filter_short_exclusion_runs_rejects_non_boolean_vectors(mask):
    with pytest.raises(ValueError, match='one-dimensional boolean'):
        filter_short_exclusion_runs(mask, min_frames=3)


def test_non_evidence_measures_steps_not_output_frames():
    n_frames = 100
    track = _speed_track(0.1, n_frames)
    track[40:80, 1] = 0.4 + 0.012 * (np.arange(40, 80) % 2)
    rally_spans = [(0, 31)]

    long_graded = np.zeros(n_frames, dtype=bool)
    long_graded[50:70] = True
    long_mask = velocity_drop_signal(
        track, rally_spans, n_frames, 25.0, non_evidence=long_graded,
    )
    assert not long_mask[60:65].any()

    short_graded = np.zeros(n_frames, dtype=bool)
    short_graded[55:57] = True
    short_mask = velocity_drop_signal(
        track, rally_spans, n_frames, 25.0, non_evidence=short_graded,
    )
    assert short_mask[53]
    assert short_mask[58]
    # The graded frames themselves still fire on neighbouring measured evidence:
    # non_evidence is a measurement rule, not an output gate over the mask.
    assert short_mask[55]
    assert short_mask[56]


def test_non_evidence_marks_a_step_when_only_its_earlier_endpoint_is_graded(monkeypatch):
    track = _speed_track(0.1, 40)
    non_evidence = np.zeros(len(track), dtype=bool)
    non_evidence[4] = True
    captured = []

    def capture_speed(values, _window):
        captured.append(values.copy())
        return np.full(len(values), np.nan)

    monkeypatch.setattr(replay_mask_module, 'rolling_nanmedian', capture_speed)
    velocity_drop_signal(track, [(0, len(track))], len(track), 25.0, non_evidence=non_evidence)

    assert non_evidence[4]
    assert not non_evidence[5]
    # Both steps touching frame 4 are unmeasured: step 4 (frame 4 as its
    # current endpoint) and step 5 (frame 4 as its earlier endpoint).
    assert np.isnan(captured[0][4])
    assert np.isnan(captured[0][5])


def test_non_evidence_removes_graded_fast_frames_from_the_baseline():
    n_frames = 80
    track = _borderline_baseline_track(n_frames)
    rally_spans = [(0, 40)]
    graded = np.zeros(n_frames, dtype=bool)
    graded[5:35] = True

    # Grading frames [5:35) NaNs every touching step, 5 through 35 inclusive:
    # the span keeps 8 finite steps (1-4 and 36-39), all 0.015. One fewer than
    # the baseline_exclude test's 9: pool exclusion keeps the boundary step 35,
    # the step-touch rule kills it.
    hand_computed_median = float(np.median([0.015] * 8))
    assert hand_computed_median == pytest.approx(0.015)
    assert 0.005 >= SLOWMO_SPEED_FRAC * hand_computed_median
    assert 0.005 < SLOWMO_SPEED_FRAC * 0.1

    contaminated = velocity_drop_signal(track, rally_spans, n_frames, 25.0)
    decontaminated = velocity_drop_signal(
        track, rally_spans, n_frames, 25.0, non_evidence=graded,
    )
    assert contaminated[50:65].all()
    assert not decontaminated[50:65].any()


def test_baseline_exclude_removes_flagged_frames_and_leaves_firing_untouched():
    n_frames = 80
    track = _borderline_baseline_track(n_frames)
    rally_spans = [(0, 40)]
    flagged = np.zeros(n_frames, dtype=bool)
    flagged[5:35] = True

    # Excluding frames [5:35) from the pool leaves frames {0-4, 35-39}: 9
    # finite steps (1-4 and 35-39), all 0.015. Step 35 survives here because
    # pool exclusion, unlike the step-touch rule, keeps boundary steps.
    hand_computed_median = float(np.median([0.015] * 9))
    assert hand_computed_median == pytest.approx(0.015)
    assert 0.005 >= SLOWMO_SPEED_FRAC * hand_computed_median
    assert 0.005 < SLOWMO_SPEED_FRAC * 0.1
    contaminated = velocity_drop_signal(track, rally_spans, n_frames, 25.0)
    decontaminated = velocity_drop_signal(
        track, rally_spans, n_frames, 25.0, baseline_exclude=flagged,
    )
    assert contaminated[50:65].all()
    assert not decontaminated[50:65].any()

    unchanged_track = _speed_track(0.1, n_frames)
    unchanged_track[40:55, 1] = 0.4 + 0.005 * (np.arange(40, 55) % 2)
    one_flagged_frame = np.zeros(n_frames, dtype=bool)
    one_flagged_frame[45] = True
    before = velocity_drop_signal(unchanged_track, [(0, 60)], n_frames, 25.0)
    after = velocity_drop_signal(
        unchanged_track, [(0, 60)], n_frames, 25.0,
        baseline_exclude=one_flagged_frame,
    )
    assert before[45]
    assert after[45]


def test_velocity_drop_prechange_regression_with_explicit_noop_masks():
    n_frames = 90
    track = np.column_stack([
        np.full(n_frames, 0.5),
        0.4 + 0.1 * (np.arange(n_frames) % 2),
        np.ones(n_frames),
    ])
    track[35:73, 1] = 0.4 + 0.012 * (np.arange(35, 73) % 2)
    track[73:79, 1] = 0.4 + 0.06 * (np.arange(73, 79) % 2)
    expected = [
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False,
        True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False,
    ]

    np.testing.assert_array_equal(
        velocity_drop_signal(track, [(0, 30)], n_frames, 25.0), expected,
    )
    false_mask = np.zeros(n_frames, dtype=bool)
    np.testing.assert_array_equal(
        velocity_drop_signal(
            track, [(0, 30)], n_frames, 25.0,
            non_evidence=false_mask, baseline_exclude=false_mask,
        ),
        expected,
    )


@pytest.mark.parametrize('case', ['all-excluded', 'all-nan'])
def test_velocity_drop_empty_baseline_pool_is_quiet(case, caplog):
    n_frames = 30
    track = _speed_track(0.1, n_frames) if case == 'all-excluded' else np.zeros((n_frames, 3))
    baseline_exclude = np.ones(n_frames, dtype=bool) if case == 'all-excluded' else None
    caplog.set_level(logging.INFO, logger='annotator.replay_mask')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        mask = velocity_drop_signal(
            track, [(0, n_frames)], n_frames, 25.0, baseline_exclude=baseline_exclude,
        )

    assert not mask.any()
    assert any(
        record.message == 'no visible rally-span frames; velocity-drop signal all-False'
        for record in caplog.records
    )
    assert not any(issubclass(warning.category, RuntimeWarning) for warning in caught)


@pytest.mark.parametrize('parameter', ['non_evidence', 'baseline_exclude'])
@pytest.mark.parametrize(
    'invalid',
    [
        np.zeros((12, 1), dtype=bool),
        np.zeros(12, dtype=np.int8),
        np.zeros(11, dtype=bool),
    ],
    ids=['wrong-ndim', 'wrong-dtype', 'wrong-length'],
)
def test_velocity_drop_validates_optional_masks(parameter, invalid):
    track = _speed_track(0.1, 12)
    with pytest.raises(ValueError):
        velocity_drop_signal(
            track, [(0, len(track))], len(track), 25.0, **{parameter: invalid},
        )


@pytest.mark.parametrize('parameter', ['non_evidence', 'baseline_exclude'])
@pytest.mark.parametrize('disabled', ['no-track', 'no-spans'])
def test_velocity_drop_disabled_signal_skips_optional_mask_validation(parameter, disabled):
    # Guard order is binding: the None/empty early return comes before optional
    # mask validation, so a disabled signal stays all-False and never raises.
    track = None if disabled == 'no-track' else _speed_track(0.1, 12)
    spans = [(0, 12)] if disabled == 'no-track' else []
    invalid = np.zeros(5, dtype=np.int8)
    mask = velocity_drop_signal(track, spans, 12, 25.0, **{parameter: invalid})
    assert not mask.any()


def test_combine_mask_threads_non_evidence_to_velocity():
    n_frames = 80
    track = _speed_track(0.1, n_frames)
    track[35:65, 1] = 0.4 + 0.015 * (np.arange(35, 65) % 2)
    non_evidence = np.zeros(n_frames, dtype=bool)
    non_evidence[45:60] = True

    combined = combine_mask(
        None, None, track, [(0, 30)], n_frames, 25.0, non_evidence=non_evidence,
    )
    expected = velocity_drop_signal(
        track, [(0, 30)], n_frames, 25.0,
        non_evidence=non_evidence, baseline_exclude=np.zeros(n_frames, dtype=bool),
    )
    np.testing.assert_array_equal(combined, expected)


def test_combine_mask_excludes_court_producer_from_baseline():
    n_frames = 80
    track = _borderline_baseline_track(n_frames)
    court_present = np.ones(n_frames, dtype=bool)
    court_present[5:35] = False
    court = court_absence_signal(court_present, n_frames, 25.0)
    result = combine_mask(court_present, None, track, [(0, 40)], n_frames, 25.0)
    without_court = combine_mask(np.ones(n_frames, dtype=bool), None, track, [(0, 40)], n_frames, 25.0)
    expected = court | velocity_drop_signal(
        track, [(0, 40)], n_frames, 25.0, baseline_exclude=court,
    )

    assert court[5:35].all()
    assert without_court[50]
    assert not result[50]
    np.testing.assert_array_equal(result, expected)


def test_combine_mask_excludes_perspective_producer_from_baseline():
    n_frames = 80
    track = _borderline_baseline_track(n_frames)
    shifted = [value + (200.0 if index % 2 == 0 else 0.0) for index, value in enumerate(REFERENCE_CORNERS)]
    rows = [
        _homography_row('v', 0, 5, REFERENCE_CORNERS),
        _homography_row('v', 5, 35, shifted),
        _homography_row('v', 35, 80, REFERENCE_CORNERS),
    ]
    perspective = perspective_shift_signal(rows, n_frames)
    result = combine_mask(None, rows, track, [(0, 40)], n_frames, 25.0)
    without_perspective = combine_mask(None, None, track, [(0, 40)], n_frames, 25.0)
    expected = perspective | velocity_drop_signal(
        track, [(0, 40)], n_frames, 25.0, baseline_exclude=perspective,
    )

    assert perspective[5:35].all()
    assert without_perspective[50]
    assert not result[50]
    np.testing.assert_array_equal(result, expected)


def _borderline_baseline_track(n_frames: int) -> np.ndarray:
    """Track with a fast baseline contaminant and a borderline candidate span."""
    track = _speed_track(0.015, n_frames)
    fast_frames = np.arange(5, 35)
    track[5:35, 1] = 0.4 + 0.1 * (fast_frames % 2)
    borderline_frames = np.arange(45, 70)
    track[45:70, 1] = 0.4 + 0.005 * (borderline_frames % 2)
    return track


def _cli_fabricated_track() -> np.ndarray:
    """Track with the recurrence pattern used by the guard's fabricated-fill tests."""
    length = 14000
    track = np.column_stack((
        np.arange(length, dtype=float),
        np.arange(length, dtype=float) * 0.37 + 1.0,
    ))
    varying = np.column_stack((
        np.arange(16, dtype=float) + 100.0,
        np.arange(16, dtype=float) * 2.0 + 200.0,
    ))
    varying_starts = [20 + episode * 100 for episode in range(50)]
    varying_starts += [7200 + (episode - 50) * 100 for episode in range(50, 100)]
    for start in varying_starts:
        track[start:start + 16] = varying
    flat_starts = tuple([5500 + episode * 50 for episode in range(25)] +
                        [12500 + episode * 50 for episode in range(25)])
    for start in flat_starts:
        track[start:start + 16] = (900.0, 901.0)
    for start in (6800, 6840, 6880):
        track[start:start + 16] = (1200.0, 1300.0)
    return np.column_stack((track, np.ones(length)))


def test_cli_non_evidence_grades_rejected_frames_and_logs_counts(caplog):
    track = _cli_fabricated_track()
    codes, _info = grade_track(track)
    caplog.set_level(logging.INFO, logger='annotator.replay_mask')

    result = _cli_non_evidence(track)

    expected = np.isin(codes, (1, 2, 3))
    np.testing.assert_array_equal(result, expected)
    assert np.all(codes[20:36] == FABRICATED)
    assert len(caplog.records) == 1
    assert str(code_counts(codes)) in caplog.records[0].message


def test_cli_non_evidence_returns_none_for_missing_track():
    assert _cli_non_evidence(None) is None


def test_cli_non_evidence_uses_configured_rejected_grades(monkeypatch):
    track = _cli_fabricated_track()
    codes, _info = grade_track(track)

    class ConfigDouble:
        rejected_grades = frozenset({FABRICATED})

    monkeypatch.setattr(replay_mask_module, 'BaseAnnotatorConfig', ConfigDouble)
    result = _cli_non_evidence(track)

    np.testing.assert_array_equal(result, codes == FABRICATED)


def test_cli_non_evidence_propagates_grade_track_errors(monkeypatch):
    track = np.ones((4, 3), dtype=float)

    def fail(_track):
        raise RuntimeError('grading failed')

    monkeypatch.setattr(replay_mask_module, 'grade_track', fail)
    with pytest.raises(RuntimeError, match='grading failed'):
        _cli_non_evidence(track)


def test_replay_mask_writer_trusts_detector_output_and_supports_maskless_mode(
    monkeypatch, tmp_path, caplog,
):
    court_path = tmp_path / 'court.npy'
    out_dir = tmp_path / 'masks'
    np.save(court_path, np.zeros(40, dtype=bool))
    raw = np.zeros(40, dtype=bool)
    raw[:14] = True
    raw[20:] = True
    monkeypatch.setattr(replay_mask_module, 'combine_mask', lambda *args, **kwargs: raw.copy())
    caplog.set_level(logging.INFO, logger='annotator.replay_mask')

    monkeypatch.setattr(sys, 'argv', [
        'replay', '--video-id', 'v', '--court-mask', str(court_path), '--out-dir', str(out_dir),
        '--rally-spans', str(tmp_path / 'spans.csv'), '--fps', '30',
    ])
    replay_mask_module.main()
    np.testing.assert_array_equal(
        np.load(out_dir / 'v_replay.npy'), raw,
    )

    monkeypatch.setattr(
        replay_mask_module, 'combine_mask',
        lambda *args, **kwargs: pytest.fail('maskless mode must skip detector computation'),
    )
    monkeypatch.setattr(sys, 'argv', [
        'replay', '--video-id', 'v', '--court-mask', str(court_path), '--out-dir', str(out_dir),
        '--rally-spans', str(tmp_path / 'spans.csv'), '--fps', '30', '--no-replay-mask',
    ])
    replay_mask_module.main()
    assert not np.load(out_dir / 'v_replay.npy').any()
    assert 'config: no_replay_mask=True' in caplog.text
