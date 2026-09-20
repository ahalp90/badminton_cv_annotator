"""Synthetic checks of the replay helpers: the chart decomposition, the step masks and the product search."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axis_replay import (
    courts_from,
    ideal_parameters,
    nearest_over_product,
    step_masks,
    sweep,
)
from paint_profiles import (
    FLANK_WINDOWS_WORKING_PX,
    features,
    profile_offsets,
)

from shared import (
    ALL_CASE_IDS,
    ALL_CASES,
    CASE_IDS,
    CASES,
    LABELS,
    PACK_OF,
    REGRESSION_CASES,
    corner_errors,
)


def mocked_filter_replay(monkeypatch, tmp_path: Path, case_id: str = 'mock_case'):
    import filter_replay
    import vp_pruning

    source = {
        'id': case_id,
        'segments_px': [[0., 1., 8., 1.], [0., 8., 8., 8.]],
        'bbox_px': [],
        'dimensions': {'width': 8, 'height': 8},
    }
    settings = vp_pruning.Settings()
    saved = {
        'case_id': case_id,
        'working_size': [8, 8],
        'settings': {},
        'estimator': {'retained_candidate_ids': [0], 'points_working': [[1., 0., 1.]],
                      'retained_support_masks': [], 'direction_lines': []},
    }
    masks = {arm: np.array([True, False]) for arm in filter_replay.ARMS}
    masks.update({'baseline': np.array([True, True]), 'person': np.array([True, False]),
                  'paint': np.array([True, False]), 'paint_person': np.array([True, False])})
    gate_calls = []
    mask_calls = []

    monkeypatch.setattr(filter_replay, 'CASE_IDS', (case_id,))
    monkeypatch.setattr(filter_replay, 'LABELS', {case_id: 'Mock'})
    monkeypatch.setattr(filter_replay, 'PACK_OF', {case_id: 'mock'})
    monkeypatch.setattr(filter_replay, 'PACKS', {'mock': tmp_path / 'pack.json.gz'})
    monkeypatch.setattr(filter_replay, 'read', lambda _path: {'cases': [source]})
    monkeypatch.setattr(filter_replay, 'frame_path', lambda _source: tmp_path / 'frame.png')
    monkeypatch.setattr(filter_replay.cv2, 'imread', lambda _path: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(filter_replay, 'prepare', lambda _source: (np.asarray(source['segments_px']), None, (8, 8)))
    monkeypatch.setattr(filter_replay, 'load_estimator', lambda _case_id: saved)

    def fake_gate(segments, size, record):
        gate_calls.append((segments, size, record))
        return settings

    def fake_masks(case_source, frame, scale):
        mask_calls.append((case_source, frame, scale))
        return masks

    monkeypatch.setattr(filter_replay, 'gate_baseline', fake_gate)
    monkeypatch.setattr(filter_replay, 'fragment_masks', fake_masks)
    return filter_replay, case_id, saved, gate_calls, mask_calls


def install_normal_route_stubs(monkeypatch, filter_replay, saved):
    def fake_stage_one(case_id, arm, source, keep, settings, control, control_directions, baseline_ids):
        row = dict.fromkeys(filter_replay.STAGE1_COLUMNS)
        row.update({'case_id': case_id, 'label': 'Mock', 'arm': arm, 'fragments': int(keep.sum()), 'dropped': int((~keep).sum()),
                    'marking_fragments': 0, 'marking_dropped': 0, 'merged_lines': 0, 'directions': 0,
                    'selection_identical_to_baseline': True})
        best = {'pair_id': 0, 'groups': [0, 1], 'max_corner_working_px': 0.}
        return row, saved['estimator'], np.zeros((2, 3)), best

    monkeypatch.setattr(filter_replay, 'stage_one', fake_stage_one)
    monkeypatch.setattr(filter_replay, 'axis_stage', lambda *args: [])
    monkeypatch.setattr(filter_replay, 'control_corners', lambda _case_id: (np.zeros((4, 2)), {}))
    monkeypatch.setattr(filter_replay, 'control_vanishing_points', lambda _control, _size: np.zeros((2, 3)))


def test_observation_inputs_only_does_not_load_control_or_e3(monkeypatch, tmp_path):
    filter_replay, case_id, _, gate_calls, mask_calls = mocked_filter_replay(monkeypatch, tmp_path)

    def records_are_not_allowed(*args):
        raise AssertionError(f'unexpected diagnostic/control load: {args}')

    monkeypatch.setattr(filter_replay, 'control_corners', records_are_not_allowed)
    monkeypatch.setattr(filter_replay, 'load_direction_record', records_are_not_allowed)
    filter_replay.write_observation_inputs_only([case_id], tmp_path / 'inputs')

    assert len(gate_calls) == 1
    assert len(mask_calls) == 1
    assert (tmp_path / 'inputs' / 'paint_observations' / 'cases' / f'{case_id}.json.gz').exists()
    assert (tmp_path / 'inputs' / 'paint_observations' / 'estimators' / f'{case_id}.json.gz').exists()


def test_observation_inputs_only_match_normal_route_bytes(monkeypatch, tmp_path):
    filter_replay, case_id, saved, _, _ = mocked_filter_replay(monkeypatch, tmp_path)
    install_normal_route_stubs(monkeypatch, filter_replay, saved)
    normal_inputs = tmp_path / 'normal_inputs'
    monkeypatch.setattr(sys, 'argv', ['filter_replay.py', '--output', str(tmp_path / 'run'), '--cases', case_id,
                                      '--inputs-dir', str(normal_inputs), '--axis-arms', 'baseline'])
    filter_replay.main()

    def records_are_not_allowed(*args):
        raise AssertionError(f'unexpected diagnostic/control load: {args}')

    monkeypatch.setattr(filter_replay, 'control_corners', records_are_not_allowed)
    monkeypatch.setattr(filter_replay, 'load_direction_record', records_are_not_allowed)
    observation_inputs = tmp_path / 'observation_inputs'
    monkeypatch.setattr(sys, 'argv', ['filter_replay.py', '--write-observation-inputs-only', '--cases', case_id,
                                      '--inputs-dir', str(observation_inputs)])
    filter_replay.main()

    normal_files = {path.relative_to(normal_inputs / 'paint_observations'): path.read_bytes()
                    for path in (normal_inputs / 'paint_observations').rglob('*') if path.is_file()}
    observation_files = {path.relative_to(observation_inputs / 'paint_observations'): path.read_bytes()
                         for path in (observation_inputs / 'paint_observations').rglob('*') if path.is_file()}
    assert observation_files == normal_files


def test_parse_defaults_remain_full_replay_defaults():
    from filter_replay import DEFAULT_AXIS_ARMS, parse_args

    args = parse_args(['--output', 'run'])
    assert args.cases == list(CASE_IDS)
    assert args.axis_arms == DEFAULT_AXIS_ARMS
    assert not args.write_observation_inputs_only
    with pytest.raises(SystemExit):
        parse_args(['--write-observation-inputs-only', '--cases', CASE_IDS[0], '--axis-arms', 'baseline'])


def random_basis(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    basis = generator.normal(size=(3, 3))
    basis[2] = [0.01, 0.02, 1.]
    return basis


def test_ideal_parameters_round_trip():
    basis = random_basis(0)
    scales_offsets = np.array([[2.5, -0.3], [1.7, 0.4]])
    homography = basis @ np.array([[2.5, 0., -0.3], [0., 1.7, 0.4], [0., 0., 1.]]) * 3.
    recovered, leak = ideal_parameters(basis, homography)
    np.testing.assert_allclose(recovered, scales_offsets, atol=1e-12)
    assert leak < 1e-12
    corners = courts_from(basis, recovered[:1], recovered[1:])
    reference = courts_from(basis, scales_offsets[:1], scales_offsets[1:])
    np.testing.assert_allclose(corners, reference, atol=1e-9)


def test_step_masks_follow_the_matching_order():
    matches = SimpleNamespace(parameters=np.zeros((6, 2)), supported=np.array([3, 2, 5, 3, 4, 3]),
                              player_compatible=np.array([True, True, False, True, True, True]),
                              distinct=np.array([2, 4, 0, 5]))
    masks = step_masks(matches, cap=3)
    assert masks['enumerated'].tolist() == [True] * 6
    assert masks['supported'].tolist() == [True, False, True, True, True, True]
    assert masks['supported_and_players'].tolist() == [True, False, False, True, True, True]
    assert masks['distinct'].tolist() == [True, False, True, False, True, True]
    assert masks['kept'].tolist() == [True, False, True, False, True, False]


def test_nearest_over_product_equals_brute_force():
    basis = random_basis(1)
    generator = np.random.default_rng(2)
    horizontal = generator.normal(size=(7, 2)) + [3., 0.]
    vertical = generator.normal(size=(5, 2)) + [3., 0.]
    control = courts_from(basis, horizontal[3:4], vertical[1:2])[0] + 0.5
    best = nearest_over_product(basis, horizontal, vertical, control)
    brute = min(float(corner_errors(courts_from(basis, horizontal[i:i + 1], vertical[j:j + 1]), control)[0])
                for i in range(7) for j in range(5))
    assert abs(best - brute) < 1e-9
    errors = sweep(basis, 0, horizontal, vertical[1], control)
    assert errors.shape == (7,) and abs(errors[3] - float(corner_errors(control[None] - 0.5, control)[0])) < 1e-9


def synthetic_profile(brightness_across: np.ndarray, saturation_value: float) -> np.ndarray:
    samples = 12
    brightness = np.tile(brightness_across, (1, samples, 1)).astype(float)
    saturation = np.full_like(brightness, saturation_value)
    return np.stack((brightness, saturation))


def test_features_pass_a_ridge_and_fail_a_step():
    offsets = profile_offsets()
    flank_near = FLANK_WINDOWS_WORKING_PX[0][0]
    ridge = np.where(np.abs(offsets) <= 1.5, 230., 150.)
    step = np.where(offsets >= 0., 230., 150.)
    plateau = np.where(offsets >= -1., 230., 150.)  # bright on one side out past every flank window
    measured = features(synthetic_profile(np.stack((ridge, step, plateau))[:, None, :], 30.))
    contrast = measured[:, 0]
    assert contrast[0] > 70, contrast
    assert contrast[1] <= 0, contrast
    assert contrast[2] <= 0, contrast
    assert measured[0, 1] == 30.
    assert flank_near > 1.5


def test_control_vanishing_points_and_angles_recover_a_known_homography():
    import cv2
    from filter_replay import (
        angles_to_control,
        control_vanishing_points,
        fragment_masks,
    )

    from experiments.annotator.independent_court import detector
    size = (960, 540)
    homography = np.array([[80., 10., 200.], [5., 60., 100.], [0.001, 0.02, 1.]])
    corners, _ = detector.project(homography[None], detector.CORNER_COURT_M)
    control_directions = control_vanishing_points(corners[0], size)
    # The homography's first two columns are the vanishing points; feed them back as selected directions.
    points = homography[:, :2].T
    angles = angles_to_control(points, control_directions, size)
    assert angles.shape == (2,) and angles.max() < 1e-4, angles
    # A box that covers the first fragment's midpoint and nothing else.
    frame = np.full((40, 60, 3), 120, dtype=np.uint8)
    cv2.line(frame, (10, 20), (50, 20), (255, 255, 255), 2)
    source = {'segments_px': [[10., 20., 50., 20.], [10., 5., 50., 5.]], 'bbox_px': [[20., 15., 40., 25.]]}
    masks = fragment_masks(source, frame, np.array([1., 1.]))
    assert masks['person'].tolist() == [False, True]
    assert masks['paint'].tolist() == [True, False]
    assert masks['baseline'].all()


def test_unused_cases_are_explicit_without_changing_regression_defaults():
    assert CASES == REGRESSION_CASES
    assert CASE_IDS == tuple(case_id for case_id, _, _ in REGRESSION_CASES)
    assert len(ALL_CASES) == 27
    assert len(ALL_CASE_IDS) == 27
    assert 'gxBQ_window_00_frame_689' in ALL_CASE_IDS
    assert PACK_OF['gxBQ_window_00_frame_689'] == 'gx'
    assert LABELS['gxBQ_window_00_frame_689'] == 'gxBQ_window_00_frame_689'
