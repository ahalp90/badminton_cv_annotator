"""Checks for the E4 saved-estimator adapter and the identity-reuse path."""

from pathlib import Path

import numpy as np
from common import BASELINE_STAGES, STAGES, read, write
from run_matcher import adapter, complete, run_case, stage_paths


def saved_record(points: list[list[float]]) -> dict:
    return {'case_id': 'synthetic', 'working_size': [960, 540],
            'settings': {'pencil_selection': 'coverage', 'angle_deg': 1.5},
            'estimator': {'points_working': points, 'direction_lines': [[1., 0., 0.]], 'normalised_to_working': np.eye(3).tolist(),
                          'candidate_ids': [0, 1], 'compatible_raw_ids': [[0]], 'retained_candidate_ids': [0, 1]}}


def arm(points: list[list[float]], name: str = 'R') -> dict:
    return {'arm': name, 'anchor_rule': 'original_foot', 'representative_rule': 'precision', 'matcher_eligible': True,
            'court_result': None, 'points_working': points, 'representative_candidate_ids': [5, 6],
            'leader_candidate_ids': [0, 1], 'support_masks': [[True], [True]], 'support_counts': [1, 1],
            'allocation_candidate_status': ['retained', 'retained']}


def test_adapter_carries_arm_points_and_drops_stale_estimator_fields() -> None:
    saved = saved_record([[1., 0., 0.], [0., 1., 0.]])
    arm_points = [[2., 1., 0.], [0., 3., 1.]]
    result = adapter(saved, arm(arm_points), 'run_x', 'saved-md5', 'e2-md5')
    assert result['working_size'] == [960, 540] and result['settings']['pencil_selection'] == 'coverage'
    assert result['estimator']['points_working'] == arm_points
    assert result['estimator']['retained_candidate_ids'] == [5, 6]
    assert 'compatible_raw_ids' not in result['estimator']
    assert result['estimator']['direction_lines'] == saved['estimator']['direction_lines']
    assert result['estimator']['source_saved_estimator_md5'] == 'saved-md5'


def test_identical_directions_reuse_the_saved_baseline(tmp_path: Path) -> None:
    points = [[1., 0., 0.], [0., 1., 0.]]
    saved = saved_record(points)
    for stage, directory in BASELINE_STAGES.items():
        write(tmp_path / directory / 'synthetic.json.gz', {'case_id': 'synthetic', 'stage_marker': stage, 'entries': []})
    output = tmp_path / 'direction_agreement/runs/run_x'
    summary = run_case('synthetic', {}, saved, {'arms': {'R': arm(points)}}, 'R', None, tmp_path, output, 'run_x',
                       {'x.py': 'md5'}, 'saved-md5', 'e2-md5')
    assert summary['status'] == 'identity_reused'
    for stage in STAGES:
        record = read(output / 'e4/R' / stage / 'synthetic.json.gz')
        assert record['stage_marker'] == stage and record['arm'] == 'R' and 'identity_reused_from' in record
    proof = read(output / 'e4/R/synthetic_identity.json.gz')
    assert proof['identical_to_baseline_directions'] and proof['points_working'] == points
    paths = stage_paths(output, 'R', 'synthetic')
    assert complete(paths, {'x.py': 'md5'}, 'saved-md5', 'e2-md5')
    assert not complete(paths, {'x.py': 'md5'}, 'saved-md5', 'e2-changed')
    assert not complete(paths, {'x.py': 'other'}, 'saved-md5', 'e2-md5')


def test_ineligible_arm_writes_empty_court_records(tmp_path: Path) -> None:
    saved = saved_record([[1., 0., 0.]])
    ineligible = {**arm([[1., 0., 0.]]), 'matcher_eligible': False,
                  'court_result': {'status': 'empty', 'reason': 'fewer than 2 directions survive allocation'}}
    output = tmp_path / 'runs/run_x'
    summary = run_case('synthetic', {}, saved, {'arms': {'R': ineligible}}, 'R', None, tmp_path, output, 'run_x', {},
                       'saved-md5', 'e2-md5')
    assert summary['status'] == 'empty'
    for stage in STAGES:
        record = read(output / 'e4/R' / stage / 'synthetic.json.gz')
        assert record['entries'] == [] and record['court_result']['status'] == 'empty'
