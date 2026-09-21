"""Synthetic checks of the replay helpers: the chart decomposition, the step masks and the product search."""

from __future__ import annotations

import hashlib
import json
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

from experiments.annotator.independent_court.case_provenance import (
    CaseProvenance,
    ImageKind,
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
    frame_path,
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
    monkeypatch.setattr(filter_replay, 'paint_masks', fake_masks)
    monkeypatch.setattr(filter_replay, 'case_provenance', lambda _case_id: object())
    monkeypatch.setattr(filter_replay, 'require_same_image_boxes', lambda record: record)
    return filter_replay, case_id, saved, gate_calls, mask_calls, source


def write_exact_detection_packet(tmp_path: Path, repair_inputs, gx_md5: str) -> Path:
    """Write a small packet with the shared detector contract and all six case records."""
    packet = {
        'schema': repair_inputs.DETECTION_SCHEMA,
        'model': {
            'basename': repair_inputs.RTMDET_M_BASENAME,
            'score_cutoff': repair_inputs.SCORE_CUTOFF,
            'score_rule': 'strict_gt',
        },
        'coordinate_order': 'xyxy',
        'cases': {},
    }
    for case_id in repair_inputs.DETECTION_CASE_IDS:
        frame_index, image, width, height = repair_inputs.DETECTION_METADATA[case_id]
        packet['cases'][case_id] = {
            'frame_index': frame_index,
            'image': image,
            'image_md5': gx_md5 if case_id == repair_inputs.GX5_CASE_ID else '0' * 32,
            'dimensions': {'width': width, 'height': height},
            'bboxes': [[1., 1., 3., 3.]],
            'scores': [0.3],
        }
    path = tmp_path / 'detections.json'
    path.write_text(json.dumps(packet))
    return path


def write_repair_detection_packet(tmp_path: Path, repair_inputs) -> Path:
    source = repair_inputs.load_source(repair_inputs.GX5_CASE_ID)
    return write_exact_detection_packet(tmp_path, repair_inputs,
                                        repair_inputs.file_md5(repair_inputs.frame_path(source)))


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
    filter_replay, case_id, _, gate_calls, mask_calls, _ = mocked_filter_replay(monkeypatch, tmp_path)

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
    filter_replay, case_id, saved, _, _, _ = mocked_filter_replay(monkeypatch, tmp_path)
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


def test_observation_inputs_only_does_not_need_boxes_or_provenance(monkeypatch, tmp_path):
    filter_replay, case_id, _, _, _, source = mocked_filter_replay(monkeypatch, tmp_path)
    source.pop('bbox_px')

    def forbidden(*_args):
        raise AssertionError('paint-only input generation resolved person provenance')

    monkeypatch.setattr(filter_replay, 'case_provenance', forbidden)
    monkeypatch.setattr(filter_replay, 'person_mask', forbidden)
    filter_replay.write_observation_inputs_only([case_id], tmp_path / 'inputs')
    assert (tmp_path / 'inputs' / 'paint_observations' / 'cases' / f'{case_id}.json.gz').exists()


def test_parse_defaults_remain_full_replay_defaults():
    from filter_replay import DEFAULT_AXIS_ARMS, parse_args

    args = parse_args(['--output', 'run'])
    assert args.cases == list(CASE_IDS)
    assert args.axis_arms == DEFAULT_AXIS_ARMS
    assert not args.write_observation_inputs_only
    with pytest.raises(SystemExit):
        parse_args(['--write-observation-inputs-only', '--cases', CASE_IDS[0], '--axis-arms', 'baseline'])


def test_full_replay_preflights_every_case_before_creating_output(monkeypatch, tmp_path):
    import filter_replay

    checked = []

    def require(record):
        checked.append(record)
        if record == 'unsafe':
            raise ValueError('same-image person boxes unavailable')
        return record

    monkeypatch.setattr(filter_replay, 'case_provenance', lambda case_id: case_id)
    monkeypatch.setattr(filter_replay, 'require_same_image_boxes', require)
    monkeypatch.setattr(
        filter_replay,
        'load_case_inputs',
        lambda _case_id: (_ for _ in ()).throw(AssertionError('measurement started before preflight')),
    )
    output = tmp_path / 'run'
    monkeypatch.setattr(
        sys,
        'argv',
        ['filter_replay.py', '--output', str(output), '--cases', 'safe', 'unsafe'],
    )
    with pytest.raises(ValueError, match='same-image person boxes unavailable'):
        filter_replay.main()
    assert checked == ['safe', 'unsafe']
    assert not output.exists()


def test_matcher_person_arms_are_quarantined_before_generation(monkeypatch, tmp_path):
    from shared import add_helper_paths

    add_helper_paths()
    import line_run_matcher

    monkeypatch.setattr(line_run_matcher, 'case_provenance', lambda _case_id: 'unsafe')

    def reject(_record):
        raise ValueError('same-image person boxes unavailable')

    monkeypatch.setattr(line_run_matcher, 'require_same_image_boxes', reject)
    monkeypatch.setattr(
        line_run_matcher,
        'read',
        lambda _path: (_ for _ in ()).throw(AssertionError('input read before provenance gate')),
    )
    with pytest.raises(ValueError, match='same-image person boxes unavailable'):
        line_run_matcher.run_case('case', 'person_observations', object(), tmp_path, tmp_path, 'run', {})
    assert not (tmp_path / 'matcher').exists()
    line_run_matcher.preflight_person_input('case', 'paint')


def test_matcher_preflights_all_ids_before_any_generation(monkeypatch):
    from shared import add_helper_paths

    add_helper_paths()
    import line_run_matcher

    checked = []
    monkeypatch.setattr(line_run_matcher, 'case_provenance', lambda case_id: case_id)

    def require(record):
        checked.append(record)
        if record == 'unsafe':
            raise ValueError('same-image person boxes unavailable')
        return record

    monkeypatch.setattr(line_run_matcher, 'require_same_image_boxes', require)
    monkeypatch.setattr(
        line_run_matcher.importlib,
        'import_module',
        lambda _name: (_ for _ in ()).throw(AssertionError('zone imported before provenance preflight')),
    )
    monkeypatch.setattr(
        line_run_matcher,
        'run_case',
        lambda *_args: (_ for _ in ()).throw(AssertionError('generation started before provenance preflight')),
    )
    monkeypatch.setattr(
        sys,
        'argv',
        ['line_run_matcher.py', '--run', 'test', '--arm', 'person', '--ids', 'safe', 'unsafe'],
    )
    with pytest.raises(ValueError, match='same-image person boxes unavailable'):
        line_run_matcher.main()
    assert checked == ['safe', 'unsafe']


def test_account_quarantines_person_arms_before_measurement(monkeypatch):
    from shared import add_helper_paths

    add_helper_paths()
    import account

    checked = []
    monkeypatch.setattr(account, 'case_provenance', lambda case_id: case_id)

    def require(record):
        checked.append(record)
        if record == 'unsafe':
            raise ValueError('same-image person boxes unavailable')
        return record

    monkeypatch.setattr(account, 'require_same_image_boxes', require)
    monkeypatch.setattr(
        account,
        'direction_experiment_module',
        lambda _name: (_ for _ in ()).throw(AssertionError('measurement setup started before provenance gate')),
    )
    monkeypatch.setattr(
        sys,
        'argv',
        ['account.py', '--run', 'test', '--arms', 'paint_person', '--cases', 'safe', 'unsafe'],
    )
    with pytest.raises(ValueError, match='same-image person boxes unavailable'):
        account.main()
    assert checked == ['safe', 'unsafe']

    checked.clear()
    account.preflight_person_inputs(['unsafe'], ['paint'])
    assert checked == []


@pytest.mark.parametrize('field, value', [
    ('case_id', 'other'),
    ('arm', 'paint'),
    ('stage', 'camera_first'),
    ('run', 'other-run'),
])
def test_account_checks_record_identity_before_hashes(monkeypatch, field, value, tmp_path):
    import account

    record = {
        'case_id': 'case',
        'arm': 'person_observations',
        'stage': 'results',
        'run': 'run',
        'input_case_md5': 'unused',
        'input_estimator_md5': 'unused',
    }
    record[field] = value
    monkeypatch.setattr(account, 'md5', lambda _path: (_ for _ in ()).throw(AssertionError('hashed too early')))
    with pytest.raises(ValueError, match=f'{field} does not match'):
        account.gate_inputs(
            'case', 'person_observations', {}, record, tmp_path,
            expected_stage='results', expected_run='run')


def account_gate_fixture(monkeypatch, tmp_path):
    import account

    source = {'id': 'case', 'frozen': 'source', 'segments_px': [[0., 0., 4., 0.]]}
    filtered = dict(source)
    written = {
        'kept_fragment_ids': [0],
        'estimator': {'directions': 'saved'},
        'settings': {'threshold': 1},
    }
    saved = {'estimator': {'directions': 'saved'}, 'settings': {'threshold': 1}}
    record = {
        'case_id': 'case',
        'arm': 'person_observations',
        'input_case_md5': 'case-hash',
        'input_estimator_md5': 'estimator-hash',
        'fragments_kept': 1,
        'fragments_total': 1,
    }
    monkeypatch.setattr(account, 'md5', lambda path: 'case-hash' if path.parent.name == 'cases' else 'estimator-hash')
    monkeypatch.setattr(account, 'read', lambda path: filtered if path.parent.name == 'cases' else written)
    monkeypatch.setattr(account, 'load_estimator', lambda _case_id: saved)
    return account, source, filtered, written, record


def test_account_hash_gate_raises_explicitly(monkeypatch, tmp_path):
    account, source, _, _, record = account_gate_fixture(monkeypatch, tmp_path)
    record['input_case_md5'] = 'tampered'
    with pytest.raises(ValueError, match='case input changed'):
        account.gate_inputs('case', 'person_observations', source, record, tmp_path)


def test_account_source_gate_raises_explicitly(monkeypatch, tmp_path):
    account, source, filtered, _, record = account_gate_fixture(monkeypatch, tmp_path)
    filtered['frozen'] = 'tampered'
    with pytest.raises(ValueError, match='case fields differ'):
        account.gate_inputs('case', 'person_observations', source, record, tmp_path)


def test_account_segment_gate_raises_explicitly(monkeypatch, tmp_path):
    account, source, filtered, _, record = account_gate_fixture(monkeypatch, tmp_path)
    filtered['segments_px'] = [[1., 0., 4., 0.]]
    with pytest.raises(ValueError, match='kept fragments differ'):
        account.gate_inputs('case', 'person_observations', source, record, tmp_path)


def test_account_saved_estimator_gate_raises_explicitly(monkeypatch, tmp_path):
    account, source, _, written, record = account_gate_fixture(monkeypatch, tmp_path)
    written['estimator'] = {'directions': 'tampered'}
    with pytest.raises(ValueError, match='baseline directions changed'):
        account.gate_inputs('case', 'person_observations', source, record, tmp_path)


def test_repair_mask_uses_inclusive_two_of_three_boxes_and_deduplicates():
    import repair_inputs

    frame_boxes = [
        np.array([[0., 0., 4., 4.], [10., 10., 12., 12.]]),
        np.array([[4., 1., 8., 5.], [10., 10., 12., 12.]]),
        np.array([[1., 1., 3., 3.], [10., 10., 12., 12.]]),
    ]
    boxes = repair_inputs.inclusive_pairwise_intersections(frame_boxes)
    np.testing.assert_array_equal(boxes, [[1., 1., 3., 3.], [4., 1., 4., 4.], [10., 10., 12., 12.]])


def test_repair_score_cut_is_strictly_above_point_two():
    import repair_inputs

    boxes = [[0., 0., 1., 1.], [2., 2., 3., 3.], [4., 4., 5., 5.]]
    selected = repair_inputs.eligible_boxes(boxes, [0.2, 0.2000001, 0.1])
    np.testing.assert_array_equal(selected, [[2., 2., 3., 3.]])


def test_repair_exact_frame_requires_native_identity_and_frozen_image_hash(monkeypatch, tmp_path):
    import repair_inputs

    assert len(repair_inputs.DETECTION_CASE_IDS) == 6
    assert repair_inputs.DETECTION_CASE_IDS[-1] == 'am4_window_00_frame_319'
    image = tmp_path / 'gxBQ_window_00_frame_00000005.png'
    image.write_bytes(b'frozen image')
    image_md5 = hashlib.md5(image.read_bytes()).hexdigest()
    source = {
        'id': 'gxBQ_window_00_frame_5',
        'image': 'images/gxBQ_window_00_frame_00000005.png',
        'dimensions': {'width': 1920, 'height': 1080},
        'provenance': {'anchor_frame_index': 5, 'line_image_file_md5': image_md5},
    }
    path = write_exact_detection_packet(tmp_path, repair_inputs, image_md5)
    detections = json.loads(path.read_text())
    monkeypatch.setattr(repair_inputs, 'frame_path', lambda _source: image)
    evidence = repair_inputs.exact_frame_evidence(source, path)
    assert evidence['frame_index'] == 5
    detections['cases']['gxBQ_window_00_frame_5']['frame_index'] = 6
    path.write_text(json.dumps(detections))
    with pytest.raises(ValueError, match='expected 5'):
        repair_inputs.exact_frame_evidence(source, path)


@pytest.mark.parametrize('mutation, match', [
    ('frame', 'expected 5'),
    ('image', 'expected'),
    ('md5', 'hexadecimal'),
    ('dimensions', 'expected 1920x1080'),
    ('boxes', 'image coordinates'),
    ('scores', 'between zero and one'),
])
def test_repair_detector_packet_pins_all_canonical_metadata(tmp_path, mutation, match):
    import repair_inputs

    path = write_exact_detection_packet(tmp_path, repair_inputs, 'a' * 32)
    packet = json.loads(path.read_text())
    record = packet['cases'][repair_inputs.GX5_CASE_ID]
    if mutation == 'frame':
        record['frame_index'] = 6
    elif mutation == 'image':
        record['image'] = 'wrong.png'
    elif mutation == 'md5':
        record['image_md5'] = 'z' * 32
    elif mutation == 'dimensions':
        record['dimensions']['width'] = 1280
    elif mutation == 'boxes':
        record['bboxes'] = [[-2., 1., 3., 3.]]
    elif mutation == 'scores':
        record['scores'] = [1.1]
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match=match):
        repair_inputs._case_detection(packet, repair_inputs.GX5_CASE_ID)

    if mutation == 'frame':
        del packet['cases'][repair_inputs.DETECTION_CASE_IDS[-1]]
        with pytest.raises(ValueError, match='must contain exactly'):
            repair_inputs._case_detection(packet, repair_inputs.GX5_CASE_ID)


def test_repair_manifest_rejects_another_arm(tmp_path):
    import repair_inputs

    path = tmp_path / 'manifest.json.gz'
    from shared import write

    write(path, {'schema': repair_inputs.SCHEMA, 'arm': 'paint', 'cases': []})
    with pytest.raises(ValueError, match='only valid for person_observations'):
        repair_inputs.load_manifest(path)


def test_repair_matcher_preflights_the_whole_batch_before_zone_import(monkeypatch):
    from shared import add_helper_paths

    add_helper_paths()
    import line_run_matcher

    calls = []
    monkeypatch.setattr(line_run_matcher.repair_inputs, 'load_manifest', lambda _path: {'manifest': True})
    monkeypatch.setattr(line_run_matcher.repair_inputs, 'validate_repair_inputs',
                        lambda ids, inputs, manifest: calls.append((ids, inputs, manifest)))
    monkeypatch.setattr(
        line_run_matcher.importlib,
        'import_module',
        lambda _name: (_ for _ in ()).throw(AssertionError('zone imported before repair preflight')),
    )
    monkeypatch.setattr(sys, 'argv', [
        'line_run_matcher.py', '--run', 'test', '--arm', 'person_observations',
        '--ids', 'first', 'second', '--inputs-dir', str(Path('/tmp/repair-inputs')),
        '--repair-manifest', '/tmp/repair-manifest.json.gz',
    ])
    with pytest.raises(AssertionError, match='zone imported'):
        line_run_matcher.main()
    assert calls == [(['first', 'second'], Path('/tmp/repair-inputs'), {'manifest': True})]


def test_repair_matcher_rejects_existing_stage_before_zone_import(monkeypatch, tmp_path):
    from shared import add_helper_paths

    add_helper_paths()
    import line_run_matcher

    output = tmp_path / 'runs' / 'test'
    existing = output / 'matcher' / 'person_observations' / 'results' / 'first.json.gz'
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b'old result')
    monkeypatch.setattr(line_run_matcher, 'HERE', tmp_path)
    monkeypatch.setattr(line_run_matcher.repair_inputs, 'load_manifest', lambda _path: {'manifest': True})
    monkeypatch.setattr(line_run_matcher.repair_inputs, 'validate_repair_inputs', lambda *_args: None)
    monkeypatch.setattr(
        line_run_matcher.importlib,
        'import_module',
        lambda _name: (_ for _ in ()).throw(AssertionError('zone imported before output preflight')),
    )
    monkeypatch.setattr(sys, 'argv', [
        'line_run_matcher.py', '--run', 'test', '--arm', 'person_observations', '--ids', 'first', 'second',
        '--inputs-dir', str(tmp_path / 'inputs'), '--repair-manifest', str(tmp_path / 'manifest.json.gz'),
    ])
    with pytest.raises(FileExistsError, match='repair output paths already exist'):
        line_run_matcher.main()


def test_repair_validation_rejects_tampered_fragments_and_manifest(monkeypatch, tmp_path):
    import repair_inputs

    from shared import read, write

    monkeypatch.setattr(repair_inputs, '_gate_saved_estimator', lambda _source, _saved: None)
    inputs = tmp_path / 'inputs'
    manifest_path = tmp_path / 'manifest.json.gz'
    detections_path = write_repair_detection_packet(tmp_path, repair_inputs)
    manifest = repair_inputs.build_repair_inputs(
        inputs, manifest_path, repair_inputs.REPAIR_CASE_IDS, detections_path)
    repair_inputs.validate_repair_inputs(repair_inputs.REPAIR_CASE_IDS, inputs, manifest)

    case_id = 'shuttleset_03_scene_0016'
    case_path = inputs / repair_inputs.ARM / 'cases' / f'{case_id}.json.gz'
    case = read(case_path)
    case['segments_px'][0][0] += 1.
    write(case_path, case)
    with pytest.raises(ValueError, match='filtered fragments'):
        repair_inputs.validate_repair_inputs([case_id], inputs, manifest)

    # Restore the input and tamper with the manifest itself.
    source = repair_inputs.load_source(case_id)
    entry = next(entry for entry in manifest['cases'] if entry['case_id'] == case_id)
    case['segments_px'][0] = source['segments_px'][entry['kept_fragment_ids'][0]]
    write(case_path, case)
    entry['kept_fragment_ids'] = [999]
    with pytest.raises(ValueError, match='kept IDs'):
        repair_inputs.validate_repair_inputs([case_id], inputs, manifest)


def test_repair_builder_requires_the_fixed_case_allowlist(tmp_path):
    import repair_inputs

    with pytest.raises(ValueError, match='exactly the five-case repair allowlist'):
        repair_inputs.build_repair_inputs(
            tmp_path / 'inputs', tmp_path / 'manifest.json.gz', ['shuttleset_03_scene_0016'])


def test_repair_builder_requires_fresh_targets(monkeypatch, tmp_path):
    import repair_inputs

    monkeypatch.setattr(repair_inputs, '_gate_saved_estimator', lambda _source, _saved: None)
    with pytest.raises(ValueError, match='outside historical'):
        repair_inputs.build_repair_inputs(
            repair_inputs.HISTORICAL_INPUTS / 'new', tmp_path / 'manifest.json.gz',
            repair_inputs.REPAIR_CASE_IDS)
    with pytest.raises(ValueError, match='outside historical'):
        repair_inputs.build_repair_inputs(
            tmp_path / 'inputs', repair_inputs.HISTORICAL_INPUTS / 'manifest.json.gz',
            repair_inputs.REPAIR_CASE_IDS)

    target = tmp_path / 'inputs'
    (target / repair_inputs.ARM).mkdir(parents=True)
    with pytest.raises(FileExistsError, match='arm directory already exists'):
        repair_inputs.build_repair_inputs(target, tmp_path / 'manifest.json.gz', repair_inputs.REPAIR_CASE_IDS)

    (target / repair_inputs.ARM).rmdir()
    manifest_path = tmp_path / 'manifest.json.gz'
    manifest_path.write_bytes(b'old')
    with pytest.raises(FileExistsError, match='manifest already exists'):
        repair_inputs.build_repair_inputs(target, manifest_path, repair_inputs.REPAIR_CASE_IDS)


def test_repair_builder_derives_all_cases_before_writing(monkeypatch, tmp_path):
    import repair_inputs

    def fail_on_second(source, _saved):
        if source['id'] == 'shuttleset_03_scene_0017':
            raise RuntimeError('synthetic derivation failure')

    monkeypatch.setattr(repair_inputs, '_gate_saved_estimator', fail_on_second)
    inputs = tmp_path / 'inputs'
    manifest_path = tmp_path / 'manifest.json.gz'
    detections_path = write_repair_detection_packet(tmp_path, repair_inputs)
    with pytest.raises(RuntimeError, match='synthetic derivation failure'):
        repair_inputs.build_repair_inputs(inputs, manifest_path, repair_inputs.REPAIR_CASE_IDS, detections_path)
    assert not (inputs / repair_inputs.ARM).exists()
    assert not manifest_path.exists()

    # A partial arm directory from an interrupted write cannot be reused.
    (inputs / repair_inputs.ARM).mkdir(parents=True)
    with pytest.raises(FileExistsError, match='arm directory already exists'):
        repair_inputs.build_repair_inputs(inputs, manifest_path, repair_inputs.REPAIR_CASE_IDS, detections_path)


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
    source = {'id': 'gxBQ_window_00_frame_0', 'segments_px': [[10., 20., 50., 20.], [10., 5., 50., 5.]],
              'bbox_px': [[20., 15., 40., 25.]]}
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


def test_amateur_frame_path_rejects_mismatched_typed_frame(monkeypatch):
    import shared

    source = {'id': 'am2_window_00_frame_150'}
    provenance = CaseProvenance(source['id'], ImageKind.SOURCE_FRAME, (151,), 151)
    monkeypatch.setattr(shared, 'case_provenance', lambda _case_id: provenance)

    with pytest.raises(ValueError, match='amateur frame path uses frame 150'):
        frame_path(source)
