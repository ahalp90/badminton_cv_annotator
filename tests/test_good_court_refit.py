from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_builder.vision import (
    load_npy_xz,
    load_pose_arrays,
    save_json_gz,
    save_npy_xz,
    save_pose_arrays,
)
from experiments.annotator.good_court_refit import runner
from experiments.annotator.good_court_refit.runner import (
    ARM_CONFIGS,
    RefitConfig,
    RefitContext,
    contact_fit_plan,
    court_conversion_job,
    expected_id_maps,
    load_video_inputs,
    run_pair,
    selected_test_ids,
    source_for,
    validate_court_map,
)
from experiments.annotator.old_court_regression.retrain import (
    TEST_VIDEO_IDS,
    Development,
    trusted_test_label_rows,
)
from experiments.annotator.old_court_regression.runner import Progress


def small_development() -> Development:
    groups = {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'sset_15': 'V'}
    ids = {'a': 5, 'b': 40, 'c': 10, 'd': 1, 'sset_15': 15}
    return Development(groups, {video: 30.0 for video in groups}, ids)


def context_with_maps(court_json: dict[str, dict[str, Path | None]]) -> RefitContext:
    empty = Path('/unused')
    config = RefitConfig(
        empty, empty, empty, empty, empty, empty, empty, empty, empty, court_json,
    )
    return RefitContext(
        config, small_development(),
        {video: (1920.0, 1080.0) for video in small_development().group_by_video},
    )


def test_test_population_excludes_only_misaligned_video_15() -> None:
    assert selected_test_ids() == tuple(video_id for video_id in TEST_VIDEO_IDS if video_id != 15)
    assert len(selected_test_ids()) == 46
    assert 15 not in selected_test_ids()


def test_contact_sampling_order_is_preserved_before_id_sorted_fit() -> None:
    sampling_order, fit_order = contact_fit_plan(small_development(), 'held_A')
    assert sampling_order == ('b', 'c', 'd')
    assert fit_order == ('d', 'c', 'b')


def test_development_sset_15_stays_in_pinned_population() -> None:
    context = context_with_maps({'development': {}, 'test': {}})
    assert 'sset_15' in expected_id_maps(context)['development']


def test_court_maps_require_every_expected_id_and_no_test_video_15() -> None:
    expected = context_with_maps({'development': {}, 'test': {}})
    with pytest.raises(ValueError, match='must match the pinned video IDs'):
        validate_court_map(expected)

    maps = {
        'development': {
            video: Path(f'/court/{video}.json') for video in small_development().group_by_video
        },
        'test': {str(video_id): Path(f'/court/{video_id}.json') for video_id in selected_test_ids()},
    }
    validate_court_map(context_with_maps(maps))
    maps['test']['15'] = Path('/court/15.json')
    with pytest.raises(ValueError, match='must match the pinned video IDs'):
        validate_court_map(context_with_maps(maps))


def test_veto_is_stored_as_a_contact_prediction_setting() -> None:
    assert not ARM_CONFIGS['base'].reject_masked_without_player
    assert ARM_CONFIGS['veto'].reject_masked_without_player


def test_tracked_test_labels_match_the_pinned_population_and_counts() -> None:
    label_path = Path(__file__).parents[1] / 'experiments/annotator/good_court_refit/shuttleset22_labels.json.gz'
    labels = trusted_test_label_rows(label_path)
    assert tuple(sorted(labels)) == selected_test_ids()
    assert 15 not in labels
    assert sum(map(len, labels.values())) == 3327
    assert sum(len(contacts) for rallies in labels.values() for _rally, contacts in rallies) == 37184


def test_preflight_court_conversion_and_current_loader_use_saved_inputs(tmp_path: Path, monkeypatch) -> None:
    identity = 'sset_15'
    frame_count = 2
    stage_root = tmp_path / 'development'
    shuttle_dir = stage_root / 'stages/shuttle' / identity
    pose_dir = stage_root / 'stages/pose' / identity
    track_path = shuttle_dir / 'shuttle_track.npy.xz'
    save_npy_xz(track_path, np.zeros((frame_count, 3), dtype=np.float32))
    save_npy_xz(shuttle_dir / 'shuttle_guard_codes.npy.xz', np.zeros(frame_count, dtype=np.uint8))

    one_frame_boxes = np.asarray(
        [[[350, 350, 450, 550], [1450, 350, 1550, 550]]], dtype=np.float32,
    )
    poses = runner.PoseArrays(
        kps=np.zeros((frame_count, 2, 17, 2), dtype=np.float32),
        bboxes=np.repeat(one_frame_boxes, frame_count, axis=0),
        scores=np.ones((frame_count, 2), dtype=np.float32),
        kp_scores=np.ones((frame_count, 2, 17), dtype=np.float32),
        ndet=np.full(frame_count, 2, dtype=np.int64),
    )
    save_pose_arrays(pose_dir, poses, frame_count)
    court_path = tmp_path / 'fresh-court.json.gz'
    save_json_gz(court_path, {
        'schema': 'court-detector-video/1', 'frame_count': frame_count, 'native_size': [1920, 1080],
        'scenes': [{
            'start_frame': 0, 'end_frame': frame_count, 'frame_index': 0, 'status': 'court',
            'corners_native_px': [[100, 100], [1820, 100], [1820, 980], [100, 980]],
            'no_court_reason': None, 'reused_from': None, 'error': None,
        }],
    })
    source_files = {}
    for name in ('shots.csv', 'groups.json', 'split.json', 'labels.json.gz'):
        path = tmp_path / name
        path.write_text('synthetic input')
        source_files[name] = path
    config = RefitConfig(
        tmp_path / 'inputs.json', tmp_path / 'output', stage_root, tmp_path, tmp_path,
        source_files['shots.csv'], source_files['groups.json'], source_files['split.json'],
        source_files['labels.json.gz'], {'development': {identity: court_path}, 'test': {}},
    )
    development = small_development()
    context = RefitContext(
        config, development,
        {video: (1920.0, 1080.0) for video in development.group_by_video},
    )
    monkeypatch.setattr(runner.retrain, 'load_development', lambda _settings: development)
    monkeypatch.setattr(runner, 'expected_id_maps', lambda _context: {'development': (identity,), 'test': ()})
    monkeypatch.setattr(runner, 'read_label_sources', lambda _context: ({identity: []}, {}))

    source = source_for(context, 'development', identity)
    assert source.frame_count == len(load_npy_xz(track_path)) == frame_count
    assert source.resolution == (1920.0, 1080.0)
    snapshot = runner.preflight(context, compare_snapshot=False)
    assert snapshot['schema'] == 'good-court-refit-snapshot/1'
    assert str(track_path.resolve()) in {entry['path'] for entry in snapshot['files']}
    summary = court_conversion_job(context, 'development', identity)
    assert summary['accepted_scenes'] == 1

    inputs = load_video_inputs(context, 'development', identity)
    assert inputs.court.resolution == (1920.0, 1080.0)
    assert inputs.court_present.tolist() == [True, True]
    assert len(load_pose_arrays(pose_dir, frame_count).bboxes) == frame_count

    malformed = runner.load_detector_json(court_path)
    del malformed['scenes'][0]['end_frame']
    save_json_gz(court_path, malformed)
    with pytest.raises(ValueError, match=r'development sset_15 .*fresh-court.json.gz.*end_frame'):
        runner.check_video_source(source)


@pytest.mark.parametrize('fail_base', [False, True])
def test_pair_runner_finishes_base_before_veto_and_stops_on_failure(tmp_path: Path, monkeypatch, fail_base) -> None:
    development = small_development()
    court_map = {
        'development': {video: Path(f'/court/{video}.json') for video in development.group_by_video},
        'test': {str(video_id): Path(f'/court/{video_id}.json') for video_id in selected_test_ids()},
    }
    context = context_with_maps(court_map)
    calls: list[tuple[str, str | None]] = []

    def fake_source_for(_context, split: str, identity: str):
        return SimpleNamespace(court_dir=tmp_path / 'converted' / split / identity)

    def fake_run_stage(stage, items, jobs, threads, output, progress: Progress):
        if stage == 'contact-fits':
            assert (jobs, threads) == (6, 5)
        del jobs, threads, output, progress
        arm = items[0].arguments[1] if stage in ('pools', 'sequence-fit') else None
        calls.append((stage, arm))
        return ['sequence-fit'] if fail_base and stage == 'sequence-fit' and arm == 'base' else []

    monkeypatch.setattr(runner, 'source_for', fake_source_for)
    monkeypatch.setattr(runner, 'run_stage', fake_run_stage)
    if fail_base:
        with pytest.raises(RuntimeError, match='sequence-fit failed'):
            run_pair(context, tmp_path / 'run')
        assert calls[-2:] == [('pools', 'base'), ('sequence-fit', 'base')]
        assert not any(arm == 'veto' for _stage, arm in calls)
    else:
        run_pair(context, tmp_path / 'run')
        assert calls[-4:] == [('pools', 'base'), ('sequence-fit', 'base'),
                              ('pools', 'veto'), ('sequence-fit', 'veto')]


def test_default_cli_rejects_blank_courts_without_creating_output(tmp_path: Path) -> None:
    output = tmp_path / 'never-started'
    with pytest.raises(SystemExit) as error:
        runner.main(['--output', str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_resume_rejects_changed_inputs_and_evaluation_needs_existing_run(tmp_path: Path) -> None:
    config = replace(context_with_maps({'development': {}, 'test': {}}).config, output=tmp_path / 'run')
    snapshot = {'files': [{'path': '/court.json', 'size': 12, 'mtime_ns': 100}]}
    with pytest.raises(FileNotFoundError, match='no completed/prepared run'):
        runner.write_or_check_snapshot(config, snapshot, write=False)
    assert not config.output.exists()
    runner.write_or_check_snapshot(config, snapshot, write=True)
    runner.write_or_check_snapshot(config, snapshot, write=False)
    changed = {'files': [{'path': '/court.json', 'size': 12, 'mtime_ns': 200}]}
    with pytest.raises(ValueError, match='inputs differ'):
        runner.write_or_check_snapshot(config, changed, write=True)


@pytest.mark.parametrize('arm', ['base', 'veto'])
def test_sequence_job_keeps_held_predictions_and_saves_both_bundle_roles(tmp_path: Path, monkeypatch, arm) -> None:
    config = replace(context_with_maps({'development': {}, 'test': {}}).config, output=tmp_path)
    context = replace(context_with_maps(config.court_json), config=config)
    identities = context.development.videos(runner.retrain.TRAINING_GROUPS)
    predictions = {identity: SimpleNamespace(sequences=(), events=()) for identity in identities}
    fit = SimpleNamespace(models=object(), cross_fitted=predictions)
    contact_models = {'final_40': object(), 'v_32': object()}
    bundles = {}

    def load(path):
        return SimpleNamespace(identity=path.stem) if path.parent.name == 'pools' else contact_models[path.stem]

    monkeypatch.setattr(runner.joblib, 'load', load)
    monkeypatch.setattr(runner.retrain, 'fit_downstream', lambda _videos: (fit, object(), np.array([1, 0, -1])))
    monkeypatch.setattr(runner, 'save_models', lambda models, path: bundles.update({path.name: models}))
    summary = runner.sequence_fit_job(context, arm)
    assert summary['training_videos'] == list(identities)
    assert bundles['bundle'].contact is contact_models['final_40']
    assert bundles['bundle_v32'].contact is contact_models['v_32']
    assert bundles['bundle'].contact_settings.reject_masked_without_player == (arm == 'veto')
    for identity in identities:
        saved = runner.load_json_gz(tmp_path / arm / 'dev_heldout' / f'{identity}.json.gz')
        assert saved == runner.retrain.stream_payload((), ())
    assert (tmp_path / arm / 'sequence/summary.json.gz').is_file()


def test_run_record_keeps_actions_and_refuses_changed_fit_environment(tmp_path: Path, monkeypatch) -> None:
    config = replace(context_with_maps({'development': {}, 'test': {}}).config, output=tmp_path)
    runner.record_invocation(config, 'fit', None)
    runner.record_invocation(config, 'validation', 'base')
    path = tmp_path / 'run.json'
    records = runner.json.loads(path.read_text())
    assert [(row['action'], row['arm']) for row in records] == [('fit', None), ('validation', 'base')]
    assert records[0]['versions']['scikit-learn'] == runner.version('scikit-learn')
    assert records[0]['git_commit']
    monkeypatch.setattr(runner.platform, 'python_version', lambda: 'changed')
    with pytest.raises(ValueError, match='Python/package versions changed'):
        runner.record_invocation(config, 'test', 'veto')
    assert runner.json.loads(path.read_text()) == records
