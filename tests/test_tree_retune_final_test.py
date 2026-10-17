"""Checks for final contact refits and the frozen five-seed test comparison."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import tree_retune_final_test as final


def test_selection_gate_precedes_any_test_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repeats = tmp_path / 'repeats'
    repeats.mkdir()
    final.grid.write_json(repeats / 'selected.json.gz', {'config_id': 'not-frozen'})
    final.grid.write_json(repeats / 'progress.json.gz', {'phase': 'repeat_validation'})

    def forbidden(*_args: object) -> None:
        pytest.fail('test data was read before frozen selection')

    monkeypatch.setattr(final, 'load_test_videos', forbidden)
    with pytest.raises(ValueError, match='complete, frozen'):
        final.freeze_plan(tmp_path, tmp_path, None)


def selected_document(root: Path) -> dict:
    spec = final.grid.CandidateSpec('shipped', {}, {})
    identity = final.grid.candidate_id(spec)
    bundles = []
    for offset in final.repeats.SEED_OFFSETS:
        bundle = final.repeats.seed_root(root, offset) / 'trials' / identity / 'bundle_v32'
        bundle.mkdir(parents=True)
        bundles.append({'seed_offset': offset, 'bundle': str(bundle)})
    row = {'id': identity, 'label': 'shipped', 'overrides_base30': {}, 'strategies': {}, 'bundles': bundles}
    return {
        'schema': final.repeats.SCHEMA, 'config_id': identity, 'label': 'shipped',
        'seed_offsets': list(final.repeats.SEED_OFFSETS), 'ranking': [identity], 'results': [row],
        'bundles': bundles, 'baseline': {'config_id': identity, 'label': 'shipped', 'bundles': bundles},
    }


def test_selection_requires_all_five_seeds_and_deduplicates_baseline(tmp_path: Path) -> None:
    document = selected_document(tmp_path)
    directory = tmp_path / 'repeats'
    final.grid.write_json(directory / 'selected.json.gz', document)
    final.grid.write_json(directory / 'progress.json.gz', {'phase': 'complete', 'selected_id': document['config_id']})
    _, candidates = final.frozen_selection(tmp_path)
    assert len(candidates) == 1
    assert [item['seed_offset'] for item in candidates[0]['bundles']] == [0, 100, 200, 300, 400]
    document['results'][0]['bundles'] = document['results'][0]['bundles'][:1]
    final.grid.write_json(directory / 'selected.json.gz', document)
    with pytest.raises(ValueError, match='every seed'):
        final.frozen_selection(tmp_path)


def original_videos() -> tuple:
    document = json.loads(final.grid.GROUPS_FILE.read_text())
    validation = document['fixed_validation_videos']
    training = [video['fixture'] for group in document['groups'] for video in group['videos']]
    return tuple(
        final.grid.SearchVideo(
            identity, 'ShuttleSet', 'validation' if identity in validation else 'train', 25, (1280, 720), (),
        )
        for identity in [*training, *validation]
    )


def test_final_contact_uses_original_40_order_and_preserves_downstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = selected_document(tmp_path)
    candidate = document['results'][0]
    videos = original_videos()
    development = final.grid.development_for(videos)
    expected = final.grid.retrain.contact_fit_videos(development)['final_40']
    models = final.grid.AnnotatorModels('old-contact', 'same-sequences', 'same-confidence')
    saved = []
    calls = []
    pose = SimpleNamespace(bboxes=None, scores=None, kps=None, ndet=None)
    inputs = SimpleNamespace(track=None, pose=pose)
    monkeypatch.setattr(final.grid, 'load_models', lambda _path: models)
    monkeypatch.setattr(final.grid, 'load_original_videos', lambda _path: videos)
    monkeypatch.setattr(final.grid, 'load_original_inputs', lambda *_args: inputs)
    monkeypatch.setattr(final.grid, 'run_video_kwargs', lambda _inputs: {})
    monkeypatch.setattr(final.grid, 'cache_sticky_result', lambda *_args: nullcontext())
    monkeypatch.setattr(final.grid, 'features_from_evidence', lambda _evidence: SimpleNamespace(rows=final.grid.np.zeros((1, 1))))

    def run_video(*_args: object, **kwargs: object) -> None:
        kwargs['capture'].contact_evidence = SimpleNamespace()

    def fit_contact(training: list, config: object, *, fit_video_order: tuple) -> SimpleNamespace:
        assert len(training) == 40
        assert tuple(video.identity for video in training) == expected
        assert fit_video_order == development.id_order(expected)
        assert config == final.repeats.seeded_settings(200).contact
        calls.append('fit')
        return SimpleNamespace(model='new-contact', selection=SimpleNamespace(labels=final.grid.np.asarray([1])))

    monkeypatch.setattr(final.grid, 'run_video', run_video)
    monkeypatch.setattr(final.grid, 'fit_contact_model', fit_contact)
    monkeypatch.setattr(final.grid, 'save_models', lambda model, _path: saved.append(model))
    paths = {name: tmp_path for name in ('shots_master', 'stages', 'development_courts')}
    record = final.fit_final_contact(tmp_path, candidate, 200, paths)
    assert calls == ['fit']
    assert saved[0].contact == 'new-contact'
    assert saved[0].sequences is models.sequences
    assert saved[0].confidence is models.confidence
    assert saved[0].preprocessing is models.preprocessing
    assert saved[0].contact_settings is models.contact_settings
    assert saved[0].side_geometry is models.side_geometry
    assert record['sampling_order'] == record['fit_order'] == list(expected)


def test_final_job_scores_all_46_and_resumes_saved_streams(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    document = selected_document(tmp_path)
    candidate = document['results'][0]
    models = SimpleNamespace(preprocessing='frozen')
    videos = [SimpleNamespace(identity=f'test-{index}') for index in range(46)]
    trial = final.trial_root(tmp_path, candidate['id'], 100)
    trial.mkdir(parents=True)
    final.grid.write_json(trial / 'contact-fit.json.gz', {'seed_offset': 100})
    calls = []
    monkeypatch.setattr(final.grid, 'load_models', lambda _path: models)
    monkeypatch.setattr(final, 'load_test_videos', lambda _path: videos)
    monkeypatch.setattr(final, 'load_test_inputs', lambda video, *_args: video)
    monkeypatch.setattr(final.grid, 'cache_sticky_result', lambda *_args: nullcontext())

    def run(inputs: object, base: object, *, models: object) -> dict:
        assert base == 'frozen'
        calls.append(inputs.identity)
        return {'events': [], 'sequences': []}

    monkeypatch.setattr(final.grid, 'run_configuration', run)
    counts = {name: 1 for name in final.grid.COUNT_FIELDS}
    monkeypatch.setattr(final.grid, 'score_stream', lambda *_args: {'5': counts, '10': counts})
    paths = {name: tmp_path for name in ('test_inputs', 'inpaint_root', 'pose_root', 'test_courts')}
    final.final_job(tmp_path, candidate, 100, paths)
    result = final.grid.read_json(trial / 'result.json.gz')
    assert len(calls) == len(result['per_video']) == 46
    assert result['totals']['10']['labelled_contacts'] == 46
    final.final_job(tmp_path, candidate, 100, paths)
    assert len(calls) == 46


def test_summary_compares_paired_means_without_choosing_on_test(tmp_path: Path) -> None:
    candidates = [{'id': 'winner', 'label': 'tuned'}, {'id': 'baseline', 'label': 'shipped'}]
    plan = {
        'candidates': candidates, 'test_videos': [f'test-{index}' for index in range(46)],
        'selected': {'config_id': 'winner', 'baseline': {'config_id': 'baseline'}}, 'comparison': 'frozen',
    }
    for candidate in candidates:
        for offset in final.repeats.SEED_OFFSETS:
            score = (0.5 if candidate['id'] == 'winner' else 0.6) + offset / 10000
            trial = final.trial_root(tmp_path, candidate['id'], offset)
            trial.mkdir(parents=True)
            final.grid.write_json(trial / 'result.json.gz', {
                'id': candidate['id'], 'label': candidate['label'], 'seed_offset': offset,
                'per_video': {identity: {} for identity in plan['test_videos']},
                'totals': {'10': {'attributed_f1': score}, '5': {'attributed_f1': score}},
            })
    summary = final.summarise(tmp_path, plan)
    assert summary['config_id'] == 'winner'
    assert summary['mean_paired_delta']['10']['attributed_f1'] == pytest.approx(-0.1)
    assert len(summary['paired_deltas']) == 5
    assert [row['seed_offset'] for row in summary['paired_deltas']] == [0, 100, 200, 300, 400]
    winner = summary['results'][0]
    assert winner['mean_test']['10']['attributed_f1'] == pytest.approx(0.52)
    assert winner['spread_test']['10']['attributed_f1']['max'] == pytest.approx(0.54)


def test_frozen_plan_pins_test_inputs_and_bundles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selected = selected_document(tmp_path)
    candidates = selected['results']
    paths = {name: tmp_path / name for name in (
        'shots_master', 'stages', 'development_courts', 'test_inputs', 'test_courts', 'inpaint_root', 'pose_root',
    )}
    test_manifest = tmp_path / 'manifest.json.gz'
    test_manifest.write_bytes(b'original')
    for bundle in selected['bundles']:
        for filename in ('models.joblib', 'metadata.json'):
            (Path(bundle['bundle']) / filename).write_bytes(b'frozen-bundle')
    videos = [SimpleNamespace(identity=f'test-{index}', dataset='ShuttleSet22', split='test') for index in range(46)]
    monkeypatch.setattr(final, 'frozen_selection', lambda _root: (selected, candidates))
    monkeypatch.setattr(final.repeats, 'verify_snapshot', lambda *_args: ({'inputs': 'original40'}, [], paths))
    monkeypatch.setattr(final, 'load_test_videos', lambda _path: videos)
    monkeypatch.setattr(final.grid, 'input_paths', lambda *_args: {'test_manifest': test_manifest})
    plan, _ = final.freeze_plan(tmp_path, tmp_path, None)
    assert len(plan['bundles']) == 10
    assert all('md5' in identity for identity in plan['bundles'].values())
    assert final.freeze_plan(tmp_path, tmp_path, None)[0] == plan
    test_manifest.write_bytes(b'changed test input')
    with pytest.raises(ValueError, match='frozen final test plan changed'):
        final.freeze_plan(tmp_path, tmp_path, None)
