"""Synthetic contracts for the ShuttleSet22 test evaluation."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from annotator.config import BaseAnnotatorConfig
from annotator.models import SideGeometry, model_files
from annotator.sequence.rally_model import RallyLayerModels
from annotator.training.workflow import DEFAULT_TRAINING_SETTINGS
from experiments.annotator.heuristic_only_tune.run import read_json, write_json
from experiments.annotator.tree_simplification import input_stages, input_test
from experiments.annotator.tree_simplification.prepare import preprocessing_record


def _write_chosen_export(root: Path, seed_offset: int = 0) -> tuple[dict, dict[str, str]]:
    export = root / 'export'
    bundle = export / 'bundle'
    bundle.mkdir(parents=True)
    for path in model_files(bundle).values():
        path.write_bytes(path.name.encode())
    hashes = {name: input_test._md5(path) for name, path in model_files(bundle).items()}

    contact_root = root / 'contacts'
    contact_root.mkdir()
    settings = asdict(DEFAULT_TRAINING_SETTINGS)
    settings.pop('rally_layers')
    settings['contact']['random_seed'] = DEFAULT_TRAINING_SETTINGS.contact.random_seed + seed_offset
    settings['confidence']['random_seed'] = DEFAULT_TRAINING_SETTINGS.confidence.random_seed + seed_offset
    settings = json.loads(json.dumps(settings, default=sorted))
    contact_settings = {
        'prepared': str(root / 'prepared'),
        'settings': settings,
    }
    fits = {name: {'fit': name} for name in ('held_A', 'held_B', 'held_C', 'held_D', 'v_32')}
    write_json(contact_root / 'contact-stage.json.gz', {
        'schema': 'tree-simplification-contacts/1',
        'seed_offset': seed_offset,
        'fits': fits,
        'settings': contact_settings,
    })

    preprocessing = preprocessing_record(BaseAnnotatorConfig())
    export_settings = {
        'schema': 'tree-simplification-raw-export/1',
        'settings': {
            'pool_settings': {
                'contact_run': str(contact_root),
                'prepared': str(root / 'prepared'),
                'settings': contact_settings['settings'],
            },
            'tree': {
                'random_seed': DEFAULT_TRAINING_SETTINGS.rally_layers.tree.random_seed + seed_offset,
            },
        },
        'preprocessing': preprocessing,
    }
    write_json(export / 'settings.json.gz', export_settings)
    write_json(export / 'result.json.gz', {
        'complete': True,
        'replay_streams_match': True,
        'per_video': {f'development-{index}': {} for index in range(8)},
    })
    return export_settings, hashes


def _write_selection(root: Path, seed_offset: int = 0) -> tuple[Path, dict, dict[str, str]]:
    export_settings, hashes = _write_chosen_export(root, seed_offset)
    export = str((root / 'export').resolve())
    seeds = [
        {
            'offset': offset,
            'export': export,
            'bundle_md5': hashes,
            'settings': export_settings,
        }
        for offset in input_test.SEED_OFFSETS
    ]
    document = {
        'schema': input_test.SCHEMA,
        'evaluation_source': {'revision': 'synthetic', 'dirty': False},
        'fit_policy': 'contact-fit-32-videos',
        'feature_units': 'raw',
        'seed_offsets': list(input_test.SEED_OFFSETS),
        'primary_tolerance': 10,
        'finalists': [{'candidate': 'candidate-a', 'objectives': ['validation'], 'seeds': seeds}],
    }
    selection_path = root / 'selection.json.gz'
    write_json(selection_path, document)
    return selection_path, document, hashes


def _runtime_patches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, fail_video: str | None = None):
    identities = tuple(f'test-{index:02d}' for index in range(46))
    videos = [
        SimpleNamespace(
            identity=identity, dataset='ShuttleSet22', split='test', fps=30.0,
            resolution=(1920.0, 1080.0), frame_count=12, rallies=(), directory=identity,
        )
        for identity in identities
    ]
    models = SimpleNamespace(
        sequences=RallyLayerModels((object(),)),
        confidence=object(),
        preprocessing=BaseAnnotatorConfig(),
        side_geometry=SideGeometry.VIDEO,
    )
    load_calls = []

    def load_model(_path):
        load_calls.append(_path)
        return models

    monkeypatch.setattr(input_test, 'load_models', load_model)
    monkeypatch.setattr(input_test, 'fitted_trees', lambda _models: ())
    test_inputs = tmp_path / 'test-inputs'
    test_inputs.mkdir()
    test_manifest = test_inputs / 'manifest.json.gz'
    write_json(test_manifest, {'items': [{'item_id': 'synthetic'}]})
    monkeypatch.setattr(input_test, 'load_paths', lambda _path: {
        'test_inputs': test_inputs,
        'inpaint_root': tmp_path / 'inpaint',
        'pose_root': tmp_path / 'pose',
        'test_courts': tmp_path / 'courts',
    })
    monkeypatch.setattr(input_test, 'load_test_videos', lambda _path: videos)
    monkeypatch.setattr(input_test, 'input_paths', lambda *_args, **_kwargs: {
        'test_inputs/manifest.json.gz': test_manifest,
    })
    monkeypatch.setattr(input_test, 'source_record', lambda: {'revision': 'synthetic', 'dirty': False})
    monkeypatch.setattr(input_test, 'load_test_inputs', lambda video, *_args: SimpleNamespace(
        identity=video.identity,
        track=object(),
        pose=SimpleNamespace(bboxes=None, scores=None, kps=None, ndet=None),
    ))
    run_calls = []

    def run_mock(*_args, **kwargs):
        identity = kwargs['video_id']
        run_calls.append(identity)
        if identity == fail_video:
            raise RuntimeError('synthetic inference failure')
        kwargs['capture'].hybrid = SimpleNamespace(
            refined=SimpleNamespace(sequences=(), events=()),
        )

    monkeypatch.setattr(input_test, 'run_video_kwargs', lambda inputs: {'video_id': inputs.identity})
    monkeypatch.setattr(input_test, 'run_video', run_mock)
    monkeypatch.setattr(input_test, 'cache_sticky_result', lambda *_args: nullcontext())
    monkeypatch.setattr(input_test, 'stream_payload', lambda *_args: {'events': [], 'sequences': []})

    def score_stream(_video: Any, _stream: dict) -> dict:
        return {'5': {}, '10': {}}

    def review_rows(
        _video: Any, identity: str, _prediction: Any,
    ) -> tuple[dict, list[dict]]:
        rankings = {
            f'{ranking}-{tolerance}': {'unjudgeable': 1}
            for ranking in ('selector', 'fitted') for tolerance in (5, 10)
        }
        rows = [
            {'video': identity, 'span': 0, 'tolerance': tolerance, 'ranking': ranking, 'correct': -1, 'score': 0.5}
            for ranking in ('selector', 'fitted') for tolerance in (5, 10)
        ]
        return rankings, rows

    monkeypatch.setattr(input_test, 'score_stream', score_stream)
    monkeypatch.setattr(input_test, 'pooled_scores', lambda _scores: {'5': {}, '10': {}})
    monkeypatch.setattr(input_stages, '_review_rows', review_rows)
    return run_calls, videos, load_calls


@pytest.mark.parametrize('mismatch', ('fit_policy', 'seed_offsets', 'bundle_hash', 'evaluation_source'))
def test_rejects_wrong_policy_or_bundle_before_test_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str,
) -> None:
    selection_path, document, _hashes = _write_selection(tmp_path)
    monkeypatch.setattr(input_test, 'source_record', lambda: {'revision': 'synthetic', 'dirty': False})
    if mismatch == 'fit_policy':
        document['fit_policy'] = 'final40'
    elif mismatch == 'seed_offsets':
        document['seed_offsets'] = [0, 100, 200, 300]
    elif mismatch == 'bundle_hash':
        document['finalists'][0]['seeds'][0]['bundle_md5'] = {
            name: '0' * input_test.MODEL_HASH_SIZE for name in document['finalists'][0]['seeds'][0]['bundle_md5']
        }
    else:
        document['evaluation_source'] = {'revision': 'older-evaluator'}
    write_json(selection_path, document)
    monkeypatch.setattr(input_test, 'load_test_videos', lambda *_args: pytest.fail('test manifest loaded too early'))

    with pytest.raises(ValueError):
        input_test.evaluate(selection_path, 'candidate-a', 0, tmp_path / 'paths.toml', tmp_path / 'output')


@pytest.mark.parametrize('dirty_source', ('current', 'selection'))
def test_dirty_source_is_rejected_before_test_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirty_source: str,
) -> None:
    selection_path, document, _hashes = _write_selection(tmp_path)
    monkeypatch.setattr(input_test, 'source_record', lambda: {
        'revision': 'synthetic', 'dirty': dirty_source == 'current',
    })
    if dirty_source == 'selection':
        document['evaluation_source']['dirty'] = True
        write_json(selection_path, document)
    monkeypatch.setattr(input_test, 'load_test_videos', lambda *_args: pytest.fail('test manifest loaded too early'))
    monkeypatch.setattr(input_test, 'load_models', lambda *_args: pytest.fail('test model loaded too early'))

    with pytest.raises(ValueError, match='clean'):
        input_test.evaluate(selection_path, 'candidate-a', 0, tmp_path / 'paths.toml', tmp_path / 'output')


def test_accepted_chosen_bundle_runs_46_videos_without_refitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_path, _document, _hashes = _write_selection(tmp_path)
    run_calls, videos, load_calls = _runtime_patches(monkeypatch, tmp_path)

    result = input_test.evaluate(selection_path, 'candidate-a', 0, tmp_path / 'paths.toml', tmp_path / 'output')

    rows = read_json(tmp_path / 'output' / 'review-rows.json.gz')
    assert result['complete'] is True
    assert set(result['per_video']) == {video.identity for video in videos}
    assert len(run_calls) == 46
    assert len(rows) == 46 * 4
    assert all(row['correct'] == -1 for row in rows)
    assert result['fit_policy'] == 'contact-fit-32-videos'
    assert result['seed_offset'] == 0
    assert len(load_calls) == 1

    resumed = input_test.evaluate(selection_path, 'candidate-a', 0, tmp_path / 'paths.toml', tmp_path / 'output')
    assert resumed['complete'] is True
    assert len(run_calls) == 46
    assert len(load_calls) == 2


def test_changed_same_stat_test_manifest_rejects_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_path, _document, _hashes = _write_selection(tmp_path)
    run_calls, _videos, _load_calls = _runtime_patches(monkeypatch, tmp_path)
    output = tmp_path / 'output'

    input_test.evaluate(selection_path, 'candidate-a', 0, tmp_path / 'paths.toml', output)
    manifest_path = tmp_path / 'test-inputs' / 'manifest.json.gz'
    saved_identity = read_json(output / 'settings.json.gz')
    assert saved_identity['test_manifest_md5'] == input_test._md5(manifest_path)
    prior_stat = manifest_path.stat()

    write_json(manifest_path, {'items': [{'item_id': 'synthetix'}]})
    changed_stat = manifest_path.stat()
    assert changed_stat.st_size == prior_stat.st_size
    os.utime(manifest_path, ns=(prior_stat.st_atime_ns, prior_stat.st_mtime_ns))
    assert manifest_path.stat().st_mtime_ns == prior_stat.st_mtime_ns

    with pytest.raises(ValueError, match='test inputs changed'):
        input_test.evaluate(selection_path, 'candidate-a', 0, tmp_path / 'paths.toml', output)
    assert len(run_calls) == 46


def test_failed_partial_run_is_never_marked_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_path, _document, _hashes = _write_selection(tmp_path)
    _runtime_patches(monkeypatch, tmp_path, fail_video='test-02')

    with pytest.raises(RuntimeError, match='synthetic inference failure'):
        input_test.evaluate(selection_path, 'candidate-a', 0, tmp_path / 'paths.toml', tmp_path / 'partial')

    result = read_json(tmp_path / 'partial' / 'result.json.gz')
    assert result['complete'] is False
