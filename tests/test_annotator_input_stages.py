"""Contracts for raw preprocessing preparation, contact fits and replay export."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from annotator.config import BaseAnnotatorConfig
from annotator.sequence.rally_model import (
    LayerConfidence,
    LayerReviewFeatures,
    RallyLayerPrediction,
)
from annotator.training.contact import ContactTrainingVideo
from annotator.training.workflow import DEFAULT_TRAINING_SETTINGS
from annotator.types import DeadMaskMode, ReentryGuardVariant, SmoothingMode, SpanOpen
from experiments.annotator.heuristic_only_tune.run import read_json, write_json
from experiments.annotator.tree_simplification import input_stages, prepare


def test_candidate_gzip_restores_existing_candidate_spec(tmp_path: Path) -> None:
    document = {
        'schema': 'input-candidates/1',
        'stage': 'base30',
        'candidates': [{'label': 'baseline', 'overrides_base30': {}, 'strategies': {}}],
    }
    gzip_path = tmp_path / 'candidate.json.gz'
    write_json(gzip_path, document)

    gzip_config, gzip_document = prepare.candidate_preprocessing(gzip_path)

    assert gzip_document == document
    assert gzip_config == BaseAnnotatorConfig()


def test_preprocessing_record_round_trips_default_and_nondefault_settings() -> None:
    default = BaseAnnotatorConfig()
    custom = replace(
        default,
        thresholds=default.thresholds._replace(smooth_window=9),
        dead_mask_mode=DeadMaskMode.COMPOSITION,
        smoothing_mode=SmoothingMode.ZERO_FILL,
        overrides_base30={'smooth_window': 9},
        span_open=SpanOpen.REGION_START,
        reentry_guard_variant=ReentryGuardVariant.REENTRY_ONLY,
        reentry_guard_buffer=0.1,
        quiet_start_window=0.25,
        rejected_grades=frozenset({1, 3}),
    )

    for config in (default, custom):
        record = prepare.preprocessing_record(config)
        assert isinstance(record['thresholds'], list)
        assert prepare.preprocessing_from_record(record) == config


def test_candidate_requires_one_existing_candidate_spec(tmp_path: Path) -> None:
    path = tmp_path / 'two.json.gz'
    row = {'label': 'baseline', 'overrides_base30': {}, 'strategies': {}}
    other = {'label': 'candidate', 'overrides_base30': {'smooth_window': 5}, 'strategies': {}}
    write_json(path, {'schema': 'input-candidates/1', 'stage': 'base30', 'candidates': [row, other]})

    with pytest.raises(ValueError, match='exactly one candidate'):
        prepare.candidate_preprocessing(path)


def test_prepare_resume_rejects_preprocessing_identity_mismatch(tmp_path: Path) -> None:
    identity = 'sset_01'
    videos = tmp_path / 'videos'
    videos.mkdir()
    (videos / f'{identity}.joblib').touch()
    write_json(videos / f'{identity}.json.gz', {
        'video': identity,
        'preprocessing': prepare.preprocessing_record(BaseAnnotatorConfig()),
    })
    requested = replace(BaseAnnotatorConfig(), gap_state_demotion_bound=60.0)

    with pytest.raises(ValueError, match='cached preprocessing differs'):
        prepare.prepare_video(SimpleNamespace(identity=identity), 'A', {}, tmp_path, requested)


def test_prepared_source_must_match_before_contact_fitting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_json(tmp_path / 'preparation.json.gz', {'source': {'revision': 'old'}, 'inputs': {}})
    monkeypatch.setattr(input_stages, 'source_record', lambda: {'revision': 'new'})

    with pytest.raises(ValueError, match='Prepared source differs'):
        input_stages.fit_contacts(tmp_path, tmp_path / 'fit')


@pytest.mark.parametrize('changed_stage', ['contact', 'selector'])
def test_export_rejects_mixed_stage_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_stage: str,
) -> None:
    prepared = tmp_path / 'prepared'
    contact = tmp_path / 'contact'
    run = tmp_path / 'selector'
    for path in (prepared, contact, run):
        path.mkdir()
    source = {'revision': 'current'}
    inputs = {
        'groups': {f'video-{index}': 'V' if index < 8 else 'A' for index in range(40)},
        'preprocessing': prepare.preprocessing_record(BaseAnnotatorConfig()),
    }
    pool_settings = {
        'prepared': str(prepared), 'contact_run': str(contact),
        'settings': {'preprocessing': inputs['preprocessing']},
    }
    write_json(prepared / 'preparation.json.gz', {'source': source, 'inputs': inputs})
    write_json(contact / 'settings.json.gz', {'prepared': str(prepared), 'settings': pool_settings['settings']})
    write_json(contact / 'contact-stage.json.gz', {
        'source': {'revision': 'old'} if changed_stage == 'contact' else source,
    })
    write_json(run / 'settings.json.gz', {'layers': 1, 'target_policy': {}, 'pool_settings': pool_settings})
    write_json(run / 'result.json.gz', {
        'complete': True, 'source': {'revision': 'old'} if changed_stage == 'selector' else source,
    })
    monkeypatch.setattr(input_stages, 'source_record', lambda: source)

    with pytest.raises(ValueError, match='sources differ'):
        input_stages._export_inputs(run, tmp_path / 'targets')


def test_contact_stage_fits_only_the_five_required_orders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Development:
        def __init__(self) -> None:
            self.group_by_video = {
                'a1': 'A', 'a2': 'A', 'b1': 'B', 'b2': 'B',
                'c1': 'C', 'c2': 'C', 'd1': 'D', 'd2': 'D',
            }

        def videos(self, groups: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(identity for identity, group in self.group_by_video.items() if group in groups)

        @staticmethod
        def id_order(identities: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(sorted(identities))

    development = Development()
    training = {
        identity: ContactTrainingVideo(identity, np.zeros((1, 1)), np.empty(0, dtype=np.int32), 30.0)
        for identity in development.group_by_video
    }
    fit_orders = {
        'final_40': tuple(reversed(tuple(training))),
        'held_A': ('d2', 'b1', 'c1', 'b2', 'd1', 'c2'),
        'held_B': ('d2', 'a1', 'c1', 'a2', 'd1', 'c2'),
        'held_C': ('d2', 'a1', 'b1', 'a2', 'd1', 'b2'),
        'held_D': ('c2', 'a1', 'b1', 'a2', 'c1', 'b2'),
        'v_32': tuple(reversed(tuple(training))),
    }
    preproc = prepare.preprocessing_record(BaseAnnotatorConfig())
    inputs = {
        'paths': {'shots_master': str(tmp_path / 'shots.csv')},
        'groups': {f'v{index}': 'A' for index in range(40)},
        'preprocessing': preproc,
    }
    monkeypatch.setattr(input_stages, '_prepared_inputs', lambda _: inputs)
    monkeypatch.setattr(input_stages, 'load_original_videos', lambda _: [])
    monkeypatch.setattr(input_stages, 'development_for', lambda _: development)
    monkeypatch.setattr(input_stages, '_training_records', lambda *_: training)
    monkeypatch.setattr(input_stages.retrain, 'contact_fit_videos', lambda _: fit_orders)
    monkeypatch.setattr(input_stages, 'source_record', lambda: {'revision': 'test'})
    monkeypatch.setattr(input_stages.pilot, 'progress', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        input_stages.pilot, 'settings_for',
        lambda *_args, **_kwargs: replace(DEFAULT_TRAINING_SETTINGS, preprocessing=BaseAnnotatorConfig()),
    )
    calls = []

    def fit_contact(videos, _config, *, fit_video_order):
        calls.append((tuple(video.identity for video in videos), tuple(fit_video_order)))
        return SimpleNamespace(
            model=object(),
            selection=SimpleNamespace(labels=np.asarray([1, 0], dtype=np.int8)),
        )

    monkeypatch.setattr(input_stages, 'fit_contact_model', fit_contact)
    monkeypatch.setattr(input_stages.retrain, 'dump_atomically', lambda _model, path: path.write_bytes(b'model'))

    input_stages.fit_contacts(tmp_path, tmp_path / 'contacts-run', threads=1)

    fit_names = {path.stem for path in (tmp_path / 'contacts-run' / 'contacts').glob('*.joblib')}
    assert fit_names == {'held_A', 'held_B', 'held_C', 'held_D', 'v_32'}
    assert len(calls) == 5
    assert calls[0] == (fit_orders['held_A'], tuple(sorted(fit_orders['held_A'])))
    assert read_json(tmp_path / 'contacts-run' / 'settings.json.gz')['settings']['preprocessing'] == preproc


def test_raw_export_replays_all_validation_streams_and_keeps_unknown_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / 'run'
    targets = tmp_path / 'targets'
    prepared_root = tmp_path / 'prepared'
    contact_run_root = tmp_path / 'contact-stage'
    pools_root = tmp_path / 'pools'
    identities = tuple(f'validation-{index}' for index in range(8))
    inputs = {'groups': {identity: 'V' for identity in identities},
              'preprocessing': prepare.preprocessing_record(BaseAnnotatorConfig())}
    pool_settings = {
        'prepared': str(prepared_root),
        'contact_run': str(contact_run_root),
        'max_edits': 3,
        'deletion_candidates': 6,
        'settings': {
            'preprocessing': inputs['preprocessing'],
            'contact_prediction': {},
        },
    }
    settings = {
        'layers': 1,
        'pools': str(pools_root),
        'pool_settings': pool_settings,
        'original_options': False,
        'target_policy': {'manifest': {'targets': 'fixed'}},
    }
    expected_stream = {'events': [], 'sequences': []}
    for identity in identities:
        path = run / 'layers-1' / 'streams' / f'{identity}.json.gz'
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, expected_stream)

    fake_videos = {
        identity: SimpleNamespace(
            video=SimpleNamespace(identity=identity, split='validation', fps=30.0, rallies=()),
            group='V', evidence=object(), features=object(),
        )
        for identity in identities
    }
    sequence = SimpleNamespace(span_id=4)
    layer_prediction = RallyLayerPrediction(
        sequences=(sequence,), events=(), pool=object(), scores=(np.asarray([0.8]),), chosen={4: 0},
    )
    confidence = LayerConfidence(LayerReviewFeatures((4,), np.zeros((1, 1))), np.asarray([0.7]))
    prediction = SimpleNamespace(refined=layer_prediction, confidence=confidence)
    monkeypatch.setattr(
        input_stages, '_export_inputs',
        lambda *_args: (settings, inputs, prepared_root, contact_run_root, 1),
    )
    monkeypatch.setattr(input_stages, 'source_record', lambda: {'revision': 'test'})
    monkeypatch.setattr(input_stages.integrate, 'fit_review', lambda *_args: object())
    monkeypatch.setattr(input_stages.integrate, 'verify_pool', lambda *_args: None)
    monkeypatch.setattr(input_stages, 'fitted_trees', lambda _models: ())
    monkeypatch.setattr(input_stages, 'predict_contact_features', lambda *_args: prediction)
    monkeypatch.setattr(input_stages, 'score_layer_confidence', lambda *_args: confidence)
    monkeypatch.setattr(input_stages, 'whole_rally_correctness', lambda *_args, **_kwargs: -1)
    monkeypatch.setattr(input_stages, 'score_stream', lambda *_args: {'5': {}, '10': {}})
    monkeypatch.setattr(input_stages, 'pooled_scores', lambda _scores: {})
    monkeypatch.setattr(input_stages.pilot, 'load_prepared', lambda _root, identity: fake_videos[identity])
    monkeypatch.setattr(input_stages.pilot, 'progress', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(input_stages, 'stream_payload', lambda *_args: expected_stream)
    saved_models = []
    monkeypatch.setattr(input_stages, 'save_models', lambda models, _path: saved_models.append(models))
    monkeypatch.setattr(input_stages, 'load_models', lambda _path: saved_models[-1])

    def load_artifact(path):
        if path.name == 'rally-models.joblib':
            return (object(),)
        return object()

    monkeypatch.setattr(joblib, 'load', load_artifact)

    result = input_stages.export_run(run, targets, tmp_path / 'export', threads=1)

    rows = read_json(tmp_path / 'export' / 'review-rows.json.gz')
    assert result['complete'] is True
    assert result['replay_streams_match'] is True
    assert set(result['per_video']) == set(identities)
    assert {row['ranking'] for row in rows} == {'selector', 'fitted'}
    assert len(rows) == 8 * 2 * 2
    assert all(row['correct'] == -1 for row in rows)

    monkeypatch.setattr(input_stages, 'stream_payload', lambda *_args: {'events': ['mismatch']})
    with pytest.raises(ValueError, match='differs from the saved rally run'):
        input_stages.export_run(run, targets, tmp_path / 'mismatched-export', threads=1)
