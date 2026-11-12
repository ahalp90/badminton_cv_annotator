"""Contracts for raw preprocessing preparation, contact fits and replay export."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from annotator.config import BaseAnnotatorConfig
from annotator.training.contact import ContactTrainingVideo
from annotator.training.workflow import DEFAULT_TRAINING_SETTINGS
from annotator.types import SmoothingMode
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

        smoothing_mode=SmoothingMode.INCLUDE_INVISIBLE,
        overrides_base30={'smooth_window': 9},
        require_fast_burst=False,
        reentry_guard_buffer=0.1,
        rejected_grades=frozenset({1, 3}),
    )

    for config in (default, custom):
        record = prepare.preprocessing_record(config)
        assert 'thresholds' not in record
        assert record['overrides_base30'] == config.overrides_base30
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
    requested = replace(BaseAnnotatorConfig(), max_high_shot_gap_frames=60.0)

    with pytest.raises(ValueError, match='cached preprocessing differs'):
        prepare.prepare_video(SimpleNamespace(identity=identity), 'A', {}, tmp_path, requested)


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
    monkeypatch.setattr(input_stages.experiment, 'contact_fit_videos', lambda _: fit_orders)
    monkeypatch.setattr(input_stages.experiment, 'progress', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        input_stages.experiment, 'settings_for',
        lambda *_args, **_kwargs: replace(DEFAULT_TRAINING_SETTINGS, preprocessing=BaseAnnotatorConfig()),
    )
    calls = []

    def fit_contact(videos, _config, *, fit_video_order):
        calls.append((tuple(video.video_id for video in videos), tuple(fit_video_order)))
        return SimpleNamespace(
            model=object(),
            selection=SimpleNamespace(labels=np.asarray([1, 0], dtype=np.int8)),
        )

    monkeypatch.setattr(input_stages, 'fit_contact_model', fit_contact)
    monkeypatch.setattr(input_stages.experiment, 'dump_atomically', lambda _model, path: path.write_bytes(b'model'))

    input_stages.fit_contacts(tmp_path, tmp_path / 'contacts-run', threads=1)

    fit_names = {path.stem for path in (tmp_path / 'contacts-run' / 'contacts').glob('*.joblib')}
    assert fit_names == {'held_A', 'held_B', 'held_C', 'held_D', 'v_32'}
    assert len(calls) == 5
    assert calls[0] == (fit_orders['held_A'], tuple(sorted(fit_orders['held_A'])))
    assert read_json(tmp_path / 'contacts-run' / 'settings.json.gz')['settings']['preprocessing'] == preproc


def test_preprocessing_record_rejects_the_retired_threshold_preset() -> None:
    record = prepare.preprocessing_record(BaseAnnotatorConfig())
    record['thresholds'] = []
    with pytest.raises(ValueError, match='Saved preprocessing fields differ'):
        prepare.preprocessing_from_record(record)
