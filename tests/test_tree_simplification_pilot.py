"""Smoke tests for the tree-simplification pilot evaluation boundary."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from annotator.config import BaseAnnotatorConfig
from annotator.models import SideGeometry
from experiments.annotator.heuristic_only_tune.data import COUNT_FIELDS
from experiments.annotator.heuristic_only_tune.run import read_json, write_json
from experiments.annotator.tree_simplification import pilot


def fitted_models(*, preprocessing: BaseAnnotatorConfig | None = None,
                  side_geometry: SideGeometry = SideGeometry.VIDEO) -> SimpleNamespace:
    return SimpleNamespace(
        preprocessing=BaseAnnotatorConfig() if preprocessing is None else preprocessing,
        side_geometry=side_geometry,
        contact_settings=object(),
        contact=object(),
        sequences=object(),
        confidence=object(),
    )


def serialised_preprocessing(config: BaseAnnotatorConfig) -> dict:
    return json.loads(json.dumps(asdict(config), default=sorted))


def write_preparation(root: Path, preprocessing: dict) -> None:
    write_json(root / 'preparation.json.gz', {'inputs': {'preprocessing': preprocessing}})


def test_evaluate_creates_stream_directory_before_writing_and_saves_eight_video_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_root = tmp_path / 'prepared'
    video_dir = prepared_root / 'videos'
    video_dir.mkdir(parents=True)
    output = tmp_path / 'evaluation'
    models = fitted_models()
    write_preparation(prepared_root, serialised_preprocessing(models.preprocessing))

    prepared_by_identity = {}
    for index in range(8):
        identity = f'validation-{index}'
        video_path = video_dir / f'{identity}.joblib'
        video_path.touch()
        write_json(video_path.with_suffix('.json.gz'), {'split': 'validation'})
        video = SimpleNamespace(identity=identity, fps=30.0, split='validation')
        prepared_by_identity[identity] = SimpleNamespace(video=video, evidence=SimpleNamespace(track=np.zeros((2, 3))))

    monkeypatch.setattr(pilot.joblib, 'load', lambda path: prepared_by_identity[Path(path).stem])
    monkeypatch.setattr(
        pilot, 'training_pool',
        lambda prepared, _contact, _settings: SimpleNamespace(
            initial=(prepared.video.identity,), events=(), pool=object(),
        ),
    )
    monkeypatch.setattr(
        pilot, 'finish_sequences',
        lambda initial, events, _pool, _models, *, fps, frame_count: SimpleNamespace(
            sequences=initial, events=events,
        ),
    )
    monkeypatch.setattr(
        pilot.retrain, 'stream_payload', lambda sequences, _events: {'identity': sequences[0]},
    )
    scored = []

    def score_stream(video: SimpleNamespace, stream: dict) -> dict:
        assert stream['identity'] == video.identity
        scored.append(video.identity)
        return {
            tolerance: {field: 1 for field in COUNT_FIELDS}
            for tolerance in ('5', '10')
        }

    monkeypatch.setattr(pilot, 'score_stream', score_stream)
    observed_writes = []
    real_write_json = pilot.write_json

    def record_write(path: Path, payload: object) -> None:
        observed_writes.append((path, (output / 'validation_streams').is_dir()))
        real_write_json(path, payload)

    monkeypatch.setattr(pilot, 'write_json', record_write)

    result = pilot.evaluate(prepared_root, models, output)

    assert len(scored) == 8
    assert len({path.name for path in (output / 'validation_streams').glob('*.json.gz')}) == 8
    assert observed_writes[0][1]
    assert result['totals']['10']['labelled_contacts'] == 8
    assert result['totals']['10']['fully_correct_rallies'] == 8
    assert read_json(output / 'validation.json.gz') == result


@pytest.mark.parametrize(
    ('prepared_preprocessing', 'side_geometry'),
    [
        ({'different': 'preprocessing'}, SideGeometry.VIDEO),
        (None, SideGeometry.SCENE),
    ],
)
def test_evaluate_rejects_preparation_or_side_geometry_mismatch(
    tmp_path: Path,
    prepared_preprocessing: dict | None,
    side_geometry: SideGeometry,
) -> None:
    prepared_root = tmp_path / 'prepared'
    prepared_root.mkdir()
    models = fitted_models(side_geometry=side_geometry)
    preprocessing = (
        serialised_preprocessing(models.preprocessing)
        if prepared_preprocessing is None else prepared_preprocessing
    )
    write_preparation(prepared_root, preprocessing)
    output = tmp_path / 'evaluation'

    with pytest.raises(ValueError, match='preprocessing or side geometry'):
        pilot.evaluate(prepared_root, models, output)

    assert not (output / 'validation_streams').exists()


def test_missing_reference_bundle_fails_before_fit_or_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / 'pilot-output'
    missing_bundle = tmp_path / 'missing-reference'
    fit = Mock(side_effect=AssertionError('fit must not start'))
    source_record = Mock(side_effect=AssertionError('pilot setup must not start'))
    monkeypatch.setattr(pilot, 'fit', fit)
    monkeypatch.setattr(pilot, 'source_record', source_record)
    monkeypatch.setattr(
        'sys.argv',
        [
            'pilot', '--prepared', str(tmp_path / 'prepared'), '--paths', str(tmp_path / 'paths.toml'),
            '--output', str(output), '--reference-bundle', str(missing_bundle),
        ],
    )

    with pytest.raises(SystemExit) as error:
        pilot.main()

    assert error.value.code == 2
    fit.assert_not_called()
    source_record.assert_not_called()
    assert not output.exists()
