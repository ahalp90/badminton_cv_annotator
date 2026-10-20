"""Development export preserves portable inference for older pilot bundles."""

from copy import deepcopy
from types import SimpleNamespace

import joblib
import numpy as np
from xgboost import XGBClassifier

from annotator.models import AnnotatorModels, fitted_trees, load_models, save_models
from annotator.sequence.features import ServeEditInputs
from annotator.sequence.rally_model import (
    RallyLayerModels,
    predict_rally_options,
    score_layer_confidence,
)
from annotator.sequence.refine import OptionPool, SequenceModels
from annotator.training.sequences import SequenceTrainingVideo
from experiments.annotator.tree_simplification import integrate
from experiments.annotator.tree_simplification.options import expand_pool


def test_export_resets_historical_pilot_device_before_replay(tmp_path, monkeypatch) -> None:
    matrix = np.array([[0.], [1.], [2.], [3.]])
    tree = XGBClassifier(n_estimators=2, n_jobs=1, device='cpu').fit(matrix, [0, 0, 1, 1])
    expected = tree.predict_proba(matrix)
    historical = deepcopy(tree).set_params(device='cuda')
    pilot = tmp_path / 'pilot'
    save_models(AnnotatorModels(historical, SequenceModels(*(historical,) * 6), historical), pilot / 'bundle')
    run, output = tmp_path / 'run', tmp_path / 'export'
    (run / 'layers-1').mkdir(parents=True)
    joblib.dump((tree,), run / 'layers-1/rally-models.joblib')
    integrate.write_json(run / 'result.json.gz', {'complete': True})
    integrate.write_json(run / 'settings.json.gz', {
        'layers': 1, 'legacy_pool': False, 'target_policy': {},
        'pool_settings': {
            'pilot': str(pilot), 'max_edits': 3, 'deletion_candidates': 6,
            'settings': {'confidence': {}},
        },
    })
    monkeypatch.setattr(integrate, 'fit_review', lambda *args: tree)
    replayed = []

    def validate(run, depth, settings, old, models, directory):
        for bundle in (old, models):
            for fitted in fitted_trees(bundle):
                assert fitted.get_params()['device'] == 'cpu'
                np.testing.assert_array_equal(fitted.predict_proba(matrix), expected)
        replayed.append(models)
        return {'totals': {}, 'review': {}}

    monkeypatch.setattr(integrate, 'validate', validate)
    monkeypatch.setattr(integrate.sys, 'argv', [
        'integrate', '--run', str(run), '--layers', '1',
        '--five-frame-targets', str(tmp_path / 'targets'), '--output', str(output),
    ])
    integrate.main()
    assert len(replayed) == 1
    restored = load_models(output / 'bundle')
    np.testing.assert_array_equal(restored.contact.predict_proba(matrix), expected)
    assert integrate.read_json(output / 'result.json.gz')['complete'] is True


def test_validation_creates_and_saves_streams_in_fresh_export(tmp_path, monkeypatch) -> None:
    prepared, pools, pilot, run, output = (tmp_path / name for name in ('prepared', 'pools', 'pilot', 'run', 'output'))
    for directory in (prepared / 'videos', pools / 'videos', pilot / 'validation_streams',
                      run / 'layers-1/streams', output):
        directory.mkdir(parents=True)
    legacy = OptionPool((), {}, ServeEditInputs({}, np.empty((0, 0))),
                        np.empty((0, 0)), np.empty((0, 0)), {})
    videos = {}
    for index in range(8):
        identity = f'validation-{index}'
        video = SequenceTrainingVideo(identity, 'V', 30.0, (), (), legacy, (), 100)
        videos[identity] = SimpleNamespace(video=video, evidence=None, features=None)
        (prepared / 'videos' / f'{identity}.joblib').touch()
        integrate.write_json(prepared / 'videos' / f'{identity}.json.gz', {'split': 'validation'})
        pool = expand_pool(video)
        joblib.dump(pool, pools / 'videos' / f'{identity}.joblib')
        stream = integrate.stream_payload((), ())
        integrate.write_json(run / 'layers-1/streams' / f'{identity}.json.gz', stream)
        integrate.write_json(pilot / 'validation_streams' / f'{identity}.json.gz', stream)
    layers = RallyLayerModels((None,))
    prediction = predict_rally_options(layers, pool.runtime)
    hybrid = SimpleNamespace(refined=prediction, confidence=score_layer_confidence(prediction, None))
    monkeypatch.setattr(integrate, 'load_prepared', lambda root, identity: videos[identity])
    monkeypatch.setattr(integrate, 'predict_contact_features', lambda *args: hybrid)
    monkeypatch.setattr(integrate, 'score_stream', lambda *args: {})
    monkeypatch.setattr(integrate, 'pooled_scores', lambda rows: {})
    settings = {'pools': str(pools), 'pool_settings': {'prepared': str(prepared), 'pilot': str(pilot)}}
    old = AnnotatorModels(None, SequenceModels(*(None,) * 6), None)
    models = AnnotatorModels(None, layers, None)
    result = integrate.validate(run, 1, settings, old, models, output)
    assert len(result['per_video']) == 8
    assert len(list((output / 'streams').glob('*.json.gz'))) == 8
    assert integrate.read_json(output / 'streams/validation-0.json.gz') == stream
    assert integrate.read_json(output / 'review-rows.json.gz') == []
