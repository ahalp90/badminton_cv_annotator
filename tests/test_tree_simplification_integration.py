"""Development export preserves portable inference for older pilot bundles."""

from copy import deepcopy

import joblib
import numpy as np
from xgboost import XGBClassifier

from annotator.models import AnnotatorModels, fitted_trees, load_models, save_models
from annotator.sequence.refine import SequenceModels
from experiments.annotator.tree_simplification import integrate


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
