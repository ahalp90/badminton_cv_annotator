"""A selected parent may have a failed child fit; replay must preserve that outcome."""
from types import SimpleNamespace

import numpy as np
import pytest

from court_detector import geometry, stripe_fitting, stripe_refit


@pytest.fixture
def failed_parent(monkeypatch):
    constraints = stripe_fitting.Constraints(
        points=np.empty((0, 2)), intervals=np.empty(0, dtype=int),
        positions=np.empty(0, dtype=int), weights=np.empty(0),
        fragment_ids=np.empty(0, dtype=int), sample_ids=np.empty(0, dtype=int),
    )
    parent = {
        'origin_key': 'parent', 'kind': 'parent', 'homography_working': np.eye(3).tolist(),
        'corners_px': geometry.CORNER_COURT_M.tolist(), 'gates': {},
        'evidence': {'stripe_assignments': {}},
    }
    attempt = {
        'origin_key': 'parent', 'status': 'insufficient_samples', 'successful': False,
        'attempted_corners_native': None, 'child_origin_key': None,
    }
    record = {'parents': [parent], 'valid_children': [], 'fit_attempts': [attempt]}
    context = SimpleNamespace(native_size=(16, 16), size=(16, 16), weights=np.empty(0),
                              observations=SimpleNamespace(fragment_ids=np.empty(0, dtype=int)))
    verifier = SimpleNamespace(detector=geometry, jsonable=lambda value: value,
                               paint_geometry=SimpleNamespace(CENTRE_SEGMENTS_M=geometry.SEGMENTS_M))
    monkeypatch.setattr(stripe_fitting, 'prepare', lambda *args, **kwargs: constraints)
    monkeypatch.setattr(stripe_refit, 'colour_planes', lambda *args: (None, None, None))
    monkeypatch.setattr(stripe_refit, 'relabel', lambda *args: (None, [], {}))
    return record, context, verifier


@pytest.mark.parametrize('replay_check', [False, True])
def test_failed_parent_fit_is_rejected_normally_with_or_without_replay(failed_parent, replay_check) -> None:
    record, context, verifier = failed_parent
    result = stripe_refit.refit_chosen(
        record, 'parent', context, np.zeros((16, 16, 3), dtype=np.uint8), verifier, {}, np.empty(0),
        replay_check=replay_check,
    )
    assert not result['corrected']['valid']
    assert result['corrected']['validity_reason'] == 'no_fit_corners'
    assert result['corrected']['status'] == 'insufficient_samples'
    if replay_check:
        assert result['original_replay']['status'] == 'matched'
        assert result['original_replay']['corners_native_px'] is None


@pytest.mark.parametrize('changed_field, value', [
    ('attempted_corners_native', [[0, 0], [1, 0], [1, 1], [0, 1]]),
    ('status', 'invalid_projection'),
    ('successful', True),
])
def test_failed_replay_still_rejects_a_different_saved_outcome(failed_parent, changed_field, value) -> None:
    record, context, verifier = failed_parent
    record['fit_attempts'][0][changed_field] = value
    with pytest.raises(ValueError, match='unlike its saved attempt'):
        stripe_refit.refit_chosen(
            record, 'parent', context, np.zeros((16, 16, 3), dtype=np.uint8), verifier, {}, np.empty(0),
        )
