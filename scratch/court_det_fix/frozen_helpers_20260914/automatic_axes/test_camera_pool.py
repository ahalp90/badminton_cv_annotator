"""Check selection replay and evidence reuse across saved camera-pool diagnostics."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rescore_camera_pool as module


@pytest.mark.parametrize('keep_all', [False, True])
def test_saved_camera_and_evidence_are_reused(monkeypatch: pytest.MonkeyPatch, keep_all: bool) -> None:
    source = {'id': 'synthetic', 'dimensions': {'width': 100, 'height': 100}}
    shortlist = []
    for candidate_id, offset, score in [('bad', 40., .9), ('cached', 0., .8), ('near', 1., .75), ('new', 10., .7)]:
        corners = np.array([[0., 0.], [20., 0.], [20., 40.], [0., 40.]]) + offset
        shortlist.append({'candidate_id': candidate_id, 'corners_px': corners.tolist(), 'shortlist_score': score})

    def measured(entry: dict) -> dict:
        return {**entry, 'stripe': {'exclusive': {'score': entry['shortlist_score']}},
                'profile': {'score': 1.}, 'gates': {'camera_error': .05}}

    cached = [measured(shortlist[1]), measured(shortlist[3])]
    baseline = {'pairs': [{'shortlist': shortlist}], 'entries': cached, 'elapsed_s': 2.,
                'generation_elapsed_s': 100., 'working_size': [100, 100], 'keep_global': 256,
                'selection_stage': 'camera_before_global_cap', 'line_winner_id': 'cached',
                'paint_winner_id': 'cached',
                'camera_prefilter': [{'candidate_id': entry['candidate_id'],
                                      'camera_error': .2 if entry['candidate_id'] == 'bad' else .05}
                                     for entry in shortlist]}
    evaluated = []

    def evaluate(_source: dict, entries: list[dict], *_args) -> list[dict]:
        evaluated.append([entry['candidate_id'] for entry in entries])
        return [measured(entry) for entry in entries]

    monkeypatch.setattr(module, 'prepare', lambda source: (np.empty((0, 4)), (), (100, 100)))
    monkeypatch.setattr(module.assignment, 'prepare_observations', lambda segments, size: None)
    monkeypatch.setattr(module, 'evaluate_pool', evaluate)
    # No camera callable: every camera result must be reused from the saved pool.
    result = module.rescore(source, baseline, SimpleNamespace(), Path('.'), True, keep_all)
    expected = ['cached', 'near', 'new'] if keep_all else ['cached', 'new']
    assert [entry['candidate_id'] for entry in result['entries']] == expected
    assert evaluated == [['cached', 'new'], ['near'] if keep_all else []]
    assert result['entries'][0] is cached[0]
    assert result['camera_eligible_before_global'] == 3
    assert result['generation_elapsed_s'] == 100.
    assert result['line_winner_id'] == result['paint_winner_id'] == 'cached'
