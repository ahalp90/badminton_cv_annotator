"""Joint-grid scheduling, cutoff reuse and fitting boundaries."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from annotator.contacts.score_candidates import (
    CONTACT_FEATURE_NAMES,
    ContactModelConfig,
    score_contact_features,
)
from experiments.annotator.tree_simplification import hp_search as search
from experiments.annotator.tree_simplification import hp_search_workers as workers
from tests.test_tree_simplification_training import training_pools


def result(candidate, rallies, f1):
    scores = {'fully_correct_rallies': rallies, 'attributed_f1': f1}
    return {'candidate': asdict(candidate), 'label': candidate.label, 'totals': {'5': scores, '10': scores}}


def test_grid_and_shortlist_keep_both_objectives_and_baseline():
    candidates = search.grid()
    assert len(candidates) == len({candidate.label for candidate in candidates}) == 192
    assert candidates[0] == search.BASELINE
    assert len({candidate.contact_label for candidate in candidates}) == 8
    assert len({candidate.family_label for candidate in candidates}) == 32
    assert len(search.grid((400,))) == 96
    rows = [result(candidate, index, 1 - index / 192) for index, candidate in enumerate(candidates)]
    selected = search.shortlist(rows)
    assert {candidate.label for candidate in selected} == {
        candidates[index].label for index in (0, 1, 190, 191)
    }


def test_circular_slots_stay_owned_until_family_finishes():
    queue = search.CircularQueue(3)
    stopped = threading.Event()
    deadline = time.time() + 10
    for index in range(3):
        slot = queue.reserve(deadline, stopped)
        queue.publish(slot, index)
    assert [queue.ready.get_nowait() for _ in range(3)] == [0, 1, 2]
    entered = threading.Event()
    def next_slot():
        entered.set()
        return queue.reserve(deadline, stopped)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(next_slot)
        assert entered.wait(1)
        assert not pending.done()
        assert queue.slots == [0, 1, 2]
        queue.release(0)
        assert pending.result(timeout=2) == 0
        queue.publish(0, 3)
        assert queue.slots == [3, 1, 2]


@pytest.mark.parametrize('fps', [25.0, 30.0, 60.0])
def test_cutoff_fanout_matches_independent_production_scoring(fps):
    rows = np.zeros(5, dtype=[('frame', 'i4'), ('interval_id', 'i4'),
                             *((name, 'f4') for name in CONTACT_FEATURE_NAMES)])
    rows['frame'] = [10, 16, 23, 50, 70]
    probabilities = np.asarray([.95, .9, .85, .8, .75], dtype=np.float32)
    calls = []
    def predict(matrix):
        calls.append(len(matrix))
        return np.column_stack((1 - probabilities, probabilities))
    model = SimpleNamespace(predict_proba=predict)
    cached = score_contact_features(rows, model, fps)
    reused = [workers.contacts_at_cutoff(cached, cutoff, fps) for cutoff in (.75, .8, .85, .9)]
    assert len(calls) == 1
    for cutoff, scored in zip((.75, .8, .85, .9), reused, strict=True):
        expected = score_contact_features(rows, model, fps, ContactModelConfig(cutoff))
        np.testing.assert_array_equal(scored.kept_indices, expected.kept_indices)
        assert scored.probabilities is cached.probabilities
        assert scored.probabilities.dtype == np.float64


def test_selector_worker_real_fit_resume_and_cpu_scoring(tmp_path: Path):
    pools = training_pools(videos_per_group=1)
    candidate = replace(search.BASELINE, selector_rounds=2)
    path = workers.fit_selector(pools, candidate, 0, tuple('ABC'), tmp_path, 'cpu', time.time() + 30)
    fitted = joblib.load(path)
    assert fitted.n_features_in_ == 2
    assert fitted.get_params()['device'] == 'cpu'
    assert fitted.get_params()['n_jobs'] == 1
    first = fitted.predict_proba(pools['train-D-0'].matrix)
    workers.fit_selector(pools, candidate, 0, tuple('ABC'), tmp_path, 'cpu', time.time() + 30)
    np.testing.assert_array_equal(joblib.load(path).predict_proba(pools['train-D-0'].matrix), first)


@pytest.mark.parametrize('validation,expected_groups', [(False, ['BCD', 'ACD', 'ABD', 'ABC']), (True, ['ABCD'])])
def test_screen_holds_groups_and_validation_fits_all_training(tmp_path, monkeypatch, validation, expected_groups):
    runner = object.__new__(search.Search)
    runner.root, runner.training_ids = tmp_path, ()
    runner.workers, runner.gpu_fits = 2, 2
    runner.device, runner.deadline = 'cpu', time.time() + 10
    runner.messages = search.Queue()
    runner.progress = search.Progress(tmp_path, time.time())
    runner.progress.update(fits_completed=0)
    runner.videos = tuple(SimpleNamespace(identity=group) for group in 'ABCDV')
    runner.development = SimpleNamespace(group_by_video={group: group for group in 'ABCDV'})
    fitted, scored = [], []
    def fit(_pools, _candidate, _seed, groups, *_args):
        fitted.append(''.join(groups))
        return Path('model')
    def score(_model, directory, videos, *_args):
        scored.extend(video.identity for video in videos)
        assert directory.name == ('validation' if validation else 'train')
        return {}
    monkeypatch.setattr(search, 'load_pools', lambda *_args: {})
    monkeypatch.setattr(search, 'fit_selector', fit)
    monkeypatch.setattr(search, 'score_fold', score)
    monkeypatch.setattr(runner, 'save_result', lambda *_args: {'complete': True})
    with ThreadPoolExecutor(2) as gpu, ThreadPoolExecutor(2) as cpu:
        runner.gpu, runner.cpu = gpu, cpu
        assert runner.run_family([search.BASELINE], 0, validation) == [{'complete': True}]
    assert sorted(fitted) == sorted(expected_groups)
    assert sorted(scored) == list('V' if validation else 'ABCD')


@pytest.mark.parametrize('error', [TimeoutError('deadline'), search.subprocess.TimeoutExpired('prepare', 1)])
def test_deadline_exit_reports_time_limit(tmp_path, monkeypatch, error):
    monkeypatch.setattr(search.sys, 'argv', ['hp_search', '--paths', str(tmp_path / 'paths.toml'),
                                          '--output', str(tmp_path), '--smoke'])
    monkeypatch.setattr(search, 'load_paths', lambda _path: {})
    def expired(*_args):
        raise error
    monkeypatch.setattr(search, 'run', expired)
    with pytest.raises(SystemExit) as stopped:
        search.main()
    assert stopped.value.code == 124
    assert search.read_json(tmp_path / 'progress.json.gz')['status'] == 'time_limit'
