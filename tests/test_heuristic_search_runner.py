"""Check rolling scheduling, saved failures, progress and resume identities."""

from __future__ import annotations

import argparse
import json
from collections import deque
from concurrent.futures import Executor, Future
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import pytest

from annotator.outcomes.point_winner import Half
from annotator.training.sequences import LabelledRally
from experiments.annotator.heuristic_only_tune import run as search
from experiments.annotator.heuristic_tuning.sweep import CandidateSpec


def _video(identity: str = 'example') -> search.SearchVideo:
    rally = LabelledRally('set_1/rally_1', (10, 20), (Half.TOP, Half.BOT))
    return search.SearchVideo(identity, 'ShuttleSet', 'train', 30.0, (1280.0, 720.0), (rally,))


def _spec(label: str, radius: int) -> CandidateSpec:
    return CandidateSpec(label, {'contact_dedup_radius_frames': radius}, {})


class RecordingExecutor:
    def __init__(self) -> None:
        self.submitted: list[tuple[Future[bool], search.SearchJob]] = []

    def submit(self, function: Any, job: search.SearchJob) -> Future[bool]:
        assert function is search.run_one_job
        future: Future[bool] = Future()
        self.submitted.append((future, job))
        return future


def test_coordinator_lock_rejects_duplicate_and_allows_resume_after_release(tmp_path: Path) -> None:
    output = tmp_path / 'run'
    with (
        search.coordinator_lock(output),
        pytest.raises(RuntimeError, match='another search coordinator is already running'),
        search.coordinator_lock(output),
    ):
        pytest.fail('second coordinator acquired the same output')
    assert (output / '.coordinator.lock').exists()
    with search.coordinator_lock(output):
        pass


def test_coordinator_lock_releases_after_run_failure(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='run failed'), search.coordinator_lock(tmp_path):
        raise ValueError('run failed')
    with search.coordinator_lock(tmp_path):
        pass


def test_worker_target_changes_drain_then_refill_without_batch_barriers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    videos = tuple(_video(f'video_{index}') for index in range(3))
    specs = [_spec(f'settings_{index}', index + 1) for index in range(3)]
    jobs = [search.SearchJob(video, spec) for spec in specs for video in videos]
    plan = search.SearchPlan(deque(jobs))
    workers = tmp_path / 'workers'
    workers.write_text('3\n')
    pool = RecordingExecutor()
    observed = []

    def finish(active: Any, *, timeout: float, return_when: Any) -> tuple[set[Future[bool]], set[Future[bool]]]:
        futures = list(active)
        observed.append((len(futures), len(pool.submitted)))
        step = len(observed) - 1
        if step == 0:
            workers.write_text('1\n')
        elif step == 2:
            workers.write_text('4\n')
            return set(), set(futures)
        elif step == 3:
            # The third video of settings_0 still runs while settings_1 starts.
            assert not pool.submitted[2][0].done()
            assert any(job.spec == specs[1] for _future, job in pool.submitted[3:])
        completed = futures[:1] if step < 3 else futures
        for future in completed:
            future.set_result(False)
        return set(completed), set(futures) - set(completed)

    monkeypatch.setattr(search, 'wait', finish)
    execution = search.schedule_jobs(cast(Executor, pool), plan, tmp_path, 3, workers)

    assert observed[:4] == [(3, 3), (2, 3), (1, 3), (4, 6)]
    assert [job for _future, job in pool.submitted] == jobs
    assert len({job.identity for _future, job in pool.submitted}) == len(jobs)
    assert [row['workers'] for row in execution['worker_targets']] == [3, 1, 4]
    assert execution['peak_active_jobs'] == 4
    assert plan.completed == 9
    status = search.read_json(tmp_path / 'status.json.gz')
    assert status['state'] == 'summarising'
    assert status['completed_jobs'] == status['total_jobs'] == 9
    assert status['active_jobs'] == status['queued_jobs'] == 0


def test_zero_workers_pauses_dispatch_until_control_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = search.SearchJob(_video(), _spec('settings', 3))
    plan = search.SearchPlan(deque([job]))
    workers = tmp_path / 'workers'
    workers.write_text('0')
    pool = RecordingExecutor()

    def resume(seconds: float) -> None:
        assert seconds == search.STATUS_INTERVAL_SECONDS
        assert not pool.submitted
        status = search.read_json(tmp_path / 'status.json.gz')
        assert status['state'] == 'paused'
        assert status['queued_jobs'] == 1
        workers.write_text('2')

    def finish(active: Any, **kwargs: Any) -> tuple[set[Future[bool]], set[Future[bool]]]:
        for future in active:
            future.set_result(False)
        return set(active), set()

    monkeypatch.setattr(search.time, 'sleep', resume)
    monkeypatch.setattr(search, 'wait', finish)
    search.schedule_jobs(cast(Executor, pool), plan, tmp_path, 1, workers)
    assert len(pool.submitted) == plan.completed == 1


@pytest.mark.parametrize('target', ['-1', '27', 'two', ''])
def test_invalid_worker_target_fails_loudly(tmp_path: Path, target: str) -> None:
    control = tmp_path / 'workers'
    control.write_text(target)
    with pytest.raises(ValueError):
        search.worker_target(2, control)


def test_failed_worker_stops_dispatch_and_saves_the_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = [search.SearchJob(_video(f'video_{index}'), _spec('settings', 3)) for index in range(2)]
    plan = search.SearchPlan(deque(jobs))
    pool = RecordingExecutor()

    def fail(active: Any, **kwargs: Any) -> tuple[set[Future[bool]], set[Future[bool]]]:
        future = next(iter(active))
        future.set_exception(ValueError('broken source inputs'))
        return {future}, set()

    monkeypatch.setattr(search, 'wait', fail)
    with pytest.raises(ValueError, match='broken source inputs'):
        search.schedule_jobs(cast(Executor, pool), plan, tmp_path, 1)
    assert len(pool.submitted) == 1
    assert plan.completed == 0
    status = search.read_json(tmp_path / 'status.json.gz')
    assert status['state'] == 'failed'
    assert status['error'] == {'type': 'ValueError', 'message': 'broken source inputs'}


def test_failed_inference_keeps_labels_in_summary_and_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = _video()
    spec = _spec('settings', 3)
    inputs = object()
    loads = []

    def load(*args: Any) -> Any:
        loads.append(args)
        return inputs

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError('inference failed')

    monkeypatch.setattr(search, 'load_video_inputs', load)
    monkeypatch.setattr(search, 'run_configuration', fail)
    monkeypatch.setattr(search, 'source_record', lambda: {'revision': 'worker-start'})
    search.initialise_worker({}, tmp_path, False)
    monkeypatch.setattr(search, 'source_record', lambda: {'revision': 'later-commit'})
    assert search.run_one_job(search.SearchJob(video, spec))
    other = _spec('other settings', 4)
    assert search.run_one_job(search.SearchJob(video, other))
    assert len(loads) == 1

    row = search.summarise(tmp_path, (video,), [spec])[0]
    assert row['failed_videos'] == ['example']
    for counts in row['by_tolerance'].values():
        assert counts['labelled_rallies'] == 1
        assert counts['labelled_contacts'] == 2
        assert counts['predicted_contacts'] == counts['correct_side_contacts'] == 0
        assert counts['attributed_f1'] == counts['fully_correct_rallies_percent'] == 0.0
    attempt = search.latest_attempt(tmp_path, video.identity, other)
    assert attempt is not None
    record = attempt['fixtures'][video.identity]
    assert record['source'] == {'revision': 'worker-start'}
    assert record['input_cache_hit']
    assert record['failure']['type'] == 'RuntimeError'
    assert record['inference_seconds'] >= 0
    assert record['score_seconds'] >= 0

    plan = search.pending_jobs((video, _video('new_video')), [spec], tmp_path)
    assert plan.completed == plan.failed == 1
    assert [job.video.identity for job in plan.jobs] == ['new_video']


@pytest.mark.parametrize('stage', ['load', 'score'])
def test_input_or_scorer_failure_propagates_without_a_completed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    job = search.SearchJob(_video(), _spec('settings', 3))

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError(f'{stage} failed')

    monkeypatch.setattr(search, 'load_video_inputs', fail if stage == 'load' else lambda *_args: object())
    monkeypatch.setattr(search, 'run_configuration', lambda *_args, **_kwargs: search.EMPTY_STREAM)
    if stage == 'score':
        monkeypatch.setattr(search, 'score_stream', fail)
    search.initialise_worker({}, tmp_path, False)
    with pytest.raises(ValueError, match=f'{stage} failed'):
        search.run_one_job(job)
    assert search.latest_attempt(tmp_path, job.video.identity, job.spec) is None


def test_cache_failure_propagates_without_becoming_an_empty_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = search.SearchJob(_video(), _spec('settings', 3))

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise search.StickyCacheError('cache read failed')

    monkeypatch.setattr(search, 'load_video_inputs', lambda *_args: object())
    monkeypatch.setattr(search, 'run_configuration', fail)
    monkeypatch.setattr(search, 'source_record', lambda: {'revision': 'worker-start'})
    search.initialise_worker({}, tmp_path, False)
    monkeypatch.setattr(search, 'source_record', lambda: {'revision': 'later-commit'})
    with pytest.raises(search.StickyCacheError, match='cache read failed'):
        search.run_one_job(job)
    assert search.latest_attempt(tmp_path, job.video.identity, job.spec) is None


def test_resume_allows_source_edits_but_preserves_inputs_and_settings(tmp_path: Path) -> None:
    array = tmp_path / 'track.npy.xz'
    array.write_bytes(b'first input')
    source = {
        'revision': 'first', 'python': '3.12', 'packages': {'numpy': '2.0'},
        'files_md5': {'src/model.py': 'old', 'pyproject.toml': 'old', 'uv.lock': 'old', 'groups.json.gz': 'fixed'},
    }
    identity = {'source': source, 'input_files': search.file_manifest({'track': array}), 'candidates': ['fixed']}
    run_path = tmp_path / 'run.json.gz'
    search.keep_run_identity(run_path, identity)
    current_source = {**source, 'revision': 'second', 'dirty': True, 'files_md5': {'groups.json.gz': 'fixed'}}
    search.keep_run_identity(run_path, {**identity, 'source': current_source})
    assert search.read_json(run_path) == identity
    assert source['files_md5']['src/model.py'] == 'old'
    for changed in (
        {**identity, 'candidates': ['changed']},
        {**identity, 'source': {**current_source, 'packages': {'numpy': '2.1'}}},
        {**identity, 'source': {**current_source, 'files_md5': {'groups.json.gz': 'changed'}}},
    ):
        with pytest.raises(ValueError, match='changed'):
            search.keep_run_identity(run_path, changed)
    array.write_bytes(b'changed input content')
    with pytest.raises(ValueError, match='changed'):
        search.keep_run_identity(run_path, {**identity, 'input_files': search.file_manifest({'track': array})})


def test_progress_snapshot_keeps_the_latest_complete_counter(tmp_path: Path) -> None:
    progress = tmp_path / 'progress.jsonl'
    event = {'activity': 'resolving outcomes', 'completed': 3, 'total': 10, 'unit': 'rallies', 'worker': 'main'}
    progress.write_bytes(json.dumps(event).encode() + b'\n' + b'{"activity": "partial')
    assert search.read_progress(progress) == event
    assert search.read_progress(tmp_path / 'missing') is None


def test_active_status_includes_real_subjob_progress(tmp_path: Path) -> None:
    job = search.SearchJob(_video(), _spec('settings', 3))
    path = search.progress_path(tmp_path, job)
    path.parent.mkdir()
    event = {'activity': 'resolving outcomes', 'completed': 3, 'total': 10, 'unit': 'rallies', 'worker': 'main'}
    path.write_text(json.dumps(event) + '\n')
    future: Future[bool] = Future()
    started = search.time.monotonic()
    search.publish_status(
        tmp_path, search.SearchPlan(deque()), {future: search.ActiveJob(job, started)},
        total=1, target=1, started=started, state='running',
    )
    active = search.read_json(tmp_path / 'status.json.gz')['active'][0]
    assert active['stage'] == 'resolving outcomes'
    assert active['completed'] == 3
    assert active['total'] == 10
    assert active['progress'] == event


def test_coordinator_starts_before_worker_initialisation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(search, '_WORKER', raising=False)
    monkeypatch.setattr(search, 'load_paths', lambda _path: {'shots_master': tmp_path})
    monkeypatch.setattr(search, 'load_candidates', lambda _path: ({'stage': 'grid'}, [_spec('shipped', 3)]))
    monkeypatch.setattr(search, 'load_original_videos', lambda _path: (_video(),))
    monkeypatch.setattr(search, 'input_paths', lambda *_args: {})
    monkeypatch.setattr(search, 'source_record', lambda: {'revision': 'parent-start'})
    monkeypatch.setattr(search, 'ProcessPoolExecutor', lambda **_kwargs: nullcontext(object()))
    monkeypatch.setattr(search, 'schedule_jobs', lambda *_args: {'worker_targets': []})
    monkeypatch.setattr(search, 'summarise', lambda *_args: [])
    args = argparse.Namespace(paths=tmp_path, candidates=tmp_path, split='train', only=None,
                              full=False, output=tmp_path, workers=1, workers_file=None)
    search.run_locked(args)
    assert search.read_json(tmp_path / 'run.json.gz')['source'] == {'revision': 'parent-start'}
    assert not hasattr(search, '_WORKER')
