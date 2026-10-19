"""Saved-bundle orchestration and arithmetic without model loading or inference."""

from __future__ import annotations

import gzip
import json
import tomllib
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import pytest

from experiments.annotator.tree_hp_tune import test_rerun as rerun


def fake_scores(matched: int = 2) -> dict[str, Any]:
    by_video = {}
    for index, identity in enumerate(sorted(rerun.TEST_IDS)):
        counts = dict.fromkeys(rerun.grid.COUNT_FIELDS, 0)
        # Unequal denominators distinguish pooled F1 from mean video F1.
        if index < 2:
            counts.update(labelled_contacts=2 if index == 0 else 8,
                          predicted_contacts=2 if index == 0 else 8,
                          matched_contacts=matched if index == 0 else 0,
                          correct_side_contacts=matched // 2 if index == 0 else 0,
                          labelled_rallies=1 if index == 0 else 3,
                          fully_correct_rallies=1 if index == 0 else 0)
        by_video[identity] = {'10': counts.copy(), '5': {**counts, 'matched_contacts': 0}}
    return {'by_video': by_video, 'by_tolerance': rerun.grid.pooled_scores(by_video)}


def write_complete(directory: Path, matched: int = 2) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    scores = fake_scores(matched)
    rerun.grid.write_json(directory / 'scores.json.gz', scores)
    for identity in rerun.TEST_IDS:
        rerun.grid.write_json(directory / f'{identity}.json.gz', {'events': [], 'sequences': []})
    return scores


def bundle(tmp_path: Path, *, stage: str = 'v32', label: str = 'shipped', offset: int = 0) -> rerun.Bundle:
    return rerun.Bundle(stage, label, offset, tmp_path / 'bundle')


def test_committed_bundle_discovery_and_job_population() -> None:
    bundles = rerun.discover_bundles()
    assert len(bundles) == len({item.path for item in bundles}) == 20
    jobs = rerun.build_jobs(bundles)
    assert Counter(job.condition for job in jobs) == {'A': 10, 'B': 20, 'C': 20}
    assert all(job.bundle.stage == 'v32' for job in jobs if job.condition == 'A')
    for item in bundles:
        identity = rerun.CONFIGURATIONS[item.label]
        assert item.path.name == f'bundle_{item.stage}'
        if item.stage == 'v32':
            root = rerun.RESULTS / ('grid48' if item.seed_offset == 0 else f'repeats/seed-{item.seed_offset}')
            assert item.path == root / 'trials' / identity / 'bundle_v32'
        else:
            assert item.path == rerun.RESULTS / 'final_test/trials' / identity / f'seed-{item.seed_offset}/bundle_final40'


def test_missing_bundle_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        rerun.discover_bundles(tmp_path)


def test_complete_jobs_skipped_and_partial_jobs_restarted(tmp_path: Path) -> None:
    directory = tmp_path / 'job'
    assert rerun.prepare_job(directory)
    assert not directory.exists()  # The benchmark runner creates it itself.
    write_complete(directory)
    sentinel = directory / 'keep.txt'
    sentinel.write_text('keep')
    assert not rerun.prepare_job(directory)
    assert sentinel.exists()
    (directory / f'{next(iter(rerun.TEST_IDS))}.json.gz').unlink()
    assert rerun.prepare_job(directory)
    assert not directory.exists()


@pytest.mark.parametrize('damage', ['truncated', 'population', 'tolerance', 'failures', 'count'])
def test_incomplete_scores_are_restarted(tmp_path: Path, damage: str) -> None:
    scores = write_complete(tmp_path)
    if damage == 'truncated':
        (tmp_path / 'scores.json.gz').write_bytes(b'not gzip')
    else:
        if damage == 'population':
            scores['by_video'].pop(next(iter(rerun.TEST_IDS)))
        elif damage == 'tolerance':
            scores['by_tolerance'].pop('5')
        elif damage == 'failures':
            scores['failed_videos'] = ['8']
        else:
            scores['by_video']['8']['10'].pop('matched_contacts')
        rerun.grid.write_json(tmp_path / 'scores.json.gz', scores)
    assert not rerun.complete(tmp_path)


def test_generated_config_has_only_runner_paths_and_one_worker(tmp_path: Path) -> None:
    paths = {name: tmp_path / f'private "{name}"' for name in ('test_inputs', 'inpaint_root', 'pose_root')}
    job = rerun.Job('C', bundle(tmp_path))
    config = rerun.write_config(job, paths, tmp_path)
    assert config.is_relative_to(tmp_path / 'configs')
    document = tomllib.loads(config.read_text())
    assert document == {'paths': {'bundle': str(job.bundle.path), 'inputs': str(paths['test_inputs']),
                                  'inpaint_root': str(paths['inpaint_root']), 'pose_root': str(paths['pose_root'])},
                        'execution': {'annotator_workers': 1}}


def test_benchmark_commands_use_unchanged_clis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = rerun.Job('B', bundle(tmp_path))
    calls = []

    def run(command: list[str], **kwargs: Any) -> None:
        calls.append(command)
        assert kwargs['check'] is True
        assert kwargs['cwd'] == rerun.REPOSITORY

    monkeypatch.setattr(rerun.subprocess, 'run', run)
    rerun.benchmark_job(job, {'test_inputs': tmp_path}, tmp_path / 'run', tmp_path / 'config.toml')
    assert Path(calls[0][1]).name == 'annotator_runner.py'
    assert calls[0][2:] == ['--config', str(tmp_path / 'config.toml'), '--condition', 'ours_our_courts',
                            '--output', str(tmp_path / 'run')]
    assert Path(calls[1][1]).name == 'score.py'
    assert calls[1][2:] == ['--inputs', str(tmp_path), '--run', str(tmp_path / 'run')]


def test_discovery_loads_saved_bundle_and_uses_final_test_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = rerun.Job('A', bundle(tmp_path))
    paths = {name: tmp_path / name for name in ('test_inputs', 'inpaint_root', 'pose_root', 'test_courts')}
    models = SimpleNamespace(preprocessing='saved')
    calls = []
    thread_limits = []

    def limit_threads(*, limits: int) -> nullcontext[None]:
        thread_limits.append(limits)
        return nullcontext()

    monkeypatch.setattr(rerun.grid, 'threadpool_limits', limit_threads)
    monkeypatch.setattr(rerun.grid, 'load_models', lambda path: models if path == job.bundle.path else None)
    videos = [SimpleNamespace(identity=identity) for identity in sorted(rerun.TEST_IDS)]
    monkeypatch.setattr(rerun.final_test, 'load_test_videos', lambda path: videos)

    def load_inputs(video: Any, inpaint: Path, pose: Path, courts: Path) -> Any:
        assert (inpaint, pose, courts) == (paths['inpaint_root'], paths['pose_root'], paths['test_courts'])
        return video

    def run(inputs: Any, base: str, *, models: Any) -> dict[str, Any]:
        assert base == models.preprocessing == 'saved'
        calls.append(inputs.identity)
        return {'events': [], 'sequences': []}

    monkeypatch.setattr(rerun.final_test, 'load_test_inputs', load_inputs)
    monkeypatch.setattr(rerun.grid, 'run_configuration', run)
    scores = fake_scores()['by_video']
    monkeypatch.setattr(rerun.grid, 'score_stream', lambda video, stream: scores[video.identity])
    output = tmp_path / 'run'
    rerun.run_job(job, paths, output, None)
    assert len(calls) == 46
    assert thread_limits == [1]
    assert rerun.complete(job.directory(output))
    rerun.run_job(job, paths, output, None)
    assert len(calls) == 46
    assert thread_limits == [1]


def test_runner_failure_fields_fail_even_after_successful_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = rerun.Job('C', bundle(tmp_path))

    def failed_scores(job: rerun.Job, paths: dict[str, Path], directory: Path, config: Path) -> None:
        scores = write_complete(directory)
        scores['failed_videos'] = ['8']
        rerun.grid.write_json(directory / 'scores.json.gz', scores)

    monkeypatch.setattr(rerun, 'benchmark_job', failed_scores)
    with pytest.raises(ValueError, match='inference failures'):
        rerun.run_job(job, {}, tmp_path, tmp_path / 'config.toml')
    assert not rerun.complete(job.directory(tmp_path))
    assert 'inference failures' in (job.directory(tmp_path) / 'traceback.txt').read_text()


def test_final40_a_reuses_committed_data_without_private_fit_records(tmp_path: Path) -> None:
    bundles = rerun.discover_bundles()
    rerun.reuse_final40(bundles, tmp_path)
    for item in bundles:
        if item.stage != 'final40':
            continue
        directory = rerun.Job('A', item).directory(tmp_path)
        assert rerun.complete(directory)
        raw = gzip.decompress((directory / 'scores.json.gz').read_bytes()).decode()
        assert 'contact_fit' not in raw
        assert str(rerun.REPOSITORY) not in raw
        assert 'bundle' not in json.loads(raw)


def test_reference_import_failure_saves_traceback(tmp_path: Path) -> None:
    saved_bundle = bundle(tmp_path, stage='final40')
    output = tmp_path / 'output'
    with pytest.raises(FileNotFoundError):
        rerun.reuse_final40([saved_bundle], output, tmp_path / 'missing-results')
    directory = rerun.Job('A', saved_bundle).directory(output)
    assert 'FileNotFoundError' in (directory / 'traceback.txt').read_text()
    assert not rerun.complete(directory)


def test_seed_statistics_pool_counts_before_rates(tmp_path: Path) -> None:
    paths = []
    for offset, matched in zip(rerun.final_test.repeats.SEED_OFFSETS, (0, 1, 2, 1, 1), strict=True):
        path = tmp_path / f'{offset}.json.gz'
        scores = fake_scores(matched)
        # Statistics use the per-video counts rather than trusting saved rates.
        scores['by_tolerance']['10']['f1'] = 0.99
        rerun.grid.write_json(path, scores)
        paths.append(path)
    statistics = rerun.seed_statistics(paths)
    assert statistics['10']['f1'] == {'mean': 0.1, 'min': 0.0, 'max': 0.2}
    assert statistics['10']['attributed_f1']['mean'] == pytest.approx(0.02)
    assert statistics['10']['fully_correct_rallies'] == {'mean': 1, 'min': 1, 'max': 1}
    assert statistics['10']['fully_correct_rallies_percent'] == {'mean': 25, 'min': 25, 'max': 25}
    assert statistics['5']['f1'] == {'mean': 0, 'min': 0, 'max': 0}


@pytest.mark.parametrize('mismatch', [False, True])
def test_summary_and_production_parity(tmp_path: Path, capsys: pytest.CaptureFixture[str], mismatch: bool) -> None:
    bundles = rerun.discover_bundles()
    for condition in rerun.CONDITIONS:
        for item in bundles:
            write_complete(rerun.Job(condition, item).directory(tmp_path))
    references = tmp_path / 'references'
    for condition in ('B', 'C'):
        scores = fake_scores()
        if mismatch and condition == 'C':
            scores['by_tolerance']['5']['matched_contacts'] += 1
        reference_path = references / str(rerun.CONDITIONS[condition]) / 'scores.json.gz'
        reference_path.parent.mkdir(parents=True)
        rerun.grid.write_json(reference_path, scores)
    if mismatch:
        with pytest.raises(ValueError, match='production parity failed'):
            rerun.summarise(bundles, tmp_path, references)
    else:
        summary = rerun.summarise(bundles, tmp_path, references)
        assert len(summary['results']) == 12
        assert all(len(row['seed_offsets']) == 5 for row in summary['results'])
    saved = rerun.grid.read_json(tmp_path / 'summary.json.gz')
    assert saved['parity']['B']['passed']
    assert saved['parity']['C']['passed'] is not mismatch
    assert '| A | v32 | grid-017 | 10 |' in capsys.readouterr().out


def test_job_failure_records_traceback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = rerun.Job('B', bundle(tmp_path))

    def fail(*args: Any) -> None:
        raise RuntimeError('simulated inference failure')

    monkeypatch.setattr(rerun, 'benchmark_job', fail)
    with pytest.raises(RuntimeError, match='simulated inference failure'):
        rerun.run_job(job, {}, tmp_path, tmp_path / 'config.toml')
    assert 'simulated inference failure' in (job.directory(tmp_path) / 'traceback.txt').read_text()


def test_pool_bounds_work_and_drains_running_jobs_after_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = [rerun.Job('A', bundle(tmp_path, offset=offset)) for offset in (0, 100, 200)]
    finished = []
    options = {}

    class Executor:
        def __init__(self, **kwargs: Any) -> None:
            options.update(kwargs)
            self.pool = ThreadPoolExecutor(max_workers=kwargs['max_workers'])

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            self.pool.shutdown(wait=True)

        def submit(self, function: Any, *args: Any) -> Future:
            return self.pool.submit(function, *args)

    def run(job: rerun.Job, *args: Any) -> None:
        finished.append(job.bundle.seed_offset)
        if job.bundle.seed_offset == 0:
            raise RuntimeError('worker failed')

    monkeypatch.setattr(rerun, 'ProcessPoolExecutor', Executor)
    monkeypatch.setattr(rerun, 'run_job', run)
    with pytest.raises(RuntimeError, match=r'job\(s\) failed'):
        rerun.run_jobs(jobs, {}, tmp_path, 2)
    assert options['max_workers'] == 2
    assert options['max_tasks_per_child'] == 1
    assert options['mp_context'].get_start_method() == 'spawn'
    assert set(finished) == {0, 100}
    assert 'worker failed' in rerun.grid.read_json(tmp_path / 'failures.json.gz')[0]['traceback']


def test_shared_benchmark_config_is_written_before_pool_starts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved_bundle = bundle(tmp_path)
    jobs = [rerun.Job(condition, saved_bundle) for condition in ('B', 'C')]
    paths = {name: tmp_path / name for name in ('test_inputs', 'inpaint_root', 'pose_root')}
    written = []
    used = []
    write_config = rerun.write_config

    def write(job: rerun.Job, paths: dict[str, Path], output: Path) -> Path:
        config = write_config(job, paths, output)
        written.append(config)
        return config

    def executor(**kwargs: Any) -> ThreadPoolExecutor:
        assert len(written) == 1
        assert written[0].is_file()
        return ThreadPoolExecutor(max_workers=kwargs['max_workers'])

    def run(job: rerun.Job, paths: dict[str, Path], output: Path, config: Path) -> None:
        used.append(config)
        assert config == written[0]

    monkeypatch.setattr(rerun, 'write_config', write)
    monkeypatch.setattr(rerun, 'ProcessPoolExecutor', executor)
    monkeypatch.setattr(rerun, 'run_job', run)
    rerun.run_jobs(jobs, paths, tmp_path, 2)
    assert len(written) == 1
    assert used == [written[0], written[0]]
