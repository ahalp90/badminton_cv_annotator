"""Run saved heuristic configurations on a frozen split, resuming each video."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import multiprocessing
import os
import platform
import resource
import subprocess
import time
import tomllib
import traceback
from collections import OrderedDict, deque
from collections.abc import Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    wait,
)
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

os.environ.update(
    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
    NUMBA_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', BLIS_NUM_THREADS='1',
)

from threadpoolctl import threadpool_limits

from annotator.models import AnnotatorModels, load_models, model_files
from dataset_builder.shuttle_evidence import GUARD_CODES_FILENAME
from dataset_builder.vision import (
    COURT_EVIDENCE_FILENAME,
    COURT_KEEP_VOTE_FILENAME,
    COURT_PRESENT_FILENAME,
    POSE_FILENAMES,
    TRACK_FILENAME,
)
from experiments.annotator.heuristic_only_tune.cache import (
    StickyCacheError,
    cache_sticky_result,
)
from experiments.annotator.heuristic_only_tune.data import (
    COUNT_FIELDS,
    REPOSITORY,
    TRACKED_SPLIT,
    SearchVideo,
    add_rates,
    load_original_inputs,
    load_original_videos,
    load_test_inputs,
    load_test_videos,
    run_configuration,
    score_stream,
)
from experiments.annotator.heuristic_tuning.corpus_sweep import (
    CandidateOutcome,
    _atomic_write_json_gz,
    candidate_id,
    latest_attempt,
    write_attempt,
)
from experiments.annotator.heuristic_tuning.sweep import (
    CandidateSpec,
    _base_and_serve,
    serialise_spec,
)
from experiments.annotator.old_court_regression.old_inputs import (
    TEST_GUARD_CODES_FILENAME,
    TEST_TRACK_FILENAME,
    OldVideoInputs,
)
from shared.progress import PROGRESS_ENV, report

MAX_WORKERS = 26
INPUT_CACHE_SIZE = 3
STATUS_INTERVAL_SECONDS = 1.0
EMPTY_STREAM: dict[str, list[Any]] = {'events': [], 'sequences': []}


@dataclass(frozen=True)
class SearchJob:
    """One independently saved configuration/video evaluation."""

    video: SearchVideo
    spec: CandidateSpec

    @property
    def identity(self) -> str:
        return f'{self.video.identity}/{candidate_id(self.spec)}'


@dataclass
class SearchPlan:
    """Remaining jobs and counts recovered from completed attempts."""

    jobs: deque[SearchJob]
    completed: int = 0
    failed: int = 0


@dataclass
class WorkerState:
    """Inputs retained within one process; each process runs one job at a time."""

    paths: dict[str, Path]
    output: Path
    models: AnnotatorModels | None
    inputs: OrderedDict[str, OldVideoInputs]


@dataclass
class ActiveJob:
    job: SearchJob
    started: float


_WORKER: WorkerState


def read_json(path: Path) -> Any:
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def write_json(path: Path, payload: Any) -> None:
    _atomic_write_json_gz(path, payload)


def load_candidates(path: Path) -> tuple[dict[str, Any], list[CandidateSpec]]:
    document = read_json(path)
    candidates = []
    for row in document['candidates']:
        spec = CandidateSpec(row.get('label', 'grid'), row['overrides_base30'], row.get('strategies', {}))
        serialise_spec(spec)
        candidates.append(spec)
    if not candidates or len({candidate_id(spec) for spec in candidates}) != len(candidates):
        raise ValueError('candidate file must contain distinct configurations')
    return document, candidates


def load_paths(path: Path) -> dict[str, Path]:
    with path.open('rb') as stream:
        document = tomllib.load(stream)
    paths = {}
    for key, value in document['paths'].items():
        candidate = Path(value).expanduser()
        paths[key] = candidate if candidate.is_absolute() else (REPOSITORY / candidate).resolve()
    return paths


def source_files(repository: Path) -> dict[str, str]:
    """Hash effective source contents, including files not yet tracked by Git."""
    files = []
    for directory in ('src', 'experiments/annotator'):
        files.extend((repository / directory).rglob('*.py'))
    for name in ('pyproject.toml', 'uv.lock'):
        path = repository / name
        if path.exists():
            files.append(path)
    return {str(path.relative_to(repository)): hashlib.md5(path.read_bytes()).hexdigest() for path in sorted(files)}


def source_record() -> dict[str, Any]:
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, text=True).strip()
    return {
        'revision': revision, 'files_md5': source_files(REPOSITORY),
        'python': platform.python_version(),
        'packages': {name: version(name) for name in (
            'numpy', 'scipy', 'scikit-learn', 'pandas', 'joblib', 'numba',
            'beartype', 'jaxtyping', 'threadpoolctl',
        )},
    }


def input_paths(paths: dict[str, Path], videos: tuple[SearchVideo, ...], full: bool) -> dict[str, Path]:
    """Name every persisted file consumed by this population and mode."""
    files: dict[str, Path] = {}
    if videos[0].split == 'test':
        files['test_inputs/manifest.json.gz'] = paths['test_inputs'] / 'manifest.json.gz'
    else:
        files['shots_master'] = paths['shots_master']
        files['development_split'] = TRACKED_SPLIT
    for video in videos:
        if video.split == 'test':
            assert video.directory is not None
            shuttle = paths['inpaint_root'] / video.directory
            track = shuttle / TEST_TRACK_FILENAME
            guards = shuttle / TEST_GUARD_CODES_FILENAME
            pose = paths['pose_root'] / video.directory
            court = paths['test_courts'] / video.identity
        else:
            shuttle = paths['stages'] / 'shuttle' / video.identity
            track = shuttle / TRACK_FILENAME
            guards = shuttle / GUARD_CODES_FILENAME
            pose = paths['stages'] / 'pose' / video.identity
            court = paths['development_courts'] / video.identity
        files[f'{video.identity}/shuttle/track'] = track
        files[f'{video.identity}/shuttle/guards'] = guards
        for field, filename in POSE_FILENAMES.items():
            files[f'{video.identity}/pose/{field}'] = pose / filename
        for filename in (COURT_EVIDENCE_FILENAME, COURT_KEEP_VOTE_FILENAME, COURT_PRESENT_FILENAME):
            files[f'{video.identity}/court/{filename}'] = court / filename
    if full:
        files.update(model_files(paths['bundle']))
    return files


def file_manifest(files: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Record the fixed input files and reject changed files when resuming."""
    records = {}
    for name, path in sorted(files.items()):
        details = path.stat()
        records[name] = {
            'path': str(path.resolve()), 'size': details.st_size, 'mtime_ns': details.st_mtime_ns,
        }
    return records


def keep_run_identity(path: Path, document: dict[str, Any]) -> None:
    if path.exists():
        previous = read_json(path)
        current = dict(document)
        # A documentation-only commit can change HEAD without changing the run.
        for identity in (previous, current):
            identity['source'] = dict(identity['source'])
            identity['source'].pop('revision', None)
        if previous != current:
            raise ValueError(f'{path}: source, inputs or configurations changed; use a fresh output directory')
    else:
        write_json(path, document)


def initialise_worker(paths: dict[str, Path], output: Path, full: bool) -> None:
    global _WORKER
    threadpool_limits(limits=1)
    models = load_models(paths['bundle']) if full else None
    _WORKER = WorkerState(paths, output, models, OrderedDict())


def load_video_inputs(video: SearchVideo, paths: dict[str, Path]) -> OldVideoInputs:
    if video.split == 'test':
        return load_test_inputs(video, paths['inpaint_root'], paths['pose_root'], paths['test_courts'])
    return load_original_inputs(video, paths['stages'], paths['development_courts'])


def progress_path(output: Path, job: SearchJob) -> Path:
    return output / 'progress' / f'{job.video.identity}-{candidate_id(job.spec)}.jsonl'


def read_progress(path: Path) -> dict[str, Any] | None:
    """Read the newest complete display event while its worker may be appending."""
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 8192))
            complete_lines = stream.read().split(b'\n')[:-1]
        for line in reversed(complete_lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    except OSError:
        return None
    return None


def run_one_job(job: SearchJob) -> bool:
    """Save one result; failed inference is scored as an empty prediction stream."""
    video, spec = job.video, job.spec
    started = time.perf_counter()
    progress_file = progress_path(_WORKER.output, job)
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text('')
    os.environ[PROGRESS_ENV] = str(progress_file)
    report('loading')
    cache_hit = video.identity in _WORKER.inputs
    if cache_hit:
        inputs = _WORKER.inputs.pop(video.identity)
    else:
        inputs = load_video_inputs(video, _WORKER.paths)
    _WORKER.inputs[video.identity] = inputs
    if len(_WORKER.inputs) > INPUT_CACHE_SIZE:
        _WORKER.inputs.popitem(last=False)
    load_seconds = time.perf_counter() - started
    base, serve_start = _base_and_serve(spec)
    models = _WORKER.models
    if models is not None:
        base = models.preprocessing
    report('inferring')
    inference_started = time.perf_counter()
    failure = None
    try:
        with cache_sticky_result(_WORKER.output / 'cache' / 'sticky', video.identity):
            stream = run_configuration(inputs, base, serve_start=serve_start, models=models)
    except StickyCacheError:
        raise
    except Exception as error:  # noqa: BLE001 -- retain failed inference in the scored population.
        failure = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        stream = EMPTY_STREAM
    inference_seconds = time.perf_counter() - inference_started
    report('scoring')
    score_started = time.perf_counter()
    counts = score_stream(video, stream)
    score_seconds = time.perf_counter() - score_started
    elapsed = time.perf_counter() - started
    outcome = CandidateOutcome(
        status='succeeded', elapsed_seconds=elapsed, aggregate=counts,
        fixtures={video.identity: {
            'stream': stream, 'failure': failure, 'load_seconds': load_seconds, 'input_cache_hit': cache_hit,
            'inference_seconds': inference_seconds, 'score_seconds': score_seconds,
            'max_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }},
    )
    write_attempt(_WORKER.output, video.identity, spec, outcome)
    return failure is not None


def pending_jobs(videos: tuple[SearchVideo, ...], specs: list[CandidateSpec], output: Path) -> SearchPlan:
    """Visit every video for each configuration, omitting durable completed jobs."""
    plan = SearchPlan(deque())
    for spec in specs:
        for video in videos:
            attempt = latest_attempt(output, video.identity, spec)
            if attempt is not None and attempt['status'] == 'succeeded':
                plan.completed += 1
                plan.failed += attempt['fixtures'][video.identity]['failure'] is not None
            else:
                plan.jobs.append(SearchJob(video, spec))
    return plan


def worker_target(initial: int, path: Path | None) -> int:
    target = initial if path is None else int(path.read_text().strip())
    if not 0 <= target <= MAX_WORKERS:
        raise ValueError(f'worker target must be between 0 and {MAX_WORKERS}')
    return target


def publish_status(
    output: Path, plan: SearchPlan, active: dict[Future[bool], ActiveJob], *,
    total: int, target: int, started: float, state: str, error: BaseException | None = None,
) -> None:
    now = time.monotonic()
    active_rows = []
    for item in active.values():
        progress = read_progress(progress_path(output, item.job))
        row = {
            'video': item.job.video.identity, 'candidate_id': candidate_id(item.job.spec),
            'label': item.job.spec.label, 'stage': 'starting' if progress is None else progress['activity'],
            'elapsed_seconds': now - item.started, 'progress': progress,
        }
        if progress is not None:
            row.update({key: progress[key] for key in ('completed', 'total', 'unit')})
        active_rows.append(row)
    write_json(output / 'status.json.gz', {
        'schema': 'heuristic-only-tune-status/1', 'state': state,
        'total_jobs': total, 'completed_jobs': plan.completed, 'active_jobs': len(active),
        'queued_jobs': len(plan.jobs), 'failed_jobs': plan.failed, 'target_workers': target,
        'elapsed_seconds': now - started, 'updated_at_utc': datetime.now(UTC).isoformat(),
        'active': active_rows,
        'error': None if error is None else {'type': type(error).__name__, 'message': str(error)},
    })


def schedule_jobs(
    pool: Executor, plan: SearchPlan, output: Path, initial_workers: int,
    workers_file: Path | None = None,
) -> dict[str, Any]:
    """Refill a shared queue without batch barriers, draining jobs on a decrease."""
    total = plan.completed + len(plan.jobs)
    started = time.monotonic()
    active: dict[Future[bool], ActiveJob] = {}
    target = worker_target(initial_workers, workers_file)
    targets = []
    peak_active = 0
    try:
        while plan.jobs or active:
            target = worker_target(initial_workers, workers_file)
            if not targets or targets[-1]['workers'] != target:
                targets.append({'seconds': time.monotonic() - started, 'workers': target})
            while plan.jobs and len(active) < target:
                job = plan.jobs.popleft()
                active[pool.submit(run_one_job, job)] = ActiveJob(job, time.monotonic())
            peak_active = max(peak_active, len(active))
            state = 'paused' if target == 0 and not active else 'running'
            publish_status(output, plan, active, total=total, target=target, started=started, state=state)
            if not active:
                time.sleep(STATUS_INTERVAL_SECONDS)
                continue
            done, _pending = wait(active, timeout=STATUS_INTERVAL_SECONDS, return_when=FIRST_COMPLETED)
            for future in done:
                failed = future.result()
                item = active.pop(future)
                progress_path(output, item.job).unlink(missing_ok=True)
                plan.completed += 1
                plan.failed += failed
                print(
                    f'{plan.completed}/{total} {item.job.video.identity} '
                    f'{candidate_id(item.job.spec)[:12]} failed={failed}', flush=True,
                )
    except BaseException as error:
        publish_status(output, plan, active, total=total, target=target, started=started, state='failed', error=error)
        raise
    publish_status(output, plan, active, total=total, target=target, started=started, state='summarising')
    return {'worker_targets': targets, 'peak_active_jobs': peak_active, 'wall_seconds': time.monotonic() - started}


def summarise(output: Path, videos: tuple[SearchVideo, ...], specs: list[CandidateSpec]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        totals = {str(tolerance): dict.fromkeys(COUNT_FIELDS, 0) for tolerance in (5, 10)}
        by_video = {}
        failed_videos = []
        seconds = 0.0
        peak_rss = 0.0
        for video in videos:
            attempt = latest_attempt(output, video.identity, spec)
            if attempt is None or attempt['status'] != 'succeeded':
                raise ValueError(f'{video.identity}: configuration {candidate_id(spec)} is unfinished')
            counts = attempt['aggregate']
            by_video[video.identity] = {tolerance: add_rates(values) for tolerance, values in counts.items()}
            record = attempt['fixtures'][video.identity]
            if record['failure'] is not None:
                failed_videos.append(video.identity)
            seconds += attempt['elapsed_seconds']
            peak_rss = max(peak_rss, record['max_rss_mib'])
            for tolerance, values in counts.items():
                for key in COUNT_FIELDS:
                    totals[tolerance][key] += values[key]
        rows.append({
            'candidate_id': candidate_id(spec), 'spec': serialise_spec(spec),
            'by_tolerance': {tolerance: add_rates(values) for tolerance, values in totals.items()},
            'by_video': by_video, 'failed_videos': failed_videos,
            'seconds': seconds, 'max_worker_rss_mib': peak_rss,
        })
    return rows


@contextmanager
def coordinator_lock(output: Path) -> Iterator[None]:
    """Keep one coordinator in an output directory, including during resume."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.coordinator.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'{output}: another search coordinator is already running') from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run(args: argparse.Namespace) -> None:
    with coordinator_lock(args.output):
        run_locked(args)


def run_locked(args: argparse.Namespace) -> None:
    if not 1 <= args.workers <= MAX_WORKERS:
        raise ValueError(f'workers must be between 1 and {MAX_WORKERS}')
    paths = load_paths(args.paths)
    document, specs = load_candidates(args.candidates)
    if args.split == 'validation' and document['stage'] != 'shortlist':
        raise ValueError('validation requires a frozen shortlist')
    if args.split == 'test' and document['stage'] != 'selected' and not args.full:
        raise ValueError('test evaluation requires settings selected on validation')
    if args.full and (len(specs) != 1 or specs[0].overrides_base30 or specs[0].strategies):
        raise ValueError('the full annotator uses its fitted settings; supply one empty reference configuration')
    if args.split == 'test':
        videos = load_test_videos(paths['test_inputs'])
    else:
        videos = tuple(video for video in load_original_videos(paths['shots_master']) if video.split == args.split)
    if args.only:
        requested = set(args.only.split(','))
        if not requested <= {video.identity for video in videos}:
            raise ValueError('requested smoke-test videos are outside this split')
        videos = tuple(video for video in videos if video.identity in requested)
    if not videos:
        raise ValueError('the selected split contains no videos')
    identity = {
        'schema': 'heuristic-only-tune/2', 'source': source_record(), 'split': args.split,
        'mode': 'full' if args.full else 'heuristic', 'smoke_test': bool(args.only),
        'paths': {key: str(path) for key, path in paths.items()},
        'input_files': file_manifest(input_paths(paths, videos, args.full)),
        'candidates': document,
        'videos': [{
            'identity': video.identity, 'dataset': video.dataset, 'split': video.split,
            'fps': video.fps, 'resolution': list(video.resolution),
            'directory': video.directory, 'frame_count': video.frame_count,
            'rallies': [[rally.identity, list(rally.frames),
                         [None if side is None else side.value for side in rally.sides]] for rally in video.rallies],
        } for video in videos],
    }
    keep_run_identity(args.output / 'run.json.gz', identity)
    if args.workers_file is not None and not args.workers_file.exists():
        args.workers_file.parent.mkdir(parents=True, exist_ok=True)
        args.workers_file.write_text(f'{args.workers}\n')
    worker_target(args.workers, args.workers_file)
    plan = pending_jobs(videos, specs, args.output)
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS, mp_context=multiprocessing.get_context('spawn'),
        initializer=initialise_worker, initargs=(paths, args.output, args.full),
    ) as pool:
        execution = schedule_jobs(pool, plan, args.output, args.workers, args.workers_file)
    targets = execution['worker_targets']
    target = targets[-1]['workers'] if targets else args.workers
    try:
        rows = summarise(args.output, videos, specs)
        write_json(args.output / 'summary.json.gz', {
            'schema': 'heuristic-only-tune-summary/1', 'split': args.split,
            'mode': identity['mode'], 'smoke_test': bool(args.only),
            'videos': [video.identity for video in videos], 'candidates': rows,
        })
        execution['wall_seconds'] = time.monotonic() - started
        write_json(args.output / 'execution.json.gz', {'initial_workers': args.workers, **execution})
    except BaseException as error:
        publish_status(
            args.output, plan, {}, total=plan.completed, target=target,
            started=started, state='failed', error=error,
        )
        raise
    publish_status(args.output, plan, {}, total=plan.completed, target=target, started=started, state='complete')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument(
        '--workers-file', type=Path,
        help='live worker target (integer 0..26); zero pauses new jobs, reductions let active jobs finish',
    )
    parser.add_argument('--only', help='comma-separated videos for timing or smoke tests')
    parser.add_argument('--full', action='store_true', help='use the full annotator and its fitted configuration')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
