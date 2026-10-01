"""Retrain the selected annotator on old-court inputs and compare it with PR149.

Run from the repository root with the current environment:

    PYTHONPATH=.:src python -m experiments.annotator.old_court_regression.runner \\
        --dev-stages DIR --test-inputs DIR --output DIR --jobs 4 --threads 4

Every item finishes by writing a summary file. Rerunning the same command
skips items that have one, so a crash or stop resumes without refitting
completed work. See README.md for inputs, outputs and guarantees.
"""

import argparse
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[3]
CLOSING_PASS = REPOSITORY / 'scratch' / 'contact_det_closing_pass'
FULL_DATASET_RECORDS = REPOSITORY / 'scratch' / 'contact_det_full_ds_fit' / 'records'
THREAD_VARIABLES = (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS',
)
STAGES = (
    'labels', 'old-streams', 'features', 'contact-fits', 'pools', 'sequence-fit', 'evaluate', 'compare', 'report',
)
# Stages with one item get the whole CPU budget in a single process.
SINGLE_ITEM_STAGES = ('labels', 'old-streams', 'sequence-fit', 'report')
# Changing these between invocations would mix results from different inputs.
INPUT_ARGUMENTS = (
    'dev_stages', 'test_inputs', 'shots_master', 'groups_file', 'split_file', 'clean_labels', 'pr149_root',
    'contact_records', 'frozen_dev_features', 'frozen_test_features', 'shuttleset22_annotations',
)


@dataclass(frozen=True)
class Item:
    """One resumable unit of work; ``done`` exists only after it succeeds."""

    name: str
    done: Path | None  # None: always rerun (cheap aggregate steps)
    function: Callable[..., dict[str, Any]]
    arguments: tuple


def utc_now() -> str:
    return datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dev-stages', required=True, type=Path,
                        help='Old dataset-builder run with stages/{shuttle,pose,court}/sset_NN/ for 40 videos')
    parser.add_argument('--test-inputs', required=True, type=Path,
                        help='ShuttleSet22 inpaint root with one "NN <name>" directory per test video')
    parser.add_argument('--output', required=True, type=Path, help='Run directory; reuse it to resume')
    parser.add_argument('--jobs', type=int, default=4, help='Parallel worker processes')
    parser.add_argument('--threads', type=int, default=4, help='Numerical threads per worker process')
    parser.add_argument('--stop-after', choices=STAGES, help='Stop after this stage completes')
    parser.add_argument('--shots-master', type=Path,
                        default=REPOSITORY / 'training/data/shuttleset/annotations/shots_master.csv')
    parser.add_argument('--groups-file', type=Path, default=FULL_DATASET_RECORDS / 'training_video_score_groups.json')
    parser.add_argument('--split-file', type=Path, default=FULL_DATASET_RECORDS / 'shuttleset_development_split.json')
    parser.add_argument('--clean-labels', type=Path, default=(
        REPOSITORY / 'scratch/contact_det_full_ds_fit/raw/shuttleset22-test-result/clean_labels.json.gz'))
    parser.add_argument('--pr149-root', type=Path, default=CLOSING_PASS)
    parser.add_argument('--contact-records', type=Path,
                        help='Optional scratch/contact_det_full_ds_fit/raw: original contact-fit selection records')
    parser.add_argument('--frozen-dev-features', type=Path,
                        help='Optional scratch/contact_det_full_ds_fit/raw/full_raw: adds a frozen-feature fit')
    parser.add_argument('--frozen-test-features', type=Path,
                        help='Optional scratch/contact_det_full_ds_fit/raw/shuttleset22-test-predictions')
    parser.add_argument('--shuttleset22-annotations', type=Path,
                        help='Optional raw ShuttleSet22 annotation root (set/match.csv) for the all-source read')
    return parser


def optional_path(path: Path | None) -> Path | None:
    return None if path is None else path.resolve()


def limit_threads(count: int) -> None:
    """Set numerical thread limits; child processes read them when they import NumPy."""
    for name in THREAD_VARIABLES:
        os.environ[name] = str(count)


class Progress:
    """Append timestamped progress lines to the run's log and standard output."""

    def __init__(self, output: Path) -> None:
        self.path = output / 'progress.log'

    def write(self, message: str) -> None:
        line = f'{utc_now()} {message}'
        print(line, flush=True)
        with self.path.open('a') as handle:
            handle.write(line + '\n')


def record_invocation(arguments: argparse.Namespace, versions: dict[str, str]) -> None:
    """Keep every invocation; refuse to resume with different inputs."""
    path = arguments.output / 'run.json'
    invocations = json.loads(path.read_text()) if path.is_file() else []
    inputs = {name: None if getattr(arguments, name) is None else str(getattr(arguments, name).resolve())
              for name in INPUT_ARGUMENTS}
    if invocations and invocations[0]['inputs'] != inputs:
        raise SystemExit(f'{path}: this output directory was started with different inputs; use a new --output')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, capture_output=True, text=True, check=True)
    status = subprocess.run(
        ['git', 'status', '--porcelain'], cwd=REPOSITORY, capture_output=True, text=True, check=True,
    )
    invocations.append({
        'started': utc_now(), 'inputs': inputs, 'jobs': arguments.jobs, 'threads': arguments.threads,
        'stop_after': arguments.stop_after, 'versions': versions, 'git_commit': commit.stdout.strip(),
        'git_changed_paths': len(status.stdout.splitlines()),
    })
    path.write_text(json.dumps(invocations, indent=2) + '\n')


def update_timings(output: Path, stage: str, seconds: float, ran: int, skipped: int) -> None:
    path = output / 'timings.json'
    timings = json.loads(path.read_text()) if path.is_file() else {}
    timings[stage] = {'seconds': round(seconds, 1), 'items_run': ran, 'items_skipped': skipped, 'finished': utc_now()}
    path.write_text(json.dumps(timings, indent=2) + '\n')


def stage_items(stage: str, retrain: Any, settings: Any, frozen_fit: bool) -> list[Item]:
    """List one stage's items in submission order, longest first where it matters."""
    development = retrain.load_development(settings)
    if stage == 'labels':
        return [Item('labels', retrain.summary_path(settings, 'labels', 'summary'), retrain.write_labels_job,
                     (settings,))]
    if stage == 'old-streams':
        return [Item('old-streams', retrain.summary_path(settings, 'old', 'summary'), retrain.old_streams_job,
                     (settings,))]
    if stage == 'features':
        return [Item(video, retrain.summary_path(settings, 'features', video), retrain.development_features_job,
                     (settings, video)) for video in development.id_order(list(development.group_by_video))]
    if stage == 'contact-fits':
        names = ['final_40', *(['frozen_final_40'] if frozen_fit else []), 'v_32', 'held_A', 'held_B', 'held_C',
                 'held_D']
        return [Item(name, retrain.summary_path(settings, 'contact', name), retrain.contact_fit_job,
                     (settings, name)) for name in names]
    if stage == 'pools':
        return [Item(video, retrain.summary_path(settings, 'pools', video), retrain.pool_job, (settings, video))
                for video in development.videos(retrain.TRAINING_GROUPS)]
    if stage == 'sequence-fit':
        return [Item('sequence-fit', retrain.summary_path(settings, 'sequence', 'summary'),
                     retrain.sequence_fit_job, (settings,))]
    if stage == 'evaluate':
        validation = [('validation', video) for video in development.videos((retrain.VALIDATION_GROUP,))]
        test = [('test', str(video_id)) for video_id in retrain.TEST_VIDEO_IDS]
        return [Item(f'{split}/{video}', retrain.summary_path(settings, 'eval', split, video, 'summary'),
                     retrain.evaluate_job, (settings, split, video)) for split, video in validation + test]
    if stage == 'compare':
        pairs = ([('test', str(video_id)) for video_id in retrain.TEST_VIDEO_IDS]
                 + [('development', video) for video in development.videos(retrain.TRAINING_GROUPS)]
                 + [('validation', video) for video in development.videos((retrain.VALIDATION_GROUP,))])
        return [Item(f'{split}/{video}', retrain.summary_path(settings, 'compare', split, video),
                     retrain.compare_job, (settings, split, video)) for split, video in pairs]
    if stage == 'report':
        return [Item('report', None, retrain.report_job, (settings,))]
    raise ValueError(f'unknown stage {stage}')


def run_stage(stage: str, items: list[Item], jobs: int, threads: int, output: Path, progress: Progress) -> list[str]:
    """Run a stage's unfinished items in fresh processes; return the names that failed.

    Per-item failures are logged and the remaining items continue. Each worker
    handles one item, then exits, so large arrays never accumulate.
    """
    pending = [item for item in items if item.done is None or not item.done.is_file()]
    skipped = len(items) - len(pending)
    progress.write(f'{stage} start items={len(items)} already_done={skipped}')
    started = time.perf_counter()
    failures = []
    if pending:
        workers = 1 if stage in SINGLE_ITEM_STAGES else min(jobs, len(pending))
        # Spawned workers copy this environment, so set limits before creating them.
        limit_threads(jobs * threads if stage in SINGLE_ITEM_STAGES else threads)
        context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, max_tasks_per_child=1) as executor:
            submitted = {}
            for item in pending:
                submitted[executor.submit(item.function, *item.arguments)] = (item, time.perf_counter())
            waiting = set(submitted)
            done_count = 0
            while waiting:
                finished, waiting = wait(waiting, timeout=60, return_when=FIRST_COMPLETED)
                if not finished:
                    elapsed = time.perf_counter() - started
                    progress.write(f'{stage} running: {done_count}/{len(pending)} complete, {elapsed:.0f}s elapsed')
                for future in finished:
                    done_count += 1
                    item, item_started = submitted[future]
                    seconds = time.perf_counter() - item_started
                    failure_path = output / 'failures' / stage / f'{item.name.replace("/", "_")}.txt'
                    try:
                        summary = future.result()
                    except Exception as error:  # noqa: BLE001 - keep other items running; the stage fails below
                        failure_path.parent.mkdir(parents=True, exist_ok=True)
                        failure_path.write_text(''.join(traceback.format_exception(error)))
                        failures.append(item.name)
                        progress.write(f'{stage} {done_count}/{len(pending)} {item.name} FAILED {seconds:.1f}s '
                                       f'{type(error).__name__}: {error} (traceback: {failure_path})')
                        continue
                    failure_path.unlink(missing_ok=True)
                    seconds = summary.get('seconds', seconds)
                    progress.write(f'{stage} {done_count}/{len(pending)} {item.name} complete in {seconds:.1f}s')
    seconds = time.perf_counter() - started
    update_timings(output, stage, seconds, len(pending) - len(failures), skipped)
    progress.write(f'{stage} end seconds={seconds:.1f} failed={len(failures)}')
    return failures


def main(argv: list[str] | None = None) -> int:
    arguments = argument_parser().parse_args(argv)
    if arguments.jobs < 1 or arguments.threads < 1:
        raise SystemExit('--jobs and --threads must be positive')
    arguments.output.mkdir(parents=True, exist_ok=True)
    limit_threads(arguments.threads)
    # Imported after the thread limits so this process's NumPy and OpenMP read them too.
    import joblib
    import numpy
    import pandas
    import sklearn

    from experiments.annotator.old_court_regression import retrain

    progress = Progress(arguments.output)
    failed_marker = arguments.output / 'FAILED'
    failed_marker.unlink(missing_ok=True)
    versions = {'python': platform.python_version(), 'numpy': numpy.__version__, 'pandas': pandas.__version__,
                'scikit-learn': sklearn.__version__, 'joblib': joblib.__version__}
    record_invocation(arguments, versions)
    settings = retrain.RunSettings(
        output=arguments.output.resolve(), dev_stages=arguments.dev_stages.resolve(),
        test_inputs=arguments.test_inputs.resolve(), shots_master=arguments.shots_master.resolve(),
        groups_file=arguments.groups_file.resolve(), split_file=arguments.split_file.resolve(),
        clean_labels=arguments.clean_labels.resolve(), pr149_root=arguments.pr149_root.resolve(),
        contact_records=optional_path(arguments.contact_records),
        frozen_dev_features=optional_path(arguments.frozen_dev_features),
        frozen_test_features=optional_path(arguments.frozen_test_features),
        shuttleset22_annotations=optional_path(arguments.shuttleset22_annotations),
    )
    progress.write(f'run start versions={json.dumps(versions)} jobs={arguments.jobs} threads={arguments.threads}')
    current_stage = 'setup'
    try:
        for current_stage in STAGES:
            items = stage_items(current_stage, retrain, settings, arguments.frozen_dev_features is not None)
            failures = run_stage(current_stage, items, arguments.jobs, arguments.threads, settings.output, progress)
            if failures:
                failed_marker.write_text(f'{utc_now()} stage {current_stage} failed items: {", ".join(failures)}\n'
                                         f'Tracebacks: {settings.output / "failures" / current_stage}\n')
                progress.write(f'run stopped: {len(failures)} {current_stage} items failed; rerun to retry them')
                return 1
            if current_stage == arguments.stop_after:
                progress.write(f'run stopped after {current_stage} as requested')
                return 0
    except BaseException as error:
        failed_marker.write_text(f'{utc_now()} stage {current_stage} crashed\n'
                                 + ''.join(traceback.format_exception(error)))
        progress.write(f'run crashed in {current_stage}: {type(error).__name__}: {error}')
        raise
    progress.write(f'run complete; see {settings.output / "comparison.md"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
