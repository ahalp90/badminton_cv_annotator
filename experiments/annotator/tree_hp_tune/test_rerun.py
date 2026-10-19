"""Score saved 32/40-video bundles on test under discovered and known rallies."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import traceback
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

# Spawned jobs and benchmark subprocesses inherit these before numerical imports.
os.environ.update(
    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
)

# Bound inline: ruff's E402 allows sys.path calls above imports, but not assignments.
sys.path.insert(0, str(REPOSITORY := Path(__file__).resolve().parents[3]))
sys.path.insert(1, str(REPOSITORY / 'src'))

from experiments.annotator.tree_hp_tune import final_test

# isort: split

from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    VIDEO_IDS,
    load_manifest,
)

grid = final_test.grid
RESULTS = Path(__file__).resolve().parent / 'results'
BENCHMARK = REPOSITORY / 'experiments/annotator/monotrack_benchmark'
CONFIGURATIONS = {
    'grid-017': 'a521305d102725a9673c7ca76934cc2eb42106ca61285b754913aee0c14bc762',
    'shipped': 'a322221a4890572e9065814a2e4e3e0ad033ba679e9b60a46ed38d9b639d7cd6',
}
CONDITIONS = {'A': None, 'B': 'ours_our_courts', 'C': 'ours_mono_courts'}
TOLERANCES = ('10', '5')
METRICS = ('f1', 'attributed_f1', 'fully_correct_rallies', 'fully_correct_rallies_percent')
TEST_IDS = frozenset(map(str, VIDEO_IDS))
SCHEMA = 'tree-retune-test-rerun-v1'


@dataclass(frozen=True)
class Bundle:
    stage: str
    label: str
    seed_offset: int
    path: Path


@dataclass(frozen=True)
class Job:
    condition: str
    bundle: Bundle

    def directory(self, output: Path) -> Path:
        return output / self.condition / self.bundle.stage / self.bundle.label / f'seed-{self.bundle.seed_offset}'


def discover_bundles(root: Path = RESULTS) -> list[Bundle]:
    bundles = []
    for stage in ('v32', 'final40'):
        for label, identity in CONFIGURATIONS.items():
            for offset in final_test.repeats.SEED_OFFSETS:
                if stage == 'v32':
                    trial = final_test.repeats.seed_root(root, offset) / 'trials' / identity
                else:
                    trial = final_test.trial_root(root, identity, offset)
                path = trial / f'bundle_{stage}'
                for model_path in final_test.model_files(path).values():
                    if not model_path.is_file():
                        raise FileNotFoundError(model_path)
                bundles.append(Bundle(stage, label, offset, path.resolve()))
    return bundles


def build_jobs(bundles: list[Bundle]) -> list[Job]:
    return [Job(condition, bundle) for condition in CONDITIONS for bundle in bundles
            if condition != 'A' or bundle.stage == 'v32']


def validate_scores(scores: dict[str, Any]) -> None:
    """Require the fixed population, both tolerances and successful inference."""
    if set(scores['by_video']) != TEST_IDS:
        raise ValueError('scores must contain all 46 test videos')
    if scores.get('failed_videos') or scores.get('failed_rallies') or scores.get('failure_stages'):
        raise ValueError('scores contain inference failures')
    for tolerance in TOLERANCES:
        for counts in [scores['by_tolerance'][tolerance],
                       *(video[tolerance] for video in scores['by_video'].values())]:
            for field in grid.COUNT_FIELDS:
                value = counts[field]
                if not isinstance(value, int) or value < 0:
                    raise ValueError(f'invalid {field} count')


def complete(directory: Path) -> bool:
    try:
        validate_scores(grid.read_json(directory / 'scores.json.gz'))
    except (OSError, EOFError, ValueError, KeyError, TypeError, zlib.error):
        return False
    return all((directory / f'{identity}.json.gz').is_file() for identity in TEST_IDS)


def prepare_job(directory: Path) -> bool:
    """Keep complete jobs; remove partial output before the runner creates it."""
    if complete(directory):
        return False
    if directory.exists():
        shutil.rmtree(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    return True


def write_config(job: Job, paths: dict[str, Path], output: Path) -> Path:
    bundle = job.bundle
    config = output / 'configs' / bundle.stage / bundle.label / f'seed-{bundle.seed_offset}.toml'
    config.parent.mkdir(parents=True, exist_ok=True)
    values = {'bundle': bundle.path, 'inputs': paths['test_inputs'],
              'inpaint_root': paths['inpaint_root'], 'pose_root': paths['pose_root']}
    lines = ['[paths]']
    for name, path in values.items():
        # JSON string escaping also produces valid TOML basic strings for paths.
        lines.append(f'{name} = {json.dumps(str(path.resolve()))}')
    lines.extend(['', '[execution]', 'annotator_workers = 1', ''])
    config.write_text('\n'.join(lines))
    return config


def discovery_job(job: Job, paths: dict[str, Path], directory: Path) -> None:
    models = grid.load_models(job.bundle.path)
    videos = final_test.load_test_videos(paths['test_inputs'])
    if {video.identity for video in videos} != TEST_IDS or len(videos) != len(TEST_IDS):
        raise ValueError('discovery requires the fixed 46 test videos')
    directory.mkdir()
    per_video = {}
    for video in videos:
        inputs = final_test.load_test_inputs(video, paths['inpaint_root'], paths['pose_root'], paths['test_courts'])
        stream = grid.run_configuration(inputs, models.preprocessing, models=models)
        grid.write_json(directory / f'{video.identity}.json.gz', stream)
        per_video[video.identity] = grid.score_stream(video, stream)
    grid.write_json(directory / 'scores.json.gz', {
        'by_video': per_video, 'by_tolerance': grid.pooled_scores(per_video),
    })


def benchmark_job(job: Job, paths: dict[str, Path], directory: Path, config: Path) -> None:
    scripts = BENCHMARK / 'benchmark_scripts'
    commands = [
        [sys.executable, str(scripts / 'annotator_runner.py'), '--config', str(config),
         '--condition', str(CONDITIONS[job.condition]), '--output', str(directory)],
        [sys.executable, str(scripts / 'score.py'), '--inputs', str(paths['test_inputs']), '--run', str(directory)],
    ]
    with directory.with_suffix('.log').open('w') as log:
        for command in commands:
            subprocess.run(command, cwd=REPOSITORY, stdout=log, stderr=subprocess.STDOUT, check=True)


def run_job(job: Job, paths: dict[str, Path], output: Path, config: Path | None) -> None:
    directory = job.directory(output)
    try:
        if not prepare_job(directory):
            return
        with grid.threadpool_limits(limits=1):
            if job.condition == 'A':
                discovery_job(job, paths, directory)
            else:
                assert config is not None
                benchmark_job(job, paths, directory, config)
        validate_scores(grid.read_json(directory / 'scores.json.gz'))
        if not complete(directory):
            raise ValueError('job did not produce all 46 predictions and complete scores')
    except Exception:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'traceback.txt').write_text(traceback.format_exc())
        raise


def reuse_final40(bundles: list[Bundle], output: Path, root: Path = RESULTS) -> None:
    for bundle in bundles:
        if bundle.stage != 'final40':
            continue
        directory = Job('A', bundle).directory(output)
        try:
            if not prepare_job(directory):
                continue
            source = final_test.trial_root(root, CONFIGURATIONS[bundle.label], bundle.seed_offset)
            result = grid.read_json(source / 'result.json.gz')
            if (result['id'], result['label'], result['seed_offset']) != (
                CONFIGURATIONS[bundle.label], bundle.label, bundle.seed_offset,
            ):
                raise ValueError(f'{source}: final40 reference identity differs')
            scores = {'by_video': result['per_video'], 'by_tolerance': result['totals']}
            validate_scores(scores)
            directory.mkdir()
            for identity in TEST_IDS:
                shutil.copyfile(source / 'test_streams' / f'{identity}.json.gz', directory / f'{identity}.json.gz')
            # Fit records contain machine paths; only scores and streams belong here.
            grid.write_json(directory / 'scores.json.gz', scores)
        except Exception:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'traceback.txt').write_text(traceback.format_exc())
            raise


def run_jobs(jobs: list[Job], paths: dict[str, Path], output: Path, workers: int) -> None:
    unfinished = [job for job in jobs if not complete(job.directory(output))]
    configs = {}
    # B and C share each bundle's config; finish writes before either runner reads.
    for job in unfinished:
        if job.condition != 'A' and job.bundle not in configs:
            configs[job.bundle] = write_config(job, paths, output)
    pending = iter(unfinished)
    failures = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1,
    ) as executor:
        active = {}
        while True:
            while not failures and len(active) < workers:
                job = next(pending, None)
                if job is None:
                    break
                config = configs.get(job.bundle)
                active[executor.submit(run_job, job, paths, output, config)] = job
            if not active:
                break
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                job = active.pop(future)
                try:
                    future.result()
                except Exception:  # noqa: BLE001 — drain active jobs before raising the batch failure.
                    failure = {'job': str(job.directory(output).relative_to(output)),
                               'traceback': traceback.format_exc()}
                    failures.append(failure)
                    grid.write_json(output / 'failures.json.gz', failures)
                print(f'{job.directory(output).relative_to(output)}: '
                      f'{"failed" if future.exception() is not None else "complete"}', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} job(s) failed; see {output / "failures.json.gz"}')
    (output / 'failures.json.gz').unlink(missing_ok=True)


def parity_check(output: Path, references: Path) -> dict[str, Any]:
    checks = {}
    for condition in ('B', 'C'):
        actual = grid.read_json(output / condition / 'final40/shipped/seed-0/scores.json.gz')['by_tolerance']
        name = str(CONDITIONS[condition])
        reference = grid.read_json(references / name / 'scores.json.gz')['by_tolerance']
        differences = {}
        # Compare counts exactly; rates are derived from them by the scorers.
        for tolerance in TOLERANCES:
            for field, expected in reference[tolerance].items():
                if isinstance(expected, int) and actual[tolerance].get(field) != expected:
                    differences[f'{tolerance}/{field}'] = {'expected': expected, 'actual': actual[tolerance].get(field)}
        checks[condition] = {'reference': name, 'passed': not differences, 'differences': differences}
    return checks


def seed_statistics(score_paths: list[Path]) -> dict[str, Any]:
    pooled = []
    for path in score_paths:
        scores = grid.read_json(path)
        validate_scores(scores)
        pooled.append(grid.pooled_scores(scores['by_video']))
    statistics = {}
    for tolerance in TOLERANCES:
        statistics[tolerance] = {}
        for metric in METRICS:
            values = [score[tolerance][metric] for score in pooled]
            statistics[tolerance][metric] = {'mean': mean(values), 'min': min(values), 'max': max(values)}
    return statistics


def summarise(bundles: list[Bundle], output: Path, references: Path | None = None) -> dict[str, Any]:
    rows = []
    for condition in CONDITIONS:
        for stage in ('v32', 'final40'):
            for label in CONFIGURATIONS:
                selected = [bundle for bundle in bundles if (bundle.stage, bundle.label) == (stage, label)]
                if [bundle.seed_offset for bundle in selected] != list(final_test.repeats.SEED_OFFSETS):
                    raise ValueError('summary requires all five seeds in order')
                score_paths = []
                for bundle in selected:
                    directory = Job(condition, bundle).directory(output)
                    if not complete(directory):
                        raise ValueError(f'{directory}: incomplete job')
                    score_paths.append(directory / 'scores.json.gz')
                rows.append({'condition': condition, 'stage': stage, 'label': label,
                             'seed_offsets': list(final_test.repeats.SEED_OFFSETS),
                             'by_tolerance': seed_statistics(score_paths)})
    parity = parity_check(output, references or BENCHMARK / 'benchmark_results/comparison')
    summary = {'schema': SCHEMA, 'results': rows, 'parity': parity}
    grid.write_json(output / 'summary.json.gz', summary)
    print_table(summary)
    if not all(check['passed'] for check in parity.values()):
        raise ValueError('production parity failed; see summary.json.gz')
    return summary


def print_table(summary: dict[str, Any]) -> None:
    print('| Condition | Stage | Configuration | ±frames | Contact F1 | Correct-player F1 | '
          'Fully correct rallies | Fully correct % |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for row in summary['results']:
        for tolerance in TOLERANCES:
            cells = []
            for metric in METRICS:
                values = row['by_tolerance'][tolerance][metric]
                scale = 100 if metric in ('f1', 'attributed_f1') else 1
                cells.append(f'{scale * values["mean"]:.2f} [{scale * values["min"]:.2f}, '
                             f'{scale * values["max"]:.2f}]')
            print(f'| {row["condition"]} | {row["stage"]} | {row["label"]} | {tolerance} | '
                  + ' | '.join(cells) + ' |')
    print('F1 columns are percentages. Cells show mean [min, max] over five seeds.')
    print('Production parity: ' + ', '.join(
        f'{condition}={"PASS" if check["passed"] else "FAIL"}' for condition, check in summary['parity'].items()
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True, help='TOML with the four existing test path keys')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=28, help='single-threaded bundle/condition jobs (default: 28)')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    paths = grid.load_paths(args.paths)
    for name in ('test_inputs', 'test_courts', 'inpaint_root', 'pose_root'):
        if not paths[name].is_dir():
            raise NotADirectoryError(paths[name])
    load_manifest(paths['test_inputs'])
    bundles = discover_bundles()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A failed rerun must never leave an older successful summary in place.
    (output / 'summary.json.gz').unlink(missing_ok=True)
    reuse_final40(bundles, output)
    run_jobs(build_jobs(bundles), paths, output, args.workers)
    summarise(bundles, output)


if __name__ == '__main__':
    main()
