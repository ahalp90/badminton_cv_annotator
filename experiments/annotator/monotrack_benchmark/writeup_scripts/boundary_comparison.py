"""Run the three-policy benchmark after retrain_boundaries.py finishes.

Generated bundles, tables and logs stay below --output. Completed jobs are reused
when their commands and required outputs agree. Inference retains saved predictions
and shares preparation and compatible rally options across policies.

Policy selection reads validation streams before test metrics are computed.
Use --plan-only to inspect commands without loading models or writing results.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shlex
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
SCRIPTS = REPO / 'experiments/annotator/monotrack_benchmark/writeup_scripts'
SEEDS = (0, 100, 200, 300, 400)
POLICIES = {'2_3': (2, 3), '3_3': (3, 3), '5_5': (5, 5)}
STAGES = ('fit40', 'reuse40', 'inference', 'select-policy', 'metrics', 'surveys')
SELECTION_PROTOCOL = {
    'split': 'validation',
    'evaluation_policy': {'before_seconds': 2.0, 'after_seconds': 3.0},
    'objective': 'maximise mean fully-correct labelled-rally coverage across all five seeds',
    'confidence_cutoff': None,
    'population': 'all found validation rallies, equivalent to keeping the lowest attainable confidence threshold',
    'tie_order': list(POLICIES),
    'test_results_used': False,
}


@dataclass(frozen=True)
class Job:
    stage: str
    label: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    cores: int


def read_json(path: Path) -> Any:
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def lock_record(path: Path, value: Any) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'{path}: inputs or selection protocol changed; use a fresh --output')
    else:
        write_json(path, value)


def normalised_paths(source: Path) -> dict[str, Path]:
    with source.open('rb') as handle:
        configured = tomllib.load(handle)['paths']
    paths = {}
    for name, value in configured.items():
        candidate = Path(value).expanduser()
        paths[name] = candidate.resolve() if candidate.is_absolute() else (REPO / candidate).resolve()
    if 'test_courts' not in paths and 'test_inputs' in paths:
        paths['test_courts'] = paths['test_inputs'] / 'discovery'
    required = ('stages', 'development_courts', 'test_courts', 'inpaint_root', 'pose_root')
    missing = [name for name in required if name not in paths]
    if missing:
        raise ValueError(f'{source}: missing inference [paths] keys: {missing}')
    paths.setdefault('shots_master', REPO / 'training/data/shuttleset/annotations/shots_master.csv')
    return paths


def check_sources(paths: dict[str, Path], args: argparse.Namespace) -> None:
    """Public runners use fixed label locations; reject divergent source copies."""
    shots_master = REPO / 'training/data/shuttleset/annotations/shots_master.csv'
    if paths['shots_master'].read_bytes() != shots_master.read_bytes():
        raise ValueError('configured shots_master differs from the public runners\' repository label file')
    inputs = REPO / 'experiments/annotator/monotrack_benchmark/benchmark_results/inputs'
    manifest = read_json(inputs / 'manifest.json.gz')
    if 'test_inputs' in paths and read_json(paths['test_inputs'] / 'manifest.json.gz') != manifest:
        raise ValueError('configured test_inputs differ from the public runners\' benchmark manifest')
    for name in ('stages', 'development_courts', 'test_courts', 'inpaint_root', 'pose_root'):
        if not paths[name].is_dir():
            raise NotADirectoryError(f'{name}: {paths[name]}')
    selected = read_json(args.monotrack_runs / 'selected.json.gz')
    if selected['selection_split'] != 'validation' or selected['seeds'] != list(SEEDS):
        raise ValueError('MonoTrack comparator must contain all five validation-selected seeds')
    if not (args.training_root / 'prepared/raw_baseline/preparation.json.gz').is_file():
        raise FileNotFoundError(args.training_root / 'prepared/raw_baseline/preparation.json.gz')


def inference_paths(args: argparse.Namespace) -> Path:
    return args.output / 'inference_paths.toml'


def models40(args: argparse.Namespace, policy: str) -> Path:
    return args.output / 'models_40' / policy


def annotator_runs(args: argparse.Namespace, policy: str) -> Path:
    return args.output / 'annotator_runs' / policy


def pair_output(args: argparse.Namespace, training: str, evaluation: str) -> Path:
    return args.output / 'metrics' / f'train-{training}__eval-{evaluation}'


def command(script: str, *arguments: object) -> tuple[str, ...]:
    return (sys.executable, str(SCRIPTS / script), *(str(argument) for argument in arguments))


def internal_command(args: argparse.Namespace, action: str, *arguments: object) -> tuple[str, ...]:
    return (
        sys.executable, str(Path(__file__).resolve()), '--training-root', str(args.training_root),
        '--paths', str(args.paths), '--output', str(args.output), '--monotrack-runs', str(args.monotrack_runs),
        '--internal-action', action, *(str(argument) for argument in arguments),
    )


def stage_jobs(args: argparse.Namespace, stage: str) -> list[Job]:
    jobs = []
    if stage == 'fit40':
        for seed in SEEDS:
            output = models40(args, '2_3') / f'seed-{seed}'
            jobs.append(Job(stage, f'seed-{seed}', command(
                'fit_contact_40.py', '--prepared', args.training_root / 'prepared/raw_baseline',
                '--tree-results', args.training_root / '2_3', '--out', models40(args, '2_3'),
                '--device', args.device, '--seeds', seed,
            ), (output / 'models.joblib', output / 'metadata.json', output / 'contact-fit.json.gz'), 2))
    elif stage == 'reuse40':
        for policy in ('3_3', '5_5'):
            for seed in SEEDS:
                output = models40(args, policy) / f'seed-{seed}'
                jobs.append(Job(stage, f'{policy}-seed-{seed}', internal_command(
                    args, 'reuse40', '--policy', policy, '--seed', seed,
                ), (output / 'models.joblib', output / 'metadata.json', output / 'contact-fit.json.gz'), 1))
    elif stage == 'inference':
        expected = []
        for policy in POLICIES:
            output = annotator_runs(args, policy)
            for seed in SEEDS:
                expected.append(output / 'validation/32_videos' / f'seed-{seed}' / 'execution.json.gz')
                for condition in ('discovery', 'our_courts', 'monotrack_courts'):
                    for family in ('32_videos', '40_videos'):
                        expected.append(output / 'test' / condition / family / f'seed-{seed}' / 'execution.json.gz')
                expected.append(output / 'test/our_courts/body_window_36' / f'seed-{seed}' / 'execution.json.gz')
        jobs.append(Job(stage, 'all-policies', command(
            'annotator_batch_runs.py', '--paths', inference_paths(args), '--training-root', args.training_root,
            '--models-40', args.output / 'models_40', '--out', args.output / 'annotator_runs',
            '--workers', args.inference_workers,
        ), tuple(expected), 2 * args.inference_workers))
    elif stage == 'select-policy':
        jobs.append(Job(stage, 'validation', internal_command(args, 'select-policy'),
                        (args.output / 'policy_selection.json.gz',), 1))
    else:
        for training in POLICIES:
            for evaluation, (before, after) in POLICIES.items():
                output = pair_output(args, training, evaluation)
                boundary_args = ('--boundary-before-seconds', before, '--boundary-after-seconds', after)
                if stage == 'metrics':
                    selected_command = command(
                        'metrics.py', '--annotator-runs', annotator_runs(args, training),
                        '--monotrack-runs', args.monotrack_runs, '--workers', args.metrics_workers,
                        '--out', output, *boundary_args,
                    )
                    expected = (output / 'contact_summary.json', output / 'evaluation-policy.json.gz')
                    cores = args.metrics_workers
                else:
                    selected_command = command(
                        'confidence_cutoffs.py', '--validation', annotator_runs(args, training) / 'validation/32_videos',
                        '--metrics', output, '--shots-master', args.shots_master, '--out', output, *boundary_args,
                    )
                    expected = (output / 'confidence_cutoffs.json.gz', output / 'validation_confidence_curve.csv.gz',
                                output / 'test_confidence_curve.csv.gz')
                    cores = 1
                jobs.append(Job(stage, f'train-{training}__eval-{evaluation}', selected_command, expected, cores))
    return jobs


def run_job(args: argparse.Namespace, job: Job) -> None:
    receipt = args.output / 'completed' / job.stage / f'{job.label}.json.gz'
    record = {'command': list(job.command), 'outputs': [str(path) for path in job.outputs], 'cores': job.cores}
    if receipt.exists():
        if read_json(receipt) != record:
            raise ValueError(f'{receipt}: job changed; use a fresh --output')
        if not all(path.is_file() for path in job.outputs):
            raise FileNotFoundError(f'{receipt}: a completed job output is missing')
        print(f'{job.stage}/{job.label}: already complete', flush=True)
        return
    environment = dict(os.environ)
    threads = '2' if job.stage == 'fit40' else '1'
    environment.update(OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                       BLIS_NUM_THREADS=threads, NUMEXPR_NUM_THREADS=threads,
                       VECLIB_MAXIMUM_THREADS=threads)
    environment['PYTHONPATH'] = os.pathsep.join((str(REPO), str(REPO / 'src')))
    log_path = args.output / 'logs' / job.stage / f'{job.label}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'{job.stage}/{job.label}: running; log {log_path}', flush=True)
    with log_path.open('w') as log:
        log.write(shlex.join(job.command) + '\n')
        log.flush()
        result = subprocess.run(
            job.command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode:
        raise RuntimeError(f'{job.stage}/{job.label}: exit {result.returncode}; see {log_path}')
    missing = [str(path) for path in job.outputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'{job.stage}/{job.label}: command returned success with missing outputs: {missing}')
    write_json(receipt, record)
    print(f'{job.stage}/{job.label}: complete', flush=True)


def reuse_contact_tree(args: argparse.Namespace) -> None:
    from annotator.models import load_models, save_models

    source40 = models40(args, '2_3') / f'seed-{args.seed}'
    source32 = args.training_root / '2_3/models' / f'raw_baseline-seed-{args.seed}'
    policy32 = args.training_root / args.policy / 'models' / f'raw_baseline-seed-{args.seed}'
    original, trained, own = load_models(source32), load_models(source40), load_models(policy32)
    if own.preprocessing != original.preprocessing or own.contact_settings != original.contact_settings:
        raise ValueError('contact reuse requires identical preprocessing and contact settings across policies')
    if own.contact.get_booster().save_raw() != original.contact.get_booster().save_raw():
        raise ValueError('policy contact tree differs from the shared 32-video contact fit')
    output = models40(args, args.policy) / f'seed-{args.seed}'
    save_models(replace(own, contact=trained.contact), output)
    write_json(output / 'contact-fit.json.gz', {
        'seed_offset': args.seed, 'source_contact_bundle': str(source40), 'source_selector_review_bundle': str(policy32),
        'contact_refitted': False, 'selector_and_review_refitted': False,
        'shared_fit': read_json(source40 / 'contact-fit.json.gz'),
    })


def select_policy(args: argparse.Namespace) -> None:
    from annotator.training.sequences import RallyBoundaryPolicy
    from experiments.annotator.monotrack_benchmark.writeup_scripts.confidence_cutoffs import (
        validation_found,
    )

    comparison = {}
    labelled_totals = set()
    for policy in POLICIES:
        found, labelled = validation_found(
            annotator_runs(args, policy) / 'validation/32_videos', args.shots_master, RallyBoundaryPolicy(2, 3),
        )
        if set(found['seed']) != set(SEEDS):
            raise ValueError(f'{policy}: validation must include all five seeds')
        correct = found[found['outcome'] == 'fully_correct'].drop_duplicates(['seed', 'video', 'rally_id'])
        counts = correct.groupby('seed').size().reindex(SEEDS, fill_value=0)
        total_correct = int(counts.sum())
        comparison[policy] = {
            'labelled_rallies': labelled, 'fully_correct_labelled_rallies_per_seed': counts.tolist(),
            'fully_correct_labelled_rallies_seed_sum': total_correct,
            'mean_fully_correct_labelled_rally_coverage_percent': 100 * total_correct / (len(SEEDS) * labelled),
            'lowest_validation_score': float(found['confidence'].min()),
        }
        labelled_totals.add(labelled)
    if len(labelled_totals) != 1:
        raise ValueError('validation labelled-rally population differs between training policies')
    # With one common denominator, integer totals preserve exact ties between policies.
    chosen = max(POLICIES, key=lambda policy: comparison[policy]['fully_correct_labelled_rallies_seed_sum'])
    record = {'protocol': SELECTION_PROTOCOL, 'selected_training_policy': chosen, 'comparison': comparison}
    lock_record(args.output / 'policy_selection.json.gz', record)
    print(json.dumps(record, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, required=True, help='completed retrain_boundaries.py --output')
    parser.add_argument('--paths', type=Path, required=True, help='training/inference [paths] TOML')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--monotrack-runs', type=Path, required=True, help='five-seed validation-selected comparator')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--inference-workers', type=int, default=14)
    parser.add_argument('--metrics-workers', type=int, default=14)
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--internal-action', choices=('reuse40', 'select-policy'), help=argparse.SUPPRESS)
    parser.add_argument('--policy', choices=('3_3', '5_5'), help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, choices=SEEDS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.inference_workers <= 14 or not 1 <= args.metrics_workers <= 28:
        parser.error('inference workers must be 1..14 and metrics workers 1..28 (aggregate limit: 28 cores)')
    for name in ('training_root', 'paths', 'output', 'monotrack_runs'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.configured_paths = normalised_paths(args.paths)
    args.shots_master = args.configured_paths['shots_master']
    return args


def main() -> None:
    args = parse_args()
    paths = args.configured_paths
    if args.internal_action:
        sys.path[:0] = [str(REPO), str(REPO / 'src')]
        if args.internal_action == 'reuse40':
            reuse_contact_tree(args)
        else:
            select_policy(args)
        return
    if not args.plan_only:
        check_sources(paths, args)
        lock_record(args.output / 'selection_protocol.json.gz', SELECTION_PROTOCOL)
        lock_record(args.output / 'inputs.json.gz', {
            'training_root': str(args.training_root), 'paths': {name: str(value) for name, value in paths.items()},
            'monotrack_runs': str(args.monotrack_runs), 'device': args.device,
        })
        with inference_paths(args).open('w') as handle:
            handle.write('[paths]\n')
            for name in ('test_courts', 'inpaint_root', 'pose_root', 'stages', 'development_courts'):
                handle.write(f'{name} = {json.dumps(str(paths[name]))}\n')
    for stage in STAGES:
        if stage not in args.stages:
            continue
        jobs = stage_jobs(args, stage)
        parallel = 5 if stage == 'fit40' else 1
        print(f'{stage}: {len(jobs)} jobs; {parallel} concurrent; '
              f'at most {parallel * max(job.cores for job in jobs)} CPU cores', flush=True)
        if args.plan_only:
            for job in jobs:
                print(shlex.join(job.command), flush=True)
            continue
        if stage in ('metrics', 'surveys') and not (args.output / 'policy_selection.json.gz').is_file():
            raise FileNotFoundError('run select-policy before reading test metrics or surveys')
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            list(executor.map(lambda job: run_job(args, job), jobs))


if __name__ == '__main__':
    main()
