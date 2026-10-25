"""Retrain the annotator for each boundary policy using the maintained stages.

Prepared vision inputs and freshly fitted contact trees are shared between
policies. Selector targets, selector fits and confidence fits are separate.
Each policy produces the tree-results layout consumed by annotator_runs.py.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
STAGES = ('prepare', 'contacts', 'pools', 'targets', 'train', 'export')
POLICIES = {'2_3': (2, 3), '3_3': (3, 3), '5_5': (5, 5), 'no_max': (None, None)}
SEEDS = (0, 100, 200, 300, 400)
CANDIDATES = ('raw_baseline', 'body_window_36')
MODULE = 'experiments.annotator.tree_simplification'


@dataclass(frozen=True)
class Job:
    command: tuple[str, ...]
    output: Path


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as handle:
        json.dump(value, handle, indent=2)


def run_job(job: Job) -> None:
    """Let each stage validate its own resumable outputs; retain failure logs."""
    job.output.mkdir(parents=True, exist_ok=True)
    completed = job.output / 'completed-command.json.gz'
    command = list(job.command)
    if completed.exists():
        with gzip.open(completed, 'rt') as handle:
            if json.load(handle) != command:
                raise ValueError(f'{job.output}: command changed; use a fresh output directory')
        return
    environment = dict(os.environ)
    threads = command[command.index('--threads') + 1] if '--threads' in command else '1'
    environment.update(OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                       NUMEXPR_NUM_THREADS=threads, BLIS_NUM_THREADS=threads)
    with (job.output / 'stage.log').open('w') as log:
        subprocess.run(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    write_json(completed, command)
    print(f'Completed {job.output}', flush=True)


def command(stage: str, *arguments: object) -> tuple[str, ...]:
    module = 'input_stages' if stage in ('contacts', 'export') else stage
    prefix = (sys.executable, '-m', f'{MODULE}.{module}')
    if stage in ('contacts', 'export'):
        prefix = (*prefix, stage)
    return (*prefix, *(str(argument) for argument in arguments))


def stage_jobs(args: argparse.Namespace, stage: str, workers_per_job: int) -> list[Job]:
    jobs = []
    upstream = args.reuse_training_root or args.output
    if args.reuse_training_root is not None and stage in ('prepare', 'contacts'):
        return jobs
    for candidate in args.candidates:
        prepared = upstream / 'prepared' / candidate
        if stage == 'prepare':
            jobs.append(Job(command(
                stage, '--paths', args.paths, '--output', prepared,
                '--candidate', args.output / 'candidates' / f'{candidate}.json.gz', '--workers', workers_per_job,
            ), prepared))
            continue
        for seed in args.seeds:
            contacts = upstream / 'contacts' / candidate / f'seed-{seed}'
            if stage == 'contacts':
                jobs.append(Job(command(stage, '--prepared', prepared, '--output', contacts,
                                        '--seed-offset', seed, '--threads', 2, '--device', args.device), contacts))
                continue
            for policy in args.policies:
                before, after = POLICIES[policy]
                root = args.output / policy / 'test-run' / 'candidates' / candidate / f'seed-{seed}'
                pools, targets, fitted = root / 'pools', root / 'targets-five', root / 'selector'
                if stage == 'pools':
                    policy_arguments = (('--no-max-buffer',) if before is None
                                        else ('--boundary-before-seconds', before, '--boundary-after-seconds', after))
                    reuse_arguments = ()
                    if args.reuse_training_root is not None:
                        source = upstream / args.reuse_policy / 'test-run' / 'candidates' / candidate / f'seed-{seed}'
                        reuse_arguments = ('--reuse-options', source / 'pools')
                    jobs.append(Job(command(
                        stage, '--prepared', prepared, '--contact-run', contacts, '--output', pools,
                        '--workers', workers_per_job,
                        *policy_arguments, *reuse_arguments,
                    ), pools))
                elif stage == 'targets':
                    jobs.append(Job(command(stage, '--pools', pools, '--output', targets,
                                            '--workers', workers_per_job), targets))
                elif stage == 'train':
                    jobs.append(Job(command(
                        stage, '--pools', pools, '--five-frame-targets', targets, '--output', fitted,
                        '--device', args.device, '--threads', 2, '--seed', 20260905 + seed,
                        '--leaves', 31, '--rounds', 400, '--layers', 1,
                        '--learning-rate', .05, '--min-child-weight', 5, '--l2', 1,
                    ), fitted))
                else:
                    exported = root / 'export'
                    reuse_arguments = ('--reuse-upstream',) if args.reuse_training_root is not None else ()
                    jobs.append(Job(command(stage, '--run', fitted, '--five-frame-targets', targets,
                                            '--output', exported, '--threads', 2, '--device', args.device,
                                            *reuse_arguments), exported))
    return jobs


def prepare_candidates(args: argparse.Namespace) -> None:
    for candidate in args.candidates:
        overrides = {'body_unit_half_window': 36} if candidate == 'body_window_36' else {}
        write_json(args.output / 'candidates' / f'{candidate}.json.gz', {
            'schema': 'annotator-boundary-comparison/1', 'stage': 'fixed-preprocessing',
            'candidates': [{'label': candidate, 'overrides_base30': overrides, 'strategies': {}}],
        })


def collect_bundles(args: argparse.Namespace) -> None:
    upstream = args.reuse_training_root or args.output
    for policy in args.policies:
        for candidate in args.candidates:
            for seed in args.seeds:
                root = args.output / policy / 'test-run' / 'candidates' / candidate / f'seed-{seed}'
                destination = args.output / policy / 'models' / f'{candidate}-seed-{seed}'
                shutil.copytree(root / 'export' / 'bundle', destination, dirs_exist_ok=True)
                contacts = root / 'contacts-fit'
                if not contacts.exists():
                    contacts.symlink_to(upstream / 'contacts' / candidate / f'seed-{seed}', target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=28)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--policies', nargs='+', choices=POLICIES, default=list(POLICIES))
    parser.add_argument('--candidates', nargs='+', choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--reuse-training-root', type=Path,
                        help='Reuse preparation, contacts and proposals from an existing training run')
    parser.add_argument('--reuse-policy', choices=POLICIES, default='5_5', help='Saved proposal policy to relabel')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if not 2 <= args.workers <= 28:
        parser.error('--workers must be between 2 and 28')
    args.output, args.paths = args.output.resolve(), args.paths.resolve()
    if args.reuse_training_root is not None:
        args.reuse_training_root = args.reuse_training_root.resolve()
    if not args.plan_only:
        record = {
            'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'paths': str(args.paths), 'policies': args.policies, 'candidates': args.candidates,
            'seeds': args.seeds, 'device': args.device,
            'reuse_training_root': str(args.reuse_training_root) if args.reuse_training_root is not None else None,
            'reuse_policy': args.reuse_policy,
        }
        receipt = args.output / 'run.json.gz'
        if receipt.exists():
            with gzip.open(receipt, 'rt') as handle:
                if json.load(handle) != record:
                    raise ValueError('Training inputs or source changed; use a fresh output directory')
        write_json(receipt, record)
        prepare_candidates(args)
    for stage in args.stages:
        if stage == 'prepare':
            workers_per_job = max(1, args.workers // len(args.candidates))
        elif stage == 'pools':
            workers_per_job = min(2, args.workers // 2)
        else:
            workers_per_job = min(4, args.workers) if stage == 'targets' else 2
        cores_per_job = workers_per_job
        if stage == 'pools' and args.reuse_training_root is None:
            cores_per_job *= 2
        parallel_jobs = max(1, args.workers // cores_per_job)
        jobs = stage_jobs(args, stage, workers_per_job)
        print(f'{stage}: {len(jobs)} jobs, up to{parallel_jobs} concurrent, {workers_per_job} workers each', flush=True)
        if args.plan_only:
            continue
        with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            list(executor.map(run_job, jobs))
    if not args.plan_only and 'export' in args.stages:
        collect_bundles(args)


if __name__ == '__main__':
    main()
