"""Run a small parallel rally-stack sweep with incremental per-configuration results."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(REPOSITORY := Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPOSITORY / 'src'))

from experiments.annotator.heuristic_only_tune.run import read_json, write_json

# isort: split

from annotator.training.trees import TreeBackend


def configurations(backend: TreeBackend) -> list[dict[str, Any]]:
    runs = [{'name': 'legacy-pool', 'leaves': 15, 'layers': 3, 'min_child_weight': 5.0, 'legacy_pool': True}]
    runs.extend([
        {'name': f'leaves-{leaves}', 'leaves': leaves, 'layers': 3, 'min_child_weight': 5.0, 'legacy_pool': False}
        for leaves in (15, 7, 31)
    ])
    if backend == TreeBackend.XGBOOST:
        runs.append({'name': 'child-10', 'leaves': 15, 'layers': 3, 'min_child_weight': 10.0, 'legacy_pool': False})
    return runs


def run_configuration(
    run: dict[str, Any], pools: Path, output: Path, backend: TreeBackend, device: str, threads: int,
) -> dict[str, Any]:
    destination = output / run['name']
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / 'result.json.gz'
    if result_path.exists() and read_json(result_path)['complete']:
        return {'name': run['name'], 'exit_code': 0, 'resumed_complete': True}
    command = [
        sys.executable, '-u', '-m', 'experiments.annotator.tree_simplification.train',
        '--pools', str(pools), '--output', str(destination), '--backend', backend.value, '--device', device,
        '--threads', str(threads), '--leaves', str(run['leaves']), '--layers', str(run['layers']),
        '--min-child-weight', str(run['min_child_weight']),
    ]
    if run['legacy_pool']:
        command.append('--legacy-pool')
    environment = {**os.environ, 'OMP_NUM_THREADS': str(threads), 'OPENBLAS_NUM_THREADS': str(threads),
                   'MKL_NUM_THREADS': str(threads)}
    started = time.perf_counter()
    write_json(destination / 'status.json.gz', {'state': 'running', 'command': command})
    with (destination / 'run.log').open('a') as log:
        completed = subprocess.run(
            command, cwd=REPOSITORY, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    record = {'name': run['name'], 'exit_code': completed.returncode, 'wall_s': time.perf_counter() - started}
    write_json(destination / 'status.json.gz', record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pools', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=list(TreeBackend), default=TreeBackend.XGBOOST)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1 or args.threads < 1 or args.jobs * args.threads > 28:
        parser.error('jobs and threads must be positive and use at most 28 CPU cores together')
    backend = TreeBackend(args.backend)
    if backend == TreeBackend.HIST_GRADIENT_BOOSTING and args.device != 'cpu':
        parser.error('HGB requires --device cpu')
    runs = configurations(backend)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        'pools': str(args.pools.resolve()), 'pool_settings': read_json(args.pools / 'pools.json.gz'),
        'backend': backend.value, 'device': args.device, 'configurations': runs,
    }
    plan_path = args.output / 'plan.json.gz'
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError('Search settings changed; use a fresh output directory')
    write_json(plan_path, plan)
    completed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(
            run_configuration, run, args.pools.resolve(), args.output.resolve(), backend, args.device, args.threads,
        ) for run in runs]
        for future in as_completed(futures):
            record = future.result()
            completed.append(record)
            print(record, flush=True)
            write_json(args.output / 'status.json.gz', {'completed': completed, 'planned': len(runs)})
    if any(record['exit_code'] for record in completed):
        raise SystemExit('Some configurations failed; inspect their run.log and rerun to resume')


if __name__ == '__main__':
    main()
