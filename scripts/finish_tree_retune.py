"""Finish an existing preprocessing grid with bounded concurrency and a memory guard."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def read_json(path: Path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def write_state(path: Path, record: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def process_alive(pid: int) -> bool:
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def worker_processes(parent: int) -> list[dict]:
    rows = subprocess.check_output(['ps', '-ww', '-eo', 'pid=,ppid=,stat=,cputimes=,rss=,args='], text=True)
    workers = []
    for line in rows.splitlines():
        pid, ppid, state, cpu, rss, command = line.split(None, 5)
        if int(ppid) == parent and 'multiprocessing.spawn' in command:
            workers.append({'pid': int(pid), 'paused': 'T' in state, 'cpu_s': int(cpu), 'rss_kib': int(rss)})
    return workers


def available_memory_gib() -> float:
    line = next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    return int(line.split()[1]) / 1048576


def controls(workers: list[dict], available: float, maximum: int) -> tuple[list[int], list[int]]:
    """Keep a few advancing fits when memory is tight; resume one at a time."""
    active = sorted((row for row in workers if not row['paused']), key=lambda row: row['cpu_s'], reverse=True)
    paused = sorted((row for row in workers if row['paused']), key=lambda row: row['cpu_s'], reverse=True)
    target = min(maximum, 1 if available < 12 else 4 if available < 32 else len(active))
    stop = [row['pid'] for row in active[target:]]
    resume = []
    # Paused workers retain their allocations. Staggering their next allocations
    # lets the most advanced fits finish and release memory before others grow.
    if paused and len(active) < maximum and (available > 56 or (not active and available >= 32)):
        resume = [paused[0]['pid']]
    return stop, resume


def process_identity(pid: int) -> tuple[str, str] | None:
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return fields[19], fields[0]
    except FileNotFoundError:
        return None


def monitor(parent: int, state: dict, state_path: Path, maximum: int, process=None) -> int | None:
    state.update(training_pid=parent, status='running')
    last_record = 0.0
    known = {}
    try:
        while process_alive(parent):
            workers = worker_processes(parent)
            for worker in workers:
                identity = process_identity(worker['pid'])
                if identity is not None:
                    known[worker['pid']] = identity[0]
            available = available_memory_gib()
            stop, resume = controls(workers, available, maximum)
            for action, pids in ((signal.SIGSTOP, stop), (signal.SIGCONT, resume)):
                for pid in pids:
                    try:
                        os.kill(pid, action)
                    except ProcessLookupError:
                        pass
            if stop or resume or time.monotonic() - last_record >= 15:
                state.update(
                    updated_utc=datetime.now(UTC).isoformat(), available_gib=available,
                    worker_count=len(workers), paused_workers=[row['pid'] for row in workers if row['paused']],
                    last_controls={'paused': stop, 'resumed': resume},
                )
                write_state(state_path, state)
                last_record = time.monotonic()
            time.sleep(2)
        surviving = [pid for pid, birth in known.items()
                     if (identity := process_identity(pid)) is not None and identity[0] == birth and identity[1] != 'Z']
        if surviving:
            raise RuntimeError(f'training parent exited with surviving workers: {surviving}')
    except BaseException:
        surviving = {pid: identity[1] for pid, birth in known.items()
                     if (identity := process_identity(pid)) is not None and identity[0] == birth and identity[1] != 'Z'}
        paused = [pid for pid, status in surviving.items() if 'T' in status]
        slots = max(0, maximum - (len(surviving) - len(paused)))
        # Resume only verified workers within both limits. If recovery needs
        # intervention, retain their identities for the failure alert.
        resumed = []
        if available_memory_gib() >= 32:
            for pid in paused[:slots]:
                try:
                    os.kill(pid, signal.SIGCONT)
                    resumed.append(pid)
                except ProcessLookupError:
                    pass
        state['paused_after_guard_error'] = [pid for pid in paused if pid not in resumed]
        state['surviving_workers_after_guard_error'] = list(surviving)
        raise
    return process.wait() if process is not None else None


def structure(stream: dict) -> dict:
    return {
        'events': [[row[0], row[2]] for row in stream['events']],
        'sequences': [
            {**{key: value for key, value in sequence.items() if key != 'events'},
             'events': [[row[0], row[2]] for row in sequence['events']]}
            for sequence in stream['sequences']
        ],
    }


def verify_baseline(output: Path, previous: Path) -> dict:
    baseline = next(path.parent for path in output.glob('trials/*/result.json.gz') if read_json(path)['label'] == 'shipped')
    result = read_json(baseline / 'result.json.gz')
    identities = sorted(result['per_video'])
    if len(identities) != 32:
        raise ValueError('baseline must cover all 32 original training videos')
    exact, differences = 0, []
    for identity in identities:
        before = read_json(previous / f'{identity}.json.gz')
        after = read_json(baseline / 'training_streams' / f'{identity}.json.gz')
        exact += before == after
        if structure(before) != structure(after):
            differences.append(identity)
    report = {'videos': 32, 'exact_streams': exact, 'different_selections': differences, 'baseline': str(baseline)}
    with gzip.open(output / 'baseline-parity.json.gz', 'wt') as stream:
        json.dump(report, stream)
    if differences:
        raise ValueError(f'baseline selections differ on {differences}; remaining work requires inspection')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--initial-pid', type=int, required=True)
    parser.add_argument('--initial-supervisor-pid', type=int)
    parser.add_argument('--previous-streams', type=Path, required=True)
    parser.add_argument('--test-paths', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.workers <= 20:
        parser.error('workers must be between one and twenty')
    root, output = args.run_root, args.run_root / 'grid48'
    state_path = root / 'automation.json'
    state = {
        'phase': 'first_batch', 'status': 'running', 'guard_pid': os.getpid(),
        'started_utc': datetime.now(UTC).isoformat(), 'max_workers': args.workers,
        'automatic_time_kill': False,
    }
    write_state(state_path, state)
    try:
        monitor(args.initial_pid, state, state_path, args.workers)
        if (output / 'failure.json.gz').exists():
            raise RuntimeError(f'initial training failed: {read_json(output / "failure.json.gz")}')
        if len(list(output.glob('trials/*/result.json.gz'))) < 26:
            raise RuntimeError('initial training exited before all 26 submitted fits completed')
        # The old timeout was stopped to revoke its automatic kill. Its training
        # child has now exited; remove only this inert supervisor, never a fit.
        if args.initial_supervisor_pid and process_alive(args.initial_supervisor_pid):
            command = Path(f'/proc/{args.initial_supervisor_pid}/cmdline').read_bytes()
            if not command.startswith(b'timeout\x00'):
                raise RuntimeError('initial supervisor identity changed')
            os.kill(args.initial_supervisor_pid, signal.SIGKILL)
        state.update(phase='baseline_check', initial_training_success=True)
        write_state(state_path, state)
        state['baseline_parity'] = verify_baseline(output, args.previous_streams)
        environment = dict(os.environ, PYTHONPATH=f'{args.repository / "src"}:{args.repository}')
        for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            environment[variable] = '1'
        commands = [
            ('remaining_grid', [
                sys.executable, '-m', 'experiments.annotator.tree_hp_tune.run', 'run',
                '--paths', str(root / 'paths.toml'),
                '--candidates', str(args.repository / 'experiments/annotator/tree_hp_tune/grid-48.json.gz'),
                '--output', str(output), '--workers', str(args.workers), '--threads', '1', '--hours', 'inf',
            ]),
            ('seed_repeats', [
                sys.executable, str(args.repository / 'scripts/tree_retune_repeats.py'),
                '--run-root', str(root), '--repository', str(args.repository), '--workers', str(args.workers),
            ]),
            ('final_test', [
                sys.executable, str(args.repository / 'scripts/tree_retune_final_test.py'),
                '--run-root', str(root), '--test-paths', str(args.test_paths), '--workers', str(args.workers),
            ]),
        ]
        for phase, command in commands:
            state.update(phase=phase)
            with (root / f'{phase}.log').open('a') as log:
                process = subprocess.Popen(command, cwd=args.repository, env=environment, stdout=log, stderr=subprocess.STDOUT)
                code = monitor(process.pid, state, state_path, args.workers, process)
            if code:
                raise RuntimeError(f'{phase} exited {code}; see {phase}.log')
        selected = read_json(root / 'repeats' / 'selected.json.gz')
        state.update(status='complete', phase='complete',
                     selected_id=selected['config_id'], selected_label=selected['label'])
    except BaseException as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        state['updated_utc'] = datetime.now(UTC).isoformat()
        write_state(state_path, state)


if __name__ == '__main__':
    main()
