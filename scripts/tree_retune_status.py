"""Show saved progress and owned worker processes for one tree-retune run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

COUNT_FIELDS = (
    'labelled_rallies', 'labelled_contacts', 'predicted_contacts', 'matched_contacts',
    'correct_side_contacts', 'fully_correct_rallies',
)


def read_object(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise TypeError(f'{path}: expected a JSON object')
    return document


def candidate_identity(candidate: dict[str, Any]) -> str:
    # Persisted trial IDs hash these two fields, excluding the display label.
    specification = {
        'overrides_base30': candidate['overrides_base30'],
        'strategies': candidate.get('strategies', {}),
    }
    encoded = json.dumps(specification, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def result_exists(path: Path, identity: str, label: str, expected_videos: int) -> bool:
    if not path.exists():
        return False
    document = read_object(path)
    if document.get('id') != identity or document.get('label') != label:
        raise ValueError(f'{path}: result identity differs from the frozen grid')
    if not isinstance(document.get('per_video'), dict) or len(document['per_video']) != expected_videos:
        raise ValueError(f'{path}: expected counts for {expected_videos} videos')
    totals = document.get('totals')
    if not isinstance(totals, dict) or not all(isinstance(totals.get(key), dict) for key in ('5', '10')):
        raise ValueError(f'{path}: missing pooled scores')
    for scores in [totals, *document['per_video'].values()]:
        for tolerance in ('5', '10'):
            counts = scores.get(tolerance) if isinstance(scores, dict) else None
            if not isinstance(counts, dict):
                raise TypeError(f'{path}: missing counts at tolerance {tolerance}')
            for name in COUNT_FIELDS:
                value = counts.get(name)
                if type(value) is not int or value < 0:
                    raise ValueError(f'{path}: invalid {name} count at tolerance {tolerance}')
    return True


def process_rows() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        ['ps', '-ww', '-eo', 'pid=,ppid=,stat=,pcpu=,rss=,etimes=,args='], text=True,
    )
    rows = []
    for line in output.splitlines():
        pid, parent, state, cpu, rss, elapsed, command = line.split(None, 6)
        rows.append({
            'pid': int(pid), 'ppid': int(parent), 'state': state, 'cpu_percent': float(cpu),
            'rss_kib': int(rss), 'elapsed_s': int(elapsed), 'command': command,
        })
    return rows


def owned_workers(rows: list[dict[str, Any]], roots: set[int]) -> list[dict[str, Any]]:
    descendants = set(roots)
    while True:
        children = {row['pid'] for row in rows if row['ppid'] in descendants}
        expanded = descendants | children
        if expanded == descendants:
            break
        descendants = expanded
    workers = []
    for row in rows:
        if row['pid'] in descendants and 'spawn_main' in row['command']:
            workers.append({**row, 'status': 'paused' if 'T' in row['state'] else 'running'})
    return sorted(workers, key=lambda row: row['pid'])


def available_memory_kib(path: Path = Path('/proc/meminfo')) -> int:
    for line in path.read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])
    raise ValueError(f'{path}: MemAvailable is absent')


def repeat_status(run_root: Path, now: float, plan: dict[str, Any] | None = None) -> dict[str, Any] | None:
    root = run_root / 'repeats'
    progress_path = root / 'progress.json.gz'
    plan_path = root / 'plan.json.gz'
    if not progress_path.exists() and not plan_path.exists():
        return None
    progress = read_object(progress_path) if progress_path.exists() else {'phase': 'planned'}
    unfinished, queued = [], []
    if plan_path.exists():
        if plan is None:
            plan = read_object(plan_path)
        for offset in plan['seed_offsets']:
            seed_root = run_root / 'grid48' if offset == 0 else root / f'seed-{offset}'
            for candidate in plan['candidates']:
                identity, label = candidate['id'], candidate['label']
                trial = seed_root / 'trials' / identity
                fitted = result_exists(trial / 'result.json.gz', identity, label, 32)
                trial_progress = trial / 'progress.json.gz'
                display = f'{label} seed+{offset}'
                if not fitted and trial_progress.exists():
                    record = read_object(trial_progress)
                    unfinished.append({
                        'id': identity, 'label': display, 'seed_offset': offset, 'stage': record['stage'],
                        'progress_age_s': max(0, now - trial_progress.stat().st_mtime),
                        'video': record.get('video'), 'fit': record.get('fit'),
                    })
                elif not fitted:
                    queued.append(display)
                elif progress['phase'] == 'repeat_validation':
                    validated = result_exists(trial / 'validation.json.gz', identity, label, 8)
                    if not validated:
                        streams = list((trial / 'validation_streams').glob('*.json.gz'))
                        timestamps = [path.stat().st_mtime for path in streams]
                        if trial_progress.exists():
                            timestamps.append(trial_progress.stat().st_mtime)
                        unfinished.append({
                            'id': identity, 'label': display, 'seed_offset': offset, 'stage': 'validation',
                            'progress_age_s': max(0, now - max(timestamps)) if timestamps else None,
                            'completed_video_streams': len(streams),
                        })
    selected_path = root / 'selected.json.gz'
    selected = read_object(selected_path) if selected_path.exists() else None
    return {
        **progress, 'unfinished_trials': unfinished, 'queued_labels': queued,
        'selected_label': selected['label'] if selected else None,
    }


def final_test_status(run_root: Path, now: float) -> dict[str, Any] | None:
    root = run_root / 'final_test'
    progress_path = root / 'progress.json.gz'
    if not progress_path.exists():
        return None
    progress = read_object(progress_path)
    unfinished, queued = [], []
    plan_path = root / 'plan.json.gz'
    if plan_path.exists():
        for candidate in read_object(plan_path)['candidates']:
            for bundle in candidate['bundles']:
                offset = bundle['seed_offset']
                trial = root / 'trials' / candidate['id'] / f'seed-{offset}'
                if result_exists(trial / 'result.json.gz', candidate['id'], candidate['label'], 46):
                    continue
                label = f"final test {candidate['label']} seed+{offset}"
                trial_progress = trial / 'progress.json.gz'
                if trial_progress.exists():
                    record = read_object(trial_progress)
                    unfinished.append({
                        'id': candidate['id'], 'label': label, 'seed_offset': offset, 'stage': record['stage'],
                        'progress_age_s': max(0, now - trial_progress.stat().st_mtime),
                        'video': record.get('video'), 'fit': record.get('fit'),
                        'items_complete': record.get('complete'), 'items_planned': record.get('planned'),
                    })
                else:
                    queued.append(label)
    return {**progress, 'unfinished_trials': unfinished, 'queued_labels': queued}


def collect_status(run_root: Path) -> dict[str, Any]:
    now = time.time()
    output = run_root / 'grid48'
    snapshot_path = output / 'snapshot.json.gz'
    snapshot = read_object(snapshot_path)
    candidates = snapshot['candidates']['candidates']
    labels = {candidate_identity(candidate): candidate['label'] for candidate in candidates}
    if len(labels) != len(candidates):
        raise ValueError('frozen grid has duplicate candidate IDs')
    automation_path = run_root / 'automation.json'
    automation = read_object(automation_path) if automation_path.exists() else {}
    roots = set()
    batch_pid_path = run_root / 'batch1.pid'
    if batch_pid_path.exists():
        roots.add(int(batch_pid_path.read_text().strip()))
    if automation.get('training_pid') is not None:
        roots.add(int(automation['training_pid']))
    processes = process_rows()
    workers = owned_workers(processes, roots)
    completed, unfinished, queued = [], [], []
    validation_complete = []
    for identity, label in labels.items():
        trial = output / 'trials' / identity
        if result_exists(trial / 'result.json.gz', identity, label, 32):
            completed.append(identity)
        else:
            progress_path = trial / 'progress.json.gz'
            if progress_path.exists():
                record = read_object(progress_path)
                unfinished.append({
                    'id': identity, 'label': label, 'stage': record['stage'],
                    'progress_age_s': max(0, now - progress_path.stat().st_mtime),
                    'video': record.get('video'), 'fit': record.get('fit'),
                })
            else:
                queued.append(label)
        if result_exists(trial / 'validation.json.gz', identity, label, 8):
            validation_complete.append(identity)
    repeat_plan_path = run_root / 'repeats/plan.json.gz'
    repeat_plan = read_object(repeat_plan_path) if repeat_plan_path.exists() else None
    shortlist_path = output / 'shortlist.json.gz'
    if repeat_plan is not None:
        shortlist = [candidate['id'] for candidate in repeat_plan['candidates']]
    else:
        shortlist = read_object(shortlist_path)['ids'] if shortlist_path.exists() else []
    if len(shortlist) != len(set(shortlist)) or not set(shortlist).issubset(labels):
        raise ValueError(f'{shortlist_path}: shortlist IDs differ from the frozen grid')
    if not set(validation_complete).issubset(shortlist):
        raise ValueError('validation results exist outside the frozen shortlist')
    if len(completed) == len(labels):
        for identity in shortlist:
            if identity in validation_complete:
                continue
            trial = output / 'trials' / identity
            stream_paths = list((trial / 'validation_streams').glob('*.json.gz'))
            progress_path = trial / 'progress.json.gz'
            timestamps = [path.stat().st_mtime for path in stream_paths]
            if progress_path.exists():
                timestamps.append(progress_path.stat().st_mtime)
            unfinished.append({
                'id': identity, 'label': labels[identity], 'stage': 'validation',
                'progress_age_s': max(0, now - max(timestamps)) if timestamps else None,
                'completed_video_streams': len(stream_paths),
            })
    failure_path = output / 'failure.json.gz'
    failure = read_object(failure_path) if failure_path.exists() else None
    error = automation.get('error')
    guard_pid = automation.get('guard_pid')
    if automation.get('status') == 'running' and guard_pid is not None:
        live_guard = any(row['pid'] == int(guard_pid) and 'Z' not in row['state'] for row in processes)
        if not live_guard:
            error = 'automation guard stopped'
    repeats = repeat_status(run_root, now, repeat_plan)
    final_test = final_test_status(run_root, now)
    grid_finished = len(completed) == len(labels) and bool(shortlist) and len(validation_complete) == len(shortlist)
    repeats_finished = repeats is None or (repeats['phase'] == 'complete' and repeats['selected_label'] is not None)
    final_finished = final_test is None or (
        final_test['phase'] == 'complete' and final_test['complete'] == final_test['planned']
    )
    finished = grid_finished and repeats_finished and final_finished and automation.get('status') != 'running'
    if repeat_plan is not None:
        repeat_jobs = len(repeat_plan['candidates']) * len(repeat_plan['seed_offsets'])
        fresh_jobs = repeat_jobs - len(repeat_plan['candidates'])
        repeats_finished = repeats is not None and repeats_finished and (
            repeats.get('fits_complete') == repeats.get('fits_planned') == repeat_jobs
            and repeats.get('fresh_fits_complete') == repeats.get('fresh_fits_planned') == fresh_jobs
            and repeats.get('validation_complete') == repeats.get('validation_planned') == repeat_jobs
        )
        final_finished = final_test is not None and final_finished and final_test['planned'] > 0
        finished = grid_finished and repeats_finished and final_finished and automation.get('status') == 'complete'
    alerts = []
    if automation.get('status') != 'complete':
        if automation.get('memory_emergency'):
            alerts.append('Memory emergency: fits paused because available memory fell below 10 GiB')
        if automation.get('memory_stalled'):
            alerts.append('Memory stalled: all fits paused and memory remains below 16 GiB; watcher action needed')
    if error or automation.get('status') == 'failed':
        phase = 'failed'
    elif finished:
        phase = 'complete'
    elif failure and not workers:
        phase = 'failed'
    elif automation.get('status') == 'complete':
        phase = 'incomplete saved results'
    elif automation.get('phase'):
        phase = automation['phase']
    elif len(completed) < len(labels):
        phase = 'training'
    elif shortlist:
        phase = 'validation'
    else:
        phase = 'awaiting validation'
    started = snapshot_path.stat().st_mtime
    if batch_pid_path.exists():
        started = min(started, batch_pid_path.stat().st_mtime)
    return {
        'utc': datetime.fromtimestamp(now, UTC).isoformat(timespec='seconds'),
        'run_root': str(run_root), 'output': str(output), 'phase': phase, 'complete': finished,
        'batch_started_utc': datetime.fromtimestamp(started, UTC).isoformat(timespec='seconds'),
        'elapsed_s': max(0, now - started), 'start_basis': 'earliest snapshot/batch1.pid modification time',
        'training': {'complete': len(completed), 'planned': len(labels)},
        'validation': {'complete': len(validation_complete), 'planned': len(shortlist)},
        'repeats': repeats, 'final_test': final_test,
        'unfinished_trials': unfinished, 'queued_labels': queued, 'workers': workers,
        'memory_available_kib': available_memory_kib(), 'automation': automation,
        'failure': failure, 'error': error, 'alerts': alerts, 'process_roots': sorted(roots),
    }


def duration(seconds: float) -> str:
    minutes = int(seconds) // 60
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes:02d}m'


def print_status(status: dict[str, Any]) -> None:
    training, validation = status['training'], status['validation']
    print(f"{status['utc']} | {status['phase']} | elapsed {duration(status['elapsed_s'])}")
    print(f"Batch started {status['batch_started_utc']} (from saved file times)")
    print(f"Training {training['complete']}/{training['planned']} | "
          f"validation {validation['complete']}/{validation['planned'] or 'shortlist pending'}")
    repeats = status['repeats']
    if repeats is not None:
        print(f"Seed repeats: {repeats['phase']} | fits {repeats.get('fits_complete', 0)}/"
              f"{repeats.get('fits_planned', '?')} | fresh fits {repeats.get('fresh_fits_complete', 0)}/"
              f"{repeats.get('fresh_fits_planned', '?')} | validation {repeats.get('validation_complete', 0)}/"
              f"{repeats.get('validation_planned', '?')}")
        if repeats['selected_label'] is not None:
            print(f"Selected configuration: {repeats['selected_label']}")
    if status['final_test'] is not None:
        final_test = status['final_test']
        print(f"Final test: {final_test['phase']} | {final_test['complete']}/{final_test['planned']}")
    workers = status['workers']
    paused = sum(worker['status'] == 'paused' for worker in workers)
    available_gib = status['memory_available_kib'] / 1024**2
    print(f"Workers {len(workers)} ({paused} paused) | memory available {available_gib:.1f} GiB")
    if status['error']:
        print(f"ERROR: {status['error']}")
    for alert in status.get('alerts', []):
        print(f'ALERT: {alert}')
    if status['failure']:
        failure = status['failure']
        print(f"Recorded failure: {failure.get('action', '?')} {failure.get('id', '?')}")
        if failure.get('traceback'):
            print(str(failure['traceback']).strip().splitlines()[-1])
    print('\nUnfinished trials with saved progress:')
    unfinished = list(status['unfinished_trials'])
    queued = list(status['queued_labels'])
    if repeats is not None:
        unfinished.extend(repeats['unfinished_trials'])
        queued.extend(repeats['queued_labels'])
    if status['final_test'] is not None:
        unfinished.extend(status['final_test']['unfinished_trials'])
        queued.extend(status['final_test']['queued_labels'])
    for trial in unfinished:
        detail = trial.get('video') or trial.get('fit') or ''
        if trial['stage'] == 'validation':
            detail = f"{trial['completed_video_streams']}/8 video streams saved"
        if trial.get('items_planned') is not None:
            detail += f" ({trial['items_complete']}/{trial['items_planned']})"
        age = trial['progress_age_s']
        updated = f'{duration(age)} ago' if age is not None else 'unavailable'
        print(f"  {trial['label']}: {trial['stage']} {detail} | last saved progress {updated}")
    if not unfinished:
        print('  none')
    print('Queued: ' + (', '.join(queued) or 'none'))
    print('\nOwned workers (PID-to-trial mapping unavailable; CPU is the ps lifetime average):')
    for worker in workers:
        print(f"  {worker['pid']} {worker['status']} [{worker['state']}] "
              f"CPU {worker['cpu_percent']:.1f}% RSS {worker['rss_kib'] / 1024**2:.2f} GiB "
              f"age {duration(worker['elapsed_s'])}")
    if not workers:
        print('  none')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--json', action='store_true', help='emit one machine-readable status object')
    args = parser.parse_args()
    try:
        status = collect_status(args.run_root)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        if args.json:
            print(json.dumps({'phase': 'status error', 'error': str(error), 'run_root': str(args.run_root)}))
        else:
            print(f'Status error: {error}', file=sys.stderr)
        raise SystemExit(1) from error
    if args.json:
        print(json.dumps(status))
    else:
        print_status(status)


if __name__ == '__main__':
    main()
