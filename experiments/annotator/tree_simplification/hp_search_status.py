"""Print progress for a running or completed annotator settings search."""

import argparse
import gzip
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any


def duration(seconds: float) -> str:
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def read_progress(output: Path) -> dict[str, Any]:
    path = output / 'progress.json.gz'
    if not path.exists():
        return {}
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)


def status_lines(output: Path, *, now: float | None = None) -> list[str]:
    """Combine saved progress with the detached process's PID and exit status."""
    now = time.time() if now is None else now
    progress = read_progress(output)
    exit_path = output / 'exit-code'
    pid_path = output / 'process.pid'
    status = progress.get('status', 'starting')
    pid = int(pid_path.read_text()) if pid_path.exists() else None
    if exit_path.exists():
        exit_code = int(exit_path.read_text())
        if exit_code == 124:
            status = 'time limit reached'
        elif exit_code:
            status = f'failed (exit {exit_code})'
        elif status not in ('complete', 'time_limit'):
            status = f'stopped ({status})'
    elif pid is not None and not process_running(pid):
        status = 'stopped without an exit status'
    status = status.replace('_', ' ')
    lines = [f'Status: {status}']
    if progress.get('stage'):
        lines.append(f'Stage: {progress["stage"]}')
    if 'completed' in progress and 'total' in progress:
        lines.append(f'Progress: {progress["completed"]} / {progress["total"]}')
    if 'fits_completed' in progress and 'fits_total' in progress:
        lines.append(f'Fits: {progress["fits_completed"]} / {progress["fits_total"]}')
    if 'started_at' in progress:
        end = exit_path.stat().st_mtime if exit_path.exists() else now
        end = progress.get('finished_at', end)
        lines.append(f'Elapsed: {duration(end - progress["started_at"])}')
    if 'updated_at' in progress:
        lines.append(f'Last update: {duration(now - progress["updated_at"])} ago')
    if progress.get('active'):
        lines.append('Active: ' + ', '.join(progress['active']))
    if progress.get('best_label'):
        best = f'Best so far: {progress["best_label"]}'
        if 'best_rallies' in progress:
            best += f' — {progress["best_rallies"]} fully correct rallies'
        if 'best_player_f1' in progress:
            best += f', player F1 {100 * progress["best_player_f1"]:.2f}%'
        lines.append(best)
    if progress.get('message'):
        lines.append(str(progress['message']))
    log_path = output / 'run.log'
    if status.startswith(('failed', 'stopped')) and log_path.exists():
        with log_path.open() as handle:
            tail = deque(handle, maxlen=8)
        lines.extend(['', 'Last log lines:', *(line.rstrip() for line in tail)])
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Search output directory')
    args = parser.parse_args()
    print('\n'.join(status_lines(args.output)))


if __name__ == '__main__':
    main()
