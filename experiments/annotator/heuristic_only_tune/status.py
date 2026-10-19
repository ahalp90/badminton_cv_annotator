"""Show or watch the saved search status without importing the annotator."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

STATUS_SCHEMA = 'heuristic-only-tune-status/1'
REFRESH_SECONDS = 1.0


def read_status(output: Path) -> dict[str, Any]:
    """Read the latest complete status snapshot written by the coordinator."""
    path = output / 'status.json.gz'
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        status = json.load(handle)
    if status['schema'] != STATUS_SCHEMA:
        raise ValueError(f'{path}: unsupported status schema {status["schema"]!r}')
    return status


def elapsed_time(seconds: float) -> str:
    """Format the elapsed duration as hours, minutes and seconds."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def render_status(status: dict[str, Any], width: int = 100) -> str:
    """Show job counts and draw bars only for stages with a measured total."""
    completed = status['completed_jobs']
    total = status['total_jobs']
    percent = 100 * completed / total if total else 0.0
    lines = [
        f'{status["state"].upper()}  {completed:,}/{total:,} jobs ({percent:.1f}%)',
        (
            f'Active: {status["active_jobs"]:,}  Queued: {status["queued_jobs"]:,}  '
            f'Failed: {status["failed_jobs"]:,}  Worker limit: {status["target_workers"]}'
        ),
        f'Elapsed: {elapsed_time(status["elapsed_seconds"])}  Updated: {status["updated_at_utc"]}',
    ]
    for job in status['active']:
        progress = job.get('progress')
        stage = progress['activity'] if progress is not None else job['stage']
        lines.append(
            f'{job["video"]}  {job["candidate_id"][:12]}  {job["label"]}  | '
            f'{stage}  | elapsed {elapsed_time(job["elapsed_seconds"])}'
        )
        if progress is None:
            continue
        count = progress['completed']
        stage_total = progress['total']
        unit = progress['unit']
        if stage_total is None or stage_total == 0:
            if count:
                lines.append(f'  {count:,} {unit} processed; total unavailable')
            continue
        # Job elapsed time includes earlier stages, so it cannot estimate this stage's rate.
        bar = tqdm.format_meter(
            count, stage_total, 0, ncols=width - 2, prefix=stage, unit=unit, ascii=True,
            bar_format='{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit}',
        )
        lines.append(f'  {bar}')
    if status.get('error') is not None:
        error = status['error']
        lines.append(f'{error["type"]}: {error["message"]}')
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='directory containing status.json.gz')
    parser.add_argument('--watch', action='store_true', help='refresh every second until the run ends')
    args = parser.parse_args(argv)
    try:
        while True:
            try:
                status = read_status(args.output)
            except FileNotFoundError:
                if not args.watch:
                    raise
                status = None
            width = shutil.get_terminal_size(fallback=(100, 24)).columns
            message = (
                f'Waiting for {args.output / "status.json.gz"}'
                if status is None else render_status(status, width)
            )
            if args.watch and sys.stdout.isatty():
                sys.stdout.write('\033[H\033[J')
            print(message, flush=True)
            if not args.watch or (status is not None and status['state'] in {'complete', 'failed'}):
                return 1 if status is not None and status['state'] == 'failed' else 0
            time.sleep(REFRESH_SECONDS)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Cannot read search status: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
