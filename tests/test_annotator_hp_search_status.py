"""Progress remains useful after the detached process exits or fails."""

import gzip
import json
import os

from experiments.annotator.tree_simplification.hp_search_status import status_lines


def test_running_search_shows_current_progress(tmp_path):
    with gzip.open(tmp_path / 'progress.json.gz', 'wt') as handle:
        json.dump({
            'status': 'running', 'stage': 'screen', 'completed': 6, 'total': 192,
            'fits_completed': 24, 'fits_total': 768,
            'started_at': 1000, 'updated_at': 4600, 'active': ['body-12'],
            'best_label': 'body-12', 'best_player_f1': 0.85,
        }, handle)
    (tmp_path / 'process.pid').write_text(str(os.getpid()))
    lines = status_lines(tmp_path, now=4660)
    assert 'Status: running' in lines
    assert 'Progress: 6 / 192' in lines
    assert 'Fits: 24 / 768' in lines
    assert 'Elapsed: 01:01:00' in lines
    assert 'Last update: 00:01:00 ago' in lines
    assert 'Best so far: body-12, player F1 85.00%' in lines


def test_failed_process_overrides_stale_running_progress(tmp_path):
    with gzip.open(tmp_path / 'progress.json.gz', 'wt') as handle:
        json.dump({'status': 'running', 'stage': 'screen', 'started_at': 1000}, handle)
    (tmp_path / 'exit-code').write_text('1\n')
    os.utime(tmp_path / 'exit-code', (4600, 4600))
    (tmp_path / 'run.log').write_text('Fitting selector\nRuntimeError: fit failed\n')
    lines = status_lines(tmp_path, now=20_000)
    assert lines[0] == 'Status: failed (exit 1)'
    assert lines[-1] == 'RuntimeError: fit failed'
    assert 'Elapsed: 01:00:00' in lines


def test_deadline_and_completed_elapsed_time(tmp_path):
    with gzip.open(tmp_path / 'progress.json.gz', 'wt') as handle:
        json.dump({'status': 'complete', 'started_at': 100, 'finished_at': 3700}, handle)
    (tmp_path / 'exit-code').write_text('0\n')
    assert 'Elapsed: 01:00:00' in status_lines(tmp_path, now=10_000)
    (tmp_path / 'exit-code').write_text('124\n')
    assert status_lines(tmp_path)[0] == 'Status: time limit reached'
