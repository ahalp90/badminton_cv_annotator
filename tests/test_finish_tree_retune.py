"""Checks for retaining work while completing a bounded refit grid."""

import gzip
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import finish_tree_retune as finish


def test_memory_controls_keep_work_and_respect_core_limit() -> None:
    workers = [{'pid': index + 100, 'paused': False, 'cpu_s': index} for index in range(26)]
    stop, resume = finish.controls(workers, 100, 20)
    assert len(stop) == 6 and not resume
    stop, resume = finish.controls(workers, 80, 20)
    assert len(stop) == 22 and not resume
    stop, resume = finish.controls(workers, 25, 20)
    assert len(stop) == 25 and not resume
    assert finish.controls(workers, 8, 20) == ([row['pid'] for row in reversed(workers)], [])
    paused = [{'pid': 100, 'paused': True, 'cpu_s': 20}]
    assert finish.controls(paused, 16, 20) == ([], [100])
    assert finish.controls(paused, 9, 20) == ([], [])
    active = [{'pid': 101, 'paused': False, 'cpu_s': 30}]
    assert finish.controls(paused + active, 60, 20) == ([], [])
    assert finish.controls(paused + active, 100, 20) == ([], [100])


def test_worker_discovery_and_pause_preserve_a_live_child() -> None:
    child = subprocess.Popen(
        [sys.executable, '-c', "import time; print('ready', flush=True); time.sleep(30)", 'multiprocessing.spawn'],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout.readline() == 'ready\n'
        workers = finish.worker_processes(os.getpid())
        assert child.pid in {row['pid'] for row in workers}
        os.kill(child.pid, signal.SIGSTOP)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            row = next(row for row in finish.worker_processes(os.getpid()) if row['pid'] == child.pid)
            if row['paused']:
                break
            time.sleep(0.01)
        assert row['paused'] and finish.process_alive(child.pid)
        os.kill(child.pid, signal.SIGCONT)
        assert child.poll() is None
    finally:
        os.kill(child.pid, signal.SIGCONT)
        child.terminate()
        child.wait(timeout=5)


def test_baseline_gate_rejects_changed_contact_selection(tmp_path: Path) -> None:
    output, previous = tmp_path / 'new', tmp_path / 'old'
    trial = output / 'trials' / 'baseline'
    (trial / 'training_streams').mkdir(parents=True)
    previous.mkdir()
    identities = [f'sset_{index:02d}' for index in range(32)]
    with gzip.open(trial / 'result.json.gz', 'wt') as stream:
        json.dump({'label': 'shipped', 'per_video': dict.fromkeys(identities)}, stream)
    payload = {'events': [[10, 0.9, 'Top']], 'sequences': []}
    for identity in identities:
        for directory in (previous, trial / 'training_streams'):
            with gzip.open(directory / f'{identity}.json.gz', 'wt') as stream:
                json.dump(payload, stream)
    assert finish.verify_baseline(output, previous)['exact_streams'] == 32
    with gzip.open(trial / 'training_streams' / f'{identities[0]}.json.gz', 'wt') as stream:
        json.dump({'events': [[11, 0.9, 'Top']], 'sequences': []}, stream)
    with pytest.raises(ValueError, match='baseline selections differ'):
        finish.verify_baseline(output, previous)
