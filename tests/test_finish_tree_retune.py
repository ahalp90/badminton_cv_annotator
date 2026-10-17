"""Checks for retaining work while completing a bounded refit grid."""

import gzip
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import finish_tree_retune as finish


def test_all_future_phases_receive_14_workers_and_two_threads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for index in range(26):
        result = tmp_path / 'grid48' / 'trials' / str(index) / 'result.json.gz'
        result.parent.mkdir(parents=True)
        with gzip.open(result, 'wt') as stream:
            json.dump({}, stream)
    selected = tmp_path / 'repeats' / 'selected.json.gz'
    selected.parent.mkdir()
    with gzip.open(selected, 'wt') as stream:
        json.dump({'config_id': 'winner', 'label': 'tuned'}, stream)
    monkeypatch.setattr(sys, 'argv', [
        'finish_tree_retune', '--run-root', str(tmp_path), '--repository', str(tmp_path),
        '--initial-pid', '99', '--previous-streams', str(tmp_path / 'previous'),
        '--test-paths', str(tmp_path / 'test.toml'), '--workers', '14', '--threads', '2',
    ])
    monkeypatch.setattr(finish, 'monitor', lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(finish, 'verify_baseline', lambda *_args: {'different_selections': []})
    commands = []

    def launch(command: list[str], *, env: dict[str, str], **_kwargs: object) -> SimpleNamespace:
        commands.append((command, env))
        return SimpleNamespace(pid=100 + len(commands))

    monkeypatch.setattr(finish.subprocess, 'Popen', launch)
    finish.main()
    assert len(commands) == 3
    for command, environment in commands:
        assert command[command.index('--workers') + 1] == '14'
        assert command[command.index('--threads') + 1] == '2'
        assert environment['OMP_NUM_THREADS'] == environment['OPENBLAS_NUM_THREADS'] == '1'
    state = json.loads((tmp_path / 'automation.json').read_text())
    assert state['status'] == 'complete'
    assert state['future_threads_per_fit'] == 2 and state['cpu_budget'] == 28


def test_future_phases_reject_more_than_28_cores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, 'argv', [
        'finish_tree_retune', '--run-root', str(tmp_path), '--repository', str(tmp_path),
        '--initial-pid', '99', '--previous-streams', str(tmp_path / 'previous'),
        '--test-paths', str(tmp_path / 'test.toml'), '--workers', '15', '--threads', '2',
    ])
    with pytest.raises(SystemExit) as error:
        finish.main()
    assert error.value.code == 2 and not (tmp_path / 'automation.json').exists()


def test_memory_controls_keep_work_and_respect_core_limit() -> None:
    workers = [{'pid': index + 100, 'paused': False, 'idle': False, 'cpu_s': index} for index in range(26)]
    stop, resume = finish.controls(workers, 100, 20)
    assert len(stop) == 6 and not resume
    stop, resume = finish.controls(workers, 80, 20)
    assert len(stop) == 22 and not resume
    stop, resume = finish.controls(workers, 25, 20)
    assert len(stop) == 25 and not resume
    assert finish.controls(workers, 8, 20) == ([row['pid'] for row in reversed(workers)], [])
    paused = [{'pid': 100, 'paused': True, 'idle': False, 'cpu_s': 20}]
    assert finish.controls(paused, 16, 20) == ([], [100])
    assert finish.controls(paused, 9, 20) == ([], [])
    active = [{'pid': 101, 'paused': False, 'idle': False, 'cpu_s': 30}]
    assert finish.controls(paused + active, 60, 20) == ([], [100])
    assert finish.controls(paused + active, 100, 20) == ([], [100])
    assert finish.controls(paused, 6.7, 16, emergency_floor=4) == ([], [100])
    assert finish.controls(active, 3.9, 16, emergency_floor=4) == ([101], [])


@pytest.mark.parametrize(('available', 'resumed'), [(31.9, []), (32, [100]), (55.3, [100]), (64, [100])])
def test_memory_guard_fills_available_slots_one_fit_at_a_time(available: float, resumed: list[int]) -> None:
    workers = [
        {'pid': 100, 'paused': True, 'idle': False, 'cpu_s': 6000},
        {'pid': 101, 'paused': True, 'idle': False, 'cpu_s': 5900},
        {'pid': 200, 'paused': False, 'idle': False, 'cpu_s': 7000},
    ]
    assert finish.controls(workers, available, 14) == ([], resumed)


@pytest.mark.parametrize(('available', 'resumed'), [(47.9, []), (48, [100]), (58.6, [100])])
def test_memory_guard_allows_third_resident_fit(available: float, resumed: list[int]) -> None:
    workers = [
        {'pid': 100, 'paused': True, 'idle': False, 'cpu_s': 6000},
        {'pid': 200, 'paused': False, 'idle': False, 'cpu_s': 7000},
        {'pid': 201, 'paused': False, 'idle': False, 'cpu_s': 6900},
    ]
    assert finish.controls(workers, available, 14) == ([], resumed)


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


@pytest.mark.parametrize(('floor', 'available'), [(4, 22.8), (10, 22.8), (4, 6), (10, 16)])
@pytest.mark.parametrize('cpu_s', [0, 6000])
def test_memory_controls_advance_fit_beside_idle_pool_worker(floor: float, available: float, cpu_s: int) -> None:
    workers = [
        {'pid': 100, 'paused': True, 'idle': False, 'cpu_s': cpu_s},
        {'pid': 200, 'paused': False, 'idle': True, 'cpu_s': 1},
    ]
    assert finish.controls(workers, available, 16, emergency_floor=floor) == ([], [100])
    workers[0]['paused'] = False
    assert finish.controls(workers, available, 16, emergency_floor=floor) == ([], [])


@pytest.mark.parametrize(('floor', 'available'), [(4, 5.9), (10, 15.9)])
def test_memory_controls_wait_for_recovery_before_replacing_worker(floor: float, available: float) -> None:
    workers = [
        {'pid': 100, 'paused': True, 'idle': False, 'cpu_s': 6000},
        {'pid': 200, 'paused': False, 'idle': True, 'cpu_s': 1},
    ]
    assert finish.controls(workers, available, 16, emergency_floor=floor) == ([], [])


def test_worker_discovery_identifies_idle_task_pipe() -> None:
    reader, writer = os.pipe()
    child = subprocess.Popen(
        [sys.executable, '-c',
         "import os, sys; print('ready', flush=True); os.read(int(sys.argv[1]), 1)",
         str(reader), 'multiprocessing.spawn'],
        pass_fds=(reader,), stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout.readline() == 'ready\n'
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            row = next(row for row in finish.worker_processes(os.getpid()) if row['pid'] == child.pid)
            if row['idle']:
                break
            time.sleep(0.01)
        assert row['idle'] and not row['paused']
        assert finish.controls([row], 8, 16) == ([], [])
        os.write(writer, b'x')
        assert child.wait(timeout=5) == 0
    finally:
        os.close(reader)
        os.close(writer)
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


def test_temporary_recovery_restores_normal_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    alive, memory = iter((True, True, False)), iter((6.7, 32))
    monkeypatch.setattr(finish, 'process_alive', lambda pid: next(alive))
    monkeypatch.setattr(finish, 'available_memory_gib', lambda: next(memory))
    monkeypatch.setattr(finish, 'worker_processes',
                        lambda pid: [{'pid': 100, 'paused': True, 'idle': False, 'cpu_s': 20}])
    monkeypatch.setattr(finish, 'process_identity', lambda pid: None)
    monkeypatch.setattr(finish.time, 'sleep', lambda seconds: None)
    signals = []
    monkeypatch.setattr(finish.os, 'kill', lambda pid, action: signals.append((pid, action)))
    state = {}
    finish.monitor(99, state, tmp_path / 'state.json', 16, emergency_floor=4)
    assert state['initial_memory_recovery_complete']
    assert state['memory_floor_gib'] == 10
    assert signals == [(100, signal.SIGCONT), (100, signal.SIGCONT)]


def test_replacement_pool_workers_do_not_block_paused_fits() -> None:
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1,
    ) as executor:
        futures = [executor.submit(os.getpid) for _ in range(2)]
        finished_pids = {future.result(timeout=10) for future in futures}
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            workers = finish.worker_processes(os.getpid())
            if len(workers) == 2 and all(row['idle'] and row['pid'] not in finished_pids for row in workers):
                break
            time.sleep(0.01)
        assert len(workers) == 2 and all(row['idle'] for row in workers)
        paused_fit = {'pid': 99999999, 'paused': True, 'idle': False, 'cpu_s': 6000}
        assert finish.controls(workers + [paused_fit], 22, 16) == ([], [paused_fit['pid']])


def test_monitor_reports_stalled_fits_beside_idle_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    alive = iter((True, False))
    workers = [
        {'pid': 100, 'paused': True, 'idle': False, 'cpu_s': 6000},
        {'pid': 200, 'paused': False, 'idle': True, 'cpu_s': 1},
    ]
    monkeypatch.setattr(finish, 'process_alive', lambda pid: next(alive))
    monkeypatch.setattr(finish, 'available_memory_gib', lambda: 15)
    monkeypatch.setattr(finish, 'worker_processes', lambda pid: workers)
    monkeypatch.setattr(finish, 'process_identity', lambda pid: None)
    monkeypatch.setattr(finish.time, 'sleep', lambda seconds: None)
    state = {}
    finish.monitor(99, state, tmp_path / 'state.json', 16)
    assert state['memory_stalled'] and state['idle_workers'] == [200]


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
