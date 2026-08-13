"""Shared process and raw-array helpers for dataset-builder pose stages."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
from types import FrameType
from typing import Mapping, Sequence

import numpy as np


POSE_CHILD_STEM = "pose"
_RAW_POSE_SUFFIXES = {
    "kps": "_raw_kps.npy",
    "bboxes": "_raw_bboxes.npy",
    "scores": "_raw_scores.npy",
    "kp_scores": "_raw_kp_scores.npy",
    "ndet": "_raw_ndet.npy",
}


def load_raw_pose_mapping(output_dir: Path, stem: str) -> dict[str, np.ndarray]:
    """Load the canonical five uncompressed arrays from a pose child."""
    loaded: dict[str, np.ndarray] = {}
    for name, suffix in _RAW_POSE_SUFFIXES.items():
        path = output_dir / f"{stem}{suffix}"
        if not path.is_file():
            raise FileNotFoundError(f"RTMLib pose subprocess did not produce {path.name}")
        loaded[name] = np.load(path, allow_pickle=False)
    return loaded


def resolve_pose_executable(executable: str | Path) -> Path:
    """Resolve and validate the configured pose-interpreter executable."""
    requested = os.fspath(executable)
    located = shutil.which(requested)
    if located is None:
        candidate = Path(requested)
        if not candidate.is_file():
            raise FileNotFoundError(f"pose interpreter is not an executable file: {requested}")
        located = os.fspath(candidate)
    resolved = Path(located).resolve(strict=True)
    if not os.access(resolved, os.X_OK):
        raise PermissionError(f"pose interpreter is not executable: {resolved}")
    return resolved


def pose_subprocess_environment() -> dict[str, str]:
    """Return the environment needed by isolated pose child processes."""
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1]
    required = [os.fspath(source_root), os.fspath(source_root / "bst_x")]
    existing = environment.get("PYTHONPATH")
    if existing:
        required.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(required)
    return environment


def run_isolated_pose_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run a pose child in an owned process group that follows cancellation."""
    handled_signals = (signal.SIGINT, signal.SIGTERM)
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in handled_signals
    }
    installed: list[signal.Signals] = []
    process: subprocess.Popen[str] | None = None
    pending_signal: int | None = None
    cleanup_started = False

    def cancel(signum: int, _frame: FrameType | None) -> None:
        nonlocal cleanup_started, pending_signal
        if pending_signal is None:
            pending_signal = signum
        if process is None or cleanup_started:
            return
        cleanup_started = True
        raise SystemExit(128 + signum)

    try:
        for signum in handled_signals:
            signal.signal(signum, cancel)
            installed.append(signum)
        if pending_signal is not None:
            cleanup_started = True
            raise SystemExit(128 + pending_signal)
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=dict(env),
            start_new_session=True,
        )
        if pending_signal is not None:
            cleanup_started = True
            raise SystemExit(128 + pending_signal)
        stdout, stderr = process.communicate()
    except BaseException:
        cleanup_started = True
        if process is not None:
            _terminate_process_group(process)
        if pending_signal is not None:
            raise SystemExit(128 + pending_signal) from None
        raise
    finally:
        for signum in reversed(installed):
            signal.signal(signum, previous_handlers[signum])

    if process.returncode != 0:
        cleanup_started = True
        _terminate_process_group(process)
    if pending_signal is not None:
        raise SystemExit(128 + pending_signal)
    return subprocess.CompletedProcess(
        list(command),
        process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate the owned child group and reap its direct process."""
    process_group = process.pid
    _signal_process_group(process_group, signal.SIGTERM)
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        _signal_process_group(process_group, signal.SIGKILL)
        process.wait(timeout=2.0)

    deadline = time.monotonic() + 2.0
    while _process_group_exists(process_group) and time.monotonic() < deadline:
        time.sleep(0.02)
    if _process_group_exists(process_group):
        _signal_process_group(process_group, signal.SIGKILL)


def _signal_process_group(process_group: int, signum: signal.Signals) -> None:
    try:
        os.killpg(process_group, signum)
    except ProcessLookupError:
        pass


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
