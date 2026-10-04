"""Optional progress events shared by the coordinator and its child processes.

The channel exists only while a terminal stage is running. Events are small
append-only JSON records. Opt-in worker logging keeps diagnostics off the display.
"""

# Display-only boundaries deliberately catch ordinary exceptions, never cancellation.
# ruff: noqa: BLE001

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout

PROGRESS_ENV = "BADMINTON_PROGRESS_FILE"
LOG_ENV = "BADMINTON_PROGRESS_LOG"


def enabled() -> bool:
    return bool(os.environ.get(PROGRESS_ENV))


def report(
    activity: str,
    completed: int = 0,
    total: int | None = None,
    unit: str = "items",
    *,
    worker: str = "main",
) -> None:
    """Publish an absolute counter, or an activity without a known total.

    A broken display channel must never fail extraction or annotation.
    Worker IDs identify disjoint portions of the same activity.
    """
    path = os.environ.get(PROGRESS_ENV)
    if not path:
        return
    try:
        payload = (
            json.dumps(
                {
                    "activity": activity,
                    "completed": completed,
                    "total": total,
                    "unit": unit,
                    "worker": worker,
                },
                ensure_ascii=True,
            ).encode()
            + b"\n"
        )
        fd = os.open(path, os.O_WRONLY | os.O_APPEND)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
    except Exception:
        # Counters are optional, including serialization of unexpected values.
        return


def progress_iter[T](
    items: Iterable[T],
    activity: str,
    *,
    total: int | None = None,
    unit: str = "items",
    worker: str = "main",
) -> Iterator[T]:
    """Count completed iterations, throttled to five events/second per worker.

    Never claim completion if the consumer raises or stops iterating early.
    """
    if not enabled():
        yield from items
        return
    if total is None:
        try:
            total = len(items)  # type: ignore[arg-type]
        except Exception:
            total = None
    report(activity, 0, total, unit, worker=worker)
    last = time.monotonic()
    completed = 0
    for item in items:
        yield item
        completed += 1
        now = time.monotonic()
        if now - last >= 0.2:
            report(activity, completed, total, unit, worker=worker)
            last = now
    report(activity, completed, total, unit, worker=worker)


class SafeOutput:
    """Best-effort display/log writes; a failed log falls back to the terminal."""

    def __init__(self, target, fallback=None, on_failure=lambda: None):
        self.target, self.fallback, self.on_failure = target, fallback, on_failure

    def _call(self, method, *args):
        if self.target is not None:
            try:
                return getattr(self.target, method)(*args)
            except Exception:
                self.target = self.fallback
                self.fallback = None
                self.on_failure()
                return self._call(method, *args)
        return None

    def write(self, text):
        self._call("write", text)
        return len(text)

    def flush(self):
        self._call("flush")

    def fileno(self):
        if self.target is None:
            raise OSError("output stream unavailable")
        return self.target.fileno()

    def isatty(self):
        return bool(self._call("isatty"))


@contextmanager
def capture_output(path=None, on_failure=lambda: None):
    """Capture Python diagnostics without making log I/O part of processing.

    Child entry points use LOG_ENV so their prints do not disturb the display.
    Exceptions raised by processing always propagate unchanged.
    """
    path = path if path is not None else os.environ.get(LOG_ENV)
    if not path:
        yield None
        return
    try:
        log = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115 - close must not mask processing errors
    except Exception:
        on_failure()
        yield None
        return
    try:
        with (
            redirect_stdout(SafeOutput(log, sys.stdout, on_failure)),
            redirect_stderr(SafeOutput(log, sys.stderr, on_failure)),
        ):
            yield path
    finally:
        try:
            log.close()
        except Exception:
            on_failure()
