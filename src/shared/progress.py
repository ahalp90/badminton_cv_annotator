"""Optional progress reporting for annotation, extraction and their worker processes.

Workers append small JSON messages to the file selected by PROGRESS_ENV.
The terminal display reads those messages. LOG_ENV can redirect worker
prints to a log while the display is active.
"""

# Display/log failures must not stop processing. Cancellation still propagates.
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


# Progress messages for the terminal display

def enabled() -> bool:
    """Whether a progress output file has been configured."""
    return bool(os.environ.get(PROGRESS_ENV))


def report(
    activity: str,
    completed: int = 0,
    total: int | None = None,
    unit: str = "items",
    *,
    worker: str = "main",
) -> None:
    """Report the completed count so far, or an activity with no known total.

    Worker IDs separate independent portions of the same activity. Reporting
    failures are ignored so a broken display cannot stop processing.
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
        # Progress is optional, even if the message could not be serialised.
        return


def progress_iter[T](
    items: Iterable[T],
    activity: str,
    *,
    total: int | None = None,
    unit: str = "items",
    worker: str = "main",
) -> Iterator[T]:
    """Yield items and report completed iterations, at most five updates per second.

    Count an item when the caller asks for the next one. A break or exception
    leaves the last yielded item uncounted.
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


# Redirect worker stdout/stderr to a log

class SafeOutput:
    """Write to a log, falling back to the terminal if writing fails.

    After a failure, try the fallback once and call on_failure. Further output
    is ignored if neither destination works.
    """

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
    """Send Python stdout/stderr to a log, keeping logging failures non-fatal.

    Use LOG_ENV if no path is supplied. If the file cannot be opened, retain
    normal output. Processing exceptions still propagate to the caller.
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
