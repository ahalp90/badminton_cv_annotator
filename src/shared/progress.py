"""Optional progress events shared by the coordinator and its child processes.

The channel exists only while a terminal stage is running. Events are small
append-only JSON records; GPU subprocess stdout/stderr remain untouched.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Iterator

PROGRESS_ENV = "BADMINTON_PROGRESS_FILE"


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
    try:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
    except OSError:
        pass


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
        except TypeError:
            pass
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
