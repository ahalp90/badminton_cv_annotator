"""Quiet tqdm terminal display driven by real processing counters."""

# Display-only boundaries deliberately catch ordinary exceptions, never cancellation.
# ruff: noqa: BLE001

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

from shared.progress import LOG_ENV, PROGRESS_ENV, SafeOutput, capture_output

LABELS = {
    "search": "Source discovery",
    "transcript": "Transcript retrieval",
    "triage": "Relevance triage",
    "selection": "Source selection",
    "download": "Download",
    "metadata": "Video metadata",
    "commentary_cleaning": "Commentary cleaning",
    "tracknet_input": "TrackNet input preparation",
    "shuttle": "Shuttle tracking",
    "pose": "Pose extraction",
    "court": "Court detection",
    "annotation": "Annotation",
    "commentary_pairing": "Commentary pairing",
    "primitive_projection": "Primitive projection",
    "artifact_index": "Artifact indexing",
    "assembly": "Record assembly",
    "report": "Run report",
}
WIDTH = max(map(len, LABELS.values()))


def elapsed(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}:{seconds:02d}"


@dataclass
class Counter:
    """Combine disjoint workers; replace absolute updates instead of double counting."""

    activity: str = "working"
    unit: str = "items"
    workers: dict[str, tuple[int, int | None]] = field(default_factory=dict)

    def update(self, event: dict) -> bool:
        activity, worker, unit = (
            event.get(key) for key in ("activity", "worker", "unit")
        )
        completed, total = event.get("completed"), event.get("total")
        if not all(isinstance(value, str) for value in (activity, worker, unit)):
            return False
        if type(completed) is not int or completed < 0:
            return False
        if total is not None and (type(total) is not int or total < completed):
            return False
        changed = activity != self.activity or unit != self.unit
        if changed:
            self.workers.clear()
        self.activity, self.unit = activity, unit
        previous = self.workers.get(worker)
        if previous is None or completed >= previous[0]:
            self.workers[worker] = completed, total
        return changed

    @property
    def completed(self) -> int:
        return sum(done for done, _ in self.workers.values())

    @property
    def total(self) -> int | None:
        totals = [total for _, total in self.workers.values()]
        return (
            sum(totals)
            if totals and all(total is not None for total in totals)
            else None
        )


def optional_display(method):
    """Catch only display work, never the processing body of a stage."""

    @wraps(method)
    def guarded(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            self.disable()

    return guarded


class RunProgress:
    def __init__(self, config, phases, mode: str = "auto", stream=None, log_dir=None):
        self.stream = SafeOutput(
            stream if stream is not None else sys.stderr, on_failure=self.disable
        )
        self.enabled = mode == "on" or (mode == "auto" and self.stream.isatty())
        if mode not in {"auto", "on", "off"}:
            raise ValueError(f"unknown progress mode: {mode}")
        self.config = config
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.video = None
        self.suppressed = set()
        if not self.enabled:
            return
        self.write(self.style("BADMINTON  /  Processing", "1"))
        skips = {}
        if config.fixed_sources is not None:
            skips["fixed source"] = ["search", "transcript", "triage", "download"]
        if not config.commentary_enabled or config.fixed_sources is not None:
            skips[
                "commentary disabled"
                if not config.commentary_enabled
                else "fixed source commentary bypass"
            ] = ["commentary_cleaning", "commentary_pairing"]
        skips = {
            reason: [name for name in names if name in phases]
            for reason, names in skips.items()
        }
        skips = {reason: names for reason, names in skips.items() if names}
        if skips:
            self.write("\nSkipped for this run")
            for reason, names in skips.items():
                names = [name for name in names if name in phases]
                self.suppressed.update(names)
                if names:
                    self.write(
                        f"  {', '.join(LABELS[name] for name in names)} · {reason}"
                    )

    def disable(self):
        self.enabled = False

    def style(self, text, code):
        return f"\033[{code}m{text}\033[0m" if self.stream.isatty() else text

    def write(self, text):
        print(text, file=self.stream, flush=True)

    @contextmanager
    def stage(self, name):
        phase, _, video = name.partition(":")
        if not self.enabled or phase in self.suppressed:
            yield None
            return
        stage = temporary = None
        previous = {key: os.environ.get(key) for key in (PROGRESS_ENV, LOG_ENV)}
        try:
            try:
                if video and video != self.video:
                    self.write("\n" + self.style(f"Video  {video}", "1;36"))
                    self.video = video
                elif not video and self.video is not None:
                    self.write("\nRun outputs")
                    self.video = None
                label = LABELS.get(phase, phase)
                detail = ""
                if phase == "pose" and self.config.pose_shards > 1:
                    detail = f"  {self.config.pose_shards} parallel shards"
                elif phase == "court":
                    detail = "  8 workers"
                self.write(
                    "  "
                    + self.style("›", "36")
                    + f" {label:<{WIDTH}}  {'working':<20}{detail}"
                )
                if self.enabled:
                    temporary = tempfile.TemporaryDirectory(
                        prefix="badminton-progress-"
                    )
                    path = Path(temporary.name) / "events.jsonl"
                    path.touch()
                    stage = StageProgress(self, path, label, detail)
                    if self.log_dir is not None:
                        self.log_dir.mkdir(parents=True, exist_ok=True)
                        stage.log_path = self.log_dir / f"{phase}-{time.time_ns()}.log"
                    os.environ[PROGRESS_ENV] = str(path)
                    stage.start()
            except Exception:
                self.disable()
            if not self.enabled:
                self._restore_environment(previous)
                # No retry or interception of processing: yield exactly once.
                yield None
            else:
                with capture_output(
                    stage.log_path if self.log_dir is not None else "", self.disable
                ) as log_path:
                    stage.log_path = Path(log_path) if log_path is not None else None
                    if log_path is not None:
                        os.environ[LOG_ENV] = str(log_path)
                    try:
                        yield stage
                    except BaseException:
                        self.write(f"  ✗ {label:<{WIDTH}}  interrupted or failed")
                        raise
        finally:
            # Each cleanup is independent; display failures cannot mask an
            # exception from processing or prevent environment restoration.
            try:
                if stage is not None:
                    stage.close()
            except Exception:
                self.disable()
            finally:
                self._restore_environment(previous)
                if temporary is not None:
                    try:
                        temporary.cleanup()
                    except Exception:
                        self.disable()

    @staticmethod
    def _restore_environment(previous):
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @optional_display
    def finish(self, name, event, record, stage):
        if not self.enabled:
            return
        phase = name.partition(":")[0]
        outcome = event.outcome.value
        if phase in self.suppressed and outcome in {"skipped", "unavailable"}:
            return
        counts = dict(record.counts)
        work = next(
            (
                f"{counts[key]:,} {key}"
                for key in (
                    "frames",
                    "rallies",
                    "scenes",
                    "videos",
                    "indexed_artifacts",
                )
                if key in counts
            ),
            "done",
        )
        if event.reused:
            work = "reused"
        elif outcome != "processed":
            work = outcome
        seconds = stage.seconds if stage is not None else 0
        icon = (
            "✓"
            if outcome == "processed"
            else "–"
            if outcome in ("skipped", "unavailable")
            else "✗"
        )
        label = LABELS.get(phase, phase)
        reason = f"  {event.reason}" if event.reason and outcome != "processed" else ""
        detail = stage.detail if stage is not None else ""
        if (
            stage is not None
            and stage.log_path is not None
            and outcome in {"failed", "unavailable"}
        ):
            reason += f"  log: {stage.log_path}"
        self.write(
            "  "
            + self.style(
                icon,
                "32"
                if outcome == "processed"
                else "31"
                if outcome == "failed"
                else "2",
            )
            + f" {label:<{WIDTH}}  {work:<20} {elapsed(seconds):>5}{detail}{reason}"
        )


class StageProgress:
    def __init__(self, owner, path, label, detail):
        self.owner, self.path, self.label, self.detail = owner, path, label, detail
        self.counter = Counter()
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.seconds = 0.0
        self.bar = None
        self.log_path = None
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def start(self):
        from tqdm import tqdm

        class Bar(tqdm):
            @property
            def format_dict(bar):
                values = super().format_dict
                # A zero-column recording PTY must not trigger tqdm's default
                # format, which includes unknown ETA placeholders.
                if values["ncols"] == 0:
                    values["ncols"] = None
                values["eta"] = ""
                rate, total, done = values["rate"], values["total"], values["n"]
                if (
                    total is not None
                    and 0 < done < total
                    and rate
                    and math.isfinite(rate)
                    and rate > 0
                ):
                    values["eta"] = (
                        f"  ETA {bar.format_interval(math.ceil((total - done) / rate))}"
                    )
                values["work"] = self.counter.activity
                if self.counter.workers:
                    count = f"{done:,}/{total:,}" if total is not None else f"{done:,}"
                    values["work"] += f" · {count} {self.counter.unit}"
                return values

        self.bar = Bar(
            total=None,
            file=self.owner.stream,
            leave=False,
            dynamic_ncols=True,
            mininterval=0.1,
            ascii=" ━",
            colour="cyan" if self.owner.stream.isatty() else None,
            bar_format="    {work} · elapsed {elapsed}",
        )
        self.thread.start()

    def consume(self, line):
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                return
            changed = self.counter.update(event)
        except (ValueError, TypeError):
            return
        if changed:
            self.bar.reset(total=self.counter.total)
        self.bar.total = self.counter.total
        done = self.counter.completed
        if done >= self.bar.n:
            self.bar.update(done - self.bar.n)
        if self.counter.total is not None and self.counter.total > 0:
            self.bar.bar_format = "    {bar:18} {percentage:3.0f}%  {work}{eta}"
        else:
            self.bar.bar_format = "    {work} · elapsed {elapsed}"

    def watch(self):
        try:
            self._watch()
        except Exception:
            self.owner.disable()

    def _watch(self):
        # Keep incomplete lines buffered: another process may be appending.
        pending = ""
        with self.path.open() as events:
            while self.owner.enabled:
                pending += events.read()
                lines = pending.split("\n")
                pending = lines.pop()
                for line in lines:
                    self.consume(line)
                self.bar.refresh()
                if self.stop.wait(0.1):
                    pending += events.read()
                    for line in pending.splitlines():
                        self.consume(line)
                    break

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join()
        self.seconds = time.monotonic() - self.started
        if self.bar is not None:
            self.bar.close()
        if self.owner.stream.isatty():
            self.owner.stream.write("\033[1A\r\033[2K")
            self.owner.stream.flush()
