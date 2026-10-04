"""Progress must preserve pipeline results and aggregate subprocess counters."""

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataset_builder.progress import Counter, RunProgress, StageProgress
from shared.progress import PROGRESS_ENV, progress_iter, report


def config(**overrides):
    return SimpleNamespace(
        **{"fixed_sources": None, "commentary_enabled": True, "pose_shards": 8}
        | overrides
    )


def test_iterator_disabled_does_not_probe_length(monkeypatch):
    monkeypatch.delenv(PROGRESS_ENV, raising=False)

    class Items:
        def __iter__(self):
            yield 7

        def __len__(self):
            raise AssertionError("display disabled should not inspect iterable")

    assert list(progress_iter(Items(), "test")) == [7]


def test_iterator_records_only_completed_work_and_preserves_errors(
    tmp_path, monkeypatch
):
    channel = tmp_path / "events"
    channel.touch()
    monkeypatch.setenv(PROGRESS_ENV, str(channel))
    with pytest.raises(RuntimeError, match="consumer failed"):
        for item in progress_iter([1, 2], "test"):
            if item == 2:
                raise RuntimeError("consumer failed")
    events = [json.loads(line) for line in channel.read_text().splitlines()]
    assert events[0]["completed"] == 0
    assert not any(event["completed"] == 2 for event in events)
    assert list(progress_iter([1, 2], "next")) == [1, 2]
    assert json.loads(channel.read_text().splitlines()[-1])["completed"] == 2


def test_concurrent_child_events_are_complete_json_and_aggregate(tmp_path, monkeypatch):
    channel = tmp_path / "events"
    channel.touch()
    monkeypatch.setenv(PROGRESS_ENV, str(channel))
    children = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "from shared.progress import report; import sys; "
                    'report("pose", 0, 10, "frames", worker=sys.argv[1]); '
                    'report("pose", 10, 10, "frames", worker=sys.argv[1])'
                ),
                str(index),
            ],
            env=dict(
                os.environ,
                PYTHONPATH=str(
                    __import__("pathlib").Path(__file__).resolve().parents[1] / "src"
                ),
            ),
        )
        for index in range(8)
    ]
    for child in children:
        assert child.wait(timeout=10) == 0
    counter = Counter()
    for line in channel.read_text().splitlines():
        counter.update(json.loads(line))
    assert counter.completed == counter.total == 80
    assert len(counter.workers) == 8


def test_absolute_updates_do_not_double_count_or_regress():
    counter = Counter()
    for worker, done in [("a", 0), ("b", 0), ("a", 4), ("a", 4), ("a", 3), ("b", 6)]:
        counter.update(
            {
                "activity": "pose",
                "unit": "frames",
                "worker": worker,
                "completed": done,
                "total": 10,
            }
        )
    assert counter.completed == 10
    assert counter.total == 20
    counter.update(
        {
            "activity": "saving",
            "unit": "files",
            "worker": "main",
            "completed": 0,
            "total": None,
        }
    )
    assert counter.completed == 0
    assert counter.total is None


@pytest.mark.parametrize(
    "event",
    [
        {},
        {"activity": "x"},
        {
            "activity": "x",
            "unit": "frames",
            "worker": "a",
            "completed": -1,
            "total": 10,
        },
        {"activity": "x", "unit": "frames", "worker": "a", "completed": 3, "total": 2},
    ],
)
def test_invalid_events_do_not_corrupt_counter(event):
    counter = Counter()
    counter.update(event)
    assert counter.workers == {}


def test_auto_non_terminal_has_no_output_or_channel(tmp_path, monkeypatch):
    monkeypatch.delenv(PROGRESS_ENV, raising=False)
    stream = io.StringIO()
    owner = RunProgress(config(), ("pose",), stream=stream)
    with owner.stage("pose:video") as stage:
        assert stage is None
        assert PROGRESS_ENV not in os.environ
    assert stream.getvalue() == ""


def test_eta_appears_only_after_counter_and_rate_are_available(tmp_path):
    channel = tmp_path / "events"
    channel.touch()
    owner = RunProgress(config(), ("pose",), mode="on", stream=io.StringIO())
    stage = StageProgress(owner, channel, "Pose extraction", "")
    stage.start()
    try:
        assert stage.bar.format_dict["eta"] == ""
        stage.consume(
            json.dumps(
                {
                    "activity": "pose",
                    "unit": "frames",
                    "worker": "a",
                    "completed": 0,
                    "total": 10,
                }
            )
        )
        assert stage.bar.format_dict["eta"] == ""
        stage.bar.mininterval = 0
        stage.consume(
            json.dumps(
                {
                    "activity": "pose",
                    "unit": "frames",
                    "worker": "a",
                    "completed": 2,
                    "total": None,
                }
            )
        )
        assert stage.bar.format_dict["eta"] == ""
        stage.consume(
            json.dumps(
                {
                    "activity": "pose",
                    "unit": "frames",
                    "worker": "a",
                    "completed": 5,
                    "total": 10,
                }
            )
        )
        assert "ETA" in stage.bar.format_dict["eta"]
        assert "?" not in stage.bar.format_dict["eta"]
        stage.consume(
            json.dumps(
                {
                    "activity": "pose",
                    "unit": "frames",
                    "worker": "a",
                    "completed": 10,
                    "total": 10,
                }
            )
        )
        assert stage.bar.format_dict["eta"] == ""
    finally:
        stage.close()


def test_cleanup_failure_and_preserved_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setenv(PROGRESS_ENV, "previous-channel")
    stream = io.StringIO()
    owner = RunProgress(
        config(), ("pose",), mode="on", stream=stream, log_dir=tmp_path / "logs"
    )
    with (
        pytest.raises(RuntimeError, match="failure"),
        owner.stage("pose:video") as stage,
    ):
        print("diagnostic detail")
        report("pose", 2, 10, "frames")
        raise RuntimeError("failure")
    assert os.environ[PROGRESS_ENV] == "previous-channel"
    assert not stage.path.exists()
    assert not stage.thread.is_alive()
    assert "diagnostic detail" in stage.log_path.read_text()
    assert "interrupted or failed" in stream.getvalue()


def test_configured_skips_are_once_at_top_and_failures_still_visible():
    stream = io.StringIO()
    owner = RunProgress(
        config(fixed_sources=object(), commentary_enabled=False),
        ("search", "download", "pose", "commentary_pairing"),
        mode="on",
        stream=stream,
    )
    from dataset_builder.models import StageOutcome

    for name in ("commentary_pairing:a", "commentary_pairing:b"):
        with owner.stage(name) as stage:
            assert stage is None
        owner.finish(
            name,
            SimpleNamespace(
                outcome=StageOutcome.UNAVAILABLE, reused=False, reason="disabled"
            ),
            SimpleNamespace(counts=()),
            stage,
        )
    assert stream.getvalue().count("Commentary pairing") == 1
    owner.finish(
        "download",
        SimpleNamespace(
            outcome=StageOutcome.FAILED, reused=False, reason="missing source"
        ),
        SimpleNamespace(counts=()),
        None,
    )
    assert "✗ Download" in stream.getvalue()
    assert "missing source" in stream.getvalue()


def test_completed_reused_and_failed_are_distinct():
    from dataset_builder.models import StageOutcome

    stream = io.StringIO()
    owner = RunProgress(config(), (), mode="on", stream=stream)
    for outcome, reused, reason in [
        (StageOutcome.PROCESSED, False, None),
        (StageOutcome.PROCESSED, True, None),
        (StageOutcome.FAILED, False, "broken input"),
    ]:
        owner.finish(
            "annotation:video",
            SimpleNamespace(outcome=outcome, reused=reused, reason=reason),
            SimpleNamespace(counts=(("rallies", 24),)),
            None,
        )
    assert "24 rallies" in stream.getvalue()
    assert "reused" in stream.getvalue()
    assert "✗ Annotation" in stream.getvalue()


@pytest.mark.parametrize(
    "failure", ["temporary", "events", "log_directory", "log_open", "renderer"]
)
def test_setup_failure_runs_processing_once(tmp_path, monkeypatch, failure):
    import builtins
    from pathlib import Path

    import dataset_builder.progress as display
    from shared.progress import LOG_ENV

    monkeypatch.setenv(PROGRESS_ENV, "prior-events")
    monkeypatch.setenv(LOG_ENV, "prior-log")
    owner = RunProgress(config(), ("pose",), "on", io.StringIO(), tmp_path / "logs")

    def broken(*args, **kwargs):
        raise OSError("display unavailable")

    if failure == "temporary":
        monkeypatch.setattr(display.tempfile, "TemporaryDirectory", broken)
    elif failure == "events":
        monkeypatch.setattr(Path, "touch", broken)
    elif failure == "log_directory":
        (tmp_path / "logs").write_text("occupied")
    elif failure == "log_open":
        monkeypatch.setattr(builtins, "open", broken)
    else:
        monkeypatch.setattr(StageProgress, "start", broken)
    results = []
    with owner.stage("pose:video"):
        results.append(42)
    assert results == [42]
    assert not owner.enabled
    assert os.environ[PROGRESS_ENV] == "prior-events"
    assert os.environ[LOG_ENV] == "prior-log"


@pytest.mark.parametrize("operation", ["write", "flush", "isatty"])
def test_broken_terminal_cannot_fail_processing(operation):
    class Broken(io.StringIO):
        pass

    def broken(*args):
        raise OSError("terminal closed")

    setattr(Broken, operation, broken)
    owner = RunProgress(config(), ("pose",), "on", Broken())
    with owner.stage("pose:video"):
        assert list(progress_iter([1, 2], "pose")) == [1, 2]
    assert not owner.enabled


@pytest.mark.parametrize(
    "processing_error", [None, RuntimeError("model failed"), KeyboardInterrupt()]
)
def test_cleanup_preserves_processing_error(tmp_path, monkeypatch, processing_error):
    import dataset_builder.progress as display
    from shared.progress import LOG_ENV

    monkeypatch.delenv(PROGRESS_ENV, raising=False)
    monkeypatch.delenv(LOG_ENV, raising=False)
    owner = RunProgress(config(), ("pose",), "on", io.StringIO(), tmp_path / "logs")
    original_close = StageProgress.close
    original_cleanup = display.tempfile.TemporaryDirectory.cleanup

    def broken_close(self):
        original_close(self)
        raise OSError("render cleanup")

    def broken_cleanup(self):
        original_cleanup(self)
        raise OSError("temporary cleanup")

    monkeypatch.setattr(StageProgress, "close", broken_close)
    monkeypatch.setattr(display.tempfile.TemporaryDirectory, "cleanup", broken_cleanup)
    try:
        with owner.stage("pose:video") as stage:
            if processing_error is not None:
                raise processing_error
    except (RuntimeError, KeyboardInterrupt) as exc:
        assert exc is processing_error
    else:
        assert processing_error is None
    assert not stage.thread.is_alive()
    assert PROGRESS_ENV not in os.environ
    assert LOG_ENV not in os.environ


def test_failed_log_write_and_close_fall_back(monkeypatch):
    import builtins

    from shared.progress import capture_output

    class FailedLog:
        def write(self, text):
            raise OSError("disk full")

        def close(self):
            raise OSError("flush failed")

    stdout, stderr = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(builtins, "open", lambda *a, **kw: FailedLog())
    error = RuntimeError("processing failed")
    with pytest.raises(RuntimeError) as caught, capture_output("log"):
        print("still processing", flush=True)
        print("useful diagnostic", file=sys.stderr, flush=True)
        raise error
    assert caught.value is error
    assert sys.stdout is stdout and sys.stderr is stderr
    assert "still processing" in stdout.getvalue()
    assert "useful diagnostic" in stderr.getvalue()


def test_renderer_thread_failure_disables_display(tmp_path, monkeypatch):
    import threading

    called = threading.Event()
    owner = RunProgress(config(), ("pose",), "on", io.StringIO())

    def broken_watch(self):
        called.set()
        raise OSError("event read failed")

    monkeypatch.setattr(StageProgress, "_watch", broken_watch)
    with owner.stage("pose:video"):
        assert called.wait(2)
    assert not owner.enabled


def test_bad_counter_serialization_and_length_are_nonfatal(tmp_path, monkeypatch):
    path = tmp_path / "events"
    path.touch()
    monkeypatch.setenv(PROGRESS_ENV, str(path))
    report("counter", object())

    class Items:
        def __iter__(self):
            yield 42

        def __len__(self):
            raise ValueError("length is not available")

    assert list(progress_iter(Items(), "counter")) == [42]


def test_child_diagnostics_go_to_log_and_keep_exit_status(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    owner = RunProgress(config(), ("shuttle",), "on", io.StringIO(), tmp_path / "logs")
    with owner.stage("shuttle:video") as stage:
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; from shared.progress import capture_output; "
                    '\nwith capture_output():\n print("worker output"); '
                    'print("worker error", file=sys.stderr); sys.exit(7)'
                ),
            ],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src")),
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
    assert child.returncode == 7, child.stderr
    assert child.stdout == child.stderr == ""
    assert "worker output" in stage.log_path.read_text()
    assert "worker error" in stage.log_path.read_text()


def test_zero_width_terminal_does_not_show_unknown_eta(tmp_path):
    channel = tmp_path / "events"
    channel.touch()
    owner = RunProgress(config(), ("pose",), "on", io.StringIO())
    stage = StageProgress(owner, channel, "Pose extraction", "")
    stage.start()
    try:
        stage.bar.dynamic_ncols = lambda stream: (0, 40)
        stage.consume(
            json.dumps(
                {
                    "activity": "pose",
                    "unit": "frames",
                    "worker": "a",
                    "completed": 0,
                    "total": 10,
                }
            )
        )
        rendered = str(stage.bar)
        assert "pose" in rendered
        assert "?" not in rendered
        assert "ETA" not in rendered
    finally:
        stage.close()
