"""Progress must preserve pipeline results and aggregate subprocess counters."""

import io
import json
import os
import subprocess
import sys
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
