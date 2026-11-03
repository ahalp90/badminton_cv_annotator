"""Run the VideoMAE queue logic without launching training or a GPU."""
import fcntl
import signal
from pathlib import Path

import pytest

from experiments.annotator.video_models_benchmark import videomae_queue as queue
from experiments.annotator.video_models_benchmark.grid import (
    read_record,
    run_state,
    write_record,
)


def initialise(tmp_path: Path) -> tuple[Path, list[dict]]:
    output = tmp_path / "queue"
    queue.initialise(output, tmp_path / "cache", Path(queue.__file__).with_name("videomae.toml"))
    return output, read_record(output / "grid.json.gz")["runs"]


def finish_run(output: Path, run: dict, *, phase: str = "complete", continuation: bool = False) -> None:
    directory = output / run["name"]
    directory.mkdir(exist_ok=True)
    write_record(directory / "status.json.gz", {"phase": phase})
    (directory / "exit-code.txt").write_text("0\n")
    if continuation:
        (directory / "continuation.json.gz").touch()


def test_initial_queue_has_two_rates_at_each_seed_and_keeps_task_settings(tmp_path: Path) -> None:
    output, runs = initialise(tmp_path)
    assert [(run["seed"], run["learning_rate"]) for run in runs] == [
        (0, 3.125e-5), (0, 1e-4), (1, 3.125e-5), (1, 1e-4),
    ]
    manifest = read_record(output / "grid.json.gz")
    assert manifest["labels"] == "positive within ±1 frame; distance-two centres ignored"
    assert all((output / f"{run['name']}.toml").exists() for run in runs)
    with pytest.raises(FileExistsError):
        queue.initialise(output, tmp_path / "cache", Path(queue.__file__).with_name("videomae.toml"))


def test_each_run_continues_before_the_next_run_starts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, runs = initialise(tmp_path)
    calls = []

    def finish(output: Path, _manifest: dict, run: dict, *, extend: bool, resume: bool) -> None:
        calls.append((run["name"], extend, resume))
        finish_run(output, run, continuation=extend)

    monkeypatch.setattr(queue, "launch_run", finish)
    queue.execute(output, resume=False)
    assert calls == [(run["name"], extend, extend) for run in runs for extend in (False, True)]
    assert read_record(output / "grid-status.json.gz")["phase"] == "complete"


def test_resume_finishes_an_interrupted_continuation_before_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, runs = initialise(tmp_path)
    calls = []

    def pause_during_first_continuation(
        output: Path, _manifest: dict, run: dict, *, extend: bool, resume: bool,
    ) -> None:
        calls.append((run["name"], extend, resume))
        if not extend:
            finish_run(output, run)
            return
        assert resume
        finish_run(output, run, phase="paused")
        (output / run["name"] / "continuation.json.gz").touch()
        (output / run["name"] / "PAUSE").touch()

    monkeypatch.setattr(queue, "launch_run", pause_during_first_continuation)
    queue.execute(output, resume=False)
    assert calls == [(runs[0]["name"], False, False), (runs[0]["name"], True, True)]
    assert run_state(output / runs[0]["name"])["phase"] == "paused"
    assert read_record(output / "grid-status.json.gz")["phase"] == "paused"

    queue.continue_queue(output)
    resumed = []

    def finish_continuation(output: Path, _manifest: dict, run: dict, *, extend: bool, resume: bool) -> None:
        resumed.append((run["name"], extend, resume))
        finish_run(output, run, continuation=extend)

    monkeypatch.setattr(queue, "launch_run", finish_continuation)
    queue.execute(output, resume=True)
    assert resumed[0] == (runs[0]["name"], True, True)
    assert resumed[1:] == [(run["name"], extend, True) for run in runs[1:] for extend in (False, True)]
    assert read_record(output / "grid-status.json.gz")["phase"] == "complete"


def test_operational_failure_stops_the_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, runs = initialise(tmp_path)
    calls = []

    def fail_first_run(_output: Path, _manifest: dict, run: dict, *, extend: bool, resume: bool) -> None:
        calls.append(run["name"])
        raise RuntimeError("training failed")

    monkeypatch.setattr(queue, "launch_run", fail_first_run)
    with pytest.raises(RuntimeError, match="training failed"):
        queue.execute(output, resume=False)
    assert calls == [runs[0]["name"]]
    assert read_record(output / "grid-status.json.gz")["phase"] == "stopped"


@pytest.mark.parametrize("under_nohup", [False, True])
def test_hangup_respects_nohup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, under_nohup: bool) -> None:
    output, runs = initialise(tmp_path)
    terminations = []

    class Child:
        def wait(self) -> int:
            signal.raise_signal(signal.SIGHUP)
            finish_run(output, runs[0])
            return 0

    child = Child()
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *_args, **_kwargs: child)
    monkeypatch.setattr(queue, "terminate_child", lambda _child, signum, _frame: terminations.append(signum))
    inherited = signal.SIG_IGN if under_nohup else signal.SIG_DFL
    original = signal.signal(signal.SIGHUP, inherited)
    try:
        queue.launch_run(output, {"cache": str(tmp_path / "cache")}, runs[0], extend=False, resume=False)
        assert terminations == ([] if under_nohup else [signal.SIGHUP])
        assert signal.getsignal(signal.SIGHUP) == inherited
    finally:
        signal.signal(signal.SIGHUP, original)


def test_finished_runs_are_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, runs = initialise(tmp_path)
    for run in runs[:2]:
        finish_run(output, run, continuation=True)
    calls = []

    def finish_rest(output: Path, _manifest: dict, run: dict, *, extend: bool, resume: bool) -> None:
        calls.append((run["name"], extend))
        finish_run(output, run, continuation=extend)

    monkeypatch.setattr(queue, "launch_run", finish_rest)
    queue.execute(output, resume=True)
    assert calls == [(run["name"], extend) for run in runs[2:] for extend in (False, True)]


def test_queue_lock_prevents_a_second_runner(tmp_path: Path) -> None:
    output, _runs = initialise(tmp_path)
    with (output / "grid.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(BlockingIOError):
            queue.execute(output, resume=False)


def test_selection_uses_minimum_loss_checkpoint_and_lower_lr_for_a_tie(tmp_path: Path) -> None:
    output, runs = initialise(tmp_path)
    for run in runs:
        finish_run(output, run, continuation=True)
        directory = output / run["name"]
        # The last row ties minimum CE but has an intentionally higher event score.
        history = [
            {"steps": 10, "validation": {"loss": 0.7},
             "events": {"10": {"fully_correct_percent": 99, "correct_player_f1": 0.99}}},
            {"steps": 20, "validation": {"loss": 0.4},
             "events": {"10": {"fully_correct_percent": 50, "correct_player_f1": 0.7}}},
            {"steps": 30, "validation": {"loss": 0.4},
             "events": {"10": {"fully_correct_percent": 99, "correct_player_f1": 0.99}}},
        ]
        write_record(directory / "history.json.gz", history)

    ranking = queue.rank_runs(output)
    assert [row["learning_rate"] for row in ranking] == [3.125e-5, 1e-4]
    assert [run["best_step"] for result in ranking for run in result["runs"]] == [20, 20, 20, 20]
    assert [row["mean_fully_correct_percent"] for row in ranking] == [50, 50]
    assert [row["mean_correct_player_f1"] for row in ranking] == [0.7, 0.7]


def test_selection_refuses_to_rank_before_all_runs_finish(tmp_path: Path) -> None:
    output, _runs = initialise(tmp_path)
    with pytest.raises(ValueError, match="Both learning rates"):
        queue.rank_runs(output)
