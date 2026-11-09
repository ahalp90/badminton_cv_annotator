"""Queue boundaries for independently recoverable benchmark runs."""
import fcntl
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from experiments.annotator.video_models_benchmark import grid


def test_grid_crosses_two_rates_ratios_and_seeds(tmp_path: Path) -> None:
    grid.initialise(tmp_path, tmp_path / "cache")
    manifest = grid.read_record(tmp_path / "grid.json.gz")
    configs = [tomllib.loads((tmp_path / f"{run['name']}.toml").read_text()) for run in manifest["runs"]]
    assert [(config["seed"], config["learning_rate"], config["contact_fraction"]) for config in configs] == [
        (seed, rate, ratio) for seed in (0, 1) for ratio in (0.5, 0.25) for rate in (0.02, 0.005)
    ]
    assert all(config["warmup_start_lr"] == config["learning_rate"] / 10 for config in configs)
    assert all(config["validation_steps"][-1] == config["max_steps"] == 30000 for config in configs)
    assert all(config["validation_stride"] == 3 for config in configs)
    with pytest.raises(FileExistsError):
        grid.initialise(tmp_path, tmp_path / "cache")


def test_grid_skips_completed_and_stops_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grid.initialise(tmp_path, tmp_path / "cache")
    runs = grid.read_record(tmp_path / "grid.json.gz")["runs"]
    first = tmp_path / runs[0]["name"]
    first.mkdir()
    grid.write_record(first / "status.json.gz", {"phase": "complete"})
    (first / "exit-code.txt").write_text("0\n")
    calls = []

    class FailedProcess:
        def __init__(self, command: list[str], **_kwargs: object) -> None:
            calls.append(command)

        def wait(self) -> int:
            return 7

    monkeypatch.setattr(grid.subprocess, "Popen", FailedProcess)
    with pytest.raises(RuntimeError, match="exit 7"):
        grid.execute(tmp_path, resume=False)
    assert len(calls) == 1
    assert str(tmp_path / runs[1]["name"]) in calls[0]
    assert grid.read_record(tmp_path / "grid-status.json.gz")["phase"] == "stopped"


def test_grid_pause_and_resume_do_not_start_next_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grid.initialise(tmp_path, tmp_path / "cache")
    runs = grid.read_record(tmp_path / "grid.json.gz")["runs"]
    grid.pause(tmp_path)
    assert all((tmp_path / run["name"] / "PAUSE").exists() for run in runs)
    calls = []

    class PausedProcess:
        def __init__(self, command: list[str], **_kwargs: object) -> None:
            calls.append(command)
            grid.write_record(tmp_path / runs[0]["name"] / "status.json.gz", {"phase": "paused"})

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(grid.subprocess, "Popen", PausedProcess)
    grid.execute(tmp_path, resume=False)
    assert calls == []
    grid.continue_queue(tmp_path)
    grid.write_record(tmp_path / runs[0]["name"] / "settings.json.gz", {})
    grid.execute(tmp_path, resume=True)
    assert len(calls) == 1 and calls[0][-1] == "--resume"
    assert grid.read_record(tmp_path / "grid-status.json.gz")["phase"] == "paused"
    assert not any((tmp_path / run["name"] / "PAUSE").exists() for run in runs)


def finished_grid(output: Path) -> list[dict]:
    grid.initialise(output, output / "cache")
    runs = grid.read_record(output / "grid.json.gz")["runs"]
    for run in runs:
        directory = output / run["name"]
        directory.mkdir()
        grid.write_record(directory / "status.json.gz", {"phase": "complete", "step": 30000})
        (directory / "exit-code.txt").write_text("0\n")
        for filename in ("last.pt", "best.pt", "history.json.gz", "validation_events.json.gz",
                         "validation_probabilities.npy.xz", "settings.json.gz"):
            (directory / filename).write_bytes(b"original data")
    grid.write_record(output / "grid-status.json.gz", {"phase": "complete"})
    return runs


def test_continuation_waits_for_original_lock_and_preserves_originals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = finished_grid(tmp_path)
    waiting, executed = threading.Event(), threading.Event()
    write_record = grid.write_record

    def record(path: Path, value: dict) -> None:
        write_record(path, value)
        if value.get("phase") == "waiting":
            waiting.set()

    def execute(output: Path, *, resume: bool) -> None:
        assert output == tmp_path / "continuation" and resume
        manifest = grid.read_record(output / "grid.json.gz")
        assert manifest["stage"] == "continuation"
        for run in runs:
            assert (output / run["name"] / "last.pt").read_bytes() == b"original data"
            assert not (output / run["name"] / "exit-code.txt").exists()
        (output / runs[0]["name"] / "last.pt").write_bytes(b"continued data")
        executed.set()

    monkeypatch.setattr(grid, "write_record", record)
    monkeypatch.setattr(grid, "execute", execute)
    with (tmp_path / "grid.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(grid.extend_grid, tmp_path)
            try:
                assert waiting.wait(5)
                assert not executed.wait(0.05)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
            future.result(timeout=5)
    assert executed.is_set()
    assert (tmp_path / runs[0]["name"] / "last.pt").read_bytes() == b"original data"


@pytest.mark.parametrize("phase", ["stopped", "paused"])
def test_continuation_refuses_unsuccessful_original_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    finished_grid(tmp_path)
    grid.write_record(tmp_path / "grid-status.json.gz", {"phase": phase})
    monkeypatch.setattr(grid, "execute", lambda *args, **kwargs: pytest.fail("Original did not finish"))
    with pytest.raises(RuntimeError, match="did not complete successfully"):
        grid.extend_grid(tmp_path)
    assert grid.read_record(tmp_path / "continuation/grid-status.json.gz")["phase"] == "stopped"


def test_continuation_pause_markers_are_honoured_and_cleared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    finished_grid(tmp_path)
    (tmp_path / "PAUSE").touch()
    monkeypatch.setattr(grid, "execute", lambda *args, **kwargs: pytest.fail("Queue is paused"))
    grid.extend_grid(tmp_path)
    assert grid.read_record(tmp_path / "continuation/grid-status.json.gz")["phase"] == "paused"
    grid.pause(tmp_path)
    assert (tmp_path / "continuation/PAUSE").exists()
    grid.continue_queue(tmp_path)
    assert not (tmp_path / "PAUSE").exists()
    assert not (tmp_path / "continuation/PAUSE").exists()


def test_continuation_launches_with_resume_and_skips_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = finished_grid(tmp_path)
    calls = []

    class CompletedProcess:
        def __init__(self, command: list[str], **_kwargs: object) -> None:
            calls.append(command)
            output = Path(command[4])
            grid.write_record(output / "status.json.gz", {"phase": "complete", "stage": "continuation", "step": 40000})
            (output / "exit-code.txt").write_text("0\n")

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(grid.subprocess, "Popen", CompletedProcess)
    grid.extend_grid(tmp_path)
    assert len(calls) == len(runs)
    assert all("--extend" in command and "--resume" in command for command in calls)
    grid.extend_grid(tmp_path)
    assert len(calls) == len(runs)


def test_status_reports_dead_continuation_waiter_while_original_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    grid.initialise(tmp_path, tmp_path / "cache")
    grid.write_record(tmp_path / "grid-status.json.gz", {"phase": "running", "pid": 1})
    continuation = tmp_path / "continuation"
    continuation.mkdir()
    grid.write_record(continuation / "grid-status.json.gz", {"phase": "waiting", "pid": 2})

    def check_pid(pid: int, _signal: int) -> None:
        if pid == 2:
            raise ProcessLookupError

    monkeypatch.setattr(grid.os, "kill", check_pid)
    assert not grid.display(tmp_path)
    assert "Continuation runner: stopped (queue process absent)" in capsys.readouterr().out


def test_full_grid_selects_paired_validation_winner_before_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid.initialise(tmp_path, tmp_path / "cache", full_grid=True, test_cache=tmp_path / "test-cache")
    manifest = grid.read_record(tmp_path / "grid.json.gz")
    runs = manifest["runs"]
    assert {(run["scheme"], run["seed"], run["contact_fraction"], run["learning_rate"]) for run in runs} == {
        (scheme, seed, fraction, rate) for scheme in ("A", "B") for seed in (0, 1)
        for fraction in (0.25, 0.5) for rate in (0.005, 0.02)
    }
    for run in runs:
        settings = tomllib.loads((tmp_path / f"{run['name']}.toml").read_text())
        assert settings["continue_until_patience"] and settings["break_alternation_ties"]
        assert settings["validation_stride"] == 3
        directory = tmp_path / run["name"]
        directory.mkdir()
        grid.write_record(directory / "status.json.gz", {"phase": "complete"})
        (directory / "exit-code.txt").write_text("0\n")
        winner = run["scheme"] == "B" and run["contact_fraction"] == 0.25 and run["learning_rate"] == 0.005
        score = 60 if winner else (90 if run["seed"] == 0 else 10)
        grid.write_record(directory / "history.json.gz", [
            {"steps": 30000, "validation": {"loss": 0.2},
             "events": {"10": {"fully_correct_percent": score, "correct_player_f1": 80}}},
            {"steps": 35000, "validation": {"loss": 0.3},
             "events": {"10": {"fully_correct_percent": 99, "correct_player_f1": 99}}},
        ])
    commands = []

    def infer(command: list[str], *, check: bool) -> None:
        assert check
        selection = grid.read_record(tmp_path / "selection.json.gz")["ranking"]
        assert len(selection) == 8
        assert selection[0]["mean_fully_correct_percent"] == 60
        assert all(run["best_step"] == 30000 for row in selection for run in row["runs"])
        commands.append(command)
        destination = Path(command[-1])
        destination.mkdir(parents=True)
        grid.write_record(destination / "status.json.gz", {"phase": "complete"})

    monkeypatch.setattr(grid.subprocess, "run", infer)
    grid.execute(tmp_path, resume=False)
    assert len(commands) == 2
    assert all("B-contact25-lr0.005" in command[command.index("--checkpoint") + 1] for command in commands)
    assert all(command[command.index("--stride") + 1] == "3" for command in commands)
    assert all(command[command.index("--offset") + 1] == "0" for command in commands)
    grid.execute(tmp_path, resume=True)
    assert len(commands) == 2
    (tmp_path / runs[-1]["name"] / "exit-code.txt").unlink()
    with pytest.raises(ValueError, match="every configured run"):
        grid.rank_configurations(tmp_path)
