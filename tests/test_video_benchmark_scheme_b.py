"""Select paired validation settings and chain fresh scheme-B runs safely."""
import fcntl
import os
import shutil
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from experiments.annotator.video_models_benchmark import grid


@pytest.fixture
def completed_a(tmp_path: Path) -> Path:
    grid.initialise(tmp_path, tmp_path / "cache")
    source = tmp_path / "continuation"
    source.mkdir()
    manifest = grid.read_record(tmp_path / "grid.json.gz")
    scores = [(90, 0, 70), (60, 60, 80), (60, 60, 90), (20, 20, 95)]
    for index, run in enumerate(manifest["runs"]):
        directory = source / run["name"]
        directory.mkdir()
        shutil.copy2(tmp_path / f"{run['name']}.toml", source / f"{run['name']}.toml")
        seed_zero, seed_one, player_f1 = scores[index % 4]
        rally = seed_zero if run["seed"] == 0 else seed_one
        loss = [0.1, 0.2, 0.3, 0.01][index % 4]
        history = [
            {"steps": 10000, "validation": {"loss": loss + 1},
             "events": {"10": {"fully_correct_percent": 99, "correct_player_f1": 99}}},
            {"steps": 35000, "validation": {"loss": loss},
             "events": {"10": {"fully_correct_percent": rally, "correct_player_f1": player_f1}}},
            {"steps": 40000, "validation": {"loss": loss + 0.5},
             "events": {"10": {"fully_correct_percent": 100, "correct_player_f1": 100}}},
        ]
        grid.write_record(directory / "history.json.gz", history)
        grid.write_record(directory / "status.json.gz", {"phase": "complete", "step": 40000})
        (directory / "exit-code.txt").write_text("0\n")
    grid.write_record(source / "grid.json.gz", {**manifest, "stage": "continuation", "max_steps": None})
    grid.write_record(source / "grid-status.json.gz", {"phase": "complete"})
    grid.write_record(tmp_path / "grid-status.json.gz", {"phase": "complete"})
    (source / "wait.lock").touch()
    return tmp_path


def test_selection_uses_mean_events_at_lowest_ce_across_both_phases(completed_a: Path) -> None:
    ranking = grid.rank_configurations(completed_a / "continuation")
    assert [(row["contact_fraction"], row["learning_rate"]) for row in ranking] == [
        (0.25, 0.02), (0.5, 0.005), (0.5, 0.02), (0.25, 0.005),
    ]
    assert [row["mean_fully_correct_percent"] for row in ranking] == [60, 60, 45, 20]
    assert all(run["best_step"] == 35000 for row in ranking for run in row["runs"])


def test_selection_retains_first_lowest_ce_checkpoint_and_stable_exact_ties(completed_a: Path) -> None:
    source = completed_a / "continuation"
    manifest = grid.read_record(source / "grid.json.gz")
    for run in manifest["runs"]:
        path = source / run["name"] / "history.json.gz"
        history = grid.read_record(path)
        history[1]["events"]["10"].update(fully_correct_percent=50, correct_player_f1=80)
        history[2]["validation"]["loss"] = history[1]["validation"]["loss"]
        grid.write_record(path, history)
    ranking = grid.rank_configurations(source)
    assert [row["runs"][0]["name"] for row in ranking] == [run["name"] for run in manifest["runs"][:4]]
    assert all(row["mean_fully_correct_percent"] == 50 for row in ranking)


@pytest.mark.parametrize("fault", ["unfinished", "missing_seed", "duplicate_seed", "wrong_scheme"])
def test_selection_requires_complete_seed_pairs(completed_a: Path, fault: str) -> None:
    source = completed_a / "continuation"
    manifest = grid.read_record(source / "grid.json.gz")
    if fault == "unfinished":
        (source / manifest["runs"][0]["name"] / "exit-code.txt").write_text("1\n")
    elif fault == "missing_seed":
        manifest["runs"].pop()
    elif fault == "duplicate_seed":
        manifest["runs"][-1]["seed"] = 0
    else:
        manifest["runs"][0]["scheme"] = "B"
    grid.write_record(source / "grid.json.gz", manifest)
    with pytest.raises(ValueError):
        grid.rank_configurations(source)


def test_scheme_b_starts_fresh_then_continues_and_resume_keeps_selection(
    completed_a: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class CompletedProcess:
        def __init__(self, command: list[str], **_kwargs: object) -> None:
            calls.append(command)
            directory = Path(command[4])
            directory.mkdir(exist_ok=True)
            extending = "--extend" in command
            grid.write_record(directory / "status.json.gz", {"phase": "complete", "step": 40000 if extending else 30000})
            (directory / "exit-code.txt").write_text("0\n")
            for filename in ("last.pt", "best.pt", "history.json.gz", "validation_events.json.gz",
                             "validation_probabilities.npy.xz", "settings.json.gz"):
                (directory / filename).write_bytes(b"B training output")

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(grid.subprocess, "Popen", CompletedProcess)
    grid.run_scheme_b(completed_a)
    destination = completed_a / "scheme-b"
    manifest = grid.read_record(destination / "grid.json.gz")
    assert manifest["max_steps"] == 30000
    assert [(run["seed"], run["contact_fraction"], run["learning_rate"]) for run in manifest["runs"]] == [
        (0, 0.25, 0.02), (0, 0.5, 0.005), (1, 0.25, 0.02), (1, 0.5, 0.005),
    ]
    assert len(calls) == 8 and all(command[5] == "B" for command in calls)
    assert all("--resume" not in command and "--extend" not in command for command in calls[:4])
    assert all("--resume" in command and "--extend" in command for command in calls[4:])
    for run in manifest["runs"]:
        config = destination / f"{run['name']}.toml"
        original = completed_a / f"A{run['name'][1:]}.toml"
        assert tomllib.loads(config.read_text()) == tomllib.loads(original.read_text())
    assert grid.read_record(destination / "runner-status.json.gz")["phase"] == "complete"
    monkeypatch.setattr(grid, "rank_configurations", lambda *_args: pytest.fail("Selection must stay frozen"))
    grid.run_scheme_b(completed_a)
    assert len(calls) == 8


def test_scheme_b_waits_for_continuation_runner_lock(completed_a: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    waiting, executed = threading.Event(), threading.Event()
    write_record = grid.write_record

    def record(path: Path, value: dict) -> None:
        write_record(path, value)
        if path.name == "runner-status.json.gz" and value["phase"] == "waiting":
            waiting.set()

    def execute(output: Path, *, resume: bool) -> None:
        assert resume
        executed.set()
        grid.write_record(output / "grid-status.json.gz", {"phase": "paused"})

    monkeypatch.setattr(grid, "write_record", record)
    monkeypatch.setattr(grid, "execute", execute)
    with (completed_a / "continuation/wait.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(grid.run_scheme_b, completed_a)
            try:
                assert waiting.wait(5)
                assert not executed.wait(0.05)
                assert not (completed_a / "scheme-b/grid.json.gz").exists()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
            future.result(timeout=5)
    assert executed.is_set()


@pytest.mark.parametrize("phase", ["stopped", "paused", "running"])
def test_scheme_b_refuses_unsuccessful_dependency(
    completed_a: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    grid.write_record(completed_a / "continuation/grid-status.json.gz", {"phase": phase})
    monkeypatch.setattr(grid, "execute", lambda *_args, **_kwargs: pytest.fail("Dependency unfinished"))
    with pytest.raises(RuntimeError, match="did not complete successfully"):
        grid.run_scheme_b(completed_a)
    assert grid.read_record(completed_a / "scheme-b/runner-status.json.gz")["phase"] == "stopped"


def test_scheme_b_pause_before_selection_and_duplicate_waiter(
    completed_a: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = completed_a / "scheme-b"
    destination.mkdir()
    monkeypatch.setattr(grid, "execute", lambda *_args, **_kwargs: pytest.fail("Paused"))
    grid.pause(completed_a)
    assert (destination / "PAUSE").exists()
    grid.run_scheme_b(completed_a)
    assert grid.read_record(destination / "runner-status.json.gz")["phase"] == "paused"
    grid.continue_queue(completed_a)
    assert not (destination / "PAUSE").exists()
    with (destination / "wait.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(BlockingIOError):
            grid.run_scheme_b(completed_a)
    assert grid.read_record(destination / "runner-status.json.gz")["phase"] == "paused"


def test_status_stays_live_during_handoff_and_follows_scheme_b(
    completed_a: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # Minimal completed A table avoids drawing on unrelated per-run metrics here.
    source = completed_a / "continuation"
    manifest = grid.read_record(source / "grid.json.gz")
    grid.write_record(source / "grid.json.gz", {**manifest, "runs": []})
    destination = completed_a / "scheme-b"
    destination.mkdir()
    grid.write_record(destination / "runner-status.json.gz", {"phase": "waiting", "pid": os.getpid()})
    assert not grid.display(completed_a)
    assert "Scheme B runner: waiting" in capsys.readouterr().out
    grid.write_record(destination / "grid.json.gz", {**manifest, "runs": [], "max_steps": 30000})
    grid.write_record(destination / "grid-status.json.gz", {"phase": "complete"})
    grid.write_record(destination / "runner-status.json.gz", {"phase": "running", "pid": os.getpid()})
    assert not grid.display(completed_a)
    assert str(destination) in capsys.readouterr().out
    grid.write_record(destination / "runner-status.json.gz", {"phase": "complete"})
    assert grid.display(completed_a)
    continuation = destination / "continuation"
    continuation.mkdir()
    grid.write_record(continuation / "grid.json.gz", {**manifest, "runs": []})
    grid.write_record(continuation / "grid-status.json.gz", {"phase": "running", "pid": os.getpid()})
    grid.write_record(destination / "runner-status.json.gz", {"phase": "running", "pid": os.getpid()})
    assert not grid.display(completed_a)
    assert f"Continuation: {continuation}" in capsys.readouterr().out
    grid.write_record(continuation / "grid-status.json.gz", {"phase": "complete"})
    grid.write_record(destination / "runner-status.json.gz", {"phase": "complete"})
    assert grid.display(completed_a)


@pytest.mark.parametrize("failure_stage", ["original", "continuation"])
def test_scheme_b_stops_on_training_failure(
    completed_a: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str,
) -> None:
    def execute(output: Path, *, resume: bool) -> None:
        if failure_stage == "original":
            raise RuntimeError("Training failed")
        grid.write_record(output / "grid-status.json.gz", {"phase": "complete"})

    def extend(_output: Path) -> None:
        assert failure_stage == "continuation"
        raise RuntimeError("Continuation failed")

    monkeypatch.setattr(grid, "execute", execute)
    monkeypatch.setattr(grid, "extend_grid", extend)
    with pytest.raises(RuntimeError, match="failed"):
        grid.run_scheme_b(completed_a)
    assert grid.read_record(completed_a / "scheme-b/runner-status.json.gz")["phase"] == "stopped"


def test_pause_and_continue_reach_b_continuation(completed_a: Path) -> None:
    destination = completed_a / "scheme-b"
    destination.mkdir()
    grid.initialise_scheme_b(completed_a / "continuation", destination)
    continuation = destination / "continuation"
    continuation.mkdir()
    manifest = grid.read_record(destination / "grid.json.gz")
    grid.write_record(continuation / "grid.json.gz", {**manifest, "stage": "continuation"})
    grid.pause(completed_a)
    for output in (destination, continuation):
        assert (output / "PAUSE").exists()
        assert all((output / run["name"] / "PAUSE").exists() for run in manifest["runs"])
    grid.continue_queue(completed_a)
    assert not list(completed_a.rglob("PAUSE"))
