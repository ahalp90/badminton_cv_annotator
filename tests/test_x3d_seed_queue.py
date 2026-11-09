"""Exercise the extra-seed queue without training a model or using a GPU."""
import fcntl
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from experiments.annotator.video_models_benchmark import grid
from experiments.annotator.video_models_benchmark import x3d_seed_queue as queue


@pytest.fixture
def queues(tmp_path: Path) -> tuple[Path, Path, Path]:
    source, after, output = (tmp_path / name for name in ("x3d", "videomae", "extra"))
    grid.initialise(source, tmp_path / "rgb", full_grid=True, test_cache=tmp_path / "test-rgb")
    manifest = grid.read_record(source / "grid.json.gz")
    runs = [run for run in manifest["runs"]
            if run["scheme"] == "B" and run["contact_fraction"] == 0.25 and run["learning_rate"] == 0.005]
    winner = {"scheme": "B", "contact_fraction": 0.25, "learning_rate": 0.005, "runs": runs}
    after.mkdir()
    for dependency in (source, after):
        (dependency / "runner.lock").touch()
        (dependency / "queue-exit-code.txt").write_text("0\n")
        grid.write_record(dependency / "grid-status.json.gz", {"phase": "complete"})
        grid.write_record(dependency / "selection.json.gz", {"ranking": [winner]})
        for seed in (0, 1):
            test = dependency / "test" / f"seed{seed}"
            test.mkdir(parents=True)
            grid.write_record(test / "status.json.gz", {"phase": "complete"})
    return source, after, output


def fake_processes(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls = []

    class Process:
        pid = 12345

        def __init__(self, command: list[str], **_kwargs: object) -> None:
            calls.append(command)
            if command[0] == "bash":
                directory = Path(command[4])
                settings = tomllib.loads(Path(command[command.index("--config") + 1]).read_text())
                assert command[5] == "B"
                assert command[6] == "2000"
                assert "--extend" not in command
                assert settings["continue_until_patience"] and settings["break_alternation_ties"]
                directory.mkdir(exist_ok=True)
                grid.write_record(directory / "settings.json.gz", settings)
                (directory / "exit-code.txt").write_text("0\n")
                (directory / "best.pt").touch()
            else:
                assert command[command.index("--role") + 1] == "test"
                assert command[command.index("--stride") + 1] == "3"
                assert command[command.index("--offset") + 1] == "0"
                assert Path(command[command.index("--checkpoint") + 1]).exists()
                directory = Path(command[command.index("--output") + 1])
                directory.mkdir()
            grid.write_record(directory / "status.json.gz", {"phase": "complete"})

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(queue.subprocess, "Popen", Process)
    return calls


def test_follow_trains_and_tests_only_three_new_seeds_then_resumes_without_repeating(
    queues: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, after, output = queues
    calls = fake_processes(monkeypatch)
    original_selection = (source / "selection.json.gz").read_bytes()
    queue.follow(output, source, after)
    manifest = grid.read_record(output / "grid.json.gz")
    assert [run["seed"] for run in manifest["runs"]] == [2, 3, 4]
    assert [run["seed"] for run in manifest["test_runs"]] == [0, 1, 2, 3, 4]
    assert [command[0] == "bash" for command in calls] == [True, False] * 3
    original = manifest["x3d_winner"]["runs"][0]
    original_settings = tomllib.loads((source / f"{original['name']}.toml").read_text())
    for run in manifest["runs"]:
        settings = tomllib.loads((output / f"{run['name']}.toml").read_text())
        assert settings == {**original_settings, "seed": run["seed"]}
    for run in manifest["test_runs"]:
        assert grid.run_state(Path(run["test_output"]))["phase"] == "complete"
    assert grid.read_record(output / "grid-status.json.gz")["phase"] == "complete"
    assert (source / "selection.json.gz").read_bytes() == original_selection
    queue.follow(output, source, after)
    assert len(calls) == 6


def test_additional_seeds_read_the_historical_tie_break_name(queues: tuple[Path, Path, Path]) -> None:
    source, after, output = queues
    winner = grid.read_record(source / "selection.json.gz")["ranking"][0]["runs"][0]
    config = source / f"{winner['name']}.toml"
    config.write_text(config.read_text().replace("break_alternation_ties", "force_alternation"))
    queue.initialise(output, source, after)
    for path in output.glob("*.toml"):
        settings = tomllib.loads(path.read_text())
        assert settings["break_alternation_ties"] is True
        assert "force_alternation" not in settings


def test_follower_waits_for_launcher_even_when_grid_already_says_complete(
    queues: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, after, output = queues
    entered = threading.Event()
    wait = queue.wait_for_queue

    def mark_wait(path: Path) -> None:
        entered.set()
        wait(path)

    monkeypatch.setattr(queue, "wait_for_queue", mark_wait)
    monkeypatch.setattr(queue, "execute", lambda _output: None)
    with (after / "runner.lock").open("r") as lock, ThreadPoolExecutor() as executor:
        fcntl.flock(lock, fcntl.LOCK_EX)
        future = executor.submit(queue.follow, output, source, after)
        try:
            assert entered.wait(timeout=5)
            assert not future.done()
            assert not (output / "grid.json.gz").exists()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
        future.result(timeout=5)
    assert (output / "grid.json.gz").exists()


@pytest.mark.parametrize(("dependency", "phase", "exit_code"), [
    ("videomae", "stopped", "1"), ("videomae", "paused", "0"),
    ("x3d", "stopped", "1"), ("x3d", "complete", "1"),
])
def test_failed_or_paused_dependency_never_initialises_training(
    queues: tuple[Path, Path, Path], dependency: str, phase: str, exit_code: str,
) -> None:
    source, after, output = queues
    failed = after if dependency == "videomae" else source
    grid.write_record(failed / "grid-status.json.gz", {"phase": phase})
    (failed / "queue-exit-code.txt").write_text(exit_code)
    with pytest.raises(RuntimeError, match="Dependency did not complete"):
        queue.follow(output, source, after)
    assert not (output / "grid.json.gz").exists()
    assert grid.read_record(output / "grid-status.json.gz")["phase"] == "stopped"


@pytest.mark.parametrize("dependency", ["x3d", "videomae"])
def test_completed_dependency_requires_both_test_outputs(
    queues: tuple[Path, Path, Path], dependency: str,
) -> None:
    source, after, output = queues
    failed = after if dependency == "videomae" else source
    grid.write_record(failed / "test/seed1/status.json.gz", {"phase": "testing"})
    with pytest.raises(RuntimeError, match="test seed 1 is incomplete"):
        queue.follow(output, source, after)
    assert not (output / "grid.json.gz").exists()


def test_pause_before_dependency_finishes_prevents_training_and_continue_resumes(
    queues: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, after, output = queues
    output.mkdir()
    (output / "PAUSE").touch()
    calls = fake_processes(monkeypatch)
    queue.follow(output, source, after)
    assert not calls
    assert grid.read_record(output / "grid-status.json.gz")["phase"] == "paused"
    queue.continue_queue(output)
    queue.follow(output, source, after)
    assert len(calls) == 6


def test_incomplete_test_is_preserved_and_completed_training_is_not_repeated(
    queues: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, after, output = queues
    queue.initialise(output, source, after)
    run = grid.read_record(output / "grid.json.gz")["runs"][0]
    training = output / run["name"]
    training.mkdir()
    grid.write_record(training / "status.json.gz", {"phase": "complete"})
    (training / "exit-code.txt").write_text("0")
    partial = output / "test/seed2"
    partial.mkdir(parents=True)
    marker = partial / "partial-result"
    marker.write_text("preserve")
    calls = fake_processes(monkeypatch)
    with pytest.raises(FileExistsError, match="Incomplete test output"):
        queue.follow(output, source, after)
    assert marker.read_text() == "preserve"
    assert not calls
    assert grid.read_record(output / "grid-status.json.gz")["phase"] == "stopped"


def test_resume_keeps_frozen_recipe_and_passes_resume_for_partial_training(
    queues: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, after, output = queues
    queue.initialise(output, source, after)
    frozen = (output / "grid.json.gz").read_bytes()
    manifest = grid.read_record(output / "grid.json.gz")
    first = output / manifest["runs"][0]["name"]
    first.mkdir()
    grid.write_record(first / "settings.json.gz", {"seed": 2})
    selection = grid.read_record(source / "selection.json.gz")
    selection["ranking"][0]["contact_fraction"] = 0.5
    grid.write_record(source / "selection.json.gz", selection)
    calls = fake_processes(monkeypatch)
    queue.follow(output, source, after)
    assert "--resume" in calls[0]
    assert "--resume" not in calls[2]
    assert (output / "grid.json.gz").read_bytes() == frozen


@pytest.mark.parametrize("failure", ["exit", "status"])
def test_failed_inference_stops_later_seeds(
    queues: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    source, after, output = queues
    calls = fake_processes(monkeypatch)
    wait = queue.wait_for_child

    def finish(process: object) -> int:
        if calls[-1][0] != "bash":
            if failure == "exit":
                return 1
            grid.write_record(output / "test/seed2/status.json.gz", {"phase": "testing"})
        return wait(process)

    monkeypatch.setattr(queue, "wait_for_child", finish)
    with pytest.raises(RuntimeError, match="Test seed 2 failed"):
        queue.follow(output, source, after)
    assert len(calls) == 2
    assert grid.read_record(output / "grid-status.json.gz")["phase"] == "stopped"
