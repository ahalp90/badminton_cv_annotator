"""Queue boundaries for independently recoverable benchmark runs."""
import tomllib
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
