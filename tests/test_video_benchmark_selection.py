"""Check that each model's queue tests the validation winner's lowest-CE checkpoints at stride three."""
import tomllib
from pathlib import Path

import pytest

from experiments.annotator.video_models_benchmark import grid
from experiments.annotator.video_models_benchmark import videomae_queue as queue
from experiments.annotator.video_models_benchmark.grid import read_record, write_record


def test_full_grid_selects_paired_validation_winner_before_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid.initialise(tmp_path, tmp_path / "cache", full_grid=True, test_cache=tmp_path / "test-cache")
    manifest = read_record(tmp_path / "grid.json.gz")
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
        write_record(directory / "status.json.gz", {"phase": "complete"})
        (directory / "exit-code.txt").write_text("0\n")
        winner = run["scheme"] == "B" and run["contact_fraction"] == 0.25 and run["learning_rate"] == 0.005
        # The winner's paired mean (60) beats every other pair (50), despite their stronger seed 0.
        score = 60 if winner else (90 if run["seed"] == 0 else 10)
        write_record(directory / "history.json.gz", [
            {"steps": 30000, "validation": {"loss": 0.2},
             "events": {"10": {"fully_correct_percent": score, "correct_player_f1": 80}}},
            {"steps": 35000, "validation": {"loss": 0.3},
             "events": {"10": {"fully_correct_percent": 99, "correct_player_f1": 99}}},
        ])
    commands = []

    def infer(command: list[str], *, check: bool) -> None:
        assert check
        selection = read_record(tmp_path / "selection.json.gz")["ranking"]
        assert len(selection) == 8
        assert selection[0]["mean_fully_correct_percent"] == 60
        assert all(run["best_step"] == 30000 for row in selection for run in row["runs"])
        commands.append(command)
        destination = Path(command[-1])
        destination.mkdir(parents=True)
        write_record(destination / "status.json.gz", {"phase": "complete"})

    monkeypatch.setattr(grid.subprocess, "run", infer)
    grid.execute(tmp_path, resume=False)
    assert len(commands) == 2
    assert all("B-contact25-lr0.005" in command[command.index("--checkpoint") + 1] for command in commands)
    assert all(Path(command[command.index("--checkpoint") + 1]).name == "best.pt" for command in commands)
    assert all(command[command.index("--stride") + 1] == "3" for command in commands)
    assert all(command[command.index("--offset") + 1] == "0" for command in commands)
    grid.execute(tmp_path, resume=True)
    assert len(commands) == 2
    (tmp_path / runs[-1]["name"] / "exit-code.txt").unlink()
    with pytest.raises(ValueError, match="every configured run"):
        grid.rank_configurations(tmp_path)


def initialise_videomae_queue(tmp_path: Path) -> tuple[Path, list[dict]]:
    output = tmp_path / "queue"
    queue.initialise(output, tmp_path / "cache", Path(queue.__file__).with_name("videomae.toml"))
    return output, read_record(output / "grid.json.gz")["runs"]


def finish_continued_run(output: Path, run: dict) -> None:
    directory = output / run["name"]
    directory.mkdir(exist_ok=True)
    write_record(directory / "status.json.gz", {"phase": "complete"})
    (directory / "exit-code.txt").write_text("0\n")
    (directory / "continuation.json.gz").touch()


def test_test_dispatch_uses_both_seeds_of_the_mean_validation_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, runs = initialise_videomae_queue(tmp_path)
    for run in runs:
        finish_continued_run(output, run)
        # The 1e-4 pair wins on its mean (60 against 45), despite the other rate's best single seed.
        score = 60 if run["learning_rate"] == 1e-4 else (90 if run["seed"] == 0 else 0)
        write_record(output / run["name"] / "history.json.gz", [
            {"steps": 40000, "validation": {"loss": 0.1},
             "events": {"10": {"fully_correct_percent": score, "correct_player_f1": 80}}},
            {"steps": 45000, "validation": {"loss": 0.2},
             "events": {"10": {"fully_correct_percent": 99, "correct_player_f1": 99}}},
        ])
    commands = []

    class Inference:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            ranking = read_record(output / "selection.json.gz")["ranking"]
            assert ranking[0]["learning_rate"] == 1e-4
            commands.append(command)
            destination = Path(command[-1])
            destination.mkdir()
            write_record(destination / "status.json.gz", {"phase": "complete"})

    monkeypatch.setattr(queue.subprocess, "Popen", Inference)
    monkeypatch.setattr(queue, "wait_for_child", lambda *_args: 0)
    queue.evaluate_selected(output, tmp_path / "test-cache")
    assert len(commands) == 2
    for seed, command in enumerate(commands):
        assert str(output / f"videomae-lr0.0001-seed{seed}" / "best.pt") in command
        assert command[command.index("--stride") + 1] == "3"
        assert command[command.index("--offset") + 1] == "0"
    queue.evaluate_selected(output, tmp_path / "test-cache")
    assert len(commands) == 2
