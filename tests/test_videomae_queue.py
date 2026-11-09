"""Run the VideoMAE queue logic without launching training or a GPU."""
import fcntl
import signal
import subprocess
import sys
import textwrap
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
        pid = 123456

        def wait(self) -> int:
            signal.raise_signal(signal.SIGHUP)
            finish_run(output, runs[0])
            return 0

    child = Child()
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *_args, **_kwargs: child)
    monkeypatch.setattr(queue.os, "killpg", lambda _pid, signum: terminations.append(signum))
    inherited = signal.SIG_IGN if under_nohup else signal.SIG_DFL
    original = signal.signal(signal.SIGHUP, inherited)
    try:
        if under_nohup:
            queue.launch_run(output, {"cache": str(tmp_path / "cache")}, runs[0], extend=False, resume=False)
        else:
            with pytest.raises(SystemExit) as stopped:
                queue.launch_run(output, {"cache": str(tmp_path / "cache")}, runs[0], extend=False, resume=False)
            assert stopped.value.code == 128 + signal.SIGHUP
        assert terminations == ([] if under_nohup else [signal.SIGTERM])
        assert signal.getsignal(signal.SIGHUP) == inherited
    finally:
        signal.signal(signal.SIGHUP, original)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_during_real_child_wait_exits_without_hanging(signum: signal.Signals) -> None:
    script = textwrap.dedent("""
        import os
        import signal
        import subprocess
        import sys
        import threading
        from experiments.annotator.video_models_benchmark.videomae_queue import wait_for_child

        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)'], start_new_session=True)
        threading.Timer(0.2, os.kill, args=(os.getpid(), int(sys.argv[1]))).start()
        try:
            wait_for_child(child)
        finally:
            assert child.poll() == -signal.SIGTERM
    """)
    result = subprocess.run([sys.executable, "-c", script, str(int(signum))],
                            check=False, capture_output=True, text=True, timeout=5)
    assert result.returncode == 128 + signum, result.stderr


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


def test_queue_tests_only_after_all_training_and_continuations_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "queue"
    test_cache = tmp_path / "test-cache"
    queue.initialise(output, tmp_path / "cache", Path(queue.__file__).with_name("videomae.toml"), test_cache=test_cache)
    runs = read_record(output / "grid.json.gz")["runs"]

    def finish(output: Path, _manifest: dict, run: dict, *, extend: bool, resume: bool) -> None:
        finish_run(output, run, continuation=extend)

    def test_finished_runs(directory: Path, cache: Path) -> None:
        assert directory == output and cache == test_cache
        assert all((directory / run["name"] / "continuation.json.gz").exists() for run in runs)
        assert read_record(output / "grid-status.json.gz")["phase"] == "testing"

    monkeypatch.setattr(queue, "launch_run", finish)
    monkeypatch.setattr(queue, "evaluate_selected", test_finished_runs)
    queue.execute(output, resume=False)
    assert read_record(output / "grid-status.json.gz")["phase"] == "complete"


def test_automatic_test_uses_validation_winner_and_skips_completed_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, runs = initialise(tmp_path)
    for run in runs:
        finish_run(output, run, continuation=True)
        events = {"fully_correct_percent": 70 if run["learning_rate"] == 1e-4 else 50, "correct_player_f1": 80}
        write_record(output / run["name"] / "history.json.gz", [
            {"steps": 30000, "validation": {"loss": 0.4}, "events": {"10": events}},
        ])
    commands = []

    class Inference:
        def __init__(self, command: list[str], **_kwargs) -> None:
            commands.append(command)
            self.directory = Path(command[command.index("--output") + 1])

        def wait(self) -> int:
            self.directory.mkdir()
            write_record(self.directory / "status.json.gz", {"phase": "complete"})
            return 0

    monkeypatch.setattr(queue.subprocess, "Popen", Inference)
    queue.evaluate_selected(output, tmp_path / "test-cache")
    queue.evaluate_selected(output, tmp_path / "test-cache")
    assert len(commands) == 2
    for command, run in zip(commands, (runs[1], runs[3]), strict=True):
        assert command[command.index("--checkpoint") + 1] == str(output / run["name"] / "best.pt")
        assert command[command.index("--role") + 1] == "test"
        assert command[command.index("--stride") + 1] == "3"
        assert command[command.index("--offset") + 1] == "0"
        assert command[command.index("--cache") + 1] == str(tmp_path / "test-cache")
    assert read_record(output / "selection.json.gz")["ranking"][0]["learning_rate"] == 1e-4


@pytest.mark.parametrize("scheme", ["A", "B"])
@pytest.mark.parametrize("fraction", [0.25, 0.5])
def test_x3d_winner_sets_both_labels_and_sampling(tmp_path: Path, scheme: str, fraction: float) -> None:
    import tomllib

    winner = {"scheme": scheme, "contact_fraction": fraction, "learning_rate": 0.02}
    queue.initialise(tmp_path, tmp_path / "cache", Path(queue.__file__).with_name("videomae.toml"),
                     test_cache=tmp_path / "test-cache", x3d_winner=winner)
    manifest = read_record(tmp_path / "grid.json.gz")
    assert manifest["scheme"] == scheme
    assert manifest["x3d_winner"] == winner
    assert len(manifest["runs"]) == 4
    for run in manifest["runs"]:
        config = tomllib.loads((tmp_path / f"{run['name']}.toml").read_text())
        assert config["contact_fraction"] == fraction
        assert config["validation_stride"] == 3
        assert config["break_alternation_ties"] and config["continue_until_patience"]
        assert config["learning_rate"] in queue.LEARNING_RATES
        assert config["max_steps"] == 30000
        assert "max_total_steps" not in config


def test_integrated_queue_resumes_without_skipping_or_reextending_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue.initialise(tmp_path, tmp_path / "cache", Path(queue.__file__).with_name("videomae.toml"),
                     x3d_winner={"scheme": "B", "contact_fraction": 0.25})
    manifest = read_record(tmp_path / "grid.json.gz")
    runs = manifest["runs"]
    finish_run(tmp_path, runs[0], phase="paused", continuation=True)
    calls = []

    def finish(output: Path, manifest: dict, run: dict, *, extend: bool, resume: bool) -> None:
        assert manifest["scheme"] == "B"
        assert resume and not extend
        calls.append(run["name"])
        finish_run(output, run, continuation=True)

    monkeypatch.setattr(queue, "launch_run", finish)
    queue.execute(tmp_path, resume=True)
    assert calls == [run["name"] for run in runs]
    queue.execute(tmp_path, resume=True)
    assert len(calls) == 4


def x3d_source(tmp_path: Path, phase: str = "complete") -> Path:
    source = tmp_path / "x3d"
    source.mkdir()
    (source / "grid.lock").touch()
    write_record(source / "grid-status.json.gz", {"phase": phase})
    write_record(source / "grid.json.gz", {"full_grid": True, "cache": str(tmp_path / "cache"),
                                           "test_cache": str(tmp_path / "test-cache")})
    write_record(source / "selection.json.gz", {"ranking": [
        {"scheme": "B", "contact_fraction": 0.25, "learning_rate": 0.005},
    ]})
    return source


def test_followup_waits_for_source_lock_and_freezes_its_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    source = x3d_source(tmp_path)
    output = tmp_path / "videomae"
    waiting, executed = threading.Event(), threading.Event()

    def write(path: Path, value: dict) -> None:
        write_record(path, value)
        if value.get("phase") == "waiting":
            waiting.set()

    def execute(directory: Path, *, resume: bool) -> None:
        assert directory == output and resume
        manifest = read_record(directory / "grid.json.gz")
        assert manifest["scheme"] == "B"
        assert manifest["settings"]["contact_fraction"] == 0.25
        assert manifest["test_cache"] == str(tmp_path / "test-cache")
        executed.set()

    monkeypatch.setattr(queue, "write_record", write)
    monkeypatch.setattr(queue, "execute", execute)
    with (source / "grid.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(queue.follow_x3d, output, source,
                                     Path(queue.__file__).with_name("videomae.toml"))
            try:
                assert waiting.wait(5)
                assert not executed.wait(0.05)
                assert not (output / "grid.json.gz").exists()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
            future.result(timeout=5)
    assert executed.is_set()
    write_record(source / "selection.json.gz", {"ranking": [{"scheme": "A", "contact_fraction": 0.5}]})
    queue.follow_x3d(output, source, Path(queue.__file__).with_name("videomae.toml"))


@pytest.mark.parametrize("phase", ["paused", "stopped", "running"])
def test_followup_refuses_an_unsuccessful_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    source = x3d_source(tmp_path, phase)
    output = tmp_path / "videomae"
    monkeypatch.setattr(queue, "execute", lambda *args, **kwargs: pytest.fail("Must not start"))
    with pytest.raises(RuntimeError, match="X3D did not complete successfully"):
        queue.follow_x3d(output, source, Path(queue.__file__).with_name("videomae.toml"))
    assert not (output / "grid.json.gz").exists()
    assert read_record(output / "grid-status.json.gz")["phase"] == "stopped"


def test_test_dispatch_uses_both_seeds_of_the_mean_validation_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, runs = initialise(tmp_path)
    for run in runs:
        finish_run(output, run, continuation=True)
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
