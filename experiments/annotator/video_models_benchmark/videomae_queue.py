"""Run both VideoMAE learning rates at two seeds, finishing each continuation."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from .grid import (
    completed_run,
    continue_queue,
    read_record,
    run_state,
    write_record,
)

LEARNING_RATES = (3.125e-5, 1e-4)


def initialise(
    output: Path, cache: Path, config: Path, *, test_cache: Path | None = None,
    x3d_winner: dict | None = None,
) -> None:
    from .train import Settings, normalise_settings_record

    settings = normalise_settings_record(tomllib.loads(config.read_text()))
    settings.setdefault("validation_stride", 3)
    scheme = "A"
    if x3d_winner is not None:
        settings.update(contact_fraction=x3d_winner["contact_fraction"], validation_stride=3,
                        break_alternation_ties=True, continue_until_patience=True)
        scheme = x3d_winner["scheme"]
    Settings(**settings)
    if settings.get("model_name") != "videomae" or settings.get("contact_fraction") not in (0.25, 0.5):
        raise ValueError("The VideoMAE queue requires VideoMAE and 25% or 50% contact sampling")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "grid.json.gz"
    if manifest.exists():
        raise FileExistsError(f"Queue already exists: {manifest}")
    runs = []
    for seed in (0, 1):
        for learning_rate in LEARNING_RATES:
            name = f"videomae-lr{learning_rate:g}-seed{seed}"
            run_settings = {**settings, "seed": seed, "learning_rate": learning_rate,
                            "warmup_start_lr": learning_rate / 10}
            lines = [f"{key} = {json.dumps(value)}\n" for key, value in run_settings.items()]
            (output / f"{name}.toml").write_text("".join(lines))
            runs.append({"name": name, "seed": seed, "learning_rate": learning_rate})
    write_record(manifest, {"cache": str(cache.resolve()), "created_at": time.time(), "runs": runs,
                            "scheme": scheme, "x3d_winner": x3d_winner,
                            "labels": ("positive within ±1 frame; distance-two centres ignored" if scheme == "A"
                                       else "positive within ±2 frames"),
                            "settings": settings, "test_cache": str(test_cache.resolve()) if test_cache else None})


def follow_x3d(output: Path, source: Path, config: Path) -> None:
    """Wait for X3D, freeze its validation winner's settings, then run VideoMAE."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "follow.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"pid": os.getpid(), "phase": "waiting", "source_grid": str(source.resolve()),
                 "updated_at": time.time()}
        write_record(output / "grid-status.json.gz", state)
        print("Waiting for X3D training and its selected test runs to finish.", flush=True)
        try:
            with (source / "grid.lock").open("r") as source_lock:
                fcntl.flock(source_lock, fcntl.LOCK_EX)
                if read_record(source / "grid-status.json.gz")["phase"] != "complete":
                    raise RuntimeError("X3D did not complete successfully; VideoMAE has not started")
                if not (output / "grid.json.gz").exists():
                    manifest = read_record(source / "grid.json.gz")
                    if not manifest.get("full_grid") or not manifest.get("test_cache"):
                        raise ValueError("The follow-up requires the full X3D grid and a test cache")
                    winner = read_record(source / "selection.json.gz")["ranking"][0]
                    initialise(output, Path(manifest["cache"]), config,
                               test_cache=Path(manifest["test_cache"]), x3d_winner=winner)
            execute(output, resume=True)
        except BaseException:
            state.update(phase="stopped", updated_at=time.time())
            write_record(output / "grid-status.json.gz", state)
            raise


def wait_for_child(process: subprocess.Popen) -> int:
    received_signal = None

    def forward_termination(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum
        os.killpg(process.pid, signal.SIGTERM)

    signals = [signal.SIGTERM, signal.SIGINT]
    # Preserve nohup's ignored hangup signal for unattended queues.
    if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:
        signals.append(signal.SIGHUP)
    previous_handlers = {signum: signal.signal(signum, forward_termination) for signum in signals}
    try:
        # Reap here: calling wait again inside the signal handler deadlocks.
        return_code = process.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if received_signal is not None:
        raise SystemExit(128 + received_signal)
    return return_code


def launch_run(output: Path, manifest: dict, run: dict, *, extend: bool, resume: bool) -> None:
    directory = output / run["name"]
    config = output / f"{run['name']}.toml"
    settings = tomllib.loads(config.read_text())
    command = ["bash", str(Path(__file__).with_name("launch.sh")), sys.executable,
               manifest["cache"], str(directory), manifest.get("scheme", "A"),
               str(settings["warmup_steps"]), "--config", str(config)]
    if extend:
        command.append("--extend")
    if (directory / "settings.json.gz").exists():
        if not resume:
            raise FileExistsError(f"Unfinished run {directory}; inspect it, then use run --resume")
        command.append("--resume")
    process = subprocess.Popen(command, start_new_session=True)
    return_code = wait_for_child(process)
    if return_code:
        raise RuntimeError(f"{run['name']} stopped with exit {return_code}; see {directory / 'train.log'}")
    if run_state(directory).get("phase") != "paused" and not completed_run(directory):
        raise RuntimeError(f"{run['name']} exited without completing or pausing")


def evaluate_selected(output: Path, cache: Path) -> None:
    ranking = rank_runs(output)
    write_record(output / "selection.json.gz", {"ranking": ranking})
    test_root = output / "test"
    test_root.mkdir(exist_ok=True)
    for run in ranking[0]["runs"]:
        directory = test_root / f"seed{run['seed']}"
        if run_state(directory).get("phase") == "complete":
            continue
        if directory.exists():
            raise FileExistsError(f"Incomplete test output {directory}; preserve it elsewhere before resuming")
        command = [sys.executable, "-u", "-m", "experiments.annotator.video_models_benchmark.infer",
                   "--checkpoint", str(output / run["name"] / "best.pt"), "--cache", str(cache),
                   "--role", "test", "--stride", "3", "--offset", "0", "--output", str(directory)]
        print(f"Testing {run['name']}", flush=True)
        with (test_root / f"seed{run['seed']}.log").open("a") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            return_code = wait_for_child(process)
        if return_code or run_state(directory).get("phase") != "complete":
            raise RuntimeError(f"Test seed {run['seed']} failed; see {log.name}")


def execute(output: Path, *, resume: bool) -> None:
    manifest = read_record(output / "grid.json.gz")
    integrated = manifest["settings"].get("continue_until_patience", False)
    with (output / "grid.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"pid": os.getpid(), "phase": "running", "updated_at": time.time()}
        try:
            for run in manifest["runs"]:
                directory = output / run["name"]
                continuation = directory / "continuation.json.gz"
                if completed_run(directory) and continuation.exists():
                    continue
                for extend in ((False,) if integrated else (False, True)):
                    if not integrated and not extend and (continuation.exists() or completed_run(directory)):
                        continue
                    if (output / "PAUSE").exists() or (directory / "PAUSE").exists():
                        state["phase"] = "paused"
                        return
                    state.update(current=run["name"], continuation=extend, updated_at=time.time())
                    write_record(output / "grid-status.json.gz", state)
                    launch_run(output, manifest, run, extend=extend, resume=resume or extend)
                    if run_state(directory).get("phase") == "paused":
                        state["phase"] = "paused"
                        return
            if manifest.get("test_cache"):
                if (output / "PAUSE").exists():
                    state["phase"] = "paused"
                    return
                state.update(phase="testing", continuation=False)
                write_record(output / "grid-status.json.gz", state)
                evaluate_selected(output, Path(manifest["test_cache"]))
            state["phase"] = "complete"
        except BaseException:
            state["phase"] = "stopped"
            raise
        finally:
            state["updated_at"] = time.time()
            write_record(output / "grid-status.json.gz", state)


def rank_runs(output: Path) -> list[dict]:
    groups: dict[float, list[dict]] = {}
    for run in read_record(output / "grid.json.gz")["runs"]:
        directory = output / run["name"]
        if not completed_run(directory) or not (directory / "continuation.json.gz").exists():
            raise ValueError("Both learning rates and both seeds must finish before selection")
        best = min(read_record(directory / "history.json.gz"), key=lambda row: row["validation"]["loss"])
        events = best["events"]["10"]
        groups.setdefault(run["learning_rate"], []).append({
            **run, "best_step": best["steps"], "validation_loss": best["validation"]["loss"],
            "fully_correct_percent": events["fully_correct_percent"],
            "correct_player_f1": events["correct_player_f1"],
        })
    if sorted(groups) != list(LEARNING_RATES):
        raise ValueError("Selection requires both pilot learning rates")
    ranking = []
    for rate, runs in groups.items():
        if sorted(run["seed"] for run in runs) != [0, 1]:
            raise ValueError("Each learning rate requires seeds 0 and 1")
        ranking.append({"learning_rate": rate, "runs": runs,
                        "mean_fully_correct_percent": sum(run["fully_correct_percent"] for run in runs) / 2,
                        "mean_correct_player_f1": sum(run["correct_player_f1"] for run in runs) / 2})
    return sorted(ranking, key=lambda row: (-row["mean_fully_correct_percent"],
                                           -row["mean_correct_player_f1"], row["learning_rate"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "run", "follow-x3d", "pause", "continue", "status", "rank"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--x3d-output", type=Path, help="Wait for this X3D grid and inherit its validation settings")
    parser.add_argument("--test-cache", type=Path, help="After training, test both seeds of the validation-selected LR")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("videomae.toml"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "init":
        if args.cache is None:
            parser.error("init requires --cache")
        initialise(args.output, args.cache, args.config, test_cache=args.test_cache)
    elif args.command == "follow-x3d":
        if args.x3d_output is None:
            parser.error("follow-x3d requires --x3d-output")
        follow_x3d(args.output, args.x3d_output, args.config)
    elif args.command == "run":
        execute(args.output, resume=args.resume)
    elif args.command == "continue":
        continue_queue(args.output)
    elif args.command == "rank":
        ranking = rank_runs(args.output)
        write_record(args.output / "selection.json.gz", {"ranking": ranking})
        print(json.dumps(ranking, indent=2))
    else:
        if not (args.output / "grid.json.gz").exists():
            if args.command == "pause":
                (args.output / "PAUSE").touch()
            state = read_record(args.output / "grid-status.json.gz")
            print(f"VideoMAE: {state['phase']} | waiting for the X3D validation winner")
            return
        manifest = read_record(args.output / "grid.json.gz")
        if args.command == "pause":
            (args.output / "PAUSE").touch()
        for run in manifest["runs"]:
            directory = args.output / run["name"]
            if args.command == "pause":
                directory.mkdir(exist_ok=True)
                (directory / "PAUSE").touch()
            else:
                state = run_state(directory)
                print(run["name"], state.get("phase", "pending"), state.get("stage", "initial"),
                      f"step {state.get('step', 0)}")


if __name__ == "__main__":
    main()
