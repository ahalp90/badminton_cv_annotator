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
from functools import partial
from pathlib import Path

from .grid import (
    completed_run,
    continue_queue,
    read_record,
    run_state,
    terminate_child,
    write_record,
)

LEARNING_RATES = (3.125e-5, 1e-4)


def initialise(output: Path, cache: Path, config: Path) -> None:
    from .train import Settings

    settings = tomllib.loads(config.read_text())
    Settings(**settings)
    if settings.get("model_name") != "videomae" or settings.get("contact_fraction") != 0.5:
        raise ValueError("The VideoMAE pilot requires VideoMAE and 50% contact sampling")
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
                            "labels": "positive within ±1 frame; distance-two centres ignored",
                            "settings": settings})


def launch_run(output: Path, manifest: dict, run: dict, *, extend: bool, resume: bool) -> None:
    directory = output / run["name"]
    config = output / f"{run['name']}.toml"
    settings = tomllib.loads(config.read_text())
    command = ["bash", str(Path(__file__).with_name("launch.sh")), sys.executable,
               manifest["cache"], str(directory), "A", str(settings["warmup_steps"]), "--config", str(config)]
    if extend:
        command.append("--extend")
    if (directory / "settings.json.gz").exists():
        if not resume:
            raise FileExistsError(f"Unfinished run {directory}; inspect it, then use run --resume")
        command.append("--resume")
    process = subprocess.Popen(command, start_new_session=True)
    signals = [signal.SIGTERM, signal.SIGINT]
    # Preserve nohup's ignored hangup signal for unattended queues.
    if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:
        signals.append(signal.SIGHUP)
    previous_handlers = {signum: signal.signal(signum, partial(terminate_child, process))
                         for signum in signals}
    try:
        return_code = process.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if return_code:
        raise RuntimeError(f"{run['name']} stopped with exit {return_code}; see {directory / 'train.log'}")
    if run_state(directory).get("phase") != "paused" and not completed_run(directory):
        raise RuntimeError(f"{run['name']} exited without completing or pausing")


def execute(output: Path, *, resume: bool) -> None:
    manifest = read_record(output / "grid.json.gz")
    with (output / "grid.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"pid": os.getpid(), "phase": "running", "updated_at": time.time()}
        try:
            for run in manifest["runs"]:
                directory = output / run["name"]
                continuation = directory / "continuation.json.gz"
                if completed_run(directory) and continuation.exists():
                    continue
                for extend in (False, True):
                    if not extend and (continuation.exists() or completed_run(directory)):
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
    parser.add_argument("command", choices=("init", "run", "pause", "continue", "status", "rank"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("videomae.toml"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "init":
        if args.cache is None:
            parser.error("init requires --cache")
        initialise(args.output, args.cache, args.config)
    elif args.command == "run":
        execute(args.output, resume=args.resume)
    elif args.command == "continue":
        continue_queue(args.output)
    elif args.command == "rank":
        ranking = rank_runs(args.output)
        write_record(args.output / "selection.json.gz", {"ranking": ranking})
        print(json.dumps(ranking, indent=2))
    else:
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
