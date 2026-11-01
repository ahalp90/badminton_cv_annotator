"""Run a small sequential grid and inspect or pause it without importing PyTorch."""
from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import os
import signal
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

from .status import show

VALIDATION_STEPS = [2500, 5000, 10000, 15000, 20000, 25000, 30000]


def read_record(path: Path) -> Any:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def write_record(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with gzip.open(temporary, "wt") as stream:
        json.dump(value, stream, indent=2)
    temporary.replace(path)


def initialise(output: Path, cache: Path) -> None:
    """Freeze the queue before training; repeat seeds use identical settings."""
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "grid.json.gz"
    if manifest.exists():
        raise FileExistsError(f"Grid already exists: {manifest}")
    runs = []
    for seed in (0, 1):
        for contact_fraction, learning_rate in ((0.5, 0.02), (0.5, 0.005), (0.25, 0.02), (0.25, 0.005)):
            name = f"A-contact{int(100 * contact_fraction)}-lr{learning_rate:g}-seed{seed}"
            config = output / f"{name}.toml"
            config.write_text(
                f"seed = {seed}\nbatch_size = 16\nworkers = 12\n"
                f"learning_rate = {learning_rate}\nwarmup_start_lr = {learning_rate / 10}\n"
                f"contact_fraction = {contact_fraction}\nmomentum = 0.9\nweight_decay = 0.00001\n"
                f"max_steps = 30000\nwarmup_steps = 2000\nvalidation_steps = {VALIDATION_STEPS}\n"
                'precision = "bf16"\n'
            )
            runs.append({"name": name, "seed": seed, "contact_fraction": contact_fraction,
                         "learning_rate": learning_rate, "scheme": "A"})
    write_record(manifest, {"cache": str(cache.resolve()), "created_at": time.time(), "runs": runs,
                            "max_steps": 30000, "validation_steps": VALIDATION_STEPS,
                            "estimated_run_hours": (30000 * 0.209034 + 7 * 10.41 * 60) / 3600})


def run_state(directory: Path) -> dict:
    path = directory / "status.json.gz"
    return read_record(path) if path.exists() else {}


def completed_run(directory: Path) -> bool:
    exit_path = directory / "exit-code.txt"
    return (exit_path.exists() and exit_path.read_text().strip() == "0"
            and run_state(directory).get("phase") == "complete")


def terminate_child(process: subprocess.Popen, signum: int, _frame: object) -> None:
    os.killpg(process.pid, signal.SIGTERM)
    process.wait()
    raise SystemExit(128 + signum)


def execute(output: Path, *, resume: bool) -> None:
    manifest = read_record(output / "grid.json.gz")
    with (output / "grid.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "PAUSE").exists():
            print("Queue paused; use the continue command before resuming.", flush=True)
            return
        active = {"pid": os.getpid(), "phase": "starting", "updated_at": time.time()}
        write_record(output / "grid-status.json.gz", active)
        try:
            for run in manifest["runs"]:
                directory = output / run["name"]
                if completed_run(directory):
                    continue
                if (output / "PAUSE").exists():
                    active.update(phase="paused")
                    break
                config = output / f"{run['name']}.toml"
                command = ["bash", str(Path(__file__).with_name("launch.sh")), sys.executable,
                           manifest["cache"], str(directory), run["scheme"], "2000", "--config", str(config)]
                if (directory / "settings.json.gz").exists():
                    if not resume:
                        raise FileExistsError(f"Unfinished run {directory}; inspect it, then use run --resume")
                    command.append("--resume")
                active.update(phase="running", current=run["name"], updated_at=time.time())
                write_record(output / "grid-status.json.gz", active)
                print(f"Starting {run['name']}", flush=True)
                process = subprocess.Popen(command, start_new_session=True)

                previous_handlers = {signum: signal.signal(signum, partial(terminate_child, process))
                                     for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
                try:
                    return_code = process.wait()
                finally:
                    for signum, handler in previous_handlers.items():
                        signal.signal(signum, handler)
                if return_code:
                    raise RuntimeError(f"{run['name']} stopped with exit {return_code}; see {directory / 'train.log'}")
                if run_state(directory).get("phase") == "paused":
                    active.update(phase="paused")
                    break
                if not completed_run(directory):
                    raise RuntimeError(f"{run['name']} exited without a completed run")
                print(f"Finished {run['name']}", flush=True)
            else:
                active.update(phase="complete")
        except BaseException:
            active.update(phase="stopped")
            raise
        finally:
            active.update(updated_at=time.time())
            write_record(output / "grid-status.json.gz", active)


def pause(output: Path) -> None:
    (output / "PAUSE").touch()
    # All unfinished runs receive the marker, including a run just being launched.
    for run in read_record(output / "grid.json.gz")["runs"]:
        directory = output / run["name"]
        if not completed_run(directory):
            directory.mkdir(exist_ok=True)
            (directory / "PAUSE").touch()
    print("Pause requested: the active run saves at its next validation boundary; no later run starts.")


def continue_queue(output: Path) -> None:
    for run in read_record(output / "grid.json.gz")["runs"]:
        (output / run["name"] / "PAUSE").unlink(missing_ok=True)
    (output / "PAUSE").unlink(missing_ok=True)
    print("Pause markers cleared. Start the queue with run --resume.")


def display(output: Path) -> bool:
    manifest = read_record(output / "grid.json.gz")
    state_path = output / "grid-status.json.gz"
    state = read_record(state_path) if state_path.exists() else {"phase": "not started"}
    phase = state["phase"]
    if phase in ("running", "starting"):
        try:
            os.kill(state["pid"], 0)
        except ProcessLookupError:
            phase = "stopped (queue process absent)"
    finished = 0
    updates = 0
    print(f"GRID {phase.upper()} | {output}")
    print("Run                                  State          Updates    Best CE  Rally%  Player F1  Pred/GT")
    for run in manifest["runs"]:
        directory = output / run["name"]
        status = run_state(directory)
        done = completed_run(directory)
        finished += done
        step = status.get("step", 0)
        updates += step
        history_path = directory / "history.json.gz"
        metrics = "—"
        if history_path.exists():
            history = read_record(history_path)
            if history:
                best = min(history, key=lambda row: row["validation"]["loss"])
                events = best["events"]["10"]
                ratio = events["predicted"] / events["labelled"]
                metrics = (f"{best['validation']['loss']:7.4f} {events['fully_correct_percent']:7.2f} "
                           f"{events['correct_player_f1']:9.2f} {ratio:8.2f}")
        run_phase = "complete" if done else status.get("phase", "queued")
        print(f"{run['name']:36} {run_phase:14} {step:7,d}    {metrics}")
    total = len(manifest["runs"]) * manifest["max_steps"]
    print(f"\n{finished}/{len(manifest['runs'])} runs complete | {updates:,}/{total:,} updates ({100 * updates / total:.1f}%)")
    print(f"Initial estimate: {len(manifest['runs']) * manifest['estimated_run_hours']:.1f}h excluding frame preparation/startup/scoring.")
    print("Metrics use the lowest-CE checkpoint at ±10 frames (scaled by fps). Partial runs can change.")
    if phase == "running" and "current" in state:
        print()
        show(output / state["current"])
    return phase != "running" and phase != "starting"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "run", "status", "pause", "continue"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "init":
        if args.cache is None:
            parser.error("init requires --cache")
        initialise(output, args.cache)
    elif args.command == "run":
        execute(output, resume=args.resume)
    elif args.command == "pause":
        pause(output)
    elif args.command == "continue":
        continue_queue(output)
    else:
        while True:
            if args.watch and sys.stdout.isatty():
                print("\033[H\033[J", end="")
            finished = display(output)
            if not args.watch or finished:
                break
            time.sleep(15)


if __name__ == "__main__":
    main()
