"""Run a small sequential grid and inspect or pause it without importing PyTorch."""
from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from functools import partial
from itertools import product
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


def initialise(output: Path, cache: Path, *, full_grid: bool = False, test_cache: Path | None = None) -> None:
    """Freeze the queue before training; repeat seeds use identical settings."""
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "grid.json.gz"
    if manifest.exists():
        raise FileExistsError(f"Grid already exists: {manifest}")
    runs = []
    for seed, scheme in product((0, 1), ("A", "B") if full_grid else ("A",)):
        for contact_fraction, learning_rate in ((0.5, 0.02), (0.5, 0.005), (0.25, 0.02), (0.25, 0.005)):
            name = f"{scheme}-contact{int(100 * contact_fraction)}-lr{learning_rate:g}-seed{seed}"
            config = output / f"{name}.toml"
            config.write_text(
                f"seed = {seed}\nbatch_size = 16\nworkers = 12\n"
                f"learning_rate = {learning_rate}\nwarmup_start_lr = {learning_rate / 10}\n"
                f"contact_fraction = {contact_fraction}\nmomentum = 0.9\nweight_decay = 0.00001\n"
                f"max_steps = 30000\nwarmup_steps = 2000\nvalidation_steps = {VALIDATION_STEPS}\n"
                'validation_stride = 3\nprecision = "bf16"\n'
                f'continue_until_patience = {str(full_grid).lower()}\n'
                f'break_alternation_ties = {str(full_grid).lower()}\n'
            )
            runs.append({"name": name, "seed": seed, "contact_fraction": contact_fraction,
                         "learning_rate": learning_rate, "scheme": scheme})
    write_record(manifest, {"cache": str(cache.resolve()), "created_at": time.time(), "runs": runs,
                            "max_steps": 30000, "validation_steps": VALIDATION_STEPS,
                            "validation_stride": 3,
                            "full_grid": full_grid, "validation_offset": 0,
                            "break_alternation_ties": full_grid,
                            "test_cache": str(test_cache.resolve()) if test_cache is not None else None,
                            "estimated_run_hours": (30000 * 0.209034 + 7 * 10.41 * 60 / 3) / 3600})


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
                if manifest.get("stage") == "continuation":
                    command.append("--extend")
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
                if manifest.get("full_grid"):
                    ranking = rank_configurations(output)
                    write_record(output / "selection.json.gz", {"ranking": ranking})
                    if manifest.get("test_cache"):
                        active.update(phase="testing", updated_at=time.time())
                        write_record(output / "grid-status.json.gz", active)
                        evaluate_selected(output, Path(manifest["test_cache"]), ranking[0])
                active.update(phase="complete")
        except BaseException:
            active.update(phase="stopped")
            raise
        finally:
            active.update(updated_at=time.time())
            write_record(output / "grid-status.json.gz", active)


def extend_grid(output: Path) -> None:
    """Wait for the original queue lock, then continue copies of its finished runs."""
    destination = output / "continuation"
    destination.mkdir(exist_ok=True)
    with (destination / "wait.lock").open("w") as wait_lock:
        fcntl.flock(wait_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read_record(output / "grid.json.gz")
        if not (destination / "grid.json.gz").exists():
            for run in manifest["runs"]:
                config = f"{run['name']}.toml"
                shutil.copy2(output / config, destination / config)
            write_record(destination / "grid.json.gz", {
                **manifest, "stage": "continuation", "source_grid": str(output), "max_steps": None,
            })
        waiting = {"pid": os.getpid(), "phase": "waiting", "updated_at": time.time()}
        write_record(destination / "grid-status.json.gz", waiting)
        print(f"Waiting for the original grid to complete: {output}", flush=True)
        try:
            with (output / "grid.lock").open("w") as original_lock:
                fcntl.flock(original_lock, fcntl.LOCK_EX)
                if (output / "PAUSE").exists() or (destination / "PAUSE").exists():
                    write_record(destination / "grid-status.json.gz", {**waiting, "phase": "paused"})
                    return
                original = read_record(output / "grid-status.json.gz")
                if original["phase"] != "complete" or not all(
                    completed_run(output / run["name"]) for run in manifest["runs"]
                ):
                    raise RuntimeError("Original grid did not complete successfully; continuation will not start")
                for run in manifest["runs"]:
                    source, target = output / run["name"], destination / run["name"]
                    target.mkdir(exist_ok=True)
                    if (target / "settings.json.gz").exists():
                        continue
                    # Settings is copied last so an interrupted initial copy can be repeated.
                    filenames = ["last.pt", "best.pt", "history.json.gz", "validation_events.json.gz",
                                 "validation_probabilities.npy.xz", "settings.json.gz"]
                    if (source / "training_progress.json.gz").exists():
                        filenames.insert(0, "training_progress.json.gz")
                    for filename in filenames:
                        shutil.copy2(source / filename, target / filename)
            execute(destination, resume=True)
        except BaseException:
            state = read_record(destination / "grid-status.json.gz")
            write_record(destination / "grid-status.json.gz", {**state, "phase": "stopped", "updated_at": time.time()})
            raise


def rank_configurations(output: Path) -> list[dict]:
    """Rank paired seeds using event scores at each run's lowest-CE checkpoint."""
    manifest = read_record(output / "grid.json.gz")
    schemes = ("A", "B") if manifest.get("full_grid") else ("A",)
    groups: dict[tuple[str, float, float], list[dict]] = {}
    for run in manifest["runs"]:
        if run["scheme"] not in schemes or not completed_run(output / run["name"]):
            raise ValueError("Selection requires every configured run to complete")
        history = read_record(output / run["name"] / "history.json.gz")
        best = min(history, key=lambda row: row["validation"]["loss"])
        events = best["events"]["10"]
        key = run["scheme"], run["contact_fraction"], run["learning_rate"]
        groups.setdefault(key, []).append({
            **run, "best_step": best["steps"], "validation_loss": best["validation"]["loss"],
            "fully_correct_percent": events["fully_correct_percent"],
            "correct_player_f1": events["correct_player_f1"],
        })
    ranking = []
    for (scheme, contact_fraction, learning_rate), runs in groups.items():
        if sorted(run["seed"] for run in runs) != [0, 1]:
            raise ValueError("Each configuration needs exactly seeds 0 and 1")
        ranking.append({
            "scheme": scheme,
            "contact_fraction": contact_fraction, "learning_rate": learning_rate, "runs": runs,
            "mean_fully_correct_percent": sum(run["fully_correct_percent"] for run in runs) / 2,
            "mean_correct_player_f1": sum(run["correct_player_f1"] for run in runs) / 2,
        })
    if set(groups) != set(product(schemes, (0.25, 0.5), (0.005, 0.02))):
        raise ValueError("Selection requires every configuration in the grid")
    # Stable sorting preserves the original grid order for exact ties.
    return sorted(ranking, key=lambda row: (-row["mean_fully_correct_percent"], -row["mean_correct_player_f1"]))


def evaluate_selected(output: Path, cache: Path, winner: dict) -> None:
    """Test both seeds only after the validation ranking has been saved."""
    for run in winner["runs"]:
        destination = output / "test" / f"seed{run['seed']}"
        if run_state(destination).get("phase") == "complete":
            continue
        if destination.exists():
            shutil.rmtree(destination)
        command = [sys.executable, "-u", "-m", "experiments.annotator.video_models_benchmark.infer",
                   "--checkpoint", str(output / run["name"] / "best.pt"), "--cache", str(cache),
                   "--role", "test", "--stride", "3", "--offset", "0", "--output", str(destination)]
        subprocess.run(command, check=True)


def initialise_scheme_b(source: Path, destination: Path) -> None:
    """Freeze the top two settings; B starts afresh with the same per-seed config."""
    ranking = rank_configurations(source)
    manifest = read_record(source / "grid.json.gz")
    runs = []
    for seed in (0, 1):
        for configuration in ranking[:2]:
            original = next(run for run in configuration["runs"] if run["seed"] == seed)
            name = "B" + original["name"][1:]
            shutil.copy2(source / f"{original['name']}.toml", destination / f"{name}.toml")
            runs.append({"name": name, "seed": seed, "scheme": "B",
                         "contact_fraction": configuration["contact_fraction"],
                         "learning_rate": configuration["learning_rate"]})
    write_record(destination / "grid.json.gz", {
        **manifest, "stage": "scheme-b", "source_grid": str(source), "created_at": time.time(),
        "runs": runs, "max_steps": manifest["validation_steps"][-1], "selection": ranking,
    })


def run_scheme_b(output: Path) -> None:
    """Wait for A's continuation, then train and continue four selected B runs."""
    source, destination = output / "continuation", output / "scheme-b"
    destination.mkdir(exist_ok=True)
    with (destination / "wait.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"pid": os.getpid(), "phase": "waiting", "updated_at": time.time()}
        status_path = destination / "runner-status.json.gz"
        write_record(status_path, state)
        print(f"Waiting for all scheme-A continuations: {source}", flush=True)
        try:
            # The continuation runner holds wait.lock through its full queue,
            # including the period before its training grid.lock is acquired.
            with (source / "wait.lock").open("r+") as source_wait:
                fcntl.flock(source_wait, fcntl.LOCK_EX)
                with (source / "grid.lock").open("a") as source_grid:
                    fcntl.flock(source_grid, fcntl.LOCK_EX)
                    if (output / "PAUSE").exists() or (destination / "PAUSE").exists():
                        state.update(phase="paused")
                        return
                    manifest = read_record(source / "grid.json.gz")
                    if read_record(source / "grid-status.json.gz")["phase"] != "complete" or not all(
                        completed_run(source / run["name"]) for run in manifest["runs"]
                    ):
                        raise RuntimeError("Scheme-A continuation did not complete successfully; B will not start")
                    if not (destination / "grid.json.gz").exists():
                        initialise_scheme_b(source, destination)
            state.update(phase="running", updated_at=time.time())
            write_record(status_path, state)
            execute(destination, resume=True)
            if ((destination / "PAUSE").exists()
                    or read_record(destination / "grid-status.json.gz")["phase"] == "paused"):
                state.update(phase="paused")
                return
            extend_grid(destination)
            state.update(phase=read_record(destination / "continuation/grid-status.json.gz")["phase"])
        except BaseException:
            state.update(phase="stopped")
            raise
        finally:
            state.update(updated_at=time.time())
            write_record(status_path, state)


def pause(output: Path) -> None:
    (output / "PAUSE").touch()
    for child in ("continuation", "scheme-b"):
        if (output / child).is_dir():
            pause(output / child)
    if not (output / "grid.json.gz").exists():
        return
    # All unfinished runs receive the marker, including a run just being launched.
    for run in read_record(output / "grid.json.gz")["runs"]:
        directory = output / run["name"]
        if not completed_run(directory):
            directory.mkdir(exist_ok=True)
            (directory / "PAUSE").touch()
    print("Pause requested: the active run saves at its next validation boundary; no later run starts.")


def continue_queue(output: Path) -> None:
    for child in ("continuation", "scheme-b"):
        if (output / child).is_dir():
            continue_queue(output / child)
    (output / "PAUSE").unlink(missing_ok=True)
    if not (output / "grid.json.gz").exists():
        return
    for run in read_record(output / "grid.json.gz")["runs"]:
        (output / run["name"] / "PAUSE").unlink(missing_ok=True)
    print("Pause markers cleared. Restart the relevant runner (run --resume, extend, or scheme-b); see README.")


def live_phase(state: dict) -> str:
    phase = state["phase"]
    if phase in ("running", "starting", "waiting"):
        try:
            os.kill(state["pid"], 0)
        except ProcessLookupError:
            phase = "stopped (queue process absent)"
    return phase


def display(output: Path) -> bool:
    manifest = read_record(output / "grid.json.gz")
    state_path = output / "grid-status.json.gz"
    state = read_record(state_path) if state_path.exists() else {"phase": "not started"}
    phase = live_phase(state)
    scheme_b = output / "scheme-b"
    scheme_b_running = False
    if (scheme_b / "runner-status.json.gz").exists():
        pending = read_record(scheme_b / "runner-status.json.gz")
        pending_phase = live_phase(pending)
        scheme_b_running = pending_phase in ("waiting", "running") and not (output / "PAUSE").exists()
        print(f"Scheme B runner: {pending_phase} | {scheme_b}")
        if (scheme_b / "grid.json.gz").exists():
            return display(scheme_b) and not scheme_b_running
    continuation = output / "continuation"
    if (continuation / "grid.json.gz").exists() and phase == "complete":
        print(f"Original 30k grid complete. Continuation: {continuation}")
        return display(continuation) and not scheme_b_running
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
    if manifest.get("stage") == "continuation" or manifest.get("full_grid"):
        print(f"\n{finished}/{len(manifest['runs'])} continuations complete | stop after 3 unsuccessful validation checks")
    else:
        total = len(manifest["runs"]) * manifest["max_steps"]
        print(f"\n{finished}/{len(manifest['runs'])} runs complete | {updates:,}/{total:,} updates ({100 * updates / total:.1f}%)")
        print(f"Initial estimate: {len(manifest['runs']) * manifest['estimated_run_hours']:.1f}h excluding frame preparation/startup/scoring.")
    if (continuation / "grid-status.json.gz").exists():
        pending = read_record(continuation / "grid-status.json.gz")
        print(f"Continuation runner: {live_phase(pending)} | {continuation}")
    print("Metrics use the lowest-CE checkpoint at ±10 frames (scaled by fps). Partial runs can change.")
    if phase == "running" and "current" in state:
        print()
        show(output / state["current"])
    return phase not in ("running", "starting", "waiting") and not scheme_b_running


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "run", "extend", "scheme-b", "status", "pause", "continue"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--full-grid", action="store_true", help="Train all eight configurations through patience")
    parser.add_argument("--test-cache", type=Path, help="Test both seeds of the validation winner")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "init":
        if args.cache is None:
            parser.error("init requires --cache")
        if args.test_cache is not None and not args.full_grid:
            parser.error("--test-cache requires --full-grid")
        initialise(output, args.cache, full_grid=args.full_grid, test_cache=args.test_cache)
    elif args.command == "run":
        execute(output, resume=args.resume)
    elif args.command == "extend":
        extend_grid(output)
    elif args.command == "scheme-b":
        run_scheme_b(output)
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
