"""Add X3D seeds 2–4 of the frozen validation winner after another queue finishes."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from .grid import completed_run, continue_queue, read_record, run_state, write_record
from .videomae_queue import launch_run, wait_for_child

EXTRA_SEEDS = (2, 3, 4)


def wait_for_queue(source: Path) -> None:
    """Wait through the launcher's exit marker, including its selected test runs."""
    with (source / "runner.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read_record(source / "grid-status.json.gz")
        if (source / "queue-exit-code.txt").read_text().strip() != "0" or state["phase"] != "complete":
            raise RuntimeError(f"Dependency did not complete successfully: {source}")
        winner = read_record(source / "selection.json.gz")["ranking"][0]
        for run in winner["runs"]:
            if run_state(source / "test" / f"seed{run['seed']}").get("phase") != "complete":
                raise RuntimeError(f"Dependency test seed {run['seed']} is incomplete: {source}")


def initialise(output: Path, source: Path, after: Path) -> None:
    """Copy the selected training recipe, changing only the three new seed values."""
    from .train import normalise_settings_record

    source_manifest = read_record(source / "grid.json.gz")
    winner = read_record(source / "selection.json.gz")["ranking"][0]
    if not source_manifest.get("full_grid") or not source_manifest.get("test_cache"):
        raise ValueError("Additional seeds require the full X3D grid and its test cache")
    original = winner["runs"][0]
    settings = normalise_settings_record(tomllib.loads((source / f"{original['name']}.toml").read_text()))
    if not settings.get("continue_until_patience") or not settings.get("break_alternation_ties"):
        raise ValueError("The source must use integrated continuation and alternation tie-breaking")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "grid.json.gz").exists():
        raise FileExistsError(f"Queue already exists: {output}")
    runs = []
    test_runs = []
    for run in winner["runs"]:
        test_runs.append({"seed": run["seed"], "training_output": str(source / run["name"]),
                          "test_output": str(source / "test" / f"seed{run['seed']}")})
    for seed in EXTRA_SEEDS:
        name = (f"{winner['scheme']}-contact{int(100 * winner['contact_fraction'])}"
                f"-lr{winner['learning_rate']:g}-seed{seed}")
        run_settings = {**settings, "seed": seed}
        (output / f"{name}.toml").write_text(
            "".join(f"{key} = {json.dumps(value)}\n" for key, value in run_settings.items()))
        runs.append({"name": name, "seed": seed, "scheme": winner["scheme"],
                     "contact_fraction": winner["contact_fraction"], "learning_rate": winner["learning_rate"]})
        test_runs.append({"seed": seed, "training_output": str(output / name),
                          "test_output": str(output / "test" / f"seed{seed}")})
    write_record(output / "grid.json.gz", {
        "source_grid": str(source), "after": str(after), "created_at": time.time(),
        "x3d_winner": winner, "scheme": winner["scheme"], "runs": runs,
        "cache": source_manifest["cache"], "test_cache": source_manifest["test_cache"],
        "test_runs": test_runs,
    })


def test_run(output: Path, manifest: dict, run: dict) -> None:
    destination = output / "test" / f"seed{run['seed']}"
    if run_state(destination).get("phase") == "complete":
        return
    if destination.exists():
        raise FileExistsError(f"Incomplete test output {destination}; preserve it elsewhere before resuming")
    destination.parent.mkdir(exist_ok=True)
    command = [sys.executable, "-u", "-m", "experiments.annotator.video_models_benchmark.infer",
               "--checkpoint", str(output / run["name"] / "best.pt"), "--cache", manifest["test_cache"],
               "--role", "test", "--stride", "3", "--offset", "0", "--output", str(destination)]
    with (destination.parent / f"seed{run['seed']}.log").open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return_code = wait_for_child(process)
    if return_code or run_state(destination).get("phase") != "complete":
        raise RuntimeError(f"Test seed {run['seed']} failed; see {log.name}")


def execute(output: Path) -> None:
    manifest = read_record(output / "grid.json.gz")
    with (output / "grid.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"pid": os.getpid(), "phase": "running", "updated_at": time.time()}
        try:
            for run in manifest["runs"]:
                directory = output / run["name"]
                if (output / "PAUSE").exists() or (directory / "PAUSE").exists():
                    state["phase"] = "paused"
                    return
                if not completed_run(directory):
                    state.update(phase="running", current=run["name"], updated_at=time.time())
                    write_record(output / "grid-status.json.gz", state)
                    print(f"Training {run['name']}", flush=True)
                    launch_run(output, manifest, run, extend=False, resume=True)
                    if run_state(directory).get("phase") == "paused":
                        state["phase"] = "paused"
                        return
                if (output / "PAUSE").exists():
                    state["phase"] = "paused"
                    return
                state.update(phase="testing", current=run["name"], updated_at=time.time())
                write_record(output / "grid-status.json.gz", state)
                print(f"Testing {run['name']}", flush=True)
                test_run(output, manifest, run)
            state["phase"] = "complete"
        except BaseException:
            state["phase"] = "stopped"
            raise
        finally:
            state["updated_at"] = time.time()
            write_record(output / "grid-status.json.gz", state)


def follow(output: Path, source: Path, after: Path) -> None:
    output, source, after = output.resolve(), source.resolve(), after.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "follow.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"pid": os.getpid(), "phase": "waiting", "source_grid": str(source),
                 "after": str(after), "updated_at": time.time()}
        write_record(output / "grid-status.json.gz", state)
        print(f"Waiting for {after}, including both winner tests.", flush=True)
        try:
            wait_for_queue(after)
            wait_for_queue(source)
            if not (output / "grid.json.gz").exists():
                initialise(output, source, after)
            else:
                manifest = read_record(output / "grid.json.gz")
                if manifest["source_grid"] != str(source) or manifest["after"] != str(after):
                    raise ValueError("Resume must use the frozen source and dependency queues")
            execute(output)
        except BaseException:
            state.update(phase="stopped", updated_at=time.time())
            write_record(output / "grid-status.json.gz", state)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("follow", "status", "pause", "continue"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--x3d-output", type=Path)
    parser.add_argument("--after", type=Path, help="Wait for this launcher's queue and selected tests")
    args = parser.parse_args()
    if args.command == "follow":
        if args.x3d_output is None or args.after is None:
            parser.error("follow requires --x3d-output and --after")
        follow(args.output, args.x3d_output, args.after)
    elif args.command == "continue":
        continue_queue(args.output)
    elif args.command == "pause":
        (args.output / "PAUSE").touch()
        manifest = args.output / "grid.json.gz"
        if manifest.exists():
            for run in read_record(manifest)["runs"]:
                directory = args.output / run["name"]
                directory.mkdir(exist_ok=True)
                (directory / "PAUSE").touch()
    else:
        state = read_record(args.output / "grid-status.json.gz")
        print(f"X3D extra seeds: {state['phase']} | {state.get('current', state.get('after', ''))}")
        manifest = args.output / "grid.json.gz"
        if manifest.exists():
            for run in read_record(manifest)["runs"]:
                training = run_state(args.output / run["name"])
                test = run_state(args.output / "test" / f"seed{run['seed']}")
                print(f"seed {run['seed']}: {training.get('phase', 'queued')} "
                      f"step {training.get('step', 0)}; test {test.get('phase', 'pending')}")


if __name__ == "__main__":
    main()
