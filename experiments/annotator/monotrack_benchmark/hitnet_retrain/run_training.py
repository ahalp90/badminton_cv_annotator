"""Build fresh features, run both six-setting grids, and export the validation winners."""
from __future__ import annotations

import argparse
import itertools
import math
import os
import shutil
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from inputs import read_json, write_json

HERE = Path(__file__).resolve().parent
GRID = tuple(itertools.product(("kept", "zeroed"), (0.0, 0.2), (0.0, 1e-6, 1e-4)))


def trial_name(arm: str, dropout: float, l2_reg: float) -> str:
    return f"{arm}_d{dropout:g}_l2{l2_reg:g}"


def run_trial(output: Path, setting: tuple[str, float, float]) -> str:
    arm, dropout, l2_reg = setting
    name = trial_name(*setting)
    trial = output / "trials" / name
    command = [sys.executable, "-u", str(HERE / "train_hitnet.py"), "--features", str(output / "features"),
               "--arm", arm, "--dropout", str(dropout), "--l2", str(l2_reg), "--output", str(trial)]
    with (output / "logs" / f"{name}.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return name


def export_winners(output: Path, destination: Path) -> dict[str, Any]:
    """Select only after all twelve complete trials are available; never select smoke runs."""
    winners: dict[str, Any] = {}
    summaries = []
    for setting in GRID:
        name = trial_name(*setting)
        summary = read_json(output / "trials" / name / "summary.json.gz")
        actual = (summary["arm"], summary["dropout"], summary["l2"])
        if actual != setting or summary["smoke"] or not math.isfinite(summary["best_val_loss"]):
            raise ValueError(f"{name}: incomplete or incompatible trial result")
        if not (output / "trials" / name / "hitnet.h5").is_file():
            raise FileNotFoundError(f"{name}: full-model checkpoint is missing")
        summary = {**summary, "trial": name}
        summaries.append(summary)
        arm = setting[0]
        if arm not in winners or summary["best_val_loss"] < winners[arm]["best_val_loss"]:
            winners[arm] = summary
    destination.mkdir(parents=True, exist_ok=True)
    for arm, summary in winners.items():
        trial = output / "trials" / summary["trial"]
        shutil.copyfile(trial / "hitnet.h5", destination / f"hitnet_{arm}.h5")
        shutil.copyfile(trial / "history.csv.gz", destination / f"{arm}_history.csv.gz")
    write_json(destination / "winners.json.gz", winners)
    write_json(destination / "grid_summary.json.gz", summaries)
    shutil.copyfile(output / "features/summary.json.gz", destination / "training_data.json.gz")
    return winners


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    parser.add_argument("--output", type=Path, required=True, help="fresh directory for bulky features and trial files")
    parser.add_argument("--export", type=Path, default=HERE / "weights", help="small winner artefacts to commit")
    parser.add_argument("--select-only", action="store_true", help="export from an already completed twelve-trial grid")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.select_only:
        export_winners(output, args.export)
        return
    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    workers = config["execution"]["feature_workers"]
    parallel = config["execution"]["parallel_trials"]
    if workers < 1 or parallel < 1:
        raise ValueError("Worker and parallel-trial counts must be positive")
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    shutil.copyfile(args.config, output / "config.toml")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    command = [sys.executable, "-u", str(HERE / "build_training_set.py"), "--config", str(args.config.resolve()),
               "--output", str(output / "features"), "--workers", str(workers)]
    with (output / "logs/features.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("Features built; training twelve trials. Logs:", output / "logs", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(run_trial, output, setting): trial_name(*setting) for setting in GRID}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except subprocess.CalledProcessError:
                failures.append(name)
                print(f"FAILED {name}; see its log", flush=True)
            else:
                print(f"Completed {name}", flush=True)
    if failures:
        raise RuntimeError(f"No winners exported: failed trials {failures}; see {output / 'logs'}")
    winners = export_winners(output, args.export)
    for arm, result in winners.items():
        print(f"{arm}: {result['trial']}, validation loss {result['best_val_loss']:.6f}", flush=True)
    print("Ready to commit:", args.export, flush=True)


if __name__ == "__main__":
    main()
