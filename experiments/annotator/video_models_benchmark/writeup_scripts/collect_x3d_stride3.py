"""Collect compact X3D scores from native validation and saved stride-three test results.

Validation inputs are the seed0/seed1 outputs of ``infer --stride 3 --offset 0``.
Test inputs are the seed0/seed1 directories from the saved-probability rescoring,
with files named ``stride3_phase0_alternating.json.gz``. Arrays stay outside Git.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from experiments.annotator.video_models_benchmark.data import read_json, write_json

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    for seed in (0, 1):
        run = f"A-contact50-lr0.02-seed{seed}"
        history = read_json(args.results / "x3d_pilot" / run / "history.json.gz")
        best_step = min(history, key=lambda row: row["validation"]["loss"])["steps"]
        sources = {
            "validation": args.validation / f"seed{seed}" / "events.json.gz",
            "test": args.test / f"seed{seed}" / "stride3_phase0_alternating.json.gz",
        }
        for split, source in sources.items():
            saved = read_json(source)
            sampling = (saved["sampling"] if split == "validation"
                        else {"stride": saved["stride"], "offset": saved["phase"]})
            if (sampling["stride"], sampling["offset"]) != (3, 0):
                raise ValueError(f"{source}: expected stride 3, offset 0")
            destination = args.results / "x3d_stride3" / split / f"seed{seed}"
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination / "scores.json.gz", {
                "split": split, "seed": seed, "checkpoint_step": best_step,
                "sampling": sampling, "hitter_rule": "alternating", "metrics": saved["metrics"],
            })
            if split == "validation":
                shutil.copy2(source.with_name("frame_metrics.json.gz"), destination / "frame_metrics.json.gz")


if __name__ == "__main__":
    main()
