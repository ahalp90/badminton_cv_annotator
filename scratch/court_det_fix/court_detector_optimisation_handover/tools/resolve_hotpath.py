#!/usr/bin/env python3
"""Resolve and print the modules imported by the W5 court-detector hot path."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path


MODULES = (
    "run_w5",
    "automatic_generation",
    "run_automatic",
    "run_given",
    "projective_seed",
    "scan_population",
    "run_population",
    "run_diagnosis",
    "zone_net",
    "line_template_source",
    "experiments.annotator.independent_court.assignment",
    "experiments.annotator.independent_court.detector",
    "experiments.annotator.independent_court.stripe_observations",
    "experiments.annotator.independent_court.fixed_stripe_refit",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--court-root",
        type=Path,
        default=Path("scratch/court_det_fix"),
        help="Path to scratch/court_det_fix from the repository root",
    )
    args = parser.parse_args()
    court_root = args.court_root.resolve()
    repo = court_root.parents[1]
    sys.path[:0] = [str(repo), str(repo / "src"), str(court_root / "w5_holistic")]

    run_w5 = importlib.import_module("run_w5")
    run_w5.add_helper_paths(court_root)

    rows = []
    for name in MODULES:
        module = importlib.import_module(name)
        raw_path = getattr(module, "__file__", None)
        path = Path(raw_path).resolve() if raw_path else None
        rows.append(
            {
                "module": name,
                "path": str(path) if path else None,
                "relative_to_repo": (
                    str(path.relative_to(repo)) if path and path.is_relative_to(repo) else None
                ),
                "sha256": digest(path) if path and path.is_file() else None,
            }
        )
    print(json.dumps({"court_root": str(court_root), "modules": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
