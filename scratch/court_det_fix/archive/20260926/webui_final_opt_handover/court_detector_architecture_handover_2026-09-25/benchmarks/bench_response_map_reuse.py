#!/usr/bin/env python3
"""Synthetic microbenchmark for exact Gaussian-response feature-map reuse.

This mirrors the hot portion of scan_population.continuous_support after
projected sample pixels are known. Baseline gathers distances and applies the
Gaussian per court sample. The reuse variant applies the same elementwise
transform once to the two dense distance maps, then gathers responses.

The benchmark deliberately includes response-map construction in the reuse
measurement and verifies bit-identical output on the local NumPy build.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
from pathlib import Path
import statistics
import time

# Set these before importing numpy when launched directly.
for name in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ.setdefault(name, "1")

import numpy as np


def median_time(fn, repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def make_line_samples(
    rng: np.random.Generator,
    courts: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = rng.uniform([0, 0], [width - 1, height - 1], size=(courts, 12, 2)).astype(np.float32)
    ends = rng.uniform([0, 0], [width - 1, height - 1], size=(courts, 12, 2)).astype(np.float32)
    fractions = np.linspace(0, 1, 64, dtype=np.float32)[None, None, :, None]
    points = starts[:, :, None, :] + fractions * (ends - starts)[:, :, None, :]
    pixel_x = np.clip(points[..., 0], 0, width - 1).astype(np.uint16)
    pixel_y = np.clip(points[..., 1], 0, height - 1).astype(np.uint16)
    visible = rng.random((courts, 12)) > 0.1
    return pixel_x, pixel_y, visible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    height, width = 540, 960
    sigma = 2.0
    maps = np.clip(rng.gamma(shape=2.0, scale=3.0, size=(2, height, width)), 0, 30).astype(np.float32)
    family = np.repeat(np.array([0, 1], dtype=np.intp), 6)[None, :, None]

    def aggregate(response: np.ndarray, visible: np.ndarray) -> np.ndarray:
        response = response.mean(axis=2)
        response *= visible
        return response.sum(axis=1) / np.maximum(visible.sum(axis=1), 1)

    def baseline(pixel_x: np.ndarray, pixel_y: np.ndarray, visible: np.ndarray) -> np.ndarray:
        distance = maps[family, pixel_y, pixel_x]
        return aggregate(np.exp(-0.5 * np.square(distance / sigma)), visible)

    def reuse(
        pixel_x: np.ndarray,
        pixel_y: np.ndarray,
        visible: np.ndarray,
        *,
        include_precompute: bool,
    ) -> np.ndarray:
        response_maps = np.exp(-0.5 * np.square(maps / sigma))
        if not include_precompute:
            # The caller passes through this branch only after a warm response map exists;
            # construction is still written here for clarity but timed outside below.
            pass
        return aggregate(response_maps[family, pixel_y, pixel_x], visible)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    batch = 256
    for courts in (256, 1024, 4096, 16384):
        pixel_x, pixel_y, visible = make_line_samples(rng, courts, width, height)
        response_maps = np.exp(-0.5 * np.square(maps / sigma))

        def run_baseline() -> np.ndarray:
            output = np.empty(courts, dtype=np.float64)
            for start in range(0, courts, batch):
                stop = min(start + batch, courts)
                output[start:stop] = baseline(pixel_x[start:stop], pixel_y[start:stop], visible[start:stop])
            return output

        def run_reuse_including_precompute() -> np.ndarray:
            local_response_maps = np.exp(-0.5 * np.square(maps / sigma))
            output = np.empty(courts, dtype=np.float64)
            for start in range(0, courts, batch):
                stop = min(start + batch, courts)
                response = local_response_maps[family, pixel_y[start:stop], pixel_x[start:stop]]
                output[start:stop] = aggregate(response, visible[start:stop])
            return output

        def run_reuse_only() -> np.ndarray:
            output = np.empty(courts, dtype=np.float64)
            for start in range(0, courts, batch):
                stop = min(start + batch, courts)
                response = response_maps[family, pixel_y[start:stop], pixel_x[start:stop]]
                output[start:stop] = aggregate(response, visible[start:stop])
            return output

        reference = run_baseline()
        reused = run_reuse_including_precompute()
        bit_identical = reference.tobytes() == reused.tobytes()
        baseline_s = median_time(run_baseline, args.repeats)
        reuse_total_s = median_time(run_reuse_including_precompute, args.repeats)
        reuse_only_s = median_time(run_reuse_only, args.repeats)
        rows.append({
            "courts": courts,
            "baseline_s": baseline_s,
            "reuse_including_precompute_s": reuse_total_s,
            "reuse_only_s": reuse_only_s,
            "speedup_including_precompute": baseline_s / reuse_total_s,
            "speedup_reuse_only": baseline_s / reuse_only_s,
            "bit_identical": bit_identical,
            "numpy_version": np.__version__,
            "width": width,
            "height": height,
            "markings": 12,
            "samples_per_marking": 64,
            "batch": batch,
        })

    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
