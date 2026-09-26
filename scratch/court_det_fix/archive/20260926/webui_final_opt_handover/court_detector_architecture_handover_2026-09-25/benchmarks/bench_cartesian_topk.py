#!/usr/bin/env python3
"""Synthetic benchmark for best-first Cartesian axis-pair proposals.

It compares a full 512x512 separable score matrix with a Python heap that
emits only the top K pairs. This measures proposal enumeration only; it does
not include geometry, player tests, or finite court support.
"""
from __future__ import annotations

import argparse
import csv
import gc
import heapq
import os
from pathlib import Path
import statistics
import time

for name in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ.setdefault(name, "1")

import numpy as np


def median_time(fn, repeats: int) -> float:
    values: list[float] = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        fn()
        values.append(time.perf_counter() - started)
    return statistics.median(values)


def full_topk(first: np.ndarray, second: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    scores = (first[:, None] + second[None, :]).ravel()
    indexes = np.argpartition(scores, -k)[-k:]
    order = np.lexsort((indexes, -scores[indexes]))
    indexes = indexes[order]
    return indexes, scores[indexes]


def heap_topk(first: np.ndarray, second: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    heap: list[tuple[float, int, int]] = [(-(first[0] + second[0]), 0, 0)]
    seen = {(0, 0)}
    indexes: list[int] = []
    scores: list[float] = []
    while heap and len(indexes) < k:
        negative, i, j = heapq.heappop(heap)
        indexes.append(i * len(second) + j)
        scores.append(-negative)
        if i + 1 < len(first) and (i + 1, j) not in seen:
            seen.add((i + 1, j))
            heapq.heappush(heap, (-(first[i + 1] + second[j]), i + 1, j))
        if j + 1 < len(second) and (i, j + 1) not in seen:
            seen.add((i, j + 1))
            heapq.heappush(heap, (-(first[i] + second[j + 1]), i, j + 1))
    return np.asarray(indexes), np.asarray(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    axis_count = 512
    first = np.sort(rng.normal(size=axis_count))[::-1]
    second = np.sort(rng.normal(size=axis_count))[::-1]
    rows: list[dict[str, object]] = []
    for k in (256, 1024, 4096, 8192):
        full_indexes, full_scores = full_topk(first, second, k)
        heap_indexes, heap_scores = heap_topk(first, second, k)
        same_scores = np.array_equal(np.sort(full_scores), np.sort(heap_scores))
        full_s = median_time(lambda: full_topk(first, second, k), args.repeats)
        heap_s = median_time(lambda: heap_topk(first, second, k), args.repeats)
        rows.append({
            "axis_count_each": axis_count,
            "cartesian_pairs": axis_count * axis_count,
            "k": k,
            "full_numpy_s": full_s,
            "heap_best_first_s": heap_s,
            "heap_vs_full_speedup": full_s / heap_s,
            "same_topk_scores": same_scores,
            "full_score_matrix_mib": axis_count * axis_count * 8 / (1024 ** 2),
            "numpy_version": np.__version__,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
