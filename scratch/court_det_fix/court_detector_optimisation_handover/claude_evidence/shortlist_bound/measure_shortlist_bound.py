"""Measure how much court scoring an exact shortlist bound could skip (follow-up item 12).

Throwaway measurement. Runs run_d17.main unchanged, with two hooks that only observe:

- around run_given.finite_scores: also compute an upper bound on every court's score from a
  subset of its samples, and time both
- around run_automatic.retain, for the per-pair call that follows: record the pair's cutoff
  (the score of its 256th retained court) and how many courts the bound could not exclude

Why the count is the saving. Each pair keeps its 256 best mutually distinct courts, greedily in
score order. Score courts in descending bound order; once every unscored court's bound is
below the 256th retained score, those courts sort after the point where the greedy stops, so
they cannot change the shortlist. The courts that must be scored are therefore those whose
bound reaches the cutoff (rounded up to the 256-court scoring batch).

Why the bound holds. A court's score only grows when any sample's response grows. Responses
are exp(-0.5 (d / sigma)^2) of a distance-map value d, and the distance map is an exact
Euclidean distance transform, so d changes by at most the pixel distance between two samples.
Samples sit at integer pixels (clip, then truncate), which adds at most sqrt(2). So an
unmeasured sample is at least (measured neighbour's d - gap - sqrt(2)) from any line.

Usage (from the checkout's scratch/court_det_fix/d17_timing), with SHORTLIST_BOUND_OUT set to a
JSON-lines output path: measure_shortlist_bound.py <run_d17.py arguments>
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
import run_d17  # pyrefly: ignore[missing-import]

STRIDES = (4, 8, 16)
SAMPLES_PER_INTERVAL = 64
SCORING_BATCH = 256
PIXEL_TRUNCATION_PX = float(np.sqrt(2))
# Float32 map rounding and float summation order; both far below these slacks.
DISTANCE_SLACK_PX = 1e-3
SCORE_MARGIN = 1e-6

pending: dict = {}


def visible_intervals(endpoints: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """detector._visible_samples's clipping, without building the samples: lower, upper, visible."""
    starts = endpoints[:, :, 0]
    vectors = endpoints[:, :, 1] - starts
    lower = np.zeros(starts.shape[:2])
    upper = np.ones(starts.shape[:2])
    visible = np.ones(starts.shape[:2], dtype=bool)
    for axis, limit in enumerate(size):
        stationary = np.abs(vectors[..., axis]) < 1e-8
        visible &= ~stationary | ((starts[..., axis] >= 0) & (starts[..., axis] <= limit - 1))
        divisor = np.where(stationary, 1, vectors[..., axis])
        first = -starts[..., axis] / divisor
        last = (limit - 1 - starts[..., axis]) / divisor
        lower = np.maximum(lower, np.where(stationary, -np.inf, np.minimum(first, last)))
        upper = np.minimum(upper, np.where(stationary, np.inf, np.maximum(first, last)))
    visible &= upper > lower
    clipped_length = (upper - lower) * np.linalg.norm(vectors, axis=-1)
    visible &= clipped_length >= 12
    return lower, upper, visible


def batch_upper_bounds(homographies: np.ndarray, maps: np.ndarray, size: tuple[int, int], measured: np.ndarray,
                       detector, assignment) -> np.ndarray:
    projected, _ = detector.project(homographies, detector.SEGMENTS_M)
    endpoints = projected.reshape(-1, 12, 2, 2)
    starts = endpoints[:, :, 0]
    vectors = endpoints[:, :, 1] - starts
    lower, upper, visible = visible_intervals(endpoints, size)
    # Same arithmetic as _visible_samples, so measured samples land on the same pixels.
    fractions = lower[..., None] + (upper - lower)[..., None] * np.linspace(0, 1, SAMPLES_PER_INTERVAL)[measured]
    samples = starts[..., None, :] + fractions[..., None] * vectors[..., None, :]
    pixel_x = np.clip(samples[..., 0], 0, size[0] - 1).astype(int)
    pixel_y = np.clip(samples[..., 1], 0, size[1] - 1).astype(int)
    family = np.repeat([0, 1], 6)[None, :, None]
    distance = maps[family, pixel_y, pixel_x].astype(np.float64)
    # Pixel distance between neighbouring samples along each visible interval.
    step = ((upper - lower) * np.linalg.norm(vectors, axis=-1) / (SAMPLES_PER_INTERVAL - 1))[..., None]
    sigma = assignment.DISTANCE_SIGMA_PX
    measured_bound = np.exp(-.5 * np.square(np.maximum(distance - DISTANCE_SLACK_PX, 0) / sigma)).sum(axis=2)
    # Every unmeasured sample between measured samples a and b is at least
    # max(d_a - gap_a, d_b - gap_b) >= (d_a + d_b - (b - a) * step) / 2 from any line.
    span = np.diff(measured)
    least_distance = ((distance[..., :-1] + distance[..., 1:] - span * step) / 2
                      - PIXEL_TRUNCATION_PX - DISTANCE_SLACK_PX)
    gap_bound = ((span - 1) * np.exp(-.5 * np.square(np.maximum(least_distance, 0) / sigma))).sum(axis=2)
    response = np.where(visible, (measured_bound + gap_bound) / SAMPLES_PER_INTERVAL, 0.)
    per_marking, marking_visible = [], []
    for intervals in assignment.MARKING_INTERVALS:
        count = visible[:, intervals].sum(axis=1)
        per_marking.append(response[:, intervals].sum(axis=1) / np.maximum(count, 1))
        marking_visible.append(count > 0)
    return np.sum(per_marking, axis=0) / np.maximum(np.sum(marking_visible, axis=0), 1) + SCORE_MARGIN


def upper_bounds(homographies: np.ndarray, maps: np.ndarray, size: tuple[int, int], stride: int,
                 detector, assignment) -> np.ndarray:
    measured = np.unique(np.r_[np.arange(0, SAMPLES_PER_INTERVAL, stride), SAMPLES_PER_INTERVAL - 1])
    bounds = np.empty(len(homographies))
    for start in range(0, len(homographies), SCORING_BATCH):
        bounds[start:start + SCORING_BATCH] = batch_upper_bounds(
            homographies[start:start + SCORING_BATCH], maps, size, measured, detector, assignment)
    return bounds


def must_score(bounds: np.ndarray, cutoff: float | None) -> int:
    """Courts scored before every remaining bound falls below the cutoff, in whole batches."""
    if cutoff is None:
        return len(bounds)
    reaching = int((bounds >= cutoff).sum())
    return min(len(bounds), -(-reaching // SCORING_BATCH) * SCORING_BATCH)


def install_hooks(run_given, run_automatic, output: Path) -> None:
    from experiments.annotator.independent_court import assignment, detector

    finite_scores = run_given.finite_scores
    retain = run_automatic.retain

    def finite_scores_with_bounds(homographies, observations, axes, size):
        started = time.perf_counter()
        scores = finite_scores(homographies, observations, axes, size)
        finite_seconds = time.perf_counter() - started
        # The same maps finite_scores builds; rebuilt here so the original call is untouched.
        families = []
        for matched in axes:
            members = np.concatenate([observations.groups[index] for index in matched.diagnostics["retained_group_ids"]])
            families.append(observations.segments[members].reshape(-1, 4))
        maps = detector._distance_maps((families[0], families[1]), size)
        bounds, bound_seconds = {}, {}
        for stride in STRIDES:
            started = time.perf_counter()
            bounds[stride] = upper_bounds(homographies, maps, size, stride, detector, assignment)
            bound_seconds[stride] = time.perf_counter() - started
        pending.update(scores=np.asarray(scores, dtype=float), bounds=bounds,
                       finite_seconds=finite_seconds, bound_seconds=bound_seconds)
        return scores

    def retain_and_record(candidates, settings):
        retained = retain(candidates, settings)
        if not pending:
            return retained  # the global pool, or a pair with no usable courts
        scores = pending["scores"]
        assert len(candidates) == len(scores), (len(candidates), len(scores))
        assert all(candidate.score == score for candidate, score in zip(candidates, scores, strict=True))
        full = len(retained) == settings.keep_candidates
        cutoff = retained[-1].score if full else None
        row = {"courts": len(scores), "shortlist_full": full, "cutoff": cutoff,
               "must_score_exact": must_score(scores, cutoff),
               "finite_seconds": pending["finite_seconds"]}
        for stride, bounds in pending["bounds"].items():
            row[f"must_score_stride{stride}"] = must_score(bounds, cutoff)
            row[f"violations_stride{stride}"] = int((bounds < scores).sum())
            row[f"bound_seconds_stride{stride}"] = pending["bound_seconds"][stride]
        with open(output, "a") as stream:
            stream.write(json.dumps(row) + "\n")
        pending.clear()
        return retained

    run_given.finite_scores = finite_scores_with_bounds
    run_automatic.retain = retain_and_record


def main() -> None:
    output = Path(os.environ["SHORTLIST_BOUND_OUT"])
    wrap = run_d17.StageClock.wrap

    def wrap_after_hooks(clock, owner, name, label=None):
        # run_d17 imports the modules inside main; hook them just before it adds its timers.
        if getattr(owner, "__name__", "") == "run_given" and name == "finite_scores":
            install_hooks(owner, sys.modules["run_automatic"], output)
        wrap(clock, owner, name, label)

    run_d17.StageClock.wrap = wrap_after_hooks
    run_d17.main()


if __name__ == "__main__":
    main()
