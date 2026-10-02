"""Measure how deep a cheap coarse score must reach to keep every pair's shortlist (follow-up item 12).

Throwaway measurement. Runs run_d17.main unchanged, with two hooks that only observe:

- around run_given.finite_scores: also score the same courts with fewer samples per marking
  interval (COARSE_COUNTS instead of 64), and time each coarse pass
- around run_automatic.retain, for the per-pair call that follows: record where each retained
  court falls in each coarse ranking

Why that rank is the number to watch. A prefilter would fully score only the K courts with the
best coarse scores, then retain from those. Retention walks courts in exact-score order and keeps
each one that is not within 2 px of a court already kept. If every court the full search keeps is
among the K, the walk over the K makes the same keep/skip decision for every court it meets, so
the pair's shortlist is unchanged. The smallest safe K for a pair is one plus the worst coarse
rank among its retained courts.

The hook also tests that argument directly. For each coarse count it retains again from only the
top needed_k courts, kept in proposal order, and records whether the shortlist is the same one.
It times full scoring of those courts at the real batch size, so a two-pass prefilter's cost per
pair is map_seconds + coarseN_seconds + coarseN_topk_full_seconds. The hook's extra work runs
inside the timed finite_scores and retain stages, so this run's D17 stage times are not clean.

Rows are JSON lines in call order: G0 pairs, then G1 pairs, each with at least one usable court.
Usage (from the checkout's scratch/court_det_fix/d17_timing), with PREFILTER_OUT set to a
JSON-lines output path: measure_prefilter.py <run_d17.py arguments>
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
import run_d17  # pyrefly: ignore[missing-import]

COARSE_COUNTS = (4, 8, 16)
# The same batch as full scoring, so coarse and full times are comparable.
SCORING_BATCH = 256
# Coarse scores at these ranks, so a score-margin rule can be costed afterwards as well as a rank rule.
LADDER_RANKS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
# Score gaps below the weakest retained court's coarse score; each row counts the courts inside each gap.
SCORE_GAPS = (0.01, 0.02, 0.05, 0.1)

pending: dict = {}
sequence = [0]


def coarse_support(homographies: np.ndarray, maps: np.ndarray, size: tuple[int, int], count: int,
                   detector, assignment) -> np.ndarray:
    """scan_population.continuous_support with count samples per interval instead of 64."""
    projected, _ = detector.project(homographies, detector.SEGMENTS_M)
    endpoints = projected.reshape(-1, 12, 2, 2)
    lower, upper, visible = detector._visible_fractions(endpoints, size)
    fractions = lower[..., None] + (upper - lower)[..., None] * np.linspace(0, 1, count)
    start_x, start_y = endpoints[:, :, 0, 0, None], endpoints[:, :, 0, 1, None]
    end_x, end_y = endpoints[:, :, 1, 0, None], endpoints[:, :, 1, 1, None]
    pixel_x = np.clip(start_x + fractions * (end_x - start_x), 0, size[0] - 1).astype(int)
    pixel_y = np.clip(start_y + fractions * (end_y - start_y), 0, size[1] - 1).astype(int)
    distance = maps[np.repeat([0, 1], 6)[None, :, None], pixel_y, pixel_x]
    response = np.exp(-.5 * np.square(distance / assignment.DISTANCE_SIGMA_PX)).mean(axis=2)
    response *= visible
    per_marking, marking_visible = [], []
    for intervals in assignment.MARKING_INTERVALS:
        visible_count = visible[:, intervals].sum(axis=1)
        per_marking.append(response[:, intervals].sum(axis=1) / np.maximum(visible_count, 1))
        marking_visible.append(visible_count > 0)
    return np.sum(per_marking, axis=0) / np.maximum(np.sum(marking_visible, axis=0), 1)


def install_hooks(run_given, run_automatic, output: Path) -> None:
    from experiments.annotator.independent_court import assignment, detector

    finite_scores = run_given.finite_scores
    retain = run_automatic.retain

    def finite_scores_with_coarse(homographies, observations, axes, size):
        # Every finite_scores call must be consumed by its own per-pair retain before the next one.
        assert not pending, "finite_scores called twice without a per-pair retain between"
        started = time.perf_counter()
        scores = finite_scores(homographies, observations, axes, size)
        finite_seconds = time.perf_counter() - started
        # The same maps finite_scores builds; rebuilt here so the original call is untouched.
        started = time.perf_counter()
        families = []
        for matched in axes:
            members = np.concatenate([observations.groups[index] for index in matched.diagnostics["retained_group_ids"]])
            families.append(observations.segments[members].reshape(-1, 4))
        maps = detector._distance_maps((families[0], families[1]), size)
        map_seconds = time.perf_counter() - started
        coarse, coarse_seconds = {}, {}
        for count in COARSE_COUNTS:
            started = time.perf_counter()
            values = np.empty(len(homographies))
            for start in range(0, len(homographies), SCORING_BATCH):
                stop = start + SCORING_BATCH
                values[start:stop] = coarse_support(homographies[start:stop], maps, size, count, detector, assignment)
            coarse[count] = values
            coarse_seconds[count] = time.perf_counter() - started
        pending.update(scores=np.asarray(scores, dtype=float), coarse=coarse, finite_seconds=finite_seconds,
                       map_seconds=map_seconds, coarse_seconds=coarse_seconds,
                       scoring_inputs=(homographies, observations, axes, size))
        return scores

    def retain_and_record(candidates, settings):
        retained = retain(candidates, settings)
        if not pending:
            return retained  # the global pool
        scores = pending["scores"]
        assert len(candidates) == len(scores), (len(candidates), len(scores))
        assert all(candidate.score == score for candidate, score in zip(candidates, scores, strict=True))
        positions = {id(candidate): position for position, candidate in enumerate(candidates)}
        retained_positions = np.asarray([positions[id(candidate)] for candidate in retained], dtype=int)
        row = {"sequence": sequence[0], "courts": len(scores), "retained": len(retained),
               "shortlist_full": len(retained) == settings.keep_candidates,
               "cutoff": retained[-1].score if retained else None,
               "finite_seconds": pending["finite_seconds"], "map_seconds": pending["map_seconds"],
               "retained_positions": retained_positions.tolist()}
        for count, values in pending["coarse"].items():
            # Stable, so tied coarse scores keep proposal order, as a prefilter's argsort would.
            order = np.argsort(-values, kind="stable")
            rank = np.empty(len(values), dtype=int)
            rank[order] = np.arange(len(values))
            retained_ranks = rank[retained_positions]
            ladder = values[order]
            needed_k = int(retained_ranks.max()) + 1 if len(retained_ranks) else 0
            top_positions = np.sort(order[:needed_k])
            again = retain([candidates[position] for position in top_positions], settings)
            homographies, observations, axes, size = pending["scoring_inputs"]
            started = time.perf_counter()
            top_scores = finite_scores(homographies[top_positions], observations, axes, size)
            topk_full_seconds = time.perf_counter() - started
            # Scores do not depend on batch boundaries, so the subset must rescore to the same bits.
            assert top_scores.tobytes() == scores[top_positions].tobytes()
            weakest_retained = values[retained_positions].min() if len(retained_positions) else None
            same_shortlist = [id(candidate) for candidate in again] == [id(candidate) for candidate in retained]
            row[f"coarse{count}_seconds"] = pending["coarse_seconds"][count]
            row[f"coarse{count}_needed_k"] = needed_k
            row[f"coarse{count}_subset_same_shortlist"] = same_shortlist
            row[f"coarse{count}_topk_full_seconds"] = topk_full_seconds
            row[f"coarse{count}_next_excluded_score"] = float(ladder[needed_k]) if needed_k < len(ladder) else None
            row[f"coarse{count}_courts_within_gap"] = (
                {str(gap): int(np.sum(values >= weakest_retained - gap)) for gap in SCORE_GAPS}
                if weakest_retained is not None else None)
            row[f"coarse{count}_retained_ranks"] = retained_ranks.tolist()
            row[f"coarse{count}_retained_scores"] = values[retained_positions].tolist()
            row[f"coarse{count}_ladder"] = {str(k): float(ladder[k - 1]) for k in LADDER_RANKS if k <= len(ladder)}
        with open(output, "a") as stream:
            stream.write(json.dumps(row) + "\n")
        sequence[0] += 1
        pending.clear()
        return retained

    run_given.finite_scores = finite_scores_with_coarse
    run_automatic.retain = retain_and_record


def main() -> None:
    output = Path(os.environ["PREFILTER_OUT"])
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
