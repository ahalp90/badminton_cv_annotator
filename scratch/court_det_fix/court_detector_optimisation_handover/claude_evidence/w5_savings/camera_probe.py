"""Save real template camera-check inputs from one D17 view and measure how much of them selection reads (throwaway).

Runs run_d17.main through the line-template stage, then exits without finishing the view.
- Saves the first BATCHES batches' homographies, corners and final errors, for offline benchmarks.
- Times the vectorised camera errors and the whole check (vector plus scalar frontier rechecks).
- At selection, finds how far down the full score order the camera-eligible scan reached: the
  prefix a check-as-you-go camera test would have had to cover, for both greedy passes.

Usage (from the checkout's scratch/court_det_fix/d17_timing), with CAMERA_PROBE_OUT set to an .npz
path: camera_probe.py <run_d17.py arguments>
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
import run_d17  # pyrefly: ignore[missing-import]

BATCHES = 80
saved: dict[str, np.ndarray] = {}
totals = {"vector_s": 0.0, "whole_s": 0.0, "rechecks": 0, "batches": 0, "homographies": 0}


def prefix_needed(scores, rectangle_ids, templates, pool, eligible, scanned) -> int:
    """Length of the score-ordered pool prefix that holds the first `scanned` eligible hypotheses."""
    order = pool[np.lexsort((templates[pool], rectangle_ids[pool], -scores[pool]))]
    eligible_positions = np.flatnonzero(eligible[order])
    return int(eligible_positions[scanned - 1]) + 1 if scanned else 0


def install(line_template_source) -> None:
    original_vector = line_template_source.vector_camera_errors
    original_whole = line_template_source.camera_errors_with_frontier_recheck
    original_select = line_template_source.select_with_visibility_floor

    def timed_vector(homographies, size):
        started = time.perf_counter()
        result = original_vector(homographies, size)
        totals["vector_s"] += time.perf_counter() - started
        return result

    def probed_whole(homographies, corners, working_size, native_scale, native_size, zone):
        started = time.perf_counter()
        result = original_whole(homographies, corners, working_size, native_scale, native_size, zone)
        totals["whole_s"] += time.perf_counter() - started
        totals["rechecks"] += result[1]
        totals["homographies"] += len(homographies)
        if totals["batches"] < BATCHES:
            batch = totals["batches"]
            saved[f"homographies_{batch}"] = homographies.copy()
            saved[f"corners_{batch}"] = corners.copy()
            saved[f"errors_{batch}"] = result[0].copy()
            saved["working_size"] = np.asarray(working_size)
            saved["native_size"] = np.asarray(native_size)
            saved["native_scale"] = np.asarray(native_scale)
        totals["batches"] += 1
        return result

    def probed_select(scores, camera_eligible, rectangle_ids, templates, corners, visibility,
                      min_visible_lengthwise, min_visible_cross_court, *rest, **options):
        admission = original_select(scores, camera_eligible, rectangle_ids, templates, corners, visibility,
                                    min_visible_lengthwise, min_visible_cross_court, *rest, **options)
        everything = np.arange(len(scores))
        admitted_pool = np.flatnonzero(admission.visibility_admitted)
        floor_prefix = prefix_needed(scores, rectangle_ids, templates, admitted_pool,
                                     camera_eligible, admission.scanned)
        zero_prefix = prefix_needed(scores, rectangle_ids, templates, everything,
                                    camera_eligible, admission.floor_zero_scanned)
        np.savez(os.environ["CAMERA_PROBE_OUT"], **saved)
        print(f"camera probe: batches {totals['batches']}, homographies {totals['homographies']}, "
              f"vector {totals['vector_s']:.2f} s, whole check {totals['whole_s']:.2f} s, "
              f"frontier rechecks {totals['rechecks']}", flush=True)
        print(f"selection: scanned {admission.scanned} eligible; score-order prefix needed {floor_prefix} "
              f"of {len(admitted_pool)} visibility-admitted ({floor_prefix / len(admitted_pool):.2%}); "
              f"floor-zero pass needs {zero_prefix} of {len(scores)} ({zero_prefix / len(scores):.2%})", flush=True)
        os._exit(0)

    line_template_source.vector_camera_errors = timed_vector
    line_template_source.camera_errors_with_frontier_recheck = probed_whole
    line_template_source.select_with_visibility_floor = probed_select


def main() -> None:
    wrap = run_d17.StageClock.wrap

    def wrap_after_hooks(clock, owner, name, label=None):
        # run_d17 imports the modules inside main; patch just before it adds its camera-check timer.
        if getattr(owner, "__name__", "") == "line_template_source" and name == "camera_errors_with_frontier_recheck":
            install(owner)
        wrap(clock, owner, name, label)

    run_d17.StageClock.wrap = wrap_after_hooks
    run_d17.main()


if __name__ == "__main__":
    main()
