"""Profile W5 parent measurement on one D17 view, then stop (throwaway).

Runs run_d17.main with cProfile switched on only inside run_w5.make_parent_record, for the first
PARENTS parents. Then prints the functions with the most own time and exits without finishing the view.

Usage (from the checkout's scratch/court_det_fix/d17_timing), with MEASURE_PROBE_OUT set to a .prof
path: measure_probe.py <run_d17.py arguments>
"""

import cProfile
import os
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import run_d17  # pyrefly: ignore[missing-import]

PARENTS = 200
profiler = cProfile.Profile()
measured = {"parents": 0}


def install(run_w5) -> None:
    original = run_w5.make_parent_record

    def profiled(*args, **kwargs):
        profiler.enable()
        try:
            return original(*args, **kwargs)
        finally:
            profiler.disable()
            measured["parents"] += 1
            if measured["parents"] == PARENTS:
                profiler.dump_stats(os.environ["MEASURE_PROBE_OUT"])
                stats = pstats.Stats(profiler)
                stats.sort_stats("tottime").print_stats(30)
                stats.sort_stats("cumulative").print_stats(40)
                sys.stdout.flush()
                os._exit(0)

    run_w5.make_parent_record = profiled


def main() -> None:
    wrap = run_d17.StageClock.wrap

    def wrap_after_hooks(clock, owner, name, label=None):
        if getattr(owner, "__name__", "") == "run_w5" and name == "make_parent_record":
            install(owner)
        wrap(clock, owner, name, label)

    run_d17.StageClock.wrap = wrap_after_hooks
    run_d17.main()


if __name__ == "__main__":
    main()
