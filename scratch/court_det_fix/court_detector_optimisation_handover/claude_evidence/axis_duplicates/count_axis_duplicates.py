"""Count bit-identical axis hypotheses among those match_axis scores (throwaway, observe only).

Runs run_d17.main unchanged, with run_given.match_axis wrapped. After each call, the rows it
scored (the player-compatible ones) are compared by the raw bits of their (scale, shift)
parameters. Only bit-identical rows are guaranteed identical scores, so near-duplicates don't count.
Writes one JSON line per call to AXIS_DUPES_OUT.
Usage (from the checkout's scratch/court_det_fix/d17_timing): count_axis_duplicates.py <run_d17.py arguments>
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
import run_d17  # pyrefly: ignore[missing-import]


def install_hook(run_given, output: Path) -> None:
    original = run_given.match_axis

    def match_axis_and_count(basis, axis, coordinates, *args, **kwargs):
        started = time.perf_counter()
        matches = original(basis, axis, coordinates, *args, **kwargs)
        seconds = time.perf_counter() - started
        scored = matches.parameters[matches.player_compatible]
        # One int64 pair per row: identical bits, including signed zeros, or not identical at all.
        unique_rows = len(np.unique(np.ascontiguousarray(scored).view(np.int64), axis=0)) if len(scored) else 0
        row = {"axis": axis, "enumerated": len(matches.parameters), "scored": len(scored),
               "unique_scored": unique_rows, "seconds": seconds}
        with open(output, "a") as stream:
            stream.write(json.dumps(row) + "\n")
        return matches

    run_given.match_axis = match_axis_and_count


def main() -> None:
    output = Path(os.environ["AXIS_DUPES_OUT"])
    wrap = run_d17.StageClock.wrap

    def wrap_after_hook(clock, owner, name, label=None):
        # run_d17 imports the modules inside main; hook just before it adds its own timer.
        if getattr(owner, "__name__", "") == "run_given" and name == "match_axis":
            install_hook(owner, output)
        wrap(clock, owner, name, label)

    run_d17.StageClock.wrap = wrap_after_hook
    run_d17.main()


if __name__ == "__main__":
    main()
