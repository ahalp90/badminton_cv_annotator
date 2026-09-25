# Court detector check, 25 September 2026

The joined court detector (`../detect.py`, run through `../run_views.py`)
reproduces the 24 September D17 baseline on all 28 views. The research scripts
it was built from still reproduce their own outputs. Joining the stages saves
about 8-12% of compute time, which is about the size of run-to-run noise.

## What ran

Everything ran on Carmack, with at most 8 processes and one numerical thread
each. The runs used commit 42b054aa. Commit 07a15523 then made the harness's
baseline comparison stricter, and the stricter checks were applied to the saved
outputs. The two commits differ only in that comparison code.

1. **Correctness**: `run_views.py --baseline --feet --artefacts` in 8 processes
   of 2-6 views (`correctness/groups.txt`), so most views run after another
   view in the same process
2. **Rerun**: `run_d17.py` on the same 28 views through
   `run_feet_variants.sh`, for comparison with the baseline run's files
3. **Timing**: `run_views.py --timing`, one process per view, with self-checks
   on (`timing/`) and off (`timing_no_checks/`)

`run_all.sh` runs these three stages in order, and
`run_timing_no_checks.sh` waits for it to finish. Their headers give the
arguments. Both were started under `timeout` so the SSH session could not hang.

## Results

| Output | Made by | Result |
| --- | --- | --- |
| `correctness/results/`, `correctness/logs/` | `run_all.sh` | 28 of 28 views pass all 13 checks |
| `recheck_artefacts.txt` | `recheck_artefacts.py` | Stricter checks over the saved artefacts: 28 of 28 equal |
| `compare_rerun.txt` | `compare_rerun.py` | Rerun equals the baseline in every file, apart from 106 path strings |
| `check_paths.txt` | `check_paths.py` | Each of the 106 paths names its own run, with the same file after it; both files exist |
| `timing.txt` | `summarise_timing.py` | Per-view and total seconds, below |

The 13 checks are listed in `../../d17_timing/WIRING.md`, section "How to check
an integrated detector". Two limits apply:

- The line templates are compared by count and metadata only, because the
  baseline never saved the templates themselves
- The W5 comparison covers the fields the detector produces. It leaves out the
  research extras listed in `../STRIPPED.md`

`compare_rerun.py` compares key order, types and values, including stage names
and call counts. It skips durations and peak memory.

Seconds summed over the 28 views, 8 processes at a time:

| Run | Seconds |
| --- | ---: |
| `run_d17.py` rerun, its stages only | 7,288 |
| Detector, self-checks on | 6,737 |
| Detector, self-checks off | 6,434 |

The search takes about 64% of the detector's time and W5 about 33%. The same
`run_d17.py` code ran 11% faster than on 24 September, and single views swing by
30-40% between runs. The detector's times leave out start-up and video decoding.

## Paths and data kept on Carmack

`<remote>` stands for the Carmack run root. It replaces the real path in the
logs, the text outputs and the rerun's `d17/` summaries, and that swap is the
only edit to any output. The baseline run is in
`<remote>/d17_camera_20260924/standing/budget16/`.

From the rerun, this folder keeps the logs, the per-view summaries (`d17/`,
with stage timings) and the input audits (`inputs/`). The large outputs stay on
Carmack, listed with size and MD5 in `left_on_carmack.tsv`:

- the correctness run's per-view artefacts, 221 MB
- the rerun's arrays, case records and G0/G1 populations, 888 MB

`recheck_artefacts.py`, `compare_rerun.py` and `check_paths.py` need that data
and the baseline run, so they only rerun on Carmack.
