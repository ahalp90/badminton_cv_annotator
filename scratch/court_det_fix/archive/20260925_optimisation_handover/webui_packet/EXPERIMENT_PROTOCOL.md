> Historical record from the web-UI handover packet of 23 September 2026,
> filed on 25 September 2026. The
> [speed-up README](../../../court_detector_optimisation_handover/README.md)
> owns current status; the [archive map](../../README.md) records the original
> paths. Content is unchanged. The packet's `tools/`, `task_manifest.json`,
> `LOCAL_MODEL_PROMPT.md` and single-file copy are in git history at commit
> 92535b6e.

# Local and HPC experiment protocol

## 1. Data policy

Use the complete frozen development population, existing references, controls,
saved user rulings, and uncommitted local inputs already available to the
checkout. Do not spend time creating new annotations or proposing a new held-out
set.

The first optimisation gates are paired **behaviour-equivalence** tests, not
claims of generalisation. Existing references are used later to detect quality
regressions, not to drive runtime pruning.

## 2. Baseline cohorts

### Bounded profiling pair

Start with two existing cases:

- **fast broadcast:** `shuttleset_03_scene_0019` or the locally fastest existing
  broadcast case;
- **slow GX:** `gxBQ_window_00_frame_0` or the locally slowest existing GX case.

Select from recorded timings, not intuition. Use one fast case for rapid
iteration and one slow case so launch overhead does not dominate.

### Depth/coverage witnesses after a patch passes

Use the already frozen set, including:

- `am2_window_01_frame_28019` — deeper-axis coverage witness;
- `gxBQ_window_00_frame_5` — case rescued by non-G0 populations;
- `am1_window_00_frame_54` — seeded-template recovery;
- existing regression cases and labelled non-court controls.

Retain G0, G1, and templates in the final combined validation. G0-only speed
tests are useful diagnostics but not sufficient quality evidence.

## 3. Environment controls

Set before importing NumPy/OpenCV:

```bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
```

Inside each process:

```python
cv2.setNumThreads(1)
```

Record:

```bash
git rev-parse HEAD
git branch --show-current
git status --short
python --version
python -m pip freeze
lscpu
free -h
nvidia-smi  # only when present; GPU use is not assumed
```

Use fresh output directories. Never overwrite the accepted baseline.

## 4. Timing boundaries

For each run, report wall and CPU time separately for:

1. import/startup;
2. view preparation and direction estimation;
3. each candidate source: G0, G1, templates;
4. axis enumeration/scoring;
5. axis-pair combination, geometry, player checks, and finite scoring;
6. per-pair/global retention;
7. full evidence and gates;
8. refit;
9. bounded net scoring and stripe correction;
10. ranking/selection;
11. serialisation;
12. video decoding/cut detection/view matching, when applicable.

Also record peak RSS with `/usr/bin/time -v`. Report cold and warm runs
separately; do not average startup into the processing target.

## 5. Profiling progression

### Smoke

Use the existing `--max-matched-pairs 1` facility on one case to validate code,
records, and tests. A smoke is not performance evidence.

### Fast full case

Run one complete fast case, one worker. Use:

```bash
/usr/bin/time -v   python scratch/court_det_fix/svd_search/run.py     --output <fresh-output>     --case shuttleset_03_scene_0019     --arm baseline     --workers 1
```

Adapt the runner to select baseline vs patched implementation without changing
inputs or caps.

### Slow full case

Run one complete slow case after the fast case passes equivalence.

### Frozen quality population

Only after both bounded cases pass, run the existing frozen quality corpus and
controls. Parallelise by case on HPC. Do not launch a broad run to diagnose a
local code bug.

## 6. Profilers

Start with deterministic instrumentation and `cProfile`:

```bash
python -m cProfile -o <out>/case.prof <runner.py> <args...>
python - <<'PY'
import pstats
pstats.Stats("<out>/case.prof").strip_dirs().sort_stats("cumulative").print_stats(80)
PY
```

Use `py-spy` or Scalene only if already installed and needed to separate native
NumPy/OpenCV time. Do not add a large dependency merely to profile.

Instrument counters alongside time:

```text
direction_pairs_seen / matched
axis_parameters_total / player_compatible / scored
score_batches
axis_pairs_total / necessary-player-compatible
geometry_valid
joint-player-valid
finite-scored
Python candidate objects constructed
distance-map cache hits/misses/bytes
gate candidates per batch
stripe candidates per batch
refit attempts / cache hits / residual calls / nfev
```

## 7. Equivalence tiers

### Tier A — exact refactor

Required for lifetime/caching/order patches:

- same case and source records;
- same pair status/order;
- same retained axis IDs, matches, and anchors;
- same per-pair/global retained candidate identities;
- same winner, acceptance, abstention, and source;
- exact integer/boolean/string fields;
- exact arrays where arithmetic did not change.

Timing, CPU, log paths, and diagnostic counters may differ.

### Tier B — arithmetic rewrite

Required for blocked algebra, analytic player mapping, vector camera errors,
batched stripe work, and analytic Jacobian:

- all Tier-A semantic identities and decisions match;
- no candidate crosses a hard threshold without baseline scalar recheck;
- candidate corners and scores within declared tolerances;
- same stable order after fallback;
- same assignment identities for stripe/refit;
- record maximum and quantile numerical deltas.

Initial suggested diagnostics, to tighten or justify rather than silently relax:

```text
corner delta: <= 1e-9 working px for pure algebra, or explicit scalar fallback
score delta:  <= 1e-12 away from thresholds/ties
hard-gate frontier: baseline re-evaluation
```

### Tier C — scene reuse

- full detector remains the fallback;
- same frozen-frame selected geometry or a documented, existing-reference
  comparison;
- no increase in false accept/abstention on existing controls;
- detector-call count and view-group count reported.

## 8. Record comparison

Use:

```bash
python /path/to/packet/tools/compare_records.py   baseline.json.gz candidate.json.gz   --ignore-key-regex '(^|_)(elapsed|wall|cpu)_s$|timing|generation_log'   --atol 0 --rtol 0
```

For Tier B, set explicit tolerances and inspect every reported frontier
difference. The comparator exits non-zero on mismatches.

Use `tools/summarise_generation.py` on both records to ensure work was actually
removed rather than moved outside the timer.

## 9. HPC execution

- Pin one Git revision and record it in every output.
- Transfer uncommitted data by explicit `rsync`; do not infer it from similarly
  named tracked files.
- Store an input manifest with path, byte size, and checksum for critical local
  inputs.
- Use one job per `(case, arm)` first, one numerical thread per job.
- Avoid nested worker pools.
- Use job arrays for independent cases and gather deterministically.
- Keep baseline and patch outputs in separate immutable directories.
- Capture stdout/stderr, exit code, scheduler job ID, host, CPU model, memory,
  and environment.
- Do not relaunch completed historical SVD jobs.

## 10. Decision rules

Accept a patch when:

- it passes the appropriate equivalence tier;
- it reduces the measured target stage and total wall time on at least one
  representative full case;
- it does not merely shift work outside the timing boundary;
- memory and CPU efficiency remain reasonable;
- the mechanism is understood.

Reject or stop when:

- a matched run shows no useful saving;
- a threshold/ranking change is unexplained;
- a cache has low hit rate or harmful memory cost;
- the patch needs search-cap/source changes to look fast;
- the result depends on reference labels at runtime;
- a broad run is being used to debug a local mismatch.

## 11. Final reporting

Produce one table per accepted/rejected patch:

| Patch | Cases | Stage wall before/after | Total wall before/after | Peak RSS | Output match | Decision |
| --- | --- | ---: | ---: | ---: | --- | --- |

Then report the complete fresh detector per distinct view and the video-level
pipeline separately:

```text
startup
first distinct-view search
additional compatible-view validation
total five-minute video processing
number of cuts
number of distinct compatible views
number of full detector calls
fallback count
```
