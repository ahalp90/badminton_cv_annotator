---

<!-- README.md -->

# Court detector optimisation handover

**Repository:** `ahalp90/badminton_cv_annotator`  
**Reviewed branch:** `fix/court-det`  
**Reviewed GitHub revision:** `2a00f77172a59374136e19246ede1d854aaaff01`  
**Static review date:** 2026-09-23

## Executive answer

Yes. The accepted detector contains several gross inefficiencies that can be
removed before changing its candidate sources, search caps, thresholds, ranking,
or accepted output semantics.

The largest high-confidence opportunities are:

1. **Apply an existing rejection gate before the expensive work it rejects.**
   `projective_seed.match_axis()` scores every axis interpretation before
   calculating `necessary_players()`, even though `player_compatible` is later
   required for eligibility. Calculate that mask first and do not score rejected
   hypotheses in the production path.
2. **Stop recomputing fixed linear algebra.** A direction-pair basis is inverted
   in `offsets()`, again while rectifying feet, and again in every 256-hypothesis
   `score_axes()` batch. Group lines, homogeneous endpoints, normalisation
   matrices, template-pair indexes, and the basis inverse should be prepared once
   at their natural lifetime.
3. **Do not invert every combined court.** `propose_role()` can combine as many as
   `512 × 512 = 262,144` axis pairs, after which
   `zone_net.player_fractions()` performs a generic batched 3×3 inverse for every
   court. Player coordinates can be obtained directly from the already rectified
   feet and the two axis scale/shift parameters.
4. **Do not project the same corners twice.** `canonicalise()` projects all
   transforms to decide the 180-degree relabelling; `geometry()` immediately
   projects the canonical transforms again. Return and reorder the first
   projection, including denominators, and validate it once.
5. **Batch the 256-candidate evidence stage.** `evaluate_pool()` calls stripe
   measurement and `gate_evidence()` one candidate at a time.
   `gate_evidence()` even rebuilds the same feet array for every candidate, while
   the underlying projection and score routines already accept arrays.
6. **Materialise records only after array-level retention.** The generator creates
   Python `Candidate` objects and large provenance dictionaries for every usable
   court, then keeps at most 256. Keep compact arrays through scoring and
   diversity selection; construct objects/JSON for survivors.
7. **Move compatible-view reuse before full search.** Current scene grouping is
   downstream of court inference, so it cannot avoid detector calls. A
   conservative validate-or-fallback path is required to reach the video-level
   target, but it should follow first-search acceleration rather than conceal it.

These are distinct from reducing G0/G1/template coverage or shrinking search
caps. The latter are semantic experiments and are deliberately outside the
first optimisation wave.

## Scope and confidence

This packet is a static source audit plus an execution plan. It uses the
repository's recorded timings, which show that fresh generation dominates and
that the later scoring/refit stage is itself tens of seconds per frame. It does
**not** pretend that a local profiler was run against the uncommitted development
corpus. The local model must verify the actual checkout, imported module paths,
working tree, hardware, and data inventory before modifying code.

The reviewed Git tree contains multiple frozen snapshots. Import precedence is
part of the program. Resolve `module.__file__` in the real local run before
editing; do not optimise a similarly named historical copy.

## Non-negotiable behaviour contract

Keep all of the following until a separately labelled semantic experiment proves
otherwise:

- G0, G1, and line-template candidate sources.
- SVD12 as the default direction-group screen, with full16 available.
- Existing candidate caps in the baseline arm.
- Existing gates, bounded net selection, source coverage, ranking, tie order,
  abstention behaviour, and selection-then-stripe-correction order.
- Float64 geometry unless a separate numerical-equivalence trial passes.
- Existing frozen development data and controls. **Do not request or create new
  annotations or a new held-out set.**
- Local uncommitted data, caches, and historical outputs. Never reset, clean, or
  overwrite them.

## Packet contents

| File | Purpose |
| --- | --- |
| `STATIC_AUDIT.md` | Call graph, complexity, concrete inefficiencies, and safety classification |
| `PATCH_QUEUE.md` | Ordered implementation backlog with invariants and stop conditions |
| `EXPERIMENT_PROTOCOL.md` | Reproducible local/HPC profiling and equivalence protocol |
| `SOURCE_MAP.md` | Actual source/import map and local-data checklist |
| `LOCAL_MODEL_PROMPT.md` | Ready-to-use prompt for the local frontier model |
| `task_manifest.json` | Machine-readable tasks, dependencies, and acceptance gates |
| `tools/bootstrap_session.sh` | Non-destructive session/bootstrap inventory |
| `tools/resolve_hotpath.py` | Print the modules actually imported by the W5 path |
| `tools/summarise_generation.py` | Quantify recorded matcher/combine work |
| `tools/compare_records.py` | Recursive JSON/JSON.GZ equivalence comparator |
| `court_detector_optimisation_handover.md` | Single-file concatenation of the packet |

## First local actions

From the repository root:

```bash
bash /path/to/packet/tools/bootstrap_session.sh
python /path/to/packet/tools/resolve_hotpath.py   --court-root scratch/court_det_fix
python /path/to/packet/tools/summarise_generation.py   scratch/court_det_fix/svd_search/run_20260923/generation/baseline/<case>.json.gz
```

Then capture one **fast broadcast** and one **slow GX** baseline with one worker,
one numerical thread, fresh output directories, `/usr/bin/time -v`, and stage
timings. Use the existing frozen cases; no annotation work is needed.

The first implementation should be **P1: lifetime-correct prepared geometry and
basis inverse reuse**, followed by **P2: player pruning before axis scoring**.
Both are bounded, measurable, and do not require changing search coverage.


---

<!-- STATIC_AUDIT.md -->

# Static audit of the accepted court detector

## 1. Reviewed execution path

At the reviewed revision, the expensive fresh G0 path is assembled from snapshot
modules rather than one production detector module:

```text
svd_search/run.py
  -> generation._new_direction(...)
  -> w5_holistic/automatic_generation.generate(...)
       -> next_steps.../run_population.prepare(...)
       -> assignment.prepare_observations(...)
       -> permutations(selected direction groups, 2)
       -> run_given.propose_role(...)
            -> projective_seed.basis_for(...)
            -> projective_seed.match_axis(..., X_COORDS)
            -> projective_seed.match_axis(..., Y_COORDS)
            -> projective_seed.combine(...)
            -> run_given.canonicalise(...)
            -> scan_population.geometry(...)
            -> zone_net.player_fractions(...)
            -> run_given.finite_scores(...)
       -> per-pair retain
       -> global retain
       -> run_automatic.evaluate_pool(...)
  -> W5 parent measurement, refit, and ranking
```

`run_w5.add_helper_paths()` places
`next_steps_20260916/webui_seed/source` before the frozen helper directories.
The local model must print actual `module.__file__` values because uncommitted
files can change this resolution.

## 2. Why the search explodes

With SVD12, the generator examines `12 × 11 = 132` ordered direction-role pairs.

For one axis, `match_axis()` enumerates:

```text
C(retained observation groups, 2)
× 2 × C(number of canonical coordinates, 2)
```

At the configured `max_groups=128`, this is 162,560 hypotheses for a five-line
coordinate axis and 243,840 for a six-line coordinate axis. Across 132 direction
pairs, that is an upper bound of roughly 53.6 million axis interpretations.

The current batch size is 256. At the same upper bound, `score_axes()` is called
about 209,616 times, and each call calculates `np.linalg.inv(basis)` even though
the basis is unchanged for the direction pair. Its endpoint residual tensor also
represents roughly 76.9 billion scalar endpoint-distance components before
filtering, although actual cases depend on retained group counts.

After axis retention, `combine()` can create `512 × 512 = 262,144` court
homographies per direction pair, or 34.6 million across 132 pairs. The current
joint player test generically inverts every combined 3×3 homography.

These are upper bounds, not measured case counts. Run
`tools/summarise_generation.py` on local generation records to obtain actual
counts.

## 3. Ranked findings

### F1 — Existing player rejection is sequenced after axis scoring

**Files/functions**

- `frozen_helpers_20260914/axis_matching/projective_seed.py`
- `match_axis()`, `score_axes()`, `necessary_players()`

`match_axis()` scores each batch and only then computes `necessary_players()`.
Eligibility is ultimately:

```python
pattern & player_compatible
```

Therefore, a hypothesis that fails `player_compatible` cannot be retained,
regardless of its score, matches, or support count.

**Safe change**

1. Rectify feet once using the prepared basis inverse.
2. Compute `player_compatible` for all parameters, in blocks.
3. Call `score_axes()` only for compatible parameter indexes.
4. Initialise rejected rows to `score=-inf`, `matches=-1`, `supported=0` in the
   lean path.
5. Keep an optional full-diagnostics mode that computes rejected rows if a
   historical record consumer requires them.

**Safety class:** semantic-preserving for retained candidates.  
**Likely impact:** potentially large; measure the rejected fraction before
claiming a speedup.

### F2 — Basis inversion and frame geometry are recomputed at the wrong lifetime

**Files/functions**

- `projective_seed.group_lines()`
- `projective_seed.offsets()`
- `projective_seed.score_axes()`
- `projective_seed.match_axis()`

Repeated work includes:

- rebuilding group-leader endpoints and homogeneous line coefficients for both
  axes of every direction pair;
- recalculating image normalisation matrices;
- inverting the same 3×3 basis in `offsets()`, feet rectification, and every
  scoring batch;
- rebuilding template coordinate-pair arrays every axis call;
- repeatedly appending homogeneous ones to fixed points.

**Safe change**

Introduce immutable prepared structures with explicit lifetimes:

```text
FramePrepared:
  group_endpoints
  group_lines
  homogeneous_group_endpoints
  group_lengths
  normaliser / inverse normaliser
  template coordinate pair tables
  homogeneous court corners and marking endpoints

PairPrepared:
  basis
  basis_inverse
  rectified feet
  line coefficients or endpoint coefficient terms
```

Pass these into both axis matches.

**Safety class:** exact work removal, subject to preserving dtype/order.

### F3 — `score_axes()` forms a large general tensor for an affine line family

For one axis, each predicted line is:

```text
inverse[axis] - predicted_coordinate * inverse[2]
```

The current code builds line arrays and evaluates every endpoint via a large
`einsum`. The endpoint terms can be prepared once:

```text
A = endpoint_h @ inverse[axis]
B = endpoint_h @ inverse[2]
signed_distance_numerator = A - predicted_coordinate * B
```

The line normal is likewise affine in the predicted coordinate. This removes
repeated matrix work and can avoid the `batch × markings × groups × endpoints`
line tensor.

**Implementation options**

1. NumPy blocked kernel using prepared `A`/`B`.
2. Numba `njit(cache=True, fastmath=False)` kernel if Numba is already available.
3. A small C++/Cython extension only if the first two are insufficient.

Do not start by moving the entire pipeline to JAX/CuPy or float32. Dependency and
numerical changes would obscure whether the algorithm is simply doing redundant
work.

**Safety class:** algebraically equivalent, but floating evaluation order changes.
Use threshold-frontier fallback and tight paired checks.

### F4 — Batch size 256 causes excessive Python/NumPy launch overhead

A worst-case axis needs 953 scoring iterations at batch 256. The kernel is already
vectorised. Select a batch size from a memory budget, typically 2,048–8,192 on
the HPC node, rather than a hard-coded 256.

Record peak RSS and choose the largest size that does not create memory pressure.
The retained order must be based on original global indexes, not batch completion
order.

**Safety class:** semantic-preserving; numerical arrays should match within
roundoff.

### F5 — Combined courts are materialised and inverted before cheap exact gates

**Files/functions**

- `projective_seed.combine()`
- `run_given.propose_role()`
- `legacy/zone_net.py::player_fractions()`

Current sequence:

1. form all retained-axis pairs;
2. materialise every homography;
3. project all corners in `canonicalise()`;
4. project all corners again in `geometry()`;
5. invert every homography in `player_fractions()`;
6. keep only geometry/player-valid courts.

The court chart coordinate of a foot can instead be calculated from the axis
parameters:

```text
rectified = image_foot @ basis_inverse.T
chart_x = rectified_x / rectified_w
chart_y = rectified_y / rectified_w
court_x = (chart_x - x_shift) / x_scale
court_y = (chart_y - y_shift) / y_scale
```

Precompute per-axis `inside_x`, `inside_y`, `far_y`, and `near_y` masks. Combine
them in axis-pair blocks. This is the same predicate as a generic inverse, without
constructing or inverting a matrix for every pair.

Also:

- apply the already-proved per-axis necessary-player masks before the Cartesian
  product;
- process axis pairs in deterministic blocks;
- project/canonicalise/validate each surviving block once;
- return corners and denominators from canonicalisation to geometry;
- materialise homographies only for candidates that need later evidence.

Use the baseline scalar path for any value within a small ULP-aware frontier of
`-0.15`, `0.5`, or `1.15`, or when scale/denominator is non-finite.

**Safety class:** exact predicate in real arithmetic; Tier-B numerical rewrite
requiring frontier checks.

### F6 — Invalid geometry still pays the full joint player cost

Even before the analytic rewrite, a low-risk patch is to evaluate
`player_fractions()` only for `np.flatnonzero(valid)`, then scatter the results
back. Invalid courts cannot become usable.

**Safety class:** exact.

### F7 — Distance transforms are rebuilt for every direction pair

`run_given.finite_scores()` constructs two raster distance maps from the retained
group IDs for each pair. Cache by a stable key containing:

```text
frame/evidence identity
working size
axis-0 retained raw group IDs
axis-1 retained raw group IDs
distance-map parameters
```

Do not key only by direction-pair or origin string. The launch notes explicitly
warn that evidence identity matters.

Use an LRU with a measured memory cap. Report hit rate; remove the cache if hits
are negligible.

**Safety class:** exact with a complete key.

### F8 — Python object/provenance creation occurs before 256-cap retention

`propose_role()` creates a `Candidate` and a provenance dictionary for every
usable combined court. `automatic_generation.generate()` creates additional
identity maps and dictionaries, then retains at most 256 per pair and 256
globally.

Keep parallel NumPy arrays:

```text
score
corners
x_axis_id
y_axis_id
rotation flag
original ordinal
```

Perform stable score ordering and the existing greedy corner-diversity rule on
arrays. Build `Candidate` objects and verbose provenance only for survivors.
Provide diagnostics levels:

- `off`: production summary only;
- `summary`: counts/timings/cap flags;
- `full`: historical arrays and pool sidecars.

**Safety class:** exact if stable tie order and diversity comparisons are
preserved.

### F9 — Candidate gates are evaluated one at a time

**Files/functions**

- `run_automatic.evaluate_pool()`
- `marking_diagnosis/run_diagnosis.py::gate_evidence()`

For each candidate, `gate_evidence()`:

- converts the same nested feet data to a NumPy array;
- builds a homography;
- calls vector-capable `_score()` with a batch of one;
- calls the scalar camera diagnostic.

Prepare feet, settings, maps, and homographies once. Batch:

- player fractions;
- detector `_score`;
- vector camera error, with the existing scalar frontier recheck pattern from
  `w5_holistic/line_template_source.py`;
- gate record construction.

`inspect_appearance.profiles()` is already batched; retain that implementation.

**Safety class:** semantic-preserving with scalar frontier rechecks.

### F10 — Stripe evidence is scalar despite shared observations and templates

`stripe_observations.measure()` loops candidate → marking → interval → position
and repeatedly projects the same positioned template geometry. Add
`measure_many()` that:

- precomputes the 36 positioned finite intervals in homogeneous court
  coordinates;
- batches homography/inverse calculation;
- reuses observation vectors, squared lengths, directions, and flattened samples;
- chunks candidates to bound memory;
- emits the same `StripeEvidence`/score records in original order.

A conservative first patch can batch only the projection and reverse-distance
parts, leaving assignment/scoring scalar.

**Safety class:** numerical-equivalent; compare assignments as well as aggregate
scores.

### F11 — Refit repeats evidence and uses an expensive finite-difference Jacobian

`fixed_stripe_refit.refine()` calls SciPy `least_squares(..., jac="3-point")` for
eight parameters. A central finite-difference Jacobian requires many residual
evaluations per solver iteration.

Proceed in this order:

1. Cache exact duplicate fits by a key containing geometry, observation/evidence
   identity, fixed assignments, model, and optimiser settings.
2. Pass nearest interval/sample information from stripe evidence into
   `fixed_stripe_refit.prepare()` instead of recomputing the same finite-segment
   distances.
3. Implement and test the piecewise analytic Jacobian of the projected
   point-to-segment residual, including the clipped-endpoint branches.
4. Keep the current solver, tolerances, and fallback. Re-run the baseline solver
   when the analytic-Jacobian result changes status, rank, projection validity,
   or objective beyond tolerance.

Changing to a different solver, `2-point`, looser tolerances, or fewer evaluations
is a semantic trial, not a free optimisation.

**Safety class:** steps 1–2 exact; analytic Jacobian Tier B.

### F12 — View grouping happens after inference

The existing ContentDetector → evidence → perceptual-hash/alignment grouping
cannot reduce first-pass court searches. Add an earlier path:

1. cheaply fingerprint a cut sample;
2. find compatible prior views;
3. estimate/validate alignment and cheap court-line/net evidence;
4. reuse/warp accepted geometry only when validation passes;
5. otherwise run the full detector and create a new view state.

Hash similarity alone is not proof of an unchanged camera. The full-search
fallback is mandatory. Test on the frozen video/development population; no new
labels are required.

**Safety class:** pipeline change with fallback. Treat separately from inner-loop
optimisation.

## 4. Lower-priority observations

- `_distance_maps()` draws line segments in a Python loop; `cv2.polylines` may
  reduce dispatch overhead, but caching and call elimination are more important.
- `_visible_samples()` repeatedly allocates identical fraction vectors; cache
  them by sample count.
- `detector.project()` repeatedly concatenates homogeneous ones for fixed
  templates; keep homogeneous constants.
- JSON compression and verbose pair records can be expensive, but recorded
  timings already stop before some reporting. Measure before prioritising.
- SVD ranking is not the bottleneck; it already removed roughly half of matcher
  work in the recorded benchmark.
- OpenCV/BLAS thread counts must remain controlled. Oversubscription can make a
  vectorised patch look slower or nondeterministic.

## 5. Things not to call “optimisations” without quality evidence

Do not mix these into the first patch series:

- removing G0, G1, or line-template sources;
- reducing direction groups below SVD12;
- reducing `keep_axes`, per-pair cap, or global cap;
- skipping difficult cases or non-court controls;
- moving stripe correction before selection;
- changing thresholds, net weight, overrun, ranking, or abstention;
- replacing float64 geometry with float32;
- using reference labels to prune runtime work;
- accepting a different winner merely because it looks close numerically.

Those alter coverage or decisions and need a separately named experiment.


---

<!-- PATCH_QUEUE.md -->

# Ordered optimisation patch queue

Every patch gets its own output directory and before/after record. Restore the
accepted baseline between one-factor trials. Do not combine patches until each
one has a measured saving and passes its own equivalence gate.

## P0 — Baseline and import lock

**Goal:** establish what is actually running.

Actions:

1. Record `git rev-parse HEAD`, branch, `git status --short`, Python/package
   versions, CPU/GPU, RAM, OpenCV build, and all thread environment variables.
2. Run `tools/resolve_hotpath.py`.
3. Inventory local-only inputs named in `SOURCE_MAP.md`.
4. Capture one fast broadcast case and one slow GX case with:
   - one process;
   - OpenCV and numerical threads set to one;
   - cold and warm runs separated;
   - stage wall/CPU time;
   - `/usr/bin/time -v`;
   - full output records.
5. Run `tools/summarise_generation.py` on the generation records.
6. Add low-overhead counters around the concrete functions that dominate the
   profile. Keep instrumentation off by default.

**Exit gate:** reproducible baseline records and resolved module paths.  
**Stop:** if the local import path differs from the reviewed path, update the
source map before editing.

## P1 — Prepared frame/pair geometry and inverse hoist

**Files:** active `projective_seed.py`, callers, focused tests.

Create immutable `PreparedObservations` and `PreparedPair` structures or
equivalent explicit arguments. Reuse:

- group leader endpoints and homogeneous lines;
- normalisation matrices;
- template pair tables;
- homogeneous fixed template points;
- basis inverse;
- rectified feet.

Do not silently mutate the public `Observations` record used by saved outputs.

**Acceptance**

- exact retained axis indexes, matches, anchors, support counts, and scores in
  full-diagnostics mode;
- exact candidate identities/order/winners;
- lower `score_axes` inverse count, ideally one inverse per pair.

## P2 — Player gate before axis score

Compute `player_compatible` before `score_axes()`. Score only compatible rows in
lean mode; retain a diagnostic switch for complete rejected-row arrays.

Add counters:

```text
axis_parameters_total
axis_parameters_player_compatible
axis_parameters_scored
```

**Acceptance**

- retained axes and all downstream candidates exactly match baseline;
- `axis_parameters_scored <= baseline`;
- useful wall-time reduction on at least one representative case;
- no misleading comparison if the selected cases happen to prune nothing.

## P3 — Larger/adaptive scoring blocks

Replace hard-coded batch 256 with a setting chosen from an explicit memory
budget. Preserve global indexes and stable sorting.

Trial only a few consequential sizes, for example 2,048 and 8,192. This is not a
parameter sweep; stop when launch overhead is no longer material or memory
becomes limiting.

**Acceptance**

- same retained rows and decisions;
- peak RSS recorded;
- no oversubscription;
- useful speedup after P1/P2.

## P4 — Prepared endpoint scoring kernel

Use precomputed endpoint coefficients for
`inverse[axis] - coordinate * inverse[2]`. Start with blocked NumPy. Use Numba
with `fastmath=False` only if NumPy remains dominant and Numba is already
available on target systems.

Implement scalar fallback for rows whose residual/support comparison is within a
small numerical frontier of `SUPPORT_DISTANCE_PX`, or whose score tie can alter
stable order.

**Acceptance**

- same support mask/counts and nearest group IDs;
- score differences within the declared tolerance;
- exact retained identities/order;
- no threshold crossing without baseline fallback.

## P5 — Lean axis-pair combination

Implement a production path that:

1. intersects retained axis indexes with the existing necessary player masks;
2. combines pairs in deterministic blocks;
3. computes joint player predicates from rectified feet and axis scale/shift;
4. projects/canonicalises/validates corners once;
5. finite-scores only usable blocks;
6. stores compact arrays;
7. performs stable array-level diversity retention;
8. materialises records for survivors only.

Keep the current implementation callable as an oracle during development.

**Acceptance**

- same pair statuses, usable IDs, per-pair retained IDs, global retained IDs, and
  downstream winners;
- scalar baseline recheck for threshold-frontier rows;
- large reduction in generic 3×3 inversions, duplicate corner projections, and
  Python object count;
- compare both fast and slow cases before expanding.

## P6 — Exact distance-map cache

Cache finite-score maps using full geometry/evidence keys. Record lookups, hits,
misses, bytes, and evictions.

**Acceptance**

- byte/equality-equivalent maps on sampled hits;
- same candidate output;
- meaningful hit rate and net wall saving;
- remove the cache if hit rate or memory economics are poor.

## P7 — Batched gate measurement

Prepare feet and homographies once, batch `_score` and player fractions, and use
vector camera errors with scalar frontier rechecks. Keep output construction in
the same candidate order.

**Acceptance**

- identical gate validity, line counts, player fractions, and eligibility;
- camera values equal within tolerance and exactly rechecked near the hard gate;
- same ranking and abstention.

## P8 — Batched stripe measurement

Add chunked `measure_many()` and reuse fixed positioned template geometry and
observation arrays.

**Acceptance**

- per-fragment marking/position assignments match;
- forward/reverse/paired scores match within tolerance;
- same parents admitted to refit and same final rank;
- peak RSS bounded.

## P9 — Refit cache and Jacobian

First implement complete-key duplicate caching and evidence-to-constraint reuse.
Only then implement an analytic Jacobian with the current solver as fallback.

**Acceptance**

- same fit status, child existence, hard validity, and final selected key;
- fitted corners within a tight working-pixel tolerance;
- objective and Jacobian diagnostics recorded;
- baseline fallback on any ambiguous frontier;
- useful reduction in residual calls and refit wall time.

## P10 — Deterministic parallelism

Parallelise at only one layer at a time.

Preferred order:

1. case/view jobs across HPC workers;
2. direction pairs within one very slow view if case-level parallelism is
   insufficient;
3. independent refits only if profiling still justifies it.

Workers return compact arrays/records. The coordinator sorts by original
`pair_id`/parent order before retention/ranking.

Set all numerical and OpenCV threads to one inside workers. Avoid nested
case-process × pair-process pools.

**Acceptance**

- same output as one worker;
- same output under at least two worker counts;
- no memory multiplication that erases the speedup;
- report CPU efficiency, not wall time alone.

## P11 — Early compatible-view reuse

After first-search latency is acceptable, add conservative view reuse before
full court inference. Use existing cuts, hashes, alignment, and scene contracts,
but require cheap geometry/evidence validation and full-search fallback.

**Acceptance**

- frozen-video outputs match full-search results or differences are explicitly
  adjudicated using existing references/rulings;
- every uncertain validation falls back;
- detector calls scale with distinct compatible views rather than raw cuts;
- startup/decoding and first-search time reported separately.

## Integration order

Merge only accepted patches in this order:

```text
P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P9 -> P10 -> P11
```

P3 and P4 may be reordered after profiling. P6 is conditional on cache hits.
P9 analytic Jacobian is optional if P7/P8 and caching already meet the refit
budget. P11 is a separate pipeline checkpoint.


---

<!-- EXPERIMENT_PROTOCOL.md -->

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


---

<!-- SOURCE_MAP.md -->

# Source and data map

## Reviewed revision

- Branch: `fix/court-det`
- GitHub revision: `2a00f77172a59374136e19246ede1d854aaaff01`
- Review date: `2026-09-23`

The local checkout may contain later commits and uncommitted modules/data.
Record actual state; do not reset to this revision.

## Resolve the live imports

Run:

```bash
python tools/resolve_hotpath.py --court-root scratch/court_det_fix
```

The reviewed `w5_holistic/run_w5.py::add_helper_paths()` establishes this
precedence:

1. `next_steps_20260916/webui_seed/source`
2. `frozen_helpers_20260914/marking_diagnosis`
3. `frozen_helpers_20260914/vp_pruning`
4. `frozen_helpers_20260914/axis_matching`
5. `frozen_helpers_20260914/legacy`
6. `src`
7. repository/court root

At the reviewed commit, the hot path is primarily:

| Concern | Reviewed path / function |
| --- | --- |
| Fresh SVD12 generation | `w5_holistic/automatic_generation.py::generate` |
| Pair proposal orchestration | `next_steps_20260916/webui_seed/source/run_given.py::propose_role` |
| Axis matching | `.../source/projective_seed.py::{basis_for, offsets, score_axes, match_axis, combine}` |
| Geometry and continuous support | `frozen_helpers_20260914/marking_diagnosis/scan_population.py::{geometry, continuous_support, retain}` |
| Player test | `frozen_helpers_20260914/legacy/zone_net.py::player_fractions` |
| Observation grouping | `experiments/annotator/independent_court/assignment.py` |
| Pool evidence | `.../source/run_automatic.py::evaluate_pool` |
| Gates | `frozen_helpers_20260914/marking_diagnosis/run_diagnosis.py::gate_evidence` |
| Stripe evidence | `experiments/annotator/independent_court/stripe_observations.py` |
| Refit | `experiments/annotator/independent_court/fixed_stripe_refit.py` |
| W5 orchestration/ranking | `w5_holistic/run_w5.py` and `w5_holistic/verifier.py` |
| Runtime benchmark runner | `svd_search/run.py` |
| Line-template vector camera precedent | `w5_holistic/line_template_source.py` |

Do not edit a frozen snapshot until the import resolver confirms it is the active
copy or the change deliberately creates a new integrated module.

## Existing tests and useful checks

- `frozen_helpers_20260914/axis_matching/test_projective_seed.py`
  - synthetic recovery;
  - degenerate directions;
  - horizon handling;
  - proof that per-axis player rules are necessary for the joint rule;
  - finite-score rotation invariance.
- `net_recovery/test_bounded_trial.py`
- W5 verifier/ranking tests named in the repository handover.
- Scoped Ruff, Pyrefly, and syntax checks described by repository instructions.

Add focused tests beside the active module. Do not build an unrelated test
framework.

## Existing timing/evidence artefacts

- `svd_runtime/RESULTS.md` and sibling result records:
  full16 versus SVD12 matcher timing.
- `svd_search/run_20260923/cases/`:
  stage timings and counts for baseline/deeper/shortlist arms.
- `svd_search/run_20260923/generation/`:
  per-pair generation records.
- `svd_search/WORKLOG.md`:
  accepted/rejected quality observations.
- `net_recovery/statistics/paired_reference_report.md` and siblings:
  paired quality results.
- `w5_holistic/verifier.py`:
  fixed cohorts, references, hard validity, ranking, and record conventions.

Use existing records rather than rerunning completed historical experiments.

## Critical local-only inputs named by the repository handover

Verify presence and checksums before a new worktree or HPC transfer:

```text
local_scratch/net_recovery/20260923/bounded/bounded_trial.json.gz
local_scratch/net_recovery/20260923/selected_polarity/bounded_results.json.gz
local_scratch/net_recovery/20260923/selected_polarity/bounded_requests.json.gz
local_scratch/external_delegate/20260923-am1-recovery-trial/seeded_pool.json.gz
local_scratch/external_delegate/20260923-net-bounded-audit/result.md
```

Also inventory all untracked development packs and frozen run outputs referenced
by `scratch/court_det_fix/FP_INDEX.md`.

Never substitute a similarly named tracked pool. Cache identity must include
image hash/dimensions, geometry, evidence source, settings, and relevant code
revision.

## Documentation drift warning

Some older handovers refer to modules such as
`w5_holistic/court_detection.py`, `candidate_cache.py`, or `scoring.py`. They are
not present in the reviewed Git tree. Treat those names as historical design
language unless the local uncommitted checkout actually contains them.

## Production integration boundary

The current accepted quality result is assembled experimentally. No single
runner at the reviewed revision combines fresh SVD12 G0/G1 generation, seeded
templates, bounded net selection, and stripe refitting. First build a fresh,
reproducible baseline adapter or use the current local integrated runner if one
now exists; do not compare a fresh patch with historical paint scores.


---

<!-- LOCAL_MODEL_PROMPT.md -->

# Prompt for the local frontier model

You are taking over performance engineering of the accepted court detector in
`badminton_cv_annotator`.

Read this packet in order:

1. `README.md`
2. `STATIC_AUDIT.md`
3. `PATCH_QUEUE.md`
4. `EXPERIMENT_PROTOCOL.md`
5. `SOURCE_MAP.md`
6. repository `scratch/court_det_fix/pickup.md`
7. repository `handover_20260923/{00_SHARED_CONTRACT,01_LAUNCH_OPTIMISATION}.md`
8. repository instructions.

## Mission

Make the detector deployable without losing the accepted recall, precision,
source coverage, ranking, abstention, or final geometry behaviour. The target is
roughly 30 seconds for a five-minute video, with 90 seconds an upper-end
tolerance; report startup separately. Expensive search should eventually scale
with distinct compatible views.

There is no annotation budget. Do not ask for new held-out samples, new labels,
or manual review campaigns. You have the existing frozen development data,
references, controls, user rulings, uncommitted local data, and HPC.

## Safety and repository rules

- Begin by recording actual Git HEAD, branch, and status. Never reset, clean, or
  delete untracked data.
- Resolve active module paths at runtime. Multiple byte-identical and historical
  snapshots exist.
- Use fresh output directories. Preserve accepted historical results.
- Keep G0, G1, templates, SVD12, existing caps, gates, ranking, tie order,
  bounded net rule, abstention, and selection-then-stripe-correction.
- Do not use reference labels in runtime pruning.
- Keep one numerical/OpenCV thread per worker and avoid nested process pools.
- Separate exact optimisations from any semantic search-space experiment.
- Commit coherent accepted patches on the feature branch only after checks pass.

## Required first pass

1. Run the packet bootstrap and import resolver.
2. Locate the locally fastest existing broadcast case and a slow GX case from
   saved timings.
3. Capture cold/warm one-worker baselines with stage wall/CPU time, peak RSS,
   concrete work counters, and complete output records.
4. Profile the actual imported code.
5. Quantify:
   - total axis interpretations;
   - player-compatible interpretations;
   - `score_axes` calls;
   - basis inversions;
   - combined axis pairs;
   - geometry/player-valid pairs;
   - full-evidence candidates;
   - refit attempts/residual evaluations.

Do not launch the full corpus first.

## Implement in this order

### 1. Prepared geometry and inverse lifetime

Prepare frame-invariant group geometry/template tables once and pair-invariant
basis inverse/rectified feet once. Pass them explicitly. Prove the same retained
axis/candidate output.

### 2. Existing player gate before axis scoring

Calculate `necessary_players()` before `score_axes()` and avoid scoring rejected
parameters in lean mode. Keep a full-diagnostics oracle mode. This gate already
exists; you are changing sequence, not policy.

### 3. Adaptive scoring blocks and prepared endpoint algebra

Increase batch size under a memory budget. Then replace repeated general line
evaluation with the affine endpoint-coefficient form. Use scalar baseline
fallback at support/tie frontiers.

### 4. Lean combined-pair path

Avoid generic inversion of every combined court. Derive joint player predicates
from rectified feet and axis scale/shift; block the Cartesian product; project
and validate corners once; retain from compact arrays; materialise only
survivors.

### 5. Batch evidence and gates

Prepare feet once. Batch player fractions, detector `_score`, camera diagnostics,
profiles, and stripe evidence. Preserve order and frontier rechecks.

### 6. Refit

Add complete-key duplicate caching and reuse stripe nearest-distance work.
Implement an analytic Jacobian only with the current solver as a fallback.

### 7. Parallelism and scene reuse

Use deterministic case/pair/refit parallelism at one level. After first-search
cost is affordable, move compatible-view matching before full inference with
validate-or-full-search fallback.

## Validation

For each patch:

- create a separate output directory;
- compare against a freshly measured control on the same revision/hardware;
- use `tools/compare_records.py`;
- compare stable candidate identities, pair status/order, retained indexes,
  winner, acceptance/abstention, gates, and final corners;
- report numerical frontiers explicitly;
- run one fast and one slow full case before the frozen corpus;
- stop a patch when it has no useful saving or an unexplained output change.

Use existing references and controls only after equivalence passes to confirm no
quality regression. Preserve contrary cases and unlabelled rows in reports.

## Deliverables

Maintain a single optimisation session directory with:

```text
worklog.md
evidence.md
mechanisms.md
runs.md
```

For each accepted patch, record mechanism, files, commit, commands, hardware,
threads, before/after stage and total timing, peak memory, output comparison,
and remaining risk.

Finish with:

1. accepted/rejected patch table;
2. complete fresh detector cost per distinct view;
3. five-minute video cost split into startup, decoding/cuts, first searches,
   validation/reuse, fallbacks, and output;
4. paired quality/control results;
5. unresolved failures;
6. exact Git/worktree/check state.

Do the work now in the current session. Do not promise background work or ask for
new annotations.
