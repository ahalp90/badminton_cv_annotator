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
