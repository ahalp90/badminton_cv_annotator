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
