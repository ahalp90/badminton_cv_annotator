# Simple matcher optimisations to check after the SVD timing

Opus identified three worthwhile leads. The source inspection supports each
mechanism; the reported speedups come from small local measurements and
extrapolation. No matcher implementation has changed. Finish the fixed-code
12-versus-16 timing before measuring any of these changes.

| Priority | Change | Why it may help | Check before adopting |
| --- | --- | --- | --- |
| 1 | Run player checks only on geometrically valid courts | `propose_role` currently checks every combined court, then discards invalid geometry. In the sampled Am2-150 pair, only 8,398 of 262,144 courts passed geometry | Require the same usable-court mask, candidate IDs and output geometry on the nine cases |
| 2 | Replace the two-endpoint maximum reduction with `np.maximum` | The endpoint dimension is exactly two. The audit measured about 17.3 to 4.9 ms per scoring batch | Require exact scores, assignments and retained axis IDs; include the remote environment |
| 3 | Compute player-projection coordinates explicitly | The audit's scalar-coordinate expression avoids a large trailing-coordinate array and several short-axis reductions | Check rounding and court-boundary classifications on Carmack; reuse canonical court constants |

## Source checks and limits

- In `frozen_helpers_20260914/axis_matching/run_given.py`, `propose_role`
  obtains `valid` from `geometry`, computes `player_fractions` for all
  transforms, then forms `usable = valid & (one == 1) & (two >= .5)`.
  Player calculations are independent for each court. Skipping invalid courts
  therefore preserves the selected valid-court calculation.
- In `axis_matching/projective_seed.py`, `score_axes` constructs distances
  with one slot per segment endpoint. Its `max(axis=3)` can be expressed as
  a maximum of the two slices. This is the smallest arithmetic change.
- In `legacy/zone_net.py`, `player_fractions` uses `einsum` for image-to-court
  projection, then reductions over the two coordinates. An explicit expression
  changes floating-point evaluation details. Sample agreement is insufficient
  to prove all near-boundary decisions unchanged.
- The audit's `micro2.py` uses float64 decimal court dimensions. Production
  must reuse `zone_net.COURT_SIZE_M`, derived from canonical court coordinates.
  The real-data check compared resulting player fractions on one case; it did
  not establish identical projected coordinates for every input.

A boundary check confirms a difference in that prototype, exit 0:
with an identity homography and feet at `(6.1 * 1.15, 3)` and `(3, 10)`, the
existing function gives a both-halves fraction of 0; the decimal-literal
rewrite gives 1. Canonical dimensions are approximately `6.0999999046` and
`13.3999996185` metres because their source array is float32. Reusing that
existing array restores equality on this example. This is a prototype issue;
the deployed matcher is unchanged.

The audit estimated a roughly fourfold gain on a heavy pair from combining
its leads. That is **not a measured end-to-end or population speedup**. Some
full-player timings were extrapolated from smaller slices. Local profiling
overlapped other preparation work, so treat the timing magnitudes as leads.

Avoid changing axis deduplication for a saving below 0.1 seconds per pair.
Reusing canonicalised corner projections did not preserve exact arithmetic
in the audit. Axis fits use pair-specific coordinate systems, so cross-pair
reuse needs more reasoning and is outside this small optimisation pass.

## Audit record

The requested model was verified as `claude-opus-5-5`, high effort. It inspected
Am2-150 pair 1, GX5 pair 0, and a random additional function (`canonicalise`).
Its profiling and equivalence scripts are preserved under
`local_scratch/external_delegate/20260923-svd-compute-audit/`.
The audit reports all commands exited 0. Source inspection supports the
mechanisms described above; full-population equivalence remains untested.
Review targets were not edited.
