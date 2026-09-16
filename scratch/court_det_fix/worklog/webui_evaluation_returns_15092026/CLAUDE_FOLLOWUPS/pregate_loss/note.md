# Courts before the player gate: the axis matching never builds the court

**Terms.** The matcher builds courts from a pair of selected directions by matching observed line groups to court markings along each direction ("axis matching", `match_axis` in the code), keeping the 512 best-scoring distinct matchings per direction ("the per-direction cap", `keep_axes`), and combining every kept horizontal matching with every kept vertical matching ("the combined courts", `combined` in the record). Two masks then run: the geometry mask (finite, convex, positive depth, visible span; `valid`) and the player mask (every sampled player foot inside the court, both halves occupied; `usable`). Courts that pass both are "the usable courts", which the earlier cap-loss check called "proposed". The direction fit (stage E3 of the direction experiment) is the least-squares court fitted to the control from the same pair of directions with the two vanishing points fixed; it says what the directions could support. The two selections tested are the midpoint-anchor selection (arm M) on GX0 and the precise-representative selection (arm R) on Amateur-3 frame 0. Distances are the maximum corner distance in working pixels (960 by 540) to the frozen control, with the 180-degree relabelling allowed.

**Bottom line.** Neither mask removes a close court. On both views the nearest court that exists before the masks is the same court that survives them: 33.8 px on GX0 and 40.1 px on Amateur-3. Nothing within 20 px exists among the 22.4 million (GX0) or 17.1 million (Amateur-3) combined courts. The direction pair behind the best direction fit is matched in both runs, and its own nearest combined court is 33.8 px on GX0 (pair 23, direction fit 19.7) and 46.1 px on Amateur-3 (pair 43, direction fit 4.3). So the loss between the direction fit and the pool lies inside the axis matching, before the masks and before the per-pair cap the cap-loss check measured. One layer stays unseparated here: the per-direction cap excluded 1,236 to 25,852 distinct matchings per direction on these pairs, so "never built" means "never built from the kept matchings".

## Run

- Remote folder `cap_loss/` under the paint-geometry root on the compute host, reused from the cap-loss run. The instrumented copies are in `remote_src/` here; `run_given.py` is the new copy, the other four match the cap-loss copies.
- Run name `pregate_20260915_115949`. Two detached jobs, `nice -n 10`, single-threaded, both exit 0: GX0 35.9 min (11:59:52 to 12:35:47 UTC), Amateur-3 20.9 min (to 12:20:46 UTC). Logs and receipts in `records/run_logs/`.
- Instrumentation: the copied court builder (`propose_role`) returns, beside its unchanged candidates, every combined court's corners (float32 working pixels, in the order the courts were combined), the geometry mask, the player mask and the two player fractions the player mask reads. The copied `run_automatic.py` stores them per matched pair in the side file beside the record, with the cap-loss arrays. Candidate construction and the JSON record are untouched.
- Records and side files in `records/new/<arm>/results/` (530 MB, cruft; re-derivable from the packs, the direction records and `remote_src/`).

## Gate 1: the instrumentation changed nothing

`gate1.py` reuses `cap_loss/gate_records.py`. Every pair status, shortlist candidate ID list, shortlist corners, final entry IDs, both winner IDs and the pooled count equal the saved generation records for both views (`gate1_output.txt`).

```
gxBQ_window_00_frame_0 arm M: PASS on all eight checks (240 pairs, 134 matched, 256 entries, winners 106:1732 and 211:19062, pooled 29696)
am3_window_00_frame_0 arm R: PASS on all eight checks (240 pairs, 100 matched, 256 entries, winners 32:2640 and 43:13964, pooled 22593)
GATE 1 PASS
```

## Gate 2: the recorded usable courts are the combined courts at the usable indices

`analyse.py` checks, for every matched pair, that the usable-court corners the cap-loss run recorded equal the combined corners at the usable indices exactly, that the record's three counts equal the array length and the two mask sums, that every usable court passes the geometry mask, and that the player mask recomputes from the stored fractions. All 134 and all 100 matched pairs pass. The nearest usable court per pair equals the cap-loss table's nearest proposed court to 0.0 px with the same index (`gate2_analysis_output.txt`).

## What the combined courts hold

| View | Direction fit | Combined | Passed geometry | Passed players | Nearest combined | Nearest passing geometry | Nearest usable | Within 20 px anywhere |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GX0, midpoint-anchor selection | 19.7 | 22,400,488 | 5,728,494 | 3,810,512 | 33.822 (23:203788) | 33.822 (same) | 33.822 (same) | 0 |
| Amateur-3, precise-representative selection | 4.3 | 17,066,316 | 3,956,934 | 2,449,344 | 40.140 (30:180443) | 40.140 (same) | 40.140 (same) | 0 |

GX0's control is visually approved; Amateur-3's is a manual reference. The court label is `pair:index into the combined array`. The nearest combined court passes both masks on both views: player fractions any 1.00 and both halves 1.00 on GX0, 1.00 and 0.97 on Amateur-3.

Per pair, the masks do remove the nearest combined court often: on GX0 the nearest combined court passes both masks in 58 of 116 pairs with combined courts, fails geometry in 25 and fails only the player mask in 33; on Amateur-3 it is 44, 37 and 15 of 96. Those removals matter only far from the control. Every pair whose nearest combined court is within 60 px keeps it through both masks: GX0 pair 23 at 33.8; Amateur-3 pairs 30, 44, 42, 43 and 32 at 40.1, 40.3, 45.5, 46.1 and 47.9.

## The best-fit direction pair is matched and cannot reach its own fit

| View | Best-fit pair | Direction groups | Direction fit | Pair status | Combined in pair | Nearest combined in pair | Passed both masks |
| --- | ---: | --- | ---: | --- | ---: | ---: | --- |
| GX0, midpoint-anchor selection | 23 | [1, 9] | 19.65 | matched | 262,144 | 33.82 | yes |
| Amateur-3, precise-representative selection | 43 | [2, 14] | 4.27 | matched | 262,144 | 46.10 | yes |

On Amateur-3 the three pairs with direction fits under 4.6 px (43, 42, 32) build nothing under 45 px; the pair that comes nearest, pair 30 at 40.1, has a direction fit of 6.9. The direction fit is a homography with the two vanishing points fixed and four free numbers; the axis matching can only reach homographies whose four numbers come from pairs of observed line groups matched to pairs of template markings. The kept part of that discrete set holds no court near the control on these pairs.

## Where the loss now sits

Every combined court is one of 512 by 512 kept matchings. The axis matching enumerates every pair of direction-compatible observed groups against every ordered pair of template markings, keeps the matchings with at least three supported markings that the necessary player rule accepts, orders the distinct matchings by score and keeps the first 512. On the pairs above the per-direction cap excluded most of the distinct matchings:

| View, pair | Horizontal distinct | Horizontal kept | Vertical distinct | Vertical kept |
| --- | ---: | ---: | ---: | ---: |
| GX0, 23 | 2,358 | 512 | 2,022 | 512 |
| Amateur-3, 43 | 23,864 | 512 | 1,748 | 512 |
| Amateur-3, 30 | 26,364 | 512 | 1,766 | 512 |

So the answer to the brief's question is: no court within a few pixels of the control exists before the masks, and no mask removes one. The axis matching, taken as the product of kept matchings, never builds the court. Whether the right matching is enumerated and then cut by the per-direction cap, fails the three-marking support rule or the necessary player rule, or is never enumerable because the observed groups it needs are missing or fall outside the 1.5 degree direction compatibility, is not separated by this recording. It can be separated locally: one axis-matching replay per direction for a handful of pairs, with the direction-fit homography decomposed into its per-direction scale and offset as the target. No fix is proposed here.

**Addendum, 2026-09-16.** The axis-matching replay (`line_identity/axis_replay.py`, `runs/axis_replay/`) separated the layers: on Amateur-3 pair 43 a matching that reaches the fit is enumerated and passes the support and player rules, and with no per-direction cap the pair's matchings combine to a 6.2 px court (pair 30: 8.7 px). The per-direction cap, ordered by the axis score, is what removes it. The statement above that the discrete set holds no close court applies to the kept matchings only.
