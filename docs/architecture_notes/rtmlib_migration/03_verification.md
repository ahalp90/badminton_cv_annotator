# rtmlib migration: verification plan

> **Detector superseded (2026-07-04):** results recorded below were measured
> with rtmdet-nano@320 (at thr 0.3, then 0.15 after the G-4 recalibration); the
> detector is now RTMDet-M@640 at 0.3
> ([07_detector_restoration.md](07_detector_restoration.md)) and G1/G6/G7/G8/G9
> are to be re-run under it (status table in 07). The gate designs and
> thresholds stay.

> Two gate families: **byte-equality** (things that must not change) and
> **parity** (the model changed, so compare, don't byte-match). Revised
> 2026-07-01 after an adversarial gate-review (findings in
> `04_adversarial_review.md`). Gate scripts land under
> `src/bst_x/validation_scripts/rtmlib_migration/` in Batch 0 and import the
> shipped `raw_extract` / adapter, never the scratchpad prototype.

## The core lesson from the review

`sticky_anchor`'s player selection (`_pos`/`_failed`) is bbox-driven, not
keypoint-driven. So the end-to-end parity (failed/pos agreement) validates the
*bbox* path and would pass even if keypoints were swapped/reordered. The
keypoint values are the actual product of this migration, so they need their own
self-certifying CPU value gate (a committed mmpose keypoint reference +
pixel-space L2), not just the parity/joints signal (which is confounded by the
intended model change). This is the #1 gate.

## What is byte-equal-able vs parity

| Property | Kind | Why |
|----------|------|-----|
| Downstream (raw to clean to collate) on FIXED raw inputs, pre vs post branch | byte-equal (precondition) | Downstream code is unchanged; identical inputs must give identical outputs. Catches an accidental downstream edit. NOT a test of the adapter. |
| rtmlib CPU inference, same clip twice, pinned threads | byte-equal | onnxruntime CPU EP is deterministic for fixed input+model+thread-count. |
| rtmlib keypoints vs a committed mmpose keypoint reference | parity, thresholded | Updated weights, so not identical, but a correct adapter is within a few px; a swap/reorder is tens to hundreds px. This is the value gate. |
| Deployed clean: sticky_anchor(rtmlib) vs committed | parity | Tests the bbox-driven selection; necessary but not sufficient (misses keypoint bugs). |

## CPU gates (local host; the user runs them)

### G1: adapter keypoint-VALUE gate (the critical one)
Committed mmpose keypoint reference for a spread of clips across distinct
video-ids (extract the per-person `keypoints`/`bbox` from the committed
`ShuttleSet_keypoints_raw`). Run the shipped adapter on those clips;
IoU-match detections; assert median per-keypoint L2 ≤ 5 px, p90 ≤ 12 px
(body7 vs the old RTMPose-L are the same family, so a correct adapter is within a
few px; an x/y swap or keypoint reorder gives tens to hundreds px). Reference-free
backstop: assert anatomical order on a clearly-standing player (nose above hips,
ankles below knees in image coords) to catch gross reorders without a reference.

A reverted RGB fix is not a px-threshold miss. RTMPose-L body7 is
channel-robust, so a BGR feed lands ~0.1 px (median) from RGB, inside the 5 px
gate (measured BGR 2.60 px vs RGB 2.54 px on 11_1_10_2, both PASS). So G1 catches
it structurally instead: `_rgb_fix_counterfactual` recomputes the pose under
an RGB and a BGR feed on the detector's own boxes and asserts the shipped adapter
equals the RGB feed byte-for-byte (a revert flips it onto the BGR feed) and that
the two feeds differ (so the equality has teeth). The default sample spans ≥5
distinct video-ids, not three clips of one match where colour matters least.

L/R model disagreement vs a systematic L/R bug (mirror-labeling). Widening the
sample surfaced a legitimate confound on `16_1_10_1`: body7 and the old RTMPose-L
assign left/right oppositely on ~6% of frames (ambiguous rotational poses), a
shoulder/hip-width L2 that inflated the *raw* confident p95 to 20.3 px (median
stayed 2.6 px, so the skeleton is correct, just mirror-labeled; verified on the 6
worst frames: correct IoU 0.74 to 0.90, an L/R swap collapses ~29 px to ~1.4 px). This
is model drift the Phase-B retrain absorbs, not an adapter defect. So the p95 is
gated on the per-person minimum of the direct and L/R-swapped distance
(`_confident_tail_lr`), which drops `16_1_10_1` to 14.9 px. A *systematic* adapter
L/R swap can't hide behind that: it needs relabeling on ~every match, tripping
`SWAP_FRAC_MAX` (0.20), and independently the raw median/p90 (a full swap moves
12/17 joints by limb-width). Negative control: a swap-every-frame adapter gives
swap_frac 0.94, median 24.6, p90 112.6, all FAIL.

### G2: adapter contract test (schema + edge cases, synthetic)
Runs the shipped adapter; asserts per-person `{keypoints (n,17,2), bbox
(n,4), bbox_score (n,), keypoint_scores (n,17)}`, COCO-17 count, dtypes, and:
- **Truncation:** inject a synthetic frame with 20 fake boxes of known
  descending `bbox_score`; assert the kept 16 are exactly the top-16 by score,
  in order (the ~0.79%-of-frames path a small clip never exercises).
- **Empty-frame guard:** inject a no-person frame; assert `ndet==0` and the
  whole frame all-NaN (rtmlib's full-image fallback must be blocked).
- **Partial-frame padding:** for `ndet==1`, assert rows `[ndet:]` are all-NaN.

### G3: dtype-parity on the detect_players_2d path
`detect_players_2d` builds `np.array([p["keypoints"] …])`; under mmpose those
were Python lists (float64), but rtmlib returns float32 arrays, so the stack stays
float32, which shifts `normalize_joints`/court-projection at the atol boundary.
Assert the adapter feeds detect_players_2d the same dtype it had under mmpose
(cast keypoints/bbox to float64 in the 2D consumer, or in the adapter for that
path), and gate `smoke_prepare_2d_bit_exact` (rtmlib-vs-rtmlib) on it.

### G4: rtmlib CPU determinism
Run the adapter twice on the same clip with `OMP_NUM_THREADS=1` pinned in the run
command (rtmlib exposes no `intra_op_num_threads` knob); assert `np.array_equal`
on all five raw arrays.

### G5: downstream byte-equality (precondition, dual-invocation)
Fixed committed mmpose raw into unchanged `apply_heuristic`(`current`,
`sticky_anchor`) + `collate_npy`, main writes reference / branch compares;
`_failed` exact, `_pos`/`_joints` atol 1e-5. Proves the branch edited nothing
downstream. (Needs numpy + pandas<3 + torch-cpu.)

### G6: deployed-output parity
`sticky_anchor(rtmlib_raw)` vs the committed clean: failed-frame agreement,
`_pos`/`_joints` median/p90, and the directional failed-frame split. Necessary
(bbox-path) but not sufficient; G1 carries the keypoint values. *(Over 20
diverse/busted clips: dF=0 on all 20, failed-agreement mean 0.98 / min 0.86 on
one hard busted clip, pos median ≤0.012.)*

**One-directional frame-loss bias (measured; adversarial review).** The residual
failed-frame disagreement is almost entirely **rtmlib-fails-where-mmpose-keeps**:
rtmlib's 320-input RTMDet misses a salient player on hard/blur contact frames, so
`sticky_anchor` can't fill a slot and the frame is zeroed, rarely the reverse.
The worst hard clip drops ~14% of frames (all data-loss), concentrated on the
hardest strokes (smash/net at contact), so it can bias per-class frame counts.
G6/G8 now report `rtLoss`/`mmLoss` per clip and G9 the signed failed-rate delta,
so the bias cannot hide behind the mean. **This is model behavior, not a bug.**
The Phase-A decision (see 06_phase_a_decision.md) resolved this: GO at 0.15. The
mitigation was the 0.15 keep-threshold (post-inference filter, model unchanged),
not a 640 detector (considered and rejected as a different ONNX).
The earlier "no clip dropped a player mmpose kept" was an overclaim from the
easy 8-clip smoke50 sample, corrected here.

## GPU gates (Bourbaki; the user runs them; never self-certified)

### G7: CUDA self-variance floor (run first)
Run the shipped `raw_extract` twice on smoke50 on the GPU; report max
per-clip failed-rate delta `ε_fail` and median keypoint L2 `ε_kp`. Every
mmpose-vs-rtmlib threshold below must sit above these. Also a
rtmlib-CPU-vs-CUDA agreement check on a few clips to bound EP divergence.

### G8: extraction parity at scale
rtmlib `raw_extract` (CUDA) on smoke50 + shards vs committed mmpose raw:
per-clip frame count, `ndet` distribution vs the captured baseline, IoU-matched
keypoint L2.

### G9: Phase-A decision gate (concrete thresholds; replaces open O3)
Apply `sticky_anchor` to the new rtmlib raw and compare to the committed clean:
- **Frame count:** 100% of clips `F_rtmlib == F_mmpose == video frame count`; any mismatch = hard fail.
- **Deployed failed-rate:** `|Δ| ≤ 2 pp` aggregate, `≤ 5 pp` per clip; per-frame failed-agreement `≥ 95%` aggregate, all above `ε_fail`.
- **Keypoint agreement (the real signal):** IoU>0.5 high-conf matched dets, median L2 ≤ 5 px, p90 ≤ 12 px, above `ε_kp`.
- **`joints` normalised delta:** report only (model-drift confounded).
- **Sample:** stratified to cover every distinct video-id in `res_df` (court homography is per-video) and all classes; ≥ ~10k frames (smoke50 + shard_00/01 + a per-video top-up).

### G10: Phase-B (if A clears)
Full 30,487-stem re-extract, then `sticky_anchor`, collate, retrain BST-X, then
macro/min-F1 vs the committed baseline.

## Reproducibility (was "print + pin"; now assert)

- `download_and_verify_models.py` asserts SHA256 against a committed constant
  and exits non-zero on mismatch; every extraction gate calls it first.
- **Vendor the two ONNX to the pool** so Phase B never re-fetches from openmmlab
  (repo `main` is newer than PyPI 0.0.15, a silent drift risk).
- **Pin exact versions:** `rtmlib==`, `onnxruntime==` / `onnxruntime-gpu==`.
- **Pin CPU threads via `OMP_NUM_THREADS`** in the determinism-sensitive gate run
  commands (G4/G5): rtmlib's `BaseTool` builds the onnxruntime session with no
  `SessionOptions`, so `intra_op_num_threads` is not a constructor knob. G4 proved
  the env pin gives bit-identical CPU output. Phase B runs on CUDA, which is
  nondeterministic regardless, so no thread pin makes it reproducible there.

## Broken-adapter coverage (post-fix)

| Broken adapter | Caught by |
|----------------|-----------|
| x/y swap, keypoint reorder | **G1** (px L2) + order-sanity backstop |
| systematic L/R swap | **G1** `SWAP_FRAC_MAX` (≈all matches relabel) + median/p90 (12/17 joints move limb-width); legit ~6% per-frame L/R model drift does *not* trip it |
| reverted RGB fix | **G1** `_rgb_fix_counterfactual` (byte-exact: shipped == RGB feed, RGB ≠ BGR); *not* the px L2, which is channel-robust (~0.1px) |
| bbox coords reordered | G1 + G6 (pos blows up) |
| truncation key / off-by-one | **G2** synthetic 20-box frame |
| empty-frame leak | **G2** synthetic no-person frame |
| dtype drift | G2 + **G3** |
| model/version drift | SHA-assert + version pins |

## Dual-invocation note

On-disk references rot when paths flip or dirs get cleaned. Every main-vs-branch
gate assumes the dual-invocation pattern (main writes the reference into scratch,
branch reads it), never a committed reference dir.
