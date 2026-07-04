# rtmlib adapter: design + pre-analysis

> **Superseded on the detector (2026-07-04):** the "hash-identical" premise
> below is false — mmpose ran RTMDet-M @640, restored in
> [07_detector_restoration.md](07_detector_restoration.md). The adapter design
> itself (score recovery, RGB fix, contract) is unchanged and current.

> The single new module both `raw_extract` and `detect_players_2d` depend on.
> Design first, then the invariants it must preserve, then the empirical
> validation from the 2026-07-01 CPU prototype spike.

## The contract to reproduce (mmpose per-person output)

mmpose `MMPoseInferencer("human")` yields, per frame, `result["predictions"][0]`
= a list of person dicts. Consumers read:

| Field | Shape | Used by |
|-------|-------|---------|
| `keypoints` | `(17, 2)` pixel coords, COCO-17 order | raw_extract, detect_players_2d |
| `bbox` then `bbox[0]` | `(4,)` xyxy | raw_extract, detect_players_2d |
| `bbox_score` | scalar | raw_extract (to `_raw_scores`, consumed by sticky_anchor) |
| `keypoint_scores` | `(17,)` | raw_extract (to `_raw_kp_scores`, consumed by no heuristic) |

The adapter must return the same per-person quadruple so both consumers stay
byte-identical in their assembly logic.

## Design

`src/bst_x/preparing_data/rtmlib_pose.py` (new). Drives rtmlib's low-level
classes (not the `Body` solution, which returns only keypoints+scores):

1. **`RTMDetScored(rtmlib.RTMDet)`**: the person ONNX has NMS baked in
   (`outputs[0]` = `dets (1,N,5)`). rtmlib's `postprocess` computes the per-box
   score then discards it; the subclass overrides `__call__` (the NMS-baked
   `(...,5)` branch, recovering the score column) to return
   `(boxes (M,4) f32, scores (M,) f32)` for boxes with `score > self.score_thr`
   (shipped `DET_SCORE_THR = 0.15`, recalibrated from 0.3; see 06_phase_a_decision.md).
   Detector fed BGR (its mean is BGR-order, correct as-is).
2. **RGB-fixed `RTMPose`**: rtmlib's RTMPose uses an RGB-order mean but never
   converts BGR to RGB (undocumented bug). The adapter feeds it
   `cv2.cvtColor(frame, BGR2RGB)` so the colour order matches training
   (warpAffine is colour-agnostic, so convert-then-warp ≡ warp-then-convert).
   Returns `(keypoints (M,17,2), keypoint_scores (M,17))`.
3. **`extract_frame(frame_bgr) -> (kps, bboxes, det_scores, kp_scores, n)`**:
   `boxes, det_scores = det(frame)`; if `n==0` return the all-NaN, `n=0` frame
   (guarding rtmlib's full-image fallback); else `kpts, ksc = pose(rgb, boxes)`;
   assemble into `N_max`-padded arrays with the top-`N_max`-by-`det_score`
   truncation. This is `raw_extract.extract_raw_frame` sourced from rtmlib.

Models: detector `rtmdet_nano...05d8511e` @ 320x320 (recorded then as
"hash-identical to mmpose's" — false; corrected 2026-07-04, RTMDet-M `235e8209`
@640 restored, see 07);
pose `rtmpose-l_simcc-body7...256x192` @ `model_input_size=(192,256)`. Device
`cpu`|`cuda` via rtmlib's `device=`. Module hard-imports rtmlib; consumers import
it lazily (inside the functions) so `prepare_train_on_shuttleset` keeps
importing without the dep.

## Invariants preserved (what a naive swap would break)

1. **Schema exact:** 5 arrays, `float32` (kps/bboxes/scores/kp_scores) + `int8`
   (ndet), NaN padding (not zero, origin is a valid coord), COCO-17 order.
2. **`N_max=16` top-by-`bbox_score` truncation** only when `n>N_max`; otherwise
   preserve detection order (matches `extract_raw_frame:61-69`).
3. **Empty-frame guard:** zero detections give `n=0`, all-NaN frame. rtmlib's
   RTMPose would otherwise fabricate a full-image "person", which must be blocked.
4. **`_raw_ndet` saved last** = resume marker (unchanged in raw_extract).
5. **`_raw_kp_scores` still written** though no heuristic reads it (keeps
   `apply_heuristic._load_raw_clip` and `RawClip` intact).
6. **detect_players_2d side (as-built):** the loop now reads `det.keypoints` /
   `det.bboxes` from the adapter, cast to float64, so `_order_two_on_court` /
   `normalize_joints` / court projection keep the precision they had under
   mmpose's `np.array`-of-lists (rtmlib returns float32; the cast is gated by
   **G3**, verified decisive by a float64-vs-float32 negative control). The
   `_order_two_on_court` helper, the `<2`/`!=2` guards, the strict-`>` y-flip,
   the `dtype=float` (float64) zero-appends on failed frames, and
   `normalize_joints(center_align)` are untouched (OUT-list §7). NOTE the
   plan's first sketch ("keep detect_players_2d unchanged, feed it an
   inferencer-shaped adapter") was dropped: the consumer indexes
   `person["bbox"][0]`, so a flat-array shim would `IndexError` in
   `normalize_joints`. Reading the adapter arrays directly avoids that IndexError and is simpler. Detection *order* is not reproduced (rtmlib NMS order ≠ RTMDet
   order); on exact coordinate ties this can flip a Top/Bottom or slot pick,
   within the parity (not byte-identity) envelope and visible only as the
   residual failed-frame disagreement in `03_verification.md`.
7. **No torch:** drop `torch.cuda.empty_cache()` (onnxruntime-managed); keep
   `gc.collect()`.

## Determinism

onnxruntime CPU EP is deterministic run-to-run while CUDA EP is not; the
thread-pinning details (`OMP_NUM_THREADS=1`, G4, no `SessionOptions` knob) live in
`03_verification.md`.

## Prototype validation (2026-07-01, CPU, clip 11_1_10_2)

Ran the design above (low-level `RTMDetScored` + RGB-`RTMPose`) against the
committed mmpose raw arrays. Result:

- **Schema/dtype: exact match**: `kps (62,16,17,2) f32`, `bb (62,16,4) f32`,
  `sc (62,16) f32`, `ksc (62,16,17) f32`, `ndet int8`.
- **Frame count: exact** (62 = 62); cv2 decode matched mmpose's decoder.
- **Detection-score recovery: works** (243 scores, [0.30, 0.85]).
- **Keypoint agreement: median 2.54 px** over 176 IoU-matched detections
  (mean/max inflated by a few occluded/mismatched boxes). The body7 RTMPose-L
  swap tracks the old extract to a couple of pixels on matched people.
- **RGB fix applied** (kp-conf 0.594 RGB vs 0.587 BGR vs 0.542 mmpose).
- **Empty-frame guard: in place** (no zero-det frames on this clip).

### The `ndet` difference (analysed, not a bug)

rtmlib detects fewer people/frame (mean 3.92, max 6) than mmpose (mean 6.63,
max 10). Not a threshold difference: mmpose's min `bbox_score` is 0.301, so
both cut at ~0.3. The cause is detector input resolution: mmpose ran the
RTMDet `.pth` at a larger test size and found more small/distant crowd boxes;
rtmlib's person ONNX is fixed at 320x320. The extra boxes are low-salience
background people (crowd, line judges) that `sticky_anchor` rejects by geometry;
rtmlib always found ≥2 (the two salient players). Whether this perturbs the
deployed 2-player output is settled by the end-to-end parity test below and
the Phase-A GPU gate, not assumed.

Options if end-to-end parity is insufficient: (a) accept (premised then on a
"hash-identical detector" — the premise the 2026-07-04 correction overturned;
cleaner detection set); (b) swap to a 640-input YOLOX-HumanArt
detector to match density (different detector, more boxes). Resolution: (b) was
considered and REJECTED, since 640 means a different ONNX (YOLOX-HumanArt),
off-limits under "don't change the model." The frame-loss motivating it was
resolved by lowering the keep-threshold to 0.15 (post-inference filter, model
unchanged). Option (a) chosen; see 06_phase_a_decision.md.

### End-to-end parity: sticky_anchor(rtmlib_raw) vs deployed clean

`end_to_end_parity.py` over 8 smoke50 clips: ran the repo's unchanged
`sticky_anchor` on rtmlib raw and compared to the committed
`ShuttleSet_keypoints_clean_sticky_anchor` (= sticky_anchor over mmpose raw):

| metric | result |
|--------|--------|
| mean failed-frame agreement | 0.988 (7/8 clips exact; one at 0.903 = 6/62 boundary frames) |
| pos median delta (normalised court coords ~[0,1]) | 0.0081 |
| joint median delta (bbox-diagonal-normalised) | 0.0227 |

This resolves the `ndet` concern on these clips; the authoritative 200-clip run
(see 06_phase_a_decision.md) re-tested it at scale and required the 0.3-to-0.15
keep-threshold recalibration. Despite rtmlib returning a smaller, cleaner
detection set, `sticky_anchor` selects the same two players on ~99% of frames,
and where both keep a frame the positions/joints agree to <1% of the court / ~2%
of the bbox diagonal. The extra mmpose crowd detections were noise `sticky_anchor`
rejected; rtmlib's set already contains the two salient players. So
`rtmdet-nano @ 320` is likely sufficient pending the at-scale gate; the residual
is model-noise Phase B will confirm at F1 level. (The authoritative run confirmed
320 with a 0.3-to-0.15 keep-threshold change; see 06_phase_a_decision.md.)

Harder-clip confirmation (20 clips: 14 diverse across matches/classes/lighting
+ 6 known-hard busted). frame-count `dF=0` on all 20; no clip dropped a player
mmpose kept (true for these 20; the authoritative 200-clip run at 0.3 did surface
dropped players, resolved at 0.15; see 06_phase_a_decision.md); failed-agreement
mean 0.981 (min 0.857 on a hard busted clip);
pos median 0.0067; joint median 0.0197. Representativeness holds across smash /
net / drive / rush / Xnet, men/women, and dark footage. See
`04_adversarial_review.md` Lens A. Caveat (from the gate review): this parity is
bbox-driven, so the keypoint *values* are gated separately by
`03_verification.md` G1 (committed-reference pixel L2), not by this table.
