# rtmlib migration: Phase-A decision (authoritative)

> The record of executing `03_verification.md`'s gate ladder via the
> `05_gpu_handoff.md` loop, and the decision it supports. This is the
> authoritative call over the stratified all-courts sample. It supersedes the
> earlier smoke50-only provisional note.

## Verdict

Phase-A: GO. Ship the migration with one shipped recalibration and one documented
residual.

- Shipped change: the detector keep-filter `DET_SCORE_THR` moves from `0.3` to `0.15`.
  This is a post-inference filter on the identical RTMDet ONNX's scores, not a model
  change (the model, the SHA-pinned weights, and the 320x320 input are all unchanged).
- Documented residual: one hard motion-blur clip (`2_1_10_2`) still exceeds the per-clip
  frame-loss ceiling. At 0.15 the migration otherwise matches or beats the mmpose baseline
  on the Phase-A parity metrics on 199/200 in-scope clips (downstream F1 is Phase B).

## How we got here

1. smoke50 (video 11, 0.3): GO (provisional). Every regression metric passed; the only
   failures were the report-only `jntMed` drift (see the `jntMed` gate inconsistency
   below). But smoke50 is one easy court, so it could not settle the frame-loss question
   `03_verification.md:100-111` flagged.
2. G-4 authoritative (200 clips, all 40 extractable courts, 0.3): NO-GO, a real frame-loss
   bias. Not the `jntMed` leak this time. The 320-input detector dropped players on hard
   clips (below).
3. Diagnosis: the drops are under-scored players, not recall failures, so the fix is the
   keep-filter, not the model.
4. G-4 re-run at 0.15, the bias resolves. Dropped players cleared, the directional loss
   balanced out, and aggregate parity went to ~0: GO.

## G-4 at the then-shipped 0.3: the frame-loss finding

NO-GO. Aggregate metrics were healthy (`kp_med 1.74px`, `kp_p90 4.58px`, agg failed-rate
Δ `0.48pp`, fmatch `0.994`) but the tail failed: 6 real per-clip fails, all the
320-detector frame-loss bias:

| clip | reason | frame-loss Δ |
|------|--------|--------------|
| `4_1_10_2/3/4/5` | dropped player (all 4 of video 4's sampled clips) | 1.7-10.3pp |
| `16_1_10_6` | dropped player | 5.1pp |
| `2_1_10_2` | fmatch 0.812, 6 frames zeroed | 18.75pp |

Directional split: 50 rtmlib-only-fails vs 7 mmpose-only, the one-directional 320-detector
bias, manifest at scale. (`5_1_10_1` also tripped a keypoint bound; it is benign, see
residuals.)

## Diagnosis: under-scored players, not recall loss

`diag_g4_fails.py` re-detected the dropped frames at a low threshold:

- No player is ever genuinely undetected at 320. All 36 detections that are truly absent
  at 320 are crowd (detector slots mm[2]-mm[15]); zero are players (mm[0]/mm[1]).
- The dropped player-slots are under-scored players: 68 events, scoring 0.10-0.30, median
  0.18 (mmpose ran RTMDet at a larger test size and scored the same players above its ~0.3
  cut; the fixed 320x320 ONNX scores them lower).

So the frame loss is the `0.3` keep-filter discarding players the detector did find. The
fix is the filter.

Why not the 640 detector (`02_adapter_design.md` option b): the RTMDet ONNX is fixed at
320x320, so "640" means swapping to a different detector (YOLOX-HumanArt), a real model
change, out of scope. The under-scored-player finding makes it unnecessary anyway.

## G-4 at 0.15: the fix

| signal | 0.3 | 0.15 |
|--------|-----|------|
| dropped-player clips | 5 | 0 |
| directional loss (rt-only : mm-only) | 50 : 7 | 15 : 20 |
| aggregate failed-rate Δ | 0.48pp | 0.01pp |
| per-frame agreement (fmatch) | 0.994 | 0.997 |
| keypoint median / p90 | 1.74 / 4.58px | 1.93 / 4.21px |
| genuine rtmlib-worse per-clip fails | 6 | 1 |

The under-scored players were recovered, the top-16 truncation did not evict them, and the
one-directional bias is gone (rtmlib now loses slightly fewer frames than mmpose). The
extra crowd 0.15 admits is geometry-rejected by `sticky_anchor`, as designed.

## Residuals at 0.15 (why the gate still prints NO-GO)

The mechanical G9 NO-GO at 0.15 is driven by artifacts plus one genuine clip:

- `2_1_10_2`, the one genuine regression: 15.62pp, 5 frames lost, both players detected
  (rt<2=0), likely frames where a player scored below 0.15 (23 of the 68 under-scores were
  in [0.10,0.15)). A single hard motion-blur clip out of 200, accepted as a documented
  residual; chasing it to 0.10 drags in more crowd for marginal gain on one clip.
- `33_1_12_2` (16.67pp) and `43_1_10_2` (10.91pp): rtmlib is better than mmpose here
  (mmpose itself dropped 10-11 frames, `mm<2`), flagged only because the per-clip gate is
  symmetric. Mechanism and fix in the gate-metric gaps table below.
- `5_1_10_1`: benign L/R model drift (`kp_p90 35.6px` but `jntMed 0.013`; 34.7% of its
  people improve under an L/R swap), the same gate-metric gap shape G1 documents on
  `16_1_10_1`. Mechanism and fix in the same gaps table.
- 11 clips: the `jntMed` leak (below).

Reading past those, rtmlib at 0.15 matches or beats mmpose on the parity metrics on
199/200 in-scope clips.

## The `jntMed` gate inconsistency (unchanged from the provisional note)

G9's policy designates the joints delta report-only and model-drift-confounded
(`phase_a_decision.py:20`, `03_verification.md`) and implements that for its own aggregate
(`phase_a_decision.py:143-145`), but `g8_ok` (`:109-110,155`) re-imports G8's per-clip
verdict, which hard-gates `jntMed` in G8's `_verdict` (`gate_gpu_parity.py`). So benign
body7 pose drift (keypoint median ~1.4px, L/R-explained) reaches the mechanical NO-GO
through a back-door that contradicts the decision layer's stated policy; we leave the gate
code unchanged and record the exception here.

## Gate-metric gaps (deferred fixes)

Three metrics flag non-defects at 0.15. All are left as-is to keep this branch scoped to
the migration; each is a small, self-contained follow-up.

| gap | effect | suggested fix | needs fixing? |
|-----|--------|---------------|---------------|
| `jntMed` reaches the decision via `g8_ok` (above) | benign body7 drift feeds the mechanical NO-GO | drop `jnt_med` from G8's `_verdict`, matching G9's report-only policy | only for a mechanically-green ladder; the decision already reads past it |
| Per-clip failed-rate is symmetric (`phase_a_decision.py` gates `abs(rt-mm)`) | flags clips where rtmlib beats mmpose (`33_1_12_2`, `43_1_10_2`), where mmpose itself dropped players | gate the directional excess `max(0, rt-mm)`; the signed split is already computed alongside | recommended before the next authoritative run; it overstates the failure count |
| G8's confident-p90 lacks G1's L/R correction | benign body7 L/R relabeling inflates the tail (`5_1_10_1`: 35px p90 vs 1.3px joint agreement) | reuse G1's `_confident_tail_lr` (per-person min of direct and L/R-swapped distance) for `kp_cp90` | low, G1 already applies it and passes the keypoint values |

None blocks the migration: the affected clips are benign or rtmlib-better, and the GO
rests on the aggregates and the directional split, which are already correct.

## Coverage

`40/44` video-ids, effectively full. The 4 absent ids (9, 10, 12, 27) have zero clips in
the 30,487-stem extraction manifest (`stems_to_extract.txt`); they contribute nothing to
extract. The sample covers 40/40 videos actually in scope.

## Reproduce

```bash
source gpu_env.sh
# Build the stratified sample (5 clips per extractable video-id):
PYTHONPATH=src/bst_x:src ~/venv-rtmlib-gpu/bin/python \
  src/bst_x/validation_scripts/rtmlib_migration/make_phase_a_sample.py \
  --per-video 5 --out phase_a_stems.txt
# G8 at the shipped 0.15 (DET_THR now defaults to the shipped DET_SCORE_THR):
PYTHONUNBUFFERED=1 PYTHONPATH=src/bst_x:src RTMLIB_GATE_DEVICE=cuda \
  RTMLIB_GATE_STEMFILE=phase_a_stems.txt RTMLIB_GATE_G8_JSON=g8_parity_full.json \
  ~/venv-rtmlib-gpu/bin/python \
  src/bst_x/validation_scripts/rtmlib_migration/gate_gpu_parity.py 2>&1 | tee g8_full.out
PYTHONPATH=src/bst_x:src RTMLIB_GATE_G7_JSON=g7_selfvariance.json \
  ~/venv-rtmlib-gpu/bin/python \
  src/bst_x/validation_scripts/rtmlib_migration/phase_a_decision.py \
  g8_parity_full.json 2>&1 | tee g9_full.out
```

`classify_phase_a_fails.py g8_parity_full.json` splits the per-clip fails into
real vs `jntMed`-only.

## Phase B (next)

Full 30,487-stem re-extract at 0.15 (the mmpose baseline covered 32,203 clips), then
`sticky_anchor`, collate, retrain BST-X, and macro/min-F1 vs the committed baseline
(`G10`). The retrain also absorbs the residual body7 model drift and measures whether
`2_1_10_2`-style hard-clip loss touches the smash/net class F1.
