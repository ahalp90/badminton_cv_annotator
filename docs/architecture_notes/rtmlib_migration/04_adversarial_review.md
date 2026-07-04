# rtmlib migration: adversarial plan-review (round 1)

> 2026-07-01. Three independent refute-biased reviewers, one lens each. Net:
> the migration approach is sound; the verification plan had real holes the
> gate reviewer caught, now folded into `03_verification.md`. No `src/` edited.

## Lens A: deployed-output parity, refutation FAILED (claim CONFIRMED)

Attack: the 8-clip smoke50 sample is unrepresentative; cv2 might decode a
different frame count than mmpose; rtmlib's smaller detection set might drop a
player on hard clips.

Evidence (ran the repo's unchanged `sticky_anchor` on rtmlib raw vs the
committed clean over 20 clips: 14 diverse (HSBC/Fuzhou/AllEngland/Thailand,
smash/net/drive/rush/Xnet, men+women, dark footage) plus 6 known-hard "busted"):

| metric | result |
|--------|--------|
| frame-count mismatches (`dF≠0`) | **0 / 20** |
| clips where rtmlib dropped a player mmpose kept | **0 / 20** (`rt<2 ≤ mm<2` always; Round-1 sample, later revised at G-4, see `06_phase_a_decision.md`) |
| mean failed-frame agreement | **0.981** (min 0.857 on busted `11_2_21_18`) |
| mean per-clip pos median | 0.0067 (max 0.0117) |
| mean per-clip joint median | 0.0197 (max 0.0329) |
| clips < 0.95 agreement | 3, all hard (`25_1_12_5`, `11_2_7_14`, `11_2_21_18`) |

Verdict: the deployed output is preserved across diverse and hard clips (Round-1
20-clip sample; the stratified G-4 run later found dropped-player clips at 0.3,
resolved at 0.15; see `06_phase_a_decision.md`). The sub-0.95 clips are marginal
on-court-boundary cases in the already-"irrecoverable" busted set; the residual is
model-noise Phase-B (retrain) will confirm at F1.

## Lens C: gate/verification soundness, VERDICT: REFUTED (holes found)

The ladder caught schema/dtype and accidental downstream edits, but not value
regressions in the adapter, because `sticky_anchor` selection is bbox-driven,
a keypoint bug (x/y swap, reorder, reverted RGB fix) would pass the parity I was
leaning on. Findings and their folds:

| # | Sev | Finding | Fold |
|---|-----|---------|------|
| H1 | HIGH | No self-certifying keypoint-VALUE gate | **G1**: committed mmpose kp ref + IoU L2 ≤5px/p90≤12px + order-sanity backstop |
| H2 | HIGH | Gates certified the prototype, not shipped `raw_extract`; truncation + empty-frame paths untested | Gate scripts import the shipped `raw_extract`; **G2** injects synthetic 20-box + no-person + partial frames |
| H3 | HIGH | Repro unpinned (SHA "print", onnxruntime/model drift) | SHA assert+exit, vendor ONNX to pool, pin `onnxruntime==`; thread pin refined as-built to `OMP_NUM_THREADS=1` in the gate run command (rtmlib exposes no `SessionOptions`/`intra_op_num_threads`) |
| M1 | MED | Phase-A thresholds (O3) hand-wave | **G9** concrete thresholds, set above the **G7** CUDA self-variance floor |
| M2 | MED | `detect_players_2d` unverified (smoke neutralised by model change) | **G3** dtype-parity + rtmlib-vs-rtmlib smoke; mitigated (dormant path) |
| M3 | MED | Device blind spot (local CPU-EP vs prod CUDA-EP) | **G7** CUDA self-variance + CPU-vs-CUDA agreement |
| L2 | LOW | RGB fix validated via kp_score mean (unused slot), wrong signal | Validate via L2: `L2_rgb ≪ L2_bgr` AND `L2_rgb ≤ 5px`, ≥20 clips/≥5 video-ids |
| L3 | LOW | Det threshold 0.3 vs `score_filter` 0.2 | Report in G8 ndet parity; revisit if ndet parity flags it. Actioned: resolved by the 0.15 recalibration (see `06_phase_a_decision.md`) |

## Lens B: adapter contract / scope / OUT-list, report lost, lens covered

The reviewer's transcript did not return (concurrent-git churn in the shared
tree; see `collab-workflow-constraints`). Its central concern, float32 vs
float64 dtype drift on the `detect_players_2d` path (mmpose lists give float64 vs
rtmlib float32 arrays, shifting `normalize_joints`/projection at the atol
boundary), was independently raised by Lens C and is folded as **G3**.
Remaining B-lens points, checked directly:
- **Detection order:** `current`/`sticky_anchor` decide on-court membership by
  geometry and Top/Bottom by a strict-`>` y-flip, so detection index order does
  not change the selection except on an exact y-tie (rare), low risk.
- **bbox format:** rtmlib returns xyxy in original pixels = mmpose
  `person["bbox"][0]`; the 2.54px prototype agreement confirms it.
- **3D quarantine / requirements split:** verified in scope analysis; no test
  imports mmpose, so moving the pins to `requirements-legacy-3d.txt` is safe.
The adapter contract gets a dedicated per-batch adversarial review at Batch 1.

## Net

Approach validated (A). Verification plan materially strengthened (C, folded into
`03_verification.md`). One real code-level fix to carry into execution: the **G3
dtype-parity** on the `detect_players_2d` path. Executed; Phase-A GO at 0.15 (see
`06_phase_a_decision.md`).
