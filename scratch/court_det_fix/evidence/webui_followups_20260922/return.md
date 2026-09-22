# Follow-ups 1 → 2 — completed

**Task 1: ranking lead for the frozen GX86088 pool/view. Task 2: sparse shared paint helps, but even three samples retain a wall-selection failure.** No detector, new image scoring, proposal generation, candidate refitting, tracking, or repository write was performed. Annotation-plane fitting below is a diagnostic, not a modified candidate.

## Reproduce and verify

Python 3.10+, NumPy and Pillow; tested here with NumPy 2.3.5 and Pillow 12.3.0. One numerical CPU thread. From a directory containing both archives:

```bash
tar -xzf followups12_inputs.tar.gz
tar -xzf followups12_completed.tar.gz
python followups12_results/run_followups12.py \
  --inputs followups12_inputs --output followups12_rerun --task all --figures
```

`--task 1` and `--task 2` also run separately. Figure generation needs task-1 outputs. The supplied bundle is labelled revision `b47eee0ff43ec7cbebcc2d84eb9e5e98075d4eea`; the revision was not independently re-fetched. Inputs retain repository-relative paths:

- `scratch/court_det_fix/evidence/pixel_temporal/evaluation_20260922/results/{gx,am3}/`: `candidates.json.gz`, `manifest.json`, `selection.json`, and the seven GX/two Am3 `scores/*.json.gz` files.
- `data/amateur_court_corners/2026-09-09/{hand_corners_landmarks,hand_corners}.csv`; original `data/amateur_court_corners/hand_corners_landmarks.csv`; `scratch/court_det_fix/frozen_views/packs/gx_extension_inputs.json.gz`; and the bundled native frames/review sheets.
- Semantics were read from `evaluation_20260922/run_temporal_union.py::{working_alignment,eligible_ids,frame_winners,temporal_winners}` and `next_steps_20260916/L3_temporal/run_l3_temporal.py::choose_winners`. Neither producer was imported or executed.

A clean rerun reproduced all 13 data/figure files byte-for-byte. Checks passed: all 31 declared file sizes; 4,608 qualified occurrences and 27,136 score records with matching identities; 78 exact published count/winner-ID checks, including G0/G1 splits; 66 observed-access-only selection replays; and synthetic checks for ties, empty pools, projection scales, symmetry and polygon/DLT calculations. GX common eligibility is **278/3,584**, Am3 **204/1,024**. GX native eligible counts in frame order are **34, 12, 46, 152, 31, 7, 4**. See `results/checks.json.gz`.

## 1. Visible-fit measurements and existing alternatives

All errors in the tables below are **working pixels at 960×540**. Every supplied native frame independently verifies as 1920×1080, so native errors are exactly twice these values. CSVs contain both units.

`S_f = diag(960/native_width_f, 540/native_height_f, 1)`; `A_work = S_anchor @ A_native @ inv(S_target)`; `H_target_native = inv(S_target) @ inv(A_work) @ H_anchor_working`. An independent native-coordinate expression agreed exactly on the tested projections. Applying native registration directly to working coordinates would move tested landmarks by up to **11.09 native pixels**. Figures 01/03 agree visually with the archived GX86088/GX0 selections; this is not a pixel-identical overlay-renderer comparison.

Match clicks by video/frame/court metres. Choose one whole-court symmetry (identity, 180°, x reflection or y reflection) by total squared origin-click error **after selection**, and hold it fixed across targets. No per-landmark reassignment; named courts/alternatives use identity. Am3’s unique video matching both frame numbers is `C6NrJyBwn6c.mkv`; raw frames and projected clicks were inspected.

Regions follow annotation orientation: far backcourt `0 ≤ y ≤ 0.76 m`; remaining far half `0.76 < y < 6.7 m`; near half `6.7 ≤ y ≤ 13.4 m`. GX86088 has **5 / 5 / 9** clicked points respectively. All 21 frame/policy rows retain regional counts, maxima and medians. The landmark CSV is the sole numerical truth. The corner CSV has 13 clicked rows and 15 extrapolated rows; four clicked-row coordinates differ from the landmark file by 0.125–1.008 native pixels. Those sources were not merged or silently reconciled, and no extrapolated corner was counted.

### GX86088 fixed-candidate comparison

`GX0` abbreviates `gxBQ_window_00_frame_0`; `GX86088` abbreviates `gxBQ_window_04_frame_86088`. Complete IDs are in every data output.

| Source / arm / candidate | All-click max / median | Far-backcourt max / median | Model strip missing from full court |
|---|---:|---:|---:|
| `GX0 / G1 / 143:158` | 16.35 / 9.38 | 11.53 / 9.56 | 53.9% |
| `GX86088 / G1 / 16:44` | 9.50 / 5.35 | 9.50 / 6.98 | 94.7% |
| `GX0 / G0 / 22:4580` | 6.65 / 4.81 | 6.22 / 5.17 | 2.0% |
| `GX0 / G1 / 22:137` | 7.57 / 4.98 | 6.94 / 5.63 | 2.0% |
| `GX0 / G0 / 22:4588` | 7.95 / 4.15 | 5.07 / 3.95 | 2.0% |

**Lower aggregate landmark error does not imply less far-end clipping.** GX86088 G1 `16:44` has smaller landmark errors than GX0 G1 `143:158`, yet excludes much more of the far strip. This supports the user’s fallback distinction without making either a clean fit.

The strip is `[0,6.1] × [0,0.76] m`. Missing fraction = reference-strip area outside the candidate’s **full court footprint**, divided by reference-strip area, measured in the annotation-derived plane after clipping to the image rectangle. The reference uses normalized DLT on 19 clicks: RMS **2.834**, median **1.892**, maximum **5.346 native pixels**; non-null condition number **5.54**. It is a fitted planar model, not exact physical ground truth. No person/net occlusion mask was supplied. “Viewport” means image bounds; a second output additionally clips to the convex hull of clicked court coordinates, an interpolation-support footprint, not an exact visibility mask.

For `143:158` versus `16:44`, viewport losses are **53.9% versus 94.7%**; clicked-support losses are **50.3% versus 99.3%**. Leave-one-click-out reference refits give viewport ranges **51.7–62.0% versus 93.5–99.1%**, and clicked-support ranges **48.4–54.7% versus 98.8–100%**. The support polygon is held fixed in that check. These are sensitivity ranges, not confidence intervals; the far strip is thin enough that small reference errors matter.

The oracle scans all **278** occurrences by clicked maximum, median, then ID. The three alternatives are separated by >1 working pixel maximum projected-landmark distance (display diversity only; distances recorded). All have smaller global maxima and less footprint clipping than the named winners. Their full/far-end overlays were inspected: they follow the same court more closely but retain stripe offsets. Footprint coverage is not baseline/service-band alignment. None is promoted to a clean or independently approved court.

**Verdict: ranking lead**, scoped to this pool/view, not “geometry solved.” Next change: add this post-selection helper to the wider-sample report; keep proposals fixed for the first ranking comparison and retain these attainable-fit witnesses. Report fields: source-qualified ID and eligibility; per-region clicked count/max/median; strip-loss model/support and fit residual when available; a separate visual ruling. Do not use the oracle scan as the selector. No numerical court-acceptance threshold was tuned or validated.

## 2. Sparse proposal/decision access

Enumerated **7 singletons, 21 pairs, 35 triples** on GX, plus two singletons and the two-frame Am3 set. A k-frame subset admits only its `512k` source occurrences and requires `status == "ok"` on those k frames, not on the unseen frames. Per-frame paint and shared median paint use the same pool. Paint ties use descending line score/median line score, then ascending full ID; shared line uses descending median line score then ascending full ID. Evaluate each observed-frame choice separately; never retrospectively choose the best one.

All **290** policy decisions were serialized to `task2_locks.json.gz` before task-2 annotation loading/evaluation. The selector receives only observed scores; the separate published-regression check’s 278-ID mask is never a sparse-selection input. Outputs preserve selected/eligible IDs, origin counts and statuses: **1,980** decision/target rows. There are **no empty pools, no selected-court unseen eligibility failures, and no selected-court projection failures**. Large residuals below therefore remain distinct from eligibility/projection failure.

For each decision, take its maximum clicked error on each unseen target and then the **worst across unseen targets**. The table summarizes that scalar across decisions. Shared policies have one decision per subset; per-frame paint has k decisions per subset. Quantiles use NumPy linear interpolation; these overlapping subsets are not independent statistical trials.

| k | Policy | Decisions | Median worst-unseen | P90 | Worst |
|---:|---|---:|---:|---:|---:|
| 1 | per frame paint | 7 | 13.37 | 486.06 | 527.21 |
| 1 | shared paint | 7 | 13.37 | 486.06 | 527.21 |
| 1 | shared line | 7 | 368.40 | 482.83 | 515.02 |
| 2 | per frame paint | 42 | 10.28 | 325.94 | 458.62 |
| 2 | shared paint | 21 | 11.08 | 16.48 | 458.62 |
| 2 | shared line | 21 | 368.40 | 763.85 | 766.30 |
| 3 | per frame paint | 105 | 10.28 | 16.48 | 458.62 |
| 3 | shared paint | 35 | 10.28 | 16.45 | 373.75 |
| 3 | shared line | 35 | 368.40 | 761.43 | 766.30 |

**Preserved failures:** singleton `{5}` selects `gxBQ_window_00_frame_5::G0::181:973` (527.21 px worst unseen); shared pair `{5111,5766}` selects `gxBQ_window_01_frame_5111::G0::136:541` (458.62 px); shared triple `{5,5111,5766}` selects `gxBQ_window_01_frame_5111::G0::0:17320` (373.75 px). All three were inspected on unobserved raw GX0 and lie on the wall rather than the annotated floor court. The triple’s pool has 193 eligible occurrences. Shared line remains a negative control, not a fix.

Adjacent `{0,5}` selects shared paint `GX0::G1::143:158`: **16.40 px** worst unseen. Other pairs: median **10.96 px**, worst **458.62 px**. Triples containing 0 and 5 are also separated in the tables. No sampling interval or best labelled subset is recommended.

**Am3 contrast:** singleton paint at frame 0 selects `am3_window_00_frame_0::G1::43:159` (15.29 px on unseen 10514); singleton 10514 selects `am3_window_01_frame_10514::G1::30:325` (11.30 px on unseen 0). Shared two-frame paint selects `am3_window_00_frame_0::G1::43:451`, with **11.07 px** maximum across the two observed frames. Its two-frame pool has 204 occurrences. There is **no unobserved target left for the two-frame Am3 test**. The three Am3 paint geometries were inspected, bringing new task-2 visual inspection to six distinct geometries total.

### First local scene experiment

**First experiment: two temporally separated proposal frames, unchanged shared median paint versus per-frame paint at equal access.** Escalate to three for an empty pool or observed-frame paint-ID disagreement; separately record existing-gate failures. ID disagreement can over-escalate equivalent geometries: **15/21 pairs**, **27/35 triples** here. Both bad multi-frame witnesses disagree, but future failure detection is unproved. At three, flag/abstain if unresolved. This adaptive policy was **not run**, and no favourable third frame was chosen.

Proposal/observed-score-cell proxies are **512/512**, **1,024/2,048**, and **1,536/4,608** for k=1/2/3, versus **3,584/25,088** for full seven-frame access. These are not timings. The 2→3 step multiplies score cells by 2.25 and still does not eliminate severe errors. Cache comparisons condition on the archived registrations (including the common anchor), not registrations estimated using only the sparse samples. Real-clip verification must check sparse registration, actual work/latency, and locked visible fit. These familiar development views establish neither held-out F1, change detection, a sampling interval, nor reliable streaming acceptance.

## Deliverables

`run_followups12.py` is the one standalone implementation. `task1_per_frame.csv.gz` has 21 published-paint measurement rows; `task1_gx86088.csv.gz` has the two named courts plus three alternatives; `task1_oracle_scan.csv.gz` preserves all 278 diagnostic rows; `task1_details.json.gz` contains transforms, symmetry IDs, plane/area diagnostics and sensitivity checks. `task2_subsets.csv.gz` has 290 decision-level rows; `task2_evaluations.csv.gz` has 1,980 target rows; `task2_distributions.csv.gz` contains sample-count/temporal-category distributions; `task2_locks.json.gz` is the selection regression fixture. `checks.json.gz` records verification. Figures **01–03** cover task 1; **04** covers the six task-2 controls.

Local verification is required before integrating the returned code or experiment proposal.
