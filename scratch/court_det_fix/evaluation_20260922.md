# Evaluation worklog

## Resume

All four pickup comparisons and the documentation/publication follow-up are
complete. Start with [pickup](pickup.md) for the next run, [FP_INDEX](FP_INDEX.md)
for files by idea, and the [GitHub packet](evidence/review_20260922/README.md)
for independent review. Existing source packets and the directory layout stay
unchanged. The wider 71-case test and scene-level detector work have not run.
Read [evaluation results](evaluation_results_20260922.md) for the measurements.

W5 full-frame review finds 19/27 qualitatively usable selections for `(3,3)`
and 21/27 for both `(4,3)` and `(5,3)`. The latter two select identical geometry.
These are development-set judgements, not held-out accuracy. No directional
floor makes automatic acceptance safe. Five of the six failures change when
the existing temporal player-support gate is applied. Full-frame inspection
finds that three GX replacements become usable; yellow and SS21-10 remain
wrong. The gated result is 24/27, with Am1 also still failing.

## Concerns and observations

- The 27 cases are development data, not an independent accuracy estimate.
- G1 scene 0029 has saved generation candidates but no preserved scoring inputs.
- The corrected person-mask comparison is complete for all five cases.
- The user authorised Claude Code Opus 5.0 for independent review.
- User review distinguishes rare tolerable fallbacks from clean fits. GX
  G1 `143:158` clips a far-left sliver; G1 `16:44` misses much of the far
  long-service-to-baseline strip. The old usability counts are not clean-fit
  counts. Frequent errors of the latter size would block deployment.

## GitHub evidence and next-test follow-up

- Scope: top-level navigation and design docs, an `FP_INDEX.md` concept map,
  and a bounded export of existing results, images and candidate geometry.
  No detector changes, new model runs, directory moves or recovery cleanup.
- Three read-only Luna max tasks check historical leads, committed sample
  coverage and evidence sizes. Findings receive a parent evidence check.
- The sample inventory was checked directly against all three packs and the
  27 saved case checkpoints: 47 frozen cases, with 20 outside the recent panel.
  Another 24 broadcast records are a separate arm (8 non-court, 16 unlabelled),
  not positive accuracy examples or new videos. The planned total is 71 cases.
- Historical net-image and pooled-fragment refinement is a genuine omitted
  mechanism. Its Yellow 221.19 → 26.01 → 21.22 px sequence uses 1280×720 and
  changes both cues and geometry; it is not an isolated temporal ablation.
  W5 source inspection confirms same-view refitting and record-only post
  diagnostics, not the older combined method.
- User clarification makes sparse multi-frame scene agreement a design
  requirement. SVD search reduction is an early speed experiment; graph/shared-
  state search follows settled matcher rules. The old fixed-group SVD test
  proves neither a speedup nor that this broader lead should be discarded.
- One bounded Claude Code review used `claude-fable-5-1` at high effort
  (reported model matched; no child reviewers; $1.72). Its record is
  `local_scratch/external_delegate/20260922-design-fable/result.md`.
  It checked the 24/27 arithmetic, inspected GX far-end imagery and a seeded
  random SS03-38 overlay, and supported wider coverage before more tuning.
  Its gate-sampling concern was checked: broadcast has 3 footpoint samples per
  case, GX 29–30, other amateur 29–31. The plan now records that difference and
  separates gated/ungated outcomes. Thresholds must be chosen before the run.
  Its claim that expanded cases have no numerical controls was too broad:
  W5's original control column is empty there, but later frozen-reference
  diagnostics exist. Those references are not independently approved truth.
  The review preceded the final SVD/graph/scene-level clarification; those
  additions remain design requirements and proposed experiments, not results.
- Validate the export against its source records and check tracked links from
  a GitHub-only perspective. Review the staged packet and sizes before commit.
  The user authorised succinct commits and pushing this feature branch.
- Evidence commit: `df4061f` — Save court detector comparison evidence.
  Published all 27 unannotated input images, 30 W5 arm-winner previews,
  gate/source alternatives, 122 G0/G1 previews, corrected-mask diagnostics,
  GX full/far-end sheets and Am3 previews. Compact source scores and geometry
  are included. The full download, duplicate PNG galleries and large working
  populations remain local; no evidence was deleted or moved.
- The export is about 65 MiB; the complete publication adds roughly 102 MiB
  including raw inputs and the canonical temporal evidence. No individual new
  file exceeds 3 MiB. Forty W5 ablation rows lack geometry in the compact
  source review: IDs, scores and overlays are published, but their exact
  geometry remains in the full local records. The packet states this limit.
- Direct equality passed for 81 W5 selections/geometry/scores, 159 G0/G1
  crossed cells and 58 temporal choices/homographies. Export review caught
  collapsed mask-stage labels; the final table preserves five cases across
  two distinct stages and includes the GX5 pre-cap/uncapped diagnostics.
  Full export and four-worker rebuild passed. Syntax, relevant Ruff, whole-
  project Pyrefly (0 errors, 39 suppressed), publication-link checks and
  `git diff --check` all returned exit 0. Twelve top-level/packet Markdown
  files have no missing or unpublished link targets. No production code changed.

## Scope and state

1. W5: evaluate all three directional floors, including full-frame regressions
   and disagreements with earlier rulings. Select a floor only if supported.
2. G0/G1: separate proposal access, scoring observations and saved stages.
3. Person masks: compare five corrected fixed-direction matcher inputs.
4. Independent/shared scoring: compare scorers on one verified automatic union.

Detector integration, CourtKeyNet removal, new representative data collection
and a new remote W5 sweep are outside this evaluation pass.

## Execution

- Three Luna max workers own separate new evaluation directories: G0/G1
  measurements, temporal replay, and W5 source/gate ablations. Their scripts
  and results receive a separate parent audit.
- Claude Opus 5 reviewed the W5 packet and all 30 distinct selected overlays.
  Its judgements agree with the full-frame review. It also read the preliminary
  review, so visual agreement is not a blind replication. Review files are in
  `local_scratch/external_delegate/20260922-w5-evaluation-server-retry/`.
- A local profile measured nine W5 candidate evaluations across three frames:
  7.725 s total, including 4.691 s in unused junction diagnostics (60.7%).
  There were 2,268 greyscale conversions. This small sample identifies waste;
  it is not a deployment benchmark. Runtime code remains unchanged.
- A reproducible paired timing check now lives in W5
  `evaluation/profile_measurement.py`. Nine measurements took 14.981 s with
  junctions and 6.322 s without them (2.37× faster). Every other evidence field
  and array matched exactly. Measurements alternated order and ran alongside
  other local analysis, so absolute times are not a clean latency benchmark.
- Eighteen missing raw frames were copied from the existing W5 checkout on
  Carmack. Existing local files and remote data were preserved.
- Corrected person-mask jobs launched on Carmack for GX5, SS03-16/17/19 and
  SS21-20. Remote checkout: `/scratch/ahalperi/court_det_fix/` followed by
  `w5_directional_20260921_r1_checkout/scratch/court_det_fix/`.
  New isolated sibling: `line_identity_eval_20260922/`.
  Run: `runs/person_observations_repair_20260922/`.
  Launch PIDs: 2439088–2439092. All five input preflights passed.
  The launcher uses frozen frame routing directly; no compatibility symlinks.
- Corrected SS03-17 and SS21-20 have completed and passed input-aware
  accounting. SS03-17 still retains a camera-eligible candidate 6.675 working
  pixels from its manual control. Full-frame inspection finds that candidate
  and the line winner (10.643 px) pragmatically usable. The paint winner
  (191.989 px) extends into the spectators. This is a selection failure, not
  loss of a useful proposal. Do not reject the proposal source on this result.
- Temporal replay's corrected GX0 smoke matches all 512 native S0 scores
  (maximum difference 4.44e-16). AM3 registration has 378 inliers and 0.556 px
  held-out median; the overlay aligns static court markings. Full GX7 scoring
  needs target-parallel execution: local smoke took 243.4 s for 512 candidates.
- Whole-project Pyrefly check: exit 0, no errors (39 suppressed). New wrapper
  and profiling scripts have direct runtime smoke checks; final lint remains
  pending until worker edits settle.
- Expanded G0/G1 is complete for all 27 cases. S0 exists for all cases; S1
  exists for 26. The selected gallery contains 122 source-qualified IDs,
  not 122 geometrically distinct courts.
- All five corrected person-mask jobs finished with exit 0 and were pulled.
  Accounting is complete. GX5 retains no camera-eligible court after the
  ordinary 256-candidate cap. Its closest pre-cap proposal remains 23.66 px
  from the approved control; the closest camera-eligible proposal is 90.67 px.
- Am3 replay is complete: 1,024 candidate occurrences, 204 eligible on both
  frames. Per-frame and shared line/paint selections all look pragmatically
  usable in the reviewed common-union overlays. Two frames cannot establish
  robustness to temporal outliers.
- The Am3 native check exposed inherited G1 floor evidence measured on S1.
  Replaying S0 changes that diagnostic on 61 and 56 native candidates.
  All native line/paint scores match; geometry and camera eligibility match.
  Floor evidence is unused by these rankings. A Luna audit traced the cause;
  Claude Opus 5 independently reproduced selection and sampled re-scoring.
  Review: `local_scratch/external_delegate/20260922-temporal-scoring-audit/`.
  Its cache-alignment concern matters if registration changes between runs;
  the present alignment was checked. Use a new output directory for changed
  registration. No extra provenance framework was added.
- GX temporal replay hit OpenCV's 32,767-row remap limit in inherited paint
  sampling. The evaluation wrapper now batches 256 courts per paint call.
  Direct equality passed for 512 courts; a 3,584-court smoke passed. Failed
  targets restarted with the same candidate/scoring contract. At most six
  remote target workers run together; native libraries use one thread each.
- Remote temporal owner: `temporal_eval_20260922/` beside the isolated matcher
  directory above. Active batches: `gx.paint_retry` (689, 5111, 5766),
  `gx.retry` (86088 remaining), and `gx.final_retry` (0, 5, then 77876).
  Earlier failure receipts remain until their target retry completes.
- The user permits commit/push locally and pull remotely for synchronisation.
  Large provisional packets remain rsync-only. No commit or push made yet.

## Completion

- GX replay completed all seven targets and reused all seven checkpoints for
  aggregation. The fixed 3,584-candidate union contains 278 all-frame-eligible
  candidates. Native / common-union per-frame / shared-median line selection
  identifies the played court on 2/7, 3/7 and 0/7 frames; paint selection does
  so on 5/7, 7/7 and 7/7. Every selected full-frame overlay was inspected.
- Far-end raw/overlay crops were added for all seven GX frames after the user
  highlighted their diagnostic value. They show residual errors hidden by
  good near-end fits, notably G1 `16:44`. GX counts are explicitly correct-court
  selection counts, not precise-fit counts. Relative far-marking accuracy
  remains a detector-quality question, not another uncompleted scoring run.
- Parent re-derived the full GX selection exactly from its matrix. Replacing
  every native row with the saved local S0 evidence changes no winner in any
  access arm. Small line-score differences are below 0.001. NumPy/SciPy differ
  between local (2.4.4/1.18.0) and remote (2.4.6/1.17.1); scorer sources match.
  A bounded diagnostic reproduces one difference through zero-tolerance
  image-border inclusion after floating-point projection. Assignment does
  not change. The experiment did not alter the scorer mid-comparison.
- GX5 person-mask pre-cap candidate `17:836` is recognisably the played court
  but imperfect. Its camera error is 0.182, over the 0.1 cutoff. This reinforces
  the need to test hard-gate rejection as well as wrong-winner rates.
- Relevant Ruff checks: exit 0. Whole-project Pyrefly: exit 0, 0 errors
  (39 suppressed). Wrapper syntax, painting batch equality/large-batch smoke,
  actual invalid-projection handling, score replay and all rendering checks
  passed. Markdown links and `git diff --check` pass. No whole pytest run:
  production model/pipeline code was not changed.
- Final report audit: exit 0, factual fidelity passed, weighted score 92.4/100.
  The fresh cold reader rated story recovery, experiment accounting, decision
  utility and visual communication 4/4 each. Mechanical score: 74.8/100
  (orientation 75, terminology 40, syntax 98.1, processing 90, continuity 40,
  visual assets 100). Remaining flags mainly concern defined experiment labels
  and case IDs; the purpose is explicit under “What was compared”. The reader
  recovered all six required answers without guessing. Precise far-end accuracy,
  held-out reliability and practical union runtime remain unmeasured, as stated.
