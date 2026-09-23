# Wider test of proposal seeds and net preference

## Resume

The user authorises a bounded follow-up experiment with graded post evidence,
priority for reliable post bases, allowance for lean and sag, and capped
influence over paint. Numerical weights require a small sensitivity check.
The first batch measures base support in the 15 reviewed gallery cases plus
seeded Am1. An independent design review runs alongside that diagnostic.
No new scoring rule or numerical weight has been adopted yet.

The previous 71-case scan is complete. Unrestricted preference loses good
courts, so core promotion remains held. The combined runner's cache is repaired,
but further proposal comparisons need freshly and consistently measured paint
scores. The first follow-up uses frozen pools without changing their scores.

The [Am1 report](../colour_consistency/AM1_RECOVERY.md) records the accepted
starting point. No production default changes during this comparison.

### Bounded follow-up plan

1. Measure existing fragment support near post bases and fragment endpoints.
   Reproduce the old whole-piece coverage exactly before interpreting the new
   diagnostics. Use the 15 reviewed gallery cases and seeded Am1 for design.
2. Choose one simple bounded, graded score after assessing whether useful base
   evidence exists. Record its formula and numerical settings before selection
   replay. Inspect a small declared weight range, not a search for a perfect
   development score. Stop if the evidence requires a new post detector.
3. Replay the fixed design across the saved pools. Keep remaining views out of
   design decisions until then. Report source families separately: nearby
   frames from a reviewed video are correlated validation views, not unseen
   sources. Centre and Am4 provide sources absent from this net-review gallery,
   although earlier detector experiments have used them. Seeded Am1 is always
   a development positive.
4. If selection remains useful, check the accepted stripe-polarity correction
   on the promising selected courts as a separate fitting comparison. Sol builds
   the review gallery from the existing SVD template. Obtain an independent
   audit, retain all substantial changes for review, and document the outcome.

In scope: scratch measurement/ranking scripts, saved-pool replays, focused
checks, galleries and current documentation. Out of scope: production defaults,
new proposal-generation work, a new object detector, manual cue selection,
player changes, colour searches and a physical net simulation. Existing gates,
candidate geometry and paint scores stay fixed during the selection experiment.
Missing evidence is neutral. Clear regressions on known good courts block
promotion. If useful base evidence or a safe bounded preference fails to emerge,
stop and preserve the findings.

Checkpoint commits remain authorised on `fix/court-det`: record the measurable
base evidence, then the bounded preference outcome and gallery. Scope-specific
checks cover exact old coverage, coordinate/sample ordering, deterministic
ranking, endpoint reversal, score bounds and no-evidence fallback. Run scoped
lint and the project type check after coherent code changes. Documentation-only
checkpoints use diff and link checks.

### Fixed policy for the first bounded replay

The base pilot and independent design review support testing a conservative
lower-post cue. A post contributes when at least one of its lowest six samples
has segment support and none of those supporting segments continues more than
4 working pixels below the projected base. The base must be in frame. Matching
reuses the existing 4 px perpendicular, 8 degree direction and 2 px extent
tolerances. Segment endpoint order is irrelevant. This is support for a plausible
base, not verified post identity. A failed check removes only that post's bonus.

`reward = (left_base_supported + right_base_supported) / 2`

`combined_score = saved_paint_score + weight * reward`

Only existing full-court candidates compete; exact ties retain original rank.
Normalising by two prevents an off-image post from increasing the reward.
Tape, tops and upper-shaft alignment do not contribute. Their idealised shape
can therefore neither strengthen nor suppress this cue directly. The projected
post direction and height still locate the lower-quarter samples. The bases
themselves depend only on the court homography. Focal-bound cases retain a
diagnostic flag rather than receiving an automatic veto.

Primary weight: **0.04**. Sensitivity arms: **0.02 and 0.08**, then base overrun
tolerances **2 and 8 px** at weight 0.04. Weight zero checks baseline equality.
These settings are fixed before the full replay; no preferred setting will be
chosen from validation outcomes. The 0.04 choice uses development evidence and
is provisional. It is not inferred from the historical/fresh score discrepancy.
At the largest bonus of 0.08, the five known regressed choices cannot displace
their baselines because each loses more than 0.08 paint score. Other candidates
may still produce new regressions; preserving the old baselines is insufficient
validation on its own.

The split is frozen in `bounded_split.json.gz`: 16 saved development cases,
49 additional views of development sources and 6 views from Centre/Am4.
Seeded Am1 is an extra development pool. It is excluded from all validation
claims. The split was recorded before the base diagnostic results were read.

Court and net colours are independent. The user explicitly rules out requiring
them to match. This experiment uses only segment geometry; the separate accepted
stripe-polarity correction uses local court-marking contrast. Colour-consistency
rejection remains closed.

## Concerns and findings

- Line-source correction: these experiments use cached DeepLSD segments, read
  directly from the frozen packs by `verifier.prepare_segments`. No Hough
  extraction or Hough composition occurs. Earlier net-trial descriptions and
  saved metadata saying "Hough" are incorrect. The court proposal path groups
  and merges separate line families; net support uses the original segments.
- The 16-case base pilot reproduces every saved coverage value exactly.
  Lower-quarter support alone does not establish a physical post base.
  Am1's accepted pair has endpoints near both predicted bases; GX views often
  lack comparable endpoint support. All 16 local frame hashes match the full
  frozen manifest, including the entries absent from the older inventory.
- The fixed rule changes 13/71 saved selections. Letterboxed78 jumps from
  full-court rank 1 to 706. Visual inspection confirms a shortened court with
  its near baseline on an internal marking. The original is substantially
  better. This exceeds any centre-versus-edge annotation ambiguity.
- Unlabelled control frame 52563 is visibly a court view. The original aligns
  well; the trial moves its far baseline above the painted court. Preserve the
  original control label in the experiment; this inspection is separate.
- Good court fits can have weak projected-net support. Frame 52563's original
  has zero tape support. Investigate projection sensitivity before proposing
  a replacement preference; do not interpret missing support as court failure.
- Opus found, and source inspection confirmed, a new replay cache bug: full
  six-field W5 gates were compared with two preliminary fields. Every template
  was therefore rescored. That comparison is repaired. An added enumeration
  order check also prevented reuse; it was removed because list position does
  not affect measurement. Exact geometry, full gates and template evidence
  remain required. The accepted Am1 net winner reproduces exactly.
- Identical geometry can receive different freshly measured paint scores.
  A bounded audit reproduces this locally. Six remote spot checks reproduce
  the fresh scores too, including the two largest sampled template changes.
  Original, later and transferred GX5 frame files have the same MD5. The
  audit's initial laptop-versus-remote explanation is therefore unproven.
  Freshly measure both baseline and trial before further seed comparisons;
  cache repair alone cannot make mixed historical/fresh scores comparable.

## Completed saved-pool scan and visual review

`scan_saved.py` completed 71/71 cases with zero failures and zero mismatches
against the earlier three-case fixture. There are 47 court views and 24
controls, including eight originally labelled non-court views. Ten court
choices and three unlabelled-control choices change. No labelled non-court
view has a strong full-court candidate. Four have strong ungated candidates
blocked by the existing full-court gate. Acceptance remains 1/8 by construction;
that equality is not new evidence of specificity.

The [gallery](gallery/index.html) reuses the shared SVD template. It shows all
13 switches plus unchanged GX5 and GX0. It contains saved pools only. Native
and working coordinates, selected IDs, image links and JavaScript syntax were
checked. The separate combined GX5 replay is not shown there.

The user authorised a handful of visual checks while away on 23 September.
Five cases were inspected using the gallery's actual selected geometry:

| Case | Visual observation |
| --- | --- |
| Letterboxed78 | Clear regression: shortened court; false near baseline. |
| Control 52563 | Clear regression: far baseline displaced above the painted court. |
| Am3-0 | Same court in both; trial shifts visible line alignment. No recovery demonstrated. |
| GX689 | Both identify the played court; no wall selection in either saved gated choice. Small alignment differences remain for user review. |
| Am3-17174 | Trial looks plausible and numeric visible-landmark agreement improves. Treat this as a possible modest gain, not a rescue or final user ruling. |

These are coordinator observations, not user acceptance labels. Screenshots
and isolated shared-template rows are under
`local_scratch/net_recovery/20260923/visual_review/`. Do not turn this selected
handful into an overall accuracy count.

### User review — 23 September

The user reviewed all 15 gallery rows. These rulings supersede the provisional
visual judgements above. Net preference gives useful gains on several views,
but also degrades good courts. Keep unrestricted net preference experimental.

| Case | User's visual judgement |
| --- | --- |
| Letterboxed78 | Substantial regression. The projected right post starts part-way up the real post. The left is a hallucination further along the court line. |
| GX5 and GX0 | Both versions are perfect. |
| GX689 | Net preference improves the court; only a tiny top-left shortfall remains. The projected right post sits slightly further along the court outline than the real post. |
| GX5111 | Same judgement as GX689. |
| Yellow156 | Net preference improves the court. It appears to follow the inner pole/tape junction, pulling the court inward. The user suggests inward net tension as the cause; this mechanism is unverified. |
| Am2-150 | Net preference degrades the court. The saved W5 version also fits the net better visually. |
| Am3-0 | Net preference pulls the court inward by following the inside of the net structure. Saved W5 avoids this problem. |
| Am3-17174 | Both courts are equally useful. Saved W5 slightly overshoots the top right; net preference slightly undershoots. The trial's projected net fits much better. |
| SS03-0016 | Both insignificantly overshoot the top left. Net preference slightly improves the court and left net outer, but fits the net top worse. |
| SS21-0034 and SS21-0044 | Both courts are fine. Vertical lines hug the stripe inner; the user asks whether the earlier centre/edge correction is active. |
| Unlabelled control 00000001 | Saved W5 undershoots the far baseline. Net preference fixes that but marginally overshoots the near baseline. The trial is roughly usable in this difficult view. |
| Unlabelled control 00052563 | Clear regression from a good saved court. Net preference overshoots onto the back tape of the court overflow area. |
| Unlabelled control 00081233 | Marginal regression. The trial's left sideline hugs the stripe inner and its right sometimes lies slightly outside. Both courts are fine. |

The control IDs retain their original experimental labels. These visual
judgements do not relabel the input set.

The gallery uses saved W5 geometry. Neither column applies the later automatic
stripe-polarity correction. The inset therefore does not show that the accepted
correction was reverted or failed. The net overlay is projected from each court;
it is not a separately detected set of posts. The rule requires both tape
halves and only one post to reach 75% fragment coverage. It neither verifies
post-foot contact nor establishes that supporting fragments form a real net.

The user proposes a graded preference: two mutually coherent posts should
contribute more than one supported post, and one more than none. This is a
candidate design, not an implemented rule. Missing posts may be obscured.
Independent fragment matches alone do not establish mutual coherence, and
any net reward still needs a bounded influence on paint-based selection.

The user also stresses that inward post lean can offset the top from the base,
and that the tape sags along its length. The current projection models upright
posts with a fixed centre height of 1.524 m versus 1.55 m at the posts, using
two straight tape halves. It cannot represent variable post lean or tape sag.
This supports investigating post-base evidence as a court anchor, with tops
and tape used more loosely to corroborate net identity. It does not establish
that tension caused every observed inset, or that post bases can already be
identified reliably from the available fragments.

### Independent review and current recommendation

Opus 5-5 xhigh reviewed the source, saved evidence and user rulings. Its report
is `local_scratch/external_delegate/20260923-net-user-review/result.md` from
the repository root. The review ran without a time limit and changed no files.

A separate replay confirms its graded-priority diagnostic: preferring any
two-post-supported candidate before any one-post-supported candidate retains
all five user-reported regressions and changes ten further choices relative
to the fixed rule. GX0 moves to full-court rank 17 and GX5111 to rank 89.
Those replacements are unreviewed; rank alone does not establish their quality.
This diagnostic tests unconditional priority, not the user's proposed bounded
weight for genuinely coherent posts.

Keep a bounded net reward as the next candidate design. Two coherent posts
can contribute more than one, while obscured posts provide no evidence.
Prioritising reliable base evidence could reduce dependence on idealised net
shape. It remains untested, and finding reliable bases is the key prerequisite.
First use the internally consistent saved pools for inexpensive selection
experiments. Further proposal-generation comparisons still need consistently
remeasured baseline and seeded pools.

The review suggests a paint band of roughly 0.01 from historical score drift.
That justification is not accepted: a difference between historical and fresh
measurements does not establish normal measurement repeatability or a safe
selection margin. No numerical cap has been adopted. The apparent separation
of gains and regressions by paint loss is development evidence only.

### What the projected-net overlay explains

Letterboxed78's trial tape follows the back edge of the floor, above the real
net tape. One projected post still has substantial fragment support. Strong
support from generic fragments therefore does not identify a real net.

On frame 52563, the original court's projected tape sits below the real tape.
The trial distorts the court while bringing its projected tape closer. Both
camera estimates hit the helper's upper focal bound of four image widths.
A diagnostic that only expands that bound to forty selects 4.439 widths for
the original, reduces camera error from 0.0894 to 0.0063, and moves the tape
about ten pixels upwards. No selections or thresholds were changed. This
identifies a projection limitation; it does not validate a new focal range.
The probe is `visual_review/focal_boundary_probe.json` under the local output
directory above. The gallery toggle displays the original fixed-rule projection.

The next useful selection experiment should limit how much paint evidence a
net preference can displace and handle uncertain projection explicitly. A
particular rank or score-gap limit has not been chosen or validated. Keeping
Am1's rank-2 recovery alone would not establish that such a limit generalises.

### Score compatibility check

`local_scratch/external_delegate/20260923-net-score-drift/result.md` contains
the bounded audit. Its decisive measurements show that fresh photometric
samples differ from the frozen evidence. The paint score contains a hard
brightness threshold, so small intensity differences can change scores by
roughly 0.01. Its attribution to different environments is qualified by the
subsequent remote checks in
`local_scratch/net_recovery/20260923/remote_remeasure.log` (exit 0).
The current local and remote OpenCV versions are both 5.0.0; NumPy is 2.4.4
locally and 2.4.6 remotely. The cause of the frozen/fresh difference remains
unresolved. The saved-pool scan uses only frozen paint scores and is unaffected.

## Current module state

- `base_probe.py`: saved-selection diagnostic of the original 24-point masks,
  lower-quarter support and fragment endpoints. All 16 cases and 30 distinct
  choices reproduce the old coverage exactly. Source frame hashes are checked
  against the complete manifest. Data/table/logs are under
  `local_scratch/net_recovery/20260923/base_probe/`.
- `bounded_split.json.gz`: fixed 71-case cohort membership plus seeded Am1.
  The bounded selection runner and a separate generic polarity replay are in
  progress; their outputs are not yet accepted results.
- `scan_saved.py`: fixed-rule selection over saved pools; no new generation.
  Complete output is `saved_net_scan.json.gz`. References enter after selection.
- `run_combined.py` and `colour_consistency/am1_recovery_trial.py`: generic
  seeded replay; cache repair complete. Original and repeated GX5 outputs are
  retained as method diagnostics, not clean seed-only comparisons. The runner
  docstring records the mixed-measurement limitation. Missing control landmark
  packs now produce an empty reference list after selection.
- `build_gallery.py`: two genuine selections per row; source geometry audited.
  A default-off projected-net toggle copies saved working-pixel segments
  directly. Source equality, build determinism and JavaScript checks pass.

## Checkpoint checks — 23 September

- Base diagnostic: syntax, scoped Ruff, exact saved coverage/fragment-union
  checks and synthetic sample-order/endpoint-reversal/in-frame checks pass.
  After replacing the incomplete inventory check with the full manifest,
  all 16 cases rerun successfully (exit 0). Whole-project Pyrefly exits 0.
  The current line-source terminology edits change descriptions only.
- Full GX5 replay and original-CLI Am1 replay: exit 0. These runs used the
  overstrict metadata cache check and are preserved under
  `local_scratch/net_recovery/20260923/repaired/`. GX5 keeps its selected ID;
  corner movement is at most 1.01e-9 native pixels. Am1's paint winner is exact.
- Net selection on the repeated Am1 pool: exit 0; the accepted net winner and
  its corners are exact. This pool has 441 children rather than the accepted
  run's 443 because all template parents were rescored.
- Final cache semantics restore the original predicate. Direct artefact checks
  find 564 reusable Am1 parents (52 templates) and 561 GX5 parents (49 templates).
  Geometry, gate and template-evidence mutations reject reuse; enumeration
  order alone permits it. No floating-point tolerance was added.
- Reconstructing both pools from the measured records with the final reuse
  predicate: exit 0. Am1 returns to 443 children and retains both accepted
  paint/net winners; GX5 has 608 children and keeps its winner. This checks
  cache assembly without rerunning generation. Script and output are
  `local_scratch/net_recovery/20260923/cache_restoration_check.{py,log}`.
- Scoped Ruff and whole-project Pyrefly after the final code change: exit 0;
  Pyrefly reports zero errors and 39 existing suppressions. Earlier syntax,
  Serena diagnostics, control-map and 71 saved-path checks passed. A full
  control replay was not run.
- Gallery JavaScript syntax, source identities, geometry, projected-net
  coordinates, image links and deterministic rebuild: exit 0. Browser captures
  were inspected for the five named cases; two also had projected nets shown.
- All 177 inline local links in the six current Markdown files resolve.
  Scan-count checks confirm 71 complete, zero failures and 13 switches (exit 0).

All seven GX views are included in the saved-pool scan. Full combined runs on
the remaining GX views are deferred until scores are measured consistently.
The 22 input files for the planned 11-case combined batch are staged on Carmack
at `/scratch/ahalperi/court_det_fix/net_recovery_inputs_20260923/`. No batch is
running there. The local manifest is
`local_scratch/net_recovery/20260923/remote_input_paths.txt`.

## Test contract

- First run the full combined change on GX5 and the recorded GX wall failures.
  Extend to the remaining GX views, Am4-319, SS21-10 and Letterboxed45 if the
  named checks remain useful. Am1 is the retained positive example.
- Separately apply the fixed net preference to all available saved wider-view
  pools, including labelled non-court controls. This isolates selection effects
  without paying for new generation or refitting. Report missing inputs.
- Keep all three sources, camera/player gates, existing paint ordering, 4096/256
  generation caps and the fixed 75% net rule. No threshold sweep or manual cue.
- Preserve measured candidate pools and scores so selection can replay cheaply.
  Separate exact geometry from immaterial diagnostic floating-point changes.
- Report changed choices, candidate rank and paint-score gap, gate status,
  visible support counts, retrospective landmarks and false-acceptance labels.
  Annotation errors are diagnostics; the user owns visual judgement.
- Use Sol high for implementation, runners and galleries. Reuse the actual
  shared gallery template. Opus may audit; its conclusions require parent checks.
- Review changed selections and requested GX cases in a compact gallery. Pause
  promotion if a known good court is lost or a clear background court is chosen.
  Finish independent checks and preserve a concrete review point.

## Scope and checkpoints

Experimental runners, outputs, galleries and current documentation are in
scope. Scene integration, a new post detector, colour-rule reopening and
parameter searches are out of this comparison. Commits on `fix/court-det` are
already authorised. Use Carmack for larger combined runs if existing inputs
and a suitable environment are available; one remote session at a time.

Checkpoint 1: reusable runners and GX5 result. Checkpoint 2: wider fixed-rule
results and changed-choice gallery. Checkpoint 3, conditional on evidence and
visual judgement: plan the smallest core integration.

## GX case identification

The saved comparison contains seven GX views: frames 0, 5, 689, 5111, 5766,
77876 and 86088. Historical notes record wall selections on GX0 and GX5 under
other proposal/selection variants. The wider W5 review reports three ungated
wall choices; its unacceptable ungated GX rows are 689, 77876 and 86088.
The notes do not uniquely name a shed-girder frame. Include all seven GX views
instead of guessing. Existing full-court gating must remain in the comparison.

Evidence: `archive/20260922/wider_evaluation_20260922.md` lines 219–229,
`archive/20260922/evaluation_results_20260922.md` lines 195–202 and
`evidence/webui_followups_20260922/return.md` line 76. The historical wall labels
refer to their named variants; do not transfer them to current gated choices.
