# Wider test of proposal seeds and net preference

## Resume

The 71-case saved-pool scan and review gallery are complete. Unrestricted net
preference loses good courts, so core promotion is held. The combined runner's
cache comparison is repaired. Next: measure baseline and seeded pools
consistently before further generation comparisons, then test a restrained
net preference. No replacement rank or score-gap limit has been adopted.

The [Am1 report](../colour_consistency/AM1_RECOVERY.md) records the accepted
starting point. No production default changes during this comparison.

## Concerns and findings

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
