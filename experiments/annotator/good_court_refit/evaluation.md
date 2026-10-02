# What did better courts change?

**Status: prepared; no new-court fits or evaluations have run.** Replace this
paragraph with the main result after evaluation: what improved, what still
fails, and which bundle should be kept. Leave unknown results blank.

This evaluation asks whether the custom court detections recover usable rally
annotations, and whether rejecting masked contacts without a nominated player
helps further. The annotation goal is complete rallies with correct contact
times and player sides, plus a useful queue for human review.

## What was compared?

| Output | Court inputs | Contact rule | Fitting |
|---|---|---|---|
| Base | Chosen new extracts | Existing candidate policy | Fresh complete bundle |
| Base + rule | Same new extracts | Masked and no nominated player | Base bundle; inference-only diagnostic |
| Veto | Same new extracts | Masked and no nominated player | Fresh downstream and confidence models |

Base and Veto share the same fitted contact trees and feature cache: the rule
changes which candidates reach sequence selection, not how the contact
tree is trained. There are two builds. Base + rule is an optional later
inference pass, not a third fit. Base → Base + rule measures the immediate rule
change; Base + rule → Veto measures the downstream refit with that rule.

Keep whole-video contact-side geometry fixed across these outputs. The rule
requires a guard grade in the fitted preprocessing's rejected grades (default
1–3), with no nominated player at any of the five existing time samples. It
applies before nearby-contact suppression and repair shortlists. A nominated
person counts even when wrists are unavailable.

Record the chosen detector files/config: **___**. Label source: **___**.
Code revision: **___**. Python/scikit-learn: **___**. Run directory: **___**.
Keep the saved config with the run; there is no need to repeat every path here.

### Populations and scoring

- Development: 32 original-ShuttleSet videos in groups A–D. Keep training-order
  and held-group diagnostics separate from validation quality claims
- Validation: eight original-ShuttleSet videos in V, scored with the 32-video
  contact tree and downstream models fitted on A–D. Choose the rule here
- Historical comparison: 46 ShuttleSet22 videos, excluding video 15 and its
  misaligned labels; expected 3,327 retained labelled rallies. These videos
  have already been inspected and helped motivate the rule
- Preserve original-ShuttleSet `sset_15`; it is a different video

Before viewing results, record how contact/side losses will affect the choice:
**___**. Do not choose solely by fully correct rally count. Any genuinely
untouched extra videos should have their own table and role; do not describe
the repeatedly inspected ShuttleSet22 set as new test evidence.

Contacts match one-to-one within ±10 frames at 30 fps, scaled to source FPS.
A fully correct labelled rally requires every contact, the right known sides,
no extra contacts and valid bounds. Report distinct recovered labelled rallies
over all labelled rallies. Also report correctness among predicted sections,
including the unjudgeable count. A completely missed rally must remain in the
first denominator even though it creates no predicted section to judge.

## How much usable annotation is recovered?

Make one table per population. Start with validation; add ShuttleSet22 after
recording the choice. Show counts as well as percentages.

| Measure | Base | Base + rule | Veto |
|---|---:|---:|---:|
| Labelled rallies / contacts | ___ | same | same |
| Distinct fully correct rallies | ___ | ___ | ___ |
| Rallies reached but not correct / wholly missed | ___ | ___ | ___ |
| Predicted / matched contacts | ___ | ___ | ___ |
| Contact precision / recall | ___ | ___ | ___ |
| Timing matches with correct known side | ___ | ___ | ___ |
| Correct / wrong / unjudgeable predicted sections | ___ | ___ | ___ |

Use paired gained/lost rally IDs to explain the change. A compact coverage plot
can show fully correct, reached but not correct, and missed rallies. A reached
rally has at least one timing-matched label in the saved contact stream; this
does not guarantee that a usable clip contains it. Keep denominators
visible and use colours that do not depend on red–green contrast.

For historical context, score old outputs on the same 46-video population and
labels. The historical model recovered 1,763 complete rallies and the fresh
old-court model 1,734 on this population. A comparison with the new run also
changes fitted models and the pinned fit order; it does not isolate the effect
of courts. An old bundle run against new courts is an optional inference-only
control if that attribution matters and its runtime environment is compatible.

## Did the new courts recover the old failure cases?

The earlier evaluation found 2,374 of 3,633 missed contacts in court-rejected
scenes (65.3%, excluding video 15). That describes where misses occurred, not
the miss rate among all contacts with rejected courts.

Measure both directions with the new outputs:

| Input state at a labelled contact | All labels in state | Missed labels | Miss rate | Share of all misses |
|---|---:|---:|---:|---:|
| Court rejected | ___ | ___ | ___ | ___ |
| Court accepted, no player nominated | ___ | ___ | ___ | ___ |
| Court accepted, one player nominated | ___ | ___ | ___ | ___ |
| Court accepted, both players nominated | ___ | ___ | ___ | ___ |

Use the same frame-level definitions in old/new comparisons. Compare the
fraction of playable footage accepted as well as contact outcomes: simply
accepting more scenes can increase false candidates.

Revisit video 53's corrupted court outline and video 17's lost far-player
selection using matched old/new frames. Report whether the new geometry fixes
the observed failure, whether player picks recover, and whether the labelled
contacts and whole rally recover. A small paired overlay is more useful here
than another aggregate plot. Also inspect a sample of newly accepted scenes
and any large per-video regression; a plausible outline can still admit a
replay or select a spectator.

## What does the contact rule remove, and what changes afterwards?

On old courts, the nomination rule rescued nine complete rallies and damaged
none, but it gained/lost 10/7 timing matches and 12/13 correct-side matches.
The losses occurred inside rallies already counted as wrong. Keep that
possibility visible in the new comparison.

| Paired change | Base → Base + rule | Base → Veto |
|---|---:|---:|
| Fully correct rallies gained / lost | ___ | ___ |
| Label timing matches gained / lost | ___ | ___ |
| Correct-side timing matches gained / lost | ___ | ___ |
| Masked candidate rows / rejected rows | ___ | ___ |
| Directly forbidden baseline final contacts | ___ | ___ |
| Those contacts matched to a label / recovered nearby | ___ | ___ |

Distinguish directly rejected contacts from changes elsewhere in the rebuilt
sequence. Inspect all lost complete rallies and a bounded sample of lost
contact/side matches. Record actual player visibility, court acceptance, guard
grade and whether a nearby replacement survived. Grades 2–3 indicate degraded
tracks, not proven non-contacts. Missing player nomination can also be caused
by court geometry or occlusion.

If results disagree across populations, report that disagreement. A small
positive total on familiar footage is insufficient reason to hide systematic
serve losses or player-side damage.

## Are the remaining errors timing, missing hits or sequence choices?

Compare first, middle and last labelled contacts, with court-accepted results
shown separately. Plot timing offsets for matched contacts and list the number
of unmatched labels beside the plot; a narrow timing histogram says nothing
about misses.

For useful residual examples, follow the contact through these stages:

1. Was its frame reachable in an accepted search interval?
2. Was a nearby candidate present, and did the rule reject it?
3. Did the initial contact survive nearby-contact suppression?
4. Was a complete correct repair option available?
5. Did its model score or the final improvement margin prevent selection?

Only expand this into a full option-coverage study if these examples suggest a
specific repair limit. The previous investigation proved that some rallies need
more than one insertion or deletion; widening candidate lists alone cannot
solve those cases. Keep new repair experiments separate from this two-build
comparison.

## Is the review queue more useful?

Compare correct rallies at the same human review budget, not just the same
confidence threshold. The fitted confidence scores can shift after a refit.
Use a precision-versus-kept-count curve and report correct, wrong and
unjudgeable sections at the chosen budget. Count duplicate sections recovering
the same rally only once when measuring rally coverage.

Choose a new threshold or review budget on V, then hold it fixed for the
ShuttleSet22 comparison. Show the old 0.757 threshold only as historical context.
Contact/rally labels do not validate winners, landings or hit heights. Inspect
representative downstream outputs and missing-value rates if the selected
bundle is intended for a dataset rebuild; do not call those fields accurate
solely because the rally score improved.

## Decision and remaining work

Chosen bundle and reason: **___**. Contact-rule setting: **___**.
Validation gains and losses supporting the choice: **___**.
Historical comparison and limits: **___**.
Known remaining failure cases: **___**.

Update the fitted-bundle instructions and the untracked post-refit documentation
checklist. Confirm the selected bundle loads with its stored policy, run the
existing integration checks, then decide whether to merge and promote it.
Keep optional side-geometry or repair studies separate unless the evaluation
reveals a concrete blocker.

The earlier [wrap-up evaluation](../../../scratch/annotator_wrapup_evaluation/README.md)
provides the historical coverage, visual examples and label checks. The
[old-court regression findings](../old_court_regression/findings.md) explain
which apparent fixes have already been tested and what they established.
