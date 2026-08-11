# Corrected findings for the accepted-contact search

The experiment must run the same GT-free search over all 239 one-to-one rallies. It moves forward past accepted impulses that have usable evidence of being junk. Once it finds a credible contact with incoming motion, it follows that trajectory backwards through earlier accepted contacts.

The 97 first impulses unmatched at ±10 base-30fps frames are an analysis slice. Ground truth (GT) does not decide whether the search runs.

## F1. Accepted contacts provide the only search nodes

PR #82 reads each frozen span's `filtered_by_rally` frames, sorts them, and checks them against the raw wrist, suppression, and exclusion fields. The search starts at the first accepted frame and may move among accepted frames only.

Earlier raw or rejected impulses remain out of scope. They become a later experiment only if the accepted-sequence search shows a clear need.

Why this matters: every search decision uses inputs available to the real system.

## F2. Forward and backward movement solve different problems

The forward step handles junk. If accepted impulse `Ai` has enough post-contact shuttle trajectory to judge but no outgoing motion, the main search skips it and checks `Ai+1`.

The backward step handles logical predecessors. If a credible `Ai` has incoming motion, the search follows that trajectory backwards and asks whether an earlier accepted impulse originated it. A connected origin becomes the next contact inspected backwards.

Why this matters: `incoming` is not shorthand for `unshown serve`. The search must first try to find an accepted origin.

## F3. Missing evidence remains unknown

The report label is `not enough shuttle trajectory to tell`.

The main rule stops with that result whenever the post-contact evidence cannot judge whether `Ai` is junk, the pre-contact evidence cannot judge whether a credible `Ai` has incoming motion, or a backwards trace breaks because the shuttle evidence becomes unusable.

A separate sensitivity check may continue forward past an unknown accepted impulse. It cannot replace the main result.

Why this matters: unavailable tracking cannot justify a skip, a visible-serve call, or an implied missing serve.

## F4. A usable trace can end without an accepted origin

If incoming motion into `Ai` remains usable while tracing backwards but no earlier accepted contact originates it, the search calls `Ai` the first visible post-serve contact. It records an implied unshown or undetected serve before `Ai` without inventing an exact serve frame.

If the trace ends because evidence becomes unusable, the result is `not enough shuttle trajectory to tell` instead.

Why this matters: absence of an accepted origin and absence of evidence are different observations.

## F5. A visible accepted serve is a terminal result

When `Ai` has outgoing motion and enough pre-contact evidence to show no incoming motion, the search calls `Ai` the visible serve and returns its accepted frame as the opener.

Why this matters: an authentic serve at the first accepted position stays unchanged. A serve reached after forward skips or backwards tracing is recovered at its accepted frame.

## F6. The existing PR #82 table fixes the baseline, not the new result

The checked PR #82 output contains 239 primary rallies. Their current first accepted impulses are:

| GT label at ±10 | Rallies |
| --- | ---: |
| contact 1 | 119 |
| contact 2 | 19 |
| later | 4 |
| unmatched | 97 |

These labels score the current start after the search has run. They never select a branch of the search.

Why this matters: the final comparison can count fixed, damaged, unchanged, and unknown results over the same 239-rally denominator.

## F7. PR #82 supplies the incoming-motion primitives

The existing work provides `closest_pre_contact_run`, `measure_incoming_motion`, `fit_robust_distance_trend`, the recurrence-clean path checks, and the fixed 0.05-BH incoming rule.

The corrected sweep still needs to map the smallest symmetric post-contact measurement and the exact continuity test between an earlier contact's outgoing path and a later contact's incoming path. Those details must be fixed before implementation and must not be tuned against GT.

Why this matters: the experiment should reuse the old measurement convention while adding only the minimum forward and connection logic.

## F8. The primary outputs are transition counts over all 239 rallies

For every rally, record the current first-impulse GT label and the reconstructed outcome. Report:

- currently wrong starts fixed
- currently correct starts damaged
- results unchanged
- results ending as `not enough shuttle trajectory to tell`
- accepted junk impulses skipped
- accepted origins found while tracing backwards
- implied unshown serves after a usable trace finds no accepted origin
- selected accepted rank and final visible-contact GT ordinal

Repeat the relevant breakdown within the 97 currently unmatched starts. Keep ±10 primary and use ±5 and ±30 only as compact sanity checks.

Why this matters: the same rule is evaluated on correct and incorrect starts, so improvement cannot hide damage to the existing successes.

## F9. The implementation remains scratch-only

No production code, rally segmentation, contact detector, raw-candidate promotion, learned model, dynamic programme, or threshold sweep belongs in this pass. The work should be one small analysis module, focused helper tests, checked compressed evidence, and a short report with roughly five examples.

Why this matters: a negative result should explain why the simple accepted-sequence reconstruction fails without accumulating another heuristic stack.
