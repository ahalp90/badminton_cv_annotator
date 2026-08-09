# Issue 28 serve-lookback decision brief

Prepared for review on 9 August 2026.

## Executive Summary

- **Do not build the tested serve-lookback prototype.** The raw-impulse plus
  central-pose rule recovered 0 of 136 target first strokes. Its evidence-only
  mask exemption selected 14 triggers, and none recovered a target.
- **The negative decision is robust to target-quality concerns.** Removing all
  26 targets whose first source row carries `flaw=1` leaves 110 targets and
  still zero recovery. Human event labels do not drive candidate selection,
  matching, or injection, so further target filtering cannot create a match.
- **The target needs precise wording.** The 136 rows are unmatched first
  ShuttleSet strokes with at least one later matched stroke. They are not 136
  confirmed missed visible serves.
- **Do not continue human review for Issue 28.** The remaining rows could
  measure how many targets are visible, broadcast-omitted, off-frame, or
  uncertain, but they cannot improve or reverse the prototype result. Park
  them unless a separately approved Issue 32 use needs exact composition.

## Decision

Resolve [Issue 28](https://github.com/ahalp90/badminton_cv_annotator/issues/28)
as a no-go for the proposed raw-impulse plus central-pose serve-lookback rule.
Do not proceed with the conditional implementation in
[Issue 30](https://github.com/ahalp90/badminton_cv_annotator/issues/30).

This decision applies to the tested rule and counterfactual. It does not prove
that every possible serve detector would fail. Another study would need a new
source of serve evidence and an explicit span-reassignment design.

## What was measured

The measurement used the reviewed `sset_01`, `sset_15`, and `sset_21`
broadcast timelines. It ran the normal pipeline with the committed replay mask,
then evaluated a narrow evidence-only exemption for candidate frames.

The unit of analysis was one ShuttleSet rally. An Issue 28 target had:

1. an unmatched first ShuttleSet stroke; and
2. at least one later stroke matched to an accepted pipeline contact.

The target first row was later verified using exactly one source row with
`ball_round == 1`. This avoids two `sset_01` cases where the first and second
strokes share one frame.

The prototype searched a short pre-anchor window for raw shuttle impulses. A
candidate needed a visible clean shuttle track, absent court detection, usable
central pose, wrist proximity, and contact-suppression clearance. Selection
kept the largest surviving impulse per lookback opportunity.

The counterfactual copied the natural accepted-contact map, injected selected
contacts, kept the pipeline spans fixed, and reran downstream results. It was a
measurement path. It did not change production behaviour.

Primary method and provenance are recorded in the
[measurement report](serve_prepend_lookback_20260808_measurement.md), the
[compressed summary](data/serve_prepend_lookback_189c5af_20260808/summary.json.gz),
and the [measurement script](measure_serve_prepend_lookback.py).

## The prototype produced no target recovery

| Result | Count | Interpretation |
| --- | ---: | --- |
| Reviewed broadcasts | 3 | `sset_01`, `sset_15`, and `sset_21` |
| ShuttleSet rallies | 292 | Full measured population |
| Any unmatched first stroke | 164 | Includes 28 whole-rally or unresolved misses |
| Issue 28 targets | 136 | First stroke unmatched, later stroke matched |
| Lookback opportunities | 344 | Claimed spans searched by the prototype |
| Raw candidate rows | 411 | Before evidence filtering |
| Evidence-passing candidates | 19 | Shuttle, court, and pose checks passed |
| Suppression-passing candidates | 18 | One candidate failed contact suppression |
| Evidence-only selected triggers | 14 | Largest surviving impulse per opportunity |
| Target first strokes recovered | 0 | No selected trigger recovered a target |
| All unmatched first strokes recovered | 0 of 164 | Fixed-span injection also recovered none |

The two tested pose bands produced identical pooled results. Widening the
central band added no useful candidate evidence.

For the evidence-only arm, target precision was 0 of 14 and target recall was
0 of 136. Both rates were 0.00%.

The 136 target outcomes explain why recovery was zero:

| Target outcome | Count | Share of 136 |
| --- | ---: | ---: |
| No raw candidate within canonical tolerance | 128 | 94.12% |
| No clean shuttle evidence at the nearby candidate | 6 | 4.41% |
| Court present, outside the prototype's intended case | 2 | 1.47% |
| Selected target match | 0 | 0.00% |

No target reached the central-pose selection stage within the canonical
matching tolerance. Pose-band tuning therefore cannot address the dominant
failure in this measurement.

## The evidence-only exemption introduced downstream risk

The committed mask selected no candidate. All 18 suppression-passing
candidates were on the definitive mask. The evidence-only sensitivity arm
cleared the mask only at selected frames and accepted 14 triggers.

Those 14 triggers recovered no target. They changed the stroke count for all
14 affected rows and changed the next-server result for 10 rows. These are
material downstream changes without measured target benefit.

The canonical human classes at those selected frames were:

| Human class | Selected triggers |
| --- | ---: |
| `cutaway` | 9 |
| `replay` | 3 |
| `live-non-standard` | 2 |
| `live` | 0 |
| `other` | 0 |

This class mix can be rechecked by the existing
[17-window annotation audit](serve_prepend_annotation_audit_runbook_20260809.md).
Any correction could refine the class description or mask interpretation. It
cannot change candidate matching, 0-of-136 target recovery, or 0-of-164
fixed-span recovery.

## Human review corrected the meaning of the target

Source-row validation found 26 of 136 target first rows with `flaw=1`. It found
25 target first rows with an unknown stroke type. Twenty-four `sset_21` targets
carry both signals.

These fields establish source-quality concerns. They do not prove that a
service was omitted by the broadcast. Conversely, an unflagged row can still
begin after an omitted service. The full source and transition evidence is in
[broadcast-omitted start and sting evidence](broadcast_omitted_start_and_sting_evidence_20260809.md).

The completed 32-row pilot reviewed all 26 flaw-marked targets plus six
deterministic workflow controls:

| Human decision | Reviewed rows |
| --- | ---: |
| Visible service contact | 19 |
| Broadcast-omitted service | 4 |
| Service action present but contact off-frame | 8 |
| Uncertain | 1 |

The pilot validates the four-state decision contract. It also proves that the
Issue 28 target contains several kinds of broadcast evidence. It does not
estimate their prevalence because the 32 rows were a quality stratum and
workflow controls, not a representative sample.

The reload-checked pilot counts and source hashes are recorded in the
[rally-start audit summary](data/rally_start_visibility_audit_20260809/summary.json.gz)
and [pilot report](rally_start_visibility_pilot_report_20260809.md).

## Why more rally-start review will not change Issue 28

The remaining 104 rows would give the exact composition of all 136 targets.
That would answer how many targets contain a visible service, an omitted
service, off-frame contact, or uncertainty.

These reviews answer a different question from the prototype test. They label
what happened at each target. They do not rerun the detector or change the 14
trigger frames that it selected.

None of those 14 triggers matched any of the 136 target first-stroke frames
within the canonical tolerance. If the completed review finds `N` visible
services, the visible-target recovery result will be 0 of `N`. Human review
can change `N`, which is the denominator. It cannot change the recovered count
from zero under the existing prototype result.

The target composition could still differ materially from the pilot because
the pilot was a quality stratum plus workflow controls. That composition may
affect whether a different detector is worth studying. It cannot turn this
prototype into a positive result.

That result would improve two later analyses:

- it would provide the correct visible-target denominator for reporting the
  prototype's recovery; and
- it would size the observed opportunity for a different future detector that
  requires a visible service contact.

It would not alter how the tested rule selected contacts or matched them to
the Issue 28 target. Filtering the 136 targets down to any reviewed subset can
remove false negatives from the denominator. It cannot turn one of the 14
selected non-matches into a target match.

The current evidence therefore supports closing Issue 28 before completing the
104 remaining reviews. If exact visibility composition is valuable for dataset
segmentation or a new detector proposal, that work belongs under a separately
approved Issue 32 decision. Do not complete the audit merely to make the Issue
28 evidence package larger.

## Remaining phases and their decision value

| Remaining work | Evidence it would add | Value for Issue 28 | Recommended status |
| --- | --- | --- | --- |
| Review 104 remaining rally starts | Full visible, omitted, off-frame, and uncertain composition | Cannot change the prototype's zero recovery or Issue 28 no-go | Park; resume only for a separately approved Issue 32 use |
| Review 17 candidate and practice windows | Rechecks human class and mask interpretation around measured candidates | Refines false-positive description only | Optional small audit |
| Run a replay-sting pilot | Tests whether match-local stings reliably bracket replay | None | Park unless a concrete Issue 32 need justifies a separate pilot |
| Audit all 179 replay intervals | Measures sting pairs, boundaries, negative cases, and post-replay state | None | Do not start without an accepted pilot result |
| Produce recording-only Phase 4 metrics | Aggregates completed visibility and sting truth | Only the visible-target denominator applies | Conditional on chosen Issue 32 audits |
| Make the Phase 5 production decision | Decides whether a replay-sting and partial-rally feature is justified | No effect on the serve-lookback no-go | Conditional on strong Phase 4 evidence |

Replay-sting work addresses a different problem. It may help preserve a
partial rally when a broadcast returns after play has started. The present
timelines show strong `other -> replay -> other` adjacency in `sset_01`, but not
in `sset_15` or `sset_21`. The current evidence does not authorise a full
179-interval audit or a production detector.

The [broadcast-start and replay-sting plan](broadcast_start_and_replay_sting_plan_20260809.md)
keeps this later work recording-only and under Issue 32.

## Recommended next steps

1. Merge the current PR as measurement, human-truth infrastructure, and a
   decision record. Keep production serve-prepend behaviour outside it.
2. Do not open a follow-on production PR for the Issue 30 prototype.
3. Post the measured no-go and precise target wording to Issue 28.
4. Close Issue 28 after the result and provenance are accepted.
5. Close Issue 30 because its conditional implementation was not justified.
6. Keep Issue 32 open for real rally-start semantics and replay-sting
   feasibility.
7. Park further human review until exact visibility composition or replay
   boundaries will inform a concrete dataset or production decision.

Any future serve-recovery PR should begin from a different evidence source and
an explicit span-reassignment design. It should cite this no-go result rather
than continue the failed Issue 30 prototype.

No production serve-prepend, replay-mask, segmentation, winner, attribution,
or pairing behaviour should change from this result.

## Further questions

- Does the project need exact visibility composition for the 136 targets, or
  is the four-state pilot sufficient for current dataset documentation?
- Is partial-rally continuation after replay important enough to justify a
  small, declared replay-sting pilot?
- If another serve detector is proposed, what new evidence source and span
  reassignment rule distinguish it from the failed prototype?

## Caveats and assumptions

- The result covers three reviewed ShuttleSet broadcasts and the explicit
  `une-189c5af-static-stride8` fixture profile.
- It rejects the measured raw-impulse plus central-pose trigger. It does not
  test every possible visual, temporal, or learned serve detector.
- Counterfactual injection kept existing spans fixed. A new span-reassignment
  design would require separate measurement.
- The 32 reviewed event rows are not a prevalence sample.
- A "false positive" in this brief means a selected trigger that did not
  recover an Issue 28 target. It does not claim that every trigger was an
  imaginary shuttle contact.
- The 17-window candidate audit and 104 remaining rally-start reviews are not
  complete.
- Replay-sting feasibility remains unmeasured against source-video event truth.

## Evidence index

- [Issue 28 measurement and decision](serve_prepend_lookback_20260808_measurement.md)
- [Issue 28 compressed evidence pack](data/serve_prepend_lookback_189c5af_20260808/)
- [Measurement entry point](measure_serve_prepend_lookback.py)
- [Shared measurement definitions](../../../src/annotator/calibration/serve_prepend_measurement.py)
- [Broadcast-omitted start and replay-sting evidence](broadcast_omitted_start_and_sting_evidence_20260809.md)
- [Rally-start pilot report](rally_start_visibility_pilot_report_20260809.md)
- [Rally-start package and hashes](data/rally_start_visibility_audit_20260809/)
- [Rally-start review runbook](rally_start_visibility_audit_runbook_20260809.md)
- [Candidate and practice audit runbook](serve_prepend_annotation_audit_runbook_20260809.md)
- [Issue 32 phased plan](broadcast_start_and_replay_sting_plan_20260809.md)
- [Issue 32 update draft](issue_32_rally_start_replay_sting_update_draft_20260809.md)
