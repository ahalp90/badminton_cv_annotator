# Court geometry repair — investigation trail and next steps

The geometry repair and subsequent [scene grouping](scene_grouping.md) are
implemented. The follow-up retains usable fallback candidates through final
validation. A comparison on original ShuttleSet supports keeping the shared median.
Frozen evaluation on three fresh broadcasts is complete; it identifies remaining
zoomed-court errors and missed grouping.

This companion records useful failed approaches and decisions from issue #148.
The [results report](scene_geometry_repair.md) contains the final implementation
summary, comparison protocol and measurements.
The [evidence bundle](../../../experiments/annotator/court_geometry_repair/README.md)
provides the selected records and runnable checks behind them.

## What the investigation ruled out

### Rejecting malformed courts alone would miss the recovery

In ShuttleSet22 video 53, scene 334, the fallback selected a weak diagonal as the
far baseline. The initial fit already put the top-right corner far below its
correct location. Refinement retained the mistake despite an acceptable fitting
residual: proximity to the selected lines did not establish a valid court.

Correcting boundary selection recovered a usable outline on the same frame.
The recovered corner was about 12.7 native-image pixels from the low-confidence
neural estimate, versus 712.8 pixels before. These distances describe agreement
with that estimate, **not ground-truth error**. Rejecting the malformed final
outline would have stopped bad geometry reaching later stages, but would have
left this court unrecovered.

### Keeping model courts was necessary but insufficient

Video 17's valid opening view occupied 73,167 frames. The old consensus gave each
scene one vote, allowing many short scenes from another framing to replace it.
Visual inspection confirmed that both framings were valid. A single court for
the whole video could not represent them.

The first scene-aware prototype preserved that opening court, but also accepted
false courts on close-ups and misaligned zoomed views. In video 53, 74 of its 75
new acceptances showed the full painted court; the remaining scene was a player
close-up. A plausible quadrilateral containing two people was insufficient
evidence of a court.

The prototype also preferred fully model-derived donor courts whenever any were
available. A synthetic check showed one such donor overriding three consistent
fallback donors. That preference was removed. Compatible donors now vote
together, and ambiguity leaves the original estimate unchanged.

Two matching corner anchors do not uniquely determine the camera projection.
Donor agreement therefore needs confirmation from the target image. Repaired
courts are also kept out of the donor pool: allowing inferred geometry to become
new evidence could spread a bad repair and make results depend on scene order.

### Painted-line coverage supplied the missing evidence

A pilot compared raw and borrowed courts on 26 selected scenes. It tested line
matching tolerances of 5, 10 and 15 pixels at 1280×720 reference resolution.
At 10 pixels, the false video-53 close-up covered only 5% and 2% of the expected
lines in the two court directions. A useful repair in video 53, scene 223 improved those
fractions from 64% / 47% to 91% / 96%.

There was an important exception: video 17's correct, fully model-derived
opening court scored 48% / 58%, with weak outer sidelines. The 50% requirement
therefore applies to fallback and borrowed courts, not fully model-derived
courts. Extending it to every detection would need separate validation.
These development examples informed the rule; they do not establish a universal
threshold for every view or lighting condition.

One rejected review proposal would have retained the original court whenever
the borrowed court failed its fresh person check. Inspection of the synthetic
example showed why that could be wrong: the original outline enclosed people
below the actual court. Keeping an acceptance is not useful when its geometry
misidentifies the playing area.

### The first full pipeline run exposed a replay dependency

With separate camera views preserved, video 17 initially collapsed from 38 raw
rally spans to two, before the learned selectors ran. The replay logic treated
every later accepted view as a perspective change from the dominant opening
court. Combined with court-absence exclusions, it masked the entire remainder
of the video. Whole-video court replacement had previously hidden this fault.

Removing camera difference from both the replay veto and its speed-reference
exclusions restored all previously correct rally sections with the same frozen
models. Camera difference alone cannot distinguish alternate live coverage
from replay. A proposed rule based on returning to the opening view would have
made the same unsupported assumption. Replay detection still needs broader
evaluation, including changes to the slow-motion reference.

## Model status and verification

**No tree models were retrained or tuned for the recorded results.** Contact
and rally-selection models kept their existing weights and settings. The neural
court detector was also unchanged. Geometry acceptance rules were refined during
development, including the painted-line coverage threshold.

Court-dependent annotations and features had to be regenerated before scoring;
rescoring saved annotations alone would miss the effect of the repair. The
reconstructed baseline reproduced the saved contact and final-section records
on both videos. Static ShuttleSet homographies were used only for matching
views; visual inspection covered camera changes and false acceptances.

At the original repair's final source revision, pytest passed with 2,104 tests passed and 29
skipped (exit 0). Whole-project Ruff and Pyrefly remained non-zero (exit 1),
with 905 existing lint findings and 11 existing import-resolution errors
respectively. Checks found no new findings attributable to the changed code.

## Follow-up status

Scene grouping now shares a median calibration across image-aligned repeated
views. The two development videos retain identical final contact/rally records
with frozen models. Existing player-feature inputs were regenerated; the
[grouping report](scene_grouping.md) records their coordinate changes and limits.

1. Completed: retain usable fallback alternatives through final validation. Review of
   revision `c9dc6ac` identified that the lowest-residual fit survives even when
   an alternative has better finite painted-line coverage. A repository-function
   probe confirms the mechanism: the distorted candidate matches only two lines
   and has a lower residual than the usable twelve-line fit. The fix passes the
   full test suite and independent review. It changed no final courts in the
   three fresh broadcasts below; a natural recovery case remains unobserved.
2. Completed: evaluate three fresh broadcasts with frozen thresholds. Accepted
   courts and masks remain unchanged by the candidate fix. Visual checks found
   existing calibration errors and conservative grouping, detailed below.
3. Completed on original ShuttleSet: compare the shared median against a fit
   using pooled visible court lines. Keep the median: the pooled fit worsened
   reference-corner accuracy on both controls. Details follow below.
4. Initial assessment complete: identify small court-space feature candidates.
   Their coverage and predictive benefit remain unmeasured. Review those results
   and tradeoffs before authorising any tree retraining or refitting.

Two small tree-feature candidates are player ground-anchor speed in court
coordinates and distance from the centre of the player's half-court. Existing
image-space features would remain the comparison. First measure finite coverage,
interpolation and out-of-court positions using exact video/frame joins. A floor
homography does not recover airborne shuttle position or physical shuttle speed.
Feature selection should use the original training data; keep fresh broadcasts
for evaluation. ShuttleSet22 is the test set: it supplies no learned-model fitting,
tuning or feature selection. Shared-fit development also uses original ShuttleSet;
the completed SS22 pass evaluated the fixed detector. The queued SS22 line-fit
comparison was cancelled before it ran. Acceptance and grouping thresholds remain fixed.
No feature implementation or tree fitting has started.

Original ShuttleSet is the preferred source for geometric development. A targeted
SS22 exception may be useful for a meaningfully different failure mode absent from
the original data. Record the reason and label those examples as development.
This exception does not permit learned-model fitting, tuning or feature selection.

Original ShuttleSet video 3 already supplies real geometry failures: seven scenes
mistook advertising-board edges for the far baseline. Video 21 is the clean
comparison. These came from neural inference and fallback recovery, not injected
bad corners. Official matching-view corner references make them suitable for
assessing shared-fit accuracy. The earlier use of SS22 videos 17 and 53 remains
labelled as development; it cannot retrospectively become untouched evaluation.

The review also identified two limits without demonstrating real-video regressions.
A low-confidence model endpoint can steer the ten-degree boundary-direction filter
away from a correct line; that filter also excludes distracting diagonals. Keep
the present filter until a targeted comparison supports a change. Normal-speed
replays with valid courts require evidence beyond camera novelty and shuttle speed.
Repeated-view hashes alone do not identify repeated action.

## Follow-up findings

The candidate-retention fix tries the remaining locally valid fits only after
preferred-court acceptance and donor repair fail. Each alternative must pass the
existing painted-line and fresh person checks. Raw diagnostics continue to describe
the preferred fit; saved active corners describe the accepted court. No acceptance
threshold changes. A regression test reproduces the lost-candidate failure and
checks recovery, preservation of a passing preferred court and person rejection.
The full suite passes: 2,114 passed, 29 skipped (exit 0). Whole-project Ruff and
Pyrefly retain the same 905/11 existing findings (exit 1).

The pooled-line comparison used original ShuttleSet videos 3 and 21. Existing
image-grouping rules found groups of 39 and 35 scenes. Five scenes per group
supplied both the comparison median and the pooled fit; the remaining 34/30
supplied independent line checks. All inputs passed local geometry checks;
fallback inputs also passed painted-line checks. Donor-derived courts were excluded.
This experiment measures geometry. It does not reproduce the production
person-acceptance gate.

| Original video | Five-scene median corner error | Pooled-fit corner error | Held-out line residual, median → fit |
| --- | ---: | ---: | ---: |
| 3 | 4.62 px | 6.73 px | 6.26 → 6.91 px |
| 21 | 4.62 px | 4.94 px | 9.10 → 2.87 px |

The production medians use all group members; their corner errors are 4.63/4.64 px
for videos 3/21. The table uses five members for a fair comparison with the fit.
All distances use 1280×720 reference pixels. Corner error is the mean distance
of the four corners from the official matching-view reference. Four spread-out
members per group were visually checked against that reference; the remaining
members passed alignment to their group representative. No reference-accuracy
claim applies to ungrouped views. Both candidates were scored against the
same fixed line assignments, so dropping difficult lines cannot improve the score.

Keep the current median. Better agreement with detected lines in video 21 did
not improve corner accuracy, and video 3 worsened on both measures. Two groups
are a bounded comparison, not proof that every pooled-fit approach would fail.
The production shared calibration remains unchanged.

## Frozen fresh-broadcast evaluation

ShuttleSet22 videos 8, 9 and 10 were selected as the first three eligible downloads,
before inspecting their results. They cover the India Open, German Open and All
England broadcasts. Neural models, tree models, acceptance thresholds and grouping
rules stayed fixed. Each before/after comparison used the same freshly decoded
frames, neural detections, pose evidence and image summaries.

| SS22 video | Scenes | Accepted scenes | Shared group sizes | Retained alternatives |
| --- | ---: | ---: | --- | ---: |
| 8 | 636 | 90 | 55, 29 | 20 |
| 9 | 317 | 66 | 17, 42 | 18 |
| 10 | 375 | 64 | 36 | 8 |

Across 1,328 scenes, retaining alternatives changed no final court records or frame
masks. Complete saved court payloads match after excluding `case_id`, which contains
the before/after comparison label; both masks match exactly. All payloads pass the strict loader. Consequently,
this pass establishes regression coverage but adds no observed real-video recovery.
Downstream contact/rally scoring was not repeated for these unchanged court inputs.

Visual checks covered one midframe for every accepted scene (220 total) and every
rejected scene with a raw quadrilateral (286 total). No obvious zoom/crop mixing
was seen within the shared groups. Some accepted model-only zooms still have
misplaced edges, especially video 8 scene 367. These predate candidate retention
and remain ungrouped. Model-only courts retain their painted-line-check exemption.
Visible courts can also fail the person gate: video 9 scenes 41/97 have two-person
fractions of 0/0.019. This is a scene-acceptance limitation, not absent court evidence.

Grouping is conservative. Some selected separate views align within one native pixel,
including the first members of video 8's two groups and several ungrouped video 9
scenes. Other pairs show little movement but fail the image-correlation threshold.
Pair compatibility does not guarantee a shared group: all-pair hash bounds,
representative alignment and final court checks also apply. No thresholds were
changed in response to these test-set findings.

This is a bounded image audit. Scenes without a raw quadrilateral were not inspected,
so overall recovery recall is unmeasured. Midframes do not establish calibration
accuracy throughout a moving shot. The findings support targeted investigation on
original ShuttleSet; they do not justify calling the detector generally accurate.

## Next work

1. Find original-ShuttleSet examples of misplaced, confidently model-derived courts,
   particularly zoomed or partial views. Assess a correction against good model-only
   views whose painted lines are weak. Applying the fallback coverage gate to every
   model court could discard useful views.
2. Trace missed grouping on original data. Keep the current rules until a bounded
   comparison can improve repeated-view coverage while preserving zoom separation.
3. Measure coverage of player court-space speed and half-court-centre distance on
   original training data. Review interpolation, missingness and geometry errors
   before deciding whether a tree-fit experiment is worthwhile. Tree fitting still
   requires an explicit check-in.

Boundary-direction filtering, normal-speed replay recognition and general moving-
camera recovery remain separate work. The pooled shared-line fit stays deferred
because the tested version worsened reference-corner accuracy.

## Explore usefulness on partial courts

The current fallback needs at least two confident neural corners, enough visible
line evidence and a static camera within the scene. Its corner-order and shape
checks also constrain strongly oblique views. Recovering missing corners under
these conditions does not establish reliable detection of arbitrary artistic
or partial-court shots.

Evaluate fixed partial views, strongly angled views and moving-camera shots
separately. Measure both geometry accuracy and how many useful scenes survive
the acceptance rules. Each view needs appropriate ground truth or visual
assessment; the official static homography cannot describe every camera setup.
Grouping matching views may help repeated partial views, but its usefulness
there remains to be tested.
