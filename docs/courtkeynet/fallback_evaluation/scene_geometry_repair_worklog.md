# Court geometry repair — investigation trail and next steps

The repair resolves the reported geometry failures, but repeated scenes from the
same camera view still receive separate court estimates. **The next step is to
group matching views and establish a shared calibration.** Otherwise, variation
between estimates can introduce unwanted coordinate drift in extracted positions
and features.

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

At the final source revision, pytest passed with 2,104 tests passed and 29
skipped (exit 0). Whole-project Ruff and Pyrefly remained non-zero (exit 1),
with 905 existing lint findings and 11 existing import-resolution errors
respectively. Checks found no new findings attributable to the changed code.

## Next: group matching camera views

Scene-specific geometry is necessary for different views, but it permits small
differences between estimates of the same view. Current donor repair does not
give every matching scene a shared final court. The resulting coordinate drift
has not yet been measured or resolved.

Group scenes that share the same view, zoom and crop, then estimate a stable
court from their combined evidence. Validate that shared court against each
scene before using it for extraction. A common physical camera is insufficient
if its framing changes. Evaluation should measure consistency of projected
positions within a group while checking that distinct views remain separate.

Regenerate court-dependent features after this work. Tree-model retraining and
tuning can then be evaluated against a stable geometry baseline.

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
