# Marking follow-up: three good ShuttleSet fits, unresolved identity failures

User inspection confirms good new fits on ShuttleSet03 scenes19/16 and
ShuttleSet21 scene20, with small paint-edge differences. Scene17 still misses
the farther horizontals, now overshooting. GX fits hall girders and a floor line;
both Amateur-2 fits are disasters. Amateur-3 is only marginally improved.
[Exact visual verdicts](marking_diagnosis_visual_judgements.md)
apply to the right-hand panels in the [gallery](marking_followup_visual_check.html).
Production and acceptance rules are unchanged.

## What was tested

The follow-up covers the same nine development views: two GX frames, three
Amateur-2/3 frames and four scenes from ShuttleSet 03/21. ShuttleSet inputs are
cached median composites. No sample is a designated holdout.

First, finite-fragment matching and two fixed-assignment refit variants examined
216 saved courts. These include reference-selected diagnostics, kept separate
from automatically retained courts. Refitting freezes each fragment's marking
identity and fits one perspective transform across its selected samples.

Next, a continuous distance score scanned the same 16,639,800 generated courts.
It scored all 3,106,429 courts that passed the original geometry/player conditions.
A greedy shortlist kept 128 per case for detailed matching and refitting.
Starts and fixed-position refits were ranked separately by finite stripe support,
with original geometry/player/camera checks. Floor failure remains visible;
this experiment does not apply a new acceptance decision. References measure
results afterwards and never choose an automatic candidate.

## What the evidence shows

Each generated court starts from four observed lines, called its seed.
All three original ShuttleSet 03 top courts use the near long-service and back
lines, only 0.76 m apart, to determine the lengthwise fit. Amateur-3's top court
uses a 0.46 m by 0.76 m seed region. These fits extrapolate to the far court.
The original scorer gives the ShuttleSet courts over 0.97 floor support, yet
finite matching finds weak far service evidence. Both far service intervals
contribute no fitting samples to those three top courts.

The new automatic refits improve finite support on both far service markings.
These are continuous coverage scores from 0 to 1, not measured accuracy or visual
approval:

| ShuttleSet 03 scene | Far long service: old → new | Far short service: old → new |
| --- | ---: | ---: |
| 17 | 0.34 → 0.92 | 0.44 → 0.94 |
| 19 | 0.24 → 0.89 | 0.38 → 0.88 |
| 16 | 0.30 → 0.94 | 0.39 → 0.92 |

Scene17's far-back coverage falls from 0.88 to 0.46. The user's visual rejection
confirms that better service support did not establish a better court.

The broader failures are clear. The coarse shortlist loses the approved GX5
court and the close Amateur-2 frame-28019 example. GX's original angle rules put
its roughly 7-degree right sidelines in the cross-court family. Its near baseline
has the opposite family problem. The continuous shortlist retains that limitation.
On Amateur-2 frame 150, a close start remains available, but refit ranking chooses
a different interpretation with 188 px maximum corner error at 1280×720. That
winning parent was already badly displaced before fitting. The tested average-score
ranking does not identify the correct court.

## Decision and checks

The visual check is complete. GX's family restriction and the rejected structural
and matting interpretations motivated the subsequent [spacing matcher](axis_matching_results.md).
That experiment begins with synthetic and supplied-direction tests, retaining
the three newly approved ShuttleSet fits as controls. The ranking tested here
has no demonstrated general replacement value.

All three experiment runs completed, exit 0. Every case reproduces the frozen
population counts, and saved seed corners reconstruct within 0.00001 native px.
The automatic arm saved 2,304 refit attempts: 1,702 converged; 602 failed the
existing solver/projection checks. Three synthetic tests, scoped Ruff/Pyrefly and
browser overlay controls pass, exit 0. Independent adapter review found no
confirmed correctness defect.

The [methods and validation](methods.md) record the settings and limits.
