# Automatic court directions: the selector loses useful precision

The automatic detector needs a direction-selection rethink. Three label-guided
controls found precise directions among the image-derived candidates that the
automatic selector discarded. The unchanged spacing matcher produced close
courts when given those directions. The latest GX0 control has visual approval:
both winners are essentially ideal. Expanding the automatic court pool exposed
a separate problem: paint ranking can prefer grossly displaced courts.

The next design step follows from that finding. Preserve or refine direction precision first,
then test ranking against the saved false winners. The existing samples expose
both problems; production promotion and final holdouts are premature.

This stage asked whether automatic directions could replace the supplied
directions in the previously inspected spacing matcher. The larger goal is
reliable single-frame court geometry across varied amateur and broadcast video.

## What was compared

All nine views are development data from five videos: two GX frames,
two Amateur-2 frames, one Amateur-3 frame and four scenes from ShuttleSet 03/21.
ShuttleSet inputs are cached median composites. There are no designated holdouts.
Manual references were used only after
automatic generation, to measure errors and diagnose losses.

Each view used the same frozen selector's 16 vanishing points: the image points
where parallel lines converge. All 240 ordered direction pairs were
considered, subject to the existing camera bound. The spacing matcher kept up
to 256 courts per pair. Three comparisons then used those saved candidate pools:

1. Original: keep 256 courts globally, then apply the camera check.
2. Camera first: apply the camera check before keeping 256 globally.
3. All camera-eligible: remove the global cap and its two-working-pixel diversity
   suppression. Keep the per-pair limits unchanged.

These are nine generation runs plus 18 rescoring runs.
The final comparison scored 19,286 camera-eligible candidates across nine views.
Three additional controls selected observed directions using supplied court
geometry, including a subsequent GX0 check against a visually approved candidate.
These controls are explicitly label-guided.

The result metric below is maximum corner distance from the manual reference,
in pixels at 1280 × 720 display size, allowing a 180-degree corner relabelling.
The metric is descriptive. There is no numerical success threshold, and the
measurements confer no visual approval. Winners pass the camera check and have an available paint score.
The line winner maximises complete-line support; the paint winner maximises
paint-profile agreement, breaking ties with line support. Existing floor gates
remain recorded but do not select these diagnostic winners. None is a detector
acceptance or emission decision.

## Results across all nine views

The table compares original winners with the all-camera-eligible winners.
Each cell is line / paint corner error in display pixels; smaller is closer.

| View | Original | All camera-eligible |
|---|---:|---:|
| GX frame 0 | 18.8 / 12.7 | 18.8 / 12.7 |
| GX frame 5 | 683.2 / 699.4 | 705.5 / 704.4 |
| Amateur-2 frame 150 | 185.1 / 10.8 | 185.1 / 10.8 |
| Amateur-2 frame 28019 | No eligible winner | 193.5 / 1189.5 |
| Amateur-3 frame 0 | 7.7 / 11.3 | 7.7 / 11.3 |
| ShuttleSet 03 scene 17 | 11.1 / 10.2 | 11.1 / 10.2 |
| ShuttleSet 03 scene 19 | 12.7 / 12.7 | 12.7 / 8390.9 |
| ShuttleSet 03 scene 16 | 8.6 / 11.3 | 8.6 / 11.3 |
| ShuttleSet 21 scene 20 | 7.2 / 7.2 | 7.2 / 7.2 |

Moving the camera check earlier gives 256 eligible candidates in every view.
It recovers an Amateur-2 frame-28019 paint winner at 22.9 display pixels.
GX frame 5 (GX5) remains badly displaced: line / paint errors are 705.5 / 588.8 pixels.
Removing the global cap then introduces the extreme paint winners shown above.

## Visual inspection

The [comparison gallery](automatic_axes_visual_check.html) has now been inspected
across all nine views. [Exact panel judgements](automatic_axes_visual_judgements.md)
identify usable line-score winners on five views and usable paint winners on four.
Six views have a usable winner under at least one ranking: Amateur-2 frame150,
Amateur-3 and all four ShuttleSet views. That combined count requires choosing
between rankings after inspection; it is not an automatic selection success rate.

GX0, GX5 and Amateur-2 frame28019 have no usable displayed winner. Their nearest-
control diagnostics are also poor. GX0's visible shear remains unacceptable even
though its measured corner errors are relatively small. The user prefers its
line winner to the numerically closer paint winner. Amateur-2 frame150's paint
winner is described as perfect, with far-end uncertainty; scene19's paint winner
is rejected as unrelated geometry. A global preference for either score therefore
remains unsupported.

Approvals apply to these exact candidates. Earlier supplied-direction panels and
the observed-bank controls are separate results. Apparent curvature and the
suspected post/person correspondences remain visual hypotheses about the cause.

## What the controls establish

For GX5, the closest saved candidate to the previously inspected court is
23.66 working pixels away before camera filtering, and fails that check.
Even without global pruning, the closest camera-eligible candidate is 89.94
working pixels away. Here working images are 960 × 540. Global retention loss
alone therefore cannot explain GX5's failure.

The full image-derived direction banks contain 5,886 GX5 candidates and 8,256
Amateur-2 frame-28019 candidates. Reference-guided selection identifies precise
direction pairs discarded by the selector: GX5's were excluded by the 16-point
cap; Amateur-2's were marked redundant. Local fits to the supplied control geometry have maximum corner distances of
0.19 and 0.16 working pixels respectively. Those fits are diagnostic optimisations,
not generated courts or certified global optima.

Feeding those observed directions into the unchanged matcher produces actual
courts with maximum corner distances of 2.38 and 1.64 working pixels from those
fitted controls. GX5's line
and paint winners both measure 10.58 display pixels from the manual reference.
Amateur-2's paint winner measures 7.27 pixels; its line winner remains poor at
193.50 pixels. This demonstrates useful matcher capacity with precise inputs,
not automatic recovery of those inputs.

Paint ranking has two distinct failure modes in the larger pool. Amateur-2's
bad frame-28019 winner passes all 11 paint profiles. ShuttleSet 03 scene 19's
bad winner gets a perfect score from only five visible markings; six are
unavailable. Both extreme winners fail the original floor gate. These examples
justify preserving the gates and checking wrong-court and background candidates
before changing ranking or acceptance.

## GX0 follow-up: both control winners are visually approved

The rejected automatic GX0 fits appeared sheared. A bounded follow-up tested
whether the observed direction bank contained enough precision to recover the
previously approved supplied-direction line winner89. This comparator is an
actual generated court, distinct from the manual reference.

Local diagnostic fits tested all 240 retained ordered direction pairs. The best
tested fit was 6.798 working pixels from approved89 at the worst corner. Selecting
four directions per axis from all 5,671 observed candidates gave sixteen further
fits. The best pair, 1183/122, reduced that distance to 0.509 working pixels.
Direction1183 had been excluded as redundant; direction122 hit the 16-direction
cap. These fits used the approved court to select directions and are not certified
optima. Working dimensions are 960 × 540.

One run of the unchanged matcher then used that observed pair. It retained 256
courts, of which 89 passed the camera check. In the
[inspected comparison](gx0_control_visual_check.html), approved89 is on the left
of each row; the two actual matcher winners are on the right:

| New GX0 control | User judgement | Maximum corner distance from approved89, 1280 × 720 pixels |
| --- | --- | ---: |
| First: line winner1864 | Essentially perfect; indistinguishable from the approved court apart from a very mild bottom-right inset to the white-line midpoint | 3.121 |
| Second: paint winner5144 | Perfect | 4.050 |

The user judged both essentially ideal. This establishes visually usable GX0
matching from observed directions that automatic selection discarded. The
selection used approved geometry, so automatic recovery remains unresolved.
Both winners retain floor rejection, as does approved89. The automatic gallery's
six-view count and its rejected GX0 fits remain unchanged.
[GX0 measurements](gx0_control_measurements.json.gz) preserve the comparator,
direction identities, diagnostic fits, winner coordinates and judgements.

## Decision and checks

Score tuning cannot restore excluded directions. The GX0 follow-up strengthens
the case for preserving or refining direction precision: its observed bank now
produces visually approved courts through the unchanged matcher. Test that next
design automatically across the existing varied amateur and ShuttleSet cases.
Keep ranking and acceptance as separate checks. No new annotations are needed
yet; production and annotations remain unchanged.

Validation passed: 22 synthetic tests, scoped lint/type checks, exact final-score
replays for all 18 rescoring cases, and original global-selection replay for all
nine views. Technical reviews checked the generator and shared evaluation. The
[methods](methods.md) record settings, limits and numerical tolerances.
The [measurement summary](measurements.json.gz) preserves all three automatic
comparison arms and the first two label-guided direction controls. The separate
GX0 diagnosis and matcher both completed with exit 0. Scoped lint/types, source
provenance checks and gallery controls passed; no algorithm or gate changed.
