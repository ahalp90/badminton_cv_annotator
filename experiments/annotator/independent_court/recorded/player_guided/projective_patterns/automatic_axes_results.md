# Automatic court directions: the selector loses useful precision

The automatic detector needs a direction-selection rethink. Two label-guided
controls found precise directions among the image-derived candidates that the
automatic selector discarded. The unchanged spacing matcher produced close
courts when given those directions. Expanding the automatic court pool exposed
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
Two additional controls selected observed directions using reference geometry;
these controls are explicitly label-guided.

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

The [comparison gallery](automatic_axes_visual_check.html) separates automatic
winners from reference-selected diagnostic courts. These new panels remain
visually unjudged; earlier approvals belong to their original panels.

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

## Decision and checks

Score tuning cannot restore excluded directions. The existing samples already
expose these failures, so no new annotations are needed yet. Production and
annotations remain unchanged.

Validation passed: 22 synthetic tests, scoped lint/type checks, exact final-score
replays for all 18 rescoring cases, and original global-selection replay for all
nine views. Technical reviews checked the generator and shared evaluation. The
[methods](methods.md) record settings, limits and numerical tolerances.
The [measurement summary](measurements.json.gz) preserves all three automatic
comparison arms and the two label-guided direction controls.
