# Spreading the seed sample across line-pair positions

Keep the original sampler. The deterministic sample finds closer generated
courts on five of seven GX frames, but accepts no GX court. It also loses an
accepted amateur control and worsens the selected broadcast court. The experiment
shows that seed choice affects coverage; this particular rule is not a replacement.

## Comparison

The question is whether a sample spread across line-pair positions can preserve
closer proposals at the existing budget. Both arms sample 4,096 convex seed
quadrilaterals from the same cached line fragments. The original random arm uses
its unchanged random seed, 20260907. The deterministic ("spread") arm groups quadrilaterals by
their line pairs' positions and separations, then samples those groups in turn.

For each family, line indices follow the original image-centre ordering. A
pair's midpoint and separation are normalised by the family's index range.
The resulting four coordinates each use eight equal bins. Within each occupied
bin, candidates nearest its centre come first using computed float64 distances;
original enumeration breaks equal computed distances. Selection visits occupied
bins in ascending lexicographic order once before taking their second candidate,
and continues until the budget is filled. This spreads coverage in line-index
coordinates, rather than uniformly in image pixels.

Both arms then apply the original 100-square-working-pixel area check without
refilling rejected samples. They retain the same court assignments, player
rule, geometry check, floor score, camera check and candidate retention.
The floor scorer uses the original image-angle families. This experiment does
not combine seed selection with the earlier directional-matching change.

The rule was specified before evaluating the new sample. Frames 0 and 5 formed
the pilot; its frame-5 coverage gain triggered the unchanged five-frame extension.
All seven frames come from the same amateur video. Reference annotations enter
after generation to measure coverage; they never choose seeds. These are
development diagnostics, not held-out accuracy results.

## GX results

Errors below use 1280×720 coordinates. Each column is the minimum over the
geometry-valid generated population, before player filtering. Maximum corner
error includes all four annotated corners, including extrapolated ones. Boundary
RMS here is the existing zero-inset boundary proxy. The two metrics can choose
different proposals. Neither metric has a newly defined usable-court cutoff.

| GX frame | Random: max corner | Spread: max corner | Random: boundary RMS | Spread: boundary RMS |
| --- | ---: | ---: | ---: | ---: |
| 0 | 27.24 | 27.24 | 7.12 | 7.12 |
| 5 | 220.52 | 25.25 | 78.25 | 6.64 |
| 689 | 21.99 | 13.23 | 5.57 | 8.33 |
| 5111 | 166.17 | 18.69 | 38.12 | 6.55 |
| 5766 | 37.86 | 35.55 | 10.20 | 8.26 |
| 77876 | 131.89 | 173.15 | 57.17 | 56.69 |
| 86088 | 216.16 | 46.94 | 19.45 | 19.86 |

Maximum corner error improves on 5/7 frames, ties on 1/7 and worsens on 1/7.
Boundary RMS improves on 4/7, ties on 1/7 and worsens on 2/7. The disagreement
on frames 689 and 86088 is a reason to retain both measures.

The random arm has one floor survivor on frame 0 and none elsewhere. The spread
arm has one on frame 0 and one on frame 5. All three are rejected by the actual
camera check. Both arms therefore retain and accept zero courts on all seven
frames. This floor-survivor count does not identify those proposals as correct.

The spread sample's closest frame-5 proposal has maximum corner error 25.25 px
and paint-inset boundary RMS 6.56 px. It passes the player rule but fails floor
support: lengthwise mean 0.4236 and two distinct lines, against required values
0.55 and three. A separate diagnostic camera check passes at 0.0894, below 0.1.
The detector never reaches that camera check for this proposal because floor
support rejects it first. This is still a reference-selected example, not an
automatic output. The earlier complete-search example remains closer at 3.37 px.

All runs use the same 4,096 pre-area budget. The random arm retains 4,060–4,077
rectangles after area filtering; the spread arm retains 4,046–4,061. This slight
difference is a consequence of the unchanged area gate. Isolated pilot detection
times were 39.8–41.9 seconds for random and 39.5–39.7 seconds for spread. The
five-frame extension ran concurrently, so its timings are recorded separately
in the saved results rather than treated as a precise speed comparison.

## Controls and checks

Both previously accepted generator controls regress. Selected-court error below
is the maximum corner error of the detector's first retained court, measured at
1280×720. It is an actual output measure, unlike the coverage minima above.

| Control | Random accepted / retained | Spread accepted / retained | Random selected error | Spread selected error |
| --- | --- | --- | ---: | ---: |
| Amateur-2 frame 150 | Yes / 1 | No / 0 | 12.30 px | No output |
| Broadcast scene 0019 | Yes / 23 | Yes / 6 | 10.31 px | 54.02 px |

The amateur control loses all three floor survivors. Its best generated maximum
corner error also worsens from 11.71 to 57.80 px. The broadcast result shows why
acceptance alone is insufficient: both arms accept, but the selected geometry
worsens substantially. Its reference has corners only, so this control uses
direct detector outputs without boundary or generated-population summaries.
An initial broadcast run failed in boundary reporting; the direct-output rerun
completed successfully. The amateur result from that first batch remains valid.

The random arm reproduces the original seven GX traces' saved counts and final
outputs. The unchanged tracer also checks that instrumentation preserves each
run's final output. Synthetic checks cover original random selection, uncapped
behaviour and selected-rectangle identities. Scoped lint, synthetic checks and
the completed comparison runs passed, exit 0.

Independent Opus review covered the selector, runner and two-frame pilot. It
found no error that changes those saved outcomes. A concrete numerical example
confirms that mathematically equal bin-centre distances can differ in the last
floating-point bit. The archive preserves the tested float64 rule, rather than
silently substituting exact-arithmetic tie handling. Ascending bin order also
favours earlier bins in the final partial round. These are limitations of this
specific sample, not evidence for uniform coverage. Extension and control runs
were outside the review's scope.

The [replay archive](seed_selection.zip) contains the tested scripts, nine saved
case comparisons and instructions using the existing frozen input archives.

## Decision

Do not adopt this sampler: the control regressions outweigh the partial GX
coverage gains. Keep the original detector and the saved comparison. Visual
usability of the new closest frame-5 proposal remains unconfirmed; the earlier
approval applies only to the complete-search court. That visual check can guide
the next search experiment, but cannot remove the control regressions here.
Floor-evidence rejection remains a separate obstacle to accepted GX courts.
