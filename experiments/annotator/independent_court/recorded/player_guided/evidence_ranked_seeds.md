# Covered-length rank improves sampling, but misses the approved court

Keep this as a diagnostic result, not a replacement sampler. Prioritising
longer-supported lines improves the closest generated court on all seven GX
frames versus random sampling and preserves both accepted controls. It still
accepts no GX court. The user judged the new frame-5 example a disaster compared
with the far superior approved complete-search court. Lower error has not
established usable recovery.

## What was tested

The goal is to recover useful courts in partial amateur views without losing
already-good outputs. The preceding simple spread sample improved some coverage
but regressed on both controls. This test adds one existing evidence cue: each
merged line's rank by covered fragment length. Rank zero has the most support;
rank is not the measured length itself, nor proof that a line is court paint.

The original merged lines already arrive in that order. Each seed rectangle's
cue is the mean normalised rank of its four lines. Within the previous selector's
position/separation bins, this cue is blended with distance from the bin centre.
The three predeclared weights are 0.25, 0.5 and 1.0; larger weights favour the
length rank more. At weight 1, rank alone orders candidates within each bin.
The bin order still preserves a spread of line-pair positions and separations.

All arms keep 4096 seeds before the original area check, without refilling
rejected seeds. Families, merging, court templates, player rules, floor scoring,
camera checks and retention remain unchanged. The original random and simple
spread runs are the comparators. Reference labels measure results only after
generation; they never choose the sampler's seeds.

This is a development comparison: seven GX frames from one amateur video, plus
one previously accepted amateur and one broadcast control. A two-frame pilot
and both controls give 12 case/weight runs; the extension adds five, for 17 total.
An arm must improve frame-5
coverage over simple spread and preserve both controls' acceptance and selected
corner error versus random, allowing only 1e-6 px arithmetic roundoff. Only
weight 1 qualifies. The other weights are not extended or discarded from reporting.

## Pilot and controls

All errors below use 1280×720 coordinates. GX values are minimum maximum corner
error over geometry-valid generated proposals, before player filtering. Control
values measure the first retained court; acceptance is shown separately.
All four corners count, including extrapolated ones. There is no new numerical
threshold for usability.

| Sampler | GX0 minimum | GX5 minimum | Amateur-2: accepted / error | Broadcast: accepted / error |
| --- | ---: | ---: | --- | --- |
| Original random | 27.24 | 220.52 | Yes / 12.30 | Yes / 10.31 |
| Simple spread | 27.24 | 25.25 | No / no court | Yes / 54.02 |
| Rank weight 0.25 | 27.24 | 25.25 | Yes / 13.97 | Yes / 9.23 |
| Rank weight 0.5 | 27.24 | 16.30 | Yes / 7.12 | No, ambiguous / 9.23 |
| Rank weight 1 | 16.65 | 16.30 | Yes / 7.72 | Yes / 8.91 |

The middle weight's broadcast court is retained but not emitted: competing
courts make the result ambiguous. Weight 0.25 worsens the amateur output.
Weight 1 preserves both controls with lower selected-court errors. These are
two individual controls, not a general acceptance or false-positive evaluation.
The broadcast references lack boundary landmarks, so no boundary comparison
or generated-population error minimum is reported for that control.

## Seven-frame extension of weight 1

Maximum corner errors retain the same generated-population definition. Boundary
RMS is the existing zero-inset boundary-landmark proxy, already scaled to
1280×720. Each metric independently selects its best proposal; minima need not
describe the same court.

| GX frame | Random corner | Spread corner | Rank corner | Random boundary RMS | Rank boundary RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 27.24 | 27.24 | 16.65 | 7.12 | 5.06 |
| 5 | 220.52 | 25.25 | 16.30 | 78.25 | 4.95 |
| 689 | 21.99 | 13.23 | 20.69 | 5.57 | 5.56 |
| 5111 | 166.17 | 18.69 | 16.67 | 38.12 | 4.91 |
| 5766 | 37.86 | 35.55 | 35.55 | 10.20 | 8.26 |
| 77876 | 131.89 | 173.15 | 11.05 | 57.17 | 4.01 |
| 86088 | 216.16 | 46.94 | 29.86 | 19.45 | 6.55 |

Weight 1 improves both minima on 7/7 frames versus random. Against simple spread,
maximum corner error improves on five, ties on one and worsens on frame 689.
It has 12 floor survivors across GX, all rejected by the actual camera check.
Both baseline samplers and weight 1 therefore accept 0/7 GX frames.

The closest new frame-5 court is a different proposal from those 12 survivors.
It passes the player rule but fails floor support: lengthwise mean 0.4444 and
two distinct lines, against required values 0.55 and three. Its separately
measured camera error is 0.0967, below 0.1; the detector never reaches that check
for this court. Its paint-inset boundary RMS is 4.84 px. The user's visual
rejection applies specifically to this 16.30 px /4.84 px example.

## Which useful seeds are still missed?

The approved complete-search frame-5 court uses pair-product 85795 and template
72, giving generation 6476322 in the full enumeration. Its four incoming line
ranks are 0, 16, 6 and 9 out of 32 per family. Its seed is absent from random,
simple spread and all three cue weights. The better-ranked sampling still fails
to preserve this known useful geometry at the same budget.

The original accepted Amateur-2 court uses pair-product 170556 and template 139.
Its line ranks are 3, 4, 9 and 2. That seed is also absent from all new samples:
the restored amateur outputs are different courts, not recovery of the old seed.
Both target geometries were reconstructed from saved IDs within 1e-5 native px.
These traces choose targets using saved results; they do not tune the sampler.

## Checks, limitations and decision

Synthetic checks reproduce the prior random and zero-weight selectors and
independently verify a rank-only example. Scoped lint, shell syntax, all 17 new
case/weight runs, target reconstruction and viewer controls passed, exit 0.
The unchanged tracer verifies direct versus instrumented final outputs.
The [replay archive](evidence_ranked_seeds.zip) preserves scripts and results.

Independent Opus review confirmed the selector and stage accounting from source
and local selection checks. It did not assess the new Carmack scoring outputs.
It identified a wording gap: equal mathematical priorities can differ in the
last floating-point bit. The tested rule sorts computed float64 priorities,
breaking equal computed values by pair-product ID. Keep that implementation
with its results. Ascending bin order also favours earlier bins in the final
partial round; this is not uniform sampling over image space.

Weight-1 direct GX detection took 39.4–54.3 seconds per case. Runs overlapped,
so these are not controlled speed comparisons. Broadcast timing combines direct
and traced detection, unlike its earlier direct-only baseline. No speedup is claimed.

Stop tuning this cue tonight. It is useful evidence that sampling can improve
without sacrificing these two controls, but the approved court is still missed
and the new frame-5 geometry is visually rejected. The next decision is whether
a small geometric or correspondence-based construction can recover the right
court more directly. A separate deep review addresses that question; this
experiment does not establish a graph solution or justify relaxing acceptance.
