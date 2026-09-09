# Paint geometry and post-base diagnostics

**The physical paint model is now explicit, but the paired trial gives mixed
results. Keep it experimental.** Post-base observations point to a better
existing alternative for Amateur-3 frame 17174. The saved pools for Amateur-3
frame 24515 and Amateur-4 frame 0 still lack a court that agrees with the visible
posts. Those cases need better proposals or fitting constraints.

The tests asked whether separating paint centres from outside boundaries improves
fits, and whether visible post bases distinguish the remaining wrong courts.
The intended detector supplies outside corners for mapping player and shuttle
landing positions onto the court. The priority is accurate visible boundaries
with sensible unseen geometry. A usable-fit cutoff has not yet been settled.

## What was compared

The paint trial used the same 682 starting courts from twenty development frames
across seven source videos. Each start was fitted under two models: the old
nominal lines treated as stripe centres, and physical stripe centres measured
from the court's outside boundary. Each model kept its own starting courts and
refits. Fragment identities stayed fixed during each fit. All 1,364 requested
fits converged; exact caching reduced the work to 1,090 distinct fits.

Both models used **stripe-only ranking**, with the existing 3:1 stripe/net score.
This orders courts by their agreement with observed line fragments and net
evidence. The previously published gallery prioritised agreement at line
intersections, called junctions. The comparison
below isolates paint geometry within stripe-only ranking; it is not a comparison
against every winner in that gallery. Generation and the existing floor, player
and camera gates were held unchanged. The new model has not been applied across
the complete detector.

Reference labels were used after ranking. The main measurement is the root mean
square (RMS) distance from clicked outer-boundary landmarks to the corresponding
finite projected court side. RMS gives larger misses more weight.
Distances use 1280×720 coordinates. This samples visible boundary alignment; it
does not measure every visible stretch of paint. Clicked corner errors are also
saved separately. Extrapolated corners never enter that clicked-corner measure,
even when their coordinates lie inside the image.

The original annotations target outside edges at the court boundary. The three
later short clips target stripe centres. Their boundary measurements therefore
use a 20 mm inset in court coordinates. The image clicks remain unchanged.
These conventions come from annotation clarification and visual review; they
are not subpixel ground truth. No new usability threshold was chosen.

## Physical model

The Badminton World Federation's (BWF) Diagram A specifies 40 mm paint,
a 420 mm clear gap between side stripes,
a 720 mm clear gap between rear stripes, and 1.98 m from the net to the nearest
short-service paint edge. The model follows those gaps and preserves outside
corners at `(0,0)…(6.1,13.4)`. [BWF Laws, Diagram A](https://usabadminton.org/wp-content/uploads/2021/05/Section-4.1-Laws-of-Badminton.pdf).

| Marking | Physical stripe centre, metres |
| --- | --- |
| Doubles sidelines | x = 0.02, 6.08 |
| Singles sidelines | x = 0.48, 5.62 |
| Centre line | x = 3.05 |
| Baselines | y = 0.02, 13.38 |
| Doubles long-service lines | y = 0.78, 12.62 |
| Short-service lines | y = 4.70, 8.70 |

Fragments can match either edge or the centre of a stripe. The correction changes
the physical fitting target rather than expanding a finished polygon in pixels.
The old geometry remains the default for historical replay and production.

## Paint result

Across the nineteen frames with an eligible court, median boundary RMS changes
from **2.17 to 2.07 pixels**. The small median change conceals both gains and
regressions. Amateur-2 frame 28019 remains empty because its known useful proposal
still fails the unchanged floor gate, which requires sufficient court-line
evidence. Eligibility means passing the existing evidence and geometry checks.
Each row below measures the highest-ranked court independently chosen by each
model. These examples show gains, regressions and persistent large errors;
the linked gallery includes all twenty frames.

| Example | Old paint RMS | Physical paint RMS |
| --- | ---: | ---: |
| Yellow 14 | 95.78 | 2.82 |
| Yellow 156 | 100.25 | 104.61 |
| Centre 64 | 1.44 | 0.89 |
| Amateur-3 frame 0 | 3.66 | 4.08 |
| Amateur-3 frame 10514 | 4.21 | 4.30 |
| Amateur-3 frame 24515 | 28.13 | 28.01 |
| Amateur-4 frame 0 | 207.35 | 205.81 |
| Amateur-4 frame 319 | 2.16 | 1.69 |

The large gain on Yellow 14 repairs a stripe-ranking error; the old junction
ranking already chose a much better court there. The large Amateur-4 error in
this table also belongs to stripe-only ranking. Its historical gallery fit was
different and is assessed in the post diagnostic below.

For the same winners, the old four-corner diagnostic changes from 10/20 to 11/20
at its 15-pixel cutoff. It measures the worst corner's distance from its reference,
including off-screen extrapolation. Boundary RMS instead measures distance to
visible boundary landmarks. Neither count establishes how many courts are usable.
No held-out or production-acceptance result is claimed.

[All twenty paired overlays](paint_overlays/index.html) show magenta outside
boundaries and dashed blue reference geometry. Each case links to its historical
gallery comparison. [Full paired measurements](paint_results.json.gz) retain
starts, attempts, assignments, gates, complete rankings and reference diagnostics.

## Do the visible posts distinguish the wrong fits?

A separate diagnostic used approximate readings of post-pole ground positions
on five existing images. Visual review confirmed the displayed observed post-base
circles. Occluded predicted posts were described as uncertain and do not provide
additional confirmed observations.
About five pixels is an illustrative reading allowance, not measured uncertainty.
Portable supports can also offset the pole from the nominal sideline. Occluded
sides contributed nothing. This is a cue test, not an automatic detector.

For each saved court, project the two doubles-sideline points at the net plane.
The table reports mean distance to the observed posts in 1280×720 pixels. The
"gallery" column uses the exact junction-ranked winner previously displayed.
"Best available" ranks the saved alternatives only by post distance. That ranking
is diagnostic and is not a proposed selection rule. A one-post mean is that
single distance. "Reference geometry" projects the posts from the manually
annotated four corners, providing a check on the approximate post readings.

| Frame | Observed posts | Gallery fit | Best available | Reference geometry |
| --- | ---: | ---: | ---: | ---: |
| Amateur-3 10514 | 1 | 3.09 | 0.83 | 4.16 |
| Amateur-3 17174 | 2 | 14.75 | 4.54 | 4.45 |
| Amateur-3 24515 | 1 | 40.92 | 26.18 | 3.49 |
| Amateur-4 0 | 1 | 57.49 | 54.47 | 4.56 |
| Amateur-4 319 | 2 | 4.06 | 4.06 | 3.94 |

For frame 17174, the better post-aligned alternative is already the stripe-ranked
winner. Its historical worst-corner error is 23.55 pixels, versus 76.53 for the
gallery winner. This supports investigating how junction ranking chose the
misplaced court. It does not require generating another court first.

The saved selection trace identifies the deciding evidence. Both courts have
one fully agreeing junction. The better court has an additional contradiction
at the near baseline: a line response extends beyond the expected centre-line
termination. The misplaced court's alignment there is too weak to make the
junction usable, so it receives no contradiction. The selector prefers zero
contradictions before comparing stripe scores (0.411 versus 0.503). Inspect the
detected fragments behind that continuation response before changing the penalty;
the same check also affects the frame-10514 control.

For frame 24515 and Amateur-4 frame 0, even the closest saved alternative remains
far from the observed post. Post-only ranking also worsens Amateur-4's historical
corner error, from 137.00 to 177.99 pixels. A post cue can expose disagreement
without being sufficient to choose the right court. Both good control frames
already agree with the observed posts within about five pixels.

Visual review supports keeping the frame-17174 alternative: it follows the court
lines closely, although the leftmost boundary follows the inner paint edge.
The frame-24515 alternative improves the bottom boundary but is too wide on both
sides. Amateur-4 frame 0 improves some line choices and has the correct right
side, but still follows inner markings at the far, left and bottom boundaries.
These partial gains do not establish a usable court in either case.

The frame-10514 control looks essentially unchanged, with slightly worse paint
edge alignment in the alternative. Both Amateur-4 frame-319 panels look correct.
In the separate paint comparison, Centre frame 64 follows outside edges better
under the physical model; both versions were judged usable. This is qualitative
feedback on named examples, not a revised success rate. Confirmation of the
post-base circles establishes visual placement, not a measured pixel tolerance. The alternative's predicted post also aligned correctly there;
the historical gallery choice's prediction was misplaced.

## Reproducibility and next work

The legacy arm reproduces the archived fitted corners within 0.00000026 native
pixels. Border rounding affects the old score: clipping
and inverse projection can put a mathematically visible sample just outside the
image. Both trial arms use the same 0.0000001-pixel allowance. Defaults remain
zero for historical replay. The largest score difference from the archive is
0.00250; it must not be described as exact score reproduction.

A separate same-machine check compared all 1,364 legacy starts and refits with
the original scoring source. With the original zero allowance, their scores
match exactly. All 682 fitting sample sets and weights also match the archive.
The candidate population and eligibility flags are preserved. This isolates
the score change from the allowance; it does not make the archived scores exact.

[Diagnostic replay bundle](paint_diagnostics.zip) contains the post readings,
per-candidate post measurements, control-check output and original scoring source.
Its README gives the control and post replay commands. To repeat the paired fit
from the repository root:

```bash
PYTHONPATH=src:. python -m experiments.annotator.independent_court.run_paint_refit \
  --recorded experiments/annotator/independent_court/recorded/player_guided \
  --annotations data/amateur_court_corners \
  --output /tmp/court-paint-replay
```

The remaining work is concrete: compare stronger evidence before discarding the
known Amateur-2 proposal; inspect the near-baseline continuation response on
frame 17174; and test
post-conditioned proposals on frame 24515 and Amateur-4 frame 0. Keep the
already-good fits as controls. The newest video adds seven annotated frames,
bringing the inventory to 27 frames across eight videos. Those seven frames and
the broadcast regression set have not been evaluated in this paint trial.

The detector remains experimental. This pass establishes the physical convention
and identifies where existing alternatives can help; it does not justify changing
the production detector or an automatic acceptance rule.
