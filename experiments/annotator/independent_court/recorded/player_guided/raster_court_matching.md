# Court evidence access with original raster distances

**Combining the two evidence-access changes lets the usable frame-5 court from
the GX amateur video pass and rejects both known wrong courts. It also rejects
one previously good Amateur-3 fit.** The result is mixed: these changes do not
establish a replacement scorer. The good-fit regression persists when distances use the original raster
calculation.

The larger aim is reliable court detection in partial amateur video. This single
experiment asks whether the scorer can recognise suitable observed lines for a
fixed court geometry. It tests access to raw line fragments and to merged lines.
Search, refitting, ranking and final detector acceptance are outside the test.
Production and the existing detector source are unchanged.

## What was tested

The test reuses the 63 fixed courts across 22 frames from the
[previous fixed-court experiment](fixed_court_matching.md): 25 examples from one
amateur video, labelled GX, and 38 older controls. Some courts were selected using
reference geometry. These are development diagnostics, not held-out accuracy
results or a test of automatic court selection.

All five arms keep the court corners, twelve painted intervals, visibility and
24 samples per interval fixed. **Interval support** is the fraction of samples
within four working pixels of an allowed observed fragment. Working images have
a maximum dimension of 960 pixels.

A court passes the **floor evidence rule** when each direction, lengthwise and
crosscourt, has mean support of at least 0.55 and at least three distinct merged
lines. An interval can count a merged line when its support reaches 0.55 and both
sampled endpoints lie within eight pixels of that line. These settings also
apply to the older controls. Passing this rule is not an accepted detection.

Every arm uses the original raster distances: round observed endpoints, draw
one-pixel lines, calculate OpenCV distance maps, then truncate projected sample
coordinates to pixel indices. None uses analytic finite-segment distances.

The arms change which observations the fixed court can access:

- **Original:** raw fragments and merged lines from the original image-angle
  families.
- **Fragments:** raw fragments within five degrees of each projected court
  marking; original family-based merged-line pools.
- **Lines:** original raw-fragment families; one merged pool built from all raw
  fragments.
- **Both:** projected-direction raw fragments and the all-fragment merged pool.
- **Unrestricted:** all raw fragments and the all-fragment merged pool.

The first four arms separate the two access changes. Both versus Unrestricted
isolates direction filtering of raw fragments. No arm adds an angle filter to
merged-line matching: all retain the original endpoint-residual rule.

Every merged pool uses the original merging algorithm, eight-pixel and
three-degree merge tolerances, and 32-line cap. Merging and the cap can change
pool membership. The original pools allow 32 lines per family; the shared pool
allows 32 in total. It is therefore not necessarily a superset of either family
pool, and this capacity difference can affect rejection.

## Results

Each cell counts fixed courts passing the floor evidence rule. Older references
comprise four amateur courts, two broadcast courts and one floor-texture
continuation case. The six saved stripe fits are previously useful detector
proposals from the first six of those frames. The two known wrong courts are a
GX floor survivor and a broadcast junction proposal.

The seven random GX proposals have no correctness labels. The 24 non-court
controls are three fixed placements on each of eight frames. Their rejection
counts do not measure a search false-acceptance rate.

| Fixed population | Original | Fragments | Lines | Both | Unrestricted |
| --- | ---: | ---: | ---: | ---: | ---: |
| GX references | 0/7 | 0/7 | 0/7 | 3/7 | 5/7 |
| GX saved starts | 0/7 | 0/7 | 0/7 | 0/7 | 1/7 |
| GX complete-search examples | 0/2 | 0/2 | 0/2 | 1/2 | 1/2 |
| GX frame-0 legacy refit | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 |
| Known wrong courts | 2/2 | 0/2 | 1/2 | 0/2 | 1/2 |
| Older references | 7/7 | 7/7 | 7/7 | 7/7 | 7/7 |
| Previously good stripe fits | 6/6 | 5/6 | 6/6 | 5/6 | 6/6 |
| Placements on non-court frames | 0/24 | 0/24 | 0/24 | 0/24 | 0/24 |
| Random GX proposals | 0/7 | 0/7 | 0/7 | 0/7 | 0/7 |

### Both changes are needed for the usable GX court

Both passes GX references at frames 0, 5 and 5111, plus the saved frame-5
complete-search court. Neither change alone passes any GX court.

For the saved frame-5 court, restored fragment support raises the lengthwise
mean from 0.4305556 to 0.6041667. The all-fragment merged pool is then needed to
raise its distinct lengthwise count from two to three. Its crosscourt mean is
0.7222222, with four distinct lines.

The user judged this specific geometry usable: it is the right-hand court in the
[frame-5 comparison](gx_trace_overlays/gxBQ_window_00_frame_5__cap.jpg). The left
and bottom follow the outside of the paint. The mild inset on the right is
acceptable, although full right-side paint width remains a preference. This
judgement does not approve detector deployment.

On that same fixed court, projected-direction access raises near-baseline
support from 0/24 to 23/24 samples. Raw fragments 132 and 187 follow the baseline
but fall outside the original crosscourt angle family. Right-singles support
rises from 3/24 to 18/24.

### Direction filtering also rejects a good Amateur-3 fit

The Amateur-3 frame-0 saved stripe fit passes Original and Lines, but fails
Fragments and Both. Direction filtering lowers far short-service support from
17/24 to 11/24 samples. Its distinct crosscourt count falls from three to two,
and its crosscourt mean falls from 0.6875 to 0.5486111, below the 0.55 threshold.
The regression persists with the original raster distances and either merged
pool.

The lost samples are 1, 7, 8, 9, 22 and 23. Raw fragments 0, 62, 84 and 108
supplied their support. Those fragments have undirected angle differences of
about 9–13 degrees from the projected short-service line.

On 13 September, the user inspected those fragments against the image.
Fragments 0, 62 and 84 are definitely not court lines; sunlight through windows
was the suggested explanation. Fragment 108 follows the outer right court line
with a slight inset. None supports the far short-service marking. The original
score therefore credited incidental crossings at these six samples. Their loss
does not justify widening the direction tolerance.

A subsequent fixed-geometry attribution explains the reference's 14/24 versus
the stripe fit's 11/24 directional samples. Samples 12, 13, 14 and 21 have the
same nearest compatible fragment, raw 186, for both courts. Their reference
distances are 3.606, 3, 2 and 3 working pixels; the stripe distances are 5, 5, 5
and 6.325 pixels. The unchanged limit is four pixels. Sample 2 changes in the
opposite direction against raw 97, from 4.123 to 3.606 pixels. Four losses and
one gain explain the net difference of three. This attributes the numerical
difference to sample placement; it does not establish every retained match's
identity or show that the marking is hidden.

### Rejection of the wrong courts has limits

For the wrong broadcast court in scene 0017, direction filtering lowers the
lengthwise mean from 0.6111111 to 0.5486111. Its distinct line counts stay at
four in each direction. Unrestricted passes this court with lengthwise support
of 0.7361111. Direction filtering therefore helps reject this known wrong court,
but the small margin below 0.55 does not establish robust separation.

The wrong GX court fails when either access change is made. In Lines, the shared
32-line cap removes its far-centre and far-long-service matches. Those groups
exist before the final cap, at zero-based ranks 36 and 46. This rejection is a
cap effect, not evidence that a broader pool recognises wrong geometry better.
No additional arm with a different cap was scored.

Every Unrestricted result on non-court placements and random GX proposals also
fails the distinct-line requirement. Many fail the support means as well. Their
rejection counts therefore do not isolate discrimination by fragment support.
Together with the Amateur-3 failure, these controls leave the distinction between
useful and wrong courts mixed.

## Decision

Keep this as a diagnostic result. Improved access restores evidence for a usable
court, but the directional rule also rejects an established good fit.

The 13 September visual check resolves the excluded-fragment question: those
matches were incidental support. The attribution then explains the remaining
reference-versus-stripe support difference through sample placement. Keep
geometry, thresholds and the matching result unchanged. The planned seed-selection
comparison can now proceed separately with the original scorer in both arms.

## Reproducibility and remaining uncertainty

The [replay archive](raster_court_matching.zip) contains the two unchanged input
packs, all five-arm results, runner, correspondence inspection and saved checks.
All 63 reconstructed baseline scores exactly match the original scorer. Their means, distinct counts,
interval support and covered samples also match the previous raster results exactly. Corners and settings are unchanged.

The full 22-case run took 11.61379 seconds, excluding imports and transfers.
All 38 relevant tests passed. Scoped lint and type checks also passed; each
check exited with code 0.

Six decisive intervals were traced to individual raw-fragment rasters. Their
four-pixel coverage decisions exactly match the union distance maps. Individual
and union distances differ by at most 7.63e-6 pixels because of float32 rounding.
The attribution check uses relative tolerance 2e-7 and absolute tolerance 1e-6.
These tolerances validate distance calculations only; they do not change the
floor support rule.

On the GX frame-0 reference, unrestricted raw fragments 160 and 301 supply all
11/24 supported far-centre samples. Neither fragment satisfies the directional
rule, which supplies 0/24. This illustrates how Unrestricted can gain support
from differently oriented fragments. Its higher pass count cannot be read as
stronger evidence for the intended markings. The far-centre marking's visibility
remains unresolved.

A focused Fable 5.1 review replayed the usable frame-5 court and a randomly drawn
broadcast stripe court. Saved scores, support and line counts agreed. It checked
all 63 courts' recorded arithmetic and identified the shared-cap limitation.
A separate verification traced 14 lost interval matches across ten courts;
12 corresponding groups survive before the final cap. No implementation defect
was established in the review's stated coverage. No new score arm was run.
