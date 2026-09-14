# Court spacing and paint: usable geometry with supplied directions

Visual inspection confirms usable generated courts in eight views. Paint ranking
is valuable on Amateur-2 but regresses GX0 compared with its excellent line-score
winner. Scene 16's new candidates remain unjudged. The
[panel-specific rulings](axis_matching_visual_judgements.md) record the differences
shown in the [comparison gallery](axis_matching_visual_check.html).

This is a diagnostic towards automatic court detection. Directions come from
previously approved fits or manual reference corners. The matcher chooses line
positions from image fragments. These results do not establish automatic recovery
of the directions. Production, annotations and original acceptance rules are unchanged.

## What was tested

All comparisons use the same nine development views from five videos: GX,
Amateur-2, Amateur-3, ShuttleSet 03 and ShuttleSet 21. There are no designated
holdouts. No new annotation is required at this stage.

Four generation arms ran on every view: the initial spacing matcher; the existing
player conditions applied earlier; a larger axis shortlist; and finite marking
support applied before the final shortlist. That is 36 completed case runs.
The final arm retains 256 candidate courts per view, giving 2,304 measured
candidates across the nine views. A separate appearance check
compares ranking by complete-line support with ranking by the existing bright-stripe
test, using the line score to break ties. Both rankings use the same candidate
pool and existing camera eligibility check. Reference corners never select either
winner. The supplied directions remain label-guided.

The table reports maximum corner distance from the manual reference at a
1280×720 display, in pixels. Lower values do not establish correct internal lines.
There is no new pixel success threshold. The panel-specific visual judgements
supplement these measurements.

| View | Line-score winner | Paint-ranked winner |
| --- | ---: | ---: |
| Amateur-2 frame 150 | 191.49 | 10.52 |
| Amateur-2 frame 28019 | 190.79 | 7.63 |
| Amateur-3 frame 0 | 9.35 | 9.23 |
| GX frame 0 | 8.67 | 20.53 |
| GX frame 5 | 9.30 | 9.30 |
| ShuttleSet 03 scene 17 | 9.82 | 9.82 |
| ShuttleSet 03 scene 19 | 8.63 | 11.36 |
| ShuttleSet 03 scene 16 | 8.75 | 9.02 |
| ShuttleSet 21 scene 20 | 8.17 | 8.66 |

## What the losses show

The early axis limit discarded close interpretations. Raising that limit alone
moved the failure downstream: Amateur-3 had a proposal 1.89 working pixels from
its supplied control before the final cut, but its closest survivor was 368.24
pixels away. Ranking by finite painted lengths before that cut preserved the
1.89-pixel proposal. Working images have a maximum dimension of 960 pixels;
these control distances use a different scale and comparator from the table.

Amateur-2 then exposed a separate ranking error. Close courts survived, but line
support favoured displaced alternatives. The paint check changes those winners.
It is not a sufficient correctness test: the previously rejected scene 17 fit
already passed every paint profile. The proposed simple 3.96 m shift does not
describe the saved Amateur-2 failures; their relative transforms contain expansion
and projective distortion.

## Outcome and subsequent experiment

The inspected results support testing automatically estimated directions while
retaining both rankings. A global paint-first rule is not established. The
[automatic-direction comparison](automatic_axes_results.md) records that completed
follow-up and its direction-selection failure. The 512-axis budget here is a
capacity diagnostic, not an established production configuration.

All 36 generation runs and nine appearance comparisons finished with exit 0.
Twelve synthetic tests, scoped lint/type checks and both browser overlay controls
pass, exit 0. Technical review covered the design, adapter, finite-ranking wiring and
appearance measurements. A diagnostic corner-ordering bug was corrected;
it does not change the closest retained control in the initial comparison.
The [methods and validation](methods.md) record settings, provenance and limits.
