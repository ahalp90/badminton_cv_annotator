# Historical impact of mismatched person boxes

## Bottom line

Some historical person-mask and junction results used bounding boxes from a
different frame than the image being measured. Every original broadcast
junction measurement used source-frame boxes on cached median images. That
broadcast result has now been repaired with both no spatial mask and a mask
supported by at least two of the three median-image frames. Both corrected arms
score 1/18, down from the invalid 3/18. The mismatch made junction-first ranking
look better; it did not cause the poor result.

The fault is bounded. Paint-only scoring, temporal player-foot gates, stripe
scoring, proposal generation, direction agreement and the W5 rankings remain
valid. No mismatched box reached any produced W5 measurement. Some broadcast
views stopped or were absent in earlier stages, and the final W5 conclusion is
still 0 affected. The current admission experiment can continue once the same
rule is driven by explicit provenance instead of case-name prefixes.

This is development evidence, not a held-out evaluation. A wrong box can hide
real court paint or fail to hide an actual player. Its bias can go either way,
so affected figures cannot be adjusted with a simple penalty.

The larger goal is a trustworthy automatic court detector that does not depend
on ground truth or CourtKeyNet. This audit keeps false precision out of the
evidence used to choose the next detector change.

## What went wrong

The three frozen packs contain 47 cases:

| Image and box relationship | Cases | Safe for person masking |
| --- | ---: | --- |
| Same source frame | 17 | yes |
| Different source frames | 10 | no |
| Cached median image with source-frame boxes | 20 | no |

The pack producers deliberately selected the nearest scheduled person
detection and recorded its frame. Some consumers ignored that frame identity
and treated `bbox_px` as if it described the line image.

The ten source-frame mismatches are GX5, GX5111, GX5766, GX77876, GX86088,
yellow14, letterboxed58, centre64, centre71 and am4-319. Every case in the
broadcast-extension pack uses a cached median image and ordinary source-frame
boxes. The earlier [marking-refit note](../../../experiments/annotator/independent_court/recorded/player_guided/marking_refit.md)
and [stripe protocol](../../../experiments/annotator/independent_court/recorded/player_guided/stripe_protocol.md)
already disclosed the frame-offset limitation.

## Impact by experiment

| Experiment or report | Affected scope | Ruling |
| --- | ---: | --- |
| Marking-refit occlusion rankings | 5 of 20 | Rerun the two masked arms or label their rows unsafe |
| Marking-refit cross-frame masked outputs | 3 of 3 windows (4 mismatched frames) | Label the saved masked summaries unsafe; keep the published unmasked table |
| Junction diagnostics and junction-first rankings | 5 of 20 | Rerun junction measurement and dependent summaries |
| Paired paint-refit result | 5 of 20 diagnostic cases | Relabel embedded occlusion diagnostics; retain paint winners and boundary metrics |
| Broadcast extension junction-first results | 19 box-consuming cases; 18 scored | Repaired: both no-mask and two-of-three-mask arms score 1/18 |
| Appearance contradiction probe | 5 of 20 | Rerun before retaining population totals |
| Line-identity person arms | 5 of 9 | Rerun or label those case-arm rows unsafe |
| W5 stages and line-template regression | 0 of 9 | No mismatched box reached any produced W5 measurement; some broadcast views stopped or were absent in earlier stages; final conclusion 0 affected |

### Marking-refit replay

In the [marking-refit replay](../../../experiments/annotator/independent_court/recorded/player_guided/marking_refit_replay.zip),
boxes enter only the `direct_occlusion` and `bidirectional_occlusion` rankings.
Five of twenty cases are affected.

These exact claims are unsafe:

- direction-aware plus boxes: 7 accurate and 3 wrong;
- bidirectional plus boxes: 8 accurate and 3 wrong.

The unmasked 7/2 and 8/2 results remain valid. Proposals, refits and gates also
remain valid. The smallest correction is to rerun the two occlusion arms for
the five affected cases. If that rerun is not worth doing, the two historical
rows should be labelled unsafe.

The replay archive's `cross_frame.json.gz` contains `direct_occlusion` output
only for three three-frame windows: `yellow_short`, `letterboxed_short` and
`centre_short`. Four mismatched frames fall in those windows; `am4-319` is
absent. Each window summary scores candidates by the median across its three
frames, so all 3/3 window summaries are unsafe. `bidirectional_occlusion`
appears in the archived `cross_frame_reverse.json.gz`, which `summarise.py`
reads into the summary's `cross_frame` section. The saved masked and unmasked
errors happen to match in each window, but that does not validate a mask from
the wrong frame. The published cross-frame table reports the unmasked arms, so
that table remains valid.

### Paired paint-refit result

The saved [`paint_results.json.gz`](../../../experiments/annotator/independent_court/recorded/player_guided/paint_results.json.gz)
did execute legacy `prepare_case`/`evidence`. Its embedded
`gate_evidence.direct_occlusion` diagnostics for the five mismatched marking
cases are unsafe. The headline paint-model winners and boundary metrics remain
valid: the imported `eligible()` uses only `scheme_eligible['original']`, each
entry score is stripe-exclusive plus net, and the ranking orders that score.
Boxes affect only the stored occlusion diagnostic. Two current readers now
require schema-2 bound results, so the tracked schema-1 result cannot be reused
silently.

### Stripe and junction diagnostics

The junction path excludes arm samples inside `bbox_px`. The same five marking
cases therefore undermine these aggregate claims across the saved 20-case,
682-geometry replay. They account for 258/682 geometries, including 132/439
with a usable junction, 161/547 sites and 39/161 disagreeing arms:

- 439 of 682 geometries with a usable junction;
- 547 sites, 161 disagreeing arms and 159 geometries;
- both junction rankings reported as 9/20;
- renewed-pool junction-first results of 9/20, 9/20 and 10/20;
- inherited selected-junction traces and summaries built from them.

The statement in archived `stripes.md` that “none of the 78 accurate entries
has a disagreement” is unsafe. The archived `reference_probe.py` also applies
person masks across all twenty cases. Its three reported disagreements are on
same-image Amateur-3 cases and remain valid, but the absence of any further
disagreement does not establish an exhaustive population claim.

Yellow14 is the only affected saved winner that differs from its stripe winner:
the junction choice reports 16.56 px while the stripe winner reports 696.24 px.
The other four currently retain the same winner, but corrected masks may still
change their order. Stripe-only results and candidate-fit accounting remain
valid.

### Broadcast extension

The original broadcast junction-first replay had nineteen box-consuming cases.
SS21-10 and SS21-39 lack verified references, so the scored denominator is
eighteen. SS21-10 also had zero eligible geometries and produced no junction
measurement. That replay applied first-frame boxes to cached median images for
the nineteen cases.

The [corrected replay](runs/broadcast_junction_box_repair_20260921/result.md)
first reproduces every historical order, then remeasures the saved candidates
with no spatial mask and with a mask supported by at least two of the three
median-image frames. Both arms produce the same winners and score 1/18 for both
original and paint-qualified observations. Paint qualification changes no full
ranking. Three winners change from the invalid run: SS03-38, SS03-16 and
SS21-29. Two previously accurate winners become wrong, so the mismatched box
inputs inflated the old 3/18 result.

The 194.27 px scene-17 winner remains unchanged and now has valid support. The
exact historical aggregate and its blanket “winners unchanged” claim remain
invalid. The broader conclusion survives: junction-first needs rethinking for
broadcast footage.

The broadcast stripe result of 16/18 and its two failures remain valid. The GX
result of 0/7 retained also remains valid because it uses temporal feet rather
than person boxes.

### Appearance contradiction probe

This probe masks raw and paint-qualified junction arms on all twenty marking
cases. Five are affected. Its 87 changed arms among 140 case-arm comparisons,
“only two selections changed”, and population-level contradiction and winner
claims are unsafe.

The saved winner changes occur on same-image cases am3-10514 and am3-17174.
The mismatched cases currently keep their winners, but that does not establish
the population-level word “only”. Paint-only fragment qualification remains
valid.

### Line identity

The `person`, `person_observations` and `paint_person` paths spatially mask line
fragments. Five of the nine cases are affected: GX5, SS03-17, SS03-19, SS03-16
and SS21-20.

Their dropped counts, direction fits, incidence bounds, axis ranks, pools and
winners are unsafe. This undermines:

- “person repairs both GX views”;
- GX5's 9.7 to 3.1 bound, 23.8 to 11.5 fit and 11.1 nearest matcher result;
- the broadcast person regressions;
- “breaks four/five other views” and the cross-population recommendation.

Same-image person results remain valid: GX0's 6.02 to 1.22 bound, 6.80 to 1.50
fit and 1.9 matcher result, plus the person rows for Am2-150, Am2-28019 and
Am3-0. The dedicated GX0 mask replay is also valid.

Paint and `paint_observations` do not use person boxes. Their nine-view results,
including the Amateur-2 improvements, remain valid. See the current
[line-identity result](../line_identity/results.md) and
[evidence record](../line_identity/evidence.md). The duplicate
[`line_identity_results.md`](../next_steps_20260916/webui_seed/reports/line_identity_results.md)
is byte-identical and needs the same caveat. Its source copy,
[`filter_replay.py`](../next_steps_20260916/webui_seed/source/filter_replay.py),
also masks with `bbox_px`.

## Repair by decision value

The mismatch does not justify rerunning every affected artefact. Correct the
results that could reopen or close a live detector branch, in this order:

| Priority | Smallest useful repair | Decision it can change |
| ---: | --- | --- |
| 1 | **Complete:** recomputed both broadcast junction-first orders on the nineteen cached median cases with no mask and a two-of-three-frame person mask; rebuilt the eighteen-reference summary | Junction-first remains poor at 1/18 under both masks |
| 2 | Use same-image boxes for GX5 and the same composite mask for SS03-17, SS03-19, SS03-16 and SS21-20; rerun `person_observations` and rebuild its nine-view tables | Whether person-filtered observations remain a useful axis-matching option |
| 3 | Recompute the junction measurements for yellow14, letterboxed58, centre64, centre71 and am4-319; merge them with the fifteen valid cases; rebuild the full junction rankings | Whether junction contradictions add a useful clue across the marking population |
| 4 | Rerun the separate appearance-contradiction records for those five cases | Whether its exact population totals and “only two selections changed” claim survive |
| 5 | Rerun `direct_occlusion` and `bidirectional_occlusion` on those same five marking cases | Whether the historical masked marking-refit arms deserve further attention |

The first repair reused the saved candidates and changed only the junction
measurement, ordering and dependent summaries. One arm used no spatial mask.
The second masked a pixel when person boxes covered it in at least two of the
three frames used to make the median. Both arms produced the same winners, so
occlusion handling does not explain the failed broadcast selections. The
machine-readable packet retains the lower-order differences for inspection.

The second repair is narrower than the original experiment. GX5 needs boxes
measured on its actual line image. The four broadcast cases can reuse the
two-of-three composite mask above. Rerun `person_observations` because it tests
the still-live idea of filtering the fragments seen by axis matching while
holding the chosen directions fixed. Replacing an invalid mask with no mask
would merely duplicate the baseline.

Do not rerun the direction-changing `person` and `paint_person` arms merely to
recount their failures. Valid same-image results already show that direction
selection becomes unstable: `person` badly regresses Am2-150 and Am3-0, while
the box-free `paint` arm shows the same broader failure pattern. Corrected boxes
could change the GX5 and broadcast numbers, but they cannot rescue that design
decision.

The marking junction and appearance probes use different sample populations.
They may share corrected mask provenance or a new measurement cache, but they
are separate reruns. The appearance totals rank below the junction selection
because their cautious conclusion still has support from the two unaffected
same-image winner changes. The masked marking-refit arms can wait until a
higher-priority rerun gives a reason to revisit spatial person masking.

Do not rerun the three saved cross-frame masked summaries or the embedded
paired-paint occlusion diagnostics merely to restore old numbers. The published
cross-frame table is unmasked, the paired-paint winners are box-independent and
both live conclusions remain supported. Keep the affected diagnostic rows
labelled unsafe. W5, line-template admission, stripe-only, temporal-feet,
proposal and camera results need no repair.

## Results that remain valid

- No mismatched box reached any produced W5 measurement. Some broadcast views
  stopped or were absent in earlier stages. The final conclusion remains 0
  affected, so no W5 rerun is needed. See the [steering record](steering_record.md)
  and [line-template result](runs/line_template_regression_20260920/result.md).
- G1 uses `paint_observations`, which is box-independent.
- Baseline generation, direction agreement, cap-loss, pregate and axis replay
  use temporal feet rather than `bbox_px`.
- Paint profiles use the measured frame and line segments, not person boxes.
- The older motion and activity diagnostics align each detection to its own
  sampled frame before storing feet and motion weights. They are unaffected.
- L3 temporal only counts `bbox_px`; its `inlier_target_bbox_px` field is the
  SIFT-keypoint extent, not a person box.

The line-template admission diagnosis also remains valid. Its two new wrong
winners, GX0 and Am2-28019, entered with too few visible court markings. Both
cases have same-image boxes, but that is not needed for the ruling: source
admission cannot read player gates. The useful GX5 candidate had stronger
two-direction visibility. This evidence still justifies testing a visibility
floor.

### What the visibility floor counts

The rule counts projected pieces of the standard court diagram, not detected
line fragments. In ordinary visual terms, the diagram has five lengthwise line
positions and seven cross-court levels when the net is included. The gate uses
a slightly different count. It excludes the net because it is not floor paint,
then counts the two disconnected centre-line halves separately. That gives six
painted pieces running along the court and six running across it.

```text
Top-down court sketch, not to scale

          lengthwise painted lines
       D      S       C       S      D
far B  +------+------+-------+------+  1
       |      |       |       |      |
far L  +------+------+-------+------+  2
       |      |       |       |      |
far S  +------+------+-------+------+  3
       |      |               |      |
net    - - - - - not floor paint - - - -
       |      |               |      |
near S +------+------+-------+------+  4
       |      |       |       |      |
near L +------+------+-------+------+  5
       |      |       |       |      |
near B +------+------+-------+------+  6

D = doubles sideline    S = singles sideline
C = centre line, counted as separate far and near pieces
B = baseline            L = doubles long-service line
```

A piece counts as visible when at least 12 pixels of its projection fall inside
the working image. A `(3, 3)` floor permits half of either family to be cropped
while excluding proposals that expose less than half of the court in one
direction. A `(4, 3)` floor strengthens only the usually well-seen lengthwise
family. A `(5, 3)` floor still permits one lengthwise piece, such as the far
centre-line segment, to be absent. This is a minimum testable-court-coverage
rule. It does not mean four detected white fragments, and visibility alone does
not claim that image evidence supports the projected line. Detected fragments
contribute to the separate support score.

The planned 27-case sweep answers a question the saved nine-case pool cannot.
The saved-pool check happens after the 256-candidate cap, so it cannot show how
pre-cap filtering and refill change the pool or whether a floor over-prunes
other camera regimes. The eighteen added scenes retain valid line and paint
evidence when person masks are unavailable. The sweep may still reject every
tested floor; it does not assume that a threshold must be adopted.

The two court directions need different treatment. Match footage usually shows
the lengthwise lines in a roughly vertical direction. Cropping along the court
length can remove one or more cross-court baselines or service lines even from
an overhead security camera, and head-height amateur footage often loses more.
A hard cross-court floor would therefore reject camera framing as if it were a
bad court. The revised sweep compares `(lengthwise, cross-court)` floors of
`(0, 0)`, `(3, 3)`, `(4, 3)` and `(5, 3)`. The `(3, 3)` arm is a broad minimum
coverage boundary. The targeted rule is `(4, 3)`; `(5, 3)` tests a stronger
pragmatic compromise and may reject footage missing two lengthwise pieces. The
pool is already capped at 256, so these floors chiefly stop barely testable
courts from occupying refinement slots rather than making an unbounded search
finite.

The W5 player-coverage sensitivity check uses temporal `all_feet_px`, not the
mismatched spatial boxes. It remains valid as a diagnostic. It remains a
secondary rule because it would couple court admission to player tracking, not
because of this box-provenance fault.

## Why the mismatch escaped review

This was mainly the coordinator's mistake. The coordinator gave a Luna Max
executor, not a low-capacity executor, a separate mapping brief that asked for
box availability and population readiness. Unlike the tracked W5 executor
contract, that earlier mapping brief did not ask whether each box described the
image that a later consumer would measure, especially when that image was a
multi-frame median. The mapper completed the narrower task. The coordinator
then accepted it without checking the mapped inputs against the W5 rule that
permits masking only with reliable same-image boxes.

The missing acceptance check matters more than the model choice. A stronger
executor could have noticed the unstated conflict, but the task did not require
that audit. Sending the same brief to a more capable model would not make the
result reliably safe. The later Carmack executor shows the expected executor
behaviour: it stopped before running invalid G1 work when its helper snapshot
disagreed with the frozen contract, then continued only after the mismatch was
resolved.

Future routing should keep tightly bounded, linear mapping and execution with
Luna. Cross-stage provenance work needs an explicit integration review owned by
the coordinator or a separate Sol Medium reviewer on the normal tier. That
review must check the measured image, the box source frame, the meaning of any
composite image, and every consumer that uses the boxes spatially. This is a
change to the task and acceptance criteria, not merely a change of executor.

## Corrective action

1. Bind every frozen case to explicit image and box provenance without changing
   the MD5-pinned pack bytes.
2. Let optional W5 masking run without a mask while recording why it is
   unavailable.
3. Make person-aware and junction consumers fail clearly when boxes do not
   describe the measured image.
4. Rerun only the affected masked arms and dependent summaries listed above.
5. Keep paint-only, temporal-feet and W5 evidence. Do not regenerate unrelated
   geometry.

An exact yellow frame-14 detection is not required for the current W5 admission
decision. It would repair one case while leaving the other nine source-frame
mismatches and twenty composite images unresolved. Exact detections should be
generated when a future person-aware experiment needs those cases.

## Remaining limit

This audit identifies which saved measurements consumed mismatched boxes. It
does not predict which corrected winners will change. The affected rows need a
rerun before they can support exact aggregate performance claims.
