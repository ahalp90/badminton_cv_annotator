# Line identity: where the court detector loses the court, and whether removing non-paint fragments helps

Run `line_identity_20260915_222437` on `fix/court-det`. Written for a colleague who knows the court-detection work but has not followed this branch. The aim was to learn two things about the automatic court detector. First, which step inside its matcher loses a court that the selected directions could support. Second, whether removing fragments that are not white paint, or that lie inside a person box, before any geometry is fitted, recovers it. Nothing here decides production readiness.

**Result.** The matcher loses close courts at one named step: the per-direction cap of its axis matching, which keeps the 512 best-scoring line-to-marking matchings per direction out of tens of thousands and scores tightly aligned clutter above the court's own markings. The masks and the per-pair cap behind it are cleared. Removing non-paint or in-person-box fragments before direction selection is a lottery: it repairs the two GX views and breaks four others, because the coverage rule reacts to which fragments exist rather than to how much clutter goes. Removing those fragments only from what the axis matching sees, with the baseline directions kept, helps where the cap had cut the court (GX0 from 34 to 8 or 13 px, Amateur-2 frame 28019 from 62 to 6 or 8) and costs at most two pixels elsewhere. The compute-host matcher run (table below) says how far those axis-stage gains reach the ranked winners.

## Terms

- Direction selection: the automatic choice of 16 vanishing directions from merged line groups (`vp_pruning.estimate`, coverage rule). The baseline selection (B) is the comparator throughout. The direction experiment's midpoint-anchor selection (M) and precise-representative selection (R) appear only where their records were reused.
- Direction fit: the least-squares court fitted to the control from one ordered pair of selected directions with both vanishing points fixed. It says what the directions could support. It is not a generated court.
- Axis matching: the matcher's step from one direction pair to courts. Along each direction it enumerates a scale and offset from every pair of direction-compatible observed line groups against every pair of court markings. It keeps enumerations with at least three supported markings that the necessary player rule accepts, drops duplicates by marking assignment, and keeps the 512 best by score (the per-direction cap). Every kept horizontal matching combined with every kept vertical matching is a combined court.
- Later masks and caps: the geometry mask, the player mask, the per-pair cap of 256 courts, and the global caps of the two rescoring stages.
- Control: the frozen reference court per view. Six are visually approved, three are manual references. Distances are maximum corner distance in working pixels (960 by 540) with the 180-degree relabelling allowed.
- Incidence bound: the lower bound on the corner error of any court that keeps one ordered pair of the selected directions. A fitted court can do no better.

## Population, comparator and evidence status

Nine frozen development views at working size 960 by 540. Two come from the GX gym video (GX0, GX5), three from amateur halls (Am2-150, Am2-28019, Am3-0) and four from ShuttleSet broadcasts (SS03-17, SS03-19, SS03-16, SS21-20). GX0, GX5, Am2-150, SS03-19, SS03-16 and SS21-20 have visually approved controls; Am2-28019, Am3-0 and SS03-17 have manual references. Nothing is held out. The comparator for every arm is the baseline detector as the direction experiment measured it. Gaps under 0.05 px are called the same. Every local replay is gated: the unfiltered path must reproduce the saved records exactly before a filtered path is measured.

## What was tested, and against what

Three diagnostics and one intervention, in order.

1. Courts before the masks (compute host, one run). The matcher's geometry and player masks were instrumented to record every combined court. Two view-selection pairs were run, GX0 under M and Amateur-3 under R, because there a close direction fit had produced no close court.
2. Axis-matching replay (local). For six direction pairs the axis matching was rerun with no per-direction cap. The court reachable after each of its rules was measured with the other direction held at the ideal matching that the direction fit implies.
3. Paint-profile study (local, descriptive). For every fragment on the nine frames: how much brighter a nearby ridge is than its two flanks, and how saturated it is.
4. Fragment filters (local replay on nine views, then the matcher on the compute host). A person-box rule drops fragments whose midpoint lies inside a person box. A paint rule keeps fragments with ridge contrast at least 20 and saturation at most 90. Each rule is tested two ways in the local replay: with its own direction selection from the filtered fragments (arms person, paint), and with the baseline directions kept and only the fragments the axis matching sees filtered (arms person_observations, paint_observations). All four arms went through the unchanged matcher on all nine views; the own-direction arms were relaunched after a first attempt with nineteen concurrent jobs stalled the host (`runs.md`).

## What happened

**The masks and the per-pair cap are innocent. The per-direction cap is where the close court goes.** Before the masks, the nearest court to the control is the same court that survives them: 33.8 px on GX0 and 40.1 px on Amateur-3. Nothing within 20 px exists among 22 and 17 million combined courts. The axis replay shows why. On Amateur-3 under R, a horizontal matching that reaches the 4.3 px direction fit is enumerated and passes the support and player rules. It sits at position 11,518 of 23,864 in the axis-score order, and the cap keeps 512. Every kept horizontal matching there supports all five markings with a score above 0.885; the near-fit one scores 0.58. With no cap the pair would give a 6.2 px court. Under the baseline selection the same pair keeps a 5.4 px matching at position 367, which is why the baseline pool holds a 4.3 px court there. The same cut hits the baseline's best-fit pair on GX0: position 1,459, a 34.4 px kept court against 6.3 px uncapped.

**Paint contrast separates court markings from clutter on the amateur and broadcast frames, not on the GX frames.** Median ridge contrast of fragments on painted markings against the rest is 41 against 8 on Amateur-3, 59 to 71 against 18 to 25 on the four broadcast views, 45 to 51 against 16 to 18 on the Amateur-2 views, and 21 to 22 against 12 on the GX views, whose far markings are dim.

**Filtering the fragments that feed direction selection is a lottery.** The incidence bound and the best direction fit per arm (working px):

| View | baseline: bound / fit | person: bound / fit | paint: bound / fit | paint_person: bound / fit | paint15: bound / fit | paint25: bound / fit |
| --- | --- | --- | --- | --- | --- | --- |
| GX0 | 6.0 / 6.8 | 1.2 / 1.5 | 15.4 / 32.8 | 15.4 / 32.8 | 8.1 / 17.6 | 15.1 / 37.9 |
| GX5 | 9.7 / 23.8 | 3.1 / 11.5 | 4.4 / 8.4 | 6.1 / 18.8 | 8.4 / 19.1 | 3.3 / 6.3 |
| Am2-150 | 2.9 / 3.3 | 6.1 / 8.9 | 8.7 / 12.0 | 12.1 / 14.3 | 7.4 / 14.3 | 3.9 / 4.9 |
| Am2-28019 | 2.6 / 3.0 | 2.1 / 5.2 | 2.2 / 2.7 | 0.3 / 0.6 | 0.5 / 0.5 | 0.7 / 0.8 |
| Am3-0 | 3.1 / 3.7 | 3.7 / 4.2 | 6.1 / 9.0 | 4.0 / 4.7 | 3.7 / 4.3 | 3.7 / 4.3 |
| SS03-17 | 1.5 / 2.0 | 1.5 / 2.0 | 1.9 / 2.1 | 1.3 / 2.1 | 1.9 / 2.0 | 1.7 / 1.9 |
| SS03-19 | 2.2 / 2.3 | 4.2 / 4.5 | 3.5 / 3.7 | 3.2 / 3.2 | 1.1 / 1.3 | 6.2 / 6.2 |
| SS03-16 | 1.0 / 1.5 | 4.9 / 5.1 | 1.6 / 2.0 | 1.6 / 2.0 | 1.6 / 2.0 | 0.5 / 0.6 |
| SS21-20 | 0.9 / 1.0 | 4.4 / 4.5 | 11.6 / 12.3 | 11.6 / 12.3 | 3.6 / 3.8 | 6.1 / 6.4 |

The person-box rule takes both GX views most of the way to their controls while removing at most one fragment on a marking. It also moves Am2-150 from 2.9 and 3.3 to 6.1 and 8.9, and three broadcast views from under 2.3 to between 4.2 and 5.1. The paint rule regresses GX0 and SS21-20 badly and improves GX5 and Am2-28019. The contrast threshold is not monotone on Am2-28019, Am3-0 and SS21-20. The coverage selection reacts to which fragments exist, not to how much clutter is removed.

**Filtering only the fragments the axis matching sees, with the baseline directions, helps where the cap had cut a near-fit matching and costs at most two pixels elsewhere.** Nearest court from the baseline's best-fit pair, kept matchings only (working px; the full table with ranks is in `evidence.md`):

| View | Arm | Pair | Direction fit | Horizontal best distinct (rank) | Horizontal best kept (rank) | Vertical best distinct (rank) | Vertical best kept (rank) | Kept by kept | No cap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GX0 | baseline | 143 | 6.8 | 6.3 (1459 of 2344) | 34.4 (178) | 7.3 (504 of 1944) | 7.3 (504) | 34.3 | 6.4 |
| GX0 | paint_observations | 143 | 6.8 | 6.3 (181 of 220) | 6.3 (181) | 24.5 (108 of 190) | 24.5 (108) | 12.6 | 12.6 |
| GX0 | person_observations | 143 | 6.8 | 6.3 (773 of 1364) | 7.6 (376) | 7.3 (376 of 1522) | 7.3 (376) | 8.3 | 6.4 |
| GX5 | baseline | 17 | 23.8 | 21.4 (1676 of 1768) | 22.3 (211) | 21.0 (874 of 2302) | 25.2 (134) | 23.7 | 20.7 |
| GX5 | paint_observations | 17 | 23.8 | 23.7 (16 of 44) | 23.7 (16) | 25.2 (58 of 198) | 25.2 (58) | 25.1 | 25.1 |
| GX5 | person_observations | 17 | 23.8 | 22.3 (117 of 1152) | 22.3 (117) | 21.0 (758 of 2038) | 25.2 (102) | 23.7 | 21.2 |
| Am2-150 | baseline | 30 | 3.3 | 5.2 (6 of 4172) | 5.2 (6) | 3.2 (702 of 3376) | 8.0 (36) | 8.7 | 5.4 |
| Am2-150 | paint_observations | 30 | 3.3 | 8.9 (2 of 1162) | 8.9 (2) | 3.3 (238 of 1240) | 3.3 (238) | 6.6 | 6.6 |
| Am2-150 | person_observations | 30 | 3.3 | 5.2 (6 of 4922) | 5.2 (6) | 3.2 (568 of 3078) | 4.6 (429) | 8.7 | 5.4 |
| Am2-28019 | baseline | 15 | 3.0 | 4.9 (3 of 7102) | 4.9 (3) | 3.9 (729 of 3958) | 60.9 (345) | 61.6 | 5.9 |
| Am2-28019 | paint_observations | 15 | 3.0 | 5.0 (1 of 298) | 5.0 (1) | 5.5 (1 of 764) | 5.5 (1) | 7.6 | 7.6 |
| Am2-28019 | person_observations | 15 | 3.0 | 4.9 (2 of 4366) | 4.9 (2) | 3.9 (190 of 2632) | 3.9 (190) | 5.9 | 5.9 |
| Am3-0 | baseline | 43 | 3.7 | 5.2 (6585 of 23758) | 5.4 (366) | 3.9 (15 of 1930) | 3.9 (15) | 4.3 | 4.3 |
| Am3-0 | paint_observations | 43 | 3.7 | 6.3 (1 of 66) | 6.3 (1) | 3.9 (8 of 290) | 3.9 (8) | 6.2 | 6.2 |
| Am3-0 | person_observations | 43 | 3.7 | 5.2 (6584 of 23758) | 5.4 (367) | 3.9 (14 of 1930) | 3.9 (14) | 4.3 | 4.3 |
| SS03-17 | baseline | 1 | 2.0 | 3.4 (506 of 1064) | 3.4 (506) | 2.2 (684 of 10254) | 3.0 (214) | 3.5 | 3.5 |
| SS03-17 | paint_observations | 1 | 2.0 | 3.4 (134 of 272) | 3.4 (134) | 2.2 (146 of 5376) | 2.2 (146) | 3.5 | 3.5 |
| SS03-17 | person_observations | 1 | 2.0 | 3.4 (506 of 1064) | 3.4 (506) | 3.0 (172 of 9484) | 3.0 (172) | 3.5 | 3.5 |
| SS03-19 | baseline | 1 | 2.3 | 3.2 (7 of 758) | 3.2 (7) | 2.9 (325 of 9469) | 2.9 (325) | 3.1 | 3.1 |
| SS03-19 | paint_observations | 1 | 2.3 | 3.2 (4 of 150) | 3.2 (4) | 2.9 (70 of 5640) | 2.9 (70) | 3.1 | 3.1 |
| SS03-19 | person_observations | 1 | 2.3 | 3.2 (6 of 584) | 3.2 (6) | 2.9 (1466 of 8761) | 2.9 (368) | 3.1 | 3.1 |
| SS03-16 | baseline | 1 | 1.5 | 2.4 (32 of 962) | 2.4 (32) | 1.5 (167 of 8358) | 1.5 (167) | 2.3 | 2.3 |
| SS03-16 | paint_observations | 1 | 1.5 | 2.4 (8 of 234) | 2.4 (8) | 1.5 (59 of 5002) | 1.5 (59) | 2.3 | 2.3 |
| SS03-16 | person_observations | 1 | 1.5 | 2.4 (32 of 962) | 2.4 (32) | 1.6 (523 of 8060) | 3.7 (196) | 3.7 | 2.3 |
| SS21-20 | baseline | 0 | 1.0 | 1.4 (0 of 1230) | 1.4 (0) | 1.8 (26 of 9162) | 1.8 (26) | 1.6 | 1.6 |
| SS21-20 | paint_observations | 0 | 1.0 | 1.4 (0 of 158) | 1.4 (0) | 1.7 (6 of 3868) | 1.7 (6) | 1.5 | 1.5 |
| SS21-20 | person_observations | 0 | 1.0 | 1.4 (0 of 1230) | 1.4 (0) | 1.8 (30 of 8642) | 1.8 (30) | 1.6 | 1.6 |

With paint-filtered observations the kept-matching nearest moves from 34.3 to 12.6 px on GX0 and from 61.6 to 7.6 on Am2-28019. It changes by at most 2.1 px on the broadcast views and Am2-150, and costs 1.9 px on Amateur-3 and 1.4 on GX5, views where the baseline had not been cut. With person-filtered observations GX0 goes to 8.3 and Am2-28019 to 5.9; six views stay within 0.1 px and SS03-16 loses 1.4.

**Matcher run.** The table gives, per view and arm at the generation stage, the nearest pooled court before the global cap, then the line winner, then the paint winner (working px to the control; "none" means no eligible winner). The baseline column is the direction experiment's accounting of the saved baseline records; every other cell comes from `runs/<run>/matcher/comparison.md`, which `account.py` wrote after gating all 36 records against their inputs.

| View | Control | baseline | paint_observations | person_observations | person | paint |
| --- | --- | --- | --- | --- | --- | --- |
| GX0 | approved | 7.7 / 16.5 / 7.7 | 9.0 / 21.1 / 21.1 | 7.7 / 10.6 / 8.3 | 1.9 / 1.9 / 10.9 | 151.9 / 3419.6 / 3419.6 |
| GX5 | approved | 23.7 / 513.6 / 525.7 | 25.1 / 2143.1 / 2143.1 | 23.7 / none / none | 11.1 / 365.9 / 365.9 | 10.0 / none / none |
| Am2-150 | approved | 8.3 / 140.9 / 12.4 | 6.6 / 12.4 / 8.3 | 8.3 / 140.9 / 12.4 | 73.3 / 1540.0 / 1540.0 | 37.2 / 137229.7 / 8099.4 |
| Am2-28019 | manual | 15.3 / none / none | 7.6 / 11.3 / 11.3 | 6.0 / 143.1 / 6.4 | 4.6 / 146.0 / 11.1 | 5.2 / 11.0 / 11.0 |
| Am3-0 | manual | 4.3 / 5.8 / 8.5 | 5.7 / 10.1 / 10.1 | 4.3 / 5.8 / 8.5 | 23.6 / 32.6 / 23.7 | 11.1 / 21.5 / 12.5 |
| SS03-17 | manual | 3.6 / 8.4 / 7.6 | 3.7 / 10.6 / 8.4 | 3.5 / 8.4 / 8.4 | 3.5 / 8.4 / 8.4 | 3.1 / 9.2 / 7.0 |
| SS03-19 | approved | 3.1 / 3.9 / 3.9 | 3.1 / 3.9 / 3.9 | 3.1 / 3.9 / 6.5 | 7.9 / 15.6 / 15.6 | 4.5 / 514.9 / 514.9 |
| SS03-16 | approved | 2.3 / 2.6 / 3.9 | 2.3 / 2.6 / 3.9 | 3.7 / 3.9 / 3.9 | 28.1 / 29.6 / 149.4 | 2.7 / 2.7 / 4.1 |
| SS21-20 | approved | 2.0 / 2.0 / 2.0 | 2.0 / 2.0 / 2.0 | 2.0 / 2.0 / 2.0 | 17.3 / 28.9 / 79.5 | 13.6 / 486.1 / 486.9 |

**Observation-only arms: the axis-stage gains reach the winners on the Amateur-2 views and cost a few pixels on GX0 and Amateur-3.** With paint-filtered observations, Amateur-2 frame 150's line winner moves from the mat border to a 12.4 px court, and frame 28019 gets winners at 11.3 px where the baseline had none. The four broadcast views keep their baseline winners within about two pixels. GX0 and Amateur-3 lose a few pixels on every measure. With person-filtered observations, GX0's line winner improves, frame 28019 gets a 6.4 px paint winner while the mat border still wins its line ranking, GX5 loses both winners, and two broadcast views slip by one to three pixels on one winner. The person mask leaves the mat border in place; the paint rule removes it because it is not white.

**Own-direction arms: the local replay's lottery is the matcher's lottery.** Reselecting directions from person-filtered fragments gives GX0 a 1.9 px line winner, the closest court any arm has produced on the hardest view, and halves GX5's nearest pooled court without an eligible winner there. The same arm breaks five of the other seven views (Amateur-2 frame 150, Amateur-3, SS03-16, SS21-20, SS03-19; table). Reselecting from paint-filtered fragments helps GX5's pool, Amateur-2 frame 28019 and SS03-16, and breaks GX0, Amateur-2 frame 150, Amateur-3, SS03-19 and SS21-20.

**The local replay predicts the matcher's nearest pooled court.** The kept-matching nearest court on the best-fit pair, measured locally in seconds, equals the matcher's nearest pooled court to within 0.1 px on most view-arms and never points the wrong way (the per-view pairs are in `evidence.md`). Where the two differ, the matcher found a closer court in another direction pair, or the proxy over-read a view whose pool is fed by a pair other than the best-fit one. The ranked winners are not predicted by that measure; the mat border winning the line ranking on the Amateur-2 views and the GX5 failures are ranking-stage outcomes.

At the final all-camera stage the pattern holds (`comparison.md`, second table). Two cells change meaning: SS03-19's baseline paint winner is a known 6,297 px false court, which the paint-filtered arm replaces with a 143.5 px false court and the person-filtered arm with a 6.5 px court; and GX0's line winner under paint-filtered observations improves to 10.7 px while its paint winner stays at 20.3.

## What the result means

Two of the detector's three known loss sites now have a mechanism and a measured lever, and the third is unchanged.

- Direction selection admits non-court structures (the wall seam, floor strip and face on GX0) and its coverage rule is unstable under small changes to the fragment set. Filtering fragments before it is therefore a lottery, and the matcher confirms it: the person-box rule gives GX0 a 1.9 px line winner, the closest any arm has produced on the hardest view, and breaks five of the other eight. That 1.9 px shows what the matcher delivers once the clutter is out of the directions; the breakages show the coverage rule handing the other views directions that lose their courts to the cap. A route through direction selection needs a selection rule that does not flip on a few fragments, before any fragment filter can be judged there.
- The axis matching's per-direction cap discards the court's own matching when many parallel non-court lines compete, because its score rewards any five tightly aligned lines and the cap keeps 512 of tens of thousands. Filtering the fragments the axis matching sees, with directions held fixed, is the first change in this series that moves views without a lottery: in the matcher it repairs both Amateur-2 views' winners and leaves the broadcast views alone, at a cost of two to four pixels on GX0 and Amateur-3, where the filter also removes some marking fragments.
- The local axis-stage replay predicts the matcher's nearest pooled court to within 0.1 px on most view-arms and never in the wrong direction. Changes to the cap's ordering or to the fragment rules can therefore be screened in minutes per view with `filter_replay.py` and `axis_replay.py` before any compute-host run; only the ranked winners still need the matcher.
- The ranking stage is unchanged in code, but the paint rule reaches it: with the mat-border fragments gone, the line ranking on Amateur-2 frame 150 picks a 12.4 px court instead of the 140.9 px border. The person mask does not remove the border and the border keeps winning under it. GX5 has no eligible winner under any arm that gets its pool within 11 px.

None of this is a readiness result. The nine views are development data, the paint rule was shaped on GX0 and its thresholds read from control-labelled quantiles on the same views, and the person-box rule masks wherever any player stood in a three-second window.

## What should usefully happen next

1. Replace the per-direction cap's order. The matchings that reach the court support five of five or five of six markings but score lower than tighter alignments of clutter. Ordering the kept set by supported marking count before score, or keeping a diverse set in scale and offset rather than the top 512 by score, can be replayed locally with `axis_replay.py` in minutes per pair before any matcher run.
2. Keep the paint rule's observation-only form as the candidate change for the matcher's fragment set: it reached the winners on both Amateur-2 views and cost GX0 and Amateur-3 two to four pixels. Before adopting it, measure why those two lose (which marking fragments the rule removes there) and whether a cap-ordering change recovers them without the filter.
3. Leave direction-stage filtering alone until the coverage rule is stabilised. The person-box rule's GX gains are real but arrive by reallocation, and the same reallocation breaks four views.
4. The per-view calibration audit stays on hold; nothing here changes the reasons given in the follow-ups schedule.

## How this affects the larger goal

The goal is a court detector that needs neither ground truth nor CourtKeyNet. This series has not produced one. It has turned "the matcher loses close courts somewhere" into two named mechanisms with local replays that reproduce the records exactly, and it has found one lever, filtering the fragments the axis matching sees, that helps the hardest views without breaking the easy ones at the axis stage. The next design changes are in the cap's ordering and the direction rule's stability, both of which can be tested locally in minutes with the scripts in this folder before spending compute-host time.
