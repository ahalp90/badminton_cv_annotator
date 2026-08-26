# Contact detector exploration

This directory is the readable record of the contact-detection work.

If you are coming back to this after doing five other things, start here. The short version is:

- the learned contact detector is much better at finding contact times than the old heuristic path;
- that improvement does **not** yet turn into a large clean set of fully correct rallies;
- most missed contacts in the pilot had a plausible candidate nearby, so simply widening the search is not the main next move;
- raw per-frame motion beat the tested frame-rate-normalised alternatives on these three videos, but that is a pilot result, not a general rule;
- a slightly wider duplicate-removal distance was the best of the five cheap decision-layer variants tested, but the distance is a pilot setting and must be chosen again on the larger dataset;
- the selected event stream now has a direct side result: **75.2%** of labelled contacts are found at the right time with the correct player side, with **73.9% joint event-and-side F1**;
- the tested broad “anchor plus one nearby alternative” shortlist failed its predeclared recovery requirement, but an oracle shows the same frozen candidate union still contains complete-rally headroom;
- a compact serve-prefix list is a cleaner bounded second-stage lead, while the tested hand-written chooser and the simple serve-lookback threshold both failed.

These results come from only three videos. They are useful for choosing what to try next, not for declaring final thresholds or constants.

If you want the shortest route through the results, open the [`visual quick guide`](VISUAL_QUICK_GUIDE.md).

## Table of contents

- [What this work was trying to learn](#what-this-work-was-trying-to-learn)
- [The useful answer in one table](#the-useful-answer-in-one-table)
- [What happened after the earlier readable checkpoint](#what-happened-after-the-earlier-readable-checkpoint)
- [What the results mean](#what-the-results-mean)
- [What should be tested on the larger dataset](#what-should-be-tested-on-the-larger-dataset)
- [Where the other reports fit](#where-the-other-reports-fit)
- [Compact reference for the old experiment codes](#compact-reference-for-the-old-experiment-codes)
- [Pilot limits](#pilot-limits)

## What this work was trying to learn

The goal is not to force an answer for every rally.

The useful product is a set of rallies that are correct from end to end, while the system abstains on difficult ones. The three-video pilot was meant to choose the **shape of the system**: where to search, what kind of model to use, how to turn model scores into events, and whether a second stage has enough useful work to do.

The working path is:

```text
broad, label-blind search region
→ HGB contact score
→ turn scores into contact events
→ existing Top/Bottom player-side rule
→ keep or reject the whole rally
```

The search region is deliberately broad. It is not the detector. Its job is to avoid throwing away real contacts before the classifier sees them.

Region version 2 searches about **31.9% of source frames** and has a candidate within ±10 base-30 frames of **98.3% of labelled contacts**. That is already enough coverage for this pilot to ask whether the classifier and event-selection layer are the real bottlenecks.

## The useful answer in one table

| Question | What was tested | What happened | What it means |
| --- | --- | --- | --- |
| Can the learned model improve contact timing? | Region-v2 histogram gradient boosting (HGB) versus the current heuristic path | Timing F1 rose from **72.6% to 87.4%** using the original HGB decisions | Yes. Learned timing is clearly worth keeping |
| Does better event timing already give clean whole rallies? | Fixed rally spans, fixed Top/Bottom rule, strict end-to-end rally scoring | Original HGB decisions gave **21 fully correct rallies out of 291 kept** with no score cut. At a 0.90 whole-rally timing cut, **9 / 51** were fully correct | No. Event F1 is not the same thing as usable complete-rally output |
| Are most missed contacts outside the search region? | Audit of the 296 contacts missed by the original HGB event stream | **244 / 296** already had a candidate nearby in the fixed search surface; this included **89 / 95** missed serves | Usually no. The next gains are more likely in scoring/selection than in simply searching more video |
| Is frame-rate-sensitive raw motion hurting the model? | Existing raw motion, motion removed, and motion scaled to a common 30 fps convention | Raw motion had **87.4% F1** and **21** fully correct rallies with no confidence cut; common-30 scaling had **87.0%** and **15**; dropping motion had **84.8%** and **16** | Raw motion won this pilot. That does **not** prove it is generally better |
| Can a cheap decision-layer change help without refitting HGB? | Five fixed ways to turn the same held-out scores into events | Using a **6-base-30-frame** duplicate-removal distance gave **88.8% timing F1** and **27 / 291** fully correct rallies | Worth carrying forward as a pilot configuration, but the distance is not a decided constant |
| How good is the selected stream when player side is included? | Existing Top/Bottom rule replayed on the selected event stream | **75.2% timing + correct-side recall**, **73.9% joint event-and-side F1**, **56.2% serve timing + correct-side recall** | The timing gain survives the side check, but player-side attribution remains a major limit |
| Does the broad nearby-alternative shortlist justify a second stage? | Keep every selected event and add one label-blind nearby alternative around each | Candidates grew from **3,238 to 6,305**; coverage rose from **90.3% to 93.4%**; only **97 of 303** misses were recovered. The gate required **152** | This exact shortlist is too large for the measured event-coverage gain |
| Is there still useful evidence inside that broad candidate union? | Exact non-deployable oracle chooses the best subset of the same frozen union | At ±10, fully correct rallies can rise **27 → 42** with timing and side both correct; timing alone is feasible for **144** rallies | Yes. There is headroom, but no validated selector has realised it |
| Can a compact serve-specific candidate list help? | Up to five frozen candidates before each detected span's first selected event | A candidate lies near **60 of 96** missed serves; timing oracle recovers **58** and raises fully correct rallies **27 → 29** | This compact list is worth a fresh-data selector experiment |
| Did the tested serve chooser work? | Fixed hand-written choice from the compact serve list | It adds 79 events, finds only 8 new serves, leaves 70 added events unmatched, and drops fully correct rallies **27 → 16** | No. Keep the candidate idea, discard the chooser |
| Does simply lowering the threshold in the serve look-back help? | Lower score only in the existing label-blind serve-lookback region | +5 events and +2 serve timing matches, but **0** more correctly sided serves and **0** more fully correct rallies | Close this exact threshold idea |

## What happened after the earlier readable checkpoint

The earlier work had already established the broad search region and the basic HGB result. The follow-up work then asked several narrower questions.

### 1. We scored the output as complete rallies, not just individual contacts

A kept rally has one fixed meaning throughout these follow-ups. It counts as fully correct only when:

- the predicted span maps to exactly one real rally;
- every labelled contact is found within the chosen timing tolerance;
- there are no extra contact events;
- every contact has a Top/Bottom answer;
- every Top/Bottom answer is correct.

A missing side answer rejects the whole rally.

That is intentionally strict because it matches the actual utility we want: a rally we can use without repairing it by hand.

The important result is that the current confidence signal is not enough to find a large clean subset. With the original HGB decisions, the system kept 291 spans with no score cut and only 21 were fully correct. Raising the whole-rally timing requirement to 0.90 kept just 51 spans, of which nine were fully correct.

The wider duplicate-removal variant improved this to 27 fully correct rallies with no score cut and 13 out of 68 at 0.90. Better, but still not a clean-output system.

![Whole-rally confidence versus yield for the original and wider duplicate-removal decisions.](figures/followup_rally_yield_curve.png)

### 2. We checked whether the misses were actually recoverable nearby

The original HGB stream missed 296 of 3,128 labelled contacts at ±10.

A candidate already existed near **244** of those misses. That included **89 of the 95 missed serves**. All **13** predicted spans that were otherwise exact apart from one missing contact also had a region-v2 candidate nearby.

That is useful because it changes the next question. The obvious problem is not “search almost the whole video.” It is “make better decisions among candidates we already have.”

There is still a real off-region problem, especially in `sset_21`, but it is not the first thing to expand for the perfect-rally goal.

### 3. We tested the frame-rate concern

Two pilot videos are 25 fps and one is 30 fps. Some motion inputs are raw movement per frame, so it was reasonable to worry that the same physical movement could have different numerical values at different frame rates.

We tested:

- the existing raw per-frame motion;
- removing the frame-rate-sensitive motion values;
- converting those motion values to a common 30 fps scale.

The existing raw values won on these three videos.

![Pilot frame-rate motion feature check.](figures/followup_motion_feature_check.png)

This is a **pilot result**, not a general conclusion that raw motion is intrinsically better. There are too few videos and too little frame-rate diversity for that claim. The larger dataset should test this again.

### 4. We tested cheap ways to turn the same HGB scores into events

HGB was not refit. The experiment only changed the small decision layer after scoring.

The best of the five tested rows used the original score cut-off and increased the distance used to remove nearby duplicate events from 5 to **6 base-30 frames**.

That variant:

- reduced predicted events from 3,350 to **3,238**;
- raised timing precision from 84.5% to **87.2%**;
- kept timing recall almost flat, 90.5% to **90.3%**;
- raised timing F1 from 87.4% to **88.8%**;
- increased fully correct rallies with no confidence cut from 21 to **27**.

A lower score threshold found more contacts and more serves, but its extra events damaged complete-rally output. Lowering the threshold only near detected rally starts did not beat the wider duplicate-removal choice either.

![Timing F1 and serve recall for the five cheap decision-layer variants.](figures/followup_decision_layer_tradeoff.png)

The **6-base-30-frame distance is not a project constant**. It is simply the best tested setting on this three-video pilot. Choose it again on the larger dataset.

### 5. We tested whether a second-stage shortlist had enough useful information

The tested shortlist was deliberately simple and label-blind:

- keep every event from the selected wider-duplicate-removal stream;
- around each one, add the strongest nearby alternative that sits outside the first event's duplicate-removal distance;
- then deduplicate the combined list.

That nearly doubled the candidate list from **3,238 to 6,305**.

At ±10, contact coverage rose from **90.3% to 93.4%**. The shortlist recovered **97 of the 303 contacts** missed by the selected event stream, but the predeclared gate required at least 152 recoveries while staying under twice the original list size.

The size test passed. The recovery test failed.

![The tested shortlist added many candidates for a modest coverage gain.](figures/followup_shortlist_tradeoff.png)

So this exact anchor-plus-nearby-alternative shortlist should stop here on the pilot.

That is not the same as saying the evidence has no headroom. A separate oracle check below shows that the frozen union contains more complete-rally potential than the current selector can realise.

### 6. We measured the selected stream with player side included

The existing Top/Bottom rule has now been replayed on the selected wider-duplicate-removal stream.

At ±10, that selected stream has:

- **83.4%** side accuracy on answered timing matches;
- **75.2%** timing + correct-side recall;
- **73.9%** joint event-and-side F1;
- **56.2%** serve timing + correct-side recall.

So the selected stream still improves the combined timing-and-side output over the old heuristics. The remaining gap is not just contact timing: player-side attribution matters too.

![The selected HGB stream improves combined timing-and-side and serve output over the old heuristics.](figures/followup_side_and_serve_summary.png)

### 7. We checked how much complete-rally headroom exists inside the frozen candidate union

The broad shortlist failed its practical gate, but that left a different question: if we could magically choose the right subset of the same already-frozen candidates, are more correct rallies even possible?

The oracle tests candidates inside unchanged spans that contain one labelled rally. Fifty-three candidates outside every span remain unassigned.

An exact oracle gives the upper bound. At ±10:

- the selected event stream has **27** fully correct rallies;
- **42** rallies are feasible when both timing and the existing Top/Bottom answers must be correct;
- **144** rallies are timing-feasible if player side is ignored.

The +15 full-rally gain passes the predeclared headroom gate.

![The frozen candidate union contains more rally-level evidence than the current selector can use.](figures/followup_candidate_union_ceiling.png)

This is **not achieved output** and it is not a deployable selector. It says the evidence itself still contains useful headroom. The much larger timing-only ceiling also shows that player-side attribution is the main evidence limit inside this union.

### 8. We tried a compact serve-prefix candidate list

The broad shortlist is noisy, so the serve-specific follow-up asks a narrower question: before a detected span's first selected event, can a tiny candidate list contain a missed serve?

Among the selected stream's **96 missed serves**:

- **60** have a frozen prefix candidate within ±10;
- a timing oracle recovers **58** new serve matches;
- no existing serve match is lost;
- fully correct rallies rise **27 → 29** with no timing-score cut.

![The compact serve-prefix list contains useful candidates, but the tested hand-written chooser finds only a small fraction of them.](figures/followup_serve_prefix_headroom.png)

The candidate list has useful headroom. The tested fixed chooser does not: it adds **79** events, finds only **8** new serves, leaves **70** added events unmatched, and drops fully correct rallies from **27 to 16**.

The useful conclusion is therefore: keep the compact candidate-list idea for a fresh-data selector experiment; discard the tested hand-written chooser.

The compact list was designed after inspecting aggregate results from this three-video pilot. Its result is development evidence, not a generalisation result.

### 9. We closed the simple serve-lookback threshold idea

A cheaper follow-up lowered the HGB score cut-off only in the existing label-blind serve-lookback region.

It adds five events and two serve timing matches, but adds:

- **zero** correctly sided serves;
- **zero** fully correct rallies.

That is enough to close this exact threshold idea on the pilot.

## What the results mean

### The contact detector itself is genuinely better

The old final heuristic path reaches **72.6% contact-timing F1**.

The original region-v2 HGB model reaches **87.4%**, and the best cheap event-selection variant reaches **88.8%** without refitting the model.

That is a meaningful gain and worth carrying into the larger experiment.

### The complete annotator is still the hard part

A high contact F1 can hide exactly the errors that break a rally: one missed stroke, one extra event, or one wrong player side.

Even after the best cheap decision-layer change, only **27 of 291** kept spans are fully correct with no timing-confidence cut. At a 0.90 cut, the result is **13 of 68**.

So the next work should keep reporting whole-rally yield and correctness first, with contact metrics as diagnostics.

### `sset_21` is the warning not to over-read pooled scores

Region v2 contains **71 of 75** serves in `sset_21`, so search coverage is not the main problem there.

The original HGB decisions find **33 of 75** serves. The wider duplicate-removal variant finds **32 of 75**.

The pooled decision-layer improvement therefore does not solve the weakest fixture. With only three videos, we do not know whether this is a one-off broadcast style, a frame-rate interaction, or a broader serve problem.

### Player-side attribution is now measured on the selected stream

For the selected wider-duplicate-removal stream, the existing Top/Bottom rule gets the correct side on **83.4%** of answered timing matches. Across all labelled contacts, **75.2%** are found at the right time with the correct side, and joint event-and-side F1 is **73.9%**.

The old final heuristics reach **70.6%** timing-plus-correct-side recall, so the selected HGB stream still improves the combined output.

The candidate-union oracle makes the remaining limit clearer: timing alone is feasible for 144 rallies at ±10, but only 42 are feasible when the current side answers must also be correct. Player-side attribution is therefore a major remaining evidence limit inside that union.

The separate rally-level alternation fit genuinely has **not** been rerun on the new contact events.

## What should be tested on the larger dataset

The larger run should refit and reselect everything that was fitted on these three videos.

1. **Refit the HGB contact model with whole-video splits.** Keep the region-v2 idea and the physical + validity inputs, but do not reuse the pilot tree.

2. **Retest raw motion against frame-rate-normalised motion.** The raw values won here, but three videos are not enough to settle the convention.

3. **Choose score cut-offs and duplicate-removal distance again.** Carry forward the idea that a slightly wider suppression window helped; do not carry forward “6” as a constant.

4. **Look at serves as a separate failure slice.** If the larger error table shows the same pattern, test start-specific features, weighting, or decision rules. Do not optimise them because one pilot video was awkward.

5. **Keep player-side scoring in the main scoreboard.** Measure contact timing, side accuracy on answered timing matches, combined timing-plus-side recall, and joint event-and-side F1 separately.

6. **Rerun the strict kept-rally curve.** Use the same fixed meaning of a fully correct kept rally so improvements remain comparable.

7. **Test a serve-prefix selector only on fresh whole videos or with nested cross-fitting.** The compact list has real timing headroom; the fixed chooser failed.

8. **Only revisit a broad second stage if it finds more useful contacts per added candidate.** The current broad shortlist failed its predeclared recovery requirement. The oracle shows the union still has theoretical headroom, but no validated label-blind selector has realised it.

9. **Treat player-side attribution as part of the second-stage problem.** The timing-only candidate-union ceiling is far above the timing-plus-side ceiling.

10. **Do not spend more pilot effort on the serve-lookback threshold.** It added no correctly sided serve and no complete rally.

11. **Only widen off-region rescue search if it buys complete rallies.** The original HGB miss audit found that all 13 otherwise-exact one-missing spans already had a region-v2 candidate nearby.

Where possible, hold out entire tournaments or broadcast packages so near-duplicate footage does not leak across training and test.

## Where the other reports fit

| File | Read it when you need to know... |
| --- | --- |
| [`README.md`](README.md) | the whole story and what to do next |
| [`VISUAL_QUICK_GUIDE.md`](VISUAL_QUICK_GUIDE.md) | the six pictures that explain the result fastest |
| [`auto_annotator_progress.md`](auto_annotator_progress.md) | what the current annotator can actually output correctly |
| [`tree_contact_detector_results.md`](tree_contact_detector_results.md) | what each contact-detector experiment tested and what it showed |
| [`bst_x_contact_detector_plan.md`](bst_x_contact_detector_plan.md) | the separate implementation plan for a BST-X neural detector |

## Compact reference for the old experiment codes

The codes are kept here only so old logs and result files remain searchable.

| Code | Plain-language description |
| --- | --- |
| `B0` | Original HGB score cut-off and 5-base-30-frame duplicate-removal distance |
| `T−` | One lower HGB score cut-off everywhere; duplicate-removal distance unchanged |
| `N−` | Original score cut-off with a smaller 4-base-30-frame duplicate-removal distance |
| `N+` | Original score cut-off with a wider 6-base-30-frame duplicate-removal distance |
| `S−` | Lower score cut-off only near label-blind detected rally starts |
| `L−` | Lower score cut-off only in the existing serve-lookback region |
| `SL−` | Lower score cut-off in both the rally-start and serve-lookback regions |

In ordinary prose, use the descriptions rather than these codes.

## Pilot limits

The measured pilot contains:

- **3 videos**: `sset_01`, `sset_15`, `sset_21`;
- **292 rallies**;
- **3,128 contacts**;
- **292 serves**.

These videos come from the same dataset and provide very little diversity. The results are useful for deciding which ideas deserve a larger test. They are not strong evidence for final constants, thresholds, frame-rate conventions, or production performance.
