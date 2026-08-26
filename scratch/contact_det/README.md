# Contact detector exploration

This directory is the readable record of the contact-detection work.

If you have been away from this project for a while, start here. The main learned detector is a tree model called histogram gradient boosting (HGB). The short version is:

- the learned contact detector is much better at finding contact times than the old heuristic path;
- the best cheap event-selection rule in this pilot reaches **88.8% contact-timing F1**, up from **72.6%** for the old final heuristics;
- that does **not** yet give a clean end-to-end annotator: the selected stream has **27 fully correct rallies among 291 predicted spans that can be scored** with no minimum-score filter, and **13 / 68** at a 0.90 filter;
- most missed contacts already had plausible evidence nearby, so simply searching more of the video is not the main next move;
- the wider duplicate-removal experiment is more interesting than it first looked: it produced 112 fewer events overall, with **105 fewer unmatched predictions** and seven fewer timing matches;
- the strict whole-rally records show a larger cleanup opportunity behind that result: on the original HGB output, **21** rallies are fully correct now, while **38** could be fully correct if an ideal cleanup stage only removed extra events;
- another **13** rallies are otherwise exact apart from one missing contact, so a small rescue path has a separate, smaller job;
- the weakest HGB score in a rally is not a good enough trust signal by itself. Short predicted event lists were cleaner. A future confidence model should test the number of predicted contacts in the rally, and may also use span duration;
- player-side attribution remains a major limit: the frozen broad candidate union can support **144 timing-exact rallies**, but only **42** when the current Top/Bottom answers must also be correct;
- `sset_21` is still the warning against over-reading pooled numbers, especially for serves;
- these are three videos from one dataset. We have a more plausible route to a high-precision abstaining annotator, but we have **not** shown near-100% kept-rally precision or cross-broadcast generalisation.

If you want the shortest route through the results, open the [`visual quick guide`](VISUAL_QUICK_GUIDE.md).

## Table of contents

- [What this work was trying to learn](#what-this-work-was-trying-to-learn)
- [The useful answer in one table](#the-useful-answer-in-one-table)
- [Are we getting closer to a near-100%-precision annotator?](#are-we-getting-closer-to-a-near-100-precision-annotator)
- [What the follow-up experiments changed](#what-the-follow-up-experiments-changed)
- [What should be tested on the larger dataset](#what-should-be-tested-on-the-larger-dataset)
- [Where the other reports fit](#where-the-other-reports-fit)
- [Pilot limits](#pilot-limits)

## What this work was trying to learn

The goal is not to force an answer for every rally.

The useful product is a set of rallies that are correct from beginning to end, while the system abstains on rallies it cannot trust. It is acceptable to keep only a small share of rallies if those rallies are genuinely clean.

The three-video pilot was meant to choose the **shape of the system**, not final settings:

```text
broad, label-blind search region
→ learned contact score
→ turn scores into contact events
→ player-side attribution
→ optional event cleanup / bounded rescue
→ keep or reject the whole rally
```

The search region is deliberately broad. It is fixed without using the true contact locations, and it is not the detector. Its job is to avoid throwing away real contacts before the classifier sees them.

Region version 2 searches about **31.9% of source frames**. It has a candidate within ±10 frames of **98.3% of labelled contacts**, after converting each video to a 30 fps-equivalent clock. That tolerance is about one-third of a second.

This is enough coverage for the pilot to focus on scoring, event selection, side attribution and whole-rally acceptance.

## The useful answer in one table

| Question | What was tested | What happened | What it means |
| --- | --- | --- | --- |
| Can the learned model improve contact timing? | Region-v2 histogram gradient boosting (HGB) versus the current heuristic path | Timing F1 rose from **72.6% to 87.4%** using the original HGB decisions | Yes. Learned timing is clearly worth keeping |
| Can a cheap event-selection change help without refitting HGB? | Five fixed ways to turn the same held-out scores into events | Wider duplicate removal reached **88.8% timing F1** and **27 / 291** fully correct rallies | Yes, but reselect the distance on larger data |
| What did wider duplicate removal actually do? | Compare its saved overall totals with the original HGB stream | It produced **112 fewer events** overall: **7 fewer matches** and **105 fewer unmatched predictions** | The gain comes from fewer unmatched predictions, not higher recall |
| Does better contact timing already give clean whole rallies? | Strict end-to-end rally scoring | Selected stream: **27 / 291** fully correct with no minimum-score filter; **13 / 68** at 0.90 | No. Event F1 is not end-to-end correctness |
| Can minimum HGB score alone identify trustworthy rallies? | Whole-rally score filters and diagnostic predicted-event-count slices | A 0.90 filter on the original stream gives **9 / 51 = 17.6%** fully correct; short predicted event lists are cleaner | No. Whole-rally confidence needs more context |
| Are most missed contacts outside the search region? | Audit of 296 original-HGB misses | **244 / 296** already had a candidate nearby; **89 / 95** missed serves did too | Usually no. Search expansion is not the first move |
| Is there useful cleanup headroom? | Upper-bound check on original strict span records | **21 → 38** fully correct if an ideal selector only deletes extras | Yes. Extra-event cleanup is a concrete second-stage job |
| Is there a separate rescue opportunity? | Otherwise-exact one-missing rallies | **13** rallies are one contact short; all 13 have a region-v2 candidate nearby | Yes, but it is smaller and should be kept bounded |
| Does the broad nearby-alternative shortlist justify a second stage? | Add one nearby alternative around selected events | Candidates **3,238 → 6,305**, but only **97 / 303** misses recovered; we had set 152 as the minimum | This exact shortlist is too noisy |
| Is there still evidence inside that broad union? | Exact non-deployable subset oracle | **27 → 42** timing-and-side-correct rallies; timing alone is feasible for **144** | Yes. Selection and side attribution are still leaving value on the table |
| Can a compact serve-specific candidate list help? | Up to five frozen candidates before each detected span's first event | A candidate lies near **60 / 96** missed serves; a label-informed upper bound could recover 58 | Worth a fresh-data selector test |
| Did the tested serve chooser work? | Fixed hand-written chooser | Only 8 new serves; fully correct rallies fall **27 → 16** | No. Keep the candidate-list idea, discard the chooser |
| Is the system shown to generalise across broadcast conventions? | Three same-dataset fixtures, including a difficult `sset_21` | Results vary materially by fixture; no external broadcast test was run | No. Generalisation is still an open requirement |

## Are we getting closer to a near-100%-precision annotator?

**Closer in system design and diagnosis: yes. Close in measured end-to-end precision: no.**

The useful progress is that the problem is becoming separable:

1. **Search is mostly broad enough.** Region v2 makes almost every labelled contact available in this pilot.
2. **Contact timing is much better.** HGB plus the selected event rule is a large step up from the old heuristics.
3. **We now know what breaks rallies.** Extra events, missing events and wrong player side can be measured separately.
4. **Cleanup has a real job.** The pilot contains many rallies that already have every true contact and correct side answer but are spoiled by extra events.
5. **Abstention is being treated as a rally-level problem.** That is the right shape for a high-precision, low-recall annotator.

But the current system is nowhere near the target yet. With the selected event stream and a 0.90 weakest-contact score filter, only **13 of 68 kept rallies (19.1%)** are fully correct. A single raw HGB score is not close to being a near-100%-precision acceptance rule.

The more promising route is therefore not “raise the score threshold until almost nothing remains.” It is:

```text
strong first-stage contact model
→ remove likely extra events
→ bounded rescue for a small one-missing set
→ reliable player-side answers
→ rally-level acceptance model using held-out predictions
→ abstain aggressively when the whole rally is not trustworthy
```

Short predicted event lists were cleaner in this pilot. That is a useful clue, not a reason to reject long rallies. A future confidence model should test the number of predicted contacts in the rally. Span duration may also help.

Use the true labelled rally length only when reporting results. Show how many short and long rallies the model keeps.

Generalisation is a separate open question. These three fixtures are too small and too similar to show that the system will survive new tournaments, camera layouts, graphics packages or production conventions. The larger programme needs whole-video and, where possible, whole-broadcast-package holdouts.

## What the follow-up experiments changed

### 1. We stopped treating contact F1 as the product score

A rally is useful only if it can be consumed without repair. Throughout the follow-ups, a kept rally is fully correct only when:

- the predicted span maps to exactly one real rally;
- every labelled contact is found;
- there are no extra predicted contact events;
- every event has a Top/Bottom answer;
- every Top/Bottom answer is correct.

This strict score is intentionally unforgiving because the intended downstream dataset is unforgiving too.

### 2. We learned that most misses are selection misses, not search misses

The original HGB stream missed **296 of 3,128** labelled contacts at ±10.

A candidate from the fixed search surface already existed near **244** of them. All **13** otherwise-exact one-missing rallies had a region-v2 candidate near the missing contact.

So “search more video” is not the obvious first response.

### 3. We checked the frame-rate concern

Two pilot videos are 25 fps and one is 30 fps.

Existing raw per-frame motion reached **87.4% F1** and 21 fully correct rallies. Common-30 scaling reached **87.0%** and 15. Removing frame-rate-sensitive motion reached **84.8%** and 16.

Raw motion won this pilot. That is not a general rule. Retest it with real frame-rate and broadcast diversity.

### 4. Wider duplicate removal taught us something about cleanup

The selected wider duplicate-removal rule changes no fitted tree.

Compared with the original HGB decisions, its output has:

- **112 fewer** predicted events;
- **7 fewer** timing matches;
- **105 fewer** unmatched predictions;
- raises timing F1 from **87.4% to 88.8%**;
- raises fully correct rallies from **21 to 27**.

Most of the change is fewer unmatched predictions. That is a strong hint that removing extras deserves to be treated as its own downstream problem.

![Where the original HGB output has cleanup headroom.](figures/followup_cleanup_headroom.png)

The original strict span records make the opportunity clearer:

- **21** rallies are fully correct now;
- **38** could be fully correct if an ideal cleanup stage only removed extra events;
- another **13** are otherwise exact apart from one missing contact.

The last step gives a loose upper bound of **up to 51**, assuming the repaired event also gets the correct player side. This is separate from the 42-rally candidate-union result because the two checks use different event lists.

For concrete inspection cases, keep the target list at [`raw/followups/rally_cleanup_targets/contact_followup_rally_targets.csv.gz`](raw/followups/rally_cleanup_targets/contact_followup_rally_targets.csv.gz).

### 5. The broad second-stage shortlist still failed

The label-blind “selected event plus one nearby alternative” shortlist nearly doubled the candidate count, from **3,238 to 6,305**. Before running it, we decided it needed to recover at least 152 missed contacts. It recovered **97 of 303**.

Stop that exact shortlist on this pilot.

The failure does **not** mean “there is no second-stage job.” The cleanup analysis says the opposite: the clearest second-stage job may be deleting bad events, not searching for many more candidates.

### 6. The candidate-union oracle still matters

Inside the frozen broad union:

- selected stream fully correct: **27** rallies;
- timing-and-side feasible: **42**;
- timing-only feasible: **144**.

That says two things at once:

- there is selection headroom;
- player-side attribution is a major remaining evidence limit.

### 7. The serve-specific lead is narrower and cleaner

The compact serve-prefix list finds a candidate near **60 of 96** missed serves. A label-informed upper bound says that a perfect chooser could recover up to 58.

The tested fixed chooser is bad: it recovers only eight new serves and damages already-correct rallies.

Keep the list idea for fresh data. Drop the chooser.

### 8. Minimum contact score is not enough for rally acceptance

For the original HGB stream:

| Diagnostic slice | Spans | Fully correct | Fully correct within slice |
| --- | ---: | ---: | ---: |
| All scorable predicted spans | 291 | 21 | 7.2% |
| Minimum HGB score at least 0.85 | 123 | 13 | 10.6% |
| Minimum HGB score at least 0.90 | 51 | 9 | 17.6% |
| At most 7 predicted events | 126 | 19 | 15.1% |
| At most 4 predicted events | 67 | 13 | 19.4% |
| Score at least 0.90 and at most 5 events | 35 | 8 | 22.9% |

These are **diagnostics, not acceptance rules**. A maximum-contact cutoff would simply bias the retained dataset toward short rallies.

The useful lesson is that short predicted event lists are cleaner, and the same score filter behaves differently across fixtures. A 0.90 filter keeps all six fully correct `sset_01` spans but only two of fourteen fully correct `sset_15` spans. The pilot does not separate the effects of true rally length, duration, fixture and difficulty.

A later acceptance model should test contact scores, player-side confidence and the number of predicted contacts in the rally. Span duration and ambiguity between nearby candidates may also help. The model can also check whether the predicted span maps cleanly to one rally.

Use the true labelled rally length only when reporting results. Show correctness and retention separately for short, medium and long rallies.

## What should be tested on the larger dataset

A sensible order is:

1. **Refit the first-stage contact model with whole-video splits.** Keep the region-v2 design and physical/validity feature idea, not the fitted pilot tree.
2. **Retest raw versus frame-rate-normalised motion.** The pilot is too small to settle this.
3. **Reselect score cut-offs and duplicate-removal distance.** Do not carry “6” forward as a constant.
4. **Test a small event-cleanup stage whose first job is deleting extras.** The 21 → 38 upper-bound result makes this the clearest second-stage opportunity.
5. **Test one bounded rescue source for otherwise-exact one-missing rallies.** Do not reopen a huge broad shortlist unless it recovers many more real contacts for each added candidate.
6. **Improve or replace player-side attribution.** Measure it directly on the selected stream because it can dominate the remaining rally gap.
7. **Learn rally acceptance from held-out predictions.** Test contact scores, ambiguity, span quality, player-side confidence, span duration and the number of predicted contacts in the rally. Use true rally length only to report correctness and retention for short, medium and long rallies.
8. **Keep serves as a separate slice.** Especially check whether the `sset_21` failure pattern repeats.
9. **Only widen off-region search when it buys otherwise-complete rallies.**
10. **Hold out entire videos and, where possible, tournaments or broadcast packages.** That is the evidence needed for a generalisation claim.

Do not spend more pilot effort tuning the failed hand-written serve chooser or the simple serve-lookback threshold.

## Where the other reports fit

| File | Read it when you need to know... |
| --- | --- |
| [`README.md`](README.md) | the whole story and what to do next |
| [`VISUAL_QUICK_GUIDE.md`](VISUAL_QUICK_GUIDE.md) | the pictures that explain the result fastest |
| [`auto_annotator_progress.md`](auto_annotator_progress.md) | what the current annotator can actually output correctly |
| [`tree_contact_detector_results.md`](tree_contact_detector_results.md) | what each contact-detector experiment tested and what it showed |
| [`bst_x_contact_detector_plan.md`](bst_x_contact_detector_plan.md) | the separate implementation plan for a BST-X neural detector |

## Pilot limits

The measured pilot contains:

- **3 videos**: `sset_01`, `sset_15`, `sset_21`;
- **292 rallies**;
- **3,128 contacts**;
- **292 serves**.

All three videos come from the same dataset. The results are useful for choosing which ideas deserve a larger test. They are not evidence for final thresholds, production precision, or generalisation across broadcast conventions.
