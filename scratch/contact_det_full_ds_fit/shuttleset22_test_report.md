# Final model and ShuttleSet22 test

This report covers a histogram gradient boosting (HGB) contact detector and its test on 47 ShuttleSet22 videos. None of those videos were used to build the detector.

**Set-up:** [Bottom line](#bottom-line) · [Question](#what-the-final-test-was-trying-to-learn) · [Label separation](#how-the-labels-were-kept-out-of-the-predictions) · [Videos](#which-shuttleset22-videos-were-used) · [Model](#the-final-model)

**Results:** [Contact timing](#contact-timing) · [New-video change](#what-changed-on-the-new-videos) · [First contacts](#first-contacts-remain-the-weak-point) · [Player side](#player-side) · [Rally sections](#how-well-the-rally-sections-worked) · [Section edges](#how-much-room-the-sections-left) · [Full outputs](#how-often-the-full-output-was-right) · [Contact scores](#what-a-high-contact-score-tells-us)

**Decision:** [Video types](#did-it-work-on-different-kinds-of-video) · [Near-100% goal](#are-whole-rallies-near-100-right) · [Next tests](#what-remains-worth-testing) · [Checks and records](#checks-and-saved-results)

## Bottom line

| Question | Result |
| --- | --- |
| Did it find single contacts? | **80.62% precision, 84.37% recall and 82.45% F1** within five frames |
| Did it choose the right player? | **92.02%** of matched contacts where both sides had an answer |
| Did it find one clean section for a rally? | **63.16% precision and 73.50% recall** |
| Was the whole output right? | **483 of all 3,982 sections, or 12.13%** |

A clean section holds every labelled contact from one rally and no contact from another rally. The 483 full outputs also got every contact and player side right.

The old report showed 16.60%. That was 493 passes among the 2,969 sections that touched one rally. It left out the 1,013 sections that touched no rally or several rallies. It also let the five-frame contact allowance reach just past a section edge.

The detector still finds useful contact candidates in new videos. It is not ready to annotate whole rallies on its own.

Nothing was tuned after the test labels were opened.

## What the final test was trying to learn

The development work had already chosen the HGB contact detector. The final test asked three questions:

1. Did it still find contacts at about the right time in new videos?
2. Did it still choose the right player?
3. Did those contact results add up to whole rallies that were right from start to finish?

This was the first broad test beyond the 40 ShuttleSet videos used in development.

## How the labels were kept out of the predictions

The development data had already set the model and all event rules.

The prediction program could not access the ShuttleSet22 labels or the code that reads them. It wrote all 47 prediction files before the scoring program opened any labels.

The combined prediction file had a SHA-256 hash before any label was read. The scorer checked that hash and all 47 smaller files first.

Only then did the scorer read the labels. Nothing in the model or its settings changed after that point.

## Which ShuttleSet22 videos were used

The official labels cover 58 matches.

The 58 official matches fell into three groups:

- 47 had downloadable videos that lined up with the official frame numbers and were not used in development. The test used these 47
- eight also appear in the base ShuttleSet development videos, so the test left them out
- three did not have public videos that we could line up with the official frame numbers, so the test left them out

The 47 test videos contain:

- 43,159 source contact rows;
- 38,218 usable contact rows; and
- 3,422 usable labelled rallies.

The model found 39,994 contacts in 3,982 video sections. It could not choose a player for 72 of those contacts.

## The final model

The final model uses the HGB setup chosen during development:

- original per-frame motion values;
- measured values plus flags that say whether the needed data was present;
- 85 input fields;
- balanced class weights;
- up to 24 non-contact rows for each real contact row;
- 31 leaves;
- learning rate 0.06;
- 180 boosting rounds;
- at least 40 training rows per leaf; and
- L2 regularisation of 1.0, which limits how strongly each tree can change the score.

Real contacts are rare in the training rows. Balanced class weights stop the many non-contact rows from dominating training.

It trained on 1,313,803 rows from all 40 development videos. Of those rows, 94,530 marked real contacts.

The model kept contact scores of 0.9 or more. It joined predictions that fell within six frames of each other on a 30 fps clock.

## Contact timing

| Tolerance | Matched contacts | Precision | Recall | F1 | First-contact recall | Later-contact recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 frame | 20,138 | 50.35% | 52.69% | 51.50% | 25.66% | 55.35% |
| 2 frames | 27,713 | 69.29% | 72.51% | 70.87% | 39.22% | 75.79% |
| 5 frames | 32,243 | **80.62%** | **84.37%** | **82.45%** | **53.92%** | **87.36%** |
| 10 frames | 32,603 | 81.52% | 85.31% | 83.37% | 58.07% | 87.99% |

At the five-frame check, predictions were 0.49 frames early on average. Half were no more than one frame away from the label. The median error was zero.

The score rises sharply when the allowed gap grows from one frame to five. The model often finds the right moment, but not the exact labelled frame.

![Timing improves quickly as the allowed frame difference widens from one to five frames.](figures/08_timing_tolerance.png)

## What changed on the new videos

The 40 development videos were split into five groups of eight. Each video was scored by a model trained on the other four groups, or 32 videos.

| Five-frame measure | Out-of-fold development result used to choose the event rules | 47-video ShuttleSet22 test | Change |
| --- | ---: | ---: | ---: |
| Precision | 90.50% | 80.62% | −9.88 points |
| Recall | 86.58% | 84.37% | −2.21 points |
| F1 | 88.49% | 82.45% | −6.04 points |
| First-contact recall | 49.39% | 53.92% | +4.53 points |
| Later-contact recall | 90.75% | 87.36% | −3.39 points |

Precision fell by almost ten points on the new videos. In plain terms, more predicted contacts had no matching label.

Recall fell by only two points. The model found a slightly larger share of first contacts, but it still found far fewer first contacts than later contacts.

![The detector keeps much of its recall on ShuttleSet22 but loses nearly ten precision points.](figures/03_contact_precision_recall_f1.png)

## First contacts remain the weak point

At five frames, the ShuttleSet22 test found about **54% of first contacts** and **87% of later contacts**.

The same gap appears at each step:

| Stage | First-contact recall | Later-contact recall |
| --- | ---: | ---: |
| Chosen eight-video validation run | 41.77% | 88.98% |
| Out-of-fold development result used to choose the event rules | 49.39% | 90.75% |
| 47-video ShuttleSet22 test | 53.92% | 87.36% |

The weak first-contact result is not just a quirk of the first eight videos. It is still there across the larger test.

![First-contact recall stays far below later-contact recall.](figures/04_first_vs_later_recall.png)

## Player side

At five frames, 32,243 predicted contacts matched a label. Almost every matched label included a known Top or Bottom player.

The detector answered 32,188 of those contacts and got 29,620 right. That gives:

- **92.02% player-side accuracy** where both sides were answered; and
- **99.83% answer coverage** among timing matches with a known human side.

At ten frames, player-side accuracy was 91.80%.

The player rule reached 92.02% accuracy on these matched contacts. One wrong player still makes the whole section wrong.

## How well the rally sections worked

A clean rally section contains every labelled contact from one rally. It contains no labelled contact from another rally.

The section finder produced 3,982 sections:

- 2,515 held one complete rally and no part of another rally
- 454 held only part of one rally
- 943 held no labelled contact
- 70 held contacts from several rallies

This gives **63.16% precision**: 2,515 clean matches from 3,982 predicted sections.

The 2,515 clean matches covered 2,515 of the 3,422 usable labelled rallies. This gives **73.50% recall**. Rally-section F1 was **67.94%**.

![All 3,982 predicted sections, split by what they contained.](figures/12_rally_section_outcomes.png)

This is a section-finding score. It does not say whether the contact detector found every contact inside the section.

ShuttleSet22 labels contact frames. It does not mark the exact visual start and end of each rally. A clean match therefore means that all contact labels are inside one section. It does not mean that the clip edges are ideal.

## How much room the sections left

The section finder did not add a fixed buffer before and after a rally.

It used 90 frames of rest, or three seconds at 30 fps, to decide that one active part of the video had ended. This was a rule for finding a break. It was not three seconds of padding at each edge.

Among the 2,515 clean sections:

- the median start was 30 frames, or 1.0 second, before the first labelled contact
- the median end was 88 frames, or 2.9 seconds, after the last labelled contact
- one quarter began within five frames of the first contact
- the middle 80% began between 1 and 273 frames before the first contact
- the middle 80% ended between 46 and 149 frames after the last contact

The end usually had useful room after the last contact. The start was much less steady. A fixed clip buffer would need to be added in a later output step if every saved clip needs the same amount of room.

![The time between each section edge and the nearest labelled contact varied widely.](figures/14_rally_section_context.png)

## How often the full output was right

The old whole-rally score first kept the 2,969 sections that touched exactly one labelled rally. It then put each section into one of these groups:

| Outcome | Sections | Share of one-rally sections |
| --- | ---: | ---: |
| Passed the old contact-and-side check | 493 | 16.60% |
| Missing contacts only | 1,147 | 38.63% |
| Extra contacts only | 243 | 8.18% |
| Missing and extra contacts | 306 | 10.31% |
| Equal contact count but wrong timing | 335 | 11.28% |
| Wrong predicted side | 437 | 14.72% |
| Predicted side unanswered | 8 | 0.27% |

There were also 44 predicted contacts outside every saved detected section.

Missing contacts caused the most failures. Wrong timing and wrong players also caused many failures.

The 16.60% in the table covers the 2,969 one-rally sections. The 943 no-rally sections and 70 multi-rally sections sit outside that calculation.

Counting the old 493 passes against all 3,982 sections gives 12.38%. That still lets the five-frame contact allowance reach past a section edge. Ten sections did this. Requiring one clean section leaves **483 of 3,982, or 12.13%**.

The scores count different things:

| Score | What counted |
| --- | --- |
| Contact timing | All 39,994 predicted contacts and all 38,218 usable labels |
| Player side | The 32,188 timing matches where both sides had an answer |
| Old section check | The 2,969 sections that touched one rally |
| Clean full output | All 3,982 predicted sections |

The 44 contacts outside every section still counted as unmatched contact predictions. They could not enter a section score.

![Missing contacts cause the most ShuttleSet22 failures, but they are not the only problem.](figures/06_external_error_mix.png)

## What a high contact score tells us

The detector already keeps only contacts with scores of 0.9 or more. Setting a lower minimum does not change the output.

At a 0.95 minimum contact score:

- 1,754 sections remained
- 1,344 of those sections matched one labelled rally and had enough human player labels to be scored
- 245 of the 1,344 scored sections passed the old contact-and-side check
- the share that passed rose only to 18.23%

The higher cut-off removes more than half the sections. The share that passed the old check rises by less than two points.

A high contact score says that one predicted contact looks likely. A clean whole section also needs the right boundaries, every real contact, no extra contacts and the right player sides.

## Did it work on different kinds of video?

The model reached 82.45% F1 on 47 new videos, so it did not simply fail outside development.

Precision still fell from 90.50% to 80.62%. The model made more false contact predictions in the new dataset. We cannot assume that it will behave the same way in another group of videos.

The test combines all 47 videos. Performance for each camera layout, set of on-screen graphics, tournament or broadcast style remains unknown.

So the result tells us three things:

- the model can find useful contacts in new videos;
- it makes more false contact predictions there; and
- we still do not know how well it works across different broadcast styles.

## Are whole rallies near 100% right?

No.

The contact detector is useful as one part of a larger system. The whole system is nowhere near 100% precision.

At the main five-frame check:

- **80.62% of predicted contacts** matched a label
- the player choice was right for **92.02%** of timing matches where both sides had an answer
- **16.60% of one-rally sections** passed the old contact-and-side check
- **12.13% of all predicted sections** were clean and fully correct

Each percentage counts a different group of events. They cannot be multiplied into one score. They simply show how accuracy falls as the task grows from one contact to a whole rally.

This work did not train a model to reject doubtful rallies. So we do not know whether the system can keep a small group of nearly perfect rallies.

![The detector finds useful single contacts, but few whole rallies are right.](figures/11_standalone_gap.png)

## What remains worth testing

The random forest (RF) and HGB comparison itself is finished.

The next work should improve:

- sections that start and end at the right time;
- a smaller and safer first-contact source;
- removal of extra contacts after section starts and ends improve;
- a way to say when the player choice is unsure; and
- a model that keeps or rejects a whole rally, tested on videos it did not train on.

That model would need two main numbers: how many rallies it keeps, and how many of those are fully right. Near-perfect accuracy means little if the model keeps almost nothing.

The model could train on predictions from the 40 development videos. Each prediction would come from a contact model that had not trained on that video. The finished model could then run once on the 47 test videos. It could look for bad section edges, a missing first contact, an extra contact or an unsure player choice.

## Checks and saved results

The project scorer and a separate recount agreed on the result:

- the prediction file had the same SHA-256 hash before and after scoring;
- all 47 child outputs were present and unique;
- the official annotation files and cleaned labels had the expected hashes;
- a separate recount did not use the project scorer; and
- that recount found the same timing and player results for the whole test and for every video.

[`records/shuttleset22_test_summary.json`](records/shuttleset22_test_summary.json) holds the short saved result and the new section recount. [`scripts/summarise_shuttleset22_sections.py`](scripts/summarise_shuttleset22_sections.py) rebuilds the recount from the saved predictions, clean labels and full test result. [`records/shuttleset_development_split.json`](records/shuttleset_development_split.json) lists the development videos.

The old plans are unchanged under [`archive/`](archive/). The larger result files stay outside Git in the ignored `raw/` folder. The full raw evidence is also kept in the GitHub release for this branch: [RF/HGB contact study raw evidence](https://github.com/ahalp90/badminton_cv_annotator/releases/tag/contact-det-rf-hgb-series-v1).
