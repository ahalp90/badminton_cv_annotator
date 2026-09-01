# RF and HGB contact results from the full dataset

This folder records the 40-video development study and the final model's frozen test on 47 ShuttleSet22 videos.

**Contents:** [Bottom line](#bottom-line) · [Quick route](#the-five-minute-route) · [Question](#what-the-work-was-trying-to-learn) · [Results](#the-main-results) · [Rally sections](#how-well-the-rally-sections-worked) · [Full output](#how-often-the-full-output-was-right) · [Decision](#are-we-closer-to-an-annotator-that-is-almost-always-right) · [Next work](#what-remains-useful) · [Files](#where-everything-lives)

## Bottom line

The nine-run RF and HGB comparison is complete. More tuning is not planned now.

HGB was the best of the simple contact models. Across 40 development videos, it reached **90.50% precision, 86.58% recall and 88.49% F1** within five frames. Each video was scored by a model trained on the other 32. The study used this development result to choose the final score cut-off and join distance. It was not an independent test.

The final model then ran once on 47 ShuttleSet22 videos that were not used in development. Every prediction file was saved before the test labels were read. It reached **80.62% precision, 84.37% recall and 82.45% F1** within five frames. It chose the right player for **92.02%** of matched contacts where both sides had an answer.

These scores tell us how well the model finds single contacts. The section finder had **63.16% precision and 73.50% recall**. It found one clean section for 2,515 of the 3,422 usable labelled rallies.

The full output was much less reliable. Only **483 of all 3,982 sections, or 12.13%,** held one complete rally and also got every contact and player side right.

The main problems are clear now:

- The first contact is much harder to find than later contacts
- The section the model finds does not always match one real rally
- Extra contacts and missing contacts still break many sections
- The model can get the timing right but give the wrong player side
- A high contact score does not tell us that the whole section is right

## The five-minute route

[`VISUAL_QUICK_GUIDE.md`](VISUAL_QUICK_GUIDE.md) gives the five-minute version.

The two main written reports are:

- [`baseline_report.md`](baseline_report.md), which explains the development experiments and the rally-start follow-up that did not pass; and
- [`shuttleset22_test_report.md`](shuttleset22_test_report.md), which explains the final model and the test on new videos

## What the work was trying to learn

The pilot showed that a tree model could improve contact timing on three videos. This study asked whether the same result held up with more videos that the model had not trained on.

The study asked six questions:

1. Which of the nine RF or HGB setups worked best on eight videos they had not trained on?
2. Were missing first contacts the main error?
3. Could a small model safely add a missing first contact?
4. Which minimum score and join distance worked across all 40 development videos?
5. Did the final detector still work on a separate ShuttleSet22 test set?
6. Did any result support a near-100%-precision, low-recall automatic annotator?

Each development score came from a model trained without the videos it scored. The ShuttleSet22 labels were not read until the model and all 47 prediction files had been saved.

![The experiment moved from a small pilot to tests on videos kept out of training, then one test on 47 new videos.](figures/01_experiment_route.png)

## The main results

| Stage | Videos | Precision | Recall | F1 | First-contact recall | Later-contact recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chosen eight-video validation run | 8 | 89.24% | 83.44% | 86.25% | 41.77% | 88.98% |
| Out-of-fold development result used to choose the final event rules | 40 | 90.50% | 86.58% | 88.49% | 49.39% | 90.75% |
| ShuttleSet22 test on new videos | 47 | 80.62% | 84.37% | 82.45% | 53.92% | 87.36% |

Every number in this table allows a five-frame timing difference. All frame counts are scaled to 30 fps.

Between the 40-video development result and the test on new videos, precision fell by **9.88 percentage points** and F1 fell by **6.04 points**. Recall fell by **2.21 points**.

![On the new videos, recall changes little while precision falls.](figures/03_contact_precision_recall_f1.png)

## How well the rally sections worked

A section counts as a clean rally match when it contains every labelled contact from one rally and no labelled contact from another rally.

The section finder produced 3,982 sections. Of those, 2,515 were clean matches. This gives:

- **63.16% precision:** 2,515 clean matches from 3,982 predicted sections
- **73.50% recall:** 2,515 clean matches from 3,422 usable labelled rallies
- **67.94% F1:** one score that balances precision and recall

The other 1,467 predicted sections were wrong for this check. Some held only part of a rally. Some held parts of several rallies. Most of the rest held no labelled contact.

![All 3,982 predicted sections, split by what they contained.](figures/12_rally_section_outcomes.png)

ShuttleSet22 labels contact frames. It does not label the exact visual start and end of a rally. The section score can therefore check whether every contact is inside. It cannot tell us whether the clip begins or ends at the ideal moment.

There was no fixed buffer before and after each rally. The section finder used 90 frames of rest, or three seconds at 30 fps, to recognise a break. Those three seconds were a break rule, not added padding.

For the 2,515 clean sections, the median start was 30 frames before the first labelled contact. The median end was 88 frames after the last. That is about 1.0 second before and 2.9 seconds after. The amount varied widely, especially at the start.

![The room before and after labelled contacts varied between sections.](figures/14_rally_section_context.png)

## How often the full output was right

The annotator first proposes a section of video. A section counts as fully correct only when:

- it contains one complete labelled rally and no part of another rally;
- every contact is present within the stated timing tolerance;
- no extra contact is present; and
- the player side is right for every contact.

This test is stricter than contact F1. One missing contact, extra contact or wrong side makes the whole section fail.

The first report counted only the 2,969 sections that touched one labelled rally. **493 passed its contact-and-side check, or 16.60%.** That old check did not require every label to sit inside the section.

That left 943 no-rally sections and 70 multi-rally sections out of the old section denominator. Contact timing used all 39,994 predictions and all 38,218 usable labels. Player-side accuracy used the 32,188 timing matches where both the prediction and label had a side answer.

When all 3,982 predicted sections are counted, the old 493 gives **12.38%**. A stricter recount also required every labelled contact to sit inside the section. Ten of the 493 missed that rule by no more than the five-frame timing allowance. The strict result was therefore **483 of 3,982, or 12.13%**.

## Are we closer to an annotator that is almost always right?

The answer is **we know more, but the full annotator is still far from the goal**.

The contact detector is useful. The next step needs to check whether the section starts and ends in the right place. It also needs to look for a missed first contact, extra contacts and a doubtful player choice.

The whole-section result barely changed when the minimum contact score rose:

| Minimum contact score | Sections kept | One-rally sections scored | Passed the old contact-and-side check |
| ---: | ---: | ---: | ---: |
| 0.90 | 3,982 | 2,969 | 16.60% |
| 0.95 | 1,754 | 1,344 | 18.23% |

The higher cut-off removed more than half the sections and raised the old pass rate by less than two percentage points.

The cleanest tested group remained well below near-100% precision. The ShuttleSet22 result combines all 47 videos, so performance for each broadcast style remains unknown.

![A high score for one contact does not mean that the whole rally is right.](figures/11_standalone_gap.png)

## What remains useful

These parts are worth keeping:

- the motion fields and the fields that say whether tracking data is present
- HGB as the simple starting contact model
- tests where the model had not trained on the video it scored
- the 0.9 score cut-off and the six-frame rule for joining nearby predictions
- the strict check for a whole section
- the finding that first contacts need separate attention
- the test method that saved every prediction before reading the labels

The nine tested RF and HGB setups were close, so more tuning of the same menu is a lower priority than whole-rally work. The next useful work is:

- sections that start and end at the right time
- a safer source for first contacts, or a safer way to choose one
- a clear way to remove extra events
- a way to tell when the player choice is doubtful
- a test at rally level that can reject many doubtful sections

The next test should train a model that keeps or rejects each rally. It could look for bad section edges, a missed first contact, extra contacts and an unsure player choice. Its training data would use predictions from models that had not trained on the video they scored. Once finished, it would run once on the 47 ShuttleSet22 videos. The report would show how many rallies it kept and how many of those were fully right.

## Where everything lives

| Item | Location |
| --- | --- |
| Five-minute picture route | [`VISUAL_QUICK_GUIDE.md`](VISUAL_QUICK_GUIDE.md) |
| Development experiments | [`baseline_report.md`](baseline_report.md) |
| Final model and test on new videos | [`shuttleset22_test_report.md`](shuttleset22_test_report.md) |
| How the detector works and where results live | [`current_system_map.md`](current_system_map.md) |
| Generated figures | [`figures/`](figures/) |
| Figure-building code | [`scripts/plot_report_figures.py`](scripts/plot_report_figures.py) |
| Machine-readable record inventory | [`records/README.md`](records/README.md) |
| Machine-readable records | 16 unchanged JSON files under `records/` |
| Larger saved files | local `raw/` and the GitHub release linked in the test report |
| Completed plans and working record | [`archive/`](archive/) |

`HANDOVER.md` is local-only. It is ignored by this folder's `.gitignore` and is not part of the report pack or release.
