# RF and HGB contact results from the full dataset

This folder records the 40-video development study and the final model's test on 47 ShuttleSet22 videos.

## Table of contents

- [Bottom line](#bottom-line)
- [The five-minute route](#the-five-minute-route)
- [What the work was trying to learn](#what-the-work-was-trying-to-learn)
- [The main results](#the-main-results)
- [What a fully correct section means](#what-a-fully-correct-section-means)
- [Are we closer to an annotator that is almost always right?](#are-we-closer-to-an-annotator-that-is-almost-always-right)
- [What remains useful](#what-remains-useful)
- [Where everything lives](#where-everything-lives)

## Bottom line

The nine-run RF and HGB comparison is complete. More tuning is not planned now.

Histogram gradient boosting, or HGB, was the best of the simple contact models. Across 40 development videos, it reached **90.50% precision, 86.58% recall and 88.49% F1** within five frames. Each video was scored by a model trained on the other 32.

The final model then ran once on 47 ShuttleSet22 videos that were not used in development. It reached **80.62% precision, 84.37% recall and 82.45% F1** within five frames. It chose the right player for **92.02%** of matched contacts where both sides had an answer.

These scores tell us how well the model finds single contacts. Whole rallies are much harder. Only **493 of 2,969 one-rally sections, or 16.60%,** were fully correct from start to finish at five frames.

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
| Fair test across all development videos | 40 | 90.50% | 86.58% | 88.49% | 49.39% | 90.75% |
| ShuttleSet22 test on new videos | 47 | 80.62% | 84.37% | 82.45% | 53.92% | 87.36% |

Every number in this table allows a five-frame timing difference. All frame counts are scaled to 30 fps.

Between the 40-video development result and the test on new videos, precision fell by **9.88 percentage points** and F1 fell by **6.04 points**. Recall fell by **2.21 points**.

![On the new videos, recall changes little while precision falls.](figures/03_contact_precision_recall_f1.png)

## What a fully correct section means

The annotator first proposes a section of video. A section counts as fully correct only when:

- it maps to exactly one labelled rally;
- every contact is present within the stated timing tolerance;
- no extra contact is present; and
- every contact has the correct Top or Bottom player answer.

This test is stricter than contact F1. One missing contact, extra contact or wrong side makes the whole section fail.

On the ShuttleSet22 test, the annotator found 3,982 sections. They fell into three groups:

- 2,969 sections matched one labelled rally
- 943 sections matched no labelled rally
- 70 sections contained several labelled rallies

Among the 2,969 one-rally sections, **493 were fully correct at five frames** and **537 were fully correct at ten frames**.

## Are we closer to an annotator that is almost always right?

The answer is **we know more, but the full annotator is still far from the goal**.

The contact detector is useful. The next step needs to check whether the section starts and ends in the right place. It also needs to look for a missed first contact, extra contacts and a doubtful player choice.

A high contact score does not tell us that the whole section is right. On the ShuttleSet22 test, a cut-off of 0.95 kept 1,754 sections instead of 3,982 at 0.90. Of those sections, 1,344 matched one labelled rally and had enough player labels to be scored. Only **18.23%** were fully right. At the 0.90 cut-off, **16.60% of 2,969** scored sections were fully right.

No test found a group of whole rallies with near-100% precision. We also do not know how the method works across different broadcast styles. The ShuttleSet22 result combines all 47 videos, so it does not show a score for each broadcast style.

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

More RF or HGB tuning is unlikely to make whole rallies reliable. The next useful work is:

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
| Machine-readable results | JSON files in this folder |
| Larger saved files | local `raw/`, to be packaged separately |
| Completed plans and working record | [`archive/`](archive/) |

`HANDOVER.md` is local-only. It is ignored by this folder's `.gitignore` and is not part of the report pack or release.
