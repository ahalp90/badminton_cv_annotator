# Visual quick guide

These pictures ask whether a contact model that worked on three videos still works on new videos, and whether it brings us closer to a near-perfect automatic annotator. The full set takes about five minutes to read.

**Main story:** [Route](#1-the-study-went-from-3-videos-to-47-new-videos) · [Model choice](#2-hgb-came-first-in-the-nine-model-runs) · [New-video result](#3-precision-fell-on-the-new-videos) · [First contacts](#4-first-contacts-stayed-much-harder) · [Section outcomes](#5-most-predicted-sections-held-one-whole-rally) · [Section scores](#6-rally-section-precision-was-632) · [Section edges](#7-the-end-usually-had-more-room-than-the-start) · [Full outputs](#8-single-contacts-are-useful-but-few-full-outputs-are-right)

**More:** [Extra plots](#extra-plots) · [Conclusion](#the-result-in-one-paragraph)

## The eight-picture story

These eight pictures show the main result.

## 1. The study went from 3 videos to 47 new videos

![The experiment path from the three-video pilot to 40 development videos and 47 new test videos.](figures/01_experiment_route.png)

The development data chose the model and all its settings. The ShuttleSet22 labels were not read until all 47 prediction files had been saved.

## 2. HGB came first in the nine model runs

![The nine-run RF and HGB comparison.](figures/02_nine_run_model_comparison.png)

The nine runs had similar contact F1 scores. The chosen HGB run also had the most fully correct sections among those kept by the development scorer. It came first on both checks.

The winning setup used the original motion values. It also used balanced class weights and up to 24 non-contact training rows for each real contact row.

## 3. Precision fell on the new videos

![Contact precision, recall and F1 across the main development tests and the test on new videos.](figures/03_contact_precision_recall_f1.png)

F1 was **88.49%** across the 40 development videos. Each video was scored by a model trained on the other 32. F1 was **82.45%** on ShuttleSet22.

Precision fell by **9.88 percentage points**. Recall fell by only **2.21 points**. The model still finds many real contacts in the new videos. It also predicts more contacts that have no matching label.

## 4. First contacts stayed much harder

![First-contact and later-contact recall across the three main tests.](figures/04_first_vs_later_recall.png)

First-contact recall rose from 41.77% on the original validation set to 53.92% on ShuttleSet22. Later-contact recall stayed near 90%.

First-contact recall stayed well below later-contact recall in all three tests. The detector still needs a separate way to handle rally starts.

## 5. Most predicted sections held one whole rally

![What all 3,982 predicted sections contained.](figures/12_rally_section_outcomes.png)

The detector made 3,982 sections. Of those, 2,515 held every labelled contact from one rally and no contact from another rally.

The other sections held part of one rally, parts of several rallies, or no labelled rally.

## 6. Rally-section precision was 63.2%

![Rally-section precision, recall and F1.](figures/13_rally_section_precision_recall.png)

The 2,515 clean matches give **63.16% precision** from all 3,982 predicted sections. They give **73.50% recall** from all 3,422 usable labelled rallies.

This check only asks whether the section holds one whole rally. It does not ask whether every predicted contact is right.

## 7. The end usually had more room than the start

![The time between each section edge and the nearest labelled contact.](figures/14_rally_section_context.png)

There was no fixed buffer before and after each rally. The three-second rule found a long rest between rallies. It did not add three seconds to both ends.

The median clean section began 1.0 second before the first labelled contact. It ended 2.9 seconds after the last. Starts varied much more than ends.

## 8. Single contacts are useful, but few full outputs are right

![What happened in the 2,969 ShuttleSet22 sections that match one labelled rally.](figures/06_external_error_mix.png)

Only **493 of 2,969 one-rally sections** passed the old contact-and-side check at five frames. That check did not require every label to sit inside the section.

Missing contacts caused the most failures. Extra contacts, wrong timing and wrong player side also broke many sections.

![The gap between contact timing, player-side answers and fully correct sections.](figures/11_standalone_gap.png)

These percentages use different groups of results:

- **80.62% contact precision** across all ShuttleSet22 predictions
- **92.02% player-side accuracy** after a timing match and two answered sides
- **16.60% passed the old check** among sections that touched one labelled rally
- **12.13% clean and fully correct** among all predicted sections

The last figure is the fairest full-output result. It counts every predicted section. It also checks that the section holds one whole rally and no part of another.

The 16.60% figure left out 943 sections with no labelled rally and 70 sections with parts of several rallies. This did not change the contact precision or recall. Those scores used all contacts.

The tests did not find any group of whole rallies with near-100% precision.

## Extra plots

### Development error mix

![What happened in the 564 development sections that match one labelled rally.](figures/05_development_error_mix.png)

The development test also failed most often because contacts were missing. Wrong player choice was the next largest group.

### Rally-start follow-up

![How often the earlier-frame list found the missed first contact, and how often the small model chose the right one.](figures/07_rally_start_followup.png)

The list contained the missed first contact for 56 of the 81 target rallies. The best small model was right only 51.7% of the times it added a contact. The test required at least 80%. The test stopped before the validation labels were read.

### Timing tolerance

![ShuttleSet22 precision, recall and F1 at one, two, five and ten frames.](figures/08_timing_tolerance.png)

F1 on ShuttleSet22 rises from 51.50% at one frame to 82.45% at five frames. The detector often finds the right moment without landing on the exact labelled frame.

### Do high contact scores find clean rallies?

![A stricter minimum contact score keeps fewer sections without making them much cleaner.](figures/09_confidence_vs_yield.png)

The score ranks single contacts, while the section result also depends on its boundaries, missing contacts, extra contacts and player sides. On development data, raising the minimum score from 0.9 to 0.95 changed the fully correct share from **16.3% to 17.1%** at a ten-frame tolerance. On the new videos, it changed the old one-rally pass rate from **16.6% to 18.23%** at a five-frame tolerance. The tolerances differ, so the cut-offs should be compared within each dataset only.

### What changed when the minimum score changed

![Development contact precision, recall and number of predictions across the tested score cut-offs.](figures/10_contact_cutoff_tradeoff.png)

Among the settings tested, 0.9 gave almost the highest precision. A higher minimum adds little precision. It loses more recall and produces fewer contacts.

## The result in one paragraph

HGB is a useful simple contact detector. It still finds contacts on ShuttleSet22, but it makes more false predictions there. The section finder gets about two-thirds of its outputs right and finds about three-quarters of the labelled rallies. Only 12.13% of all outputs have a clean section plus every contact and player side right. The next work needs safer rally starts, fewer extra contacts and a way to reject doubtful full outputs.
