# Development experiments: the full RF and HGB story

This report covers the work on 40 development videos. It starts with nine RF and HGB runs. It then looks at missed contacts and rally starts. It ends with the rules used by the final model.

RF means random forest. HGB means histogram gradient boosting. Both models use decision trees. HGB worked better here. F1 is one score that combines precision and recall.

**Set-up:** [Bottom line](#bottom-line) · [Question](#what-the-experiments-were-trying-to-learn) · [Held-out design](#how-each-video-was-kept-out-of-its-model) · [Experiment path](#the-experiment-path)

**Experiments:** [Nine runs](#the-nine-rf-and-hgb-runs) · [Model choice](#why-the-chosen-hgb-run-won) · [Full-section check](#what-whole-section-scoring-revealed) · [Errors](#what-was-going-wrong) · [Missed contacts](#the-missed-contact-check) · [Rally-start candidates](#the-rally-start-candidate-list) · [Small model](#the-small-rally-start-model)

**Outcome:** [Final rules](#the-final-rules-across-all-40-videos) · [Meaning](#what-this-means) · [Next work](#useful-work-after-this-series) · [Saved results](#saved-results)

## Bottom line

The chosen development model is HGB. It uses the original per-frame motion values and balanced class weights. It trains on up to 24 non-contact rows for each real contact row.

On the original eight validation videos, it reached:

- **89.24% precision**;
- **83.44% recall**;
- **86.25% F1**; and
- **99 fully correct sections among 609 accepted sections** when the whole section was checked within ten frames.

The model found later contacts well, but often missed the first contact in a rally. At five frames, it found **41.77% of first contacts** and **88.98% of later contacts**.

A later test found useful frames near missed first contacts. A small model then tried to choose which contact to add. At best, it chose the right contact in only **51.7%** of cases. The test required **80%**.

The HGB contact model stayed unchanged. Next, each of the 40 videos was scored by a model trained on the other 32. This test kept contacts scoring at least **0.9**. It also joined predictions that were no more than **six frames** apart on a 30 fps clock.

## What the experiments were trying to learn

The three-video pilot showed that tree models could improve contact timing. The larger work had five jobs:

1. compare nine RF and HGB setups;
2. choose one model without using the final test;
3. find out whether missed contacts or extra contacts caused more sections to fail;
4. test one narrow way to recover missed first contacts; and
5. choose the final event rules from videos that each model had not trained on.

The planned annotator may reject a rally when its contact or section checks show a problem. A small set of correct rallies would still be useful. This means the whole rally matters more than the score of one contact.

## How each video was kept out of its model

The study used 40 ShuttleSet videos. It split them into 32 training videos and eight validation videos.

The eight validation videos were `sset_18`, `sset_22`, `sset_24`, `sset_25`, `sset_30`, `sset_31`, `sset_39` and `sset_40`. The split includes unseen players and a mix of women's and men's matches.

The nine model setups were written down before the test ran. The test also had 19 possible score cut-offs and three possible distances for joining nearby predictions.

Later, five models made predictions for all 40 videos. Each model trained on 32 videos and scored the other eight. No model scored a video it had trained on.

None of these choices used ShuttleSet22 labels.

## The experiment path

![The work moved from nine model runs to the first-contact test and then the final HGB model.](figures/01_experiment_route.png)

| Step | What it asked | Result |
| --- | --- | --- |
| Feature replay | Did the larger feature build reproduce the pilot rows? | Yes. All 130,624 pilot rows matched exactly |
| Nine-run comparison | Which RF/HGB setup worked best? | HGB with original motion and more non-contact examples |
| Error check | Were extra contacts or missed contacts the larger problem? | Missed contacts, especially the first contact |
| Candidate list | Was a useful first contact nearby? | Often, but the list still held too many wrong choices |
| Small addition model | Could it safely add one earlier contact? | No choice was right at least 80% of the time |
| Out-of-fold development result used to choose the event rules | Did the 0.9 cut-off and six-frame rule still work? | Yes |
| Final fit | Could the chosen HGB model train on all 40 videos? | Yes. Reloading the saved model gave the exact same check scores |

## The nine RF and HGB runs

For the contact scores, predictions can be up to five frames from the label after scaling to 30 fps. The last column checks complete sections within ten frames. It shows the number that were fully right, followed by the number kept by the development scorer.

| Model run | Precision | Recall | F1 | Player side | Fully correct / kept by scorer |
| --- | ---: | ---: | ---: | ---: | ---: |
| HGB, original motion, balanced, 12 non-contacts | 88.89% | 83.30% | 86.01% | 90.41% | 96 / 614 |
| HGB, motion adjusted to 30 fps, balanced, 12 non-contacts | 88.91% | 83.30% | 86.01% | 89.95% | 90 / 609 |
| RF, original motion, balanced, 12 non-contacts | 88.44% | 82.32% | 85.27% | **92.03%** | 91 / 606 |
| RF, motion adjusted to 30 fps, balanced, 12 non-contacts | 86.92% | 83.29% | 85.06% | 91.51% | 85 / 612 |
| HGB, original motion, no class weights, 12 non-contacts | 88.66% | **83.74%** | 86.13% | 90.05% | 97 / 606 |
| RF, original motion, no class weights, 12 non-contacts | 88.01% | 82.64% | 85.24% | 90.42% | 68 / 620 |
| HGB, 15 leaves, original motion, balanced, 12 non-contacts | 88.29% | 81.92% | 84.98% | 89.76% | 78 / 616 |
| HGB, slower learning, original motion, balanced, 12 non-contacts | 88.91% | 83.30% | 86.01% | 89.95% | 91 / 615 |
| **HGB, original motion, balanced, 24 non-contacts** | **89.24%** | 83.44% | **86.25%** | 90.28% | **99 / 609** |

![The nine runs are close on contact F1, while the chosen HGB run also has the most fully correct sections.](figures/02_nine_run_model_comparison.png)

All nine runs joined predictions that were no more than six frames apart. The chosen HGB run kept contacts scoring 0.9 or more.

## Why the chosen HGB run won

The chosen run had the highest contact F1. It also had the most fully correct sections among those kept by the development scorer. The plan named these two measures before the test ran.

Other models had single strengths. The RF model with original motion and balanced weights had the best player-side accuracy. The HGB model without class weights had slightly higher recall. Neither gave the same overall result.

The nine F1 scores are within about 1.3 percentage points. The difference between the model families is small. Of these nine setups, the chosen HGB was the best simple starting model.

## What whole-section scoring revealed

The chosen run produced 677 detected video sections. Of these, 564 matched exactly one labelled rally. Six contained contacts from several labelled rallies. 107 did not match a labelled rally.

The old development scorer kept 609 sections and called them “accepted sections”. Of these, 557 matched one labelled rally. It removed 68 sections, including seven that matched one labelled rally. The saved result does not say why it removed each section.

The error mix below uses all 564 detected sections that matched one labelled rally. Only 99 were fully correct. This is why the total is seven higher than the 557 accepted one-rally sections.

A fully correct section has every contact, no extra contacts, and the right player side for every event. A section that joins several real rallies cannot pass this check.

Raising the minimum contact score did little:

| Minimum score | Sections kept | Fully correct | Fully correct among kept |
| ---: | ---: | ---: | ---: |
| 0.90 or lower | 609 | 99 | 16.3% |
| 0.95 | 322 | 55 | 17.1% |

The model already keeps only contacts scoring at least 0.9. Raising that minimum almost halves the output. The share of sections that are fully right rises by less than one percentage point.

## What was going wrong

The 564 detected one-rally sections contained 99 fully correct sections and 465 failures.

| What happened | Sections |
| --- | ---: |
| Fully correct | 99 |
| Missing contacts, with no extra contact | 266 |
| Extra contacts, with no missing contact | 42 |
| Both missing and extra contacts | 65 |
| Contact times correct, but player side wrong | 92 |

![Missing contacts dominate the development failure mix.](figures/05_development_error_mix.png)

There were 94 sections that were exactly one contact short. Every found contact in these sections had the right timing and player side. Of the 94 sections, 81 were missing the first contact.

Only ten sections had exactly one extra contact while everything else was right. This supported testing added first contacts before testing the removal of extra contacts.

## The missed-contact check

At five frames, the chosen model missed 389 first contacts and 554 later contacts.

Here, “first contact” means the first labelled contact in each official rally. A bad predicted start or end does not change which labelled contact counts as first.

The pattern was different:

- **290 of the 389 missed first contacts** had a saved candidate nearby that scored below the 0.9 cut-off;
- **143 of the 554 missed later contacts** had such a candidate; and
- 379 later contacts had no saved candidate nearby.

All 94 otherwise-good one-short sections had a nearby saved candidate at the ten-frame check. For the 81 sections missing the first contact, 39 candidates were available only before the detected section began.

The saved candidates often included the needed frame. They did not show that a model could choose that frame safely.

![First contacts remain much harder than later contacts at every development stage.](figures/04_first_vs_later_recall.png)

## The rally-start candidate list

The candidate list kept the first contact already found in each section. It added no more than two earlier choices.

Across the eight validation videos it contained:

- 615 section lists;
- 1,845 total entries;
- 615 existing first contacts; and
- 1,230 earlier candidates.

At ten frames, the list covered **56 of the 81** target first contacts. Thirty were covered only by candidates before the detected section.

For each target contact it covered, the list added **21.96 earlier choices** on average. This was within the limit set before the result was opened, so the small model test went ahead.

## The small rally-start model

The next test used contact scores from the 32 training videos. Each score came from a model that had not trained on that video. It compared logistic regression and shallow HGB at three score cut-offs each.

There were 5,242 earlier training candidates and 271 sections where a correct addition was possible.

| Choice | Contacts added | Right additions | Share right | New fully correct sections | Fully correct sections lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression, 0.5 | 1,134 | 206 | 18.2% | 74 | 118 |
| Logistic regression, 0.7 | 647 | 186 | 28.7% | 66 | 40 |
| Logistic regression, 0.9 | 195 | 83 | 42.6% | 35 | 4 |
| Shallow HGB, 0.5 | 748 | 199 | 26.6% | 68 | 44 |
| Shallow HGB, 0.7 | 484 | 177 | 36.6% | 59 | 17 |
| Shallow HGB, 0.9 | 147 | 76 | **51.7%** | 30 | **0** |

The test required at least 80% of the added contacts to be right. Every choice fell short.

The shallow HGB choice at 0.9 kept every already-correct section right. It still added the wrong contact almost half the time. That is not suitable for an annotator aiming for very high precision.

The model stopped at this point. It did not score the validation candidates or read their labels.

![The missed first contact was often nearby, but none of the six model choices was right often enough to continue.](figures/07_rally_start_followup.png)

## The final rules across all 40 videos

After the rally-start test stopped, the unchanged HGB contact model made predictions for all 40 development videos. Each prediction came from a model that had not trained on that video.

Each of five models trained on 32 videos and scored the other eight. The combined file contained 1,477,290 candidate scores.

The all-40 test tried the same 57 pairs of minimum scores and join distances. The final rules stayed at:

- contact score cut-off: **0.9**;
- nearby-contact distance: **six frames at 30 fps**.

At five frames, the result across all 40 videos was:

| Measure | Result |
| --- | ---: |
| Labelled contacts | 33,267 |
| Predicted contacts | 31,824 |
| Matched contacts | 28,801 |
| Precision | **90.50%** |
| Recall | **86.58%** |
| F1 | **88.49%** |
| First-contact recall | **49.39%** |
| Later-contact recall | **90.75%** |

The final model trained on 1,313,803 rows from all 40 videos. It used 85 input fields. Loading the saved model again gave the exact same 80 check scores.

Higher cut-offs raise precision, lower recall and produce fewer contacts. The chosen 0.9 cut-off is already near the high-precision end of the tested range.

![Contact precision, recall and number of predictions across the tested cut-offs.](figures/10_contact_cutoff_tradeoff.png)

## What this means

The HGB contact detector is useful and gives the same result when rerun.

The full annotator still has important problems. Contact F1 alone hides them:

- it often misses first contacts;
- some detected sections do not describe one rally;
- some sections miss contacts or add extras; and
- player-side errors spoil timing that would otherwise be useful.

The rally-start test also tells us what did not work. A nearby frame by itself is not enough. A future model needs to check the section edges, missing and extra contacts, and the player choice. It may also need to act in far fewer cases.

## Useful work after this series

More tuning of the same RF and HGB menu is unlikely to be the best use of time.

The next useful questions are:

- can sections start and end at the right time before contacts are cleaned up?
- can a new first-contact source give a much smaller and cleaner candidate list?
- can extra contacts be removed once the section starts and ends are better?
- can the system tell when its player choice is doubtful? and
- can a model keep only a small group of whole rallies that are almost always right on new videos?

A model that keeps or rejects each whole rally is the clearest first test. It could check the section edges, missing and extra contacts, and the player choice. Its report would show how many rallies it keeps and how many of those are right. Both numbers matter because a model can look very accurate by keeping almost nothing.

A later model must not train on scores from a contact model that saw the same video. The final report can show separate results for short and long rallies. The model cannot use the true rally length as an input because the annotator will not know it.

## Saved results

| Record | What it contains |
| --- | --- |
| [`records/baseline_runs.json`](records/baseline_runs.json) | The nine model setups and the tested event rules |
| [`records/baseline_summary.json`](records/baseline_summary.json) | The short eight-video result and error counts |
| [`records/missed_contact_summary.json`](records/missed_contact_summary.json) | Missed first/later contact diagnosis |
| [`records/rally_start_candidate_summary.json`](records/rally_start_candidate_summary.json) | Candidate-list coverage and size |
| [`records/rally_start_model_summary.json`](records/rally_start_model_summary.json) | All six small-model results and why the test stopped |
| [`records/final_contact_score_inputs.json`](records/final_contact_score_inputs.json) | The all-40 input files and their hashes |
| [`records/final_video_score_groups.json`](records/final_video_score_groups.json) | The five groups used to score all 40 videos without training on them |
| [`archive/`](archive/) | The original plans, stage reports and full worklog |

The larger score arrays, detailed results and final model live under the ignored `raw/` folder.
