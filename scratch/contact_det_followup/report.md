# Contact detector follow-up result

## Bottom line

The whole-rally player-side vote is the one clear low-cost gain. It raises fully correct outputs on the frozen 47-video test from 483 to 901 at ±5 frames. At ±10, it raises the count from 524 to 995. It breaks no previously correct section at either tolerance.

The remaining checks found useful clues but no safe follow-up rule. A cautious first-contact model repaired six sections on the untouched V videos. The cut-off sweep added 13 timing-complete development sections at ±5 and 25 at ±10. The learned delete chooser lost 46 net sections at ±5 and 55 at ±10. The keep-or-review model also missed its precision target by a wide margin.

The current scores still contain missed answers. Labels can find many of them in a small search. The label-free models cannot choose those answers reliably enough. Further work would need better evidence or a new contact model.

## Main results

The tables use base-30 frames. The ±5 result is the primary result. The ±10 column shows how the same fixed choice behaves with a wider timing allowance.

A-D means the 32 videos used to develop the small follow-up models. V means the eight videos held aside for one final first-contact check.

| Check | ±5 frames | ±10 frames | Result |
| --- | ---: | ---: | --- |
| Frozen baseline, fully correct outputs | 483 / 3,982 (12.13%) | 524 / 3,982 (13.16%) | Reference |
| Whole-rally side vote | 901 / 3,982 (22.63%) | 995 / 3,982 (24.99%) | Keep |
| Pooled first-contact choice, A-D | 750 / 2,850 (26.32%) | 850 / 2,850 (29.82%) | Validate once |
| Safe first-contact model, untouched V | 183 / 677 (27.03%) | 192 / 677 (28.36%) | Stop after validation |
| Best learned delete setting, A-D | 680 / 2,850 (23.86%) | 767 / 2,850 (26.91%) | Stop |

The first-contact rows start from 726 A-D sections and 177 V sections at ±5. At ±10, they start from 822 A-D sections and 186 V sections.

## What each check found

### Whole-rally player sides

The side vote chooses the better-supported alternating Top/Bottom pattern for each full contact list. Development labels set one fixed vote-gap rule. The frozen test labels only score that rule.

| Tolerance | Baseline | Revised | Repaired | Broken | Net |
| --- | ---: | ---: | ---: | ---: | ---: |
| ±5 | 483 | 901 | 418 | 0 | +418 |
| ±10 | 524 | 995 | 471 | 0 | +471 |

The rule changes 2,100 of 3,982 test sections. It captures 418 of the 434 side-pattern repairs available in the ±5 label-guided ceiling.

The rule favours a complete alternating rally over individual side answers. At ±5, matched-contact side accuracy falls from 92.02% to 91.13%. Contact-and-side F1 falls from 75.74% to 75.07%. The strict full-rally count still rises by 418.

### Nearby duplicates and global settings

The duplicate audit found no adjacent opposite-side events within two frames. That leaves nothing for the proposed cleanup rule to remove.

The 57-setting development sweep favours a contact cut-off of 0.85 with the existing six-frame merge distance. At ±5, timing-complete sections rise from 940 to 953. The setting repairs 139 sections and breaks 126, for a net gain of 13. This stays below the planned 25-section signal.

At ±10, timing-complete sections rise from 1,045 to 1,070. The setting repairs 167 sections and breaks 142, for a net gain of 25. This meets the working signal exactly. First-contact recall rises from 53.91% to 58.47%, while contact F1 changes from 89.48% to 89.46%.

The saved alternative frames lack player sides, so this sweep measures timing only. The frozen test pack also lacks the raw per-frame test scores needed for a cheap 0.85 rescore. The ±10 result remains a small development lead if the wider tolerance becomes the release measure.

### First contact

The ±5 label-guided timing ceiling raises complete A-D sections from 745 to 1,063. It finds 318 possible repairs. Timing followed by the rally-wide side vote finds 300 strict repairs.

The cautious label-free model uses a small gradient-boosted tree model with a 0.9 choice cut-off. Its held-out A-D predictions make 62 changes. The pooled A-D comparison repairs 24 strict sections at ±5 and 28 at ±10, with no breaks.

The pooled comparison uses all A-D labels to choose the model setting. A stricter nested check chooses the setting without each outer group's labels. That check repairs seven sections at ±5 and eight at ±10, with no breaks.

The fixed model then makes 15 changes on the untouched V group. It repairs six sections at both tolerances and breaks none. Contact-and-side F1 rises from 81.33% to 81.41% at ±5. It rises from 81.87% to 81.96% at ±10.

The safe model captures less than one third of the ±5 ceiling. The first-contact line stops after the V check.

### Whole-rally event choices

The combined ceiling allows one early add or replacement, one deletion, and either alternating player-side pattern. Labels choose the best allowed result for each rally. This is a measure of room to improve.

| Ceiling | ±5 frames | ±10 frames |
| --- | ---: | ---: |
| Baseline A-D sections | 726 | 822 |
| Start edit only | 1,050 | 1,186 |
| Delete only | 831 | 945 |
| Combined | 1,198 | 1,356 |

A deletion is needed for 90 repairs at ±5 and 106 at ±10. Another 15 and 17 repairs come from choosing the other side pattern without editing the event list. The combined search repairs 472 sections at ±5 and 534 at ±10. These gains occur when labels select the action and side pattern.

The learned delete chooser uses only evidence available while the annotator runs. Its best descriptive setting selects 497 deletions.

| Tolerance | Repaired | Broken | Net |
| --- | ---: | ---: | ---: |
| ±5 | 42 | 88 | −46 |
| ±10 | 49 | 104 | −55 |

No setting passes the planned gate of at least 30 net repairs with at most one break per five repairs. The nested held-out check therefore keeps every group unchanged. The whole-rally chooser stops.

### Keep or review

The keep-or-review model estimates whether a whole predicted rally is correct. It uses group-held-out scores from the 32 A-D videos.

The nearest tested threshold above 10% coverage accepts 460 of 2,850 sections, or 16.14%. Precision is 40.87% at ±5 and 45.87% at ±10. The target was 90% precision at 10% coverage.

The next threshold accepts only 5.26% of sections. Its precision reaches 45.33% at ±5 and 50.00% at ±10. This line stops.

## How to read these results

The frozen baseline and side-vote result use 47 test videos. Development choices cannot use those labels. The first-contact validation uses eight V videos after the choice was fixed on A-D.

Label-guided ceilings may use labels to choose an action for each rally. They show how many answers exist in the saved candidates. They do not describe a rule that can run on a new video.

Label-free model results use only saved scores, events, pose, shuttle, and section facts when making a choice. Labels provide training targets on development groups and score held-out predictions.

The section split and extension audit stayed in reserve. The main whole-rally chooser already failed its safety gate. A section pass would answer a separate rally-boundary question and would add a new branch of work.

## Recommendation

Use the whole-rally side vote when a fully correct alternating rally is the product goal. Keep its matched-contact side and F1 trade-off visible in any release note.

Keep the cautious first-contact result as evidence for future model work. Its six-section V gain is real but small.

Stop the remaining post-scoring experiments here. Keep the 0.85 cut-off as a small follow-up lead if ±10 becomes the release measure. Better contact or side evidence is more likely to help than another small chooser over the same saved inputs.
