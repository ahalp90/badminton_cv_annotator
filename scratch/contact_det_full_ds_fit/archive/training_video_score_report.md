# Training-video contact scores

## Bottom line

The four fixed model runs are complete. Every one of the 32 training videos
now has contact scores from a model that did not train on that video.

These scores are ready for the next rally-start candidate check. They do not
change the chosen first contact model, its score cut-off or its rule for
removing nearby duplicate predictions.

## What ran

Each run trained the chosen histogram gradient boosting (HGB) model on 24
videos and scored the separate group of eight videos. The eight validation
videos were not used to train any of the four models.

The model settings stayed fixed:

- original per-frame motion values
- balanced class weights
- up to 24 negative examples per positive example
- random seed 20260824
- score cut-off 0.9
- six-frame nearby-contact distance at 30 frames per second

| Group | Training rows | Positive training rows | Scored rows | Kept predictions |
|---|---:|---:|---:|---:|
| A | 738,652 | 53,399 | 367,951 | 8,470 |
| B | 800,193 | 60,337 | 294,802 | 6,430 |
| C | 838,570 | 61,773 | 255,293 | 5,710 |
| D | 819,001 | 61,044 | 275,881 | 5,849 |
| Total scored |  |  | 1,193,927 | 26,459 |

The row counts differ because the videos have different lengths and search
areas. They are not accuracy comparisons between groups.

## Checks

Group A ran twice before groups B, C and D. Its score file and result file
matched byte for byte.

After all four runs, the combination was also made twice. Both copies matched
byte for byte. The full checker also confirmed that:

- all four results are complete and use source commit `3b6297ca`;
- each model trained on exactly 24 videos;
- no scored or validation video entered that model's training examples;
- all 32 training videos appear once in the combined scores;
- all 1,193,927 expected score rows appear once and in the fixed group order;
- every score is finite and between zero and one;
- every kept flag can be reproduced from the fixed 0.9 cut-off and six-frame
  nearby-contact distance; and
- all saved file hashes match the result records.

The combined score file has SHA-256
`cfb32a139081105c48fbcc80d8fcd57fabdbe6b3f9a8fc53a47d8e5374147605`.

## What this result does not say

This stage did not use these scores to choose a rally-start candidate. It did
not tune the contact cut-off, change any saved validation prediction, fit on
all 40 videos or open ShuttleSet22 labels.

The next plan must say exactly how a candidate will be chosen and how that
choice will be checked on the eight validation videos. No new model should be
trained until that plan is written and reviewed.
