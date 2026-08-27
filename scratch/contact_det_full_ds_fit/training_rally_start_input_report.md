# Rally-start training inputs

## Bottom line

The saved training inputs are complete for all 32 training videos. Every
contact score came from a model that did not train on that video. The files are
ready for a small model that decides whether an earlier rally-start candidate
should be used.

This stage did not read human contact, rally or player-side labels. It did not
train a model or change any contact prediction.

## What was saved

| Group | Videos | Kept contacts | Candidate lists | Candidate entries |
|---|---:|---:|---:|---:|
| A | 8 | 8,470 | 803 | 2,409 |
| B | 8 | 6,430 | 649 | 1,947 |
| C | 8 | 5,710 | 579 | 1,737 |
| D | 8 | 5,849 | 590 | 1,770 |
| Total | 32 | 26,459 | 2,621 | 7,863 |

The existing video pipeline detected 2,850 sections. Of these, 2,621 contain
a kept contact and therefore have a candidate list. The other 229 sections do
not have a kept contact and were counted but left without a list.

Every list contains the first kept contact and two earlier candidates. This is
the same rule that produced the frozen 1,845 validation entries.

## Where the earlier candidates are

The lists contain 5,242 earlier candidates:

- 2,419 are before the detected section starts;
- 2,823 are inside the detected section; and
- 34 are contacts already kept outside the section.

The last group must move the section start if selected. It must not add the
same contact twice.

## Player-side answers limit the next step

The existing player-side rule gave an answer for 2,793 of the 5,242 earlier
candidates. It gave no answer for the other 2,449.

This matters because a correctly timed added contact still cannot make a rally
fully correct when its player side is unknown. The next plan must report this
limit and must not treat timing-only recovery as a complete-rally gain.

The side rule was much more complete for existing contacts. Only 65 of 26,459
kept contacts had no answer. Thirteen of the 2,621 fixed contacts in the short
lists had no answer.

## Checks

The three-video replay passed before the full run. The full run then completed
in fixed A, B, C and D order. A repeat run produced the same combined file byte
for byte.

The saved file checks also confirmed that:

- all 32 videos appear once and use the right held-out group;
- no video occurs in the training list for the model that scored it;
- all 1,193,927 score rows match the raw feature rows and the four group files;
- all 26,459 kept contacts appear once;
- no section has more than three candidates;
- no candidate identity is repeated between sections;
- player-side answers are only `Top`, `Bot` or absent;
- all saved input hashes match; and
- the saved files contain no machine paths or access details.

The combined file has SHA-256
`49236a091efde5ee9fcc6ac52616a716a276c992abe833c46830e30c5ec7e784`.

## Next step

Write and review a separate plan for the candidate selection model before
opening human labels. Keep the model list and settings small. Train only from
these held-out first-model scores, then check the fixed result on the eight
validation videos.

Do not use the failed hand-written pilot rule or a largest-score rule. Do not
open ShuttleSet22 labels.
