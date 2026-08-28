# ShuttleSet22 contact test result

## Bottom line

The fixed detector reached 82.45% contact-timing F1 within five frames on the
47-video ShuttleSet22 test. Player side was correct for 92.02% of timing
matches where both the human label and detector gave an answer.

Whole-rally accuracy was much lower. At five frames, 493 of 2,969 sections
with one mapped rally were fully correct, or 16.60%. At ten frames, 537 were
fully correct, or 18.09%.

This is the final test of the already chosen detector. The result does not
change the model, features, 0.9 score cut-off or six-frame nearby-contact
distance.

## Data used

The official annotation corpus contains 58 matches. The test uses the 47
downloadable, non-overlapping videos. Eight official matches overlap the base
ShuttleSet development data, while three have unresolved frame-aligned public
video sources. `shuttleset22_test_plan.md` records the exact IDs, source
findings and checksum rule.

The scorer authenticated the complete 58-match annotation corpus before it
selected the fixed 47 test matches. The cleaned test labels contain:

- 43,159 source contact rows
- 38,218 usable contact rows
- 3,422 usable rallies

The frozen predictions contain 3,982 detected sections and 39,994 contacts.
Seventy-two predicted contacts have no player-side answer.

## Contact timing

| Tolerance | Matched | Precision | Recall | F1 | First-contact recall | Later-contact recall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 frame | 20,138 | 50.35% | 52.69% | 51.50% | 25.66% | 55.35% |
| 2 frames | 27,713 | 69.29% | 72.51% | 70.87% | 39.22% | 75.79% |
| 5 frames | 32,243 | 80.62% | 84.37% | 82.45% | 53.92% | 87.36% |
| 10 frames | 32,603 | 81.52% | 85.31% | 83.37% | 58.07% | 87.99% |

At five frames, matched predictions are 0.49 frames early on average. Their
median signed error is zero and their median absolute error is one frame.

The five-frame F1 is 6.04 percentage points below the 88.49% held-out
development result. Most of that change comes from precision, which falls
from 90.50% to 80.62%. Recall falls from 86.58% to 84.37%.

## Player side

At five frames, 32,242 of the 32,243 timing-matched labels have a human-side
answer. The detector answers 32,188 of those and gets 29,620 correct. This is
92.02% accuracy with 99.83% prediction coverage.

At ten frames, player-side accuracy is 91.80%. The detector answers 32,547 of
32,602 known human sides.

## Whole rallies and detected sections

The 3,982 detected sections map to labels as follows:

- 2,969 contain exactly one labelled rally
- 943 contain no labelled rally
- 70 contain several labelled rallies

At five frames, the 2,969 one-rally sections have these exclusive outcomes:

| Outcome | Sections |
|---|---:|
| Fully correct | 493 |
| Missing contacts only | 1,147 |
| Extra contacts only | 243 |
| Missing and extra contacts | 306 |
| Equal contact count but wrong timing | 335 |
| Wrong predicted side | 437 |
| Predicted side unanswered | 8 |
| Human side unassessable | 0 |

There are 44 predicted contacts outside every saved detected section.

The confidence curve is descriptive only. Requirements from 0.0 through 0.9
retain the same sections because the fixed detector already keeps only scores
of at least 0.9. Raising the requirement to 0.95 retains 1,754 sections. It
raises fully correct accuracy among assessable retained sections from 16.68%
to 18.23%, but leaves only 245 fully correct sections. This does not justify
changing the fixed cut-off after the test.

## Integrity checks

The combined prediction file was saved before label access and kept its fixed
SHA-256 identity. The source manifest, official annotation corpus and complete
annotation tree also matched their pinned identities. The raw result and
cleaned labels stay outside Git; `shuttleset22_test_summary.json` is the
path-free tracked record.

An independent standard-library recount used only the frozen prediction and
cleaned-label files. It did not import the scorer. The recount reproduced the
saved population, timing and player-side totals at all four tolerances.
