# Plan for scores on held-out training videos

## Aim

Make HGB scores for each of the 32 training videos using a model that did not
train on that video.

These scores are needed before a trained method can choose a rally-start
candidate. This stage only makes and checks the first-model scores. It does not
train the candidate-choice method.

## Fixed video groups

Split the 32 current training videos into four groups of eight. Each group has
four 25 fps videos and four 30 fps videos. The seven women's matches are spread
as evenly as possible: two in groups A, B and C, and one in group D.

The exact list is in `training_video_score_groups.json`.

| Group | 25 fps videos | 30 fps videos |
|---|---|---|
| A | 01, 05, 11, 19 | 21, 29, 32, 41 |
| B | 02, 06, 13, 15 | 26, 33, 34, 38 |
| C | 03, 07, 16, 17 | 28, 35, 36, 42 |
| D | 04, 08, 14, 20 | 23, 37, 43, 44 |

For each group, train on the other 24 current training videos and score only
the group's eight videos. The existing eight validation videos must not train
any of these four models.

For each fit, the training label rows must belong to exactly its 24 training
videos. Rows from the scored group and the fixed validation videos may not
enter the training labels or training examples. The shared label file may be
opened to extract the allowed 24 videos, but all other rows must be excluded
before the training selection is made.

Players appear in more than one group. The agreed rule separates whole videos,
not players. A player-separated split is not practical because the 25 fps
matches are connected through several players who appear repeatedly. Describe
the result as held out by video, not held out by player.

## Fixed model and contact selection

Use the chosen baseline design without changes:

- reference HGB with 31 leaves, learning rate 0.06, 180 iterations,
  minimum 40 samples per leaf and L2 value 1.0;
- original per-frame motion values;
- balanced class weights;
- positive, ignored and nearby-negative distances of 1, 4 and 15 frames at
  30 frames per second;
- up to 24 negative examples per positive example;
- random seed 20260824;
- score cut-off 0.9; and
- six-frame duplicate-removal distance at 30 frames per second.

The cut-off and duplicate distance are copied from the baseline. Do not tune
them on these training videos.

## Four fixed fits

Run exactly four HGB fits:

1. train on B, C and D; score A;
2. train on A, C and D; score B;
3. train on A, B and D; score C; and
4. train on A, B and C; score D.

Do not add another HGB setting, score cut-off, duplicate distance, sampling
rule or group after the outputs are read.

## Saved output

For each fit, save:

- the training and scored video lists;
- hashes of the group file, development split, raw feature record, model menu,
  baseline summary, chosen baseline result and contact-label file;
- the fixed model settings and training-example counts for each video;
- every held-out score with its video, search interval, frame and frame rate;
- the kept-contact flag made with cut-off 0.9 and distance six; and
- a complete or failed status that is written before a result can be reused.

After all four fits finish, combine them in group order. Check that each of the
seven existing `region_*` search flags is an unsigned byte with value zero or
one. Score a row when at least one flag equals one. These are the same candidate
rows used by the baseline scorer. Each row is identified by video, search
interval, frame and frame rate.

The raw feature record calls the selected count for each video
`seeded_frame_count`. The 32 expected per-video counts are repeated in
`training_video_score_inputs.json`. They sum to exactly 1,193,927 candidate
score rows. The raw feature files have 1,209,642 rows for those videos; 15,715
rows have all seven flags equal to zero and are not scored.

The combined file must contain one copy of every selected row. Mark the source
group for every scored row. Keep group files and the combined file outside Git,
with compact tracked summaries added only after the result is checked.

## Checks before launch

Before any full fit:

- load `training_video_score_inputs.json` and check every listed filename and
  SHA-256 hash against its file bytes;
- check that the group file matches the fixed development split;
- check that the four groups are disjoint and cover exactly the 32 training
  videos;
- check that every group has four videos at each frame rate;
- check that no excluded or validation video appears in a group;
- check that each fit's 24 training videos are exactly the other three groups;
- check that the fixed HGB and negative-example settings match the chosen
  baseline result;
- check every one of the 32 feature files against the filename and hash in the
  already checked raw feature record before opening the contact labels;
- check that the label rows passed to each fit contain exactly its 24 training
  video IDs and none from the scored or fixed validation videos;
- check that score construction cannot read held-out results or tune settings;
- run focused tests and a small smoke test; and
- run group A twice before launching B, C and D. Its saved scores and result
  must be byte-for-byte equal. All four fits use the same function and checks,
  so this repeated first group checks the new run path before the remaining
  three fits.

Run long work in `tmux`. Check it after meaningful milestones rather than at a
fixed short interval.

## Checks after all four fits

- every group result is complete and linked to the committed source and input
  hashes;
- each training list has 24 videos and excludes its scored group;
- the existing validation videos appear in no training list;
- score identities are unique within each group and disjoint between groups;
- scores are finite and between zero and one;
- all 32 training videos appear once in the combined scores;
- the combined score identities match the raw feature candidate rows exactly;
- the kept flags can be reproduced from cut-off 0.9 and distance six; and
- a repeated combination produces identical bytes.

Stop on any failed check. Do not use contact accuracy to revise the groups or
model settings in this stage.

## Outside this stage

- no candidate-choice training
- no change to the fixed rally-start candidate list
- no complete-rally comparison
- no final cut-off selection across all 40 videos
- no fit on all 40 videos
- no ShuttleSet22 labels

The planned group-and-plan commit is:

`Set the held-out training score groups`
