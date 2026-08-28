# Plan for the final contact model

## Aim

Choose the final contact score cut-off and nearby-contact distance from all 40
development videos, then fit the chosen HGB model once on all 40. Each video
must be scored by a model that did not train on that video while the two final
settings are chosen.

This is the final fit of the original contact model. The failed rally-start
addition is not included.

## Fixed model and menu

Keep `hgb_reference_raw_more_negatives` unchanged. It uses the original motion
values, balanced class weights, the reference HGB settings and up to 24
negative examples per positive example.

Choose only from the 19 score cut-offs from 0.05 to 0.95 and the nearby-contact
distances of four, five and six frames at 30 frames per second in
`baseline_runs.json`. Use its existing tie order:

1. higher contact timing F1 at five frames;
2. higher recall;
3. higher precision;
4. larger nearby-contact distance; and
5. higher score cut-off.

Do not add a model, feature, cut-off, distance or tie rule after the results are
read.

## Five video groups

Use the fixed groups in `final_video_score_groups.json`. Groups A, B, C and D
are unchanged. Group V contains the eight former validation videos.

For each group, train on the other 32 videos and score its eight videos. Keep
the 32 training videos in the development split's numeric order. This makes
the group V fit exactly repeat the original chosen model fit and gives a
direct check against its saved validation scores.

The old A–D score files are not inputs to this stage. Those models trained on
24 videos. Rerun A–D with group V included in training.

## Input and label checks

Before reading a label row:

- check the fixed group, split, model, feature, baseline and label-file hashes;
- load and check the raw feature files for all 40 videos;
- reproduce the fixed 1,477,290 candidate rows and their video, interval,
  frame and frame-rate identities; and
- write the group result as `running`.

Read labels for only the 32 training video IDs used by a fit. Check the label
file hash again after the read. The model fitting function must receive no
label or training row from the eight videos it scores.

For A, B, C and D, those 32 training videos include group V. Their label
loaders may read the group V rows needed for training. The separate all-40
label read used to choose the final settings starts only after all five raw
score files are complete.

All 40 development labels may be used after the five raw score files are fixed
to choose the final cut-off and nearby-contact distance. This is part of model
fitting. It is not an independent test result.

## Save five raw score files

For each group:

- reset the fixed random seed;
- choose training examples with the unchanged positive, ignored, nearby and
  sampled-negative rules;
- fit the unchanged HGB model on exactly 32 videos;
- save one raw contact score for every candidate row in the eight held-out
  videos; and
- record the exact training videos, scored videos, model settings, training
  counts, input hashes, score count and score hash.

Do not write kept-contact flags using the old 0.9 cut-off and six-frame
distance. The raw score schema contains video, interval, frame, frame rate,
probability and source group only. It has no kept-contact field.

Write each group in its listed video order. Within one video, keep the
candidate order from the checked raw feature file. Record the implementation
source commit in every group result and require the same commit when results
are combined.

The group V video, interval, frame, frame-rate and probability values must
match the saved validation scores from the chosen baseline run exactly. Its
old kept-contact flags are checked against the old settings but are not copied
into the new raw score file. Stop if any identity or probability differs.

## Combine and choose the final settings

Combine the groups in A, B, C, D and V order. Keep each group's saved row order
unchanged. Require every expected candidate identity exactly once and require
every row's source-group field to match its video. Record and check the same
implementation source commit in the combined result.

Save the combined raw scores before choosing settings. Then read the 40-video
timing labels, check their file hash and score every one of the 57 fixed
cut-off and distance pairs. Save every pair's matched contacts, prediction
count, precision, recall and F1.

Choose the final pair with the fixed tie order. Apply it to a copy of the raw
scores and save the kept-contact flags separately. Report results at one, two,
five and ten frames after adjustment to 30 frames per second. Also report
first-contact and later-contact recall and per-video results at five frames.

Compare the selected pair with the old 0.9 cut-off and six-frame distance on
the same held-out all-40 scores. Do not use whole-rally results or the failed
rally-start model to change the final pair.

## Fit and save the all-40 model

After the pair is fixed, choose training examples from all 40 videos with the
same unchanged rules and fit the HGB model once. Save the model outside Git.
Record its file hash, model settings, training counts, library versions and
the selected cut-off and distance without machine paths. Record the
implementation source commit and bind it to the five group results.

Before saving, predict the first and last candidate row from each video in the
development split's numeric order. Save those 80 video, interval, frame and
frame-rate identities, their probabilities, and the raw feature record and
per-video feature hashes. Load the saved model again and require identical
probabilities for those exact rows. The final cut-off and distance remain
fixed even if the score distribution moves after training on all 40 videos.

Implement this stage with a separate five-group runner and final-fit functions,
or with clearly parameterised new functions. Do not weaken the old scorer's
four-group and 24-video checks. Add direct tests for group V, 32-video training,
the raw-score schema and order, source-group checks, the menu tie order and the
80-row model reload check.

Do not inspect or score ShuttleSet22 while making these choices.

## Repeat and review checks

- Repeat one A–D group and require equal raw score bytes.
- Require group V to match the earlier saved validation identities and scores.
- Build the combined raw file and final kept-contact file twice and require
  equal bytes.
- Recalculate the chosen pair and headline metrics from the saved raw scores.
- Have a fresh read-only reviewer check the code before each implementation
  commit and the result before its report commit.
- Run focused tests, Ruff, the pinned type check and the whole test suite.

The planned commits are:

- `Plan the final contact fit`
- `Score every development video`
- `Fit the final contact model`
- `Record the final contact fit`

## Outside this stage

- another contact model or model setting
- another score cut-off or nearby-contact distance
- the failed rally-start candidate model
- player-side or whole-rally tuning
- ShuttleSet22 labels or results
- production-code changes
