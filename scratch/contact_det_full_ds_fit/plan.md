# Full-data contact experiment plan

## Where the work is now

The scope, video split and safeguards are agreed. The saved 40-video list is checked and committed. The current task is to prepare the same contact features for each listed video.

The old three-video experiment stays unchanged. All new code and results live in this directory.

## Work completed

### Agree what will be tested

The user accepted:

- 32 training videos and eight fixed validation videos
- HGB and RF as the first models
- no more than 12 full model runs in the first comparison
- out-of-fold predictions for any later learned step
- removing extra contacts before trying to add missing contacts, when the errors support that order
- out-of-fold predictions across all 40 videos to choose the final cut-offs
- one final test on non-overlapping ShuttleSet22 videos

### Inspect the three-video code

The inspection found that the feature calculations and rally scoring functions already work one video at a time.

The fixed three-video assumptions are in the code around these functions. That code chooses the video list, loads labels, trains the models, checks result files and applies the Top/Bottom player rule again.

The source details are in `current_system_map.md`.

## Completed change: add the video list and split checks

Add one JSON file with all 40 eligible videos. Each row records the video ID, frame rate, resolution, match details and whether the video is for training or validation.

Add Python code that:

- rejects repeated video IDs or names
- rejects excluded videos in the split
- checks the expected 32/8 counts
- checks every value against the saved ShuttleSet tables
- keeps all machine paths and access details out of the file

Small tests cover these failure cases. The planned commit is:

`Add the full-dataset contact split`

## Current change: prepare features for any listed video

Use the tested pilot feature function once per video from the new list. Do not change its calculations. Save a separate checked file for each video so a stopped run leaves clear progress.

The saved files will record:

- the source commit
- the hash of the video-list file
- frame rate and row intervals for each video
- the size and hash of each input file, without its machine path

Before preparing all 40 videos, run the new code on the three pilot videos. Its feature rows must be exactly equal to the saved pilot rows. Check both the saved file hashes and the rows read back from the files.

Large feature files stay out of Git. The planned commit is:

`Freeze contact features for any video roster`

The approved commit wording uses “freeze”. Here, that means saving the feature rows and enough checks to show which inputs produced them.

## Then: train and compare HGB and RF

Before training begins, write down no more than 12 full model runs. The list may use:

- the original motion values or motion scaled to 30 frames per second
- the existing HGB and RF settings from the pilot
- at most two small, pre-set changes to the HGB settings
- no class weighting or balanced class weighting
- a small fixed list of duplicate-removal distances
- the pilot rule for choosing negative examples or one alternative

The list is not the full combination of every choice. It will not grow after validation results are read without a recorded reason and user approval.

For each full model run:

1. Check all saved feature rows before loading contact labels.
2. Build training examples from the 32 training videos only.
3. Train the model on those 32 videos.
4. Score every candidate frame in the eight validation videos.
5. Choose the score cut-off and the distance used to merge nearby duplicate contact predictions from the validation videos.
6. Save every validation score and its video, interval and frame identity.
7. Report combined and per-video timing results.

A repeated run must produce the same identities, scores and chosen settings. The planned commit is:

`Score fixed contact train and validation splits`

## Then: score complete rallies

After contact frames are fixed, apply the existing Top/Bottom player rule again at those frames. Load the player-side labels only after the predicted sides are fixed.

Reuse the pilot functions that assign events to rally ranges that include the start frame and stop before the end frame. Reuse the function that checks whether a whole rally is correct.

The report will show:

- contact timing
- player-side accuracy
- complete rally accuracy
- how many rallies remain as the required confidence rises
- the main reasons that rallies fail

The new code must reproduce the saved three-video result before using the larger data. The planned commit is:

`Add full-dataset rally scoring`

## Record the first 40-video result

Save the compact result files, plots and plain-language report. Repeat the important totals from the saved per-video predictions to catch reporting mistakes.

Independent reviewers will check the numbers and the proposed next experiment. The planned commit is:

`Record the full-dataset contact baseline`

## Checks before long runs

Before preparing all 40 videos, check:

- the accepted video split
- the absence of machine paths
- the rule that feature preparation does not read contact labels
- exact equality with the pilot feature rows

Before training HGB or RF, check:

- training and validation separation
- the order in which labels load
- negative sampling
- score cut-off and duplicate-removal selection
- random seeds
- links between feature files and result files

After the first result, independently recalculate the main contact and rally totals from the saved predictions.

Before the ShuttleSet22 test, check:

- the fixed model design
- the videos removed because they overlap
- the record of training on all 40 videos
- the rule that ShuttleSet22 labels only score the result

## Possible later work

Later experiments are chosen only after the first complete-rally errors are checked.

If extra contacts remain the main cause of bad rallies, first test a model that removes likely extras. Test a limited way to add missing contacts only when otherwise-good rallies are often one contact short.

Any later trained model that removes contacts, adds contacts or decides whether to keep a rally must use predictions from the first contact model. Each training video must be predicted by a first contact model that did not train on that video.

## Final training and ShuttleSet22 test

Choose the model design with the 32/8 result. Then make out-of-fold predictions across all 40 ShuttleSet videos and use them to choose the final score cut-off and duplicate-removal distance.

Fit the chosen model once on all 40 videos. Test the finished setup once on the non-overlapping ShuttleSet22 videos.

## Review help

- The main agent owns code integration and experiment judgement.
- Luna handles small repeatable checks and reads code without making changes.
- A fresh reviewer checks each code change before its commit.
- DeepSeek V4 Flash and agy Opus review the complete first result before another experiment starts.
