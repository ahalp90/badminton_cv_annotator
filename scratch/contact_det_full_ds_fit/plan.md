# Full-data contact experiment plan

## Where the work is now

The scope, video split, feature calculations and nine model runs are fixed. The
timing, player-side and complete-rally results are saved and checked. The
reference raw-motion HGB run with more negative examples is the first baseline.
The next task is a small follow-up aimed at missed rally starts.

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

## Completed change: prepare features for any listed video

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

## Completed change: train and compare HGB and RF

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

## Completed change: score complete rallies

After contact frames are fixed, apply the existing Top/Bottom player rule again at those frames. Load the player-side labels only after the predicted sides are fixed.

Reuse the pilot functions that assign events to rally ranges that include the start frame and stop before the end frame. Reuse the function that checks whether a whole rally is correct.

The report will show:

- contact timing
- player-side accuracy
- complete rally accuracy
- how many rallies remain as the required confidence rises
- the main reasons that rallies fail

The new code reproduced the saved three-video result before it was used on the larger data. The commit is:

`Score whole rallies after fixing the predictions`

## Completed change: record the first 40-video result

Save the compact result and plain-language report. Repeat the important totals
from the saved per-video predictions to catch reporting mistakes.

A fresh read-only reviewer and a separate local recount checked the numbers and
the proposed direction. The commit is:

`Record the full-dataset contact baseline`

The local recount supports the saved totals. The leading run reaches 0.8625
contact F1 at five frames and 99 fully correct sections out of 609 accepted at
ten frames. Missing contacts are much more common than extra contacts. Ninety-four
single-rally sections are exactly one contact short with every predicted time
and player side otherwise correct.

## Completed change: check where contacts are missed

Before another model run, check whether the chosen HGB model has saved
candidate frames near its missed contacts. Separate first contacts from later
contacts. Repeat the check for the 94 otherwise-good sections that are exactly
one contact short.

Keep this read-only. Do not train a model, tune a cut-off or change the saved
events. The fixed counts and stop rules are in `missed_contact_check_plan.md`.

The planned implementation commit is:

`Check where the baseline misses contacts`

The result confirms that missed rally starts usually have candidate frames to
work with. At ten frames, 284 of 364 missed first contacts have a saved
candidate nearby. All 94 otherwise-correct one-short sections have a nearby
candidate, and 81 are missing the first contact.

## Completed change: test a small rally-start candidate list

Build a short list without reading contact labels. Include candidates just
before each detected section as well as candidates inside its start. Fix the
maximum list size and required contact coverage before scoring it.

This is a candidate-list check, not a trained second model. Stop if it adds too
many frames for the first contacts it makes available. Do not repeat the broad
pilot list or its failed hand-written chooser.

No code or scoring starts until a separate reviewed plan gives exact values for
the maximum candidates per section, maximum total candidates, minimum
first-contact coverage and maximum added candidates per newly covered contact.

Those values are now fixed in `rally_start_candidate_plan.md`. Each section
may have its first kept contact and at most two earlier score rows. Across the
validation result, the list must contain no more than 1,845 entries, cover at
least 50 of the 81 target first contacts, and add no more than 25 entries per
covered contact.

The next implementation only builds and measures this list. It does not choose
or add a contact.

The implementation now checks every saved input, builds the list twice and
saves identical candidate bytes before it opens the missed-contact detail. It
also assigns the wider missed-first-contact count through the detected section
for that labelled rally, so one section cannot borrow a nearby candidate from
another.

The planned implementation commit is:

`Build the rally-start candidate list`

The fixed list passes all four limits. It covers 56 of the 81 target first
contacts at ten frames and adds 21.96 earlier entries per covered contact.
Thirty contacts are covered only because the list includes frames before the
detected section starts.

The list is kept for the next stage. Its 1,230 earlier entries have not been
added to the baseline.

## Current change: choose one earlier contact

Do not reuse the pilot's failed hand-written rule. Write a separate plan for a
small trained choice method.

Before that method can train, make first-model scores for its training videos
with models that did not train on those videos. The fixed groups and run rules
are in `training_video_score_groups.json` and
`training_video_score_plan.md`.

Use four groups of eight current training videos. For each group, train the
chosen HGB on the other 24 training videos and score the eight held-out videos.
The existing eight validation videos train none of these models and remain the
later check for the candidate-choice method.

This next stage only makes the held-out first-model scores. It does not train
the candidate-choice method or change the rally-start candidate list.

The scorer and its tests are now committed in `3b6297ca`. It checks the fixed
input hashes and all feature files before reading contact labels. Each fit then
receives labels from only its 24 training videos. The combined result must
match every expected candidate row from all 32 videos.

All four fits are complete. Group A and the final combination were each run
twice and produced identical files. The combined file has 1,193,927 score rows
for all 32 training videos. An independent check confirmed the fixed groups,
training separation, input hashes and kept-contact decisions.

The next stage saves the detected sections, kept contacts, short rally-start
candidate lists and predicted player sides needed for training. It does not
open human label rows or train the candidate choice method. The exact inputs
and checks are in `training_rally_start_input_plan.md`.

That stage is complete. The saved result has 2,621 section lists and 5,242
earlier candidates across all 32 training videos. It reproduces the frozen
validation rule, passed a three-video replay and produced the same combined
bytes on a full repeat.

The candidate selection plan is now fixed in
`rally_start_selection_plan.md`. It uses two small models and three fixed
selection cut-offs. It learns from held-out candidate scores on the 32
training videos and uses the eight validation videos once after the choice is
fixed.

The existing player-side rule has no answer for 2,449 earlier candidates. A
useful training answer must match both the first-contact time and its player
side. The plan keeps timing-only matches in the report but does not count them
as complete-rally gains.

The model may add one earlier contact or add nothing. It cannot remove or
replace an existing contact. An independent read found five gaps in the first
draft. The final plan separates training and validation label loads, names the
exact player-side join, prevents a labelled rally from being counted twice,
avoids the old replacement helper and gives exact denominators for its
percentage checks.

The frozen validation list does not save player-side answers for unkept
candidates. Add one small label-free preparation step before model fitting. It
will reproduce the frozen list, replay the existing Top/Bot rule for every
candidate and save the same fields already available for the 32 training
videos.

That preparation is complete. The two full saves are equal byte for byte. The
result keeps all 615 fixed lists and 1,845 entries. The player-side rule has an
answer for 629 of the 1,230 earlier candidates and no answer for 601.

The fixed model comparison is also complete. None of its six choices passed
the training rules. Shallow HGB at 0.9 was the only choice that gained fully
correct sections without losing one, but only 76 of its 147 actions were
correct. Its 51.7% correct-action rate is below the fixed 80% minimum.

Stop this addition follow-up without opening validation labels. Keep the
original first contact model and continue to held-out first-model predictions
across all 40 development videos for the final settings.

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

The exact work is fixed in `final_contact_fit_plan.md`. Use A, B, C, D and the
former validation videos as group V. Rerun all five groups with the other 32
videos used for training. Save raw scores before choosing from the original 57
cut-off and nearby-contact pairs.

Fit the chosen model once on all 40 videos. Test the finished setup once on the non-overlapping ShuttleSet22 videos.

## Review help

- The main agent owns code integration and experiment judgement.
- Luna handles small repeatable checks and reads code without making changes.
- A fresh reviewer checks each code change before its commit.
- DeepSeek V4 Flash and agy Opus review the complete first result before another experiment starts.
