# What the full-data contact experiment will do

## Aim

Choose between two simple tree classifiers for contact detection:

- histogram gradient boosting (HGB)
- random forest (RF)

We will choose the model using 32 training and eight validation videos from ShuttleSet. We will then test the finished setup once on ShuttleSet22 videos that do not overlap with the training data.

## What counts as done

- A saved file lists the eligible videos and the 32/8 training and validation split. It contains no machine-specific paths.
- Every model choice uses the same 32 training videos and eight validation videos.
- The chosen model design is trained again on all 40 eligible ShuttleSet videos only after its settings are fixed.
- ShuttleSet22 is used once as the test set after overlapping videos are removed.
- The report separates contact timing, player side and complete rally correctness.
- A rally is completely correct only when it has every contact, no extra contacts and the correct player side for every contact.
- The main report shows how accuracy changes as the system keeps or rejects more rallies.
- Saved commands, random seeds and compact result files are enough to repeat the work.

## Included

- The 40 completed ShuttleSet video extracts. Videos 9, 10, 12 and 27 stay excluded because their labels or video files are unsuitable.
- A fixed split with 32 training videos and eight validation videos. Whole videos stay together.
- HGB and RF using the same wide set of possible contact frames as the three-video pilot.
- The first comparison may choose among motion scaling, model settings, score cut-offs, distances used to merge nearby duplicate contacts, class weighting and rules for choosing negative examples.
- The existing Top/Bottom player rule and the existing complete-rally score.
- The three manually reviewed videos as checks for scene-cut handling.
- New scripts, tests, video lists, saved settings, compact results and reports under this directory.

## Not included

- `scratch/contact_det/` stays unchanged because it is the three-video pilot record.
- Production code under `src/` stays unchanged. If new experiment code cannot work without a production change, stop and ask first.
- TrackNet, pose, court and annotation outputs are reused. They are not regenerated or retrained.
- The BST-X neural contact detector is separate work.
- A new player-side model waits until the first contact result is understood.
- Pilot trees and results are for comparison only. They are not selected settings for the larger data.
- XGBoost is outside the first baseline work.
- Large feature arrays and source data stay out of Git.
- Machine paths, hostnames, connection scripts, credentials and remote commands stay out of saved project files and commit messages.
- Nothing will be pushed or merged. No commit will be made on `main`.

## Rules that must stay true

- Feature preparation must not read ShuttleSet contact labels. Labels load only after the saved features and their file identities have been checked.
- One video cannot appear in more than one of training, validation and test.
- Every frame from one video stays in the same split.
- Validation labels may choose settings. ShuttleSet22 labels may only score the finished setup.
- Frame numbers and frame rates stay explicit. Each saved rally range includes its start frame and stops before its end frame.
- A failed long run keeps enough information to explain the failure. It must not look like a finished result.
- Any later model that removes extra contacts, adds missing contacts or decides which rallies to keep uses predictions from a first contact model that did not train on the same video. These are called out-of-fold predictions.

## Keep the first comparison small

The exact list of model runs will be written down before training begins. It will contain no more than 12 full model runs.

The list may cover:

- the original motion values and motion scaled to 30 frames per second
- the existing HGB and RF settings from the pilot
- at most two HGB settings with small, pre-set changes
- no class weighting and balanced class weighting
- a small fixed list of distances used to merge nearby duplicate contact predictions
- the pilot rule for choosing negative examples and one alternative

The list will not grow after the validation results are read unless the user approves a recorded reason.

## What may follow the baseline

Later work depends on the complete-rally errors from the 40-video baseline.

If extra contacts remain the main problem, first test a model that removes contacts likely to be extra. Test a limited model that adds likely missing contacts only when otherwise-good rallies are often one contact short.

Any later trained model must use predictions from the first contact model. Each training video must be predicted by a first contact model that did not train on that video.

## Final training and test

The model design is chosen with the 32/8 split. After that choice is fixed, the first model will make out-of-fold predictions across all 40 ShuttleSet videos. Those predictions will set the final score cut-off and duplicate-removal distance.

The chosen model is then trained once on all 40 videos. The full setup is tested once on the non-overlapping ShuttleSet22 videos.

## Known risks

- Some players and tournaments occur in both training and validation. The chosen validation split gives more weight to unseen players. ShuttleSet22 provides the later dataset change.
- Preparing features for 40 full videos is much larger than the pilot. Record progress and file checks for each video.
- The pilot scripts assume exactly three videos. The new tests must catch missing, repeated or misplaced videos before a full run.
- The ShuttleSet22 overlap list and prepared input location are not recorded yet. The final test waits until both are clear.

## Authority

- Local edits and local commits on `contact-det-feasibility` are authorised.
- Commit messages are short, natural notes about the useful change.
- Compute jobs and experiment-only checkout changes are authorised for the supplied data.
- Saved project records use dataset names, video IDs and relative paths only.
- Reviewers receive only the named source files, diffs and non-sensitive results they need.
