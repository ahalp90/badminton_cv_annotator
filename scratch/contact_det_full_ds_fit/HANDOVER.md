# Contact detection experiment handover

## Start here

The final contact model and the ShuttleSet22 shuttle inputs are complete and
checked. The remaining work is the independent ShuttleSet22 contact test.

Start by writing `shuttleset22_test_plan.md`. Then prepare and save all 47
videos' contact predictions before opening any ShuttleSet22 contact label.
The fixed steps are under **Remaining ShuttleSet22 test plan** below.

## Resume

- Stage: plan the independent ShuttleSet22 contact test
- Exact next action: write and check `shuttleset22_test_plan.md` without opening test labels
- Branch: `contact-det-feasibility`
- Last runtime commit: `e6a16084` (`Allow original InpaintNet output range`)
- Current blocker: none
- Active worker or audit: none
- Last verified gate: all 47 inpaint outputs passed their full reload checks
- Large outputs: ignored under `scratch/contact_det_full_ds_fit/raw/`
- Unrelated local state: an old untracked `scratch/contact_det/scripts/__pycache__/`; leave it alone

Before doing work, read this file, the repository `AGENTS.md`,
`.github/AGENTS.md` and `.codex/context.md`. Open the detailed experiment files
below only when their stage needs them.

## Working rules

- Keep all new experiment work under `scratch/contact_det_full_ds_fit/`
- Use plain Australian English and normal speech
- Do not invent terms or define new jargon
- Keep machine paths, hostnames, access commands and credentials out of Git
- Never read `.env` or another credentials file
- Keep large arrays, models, logs and full results outside Git under `raw/`
- Work only on `contact-det-feasibility`; do not push or merge
- Stage files by exact path and make small local commits as work becomes stable
- Use short, natural commit messages
- Before each substantial stage, write the plan and have it checked
- Keep `worklog.md` and `RESUME.md` current before and after every long run
- Run long remote work in `tmux` and record meaningful milestones
- The Serena/Pyrefly MCP was not visible in the previous session

The user's earlier approval covers local edits, local commits and remote
experiment work for the agreed test. A choice that changes the fixed test
input, model, cut-off or nearby-contact distance needs the user's answer first.

## What the experiment set out to do

The experiment compares simple tree models for badminton contact detection.
It uses 40 suitable ShuttleSet videos to choose and fit the model. It then uses
non-overlapping ShuttleSet22 videos once as an independent test.

The contract is in `contract.md`. Its important rules are:

- a whole video stays in one data role
- no test video overlaps the 40 development videos
- ShuttleSet22 labels may score the finished setup but may not change it
- features and predictions must be saved before test labels are read
- contact timing, player side and complete-rally results are reported separately
- a complete rally needs every contact, no extra contact and the right player side
- large and machine-specific files stay out of Git

## Work completed

### Development split and features

The eligible ShuttleSet videos are fixed in
`shuttleset_development_split.json`. Videos 9, 10, 12 and 27 were excluded.
The remaining 40 videos were split into 32 training and eight validation
videos while the model design was chosen.

The feature preparation reuses the three-video pilot calculation without
changing it. The 40-video run produced 1,477,290 candidate rows. The three
pilot videos reproduced their old feature rows exactly before the full run.

Read these files when checking this stage:

- `contract.md`
- `current_system_map.md`
- `feature_preparation_audit.md`
- `scripts/freeze_contact_features.py`
- `scripts/feature_dataset.py`

### First model comparison

The first comparison used a fixed menu of nine HGB and random-forest runs.
The chosen model was `hgb_reference_raw_more_negatives`:

- histogram gradient boosting
- original per-frame motion values
- balanced class weights
- 31 leaves
- learning rate 0.06
- 180 iterations
- at least 40 samples per leaf
- L2 value 1.0
- at most 24 negative rows per positive row

On the eight validation videos, this run reached 0.8924 precision, 0.8344
recall and 0.8625 F1 within five frames at 30 frames per second. It produced
99 fully correct detected sections out of 609 accepted sections at ten frames.

The main weakness was the start of a rally. The model found 41.8% of first
contacts and 89.0% of later contacts at five frames. Missing contacts were much
more common than extra contacts.

The detailed result is in `baseline_report.md`.

### Rally-start follow-up

The experiment checked a small way to add one earlier contact to an otherwise
good detected section. Training-video scores came from first-stage models that
did not train on the video being scored.

The fixed follow-up compared logistic regression and shallow HGB at three
cut-offs. All six choices failed the pre-set training rules. The safest choice
made 147 changes, but only 76 were correct. No choice was tested on the eight
validation videos, and no validation label row was opened for this stage.

The final model therefore contains no rally-start addition. Read
`rally_start_model_report.md` for the checked result.

### Final setting from all 40 videos

The final cut-off and nearby-contact distance were chosen from held-out scores
for all 40 development videos. Five fits scored groups A, B, C, D and V. Each
fit trained on the other 32 videos.

The group row counts were:

- A: 367,951
- B: 294,802
- C: 255,293
- D: 275,881
- V: 283,363

They sum to 1,477,290. Group V reproduced the earlier validation identities
and probabilities exactly. Group A and the combined outputs matched their
repeats byte for byte.

The fixed 57-setting comparison kept the existing choice:

- score cut-off: 0.9
- nearby-contact distance: six frames at 30 frames per second

Across all 40 held-out score streams, the five-frame result was:

- 33,267 labelled contacts
- 31,824 predicted contacts
- 28,801 matched contacts
- 0.9050 precision
- 0.8658 recall
- 0.8849 F1
- 1,659 of 3,359 first contacts found, or 49.4%
- 27,142 of 29,908 later contacts found, or 90.8%

Read `final_contact_fit_plan.md` and `final_contact_fit_report.md` for the
method and result.

### Final model fit

The chosen HGB model was fitted once on all 40 development videos. It used
1,313,803 selected training rows, including 94,530 positive rows, and 85 model
input fields.

The final files remain outside Git:

- `raw/final_contact_model/contact_model.joblib`
- `raw/final_contact_model/final_contact_model_result.json`

Their identities are:

- model SHA-256: `ef7b66042ce2ed594572424ddd2c13f23092afcc8b259bccc8758af8cc11a8dc`
- fit-result SHA-256: `77d8ee023b53a68309055770a62757368eaa1ed22ba86deb584f924fdecae76e`
- final-setting result SHA-256: `82e6272ab4aa1ebc8c3c1a4fb2a45692dd74cdd05871501b9ea918ed80929e30`
- combined held-out score SHA-256: `d464d396af9ff451878f40ead57d46d2dbde3a61ebfbe70adee14519334707d9`
- final kept-contact file SHA-256: `947b87f3341edbb2a8a5f60bfacfd023f9a0ef45df507d38dbad6820b4f3471e`

The saved model reproduced the same probabilities for the first and last
candidate row from every development video after it was loaded again. An
independent audit checked all 40 training counts, all 80 reload rows, the
model settings, the hashes and the source commits. It found no blocker.

Use the same fitting environment for the test. The recorded versions are
Python 3.11.13, NumPy 2.2.6, scikit-learn 1.6.1 and joblib 1.5.3.

## ShuttleSet22 facts already checked

The source manifest has 58 IDs. Eight refer to videos already present in the
original ShuttleSet data:

- ShuttleSet22 1 maps to ShuttleSet 23
- ShuttleSet22 2 maps to ShuttleSet 38
- ShuttleSet22 3 maps to ShuttleSet 39
- ShuttleSet22 4 maps to ShuttleSet 41
- ShuttleSet22 5 maps to ShuttleSet 42
- ShuttleSet22 6 maps to ShuttleSet 43
- ShuttleSet22 7 maps to ShuttleSet 44
- ShuttleSet22 58 maps to ShuttleSet 24

IDs 14, 45 and 56 have no frame-aligned source. These 11 IDs are already
absent from the prepared set.

The 47 intended test IDs are:

```text
8 9 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
31 32 33 34 35 36 37 38 39 40 41 42 43 44 46 47 48 49 50 51 52
53 54 55 57
```

ShuttleSet22 IDs and ShuttleSet IDs are separate number systems. For example,
ShuttleSet22 ID 27 is not the excluded ShuttleSet video 27.

All 47 prepared videos report exact 30 fps and 1920 by 1080 frames. Each has:

- a compressed shuttle track and raw ball CSV
- five pose arrays
- court evidence and two court arrays
- a court receipt that checks the supplied files

The preparation records have these fixed identities:

- source manifest SHA-256: `746225f6b9bb1b257052224648c39e813792a75a7eb8711443688ca93fad7463`
- annotation tree SHA-256: `55f832221646229b8b65dea31e24e8d02e0876fd6d0799cb0f6eff12583e1485`
- prepared-artifact identity SHA-256: `dffe2cc2afc75f78eb89b30236477eb732f92a824b22ee3a01a4f893a673864e`
- official annotation SHA-256: `2c0208d13d13a4b72a9005ec16e92c442bfe5f223e0f9c499ea5a36f4339052c`

Before the current run, the search for existing inpaint sidecars and saved
shuttle guard arrays found none for a non-overlapping ShuttleSet22 video. The
47 prepared TrackNet CSVs use binary visibility only: 3,763,918 rows have zero
and 2,411,365 have one.

The preserved ShuttleSet22 extractor uses stride-8 TrackNet, large-video mode
and `enable_inpainting=False`. A later extraction report used stride 8 with
inpainting enabled, but that report covers one failed attempt to recover an
overlapping video. It produced no usable extract.

The guard array is separate from the inpaint fill record. The completed run
derives guard codes from repeated coordinate patterns in each finished shuttle
track.

The prepared 47-video set also has no annotation-stage output. Any test path
must produce the annotation result and exclusion mask needed by the existing
contact feature calculation.

Machine locations and access details are intentionally absent here.

## ShuttleSet22 inpaint run

The user chose the saved-coordinate InpaintNet path on 2026-08-28. The full
original-video GPU check found at most one-pixel changes on exactly
reconstructable inputs. The fabricated guard count stayed unchanged. Four
extra frames were marked degraded. The user accepted that difference.

The finished run used the normal non-overlapping 16-frame path on GPU. It read
the saved stride-8 TrackNet coordinates, used every visible coordinate in each
window and applied the normal 54-pixel gap rule. It kept one InpaintNet model
loaded and processed 16 windows in each GPU batch.

The run completed all 47 intended videos:

- 6,175,283 frames
- 2,916,960 frames selected for InpaintNet filling
- 2,411,365 visible coordinates before filling
- 5,322,221 visible coordinates after filling
- 47 complete per-video receipts

The result keeps the prepared pose and court files unchanged through links. It
adds an inpainted CSV, normalised track, fill sidecar, shuttle guard codes,
guard details and a receipt beside them in a writable mirror. The mirror's
folder name is `shuttleset22-inpainted-extract`. Its machine path stays outside
Git.

The final reload ran the complete validator again. It checked every input and
output hash, reloaded every CSV and track, checked every fill count, regenerated
every shuttle guard result and compared those results exactly. It finished with
`all requested videos are already complete`.

One output row is one pixel below the picture: video 51 has coordinate
`(1909, 1080)`. The original TrackNetV3 path does not clip InpaintNet output to
the frame, and the normal shuttle loader accepts and normalises such a value.
The validator now keeps strict bounds for the saved TrackNet inputs but accepts
the original InpaintNet output range. No other output row is outside the frame.

The checked run records are:

- aggregate receipt SHA-256: `ee5c55ec1ab0833e4bf0525dcabcf5b9eab5fde7c01dc08c47ab362ca447b160`
- deployed runner SHA-256: `2e4fe812168ef2a7abadbd8594cc3ba9bf92f0b2f4677edfbc80466fa018e1b1`
- tracked runner: `scripts/inpaint_shuttleset22_tracks.py`
- run plan: `shuttleset22_inpaint_plan.md`

## Remaining ShuttleSet22 test plan

The test design below was fixed before the input problem was found. Its
shuttle step is now complete. Keep every model and scoring choice unchanged.

### 1. Freeze the test contract

Add a tracked, path-free file with the exact 47 IDs, eight overlaps, three
unresolved IDs, dataset hashes, final-model hashes, fixed cut-off and fixed
nearby-contact distance.

Write `shuttleset22_test_plan.md` before implementation. State the chosen
shuttle preparation and its limitations. Do not read contact labels while
writing or implementing the label-free half.

Suggested commits are:

- `Plan the ShuttleSet22 test`
- `Prepare the ShuttleSet22 predictions`
- `Record the ShuttleSet22 result`

### 2. Prepare predictions without labels

For every test video:

1. Check the source manifest and completed inpaint receipt identities.
2. Load and validate the inpainted shuttle track, fill sidecar, guard arrays,
   pose arrays and court evidence.
3. Run the standard annotation stage with the chosen inpainting inputs.
4. Save the annotation result and exclusion masks outside Git.
5. Build the unchanged raw-per-frame contact features with `_fixture_rows`.
6. Keep candidate rows selected by at least one of the seven existing search regions.
7. Load the final model and save one probability for every candidate row.
8. Apply the fixed 0.9 cut-off and six-frame nearby-contact distance.
9. Replay the existing Top/Bottom player-side rule at every kept contact.
10. Save detected rally ranges, contact frames, probabilities and player sides.

Write one restartable result per video. A stopped run must show which videos
are complete. Save a complete combined prediction file before importing any
ShuttleSet22 label loader.

Build the combined predictions twice from the saved per-video files and
require equal bytes. This is a repeat check, not another model choice.

### 3. Read labels once

Use the established ShuttleSet22 cleaning rule:

- read each `set*.csv` table for the video
- parse `frame_num` as a number
- reject a whole rally when any frame is invalid or any row has `flaw` filled in
- order contacts by `ball_round`, then `frame_num`
- reject a rally whose contact frames are empty or do not increase strictly

The earlier ShuttleSet22 preparation reported 43,159 source rows, 38,218
usable rows and 3,422 usable rallies across these 47 videos. Treat those as a
recount check only. Do not change the cleaning rule to reproduce the totals.

Derive each labelled contact's Top/Bottom side from the player and opponent
vertical positions using the existing ShuttleSet22 rule. Report labels with no
side answer rather than guessing one.

### 4. Score the fixed result

Report contact timing at one, two, five and ten frames. Include:

- matched, labelled and predicted contact counts
- precision, recall and F1
- first-contact and later-contact recall
- per-video counts and rates
- timing error summaries

Report player-side accuracy separately among timing matches with a known human
side. Include both human-label coverage and prediction coverage.

Use the existing detected-section scorer for complete rallies. Report the
number of sections that match no labelled rally, one rally or several rallies.
At five and ten frames, separate missing contacts, extra contacts, mixed timing
errors, unanswered player sides and wrong player sides.

Keep the existing rally confidence curve. It shows how many sections remain
and how many are fully correct as the minimum saved contact probability rises.
Do not use this curve to revise the 0.9 event cut-off or any model setting.

### 5. Audit and close out

Before the result commit, independently recount the headline contact and rally
totals from the saved predictions. Check that:

- all 47 test videos appear exactly once
- none of the eight overlap videos appears
- model and input hashes match the plan
- all probabilities are finite and between zero and one
- the fixed cut-off and nearby-contact distance were applied exactly
- labels were first read only after the complete prediction file existed
- no machine path or access detail entered a tracked file
- the report does not suggest a setting change from the test result

Run the experiment tests, Ruff, the pinned Pyrefly check and the whole test
suite. Record every exit code. Write a plain-language result report and update
`decisions.md`, `worklog.md` and `RESUME.md`.

## Code expected to be reused

The test should be a new script under this experiment directory. Keep
production code unchanged unless the existing interface makes the test
impossible. Stop and ask before changing production code.

Useful existing functions are:

- `dataset_builder.vision.run_full_annotation_stage`
- `scratch.contact_det.scripts.freeze_tree_contact_features._fixture_rows`
- `scripts.score_contact_baseline.collect_candidate_rows`
- `scripts.score_contact_baseline.predictions_for_settings`
- `scripts.score_contact_baseline.contact_counts`
- player-side replay helpers in `scripts/save_validation_rally_predictions.py`
- complete-rally helpers in `scratch/contact_det/scripts/score_contact_rallies.py`

The old ShuttleSet22 feature comparison contains useful input and label
validation code. It is not part of the current checkout. Locate its preserved
source only when implementing the test. Reuse its checks rather than copying
its unrelated feature calculations.

## Checks already passed

At commit `31fbd55a`, before the final fit ran:

- experiment tests: 145 passed, exit 0
- whole project: 1,893 passed and 29 skipped, exit 0
- Ruff for the experiment directory: exit 0
- pinned Pyrefly: no errors and 21 suppressed messages, exit 0

The final remote fit then completed and passed its saved reload checks. The
independent result audit found no blocker.

The ShuttleSet22 inpaint run later passed these checks:

- all 47 base CSVs and saved tracks matched exactly before inference
- the InpaintNet checkpoint, model code and guard code matched the fixed hashes
- video 8 passed a full write, reload and input-identity check before the long run
- all 47 per-video receipts and outputs passed the final full reload
- the changed runner tests passed: 9 tests, exit 0
- Ruff passed for the changed runner and test files, exit 0
- the full project tests passed: 1,893 passed and 29 skipped, exit 0
- the pinned Pyrefly check found 0 errors and 25 suppressed messages, exit 0
- whole-project Ruff reported the same 863 existing findings outside this work,
  exit 1

## Stop conditions

Stop and ask the user when:

- a prepared video or expected hash is missing or different
- a test video overlaps the development data
- label code can run before the combined predictions are complete
- the completed inpaint receipt or any per-video output fails its saved check
- the test requires a production-code change
- a saved model does not load under the recorded library versions
- any completed result contains fewer or more than the fixed 47 videos

Do not soften a failed check to finish the run. Save enough state to explain
the failure and leave the next exact action in `RESUME.md`.
