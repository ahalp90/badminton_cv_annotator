# ShuttleSet22 contact test plan

## Aim

Score the finished contact detector once on the 47 non-overlapping
ShuttleSet22 videos. Save every feature, model probability, contact decision
and player-side prediction before reading a ShuttleSet22 contact label.

The test measures the finished setup. Its result cannot change the model,
features, score cut-off or nearby-contact distance.

## Unknown human-side result

The ShuttleSet22 tables are expected to give an implicit human Top/Bottom
answer for every accepted contact. A contact has no human answer only when
its player or opponent vertical position is missing, or when the two
positions are equal.

If this unexpected case occurs:

- keep these contacts and rallies in timing and section-mapping counts;
- exclude timing matches without a known human side from player-side accuracy;
- classify a rally containing an unknown human side as
  `human side unassessable`; and
- report fully correct rally accuracy among rallies whose human sides are all
  known, alongside the count and share that are unassessable.

This avoids guessing a human answer and avoids counting missing human data as
a model error. The user accepted this fallback on 2026-08-28.

The completed inpaint mirror's machine path and the exact DeepSeek audit model
are run details. They stay outside this file and do not affect the test.

## Fixed video set

Use these ShuttleSet22 IDs:

```text
8 9 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
31 32 33 34 35 36 37 38 39 40 41 42 43 44 46 47 48 49 50 51 52
53 54 55 57
```

Use the existing court-evidence identity as the annotation and feature fixture
name: the decimal ShuttleSet22 ID without a leading zero, from `8` through
`57`. `freeze_tree_contact_features._fixture_rows` passes this identity to the
court loader, so it must remain equal to the saved `court_evidence.json.gz`
`video_id`. The ShuttleSet22 ID also remains an integer in every saved record.

Use `ss22_08` through `ss22_57` only as path-free output directory names. The
directory prefix makes the two dataset number systems clear without changing
an input identity.

Exclude the eight known overlaps:

| ShuttleSet22 ID | Existing ShuttleSet ID |
|---:|---:|
| 1 | 23 |
| 2 | 38 |
| 3 | 39 |
| 4 | 41 |
| 5 | 42 |
| 6 | 43 |
| 7 | 44 |
| 58 | 24 |

IDs 14, 45 and 56 have no frame-aligned source and remain excluded.

Stop if the source manifest, prepared inputs, prediction files or label tree
contain a different set of test IDs.

### Why the official 58 matches become 47 test videos

The official ShuttleSet22 annotation set contains all 58 match IDs. The fixed
source manifest accounts for every one of them:

- 47 have a separate downloadable video that was prepared for this test;
- eight reuse videos already present in the base ShuttleSet dataset; and
- IDs 14, 45 and 56 have no resolved frame-aligned public video source.

The three unresolved IDs still have official annotations. Their exclusion is
about video availability, not missing labels. The source manifest records the
reason found during source preparation:

- ID 14: the public full-session video is not frame-aligned with the trimmed
  annotations;
- ID 45: the annotations use a trimmed video, with no frame-aligned full source
  found publicly; and
- ID 56: the exact 2022 semifinal video could no longer be found, while the
  available search results were different matches.

These are source-search findings, not independently verified explanations from
the dataset authors. Keep their manifest status as `unresolved`.

The eight overlaps are excluded to keep the final test separate from the
development dataset. This leaves the 47 downloadable, non-overlapping videos
listed above.

The source manifest is the durable record of this accounting. It must contain
58 ordered entries and classify them as exactly 47 `download`, eight
`shuttleset_overlap` and three `unresolved` entries. The scorer authenticates
the complete 58-match annotation corpus, checks the official match ID and
directory map, then reads `set*.csv` tables only for the fixed 47 test IDs.

## Fixed identities

Require these SHA-256 identities:

### ShuttleSet22 inputs

- source manifest:
  `746225f6b9bb1b257052224648c39e813792a75a7eb8711443688ca93fad7463`
- prepared annotation tree:
  `55f832221646229b8b65dea31e24e8d02e0876fd6d0799cb0f6eff12583e1485`
- prepared artifact identity:
  `dffe2cc2afc75f78eb89b30236477eb732f92a824b22ee3a01a4f893a673864e`
- official annotation corpus:
  `2c0208d13d13a4b72a9005ec16e92c442bfe5f223e0f9c499ea5a36f4339052c`
- completed inpaint run record:
  `ee5c55ec1ab0833e4bf0525dcabcf5b9eab5fde7c01dc08c47ab362ca447b160`
- deployed inpaint runner:
  `2e4fe812168ef2a7abadbd8594cc3ba9bf92f0b2f4677edfbc80466fa018e1b1`

The source manifest is the `configs/shuttleset22/sources.toml` blob at commit
`ba24a95c334300c78e30a8d1b7c2a6134b8b5fa9`. Its blob is still available in
the repository history and has the required source-manifest hash. Materialise
that exact blob outside Git for the run rather than reconstructing a new
manifest from directory names.

The official annotation provenance is the public
[`CoachAI-Challenge-IJCAI2023/ShuttleSet22`](https://github.com/wywyWang/CoachAI-Projects/tree/45517f7d4cb936b03f3eabf939cc7959d39226fe/CoachAI-Challenge-IJCAI2023/ShuttleSet22)
tree at upstream commit
`45517f7d4cb936b03f3eabf939cc7959d39226fe`. The pinned corpus and tree hashes
above authenticate that snapshot.

The label-free program checks the source manifest, prepared artifacts and
inpaint records. It receives no annotation-tree or official-annotation path.
The scoring program checks the two annotation identities only after it has
accepted the completed combined prediction file.

The official annotation-corpus identity is calculated from
`set/match.csv`, followed by every sorted `set/*/set*.csv` path. For each file,
hash its annotation-root-relative POSIX path and file bytes. Prefix both with
their lengths as unsigned eight-byte big-endian integers. This is the pinned
historical ShuttleSet22 checksum rule; do not replace it with a generic tree
hash.

### Final detector

- model:
  `ef7b66042ce2ed594572424ddd2c13f23092afcc8b259bccc8758af8cc11a8dc`
- final-fit result:
  `5428bb69be41aea034fe56f5b812594404d1ac458392681f853b74a26600b4ed`
- final-setting result:
  `9c21575c457742bf71dea6a9105ba91234f1b0038ee70bc3f9d885c56ce8ac83`
- combined held-out development scores:
  `d464d396af9ff451878f40ead57d46d2dbde3a61ebfbe70adee14519334707d9`
- final kept-development contacts:
  `947b87f3341edbb2a8a5f60bfacfd023f9a0ef45df507d38dbad6820b4f3471e`

Load the model under Python 3.11.13, NumPy 2.2.6, scikit-learn 1.6.1 and
joblib 1.5.3. Require these versions before loading the model. Stop if the
model, result or recorded model-input fields differ.

## Fixed detector settings

Use the setup chosen by development run
`hgb_reference_raw_more_negatives`. The final-fit result records the fitted
estimator name as `hgb_reference`; the longer run ID also identifies the raw
motion, balanced-weight and negative-sampling choices below.

- histogram gradient boosting;
- original per-frame motion values;
- balanced class weights;
- 31 leaves;
- learning rate 0.06;
- 180 iterations;
- at least 40 samples per leaf;
- L2 value 1.0; and
- at most 24 negative rows per positive row during the completed fit.

Use every saved model-input field in the order recorded by
`final_contact_model_result.json`.

Apply a contact score cut-off of 0.9. Within each search interval, keep the
strongest score from contacts no more than six frames apart at 30 frames per
second. All test videos are exactly 30 frames per second, so the applied
distance is six frames.

Do not use the failed rally-start addition.

## Fixed shuttle preparation

Use the completed InpaintNet outputs made from the saved stride-8 TrackNet
coordinates. The run used non-overlapping 16-frame windows, every visible
coordinate in each window and the normal 54-pixel gap rule. Its guard codes
come from the finished inpainted tracks.

The accepted limitations are:

- GPU output differed by at most one pixel where old original-video inputs
  could be reconstructed exactly;
- four more frames in that check were marked degraded;
- ShuttleSet22 video 51 has one inpainted coordinate at `(1909, 1080)`; and
- the normal shuttle loader accepts and normalises this original unclipped
  InpaintNet output.

Keep strict frame bounds for the saved TrackNet inputs. Accept the original
InpaintNet output range when checking the completed inpainted CSV and track.
Do not clip, replace or rerun an inpainted coordinate.

## Program boundary

Use two separate programs under `scripts/`:

1. `prepare_shuttleset22_predictions.py` validates vision inputs, runs the
   annotation stage, builds features, loads the final model and freezes all 47
   predictions. It has no label-root argument and imports no ShuttleSet22
   label reader.
2. `score_shuttleset22_test.py` first validates the complete combined
   prediction file without inspecting the label root. Only then does it load
   the ShuttleSet22 labels and calculate the fixed result.

The label reader lives in the scoring program. The prediction program must be
able to import and run in an environment where the label tree is absent.

The scorer writes a `running` result before reading labels. A failed score
keeps its error type and whether label reading started. It cannot look
complete.

## Prepare one video without labels

For each fixed video, perform these steps in order.

### 1. Check inputs

- Require one source-manifest row, one prepared directory and one completed
  inpaint directory for the ID.
- Reuse the completed inpaint validator. Check its receipt and every saved
  input and output hash.
- Validate the court receipt, the five pose arrays, three court files,
  inpainted shuttle track, inpaint fill sidecar and shuttle guard array.
- Require exact 30 fps, 1920 by 1080 pixels and equal frame counts across all
  frame-aligned inputs.
- Rebuild the boolean fill mask from the saved half-open sidecar intervals.
  Require its count to equal the inpaint receipt.
- Record basenames, sizes and hashes. Do not save a machine path.

### 2. Run the standard annotation stage

Load the inpainted shuttle track, fill mask, guard codes, pose arrays and court
evidence. Call `dataset_builder.vision.run_full_annotation_stage` with the
normal shipped configuration.

Save these outputs outside Git:

- `annotator_result.json.gz`;
- `raw_replay_mask.npy.xz`;
- `definitive_exclusion_mask.npy.xz`; and
- `shuttle_quality.json.gz`.

Use a temporary standard stage layout so the unchanged feature helper can
load its normal filenames. The temporary layout may link the checked read-only
inputs. Remove that layout before publishing the completed per-video result.
Keep the four annotation outputs as ordinary files.

### 3. Build and save features

Call
`scratch.contact_det.scripts.freeze_tree_contact_features._fixture_rows` with
`motion_mode="raw_per_frame"`. Save the returned full structured feature rows
and feature summary.

Require:

- the seven existing region fields and the recorded 85 model-input fields;
- ordered, unique frame identities within each search interval;
- a fixture identity equal to the decimal ShuttleSet22 ID stored by its court
  evidence;
- exact 30 fps in every row; and
- at least one row selected by the union of the seven search regions.

Keep candidate rows selected by at least one of the seven region fields. Save
the full features and candidate rows before model scoring.

### 4. Score and keep contacts

Load the checked final model once for the run. Require model classes `[0, 1]`.
Save one finite probability between zero and one for every candidate row.

Apply the fixed 0.9 cut-off and six-frame nearby-contact distance with
`predictions_for_settings`. The saved kept flags must reproduce the returned
frames exactly.

For every kept frame, rebuild the existing sticky player inputs and call
`annotator.point_winner.attribute_half`. Save `Top`, `Bot` or no answer. Do
not replace a missing answer.

Save the annotation spans as half-open ranges. For every kept contact save:

- frame;
- model probability;
- predicted player side; and
- containing span ID, or no span when the contact is outside every span.

### 5. Publish atomically

Write one video under a `.working` directory. After all semantic checks pass,
write a path-free receipt and rename the directory to its final `ss22_XX`
output name.

A restart accepts a completed video only after reloading and checking every
saved hash, feature identity, candidate probability, kept decision, player
side and annotation span. Stop when a `.working` directory remains or a
completed result has changed. Leave the partial state for diagnosis.

The run-state file lists all expected IDs, completed IDs and counts. Update it
after each completed video.

## Saved prediction layout

Keep all large files under the ignored directory
`raw/shuttleset22_test_predictions/`:

```text
run_state.json
combined_predictions.json.gz
videos/
  ss22_08/
    annotation/
      annotator_result.json.gz
      raw_replay_mask.npy.xz
      definitive_exclusion_mask.npy.xz
      shuttle_quality.json.gz
    contact_features.npy.xz
    candidate_scores.npy.xz
    predictions.json.gz
    result.json
  ...
```

`candidate_scores.npy.xz` contains candidate identity, probability and kept
flag in stable row order. `predictions.json.gz` contains spans and the kept
contact details. `result.json` binds all input and output identities, model
identity, settings, source commit and `labels_read: false`.

Use deterministic JSON and gzip encoding. Keep timestamps and elapsed times
out of files used by the repeat check.

## Freeze all 47 predictions

After every per-video result passes its reload check:

1. require all 47 fixed IDs exactly once and no other ID;
2. require the fixture order from this plan;
3. require the same model, code and fixed setting identities for every video;
4. combine the saved per-video prediction files without recomputing a feature,
   probability, kept decision or player side;
5. write `combined_predictions.json.gz`;
6. rebuild it independently from the saved per-video files; and
7. require equal bytes between the first and second builds.

The run is complete only after the combined file passes a fresh reload and
the run-state file records all 47 IDs. Record the combined file's hash and
creation time outside the combined payload. That file must exist before the
scoring program is implemented or run.

## Read the labels once

The scoring program first checks the complete combined prediction file and
its 47 per-video children. This validation accepts no label path and performs
no label import.

After that check passes:

1. require the source manifest identity again;
2. require the official 58-match annotation-corpus and annotation-tree hashes;
3. require the official match ID and directory map, then select exactly the 47
   fixed videos; do not read the eight overlap or three unresolved directories;
4. read every `set*.csv` table for a video;
5. add the set filename stem as `set_id`;
6. group rows by `set_id` and `rally`;
7. parse `frame_num` with invalid values changed to missing;
8. reject a whole rally when any frame is missing, negative or outside the
   video's saved frame count;
9. reject a whole rally when any row has `flaw` filled in;
10. order accepted rows by `ball_round`, then `frame_num`; and
11. reject a rally whose contact list is empty or does not increase strictly.

This reproduces the preserved rule from
`annotator.calibration.shuttleset22_features.load_annotation_rallies` at
commit `40beec6`. The earlier totals of 43,159 source rows, 38,218 usable rows
and 3,422 usable rallies are recount checks. A different total stops the run
for inspection; it does not change the cleaning rule.

For each accepted contact, parse `player_location_y` and
`opponent_location_y` as numbers. Save no human side when either value is not
finite or the values are equal. Otherwise save `Top` when the player's value
is smaller and `Bot` when it is larger. This is the preserved `_player_slot`
rule from the same source.

Save the clean label result and its exclusions outside Git before scoring.
Record source-row, usable-row, usable-rally and exclusion counts by video and
in total.

## Score the fixed result

Use the existing closest-first, one-to-one contact matcher. Report timing at
one, two, five and ten frames. For each tolerance report:

- labelled, predicted and matched contact counts;
- precision, recall and F1;
- first-contact and later-contact recall;
- signed and absolute timing error summaries; and
- the same counts and rates for every video.

Report player side separately among timing matches whose human side is known.
Include:

- human-side coverage among timing-matched labels;
- prediction coverage among timing matches with a known human side;
- correct player sides and accuracy when both sides are answered; and
- counts by video and timing tolerance.

Use the existing half-open detected sections and complete-rally scorer. For
every detected section, report whether it maps to no labelled rally, one
labelled rally or several labelled rallies. Also report retained contacts that
fall outside every detected section.

At five and ten frames, separate these outcomes:

- missing contacts only;
- extra contacts only;
- both missing and extra contacts;
- timing mismatch with equal contact counts;
- predicted player side unanswered;
- wrong predicted player side;
- human side unassessable; and
- fully correct.

Keep the existing minimum-contact-probability confidence curve. For each
requirement report the number of sections retained, the number fully correct
and their ratio. The curve is descriptive only. It cannot change the 0.9
event cut-off, six-frame distance or any model setting.

## Risk and checks

### A. Plan and report changes

Risk: wording or a saved identity could misstate the fixed test.

Checks:

- compare every ID, hash and setting with `HANDOVER.md` and the saved final-fit
  result;
- search tracked changes for machine paths, hostnames and access details; and
- run `git diff --check`.

### B. Label-free prediction code

Risk: a changed feature, row order, model input or nearby-contact rule could
silently change predictions.

Checks:

- focused tests for input receipts, fill-mask rebuilding, fixture identities,
  region-union selection, exact model-field order and contact tie ordering;
- a test proving the prediction program imports and runs with no label tree;
- a test proving label-related arguments and imports are absent;
- an atomic-write and restart test;
- a repeat combined-file byte check;
- a one-video real-input smoke test in the recorded model environment;
- a fresh reload of all 47 saved videos after the long run.

### C. One-time label scoring

Risk: labels could be opened early, cleaned differently or used to revise the
finished detector.

Checks:

- test that an incomplete or changed combined prediction file stops before
  the table reader is called;
- focused tests for whole-rally rejection, row order, non-increasing frames,
  unknown human sides and all reported score categories;
- recount the historical 43,159 source rows, 38,218 usable rows and 3,422
  usable rallies without changing the fixed rule;
- independently recount headline contact and rally totals from the saved
  predictions and cleaned labels;
- require all 47 test videos exactly once and no overlap ID;
- require every probability to be finite and between zero and one;
- require kept flags to reproduce the 0.9 cut-off and six-frame rule exactly;
- verify the prediction file predates the first recorded label read; and
- independently recount the headline result from the saved result files.

For each code stage, run the focused experiment tests, Ruff on the changed
files, the pinned whole-project Pyrefly check and the whole test suite. Run the
repository-wide Ruff command for the required final record and report its
exit code separately from changed-file Ruff.

## Files outside this work

Do not change:

- production code under `src/`;
- the three-video pilot under `scratch/contact_det/`;
- the final model or any saved development feature, score or result;
- the completed ShuttleSet22 inpaint mirror or prepared source extract;
- the fixed test IDs, input identities, model, model-input fields, 0.9 cut-off
  or six-frame nearby-contact distance;
- a label file or source annotation; or
- the user's existing uncommitted `HANDOVER.md` addition.

Do not push, merge or rewrite history. Stop and ask before a production-code
change or any choice that changes a detector output.

## Planned commits

Use these already agreed local commit messages:

- `Plan the ShuttleSet22 test`
- `Prepare the ShuttleSet22 predictions`
- `Record the ShuttleSet22 result`

The first commit contains this reviewed plan only. The second contains the
label-free code, focused tests and tracked run record. The third contains the
scoring code, focused tests, path-free summary, report and updated experiment
records.

Use focused tests and inspect the saved outputs at each stage. Further
independent reviews are optional when a focused check gives the same evidence.

## Stop conditions

Stop and leave a clear failed or partial record when:

- a fixed video, source row, prepared input, receipt or expected hash is
  missing or different;
- any test video overlaps the 40 development videos;
- label code can load before the complete prediction check passes;
- a completed inpaint or per-video prediction output fails its saved check;
- the final model cannot load under the recorded library versions;
- the test needs a production-code change;
- the combined prediction contains fewer or more than 47 videos;
- the combined repeat differs by one byte; or
- the fixed label-cleaning rule cannot be applied to the supplied tables.

Do not weaken a failed check to finish the test. Update `worklog.md` and
`RESUME.md` with the failure and the next exact action.
