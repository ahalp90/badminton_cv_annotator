# Plan for rally-start training inputs

## Aim

Prepare the saved inputs needed to train a small rally-start candidate chooser.

For each of the 32 training videos, save:

- the detected section boundaries from the existing video run;
- every kept contact from the model that did not train on that video;
- the same fixed rally-start candidate list already checked on validation; and
- the existing Top/Bot player-side answer for every kept contact and
  candidate.

This stage does not read contact, rally or player-side labels. It does not
train a candidate chooser.

## Why this stage is needed

The combined training score file contains the video, search interval, frame,
frame rate, contact score, kept flag and source group for every score row. It
does not contain detected section boundaries.

The candidate rule needs both search intervals and detected sections. They are
different parts of the video run and cannot stand in for each other. The
existing checked video outputs remain the source of the section boundaries.
These are sections detected by the existing video pipeline. They are stored in
the saved `annotator_result` file and are separate from ShuttleSet's human
contact and rally labels. Require the expected result schema and video identity,
then check the saved file's name, size and hash before reading the sections.

## Fixed inputs

Use:

- `training_video_score_summary.json`;
- the combined training score result and score file named by that summary;
- `training_video_score_inputs.json`;
- `training_video_score_groups.json`;
- `shuttleset_development_split.json`;
- the raw-motion feature record and its 40 checked feature files; and
- the saved shuttle, pose, court and annotation files already
  named and hashed by each video's feature record.

Check every tracked file and saved result hash before opening a large video
file. Check each video's saved input filename, size and SHA-256 hash before
reading that input.

The contact-label file may be checked as bytes because its hash is already in
the fixed input list. Do not parse any row from it in this stage.

## First reproduce the validation list

Before preparing the 32 training videos, run the shared candidate-building
function on the eight validation videos.

Require exact equality with the frozen validation construction for:

- video and section order;
- section start and end frames;
- search interval and prefix start;
- fixed contact frame;
- scaled nearby-contact distance;
- candidate frame, score and fixed-contact marker; and
- all 1,845 candidate entries.

The new file may contain extra checked fields such as the kept flag and
predicted player side. Compare the shared fields rather than the compressed
file bytes.

Stop if any shared field differs.

Also replay the predicted side for every kept validation contact. Require exact
equality with the saved validation rally-prediction file. This checks the new
side path without opening its labelled comparison rows.

## Fixed candidate rule

Keep the existing rule unchanged for every video:

1. Use detected sections in their saved order.
2. Skip a section with no kept contact and count it separately.
3. Use the earliest kept contact inside the section as the fixed contact.
4. Use score rows from the same search interval as that fixed contact.
5. Start the earlier search at the interval start. When the preceding section
   ends inside the same interval, start at that preceding end instead.
6. Include the search start and stop before the fixed contact frame.
7. Rank earlier rows by higher contact score, then earlier frame.
8. Keep a row only when it is farther from every already chosen frame than the
   fixed six-frame distance after adjustment for frame rate.
9. Save the fixed contact first, followed by at most two earlier candidates.

Keep at most three entries per section. Do not copy the validation-wide limit
of 1,845 entries to the larger 32-video set. Report the resulting total instead
of cutting the lists short.

An earlier candidate can already be a kept contact outside the detected
section. Save its kept flag. The later chooser may then move the section start
to include that contact without adding a duplicate event.

## Player-side answers

Use the same `attribute_half` Top/Bot rule used for the validation result.
Run it once for every distinct kept or candidate frame.

Before using the answer, check the centre-frame feature values against the
saved shuttle, pose and court inputs in the same way as the validation
player-side replay. Rebuild the existing sticky player values from those
inputs.

Save `Top`, `Bot` or no answer. Do not open the labelled player-side column.

## Saved output

Write one compressed JSON file per video and one combined compressed JSON file
outside Git. Each result says `running` before checks begin and `complete` only
after its checks pass.

Save only portable information:

- source commit, input filenames and hashes;
- `labels_read: false`;
- videos in fixed A, B, C and D group order;
- frame rate, frame count and checked stage-file records per video;
- ordered, non-overlapping half-open section boundaries;
- kept contacts with frame, score, interval, section and predicted side;
- candidate lists with the existing construction fields, kept flag and
  predicted side;
- sections skipped because they contain no kept contact; and
- per-video, per-group and full counts.

Do not save machine paths, server names or access commands.

For each video, build the saved value twice from the already loaded arrays and
require equal JSON bytes before writing it. A resumed run may reuse a complete
video only after rechecking all of its input hashes and saved contents.

After all videos pass, combine the checked per-video files twice in fixed A, B,
C and D order. Require equal JSON bytes. Read every compressed file back and
require the same bytes.

## Checks before the full run

- the combined score file is complete and has exactly 1,193,927 unique rows;
- each score row has the seven fixed fields, including its group;
- every score row belongs to its recorded group and expected video;
- each group result names the expected eight scored videos and the other 24
  training videos;
- no scored video occurs in the training list for the model that scored it;
- the 32 videos occur in fixed A, B, C and D group order;
- no validation video occurs in the training score rows;
- search interval IDs and frames match the raw feature record;
- sections are ordered, half-open, non-overlapping and inside the video;
- validation candidate construction matches all 1,845 frozen entries;
- validation kept-contact sides match the saved validation prediction file;
- player-side replay checks the saved centre-frame values;
- contact and player-side label readers cannot run;
- focused tests cover the training row type, group order, model separation,
  detected-section checks and side replay;
- a small three-video replay passes; and
- an independent code review finds no remaining blocker.

Run the 32 videos in `tmux`. Save progress per video so a stopped run has a
clear last completed video. A resumed run may reuse a video only after checking
its result and every input hash again.

## Checks after the full run

- all 32 videos are complete and appear once;
- every group and video count adds to the saved full count;
- every kept score row appears once in the kept-contact records;
- no candidate identity is repeated between sections;
- no section has more than three candidates;
- every candidate identity maps to exactly one combined score row;
- all scores and predicted sides match their checked source rows;
- a full repeated build produces the same bytes; and
- the copied local file has the same SHA-256 hash as the compute copy.

## Outside this stage

- no contact, rally or player-side label rows
- no positive or negative candidate labels
- no logistic regression, decision tree or other chooser
- no chooser score cut-off
- no change to the first contact model
- no change to validation predictions
- no final fit on all 40 videos
- no ShuttleSet22 labels

The planned implementation commit is:

`Save rally-start inputs for training videos`
