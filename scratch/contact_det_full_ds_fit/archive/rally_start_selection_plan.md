# Plan for choosing an earlier rally-start contact

## Aim

Train a small model to decide whether one earlier contact should be added to a
detected section. Train and choose the model from the 32 training videos. Use
the eight validation videos once after every choice is fixed.

The model may add one contact or add nothing. It does not remove or replace an
existing contact.

## Fixed inputs

Use the completed 32-video rally-start input and the frozen validation input.
The candidate lists, first contact model, 0.9 contact score cut-off and
six-frame nearby-contact distance stay unchanged in this experiment.

The frozen validation candidate file does not contain player-side answers for
its unkept candidates. Before model fitting or label reading, save one checked
validation input that adds those answers:

- reproduce all 1,845 frozen candidate entries without changing them;
- replay the existing Top/Bot rule at every distinct kept or candidate frame;
- require the replay identities to equal those distinct video and frame pairs,
  with no missing or extra answer;
- require the kept-contact answers to match the saved validation rally file;
- save the section bounds, kept contacts and enriched candidate lists twice
  and require equal bytes; and
- save input names and hashes without machine paths or access details.

This is input preparation only. It does not read labels, train a model or
change a contact.

Before reading a human label row:

1. Check the split, model, score, section, candidate and player-side input
   hashes.
2. Check that all 32 training videos have first-model scores from a model that
   did not train on that video.
3. Check the saved 32-video counts and the combined-file hash recorded in
   `training_rally_start_input_summary.json`.
4. Reproduce every shared field in the frozen 1,845-entry validation list.
5. Write the result as `running`.

Read timing labels for the 32 training videos first. Check the file hash again.
Then read their player-side labels and check the hash again. Do not read a
validation label until the six model and cut-off choices have been scored on
the 32 training videos and one choice has been fixed. Do not read ShuttleSet22
labels. The training-label loader must return rows for the 32 training video
IDs only. The validation-label loader must be a separate call that cannot run
from the training or model-choice code.

## Training answer for each section

Use the original detected section bounds to find labelled rallies. A candidate
before the section does not change this assignment. The start frame is
included and the end frame is not.

- A section containing no labelled rally has no contact to add. Its two
  earlier candidates are negative training examples.
- A section touching more than one labelled rally is left out of training and
  counted in the report.
- If one labelled rally touches more than one detected section, leave all of
  those sections out of training and count them. This stops two sections from
  learning the same rally start.
- A section with exactly one rally that belongs to no other section uses that
  rally's first labelled contact.

If an existing kept contact in the section already matches the first labelled
contact under the ten-frame limit after adjustment to 30 frames per second,
the right answer is to add nothing. This remains true when the existing
contact has the wrong player side because an addition-only model cannot
replace it.

Otherwise, an earlier candidate is a correct addition only when:

- its frame is within the same ten-frame limit of the first labelled contact;
- it has a predicted player side; and
- that side matches the side label at the matched first-contact frame.

When both earlier candidates meet these rules, use the one with the smaller
frame error. Break an equal-error tie with the higher first-model contact
score, then the earlier frame. The other candidate is negative. This gives at
most one positive example per section.

A candidate with no predicted side can be a timing match, but it is negative
for model training because it cannot make a fully correct rally. It is not
counted as a wrong player-side answer. Keep these rows and report the
timing-only count separately.

## Fixed model inputs

Build one row for each earlier candidate in an included section. Use only
values saved before labels were opened:

- the candidate contact score;
- the fixed contact score;
- the candidate-to-fixed gap, adjusted to 30 frames per second;
- the signed candidate position from the section start, adjusted to 30 frames
  per second;
- the section length, adjusted to 30 frames per second;
- whether the candidate is already a kept contact;
- whether the candidate has a player-side answer;
- whether the fixed contact has a player-side answer; and
- whether both answers are present and name the same side.

Do not use a video name, video ID, player, match, tournament, group, absolute
frame number or any value from a human label as a model input.

## Small fixed model menu

Test two models:

1. Logistic regression with balanced class weights, `C=1.0` and at most 1,000
   fitting steps, using the standard `lbfgs` solver. Standardise its numeric
   inputs using values from its training groups only.
2. Histogram gradient boosting with balanced class weights, learning rate
   `0.05`, 100 fitting steps, at most seven leaves per tree, at least 20
   examples per leaf, L2 regularisation `1.0`, no early stopping and random
   seed `20260824`.

For each model, test fixed selection cut-offs of `0.5`, `0.7` and `0.9`. This
is six choices in total. Do not add settings or cut-offs after results are
read. Do not use XGBoost.

## Choose the model from the 32 training videos

Use the existing A, B, C and D video groups. In turn, train the candidate
model on three groups and predict the fourth. Keep every section and both of
its candidates in the same group. Fit class weights and logistic-regression
scaling from the three training groups only.

Combine the four held-out prediction files in A, B, C and D order. Each video
must therefore meet both separation rules:

- its first-model scores came from a first contact model that did not train on
  that video; and
- its candidate scores came from a candidate model that did not train on that
  video.

For each of the six fixed choices, select no candidate when neither score
reaches the cut-off. Otherwise select the higher candidate-model score. Break
an equal score with the higher first-model contact score, then the earlier
frame. A candidate without a player-side answer cannot be selected.

Apply each choice to a copy of the saved baseline contacts:

- keep every existing contact;
- add the selected candidate when it is not already kept;
- include an already-kept candidate once rather than adding a duplicate;
- move the section start to the candidate frame when the candidate is before
  the old start; and
- keep the old section end and every other section boundary unchanged.

Do not reuse the old helper that can replace a nearby fixed contact. The frozen
list puts each earlier candidate farther than the six-frame nearby-contact
distance from the fixed contact. Check that rule again and stop if it fails.

Also stop if a selected candidate belongs to another section, appears in more
than one list, or would move a start before the preceding section's end.

Score the result with the existing strict whole-rally code at the ten-frame
limit. A section is fully correct only when it has one labelled rally, has
one-to-one greedy timing matches with equal event and contact counts, passes
the stated timing-confidence requirement, and has an answered, correct side
for every event. Use zero as the whole-section confidence requirement for the
main comparison because the selected contact is expected to have a first-model
score below 0.9. Also report the existing 0.9 confidence view, the five-frame
timing result and every added or lost fully correct section.

Compare changes by stable video and section identity. Also report the labelled
rally identity for every fully correct section. Fail if one labelled rally is
counted as fully correct more than once. Report every labelled rally that
touches more than one validation section, and leave those sections out of the
recovery-rate denominator even when neither becomes fully correct.

A choice may continue to validation only when its held-out 32-video result:

- gains at least ten fully correct sections at the ten-frame limit;
- loses no baseline fully correct section at zero or 0.9 confidence;
- selects a correctly timed and correctly sided first contact for at least 80%
  of its non-empty choices, with the numerator and denominator reported; and
- recovers at least 20% of the unambiguous one-rally sections that have a
  correctly timed candidate with a usable predicted side, again with both
  counts reported.

A lost section means that the same video and section was fully correct in the
baseline at that confidence requirement and is not fully correct after the
addition.

Among choices that pass, use the one with the most fully correct sections.
Break ties with fewer added contacts, then higher correct-addition rate, then
logistic regression, then the higher selection cut-off. The last rule makes an
otherwise equal choice more cautious. If none pass, record that result and
stop this rally-start follow-up.

## One validation check

Fit the chosen model on all 32 training videos. Do not change its inputs,
settings, cut-off or tie rules. Predict the two earlier candidates in the
eight validation videos. Fix and save those scores before reading validation
timing or player-side labels. Read validation timing first, check the label
file hash, then read player side and check the hash again.

Keep the model only when the validation result:

- gains at least five fully correct sections at the ten-frame limit;
- loses no baseline fully correct section at zero or 0.9 confidence;
- leaves every video's fully correct count at least as high as its baseline;
- selects a correctly timed and correctly sided first contact for at least 80%
  of its non-empty choices, with both counts reported;
- recovers at least 20% of the unambiguous one-rally sections that have a
  correctly timed candidate with a usable predicted side, with both counts
  reported; and
- does not reduce contact F1 at the ten-frame limit.

Report the five-frame contact result as a stricter timing check. Report
timing-only matches and missing player-side answers separately. Do not describe
a timing-only match as a fully correct rally.

## Repeat and review checks

- Check every saved row against its video, section, candidate and label
  identity.
- Save each training row's video and section identity, candidate identity,
  rally-assignment status, target first-contact identity when there is one,
  and target action. Keep this detailed file outside Git.
- Check that candidate identities are unique within a section and that no
  section receives more than one positive training answer.
- Check all four candidate-model training lists and their held-out videos.
- Save every held-out candidate score before choosing among the six fixed
  choices.
- Build every changed contact stream twice and require equal bytes.
- Recount the main training and validation results from the saved detail.
- Run focused tests, Ruff, the pinned type check and the whole test suite.
- Have a fresh read-only reviewer check the code before each implementation
  commit and the result before its report commit.

The planned commits are:

- `Plan the rally-start contact choice`
- `Save the validation rally-start inputs`
- `Train the rally-start contact model`
- `Record the rally-start contact result`

## After the validation result

If the model passes, freeze its inputs, settings and selection cut-off. Then
make the final first-model predictions across all 40 development videos. Use
five groups of eight: the existing A, B, C and D groups and the existing eight
validation videos as group V. Train on four groups and score the fifth so each
video is scored by a first contact model that did not train on it. Use those
predictions to choose the final first-model score cut-off and nearby-contact
distance from the already fixed menu.

Rebuild the 40 candidate lists with those final settings. Use the same five
groups to make held-out candidate-model predictions across all 40 videos. The
candidate model that scores a group must train on the other four groups.
Recheck the frozen model design without changing it, then fit the candidate
model on all 40 held-out first-model results.

Fit the first contact model on all 40 videos. Only then run the finished setup
once on the non-overlapping ShuttleSet22 videos.

## Outside this stage

- changing the first contact features or model
- changing the candidate-list construction
- removing or replacing contacts
- training a rally acceptance model
- adding another player-side model
- XGBoost
- opening ShuttleSet22 labels or using ShuttleSet22 to make any choice
- production-code changes
