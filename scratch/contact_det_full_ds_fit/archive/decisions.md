# Decisions

## Validation videos

Use these eight videos for validation:

`sset_18, sset_22, sset_24, sset_25, sset_30, sset_31, sset_39, sset_40`

This gives:

- one 25 fps video and seven 30 fps videos
- four women's matches and four men's matches
- ten players who do not appear in the 32 training videos
- matches from All England, YONEX Thailand Open, Toyota Thailand Open and the World Tour Finals

The ten unseen players are SHI Yuqi, Mia BLICHFELDT, Busanan ONGBAMRUNGPHAN, Rasmus GEMKE, Supanida KATETHONG, Sameer VERMA, Neslihan YIGIT, LEE Zii Jia, Evgeniya KOSETSKAYA and Michelle LI.

The main alternative was `sset_18` plus `sset_38` through `sset_44`. That would hold out all videos from one 30 fps broadcast recording set. It would include fewer unseen players and a less balanced mix of women's and men's matches. ShuttleSet22 already provides a later test on a different dataset, so the unseen-player split is more useful here.

The user accepted this split on 2026-08-27.

## How the final cut-offs will be chosen

First choose the model design with the 32 training and eight validation videos.

Then make predictions for all 40 ShuttleSet videos using several trained models. Each video is predicted by a model that did not train on that video. These are out-of-fold predictions.

Use those predictions to choose the final score cut-off and the distance used to merge nearby duplicate contact predictions. Then train the chosen model once on all 40 videos.

This gives cut-offs based on unseen-video scores without using ShuttleSet22 labels.

Use five groups of eight: the existing A, B, C and D groups plus group V with
the eight former validation videos. Rerun every group with exactly the other
32 videos used for training. The old A–D files used 24 training videos and are
not part of the final setting choice.

Save all raw scores before choosing the final pair. Use only the 19 cut-offs,
three nearby-contact distances and tie order already fixed in
`baseline_runs.json`. The all-40 result is part of model fitting, not an
independent test.

## Rules for any later learned step

A later model that removes extra contacts, adds missing contacts or decides which rallies to keep must use out-of-fold predictions from the first contact model.

If extra contacts cause most complete-rally errors, try removing extras first. Try adding likely missing contacts only when many otherwise-good rallies are one contact short.

## Keep the first comparison small

Write down no more than 12 full model runs before training begins. Keep XGBoost outside the first comparison.

## First baseline chosen

Use `hgb_reference_raw_more_negatives` as the first contact baseline.

It has the best contact F1 and the most fully correct accepted sections among
the nine fixed runs. The run uses the reference HGB model, original motion
values, balanced class weights and up to 24 negative examples per positive
example.

Missing contacts are the main timing problem. In particular, the model finds
41.8% of first contacts within five frames at 30 frames per second, compared
with 89.0% of later contacts. The next small contact test should focus on rally
starts or on adding one missed contact. The result does not support removing
extra contacts next.

The saved-score check narrows that next step. At ten frames, 284 of 364 missed
first contacts have a saved candidate nearby, usually below the 0.9 cut-off.
All 94 otherwise-correct one-short sections have a nearby candidate, and 81
are missing the first contact. Test a small rally-start candidate list next.
Include a limited number of frames before the detected section because 39 of
those 81 sections have candidates only before the section starts.

Do not lower the cut-off for every contact. Most missed later contacts have no
saved candidate nearby, so a broad cut-off change would add predictions without
addressing that problem.

## Rally-start candidate limits

Use the first kept contact in each detected section and no more than two
earlier HGB score rows. The list must contain no more than three entries for
one section or 1,845 entries across validation.

Continue only if the list covers at least 50 of the 81 otherwise-correct
sections missing their first contact. It must also add no more than 25 list
entries for each covered first contact.

These limits were fixed and independently checked before the candidate list
was built or the saved missed-contact detail was opened.

## Rally-start candidate result

Keep the fixed list. It covers 56 of the 81 target first contacts at ten
frames. It adds 21.96 earlier entries per covered contact, so all four limits
pass.

Do not add all candidate entries to the contact stream. Do not reuse the
pilot's hand-written choice rule. The matching and non-matching candidate
scores overlap too much to support a simple largest-score rule.

Plan a small trained choice method next. Each training video's first-model
scores must come from a model that did not train on that video. Keep the
candidate construction fixed and evaluate the choice method on held-out
videos.

## Training-video score groups

Split the 32 current training videos into four fixed groups of eight. Each
group has four 25 fps and four 30 fps videos. The seven women's matches are
spread 2, 2, 2 and 1 across the groups.

For each group, train the chosen HGB on the other 24 current training videos
and score the held-out eight. The fixed validation videos train none of these
four models.

Player overlap between groups is accepted. The agreed safeguard is whole-video
separation. A player-separated split is not practical for the heavily repeated
25 fps player set, so report these as video-held-out scores.

## Training-video score result

Keep the combined score file. It contains 1,193,927 rows from all 32 training
videos. Each video was scored by the model trained on the other three groups.
The four group results and final combination passed their repeat and input
checks.

Before training the rally-start choice method, save the existing detected
sections and predicted player sides for these same videos. Build the unchanged
short candidate list from the held-out scores. Keep this preparation separate
from human labels and model fitting.

## Rally-start training input result

Keep the 32-video input file. It contains 2,621 section lists and 5,242 earlier
candidates. The candidate rule exactly reproduces the frozen validation list.

The existing player-side rule has no answer for 2,449 earlier candidates. The
next model check must therefore report both timing recovery and fully correct
rallies. A timing match with no player-side answer is not a fully correct
rally.

Write and review the candidate selection plan before joining human labels. Use
only these held-out first-model scores for training. Keep the model list small
and leave the eight validation videos for the fixed comparison.

## Rally-start contact selection plan

Use one fixed comparison with logistic regression and shallow histogram
gradient boosting. Each model has one fixed setting and selection cut-offs of
0.5, 0.7 and 0.9. Choose among the six results using held-out predictions from
the 32 training videos before opening validation labels.

Train a correct-addition answer only when the candidate matches the labelled
first-contact time and player side. A candidate with no predicted side cannot
be selected. Keep every baseline contact and add at most one earlier contact.
Do not reuse the pilot helper that can replace the fixed contact.

Use the existing A, B, C and D groups for held-out candidate-model scores.
Continue to validation only when the training result meets the fixed gain,
loss, correct-addition and recovery checks in
`rally_start_selection_plan.md`. The eight validation videos are one final
check of the fixed choice.

## Validation rally-start input result

Keep the completed label-free validation input. It reproduces all 615 fixed
candidate lists and all 1,845 entries. Its two full saves match byte for byte.

The existing player-side rule answers 629 of the 1,230 earlier candidates and
has no answer for 601. Keep every row for scoring, but do not let the model
select a candidate without a player-side answer.

## Rally-start contact model result

Do not add the rally-start candidate model. All six fixed choices failed the
training rules, so none went to validation.

Shallow HGB at 0.9 was the safest result. It gained 30 fully correct sections
without losing one, but only 76 of its 147 actions were correct. The fixed
minimum was 80%, while this result reached 51.7%.

Keep the original first contact model for the final fit. Do not tune another
candidate-model cut-off from these results. Continue with held-out first-model
predictions across all 40 development videos.

## Other accepted points

- Use 32 videos for training and eight for validation
- Train the chosen model again on all 40 eligible ShuttleSet videos
- Use only non-overlapping ShuttleSet22 videos for the final test
- Keep new work in `scratch/contact_det_full_ds_fit/`
- Make small local commits and keep machine access details out of Git

## ShuttleSet22 inpaint input

Run InpaintNet on the saved no-inpaint TrackNet coordinates for all 47 test
videos. Use the normal non-overlapping 16-frame path on GPU. Save the original
prepared files unchanged and put the new CSV, track, sidecar and guard files in
a writable sibling-file mirror.

The full original-video trial found only one-pixel GPU differences on frames
whose earlier inputs could be reconstructed exactly. The fabricated guard
count stayed unchanged. Four extra frames were rejected as degraded. The user
accepted this difference on 2026-08-28.

## ShuttleSet22 test-set accounting

Treat the official ShuttleSet22 annotations as a 58-match corpus. The final
contact test uses 47 of those matches because they have separate downloadable,
frame-aligned videos and do not overlap the base ShuttleSet development data.

Eight official matches overlap base ShuttleSet videos: ShuttleSet22 IDs 1–7
and 58. Three more official matches, IDs 14, 45 and 56, have unresolved
frame-aligned public video sources. These 11 matches still have annotations;
they are excluded because of overlap or video availability.

Keep the historical source manifest as the authority for the 47/8/3 split.
Authenticate the complete official annotation corpus with its pinned checksum
before scoring. After that check, read label tables only for the fixed 47 test
matches. This preserves the full-corpus provenance without bringing overlap or
unresolved matches into the result.

## Final contact fit result

Keep the reference raw-motion HGB model with balanced class weights and up to
24 negative rows per positive row. Held-out scores from all 40 development
videos keep the score cut-off at 0.9 and the nearby-contact distance at six
frames at 30 frames per second.

The final model is trained on all 40 development videos. Use this saved model
without changing its features or settings for the non-overlapping
ShuttleSet22 test.

## Final ShuttleSet22 result

Keep the fixed test result without tuning from it. At five frames, the final
detector reaches 80.62% precision, 84.37% recall and 82.45% F1 across the 47
non-overlapping ShuttleSet22 videos. Player side is 92.02% accurate where both
the label and detector give an answer.

Whole-rally accuracy is 16.60% at five frames and 18.09% at ten frames among
the 2,969 detected sections that map to exactly one labelled rally. The lower
whole-rally result is part of the final finding. Do not use it to change the
model, features, score cut-off or nearby-contact distance.

The path-free result is in `shuttleset22_test_summary.json`. The readable
account is in `shuttleset22_test_report.md`. Keep the raw detailed result and
cleaned labels outside Git.
