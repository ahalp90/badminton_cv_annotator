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

## Other accepted points

- Use 32 videos for training and eight for validation
- Train the chosen model again on all 40 eligible ShuttleSet videos
- Use only non-overlapping ShuttleSet22 videos for the final test
- Keep new work in `scratch/contact_det_full_ds_fit/`
- Make small local commits and keep machine access details out of Git
