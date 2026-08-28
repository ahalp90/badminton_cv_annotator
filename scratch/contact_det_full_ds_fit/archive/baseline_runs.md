# The first contact-model comparison

## Bottom line

The first comparison contains nine full model runs. The limit is 12. This list is fixed before any of the eight validation videos are scored.

Histogram gradient boosting is shortened to HGB below. Random forest is shortened to RF.

## What stays the same

Every run will:

- train on the same 32 videos;
- score the same eight validation videos;
- use motion and player-distance inputs, plus flags for missing values;
- use random seed `20260824`;
- treat frames within one frame of a labelled contact at 30 frames per second as positive examples;
- leave frames within four frames of a contact out of the negative examples;
- keep all harder negative examples within 15 frames of a contact;
- choose among score cut-offs from 0.05 to 0.95 in steps of 0.05;
- choose among duplicate distances of 4, 5 and 6 frames at 30 frames per second;
- save the score for every possible contact frame.

The score cut-off and duplicate distance are chosen by contact timing on the eight validation videos. The order for a tie is timing F1, recall, precision, larger duplicate distance, then higher score cut-off.

F1 is one number that balances precision and recall. Precision measures how many predicted contacts were right. Recall measures how many labelled contacts were found.

The contact timing result does not choose the final model by itself. The whole-rally result is checked before that choice is made.

## The nine runs

| Run | Motion values | Model | Class weighting | Negative examples |
| --- | --- | --- | --- | --- |
| 1 | Original per-frame values | Reference HGB | Balanced | Up to 12 per positive example |
| 2 | Scaled to 30 frames per second | Reference HGB | Balanced | Up to 12 per positive example |
| 3 | Original per-frame values | Reference RF | Balanced within each tree | Up to 12 per positive example |
| 4 | Scaled to 30 frames per second | Reference RF | Balanced within each tree | Up to 12 per positive example |
| 5 | Original per-frame values | Reference HGB | None | Up to 12 per positive example |
| 6 | Original per-frame values | Reference RF | None | Up to 12 per positive example |
| 7 | Original per-frame values | HGB with 15 leaf nodes instead of 31 | Balanced | Up to 12 per positive example |
| 8 | Original per-frame values | HGB with a 0.04 learning rate and 270 rounds | Balanced | Up to 12 per positive example |
| 9 | Original per-frame values | Reference HGB | Balanced | Up to 24 per positive example |

The two changed HGB settings are small changes to the pilot model. Run 7 makes each tree smaller. Run 8 takes smaller learning steps for more rounds while keeping the other settings fixed.

The second rule for negative examples changes only the maximum count. It still keeps every harder negative example.

## Why this is enough

Runs 1 to 4 compare the two models and the two motion choices. Runs 5 and 6 check whether class weighting helps. Runs 7 and 8 make the two allowed small HGB changes. Run 9 checks the one allowed change to the number of negative examples.

This is not every possible combination. Adding more runs after reading the validation result needs a written reason and the user's approval.
