# How the three-video code becomes a 40-video experiment

## Main finding

The pilot already has reusable feature calculations and rally-scoring functions. The new work mainly replaces the parts that assume there are exactly three videos.

The pilot files stay unchanged. New code in this directory supplies the 40-video list and the fixed training and validation split.

## What the pilot does now

```text
three fixed videos
    -> calculate and save candidate-frame features
    -> check the saved feature file
    -> load 292 rally labels and 3,128 contact labels
    -> train on two videos and test on the third
    -> choose a score cut-off from the two training videos
    -> save HGB and RF contact predictions
    -> replay the Top/Bottom player rule at predicted frames
    -> assign contacts to rally ranges
    -> score complete rallies
```

## What the larger experiment will do

```text
40 listed videos with a fixed 32/8 split
    -> calculate and save the same features for each video
    -> check every saved row before loading labels
    -> train HGB and RF on 32 training videos
    -> choose the model and settings on eight validation videos
    -> save every validation score and whether each candidate frame is kept as a contact
    -> replay the Top/Bottom player rule at fixed contact frames
    -> score complete validation rallies
    -> make out-of-fold predictions across all 40 videos
    -> choose the final score cut-off and duplicate-removal distance
    -> train the chosen model on all 40 videos
    -> test once on non-overlapping ShuttleSet22 videos
```

An out-of-fold prediction is made by a model that did not train on the video being predicted.

## Code that can be reused

### Candidate-frame features

`scratch/contact_det/scripts/freeze_tree_contact_features.py` already has a function named `_fixture_rows` that processes one supplied video. Reuse this function and its feature names, NumPy row format and compressed-file writer.

`scratch/contact_det/scripts/freeze_contact_evidence.py` already knows how to load one video's shuttle, pose, court and annotation files. It also records rally ranges that include the start frame and stop before the end frame.

All 40 eligible ShuttleSet videos are 1920 by 1080. The new video list still records width and height so the assumption is checked.

### Contact model and timing score

The pilot code already has reusable functions for:

- choosing the model input columns
- building a NumPy input table
- measuring distance to the nearest labelled contact
- creating the reference HGB and RF models
- removing nearby duplicate predictions
- matching predictions to labels one-to-one
- writing repeatable compressed results

New code is needed to check a 40-video feature file, load labels for named videos, build training examples from training videos only, and train with one fixed 32/8 split.

### Complete rally score

`score_contact_rallies.py` already has reusable data records and functions for:

- reading retained contact events
- assigning contacts to rally ranges that include the start frame and stop before the end frame
- finding events outside every rally range
- checking one predicted rally against one labelled rally
- showing how accuracy changes as more rallies are kept

New code is needed to load timing and player-side labels for the larger video list. It must also replay the existing Top/Bottom rule without relying on the old three-video list.

## Three-video assumptions that must be replaced

- The video list contains only `sset_01`, `sset_15` and `sset_21`.
- Saved feature checks require exactly those three names and frame rates.
- Label loading requires exactly 292 rallies and 3,128 contacts.
- Each video becomes a test video while the other two become training videos.
- Saved results require one trained model for each of the three videos.
- The Top/Bottom check compares two saved versions of the pilot features for the same three inputs.
- Complete rally scoring loads rally ranges and labels only from the three-video comparison files.

## Order that keeps labels out of feature preparation

The saved annotation ranges and possible contact-frame inputs are predictions from the existing video code. They are not ShuttleSet contact labels.

The new code must keep this order:

1. Check the video list and split.
2. Check every feature row and input-file identity.
3. Set which videos will train the model and which videos it will predict.
4. Load contact timing labels for those named videos.
5. Set the predicted contact frames and Top/Bottom answers.
6. Load player-side labels and score the rallies.

Validation labels may choose the model and its settings. ShuttleSet22 labels only score the finished setup.

Any later trained model must use predictions from the first contact model. Each training video must be predicted by a first contact model that did not train on that video.

## Mistakes the tests must catch

- repeated video IDs or names
- a video placed in more than one split
- missing or unexpected videos in the feature file
- different frame rates in the video list, saved file and feature rows
- feature rows outside ranges that include the start frame and stop before the end frame
- contact labels loading before feature checks finish
- negative examples drawn from validation videos
- score cut-offs chosen from training scores instead of validation scores
- saved scores missing the video, range or frame number they belong to
- Top/Bottom replay missing a retained contact
- training on all 40 videos changing a setting that was already fixed
- a later model reading predictions from a first contact model that trained on the same video
- ShuttleSet22 labels being used to choose settings

## Independent code checks

One read-only Luna pass followed the pilot feature and contact-model code. A second pass followed Top/Bottom scoring, whole-rally scoring and failure reports. The main agent checked the important split, label order and Top/Bottom findings directly in the named functions.
