# Binary contact detector options

## Recommendation

Use histogram gradient boosting as the reference and test BST-X next. Keep X3D-S for a later experiment if BST-X shows that pose and shuttle evidence are insufficient.

The tree trial is complete. Across three leave-one-fixture-out folds, HGB with physical features and validity masks reaches 84.5% precision, 90.5% recall and 87.4% F1 at ±10 base-30 frames. Random forest is worse and needs no further tuning.

BST-X is the most practical neural step because its pose, shuttle and court inputs already exist. X3D-S could add racket and body-motion evidence from RGB, but it first needs a frame-exact RGB and crop pipeline.

## The comparison

| Model | Evidence | New work | Current status |
| --- | --- | --- | --- |
| Histogram boosting | Hand-built shuttle, wrist, ankle and validity features | None for the baseline | Measured at 87.4% F1 |
| BST-X | Consecutive pose, shuttle and court sequences | Contact-centred windows, validity masks and a contact head | Recommended next pilot |
| X3D-S | Cropped RGB sequence | Decoder, court crops, alignment and temporal contact output | Hold until RGB is justified |

All three models should score the same label-blind region-v2 centres: every frame selected by at least one frozen seed channel. An eligible court-view interval has active court tracking and a clear definitive exclusion mask. Region v2 also adds a backwards 45-base-30-frame pre-roll outside each interval. Before model scoring, this surface contains a centre within ±10 of 2,790/2,836 non-serves (98.4%) and 286/292 serves (97.9%). `sset_21` remains the limiting fixture; the exact coverage and failure audit are in the [tree trial report](tree_contact_detector_results.md).

## Shared detector shape

1. Build short regions from relaxed shuttle impulse, wrist evidence, visibility changes, rally starts, scene starts and the serve pre-roll.
2. Score every possible centre frame inside the merged regions.
3. Select local peaks above a threshold, then apply temporal NMS.
4. Choose thresholds and suppression distance using only inner validation videos from the two training fixtures. Keep the outer test fixture untouched.

The regions and model inputs must remain label-blind during evaluation. Ground truth can label frozen centres and tune development settings. It cannot add a region that the runtime search missed.

## Option 1: histogram boosting

The tree converts each candidate centre into one numeric row. The measured physical row includes:

- shuttle visibility, speed, direction change and impulse strength
- distance and relative position between the shuttle and each wrist
- wrist and ankle motion at nearby frame offsets
- explicit shuttle, pose and player validity flags

Absolute image position, player size, interval progress, scene timing and proposal-source flags were excluded from the best model.

HGB is the useful tree. On the same version-2 rows, random forest reaches 84.6% F1. Adding context lowers HGB to 85.5% F1. Context-only and missingness-only controls are far weaker, so the headline result is not explained by broadcast timing alone.

The [tree trial report](tree_contact_detector_results.md) contains the exact per-fixture results, controls and boundary sensitivity. The remaining limitation is structural: a tree only sees the temporal differences and offsets supplied as features. It does not learn motion directly from the sequence.

## Option 2: BST-X contact detector

BST-X already accepts two-player pose, shuttle position and player court position over time. That makes it the cleanest test of whether learned temporal physical evidence can beat the tree.

The current BST-X is a whole-clip shot classifier. A contact detector needs:

- consecutive contact-centred source frames rather than resampled stroke clips
- explicit pose, shuttle and source-frame validity masks
- a temporal contact output or one contact score for the proposed centre
- a separate Top/Bottom side output that may abstain

Changing the present 14-class head to a binary head is not enough. The current loader also zero-fills missing pose and discards shuttle visibility. Those missing values must become explicit masks.

The model is small enough to score every centre in the bounded regions. The [BST-X contact detector plan](bst_x_contact_detector_plan.md) contains the checked parameter count, exact window and label design, ShuttleSet22 split, tests and acceptance rule.

## Option 3: X3D-S over cropped RGB

X3D-S adds evidence the numeric models cannot see: racket motion, body motion, shuttle blur and broadcast view context. A detected-court crop with a small buffer is a sensible first RGB view.

The forward pass is not the main risk. The repository still needs:

- frame-exact RGB decoding
- saved crop geometry and crop-validity masks
- alignment between RGB, pose, shuttle and labels
- handling for scene changes and missing court views
- a temporal contact output rather than a whole-clip class

The official PyTorchVideo X3D-S uses 13 sampled frames at a sample rate of 6. It is listed at 3.79 million parameters and 2.96 GFLOPs per view. Its standard head pools over space and time. A stride-1 detector is therefore a real fine-tuning change. See the [official model table](https://github.com/facebookresearch/pytorchvideo/blob/main/docs/source/model_zoo.md) and [official X3D code](https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/x3d.py).

A roughly 20-frame window of consecutive source frames at stride 1 remains a reasonable first test. The exact width should be chosen on held-out data.

## Labels and evaluation

Use one contact score and a separate Top/Bottom score. A low contact score means no contact. Weak or missing side evidence means the side head abstains.

Training needs quiet negatives, failed heuristic proposals, swings without contact, offsets around real contacts, and scene or visibility failures. Use positives within ±2 base-30 frames and ignore offsets 3 to 5. A visible positive takes priority over every ignore band. Sample a centre as negative only when it is at least 6 base-30 frames from every contact.

Use whole-video splits for this experiment. Neighbouring windows cannot be split randomly because they contain nearly identical frames.

Treat an offscreen or broadcast-omitted impact as an ignored interval with no contact or side loss. A model may infer a likely serve region from the lead-in, but it should not be trained to claim that the unseen impact is visible.

## Next step

Run one bounded BST-X pilot on the same frozen region, leave-one-fixture-out folds, event matcher and temporal NMS as HGB. Require pooled F1 of at least 89.9% and precision of at least 87.5% at ±10 before expanding the experiment.

Contact event quality decides whether the pilot expands. Report side accuracy and answer coverage separately; the side head is auxiliary because direct geometry already reaches about 89% on matched contacts.

If BST-X misses the contact thresholds, stop after the pilot and inspect the errors. Build X3D-S only when the remaining in-region failures cluster around visible racket/body evidence or view types that the numeric inputs cannot represent. Handle uncovered off-court contacts as a separate search problem.
