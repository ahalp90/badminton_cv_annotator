# Binary contact detector options

## Short answer

Start with a boosted tree over the evidence we already compute. It is the fastest way to find out whether the current rules contain enough information to place contacts reliably.

If the tree reaches a ceiling, try BST-X next. BST-X already uses pose, shuttle and player position. It is a better fit than X3D-S for a first learned contact detector because its inputs and most of its data path already exist.

Keep X3D-S as the next source of genuinely new evidence. RGB can see the racket, body motion and broadcast context that the numeric inputs omit. It also needs a new RGB data path and careful crop handling, so it is the easiest option to lose time on before we know whether RGB is needed.

All three models should solve the same small problem. Simple rules make short search regions. The learned model scores each possible centre frame inside those regions. Nearby hits are reduced to the frame with the strongest contact score.

The regions cannot come only from today's raw shuttle-impulse proposals. At the 10-frame tolerance, those proposals cover 83.8% of later contacts and 66.1% of serves. That is the highest recall any scorer could reach if it never looks elsewhere.

## The shared search method

There is little reason to slide a learned model across every frame of a match. The current shuttle impulse and player checks already provide useful priors about where a contact may be.

A practical search is:

1. Keep every plausible shuttle-impulse proposal, including proposals that the final rule currently suppresses.
2. Add other cheap region seeds: wrist-motion peaks, relaxed impulse peaks, shuttle loss or return, and suspected serves near a scene or rally start.
3. Make a short search region around each seed and merge overlaps.
4. Score a tightly centred window at each frame in that region.
5. Find local score peaks.
6. When nearby peaks appear to be the same contact, keep the strongest one.

Use temporal NMS over a tree probability or neural-network logit. Set each model's threshold on held-out videos before comparing them.

The deduplication distance needs to be tuned on held-out rallies. It should not simply equal the whole input-window width. A wide input window can contain useful lead-in and follow-through while two real contacts may still occur closer together than that width.

The proposal rules must remain label-blind at test time. Ground truth can set search widths and thresholds on development videos, but it cannot add a missed region during evaluation.

## Option 1: boosted tree or random forest

A tree model can score a candidate centre frame from a fixed row of numeric features. The row can include the centre frame, several nearby offsets, and short-window summaries.

Useful inputs already exist:

- shuttle visibility, position, speed, direction change and impulse strength;
- the local impulse floor and the ratio above that floor;
- shuttle distance to each wrist, scaled by body height;
- pose visibility and missing-data flags;
- wrist, ankle and player-box positions;
- distance from each player to the net band and court edges;
- the current wrist and proximity rule results;
- whether the current proposal was suppressed, kept only as an optional record of the old rule rather than a core input;
- distance from a scene cut, rally start, shuttle gap or tracking reset;
- values from a few frames before and after the proposed centre.

Missing-data flags matter. A missing shuttle or wrist is not the same as a measured coordinate of zero.

### Random forest or boosted tree?

Use histogram gradient boosting as the main tree baseline. Boosting is usually better at combining many weak, related clues such as impulse strength, wrist distance and visibility. A random forest is still worth running as a check because it is forgiving and needs little tuning.

The current environment has scikit-learn 1.8 with both models. It does not have XGBoost, LightGBM or CatBoost. There is no need to add one of those packages for this test.

The tree baseline has three strong advantages:

- it uses data the pipeline already computes;
- training and inference are cheap;
- feature importance and reruns with feature groups removed can show whether the gain came from shuttle motion, player evidence, scene context or an accidental shortcut.

Its main limit is also clear. Trees do not learn motion directly from a sequence. We must give them useful temporal differences, offset samples and window summaries. They may also learn a particular camera layout unless the split and coordinate normalisation are careful.

## Option 2: BST-X with a contact output

BST-X is the most practical neural model for the existing evidence. It already accepts two-player pose, shuttle position and player court position over time.

The present BST-X is a clip classifier. It reduces the whole clip to three summary vectors and emits one shot-class result. A contact detector therefore needs more than changing the last layer from 14 classes to two. It needs:

- short contact-centred training windows;
- a time-aligned contact output, or one score for each proposed centre;
- contact-frame labels and an ignore flag for invisible or uncertain contacts;
- explicit pose and shuttle visibility inputs;
- a training loss for contact rather than stroke type.

The current data loader uniformly resamples a whole stroke clip when the clip is too long. Setting its sequence length to 20 would not make a 20-frame contact window. The window builder must select consecutive source frames around a candidate centre.

The model is small. With the repository's current 17-joint plus bone input and 14-class head, it has about 1.84 million parameters at 20 or 39 frames. A binary contact version would be similar. The forward pass is cheap enough that sliding over a short proposal region is reasonable.

BST-X's largest data problem is missing evidence. The current prepared arrays zero-fill failed pose frames. They also discard shuttle visibility, so a missing shuttle becomes `(0, 0)`. A contact model must keep the visibility flags instead of asking the network to guess what zero means.

## Option 3: X3D-S over cropped RGB

X3D-S adds information that the other two models cannot see: racket motion, body motion, a shuttle blur that TrackNet missed, and broadcast context. For this contact detector, a court crop with a small buffer is a sensible first RGB view. This is separate from the repository's planned wrist-crop X3D-S stroke classifier.

The cost is mostly in the data path, not the forward pass. The repository does not currently have an X3D implementation or a contact-centred RGB dataset. It would need:

- frame-exact RGB decoding;
- a saved court crop and crop-validity record;
- contact-centred windows and labels;
- handling for scene changes and missing court views;
- a temporal contact output instead of the standard clip-classification output;
- tests for frame alignment between RGB, shuttle and pose.

The official PyTorchVideo X3D-S model uses 13 sampled frames at a sample rate of 6. It is listed at 3.79 million parameters and 2.96 GFLOPs per view. Its standard head pools over space and time to produce one clip result. A stride-1 contact detector is therefore a real fine-tuning change, not just a different call to the stock model. See the [official model table](https://github.com/facebookresearch/pytorchvideo/blob/main/docs/source/model_zoo.md) and [official X3D code](https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/x3d.py).

A roughly 20-frame stride-1 window is a reasonable experiment, but the exact width should be treated as a measured choice. The window must be tight enough that the label refers to one contact. It must still show enough lead-in to recognise a serve or a partly hidden hit.

## Labels and negative examples

Use one contact score and a separate Top/Bottom score. “No contact” should come from the contact threshold, not from forcing Top, Bottom and None into one three-way choice. The side score can abstain when player evidence is missing or weak.

The training set needs several kinds of negatives:

- easy quiet frames;
- raw impulse proposals rejected by the current player checks;
- suppressed neighbouring proposals;
- racket swings and player motion without contact;
- offsets before and after a real contact;
- scene-cut and visibility failures that look tempting to the rules.

Offsets near a labelled contact need an ignore band. The source label may be a frame or two late, and a model should not be punished for placing its peak slightly closer to the visible impact.

A sensible first labelling rule is positive within about two frames of a reviewed contact, then an ignored band before clear negatives begin. Offset negatives must also be checked against every other contact in the rally.

Split by whole video, or at least by whole match section. Randomly splitting neighbouring windows would put almost identical frames in training and validation and give a misleading result.

For an offscreen or broadcast-omitted serve, use an unknown label when the exact frame cannot be supported by the video. A scene-cut lookback rule may still propose a likely serve region. No model should be trained to claim that it saw an invisible impact.

## Smallest useful experiment

1. Freeze candidate regions and per-frame features without using GT.
2. Measure how many GT contacts fall inside those regions. This is the ceiling for every learned scorer in the test.
3. Add GT afterwards to make centre-frame labels, ignore bands and grouped train/test splits.
4. Train scikit-learn histogram gradient boosting and a random forest on the same rows.
5. Score contact recall at 5, 10 and 15 base-30 frames, plus extra predictions per real contact.
6. Inspect failures and rerun without each main feature group.
7. Build the BST-X contact-window path only if the tree leaves useful misses that temporal learning could plausibly fix.
8. Build X3D-S when those remaining misses call for RGB evidence.

This order gives a useful answer after the first small experiment. It also leaves one clean reason to try each larger model.

## Recommendation

For the next detector experiment, use histogram gradient boosting inside heuristic search regions. Run a random forest beside it as a cheap check.

Between the two neural choices, use BST-X first. It fits the data already present and needs less new setup. Use X3D-S when the measured failures show that pose and shuttle evidence are insufficient, especially for unusual serves, racket-only cues and difficult broadcast views.
