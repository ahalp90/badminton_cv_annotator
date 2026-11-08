# How the trees work together

The selected annotator uses three XGBoost classifiers: a contact model, a rally
selector and a review model. They answer different questions. The contact model
scores individual frames; the selector compares complete contact sequences; the
review model ranks the chosen rallies for inspection.

The original eight-model architecture is retired. Its results remain in the
[development history](../../experiments/annotator/development.md). The current
baseline is documented in the [selected fit settings](../../experiments/annotator/tree_simplification/results/hp_search/fit-settings.json.gz).

## The selected model

These settings describe the [selected seed-0 bundle](../../experiments/annotator/tree_simplification/results/hp_search/models/seed-0/).
The baseline has no maximum outward context limit in training or discovery.
The five benchmark runs add offsets 0, 100, 200, 300 and 400 to the base seeds.

| Model | Question | Input values | Maximum rounds | Maximum leaves | Learning rate | Base seed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Contact | Is this frame a racket contact? | 85 | 360 | 63 | 0.06 | 20260824 |
| Rally selector | Does this proposed contact sequence match a complete rally? | 478 | 400 | 63 | 0.05 | 20260905 |
| Review | How likely is the chosen rally to be correct? | 27 | 100 | 7 | 0.05 | 20260905 |

All three use minimum child weight 5 and L2 regularisation 1.

## Contact model

The contact features describe shuttle motion and tracked players at five time
offsets. The model scores candidates within the search regions created by the
heuristics. High-scoring candidates compete with nearby frames, then the 0.9
cutoff determines the initial contact stream.

The scored candidates remain available after that cutoff. A lower-scoring frame
may still be selected when the rally model compares whole sequences.

## Rally selector

For each rough rally, keep the initial contact sequence as the first option.
Generate alternatives by adding shortlisted earlier or later contacts and
deleting from the first contact plus the five weakest remaining contacts. The
default budget allows up to three additions/deletions combined. Temporal
separation and neighbouring-rally bounds restrict those combinations.

Each row contains:

- summaries of the original and proposed contacts;
- agreement between their individual court-half guesses;
- counts and probabilities of added and deleted contacts;
- physical features of the original first contact, new first contact,
  strongest addition and weakest deletion;
- evidence from unused candidates and the gaps between contacts.

The highest score wins within each initial rally. Equal scores preserve the
proposal order. The chosen window keeps its outward context. Finishing can add
10 frames of contact margin at 30 FPS, subject to neighbour, video-edge and
contact-membership checks. The chosen contacts are final inside each chosen
span; detector contacts outside all chosen spans remain in the full stream.
The finishing step also fits the hitter order from the retained contacts.
Supplied rally bounds stay fixed.

## Review model

The review model sees the selected proposal's evidence, selector score and margin,
plus a summary using the final boundaries. It measures side agreement from the
original individual guesses, before enforced alternation.

Its score is saved as `rally_confidence`. It does not change annotations or apply
an acceptance threshold. If a bundle has no fitted review model, the selected
proposal's final selector score supplies the ranking instead.

## What the training labels mean

By default, contact training labels frames inside broad search regions: within
one base-30 frame of a labelled contact is positive, distances of two to four frames are
ignored, and five or more frames are negative candidates. It keeps nearby
negatives and samples more distant ones. See the
[worked example and measured contact coverage](how_it_works.md#how-contact-training-uses-ground-truth).
Sampling order and fit-row order are recorded separately because both can
affect reproducibility.

Rally training judges each proposal directly. Boundary padding happens after
selection during inference. A proposal is wrong if it merges rallies, misses or adds
contacts, contradicts known player sides, or cuts into a labelled rally beyond
the contact-timing allowance. Outward context is unlimited. Training judges the
proposal's window; benchmark discovery assessment uses its first and last
accepted contacts.
Player correctness is judged after fitting the alternating order, including the
wrist-distance tie-break. The selector still receives the original player guesses.
Thus a target describes whether the eventual output would be correct, rather than
whether every raw guess already matches ground truth. Missing labels can make a
proposal unjudgeable; these rows are excluded from fitting.

By default, the selector prefers labels matched within five frames at 30 FPS
when a rally has a positive option at that tolerance. Otherwise it uses the ten-frame labels.
Both tolerances scale with frame rate. Each judged rally receives total training
weight 372 by default, divided equally among its judged options. Set
`RallyFitConfig.span_weight` to retune that scale independently of proposal count.

Review labels describe the final selected rally at the default ten-frame
tolerance. Review training excludes unjudgeable rallies and uses held-group
predictions. Contact radii and selector/review tolerances can be changed through
the fit settings; frame counts scale from 30 fps to each video.

## Groups and validation

Training videos are grouped so that downstream features can be computed from
models fitted without the video's own group. The contact stage fits one model
per held-out group, plus the final model using all training videos. The selector
is fitted once per held-out group to provide predictions for review training.

This is the existing training procedure. Contact scores retain indirect paths
between groups through other groups' features. Separate validation videos remain
the measure of generalisation; held-group training predictions do not replace
validation.

The benchmark uses 32 development-training videos and eight validation videos.
The five runs are independent seeded fits with the same split, not five-fold
cross-validation. The supplemental 40-video comparison refits the contact model
only; it retains the development-fitted selector and review model.

## Saved contract

[`models.py`](../../src/annotator/models.py) saves the fitted classifiers with
preprocessing and contact-selection settings. The metadata records
the runtime versions, ordered feature names and proposal budgets. The fitted
review model is stored in `models.joblib`.
Changing feature order or proposal semantics requires a new fit and evaluation.

The schema for newly fitted bundles is `annotator-three-models/5`. Earlier
bundles use the source and environment they were fitted with.

See the [refit guide](retuning.md) for commands, configuration and saved results.
