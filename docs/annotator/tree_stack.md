# How the trees work together

The selected annotator uses three XGBoost classifiers: a contact model, a rally
selector and a review model. They answer different questions. The contact model
scores individual frames; the selector compares complete contact sequences; the
review model ranks the chosen rallies for inspection.

The original eight-model architecture is retired. Its results remain in the
[development history](../../experiments/annotator/development.md). The current
baseline is documented in the [rally-model report](../../experiments/annotator/reports/tree_simplification.md).

## The selected model

These settings describe the saved `raw_baseline-seed-0` bundle. The five benchmark
runs add offsets 0, 100, 200, 300 and 400 to the base seeds.

| Model | Question | Input values | Maximum rounds | Maximum leaves | Learning rate | Base seed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Contact | Is this frame a racket contact? | 85 | 180 | 31 | 0.06 | 20260824 |
| Rally selector | Does this proposed contact sequence match a complete rally? | 478 | 400 | 31 | 0.05 | 20260905 |
| Review | How likely is the chosen rally to be correct? | 27 | 100 | 7 | 0.05 | 20260905 |

All three use minimum child weight 5 and L2 regularisation 1. The selected bundle
has one rally selector. Additional selector layers remain available for research;
each receives eight extra values from every preceding layer.

## Contact model

The contact features describe shuttle motion and tracked players at five time
offsets. The model scores candidates within the search regions created by the
heuristics. High-scoring candidates compete with nearby frames, then the 0.9
cutoff determines the initial contact stream.

The scored candidates remain available after that cutoff. A lower-scoring frame
may still be selected when the rally model compares whole sequences.

The optional guarded-candidate rule removes unreliable shuttle frames without
nearby player evidence before scoring. It is off in the selected bundle. See
[Fixed heuristics](heuristics.md#optional-rule-for-guarded-candidates).

## Rally selector

For each rough rally, the proposal builder keeps the initial contacts and
constructs alternatives from shortlisted earlier serves and later contacts.
It includes serve repairs, deletions and combinations of additions/deletions.
Temporal separation and neighbouring-rally rules restrict those alternatives.

Each row contains:

- summaries of the original and proposed contacts;
- agreement between their individual court-half guesses;
- counts and probabilities of added and deleted contacts;
- physical features of the original first contact, new first contact,
  strongest addition and weakest deletion;
- evidence from unused candidates and the gaps between contacts.

The highest score wins within each initial rally. Equal scores preserve the
proposal order. The finishing rules then adjust boundaries without changing
contact membership and choose the better-fitting alternating player pattern.

With additional layers, each layer receives the preceding scores, winning score,
margin, rank, choice and contact differences. All layers score the same ordered
proposals. They do not rebuild proposals or repeatedly edit the previous winner.

## Review model

The review model sees the selected proposal's evidence, selector score and margin,
plus a summary using the final boundaries. It measures side agreement from the
original individual guesses, before enforced alternation.

Its score is saved as `rally_confidence`. It does not change annotations or apply
an acceptance threshold. If a bundle has no fitted review model, the selected
proposal's final selector score supplies the ranking instead.

## What the training labels mean

Contact training marks frames close to a human-labelled hit as positive. It
omits an uncertain band around those positives, keeps nearby negatives and
samples more distant negatives. Sampling order and fit-row order are recorded
separately because both can affect reproducibility.

Rally training judges each proposal after applying the boundary rules with its
initial neighbours. A proposal is wrong if it merges rallies, misses or adds
contacts, or contradicts known player sides. Missing labels can make a proposal
unjudgeable; these rows are excluded from fitting.

The selector prefers labels matched within five frames at 30 FPS when a rally
has a positive option at that tolerance. Otherwise it uses the ten-frame labels.
Both tolerances scale with frame rate. Each judged rally receives equal total
row weight, then a fixed scale based on the rule-based repair set is applied.
Expanding the proposal set therefore does not silently change that scale.

Review labels describe the final selected rally at ten-frame tolerance. Review
training excludes unjudgeable rallies and uses held-group predictions.

## Groups and validation

Training videos are grouped so that downstream features can be computed from
models fitted without the video's own group. The contact stage fits one model
per held-out group, plus the final model using all training videos. Rally layers
exclude the relevant groups when producing inputs for later layers and review.

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
preprocessing, contact-selection settings and side geometry. The metadata records
the runtime versions, ordered feature names, proposal budgets and review mode.
Changing feature order or proposal semantics requires a new fit and evaluation.

The `RallyLayerModels` class name and module path are retained so current saved
XGBoost bundles load. The schema is `annotator-rally-layers/2`.

See the [refit guide](retuning.md) for commands, configuration and saved results.
