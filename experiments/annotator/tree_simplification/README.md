# Annotator tree simplification

This experiment compares smaller stacks against the existing eight-model
annotator. The first pilot compares HGB and GPU XGBoost with the existing
architecture. It uses the original 32 training videos and eight validation
videos. Fully correct rally recovery and player-attributed contact F1 are the
primary measures, using the existing benchmark scorer.

Install the `annotator-experiments` extra for XGBoost. HGB remains the default
training backend. The separate MonoTrack environment is not needed here.

Prepare contact features and heuristic inputs from the saved shuttle, pose and
court artefacts once:

```bash
python -m experiments.annotator.tree_simplification.prepare \
  --paths paths.toml --output RUN/prepared --workers 8
```

`paths.toml` has a `[paths]` table containing `shots_master`, `stages` and
`development_courts`, as in the earlier tree retune. Use `--only sset_01` for a
one-video preparation smoke test. This step runs no vision models.

Fit and evaluate the unchanged architecture with GPU XGBoost:

```bash
python -m experiments.annotator.tree_simplification.pilot \
  --paths paths.toml --prepared RUN/prepared --output RUN/xgb-pilot \
  --backend xgboost --device cuda --threads 4
```

For HGB use `--backend hist_gradient_boosting --device cpu`. A reference bundle
can be evaluated without refitting by passing `--reference-bundle`; add
`--expected-validation` to check every saved per-video benchmark count.

Each pilot writes progress, fit times, a model bundle, validation predictions
and pooled scores. Use a separate output directory for each configuration.
The backend comparison retains the existing cross-fitting procedure; its
internal contact-score reuse is discussed in the earlier tree-retune report.
Validation videos stay outside every fit.

The XGBoost pilot matches the objective, class weighting, leaf budget, learning
rate and boosting-round budget where possible. Its `min_child_weight` is a
separate setting from HGB's minimum samples per leaf. Contact early stopping
uses only an internal split of training rows. Predictions and stopping points
are not expected to match between libraries. Compare validation results and
fit times against the earlier HGB seed spread before choosing a backend.
Use `--min-child-weight 10` for a small regularisation sensitivity check against
the initial value of 5. Neither value is exactly equivalent to HGB's row minimum.
Trees train on the requested device and use CPU prediction for portable bundles
and the existing NumPy feature pipeline. The reported downstream time includes
feature construction and intermediate predictions; it is not pure GPU fit time.
The result records thread count and whether this invocation fitted a bundle.

## Smaller stacks

The simplified stack has one contact model followed by one, two or three rally
models. Every rally layer scores the same options. Later layers also see each
earlier layer's scores, selected option, score margin, rank and contact changes
relative to that selection. They can revise any earlier choice.

The pool retains the existing choices and adds combinations of up to three
contact additions or deletions. Proposals and features use detector evidence.
Targets measure whole-rally correctness with neighbouring spans at their
initial choices. Evaluation applies all selected changes together using the
production boundary and player-side rules. Training gives each judged span
the same total weight. That total is the average number of judged legacy options
per judged training span, calculated once from all 32 training videos. Every
pool arm, subset and layer uses this same value, so adding options does not
change the total training weight or the scale of weight-based regularisation.
HGB's minimum rows per leaf still counts options rather than weighted spans.

Build the pools once from a pilot's five fitted contact models:

```bash
python -m experiments.annotator.tree_simplification.pools \
  --prepared RUN/prepared --pilot RUN/xgb-pilot --output RUN/pools --workers 14
```

Use `--only sset_01` for a pool-building smoke test. Each video records target
counts by edit kind and the number of spans with a correct option in the old
and expanded pools. These span counts are diagnostics, not a recovery ceiling:
spans need not map uniquely to labelled rallies.

Run the small parallel search:

```bash
python -m experiments.annotator.tree_simplification.search \
  --pools RUN/pools --output RUN/search --backend xgboost --device cuda \
  --jobs 4 --threads 2
```

The search compares 7, 15 and 31 leaves and evaluates all three stack depths.
XGBoost also gets a stronger minimum-child-weight setting. One additional
configuration uses only the old options at all three depths, with the same
features and targets, to distinguish pool expansion from the effect of fewer models.
Use HGB with `--backend hist_gradient_boosting --device cpu`. The contact
models remain fixed to the selected pilot; changing the rally backend alone
does not change the contact backend.

Each configuration writes its own log, checkpoints, validation streams and
scores as soon as each stack depth finishes. Successful configurations survive
a restart; failed configurations resume their completed fits. The search exits
unsuccessfully if any configuration fails. `train --help` exposes individual
settings for a focused follow-up.

For a compact progress and results table, run:

```bash
python -m experiments.annotator.tree_simplification.status RUN
```

This reads only small JSON records, so it also works on an incremental local
copy that excludes the model and prepared-input files. Add `--tolerance 5` for
the stricter contact-matching window.

Training keeps the historical one-level contact cross-fitting procedure:
each training video's contact model excludes its group. Rally layers recursively
exclude the predicted group when generating earlier-layer training scores.
Contact features still have indirect paths between training groups' labels,
as in the previous stack. Internal fits are therefore not independent estimates
of generalisation. The eight validation videos are excluded from every fit and
provide the comparison scores. Initial results use one seed; close differences
need repeat fits before choosing a final configuration.
