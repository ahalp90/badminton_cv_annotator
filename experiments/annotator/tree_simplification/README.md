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
