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

## Specialised layer comparisons

The structural screen tests a physical serve-repair specialist, an insertion
specialist, and both, each feeding one then two whole-rally selectors. The specialist
labels are the existing local useful-edit targets: a contact repair can be
positive even when another error keeps the rally globally wrong. The selector
keeps the original whole-rally target, span weights and tree settings.

```bash
python -m experiments.annotator.tree_simplification.search \
  --pools RUN/pools --output RUN/structures --backend xgboost --device cuda \
  --structures --jobs 3 --threads 2
```

These arms use legacy options so they can be compared directly with the
existing legacy-pool general-layer control. Specialist scores map to the same
resulting sequence; duplicate edit routes use the largest finite score.
Absent edit types remain NaN. Each selector receives four extra features:
serve/insertion scores for the option and the best of each within its span.
Specialists exclude the predicted training group at every recursive rally fit.
The final specialists use all 32 training videos. No validation labels enter
specialist fitting or feature construction.

One specialist plus one selector gives three total models including contact;
both specialists plus one selector gives four. An extra correction selector
adds one model. To extend a promising arm, rerun `train` with the same settings
and output directory but a larger `--layers` value. Completed prefixes and
validation records are reused. Other setting changes require a new directory.

The 0.05 legacy switching margin and tighter contact timing are separate
questions from whether specialised supervision is useful. Compare both five-
and ten-frame validation tolerances before interpreting near-parity results.

Prepare the tighter target without rebuilding options or features:

```bash
python -m experiments.annotator.tree_simplification.targets \
  --pools RUN/pools --output RUN/targets-five --workers 4
python -m experiments.annotator.tree_simplification.train \
  --pools RUN/pools --output RUN/tightest-legacy --legacy-pool --layers 1 \
  --five-frame-targets RUN/targets-five --threads 2
```

Within the selected legacy options, a span uses five-frame labels if any judged
option is correct at that tolerance. Otherwise it retains ten-frame labels.
Original unknown labels remain excluded. This changes the target, with the
candidate rows, features and span-weight anchor held fixed.

Replay a fixed final-choice margin on an existing completed configuration:

```bash
python -m experiments.annotator.tree_simplification.margins \
  --run RUN/search/legacy-pool --output RUN/margins/legacy-pool --threads 2
```

This compares the final model's best option with its score for the initial
sequence. It evaluates margins zero and 0.05 without changing intermediate
layer histories or fitting models. Zero must reproduce every saved validation
count before either replay is reported. The fixed margin is a diagnostic, not
a calibration claim or a threshold tuned on validation.

The old eight-model reference includes a confidence-ranking model that does
not change the scored contact stream. These structural comparisons concern
rally recovery; production confidence ranking still needs assessment before
replacing the deployed stack.

## Conditional contact-fold comparison

After selecting a plausible downstream configuration, compare its inherited
contact OOF24 training inputs with eight fixed four-video contact folds. Each
new contact model fits 28 training videos. The downstream groups stay A–D;
this is a change to contact training inputs and the candidate pools they induce.

```bash
python -m experiments.annotator.tree_simplification.contact_folds \
  --prepared RUN/prepared --pilot RUN/xgb-5 \
  --original-pools RUN/pools-xgb-5 --output RUN/contact-oof28 \
  --threads 2 --workers 4
```

The eight fits reuse the pilot's contact settings and sampling-order convention.
They run sequentially before pool preparation. `--only sset_01` builds one pool
for a smoke test, then the same output can resume without `--only`. The command
rebuilds 32 training pools and copies all eight validation artefacts exactly.
Validation still uses the original final contact32 model. Saved fold assignments
and source settings must match when resuming.

Run the matched downstream configuration with the original arm's exact saved
`span_weight`. For the initial OOF24 screen this is `371.86228729719033`:

```bash
python -m experiments.annotator.tree_simplification.targets \
  --pools RUN/contact-oof28/pools --output RUN/contact-oof28/targets-five --workers 4
python -m experiments.annotator.tree_simplification.train \
  --pools RUN/contact-oof28/pools --output RUN/contact-oof28/selected \
  --five-frame-targets RUN/contact-oof28/targets-five --layers 1 --threads 2 \
  --span-weight 371.86228729719033
```

Copy the selected arm's other settings exactly. Regenerate tighter targets for
these new pools because their candidate rows can differ. Report changed initial
contacts, candidate counts and judged-span support with validation recovery/F1.
The inherited indirect training-label paths remain; internal predictions are
training features. The eight separate validation videos stay excluded from fits.
