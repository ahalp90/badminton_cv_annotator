# Annotator tree simplification

This experiment cut the annotator's fitted models from eight to three. The new
stack has a contact model, one rally selector and a review ranker. The rally
selector is an XGBoost model with 31 leaves and 400 boosting rounds. The review
ranker has 7 leaves and 100 rounds and only orders the output for manual
checking. The [report](../reports/tree_simplification.md) gives the results and
the reasons behind each choice. This guide says what the repository keeps and
how to run each step again.

All models fit on 32 ShuttleSet training videos. Eight other videos serve as
validation: 668 labelled rallies and 5,696 contacts. Validation chose every
setting. The 46-video ShuttleSet22 test set was used only after those choices
were made.

The two measures come from the existing benchmark scorer:

- **Correct-player F1** counts a predicted contact only when it matches a
  labelled contact within ±10 frames (or ±5) and names the right player.
- **A fully correct rally** has every labelled contact matched with the right
  player and no extra contacts.

Terms used in the commands and code:

- `RUN` is a run folder you choose. Use a fresh one for each experiment.
- A **span** is one proposed rally section. Its **option list** (a "pool" in
  the code) holds the candidate contact sequences a selector chooses from.
- A **layer** is one rally selector. **Depth** is how many run in sequence.
  Saved bundles use these names, so the code keeps them.
- The **contact run** is the folder whose fitted contact models feed the
  option lists.
- A **seed offset** (0, 100, 200, 300 or 400) is added to every base seed. Fits
  at different offsets share their videos, so they show fitting variation, not
  independent samples.

The base install includes XGBoost. The separate MonoTrack environment is not
needed here. Run every command from the repository root.

## Tracked models and prediction files

Everything behind the report's numbers sits in [results/](results/). The
[results index](results/README.md) lists every file with its checksum, the
label defaults, each preprocessing configuration's exact setting and what each
archived run folder held.

- `results/*.json.gz` are the evidence files. Each one backs a table or claim in
  the report.
- `results/models/<label>/` holds 12 saved bundles (`models.joblib` and
  `metadata.json`). `selected/` is the chosen model. `raw_baseline-seed-<N>/`
  and `body_window_36-seed-<N>/` are the ten test-run bundles.
  `old-eight-model-stack-xgboost-seed-0/` is the old stack refitted with
  XGBoost, kept for the timing comparison. `annotator.models.load_models`
  loads any of them.
- `results/streams/<table>/<label>/seed-<N>/<video>.json.gz` are the per-video
  prediction files. `selection/` covers the selector settings and the old
  stack, `fps/` the frame-rate scaling variants, `input-validation/` the
  preprocessing sweep and `input-test/` the test run. `input-test/` also holds
  per-video test scores under `scores/`.
- `results/test-run/` holds the records the test runner needs to load the ten
  test bundles again. Run paths in them start with the placeholder `SWEEP/`.
- The review rows and the review thresholds chosen on validation are
  `results/input-validation-review-rows.json.gz`,
  `results/input-test-review-rows.json.gz` and
  `results/input-review-thresholds.json.gz`. `results/input-stage-costs.csv`
  gives the time and memory of every sweep and test command.

Some steps need data the repository does not hold. The saved records above are
enough to load and check every bundle, re-score every prediction file and redo
the threshold analysis. Running test inference again also needs the 46 test
videos' prepared inputs (see
[Test run and review thresholds](#test-run-and-review-thresholds)). Fitting
anything again needs the prepared training and validation inputs that
`prepare` builds.

## Analysis scripts

The [analysis folder](analysis/README.md) rebuilds and checks the evidence
files and figures. Most scripts read tracked files only, recompute the reported
numbers and compare them with the committed evidence. The analysis README lists
each script's inputs and output. Four need data from outside the repository:
`input_selection.py rows` and `screening_evidence.py` read the archived raw run
records, `profile_selected.py` needs prepared arrays, and `pool_coverage.py`
needs option lists from a fresh run.

`analysis/test_run.py` rebuilds a working test-run folder from
`results/test-run/` and `results/models/`. It then loads all ten test bundles
through the test runner's own checks. The commands are under
[Test run and review thresholds](#test-run-and-review-thresholds).

## Prepared inputs and the backend pilot

The first step compared HGB and GPU XGBoost with the old eight-model
architecture unchanged. Later steps build on that pilot's contact models.

Prepare contact features and heuristic inputs from the saved shuttle, pose and
court outputs once:

```bash
python -m experiments.annotator.tree_simplification.prepare \
  --paths paths.toml --output RUN/prepared --workers 8
```

`paths.toml` has a `[paths]` table containing `shots_master`, `stages` and
`development_courts`, as in the earlier tree retune. Use `--only sset_01` for a
one-video smoke test. This step runs no vision models.

Fit and evaluate the unchanged architecture with GPU XGBoost:

```bash
python -m experiments.annotator.tree_simplification.pilot \
  --paths paths.toml --prepared RUN/prepared --output RUN/xgb-pilot \
  --backend xgboost --device cuda --threads 4
```

For HGB use `--backend hist_gradient_boosting --device cpu`. Pass
`--reference-bundle` to evaluate a saved bundle without refitting. Add
`--expected-validation` to check every saved per-video benchmark count.

Each pilot writes progress, fit times, a model bundle, validation predictions
and pooled scores. Use a separate output folder for each configuration. The
pilot keeps the existing cross-fitting procedure. The earlier tree-retune
report discusses how it reuses contact scores internally. Validation videos
stay outside every fit.

The XGBoost pilot matches the objective, class weighting, leaf budget, learning
rate and boosting-round budget where possible. Its `min_child_weight` is a
separate setting from HGB's minimum samples per leaf. Use
`--min-child-weight 10` for a small sensitivity check against the default of 5.
Neither value is exactly equivalent to HGB's row minimum. Contact early
stopping uses only an internal split of training rows. Predictions and stopping
points are not expected to match between libraries. Trees train on the
requested device and predict on CPU, so bundles stay portable and the existing
NumPy feature pipeline is unchanged. The reported downstream time includes
feature construction and intermediate predictions, so it is not pure GPU fit
time. The result records the thread count and whether this run fitted a bundle.

The archived evidence calls the XGBoost pilot folder `RUN/xgb-5` (child weight
5) and its option lists `RUN/pools-xgb-5`. The commands here call them
`RUN/xgb-pilot` and `RUN/pools`.

## Smaller stacks

The simplified stack has one contact model followed by one, two or three rally
selectors. Every selector scores the same options. Later selectors also see
each earlier selector's scores, chosen option, score margin, rank and contact
changes relative to that choice. They can revise any earlier choice.

The option list keeps the old stack's original options and adds combinations
of up to three contact additions or deletions. Proposals and features use
detector evidence. Targets measure whether the whole rally is correct, with
neighbouring spans left at their initial choices. Evaluation applies all chosen
changes together, using the production boundary and player-side rules.

Training gives each judged span the same total weight. That total is the
average number of judged original options per judged training span, calculated
once from all 32 training videos. Every option list, subset and selector count
uses this same value. Adding options therefore does not change the total
training weight or the strength of weight-based regularisation. HGB's minimum
rows per leaf still counts options rather than weighted spans.

Build the option lists once from the pilot's five fitted contact models:

```bash
python -m experiments.annotator.tree_simplification.pools \
  --prepared RUN/prepared --contact-run RUN/xgb-pilot --output RUN/pools --workers 14
```

Use `--only sset_01` for a smoke test. Each video records target counts by edit
kind, and the number of spans with a correct option in the original and full
option lists. These span counts are diagnostics, not a ceiling on fully correct
rallies, because spans need not map one-to-one to labelled rallies.

Run the small parallel search:

```bash
python -m experiments.annotator.tree_simplification.search \
  --pools RUN/pools --output RUN/search --backend xgboost --device cuda \
  --jobs 4 --threads 2
```

The search compares 7, 15 and 31 leaves at all three depths. XGBoost also gets
a run with minimum child weight 10. One more run, `original-options`, uses only
the original options at all three depths, with the same features and targets.
It separates the effect of the larger option list from the effect of fewer
models. For HGB use `--backend hist_gradient_boosting --device cpu`. The contact
models stay those of the chosen pilot, so changing the rally backend alone does
not change the contact backend.

Each configuration writes its own log, checkpoints, validation streams and
scores as soon as each depth finishes. Finished configurations survive a
restart, and failed ones resume from their completed fits. The search exits
with an error if any configuration fails. `train --help` lists the individual
settings for a focused follow-up.

Training keeps the old one-level contact cross-fitting. The 32 training videos
form four groups, A to D. Each training video's contact scores come from a
contact model fitted without its group. Selectors also leave out the predicted
group when they generate earlier-selector training scores. Contact features
still carry indirect paths between groups' labels, as in the old stack, so
internal fits are not independent estimates of generalisation. The eight
validation videos stay out of every fit and give the comparison scores. Single
fits cannot separate close results, so close differences need repeat fits.

## Specialised layer comparisons

The structural screen tests a serve-repair specialist, an insertion specialist,
and both, each feeding one and then two whole-rally selectors. The specialist
labels are the existing local useful-edit targets: a contact repair can count
as useful even when another error keeps the rally wrong. The selector keeps the
original whole-rally target, span weights and tree settings.

```bash
python -m experiments.annotator.tree_simplification.search \
  --pools RUN/pools --output RUN/structures --backend xgboost --device cuda \
  --structures --jobs 3 --threads 2
```

These runs use the original options, so they compare directly with the
`original-options` search run. Specialist scores map to the resulting
sequence. When several edits lead to the same sequence, the largest finite
score wins. Missing edit types stay NaN. Each selector gets four extra
features: the option's serve and insertion scores, and the best of each within
its span. Specialists leave out the predicted training group at every
recursive rally fit. The final specialists use all 32 training videos. No
validation labels enter specialist fitting or feature construction.

One specialist plus one selector makes three models including contact. Both
specialists plus one selector make four. Each extra selector adds one model. To
extend a promising run, rerun `train` with the same settings and output folder
but a larger `--layers` value. Completed selectors and validation records are
reused. Any other setting change needs a new folder.

The old stack's 0.05 switching margin and ±5-frame contact timing are separate
questions from whether specialised labels help. Compare both the ±5 and ±10
validation tolerances before reading anything into near-equal results.

Build ±5-frame labels without rebuilding options or features, then train on
them:

```bash
python -m experiments.annotator.tree_simplification.targets \
  --pools RUN/pools --output RUN/targets-five --workers 4
python -m experiments.annotator.tree_simplification.train \
  --pools RUN/pools --output RUN/five-frame-preferred-original-options --original-options --layers 1 \
  --five-frame-targets RUN/targets-five --threads 2
```

A span uses ±5-frame labels if any of its judged options is correct at that
tolerance. Otherwise it keeps ±10-frame labels. Options with unknown labels in
the original target stay excluded. This changes only the target. The option rows, features and
span weight stay the same.

Replay a fixed final-choice margin on a finished run:

```bash
python -m experiments.annotator.tree_simplification.margins \
  --run RUN/search/original-options --output RUN/margins/original-options --threads 2
```

This compares the final selector's best option with its score for the initial
sequence. It evaluates margins of 0 and 0.05 without refitting or changing
earlier selectors. Margin 0 must reproduce every saved validation count before
either replay is reported. The fixed margin is a diagnostic. It is not a
calibrated threshold, and it was not tuned on validation.

The old eight-model stack includes a confidence-ranking model that does not
change the scored contacts. These structural comparisons only measure fully
correct rallies and F1. Review ranking is assessed separately, in
[Production export and review ranking](#production-export-and-review-ranking).

## Conditional contact-fold comparison

By default, each training video's contact scores come from a contact model
fitted on the 24 training videos outside its group. This check swaps those for
eight fixed folds of four videos, so each contact model fits 28 videos. The
selector groups stay A to D. Only the contact training inputs change, along
with the option lists built from them.

```bash
python -m experiments.annotator.tree_simplification.contact_folds \
  --prepared RUN/prepared --contact-run RUN/xgb-pilot \
  --original-pools RUN/pools --output RUN/contact-fit-28-videos \
  --threads 2 --workers 4
```

The eight fits reuse the pilot's contact settings and sampling order. They run
one after another before the option lists are built. `--only sset_01` builds
one option list for a smoke test, and the same output can then resume without
`--only`. The command rebuilds the 32 training option lists and copies all
eight validation ones unchanged. Validation still uses the contact model fitted
on all 32 training videos. Saved fold assignments and source settings must
match when resuming.

Train with the comparison run's exact saved `span_weight`. For the seed-0 run
with 24-video contact models, this is `371.86228729719033`:

```bash
python -m experiments.annotator.tree_simplification.targets \
  --pools RUN/contact-fit-28-videos/pools --output RUN/contact-fit-28-videos/targets-five --workers 4
python -m experiments.annotator.tree_simplification.train \
  --pools RUN/contact-fit-28-videos/pools --output RUN/contact-fit-28-videos/selected \
  --five-frame-targets RUN/contact-fit-28-videos/targets-five --layers 1 --threads 2 \
  --span-weight 371.86228729719033
```

Copy the comparison run's other settings exactly. Rebuild the ±5-frame labels
for these option lists, because their rows can differ. Report changed initial
contacts, option counts and the number of judged spans, next to validation
fully correct rallies and F1. The indirect paths between training labels
remain, and internal predictions are training features only. The eight
validation videos stay out of every fit.

## Paired seed repeats

Repeat the whole system at seed offsets 100, 200, 300 and 400. Pass each one to
`pilot --seed-offset`. The offset changes the contact, serve, rally and
confidence seeds together. Offset 0 keeps the original settings. A non-zero
offset fits a new pilot, so it cannot be combined with `--reference-bundle`.

For each offset, reuse the prepared inputs and run the pilot in a fresh output
folder. Rebuild all 40 option lists from that pilot, then rebuild its ±5-frame
labels. The old stack and the simplified stack at one offset must share that
offset's contact models, including the 32-video contact model's validation
predictions. Reusing the seed-0 option lists would hide contact sampling
variation.

Train each selector with `--seed` set to 20260905 plus the offset. Keep its
leaves, rounds and other settings fixed. Each run calculates its own span
weight as usual. The selected 31-leaf, 400-round runs used these values:

| Seed offset | Contact seed | Selector seed | Span weight |
|---:|---:|---:|---:|
| 0 | 20260824 | 20260905 | 371.86228729719033 |
| 100 | 20260924 | 20261005 | 372.0 |
| 200 | 20261024 | 20261105 | 371.53125 |
| 300 | 20261124 | 20261205 | 371.78125 |
| 400 | 20261224 | 20261305 | 371.91491887613773 |

Compare fully correct rallies and correct-player F1 at both tolerances, seed by
seed. The repeats show how much fitting varies on the eight validation videos.

## Production export and review ranking

The chosen selector came from this run, which fitted two selectors so both
depths could be compared:

```bash
python -m experiments.annotator.tree_simplification.train \
  --pools RUN/pools --five-frame-targets RUN/targets-five \
  --output RUN/selector-31-leaves-400-rounds --backend xgboost --device cuda \
  --threads 2 --seed 20260905 --leaves 31 --rounds 400 \
  --learning-rate 0.05 --min-child-weight 5 --layers 2
```

The archived evidence calls this folder `RUN/xgb-followup/longer31`. Its exact
settings are in [selected-settings.json.gz](results/selected-settings.json.gz).

Export it with one selector, without refitting the contact or final selector
models:

```bash
python -m experiments.annotator.tree_simplification.integrate \
  --run RUN/selector-31-leaves-400-rounds --layers 1 \
  --five-frame-targets RUN/targets-five \
  --output RUN/production-31-leaves-400-rounds --threads 2
```

The export needs the 32 training videos and checks the saved option-list and
target records. It copies the selector checkpoints into the export folder,
fits any missing held-group selectors and trains a small review ranker on
their finished sequences. Validation labels never enter fitting. The existing
indirect contact-label paths within training stay as they were.

The export rebuilds one training option list and its targets exactly. It then
rebuilds every validation option list through the production inference path.
All eight output streams must match the selected run. The bundle is saved and
loaded again before replay, which runs its metadata checks. The old stack's
streams must also match its pilot run. Model inference runs on CPU.

The result reports fully correct rallies, F1 and review ranking separately at
±5 and ±10. Selector-only scores and the fitted review scores rank the same
annotations. Ranking measures are correct-rally AUROC, wrong-rally average
precision, and the share of wrong rallies found in the first 10%, 25% and 50%
of the review queue. Sections that cannot be judged against labels are removed
before ordering, so these shares do not describe the full queue. The old
stack's ranking uses its own annotations and correctness labels, so it gives
context rather than a matched comparison. Scores are not calibrated
probabilities.

Outputs include `bundle/`, `result.json.gz`, per-video streams and review rows.
The run counts as complete only after every match check passes. Use a separate
export folder for each setting. The saved input record stops a resume with a
different source run or depth. This export is a validation comparison. It is
not a test result, and it does not change production defaults.

## Production inference timing

Compare two saved bundles through the same production entry point. The script
uses only prepared validation videos and two CPU threads. It reverses the model
order on the second repetition and checks every stream against its reference.
Use a fresh output folder.

```bash
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.benchmark \
  --prepared RUN/prepared \
  --old-bundle RUN/xgb-pilot/bundle \
  --new-bundle RUN/production-31-leaves-400-rounds/bundle \
  --old-streams RUN/xgb-pilot/validation_streams \
  --new-streams RUN/production-31-leaves-400-rounds/streams \
  --output RUN/timing-31-leaves-400-rounds
```

The timed stage covers contact prediction, building fresh options, rally
selection and review scoring. It leaves out vision processing, initial
contact-feature construction, file reads and metric scoring. It is a
descriptive CPU comparison, not a benchmark of the whole video pipeline.

The [report](../reports/tree_simplification.md#inference-cost) gives the
measured times. The [results index](results/README.md) describes the saved
timing and profile records.

## FPS motion-unit experiment

Raw features stay. Both scaling variants lowered mean validation fully correct
rallies and correct-player F1 over five seeds.
[Appendix B of the report](../reports/tree_simplification.md#appendix-b-frame-rate-scaling-of-motion-features)
has the numbers and limits.

`fps_normalisation` creates a separate prepared family. Its shared learned
motion columns are expressed in equivalent 30 fps units. First differences are
scaled by `fps / 30`, and the second-difference impulse by its square. The
impulse ratio is recomputed with its denominator floor in the same units. Frame
windows, candidate-region flags and vision evidence stay fixed. Both the
contact and rally models use these columns, so this measures a change to the
whole stack's features.

```bash
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.fps_normalisation prepare \
  --source RUN/prepared --output RUN/fps/prepared --workers 4
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.fps_normalisation contacts \
  --prepared RUN/fps/prepared --output RUN/fps/contacts-fit --threads 2
```

Then use the usual `pools`, `targets` and `train` commands with these new
paths. The contact stage fits the four held-group models on 24 videos each and
the final contact model on all 32 training videos. Keep the selector at 31
leaves, 400 rounds, one selector. Match the seed and span weight to the raw
run at the same offset (see [Paired seed repeats](#paired-seed-repeats)). Then
fit the review ranker and check all eight rebuilt validation option lists and
streams:

```bash
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.fps_review \
  --run RUN/fps/selector --five-frame-targets RUN/fps/targets-five \
  --output RUN/fps/review --threads 2
```

These checkpoints need the scaled preparation. The raw pilot and production
export commands reject this family, because production bundle metadata does
not record motion units. Prepared files and input records are checked before
fitting or replay. Use a fresh folder when inputs or the scaling code change.

For the velocity-only variant, add `--velocity-only` to preparation and use a
new output family. It leaves the impulse and ratio columns in raw units. A
separate units identifier stops its checkpoints mixing with the full variant.

For paired repeats, reuse each variant's prepared family and give every seed
its own output folder. Pass offsets 0, 100, 200, 300 and 400 to
`contacts --seed-offset`, and 20260905 plus that offset to `train --seed`. Pass
the raw run's span weight from the table to `train --span-weight`. Keep tree
settings and the validation videos fixed. Compare paired differences across
seeds. Five fits on the same videos are not five independent validation sets.

Scaled derivatives still depend on discrete smoothing windows, tracking noise
and missing observations. Validation at 25 and 30 fps cannot show transfer to
60 fps or amateur video. The test set played no part in this comparison.

## Refit changed preprocessing with raw features

`prepare --candidate` takes a gzip JSON candidate document with one entry in
`candidates`, using the existing `label`, `overrides_base30` and `strategies`
fields. The document also names its `schema` and `stage`. Without the option,
`prepare` keeps the default preprocessing. Each changed configuration needs its
own prepared family and its own fits. The saved configuration travels into the
bundle.

The sweep kept each configuration in `candidates/<configuration>/`, with one
`seed-<N>/` folder per seed offset. `results/test-run/` keeps the same layout.
For one configuration at seed offset 0:

```bash
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.prepare \
  --paths PATHS.toml --candidate candidate.json.gz \
  --output RUN/prepared --workers 4
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.input_stages contacts \
  --prepared RUN/prepared --output RUN/seed-0/contacts-fit \
  --seed-offset 0 --threads 4
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.pools \
  --prepared RUN/prepared --contact-run RUN/seed-0/contacts-fit \
  --output RUN/seed-0/pools --workers 4
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.targets \
  --pools RUN/seed-0/pools --output RUN/seed-0/targets-five --workers 4
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.train \
  --pools RUN/seed-0/pools --five-frame-targets RUN/seed-0/targets-five \
  --output RUN/seed-0/selector --backend xgboost --device cuda --threads 4 \
  --seed 20260905 --leaves 31 --rounds 400 --layers 1 \
  --learning-rate 0.05 --min-child-weight 5 --l2 1
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.input_stages export \
  --run RUN/seed-0/selector --five-frame-targets RUN/seed-0/targets-five \
  --output RUN/seed-0/export --threads 4
```

The contact stage fits the four held-group models on 24 videos each and the
final contact model on all 32 training videos. Each sweep run calculated its
own span weight. For another offset, pass it to `contacts --seed-offset` and
pass 20260905 plus the offset to `train --seed`. Repeat each configuration you
keep at offsets 0, 100, 200, 300 and 400.

Export checks all eight validation option lists and streams against the fitted
run. It saves a production bundle, results at both tolerances, and per-span
review rows for both the fitted review ranker and the selector-score fallback.
Unknown correctness labels stay in those rows. No filtering threshold is
applied.

## Test run and review thresholds

Validation chose two configurations for the test run, `raw_baseline` and
`body_window_36`, and the review thresholds, before the test run.
Baseline preprocessing stays the default, and review filtering stays off.
[The test run](../reports/tree_simplification.md#the-shuttleset22-test-run) and
[appendix D](../reports/tree_simplification.md#appendix-d-review-ranking-and-filtering)
of the report give the results.

`input_test` reads a selection record with schema `annotator-input-finalists/1`.
The record holds `fit_policy: contact-fit-32-videos`, `feature_units: raw`, the
five ordered `seed_offsets`, `primary_tolerance: 10`, and the evaluator's
`source_record()` as `evaluation_source`. Each entry in its `finalists` list
names a `candidate`, its `objectives` and five `seeds`. Each seed records its
`offset`, an absolute `export` folder, the export's full `settings`, and
`bundle_md5` hashes keyed by `model_files()`.

The repository keeps two kinds of input for the test run:

- **Saved records, tracked.** `results/test-run/` holds the selection record,
  each test bundle's export settings and validation results, and each contact
  stage record. `results/models/` holds the ten bundles. These are enough to
  load and check every test bundle.
- **External fixtures, not tracked.** Running test inference needs the 46 test
  videos' prepared inputs. A paths file names them with `test_inputs`,
  `test_courts`, `inpaint_root` and `pose_root`. Check these entries against
  the [paths example](../heuristic_only_tune/paths.example.toml). A paths file
  for training and validation only cannot run this evaluation.

Rebuild a working test-run folder from the tracked records, then load all ten
bundles through the test runner's own checks:

```bash
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.analysis.test_run materialise \
  --output DIR
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.analysis.test_run check \
  --output DIR
```

`materialise` copies `results/test-run/` into `DIR`, writes `DIR/` in place of
`SWEEP/`, and copies each bundle into its export folder. Add
`--use-current-checkout` before running inference from a later commit.
`input_test` requires a clean checkout whose `source_record()` equals the
record's `evaluation_source`, and this option writes the current checkout's
record into the copy. A re-run at a later commit therefore gets new evaluation
hashes.

Run inference for one configuration and seed:

```bash
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.input_test \
  --selection DIR/selection.json.gz --candidate raw_baseline --seed-offset 0 \
  --paths PATHS.toml --output RUN/test/raw_baseline/seed-0 --threads 4
```

The evaluator checks the chosen bundle and source before it loads the 46 test
videos. It uses the saved 32-video fit without refitting. Saved per-video
results can resume only with the same inputs, bundle and evaluator identity.

For the threshold analysis, combine the exported review rows and add
`candidate`, `seed_offset` and `split` (`validation` or `test`) to each row.
Keep the integer `span`, `video`, `tolerance`, `ranking`, `correct` and `score`
fields. The tracked row files already have this form. Choose thresholds from
all five validation seeds before looking at test outcomes:

```bash
python -m experiments.annotator.tree_simplification.review_thresholds choose \
  --rows experiments/annotator/tree_simplification/results/input-validation-review-rows.json.gz \
  --output thresholds.json.gz
python -m experiments.annotator.tree_simplification.review_thresholds evaluate \
  --rows experiments/annotator/tree_simplification/results/input-test-review-rows.json.gz \
  --thresholds thresholds.json.gz \
  --output threshold-evaluation.json.gz
```

`choose` reproduces `results/input-review-thresholds.json.gz` by content. To
evaluate the tracked thresholds directly, pass that file to `--thresholds`.

The analysis aims to keep 50%, 70%, 80%, 90% and 95% of predicted rally
sections, and adds an unfiltered reference. It chooses each threshold from
validation scores at ±10 and applies it unchanged at both tolerances. Sections
with no score stay in manual review. Sections with unknown correctness labels
count towards the share kept but stay out of error rates. Reports give pooled,
per-seed and per-video summaries. These thresholds are optional settings, not
calibrated probabilities, and filtering stays off by default.
