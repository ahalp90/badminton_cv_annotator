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

The current default uses the [boundary-policy refit](results/boundary_retrain_v7/README.md):
all three trees were retrained, and validation selected the 2-second / 3-second
policy. Its ten bundles and fit settings are under `results/boundary_retrain_v7/2_3/`.
The earlier tree-simplification evidence below retains its original models.

Everything behind the report's numbers sits in [results/](results/). The
[results index](results/README.md) lists every file with its checksum, the
label defaults, each preprocessing configuration's exact setting and what each
archived run folder held.

- `results/*.json.gz` are the evidence files. Each one backs a table or claim in
  the report.
- `results/models/<label>/` holds 12 saved bundles (`models.joblib` and
  `metadata.json`). `raw_baseline-seed-0/` is the default current model. `selected/` is an earlier export. `raw_baseline-seed-<N>/`
  and `body_window_36-seed-<N>/` are the ten test-run bundles.
  `old-eight-model-stack-xgboost-seed-0/` is the old stack refitted with
  XGBoost, kept for the timing comparison. The current loader accepts the rally-selector bundles; the old eight-model bundle requires its historical code revision.
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
enough to load and check current bundles, re-score every prediction file and redo
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

## Reproduce the current fit

The maintained path is preparation, contact fitting, rally proposals, training
targets, selector fitting and export. It uses XGBoost throughout. The old backend
pilot and eight-model fitting commands have been retired; their measured results
remain in `results/` and the report.

A paths file names the extracted inputs. Start with the
[paths example](../heuristic_only_tune/paths.example.toml), using your own data
locations. `prepare --candidate` accepts a gzip JSON document containing one
preprocessing candidate with `schema`, `stage` and `candidates` fields. Without
that option, preparation uses the default preprocessing. Use the saved
`raw_baseline` candidate for a matched benchmark reproduction.

For one preprocessing configuration and seed offset 0:

```bash
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.prepare \
  --paths PATHS.toml --candidate candidate.json.gz \
  --output RUN/prepared --workers 4
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.input_stages contacts \
  --prepared RUN/prepared --output RUN/seed-0/contacts-fit \
  --seed-offset 0 --threads 2 --device cuda
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.pools \
  --prepared RUN/prepared --contact-run RUN/seed-0/contacts-fit \
  --output RUN/seed-0/pools --workers 4 \
  --boundary-before-seconds 2 --boundary-after-seconds 3
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.targets \
  --pools RUN/seed-0/pools --output RUN/seed-0/targets-five --workers 4
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.train \
  --pools RUN/seed-0/pools --five-frame-targets RUN/seed-0/targets-five \
  --output RUN/seed-0/selector --device cuda --threads 2 \
  --seed 20260905 --leaves 31 --rounds 400 --layers 1 \
  --learning-rate 0.05 --min-child-weight 5 --l2 1
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.input_stages export \
  --run RUN/seed-0/selector --five-frame-targets RUN/seed-0/targets-five \
  --output RUN/seed-0/export --threads 2 --device cuda
```

The contact stage fits four held-group models on 24 videos each and one final
model on all 32 training videos. It preserves negative-sampling order separately
from fit-row order. Validation labels never enter those fits.

The proposal stage uses held-group contact scores for training videos and the
final contact model for validation. It shares the production proposal/feature
builders. Targets prefer five-frame correctness where a judged rally retains a
positive option, falling back to ten-frame labels otherwise. Both target views
use the same boundary policy: the maximum time before the first labelled hit
and after the last. These are upper limits; shorter valid spans remain positive.
The boundary-policy comparison and full-stack retraining driver are documented
in the [benchmark scripts](../monotrack_benchmark/writeup_scripts/README.md).

Selector training gives each judged rally equal total row weight. A scale based
on the rule-based repair set anchors that weight across proposal expansions.
`--span-weight` supplies a fixed scale for matched experiments; otherwise it is
calculated from the current training proposals.

Each stage saves its settings and source/input identity. Reusing a directory with
different inputs or settings fails. Use fresh directories when changing a
configuration. Prepared features from a different source revision need explicit
provenance and numerical verification; changing a receipt is not a reproduction.
Old proposal caches must be rebuilt after the representation refactor.

## Paired seed repeats

Repeat contact fitting, proposals, targets, selector fitting and export in a
separate directory for each seed offset. Preparation can be shared when its
configuration and source are the same.

| Offset | Contact seed | Selector/review seed |
| ---: | ---: | ---: |
| 0 | 20260824 | 20260905 |
| 100 | 20260924 | 20261005 |
| 200 | 20261024 | 20261105 |
| 300 | 20261124 | 20261205 |
| 400 | 20261224 | 20261305 |

Pass the offset to `input_stages contacts --seed-offset` and the selector seed to
`train --seed`. Export derives the review settings from the saved run. Each seed
rebuilds its own contact-dependent proposals and weight scale. Five runs measure
fitting variation with the same fixed split; they are not independent datasets.

## Production export and review ranking

`input_stages export` fits the review model from held-group rally predictions,
then combines the final contact model, fitted selectors and review model into
`export/bundle/`. It retains the preprocessing configuration that produced the
features.

Export reloads the bundle through the production loader and replays all eight
validation videos. Its proposal arrays and final streams must agree with the
saved selector run. Review scores are saved separately from annotation decisions.
Inference uses CPU even when fitting used CUDA.

The [production refit guide](../../../docs/annotator/retuning.md) offers a simpler
manifest-based API for new datasets. The experiment commands above retain the
fixed split, ordering and records of this benchmark.

## Further experiments

The selector command exposes rounds, leaves, learning rate, minimum child weight,
regularisation, seed, device, layer count and fixed span weight. Proposal building
exposes the edit and deletion budgets. `--original-options` restricts training
to the rule-based repair set for a matched comparison.

`contact_folds.py` supports the separate eight-fold contact comparison.
`fps_normalisation.py` and `fps_review.py` retain the frame-rate feature experiment.
The [analysis guide](analysis/README.md) covers evidence reconstruction,
review thresholds and saved-result inspection. These experiments remain separate
from the production inference path.

Historical backend, specialist and eight-model commands require the repository
revision that produced their results. Current model loading deliberately rejects
the retired bundle schema.

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
  --paths PATHS.toml --output RUN/test/raw_baseline/seed-0 --threads 2
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
