# Annotator tree simplification

This experiment cut the annotator's fitted models from eight to three. The new
stack has a contact model, one rally selector and a review ranker. The rally
selector is an XGBoost model with 31 leaves and 400 boosting rounds. The review
ranker has 7 leaves and 100 rounds and only orders the output for manual
checking. The [report](../reports/tree_simplification.md) gives the results and
the reasons behind each choice. This guide says what the repository keeps and
how to run each step again.

The [joint settings search plan](hp_search.md) proposes a bounded search over
preprocessing, contact-tree settings, contact cutoffs and rally-selector settings.

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

The current default uses the [selected model](results/hp_search/fit-settings.json.gz),
with no maximum context limit in training or discovery. Its ten bundles are
under `results/hp_search/`. The tree-simplification evidence below
retains the models used in that experiment.

Everything behind the report's numbers sits in [results/](results/). The
[results index](results/README.md) lists every file with its checksum, the
label defaults, each preprocessing configuration's exact setting and what each
archived run folder held.

- `results/*.json.gz` are the evidence files. Each one backs a table or claim in
  the report.
- `results/models/<label>/` holds 12 saved bundles (`models.joblib` and
  `metadata.json`). `raw_baseline-seed-0/` is the experiment baseline. `selected/` is an earlier export. `raw_baseline-seed-<N>/`
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

The earlier test-run records require the checkout that produced their weights.
Use the current evaluator directly with a model directory; see
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
  --output RUN/seed-0/pools --workers 4
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.targets \
  --pools RUN/seed-0/pools --output RUN/seed-0/targets-five --workers 4
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.train \
  --pools RUN/seed-0/pools --five-frame-targets RUN/seed-0/targets-five \
  --output RUN/seed-0/selector --device cuda --threads 2 \
  --seed 20260905 --leaves 31 --rounds 400 \
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
allow unlimited outward context while rejecting spans that join two labelled
rallies. Contacts must all match within the chosen tolerance, with the right
players and no extras.
The batched [retraining driver](../monotrack_benchmark/writeup_scripts/README.md#fit-and-run-the-baseline)
shares preparation across seeds and can reuse contact fits and matching option pools.

Selector training divides each judged rally's total weight equally among its
judged options. The default total is 372; `--span-weight` changes it explicitly
for a fit. Proposal count does not determine the scale.

Each stage saves its settings and current input paths. Use fresh output
folders when changing a fit configuration. Export checks preprocessing and
target settings, then runs current validation annotation and scoring.

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
rebuilds its own contact-dependent proposals with the same configured weight scale. Five runs measure
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
regularisation, seed, device and fixed span weight. Proposal building exposes
the edit and deletion budgets for the single contact-toggle generator.

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

Evaluate a current model bundle directly:

```bash
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.input_test \
  --models MODELS --paths PATHS.toml --output RUN/test --threads 2
```

Use a new or empty output directory. The paths file must name `test_inputs`,
`test_courts`, `inpaint_root` and `pose_root`; see the
[paths example](../heuristic_only_tune/paths.example.toml).

The evaluator scores all 46 videos with the benchmark's 524 excluded-rally
windows. It checks the final population of 3,327 scored rallies and 37,184
labelled contacts. Contact, player and whole-rally results use the benchmark
scorer at five- and ten-frame tolerances, scaled for each video's frame rate.

The output includes streams, per-video judgements, totals and review ranking.
Review rows use the same excluded-window judgements. Rows with missing rally or
player labels remain unjudgeable; the selector and fitted review scores are
retained for the other rows. Select models on validation before this test run.
Test review correctness is judged after removing excluded windows; validation
review labels use the full rally boundaries. Interpret the two splits separately.

The earlier ten-bundle selection records remain historical report inputs; use
their original checkout to load those weights.

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
