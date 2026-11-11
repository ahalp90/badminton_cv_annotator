# Annotator tree simplification

This historical experiment cut the annotator's fitted models from eight to three.
That stack used a contact model, one rally selector and a review ranker. The rally
selector is an XGBoost model with 31 leaves and 400 boosting rounds. The review
ranker has 7 leaves and 100 rounds and only orders the output for manual
checking. The [report](../reports/tree_simplification.md) gives the results and
the reasons behind each choice. This guide says what the repository keeps and
how to run each step again.

The current annotator adds a separately trained selector for split pieces; see
the [selected four-model method](../post_split_correction/README.md).

The historical [joint settings search plan](hp_search.md) describes a search over
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
- The **contact run** is the folder whose fitted contact models feed the
  option lists.
- A **seed offset** (0, 100, 200, 300 or 400) is added to every base seed. Fits
  at different offsets share their videos, so they show fitting variation, not
  independent samples.

The base install includes XGBoost. The separate MonoTrack environment is not
needed here. Run every command from the repository root.

## Tracked models and prediction files

Use the [current deployed bundle](../../../runtime/deployed/annotator/) for
annotation. The models below belong to the earlier three-model experiments;
load them with the Git revision that produced them.

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
  XGBoost, kept for the timing comparison. The current loader accepts four-model bundles only; these older bundles require their historical code revisions.
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
enough to inspect the historical models, re-score every prediction file and redo
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

Use the normal training command for a complete four-model bundle. The current
saved-input runner below calls the same production fitter for split correction
and confidence. Reuse saved settings to keep the tuned tree settings, preprocessing and seed;
see the [refit guide](../../../docs/annotator/retuning.md)
for the manifest format.

```bash
PYTHONPATH=src python -m annotator.training fit \
  --manifest training-manifest.json \
  --settings runtime/deployed/annotator/fit-settings.json.gz \
  --output RUN/models --device cuda --workers 6 --gpu-fits 3
```

To repeat the split-correction experiment while retaining its original contact
and first-selector weights, use the saved high-shot-3 baseline inputs and fits:

Create `BASELINE` with the historical producer at Git commit `2e24e551`.

```bash
PYTHONPATH=src:. python -m experiments.annotator.tree_simplification.post_split_training \
  --paths PATHS.toml --prepared PREPARED --baseline BASELINE \
  --output RUN --seed 100 --workers 6 --gpu-fits 3 --device cuda
```

This command calls the normal follow-up/confidence fitter. It saves the complete
bundle under `RUN/seed-100/bundle/`, including `fit-settings.json.gz`, and scores
validation and test under `RUN/evaluation/`. Use validation to make choices;
read test as a larger-population check.

### Video order

Keep the original 32 training videos and their four groups for a matched
benchmark comparison. The benchmark's full contact fit sampled videos in
ShuttleSet ID order. Its held-group contact fits sampled in group order, keeping
the original order within each group. Both then fitted rows in ShuttleSet ID
order. Negative sampling consumes randomness, so sampling order and fit-row
order are separate choices.

The benchmark fits selector rows in video ID order: ordinary pools first, then
split-piece pools. Confidence rows follow groups A, B, C, D, with video ID order
within each group. The saved-input command above preserves these orders and
reuses the existing contact fits.

For a new dataset, the normal command keeps manifest order for contact sampling,
contact fit rows and selector pools. It groups final confidence rows by sorted
group name, preserving manifest order inside each group. Reusing settings alone
does not recreate the historical benchmark's separate contact-sampling order;
use its saved fits for that matched comparison.

Both selectors use the same proposal rules and target policy. Prefer five-frame
correctness when a judged rally retains a positive option, falling back to ten
frames otherwise. Keep unjudgeable options out of fitting. Divide the configured
weight of each judged rally equally over its judged alternatives; the default
total is 372 for both ordinary rallies and split pieces.

The old single-selector export commands have been retired. Reproducing those
historical fits requires their matching Git checkout.

## Paired seed repeats

Use a separate output directory and the corresponding saved settings for each
seed offset. Prepared inputs can be shared when preprocessing is unchanged.

| Offset | Contact seed | Selector/review seed |
| ---: | ---: | ---: |
| 0 | 20260824 | 20260905 |
| 100 | 20260924 | 20261005 |
| 200 | 20261024 | 20261105 |
| 300 | 20261124 | 20261205 |
| 400 | 20261224 | 20261305 |

The saved-input split-correction command accepts the offset through `--seed`.
For normal training, use each bundle's `fit-settings.json.gz`. Every seed has its
own contact-dependent proposals. These five runs measure fitting variation on
the same fixed split, not five independent datasets.

## Model bundles and review ranking

The normal fitter saves the contact model, first selector, follow-up selector
and final confidence model together. Temporary fits exclude the group whose
final predictions become confidence-training rows. Confidence ranks the final
rallies without changing their selected contacts.

Use the [refit guide](../../../docs/annotator/retuning.md) to fit and evaluate a
new dataset. The benchmark orders above matter when comparing with saved results.

## Further experiments

The selector command exposes rounds, leaves, learning rate, minimum child weight,
regularisation, seed, device and fixed span weight. Proposal building exposes
the edit and deletion budgets for the single contact-toggle generator.

`contact_folds.py` supports the separate eight-fold contact comparison.
`fps_normalisation.py` retains the frame-rate feature experiment. Its old bundle
export requires the historical checkout.
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
