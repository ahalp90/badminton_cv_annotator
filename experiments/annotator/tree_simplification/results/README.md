# Tree simplification results

This folder holds what you need to check the numbers in the
[campaign report](../../reports/tree_simplification.md) without running the
experiment again. It has the evidence file behind each table, the chosen model,
the ten test models and every validation and test prediction file. It also has
the records that let the test models load again, and the time and memory each
run took. The [reproduction guide](../README.md) has the commands that made them.

Two groups of videos appear throughout:

- **Validation:** 8 videos with 668 labelled rallies and 5,696 contacts. All
  model choices were made on these videos.
- **Test:** 46 ShuttleSet22 videos with 3,327 rallies and 37,184 contacts. Only
  the two preprocessing configurations chosen on validation ran on them.

Five seed offsets (0, 100, 200, 300 and 400) refit each model on the same
training videos. They show how much a result moves between fits. They are not
independent samples.

## What each file holds

| File or folder | What it holds |
|---|---|
| `five-seed-evidence.json.gz` | Validation results at all five seeds for the old eight-model stack refitted with XGBoost, and for 7-leaf and 15-leaf selectors with one and two selectors. Also the six-setting screen at seed 0. |
| `selector-settings-evidence.json.gz` | Validation results at all five seeds for 31-leaf selectors with 200 and 400 rounds, one and two selectors. Per-video counts, pooled scores and paired differences. |
| `previous-hgb-validation.json.gz` | Validation means of the best earlier HGB setting (grid-017), with its branch, commit and result path. A descriptive comparison, not a paired one. |
| `fps-normalisation-evidence.json.gz` | The frame-rate test at all five seeds: motion features scaled to 30 fps units (all of them, or velocity only) against raw features. |
| `selected-settings.json.gz` | The chosen model's source run, selector count, and exact training and option-list settings. |
| `selected-production.json.gz` | The chosen seed-0 bundle replayed through the production code, and the review-ranking comparison. |
| `selected-review-rows.json.gz` | The per-section review rows behind that ranking comparison. |
| `selected-timing.json.gz` | Inference time of the old and new bundles on the eight validation videos, in both orders, on two CPU threads. Every output is checked against the saved predictions. |
| `selected-profile.json.gz` | A profile of the chosen model on `sset_30`. Building and describing the larger option list takes most of the time. Profiling adds its own overhead. |
| `input-preprocessing-evidence.json.gz` | The preprocessing sweep: validation results for all 20 configurations, the two chosen for the test, test results and analysis, the review-threshold evaluation and run costs. |
| `input-validation-review-rows.json.gz` | 30,160 review rows from validation. One row per proposed rally section, tolerance (±5 or ±10 frames) and ranking (`fitted` reviewer or `selector` score), for both test configurations at all five seeds. |
| `input-test-review-rows.json.gz` | The same kind of rows from the test run, 160,840 in all. |
| `input-review-thresholds.json.gz` | The review thresholds chosen on validation rows before any test access. Each target share of sections to keep maps to a score threshold. |
| `input-review-tradeoff.svg` | Figure for baseline preprocessing: the share of test sections kept against the error among judged kept sections, at ±5 and ±10 frames. |
| `input-stage-costs.csv` | Time and memory of every timed command in the sweep and the test run. See [Stage costs](#stage-costs). |
| `screening-evidence.json.gz` | The search arms dropped before the five-seed repeats, all single seed-0 fits on validation: the backend pilot, the first grid with XGBoost and HGB, the label and option-list comparison, smaller trees, specialist selectors, keep-margin replays and contact models fitted on 28 videos. Also the option-list ceiling and the frame-rate contact diagnostics. |
| `target-screen.svg` | Figure for the report's appendix A3: fully correct rallies against correct-player F1 for the first grid, the ±5-frame label runs and the old stack refitted with XGBoost. |
| `models/` | 12 model bundles. See [Model bundles](#model-bundles). |
| `streams/` | 1,220 prediction files and 460 per-video test score files. See [Prediction files](#prediction-files). |
| `test-run/` | 31 records that let the ten test bundles load again through `input_test`'s own checks. See [Test-run records](#test-run-records). |

The md5s of both review-row files and the threshold file match the values that
`input-preprocessing-evidence.json.gz` records (`validation_rows_md5`,
`test_rows_md5` and `thresholds_md5`).

## Labels and seeds

Folder names under `models/` and `streams/` say what the model is.

- A selector picks one reading of a rally from a fixed list of options. A
  second selector sees the first one's scores and can change its pick. The code
  calls each selector a "layer".
- Selector labels give leaves per tree, boosting rounds and the number of
  selectors, for example `31-leaves-400-rounds-1-selector`. Learning rate 0.05
  and minimum child weight 5 apply unless the label says otherwise.
- `31-leaves-400-rounds-1-selector` is the selected setting.
- `old-eight-model-stack-xgboost` is the old eight-model annotator refitted
  with XGBoost. It is the like-for-like reference on validation.
- `all-motion-scaled` and `velocity-only-scaled` are the two frame-rate arms.
- Preprocessing configurations keep their sweep ids, such as `raw_baseline`
  and `body_window_36`. See [Sweep configurations](#sweep-configurations).
- `seed-<N>` means seed offset N, added to every base seed: 20260824 for the
  contact model and 20260905 for the selectors and the review ranker. Offset 0
  keeps the original seeds.

## Model bundles

Each folder under `models/` holds `models.joblib` and `metadata.json`.
`annotator.models.load_models(folder)` loads any of them.

| Folder | What it is | `models.joblib` md5 | `metadata.json` md5 |
|---|---|---|---|
| `selected` | The chosen model: one 31-leaf, 400-round selector, seed 0, baseline preprocessing, exported at `e2b948d2` | `9f96c0f40ab42b9498cb0c6a2533951d` | `77899da271f783a20a7128bc8625050d` |
| `raw_baseline-seed-0` | Test bundle, baseline preprocessing | `44c858b2d92c4c8658f26e22fde6c9ad` | `77899da271f783a20a7128bc8625050d` |
| `raw_baseline-seed-100` | Test bundle, baseline preprocessing | `edb8a45d579d9822d96fdde6f627ff2e` | `77899da271f783a20a7128bc8625050d` |
| `raw_baseline-seed-200` | Test bundle, baseline preprocessing | `53de8ecb4bd4a4c8799d4e9f73dca5c9` | `77899da271f783a20a7128bc8625050d` |
| `raw_baseline-seed-300` | Test bundle, baseline preprocessing | `0f726d46b444fde15690de47a6105b9a` | `77899da271f783a20a7128bc8625050d` |
| `raw_baseline-seed-400` | Test bundle, baseline preprocessing | `7c1f0c4e808a9a412eecfef8439576c1` | `77899da271f783a20a7128bc8625050d` |
| `body_window_36-seed-0` | Test bundle, 36-frame body-size half-window | `c5daa4b22a02c90bd2c01a69506768ff` | `77899da271f783a20a7128bc8625050d` |
| `body_window_36-seed-100` | Test bundle, 36-frame body-size half-window | `0f9c4be9bc959c00740b9726cc840ef9` | `77899da271f783a20a7128bc8625050d` |
| `body_window_36-seed-200` | Test bundle, 36-frame body-size half-window | `df46dcc9fda2a9066af2317f1b26f2bd` | `77899da271f783a20a7128bc8625050d` |
| `body_window_36-seed-300` | Test bundle, 36-frame body-size half-window | `373ec6618329b0b376d8becf2a08e70f` | `77899da271f783a20a7128bc8625050d` |
| `body_window_36-seed-400` | Test bundle, 36-frame body-size half-window | `67c38de9b275130a64ef65a18e618f18` | `77899da271f783a20a7128bc8625050d` |
| `old-eight-model-stack-xgboost-seed-0` | The old eight-model stack refitted with XGBoost, seed 0; the reference in the timing comparison | `1e77654b05acb255e057cdf6b617d390` | `a87f68ff2acc886c4aed83fc4f095ef4` |

The ten test bundles' md5s equal `bundle_md5` in `test-run/selection.json.gz`.
The preprocessing setting lives inside `models.joblib`, so every new-stack
bundle has the same `metadata.json`.

`selected` and `raw_baseline-seed-0` differ in bytes but not in output. Both are
seed-0 fits of the selected setting with baseline preprocessing. The
`raw_baseline` sweep predictions equal the `31-leaves-400-rounds-1-selector`
predictions in content for all 40 validation files (five seeds, eight videos).
The chosen bundle's own export reproduces the eight seed-0 files.

The chosen model is the original seed-0 fit. No seed was picked by its
validation score.

## Prediction files

Paths follow `streams/<table>/<label>/seed-<N>/<video>.json.gz`. Each file holds
one video's predicted contacts (`events`) and rally sections (`sequences`).
The table folder says which part of the report the files feed.

| Folder | Files | What they cover |
|---|---:|---|
| `selection/` | 360 | The old stack and the eight selector labels, five seeds, eight validation videos |
| `fps/` | 80 | `all-motion-scaled` and `velocity-only-scaled`, five seeds, eight validation videos |
| `input-validation/` | 320 | All 20 sweep configurations at seed 0, plus `raw_baseline`, `body_window_36`, `body_window_high`, `end_rest_180` and `end_rest_high` at the other four seeds |
| `input-test/` | 460 | `raw_baseline` and `body_window_36`, five seeds, 46 test videos |
| `input-test/<label>/seed-<N>/scores/` | 460 | Each test video's scored counts at ±5 and ±10, review summaries and review rows, the md5 of its prediction file and its evaluation hash |

Test videos are named by their ShuttleSet22 number, for example `10.json.gz`.

## Test-run records

`test-run/` copies the sweep's folder layout, because the records name each other
by path.

- `selection.json.gz` names the two configurations chosen on validation and
  their objectives. For each seed it gives the export folder, the full export
  settings and the bundle md5s. `input_test --selection` reads this file.
- `candidates/<configuration>/seed-<N>/export/settings.json.gz` equals that
  seed's `settings` in the selection record.
- `candidates/<configuration>/seed-<N>/export/result.json.gz` holds the
  validation counts and the replay check from export.
- `candidates/<configuration>/seed-<N>/contacts-fit/contact-stage.json.gz`
  lists that seed's contact fits (four fits on 24 videos each, and one on all
  32), with their seed offset and settings.

Paths in these records start with `SWEEP/`. To load the bundles again, copy
`test-run/` to a new folder and replace `SWEEP/` with that folder's absolute
path. Then copy each bundle from `models/<configuration>-seed-<N>/` into
`candidates/<configuration>/seed-<N>/export/bundle/`. `input_test` checks the
records, seeds and bundle md5s before it loads a bundle.

These copies use the current key names. Apart from the `SWEEP/` prefix and
those renamed keys and values, each record equals its original.

**What is saved here, and what is not.** The records, bundles and prediction
files are enough to load and check all ten test bundles, score every prediction
file again and redo the threshold analysis. Running test inference again also
needs the 46 test videos' prepared inputs: shuttle tracks, courts, inpainted
frames and poses. A paths file names them as `test_inputs`, `test_courts`,
`inpaint_root` and `pose_root`. They are not in this repo. Refitting any model
also needs prepared training inputs. `prepare` rebuilds them from the
development videos' saved shuttle, pose and court outputs, which are not in
this repo either.

**Evaluation hashes.** The `evaluation_sha256` values in the score files and the
evidence came from the code at `94099e16`. Commit `927cd260` later renamed a
key in the test-run identity those hashes cover, so a new run of the same
bundle gets a different hash. The recorded hashes stay valid for the files they
describe.

## Sweep configurations

Each id changes one preprocessing input from the baseline. Values are at 30 fps.
Frame counts and speed thresholds scale with each video's frame rate.

| Id | Input changed | Baseline | This configuration |
|---|---|---:|---:|
| `raw_baseline` | nothing | | |
| `rest_speed_low` | rest-speed threshold (`rest_speed`) | 0.002 | 0.001 |
| `rest_speed_high` | rest-speed threshold (`rest_speed`) | 0.002 | 0.0125 |
| `end_rest_low` | frames of rest before a rally ends (`end_rest_frames`) | 90 | 30 |
| `end_rest_120` | frames of rest before a rally ends (`end_rest_frames`) | 90 | 120 |
| `end_rest_high` | frames of rest before a rally ends (`end_rest_frames`) | 90 | 150 |
| `end_rest_180` | frames of rest before a rally ends (`end_rest_frames`) | 90 | 180 |
| `gap_demotion_low` | gap-state demotion bound (`gap_state_demotion_bound`) | 75 | 15 |
| `gap_demotion_high` | gap-state demotion bound (`gap_state_demotion_bound`) | 75 | 150 |
| `body_window_low` | body-size half-window (`body_unit_half_window`) | 12 | 1 |
| `body_window_18` | body-size half-window (`body_unit_half_window`) | 12 | 18 |
| `body_window_high` | body-size half-window (`body_unit_half_window`) | 12 | 24 |
| `body_window_36` | body-size half-window (`body_unit_half_window`) | 12 | 36 |
| `smooth_window_low` | trajectory smoothing window (`smooth_window`) | 3 | 1 |
| `smooth_window_high` | trajectory smoothing window (`smooth_window`) | 3 | 5 |
| `start_speed_0075` | start-speed threshold (`start_speed`) | 0.015 | 0.0075 |
| `start_speed_low` | start-speed threshold (`start_speed`) | 0.015 | 0.01 |
| `start_speed_high` | start-speed threshold (`start_speed`) | 0.015 | 0.0225 |
| `start_speed_0300` | start-speed threshold (`start_speed`) | 0.015 | 0.03 |
| `span_region_start` | how a rally section opens (`span_open`) | back-fill | region start |

The body-size half-window smooths each player's box height, which turns
shuttle-to-wrist distance into body-height units. The 36-frame value is the
half-window, not the total window length. `candidate_specs` in
`input-preprocessing-evidence.json.gz` records each configuration's exact
override.

## Stage costs

`input-stage-costs.csv` has one row per timed command: 220 from the sweep and 10
from the test run. GNU `time` measured each one.

| Column | Meaning |
|---|---|
| `configuration` | Sweep configuration id |
| `seed_offset` | Seed offset; empty for `prepare`, which ran once per configuration |
| `stage` | `prepare` (prepared inputs), `contacts` (contact models), `pools` (option lists), `targets` (±5-frame labels where possible), `selector` (selector fit), `export` (review ranker, bundle and validation replay) or `test` (test inference and scoring) |
| `elapsed_s` | Elapsed seconds |
| `user_cpu_s` | User CPU seconds |
| `max_rss_kib` | Peak resident memory of that command, in KiB |

- The 220 sweep rows sum to 75,904.81 s elapsed (21.08 h) and 126,599.23 s of
  user CPU (35.17 h). Summed elapsed time is not batch wall time, because runs
  went in parallel.
- Peak memory is per command, not the total in use at once. The largest is
  18,950,460 KiB (18.07 GiB), from `end_rest_low`, seed 0, `selector`.
- Each `reported_rss_peak_file` in `input-preprocessing-evidence.json.gz`, such
  as `remote/candidates/end_rest_low/seed-0/selector.time`, names one row of
  this table by configuration, seed and stage.
- Per-stage medians, 90th percentiles, maxima and sums in that evidence file's
  `validation_costs.stage_distributions` match this table.
- The ten test runs took 33 minutes each on average.

## Paths in the evidence

The evidence keeps source paths, with the private part replaced by a
placeholder:

| Placeholder | Meaning |
|---|---|
| `RUN/` | The experiment's run folder. The guide's commands use the same name. |
| `SWEEP/` | The preprocessing sweep's run folder. |
| `CHECKOUT/` | The code checkout that ran the profile. |
| `VENV/` | That checkout's Python environment. |
| `USER_DATA/` | The folder holding that environment's Python interpreter. |

The run folders themselves are not in this repo. Their names stay as written,
so each path still says which run produced a number. Fresh runs from the guide
use plainer folder names:

| Folder in evidence paths | What it holds | Folder in the guide's commands |
|---|---|---|
| `RUN/prepared` | Prepared inputs for the 40 development videos | `RUN/prepared` |
| `RUN/xgb-5` | The old eight-model stack refitted with XGBoost, seed 0. Its contact models built the option lists in `RUN/pools-xgb-5`. | `RUN/xgb-pilot` |
| `RUN/pools-xgb-5` | Option lists for the 40 videos, from `RUN/xgb-5`'s contact models | `RUN/pools` |
| `RUN/tight-targets/expanded-7`, `RUN/tight-targets/expanded-15` | 7-leaf and 15-leaf, 200-round selectors (one and two) on the larger option list with ±5-frame labels where possible, seed 0 | a `train` run with `--five-frame-targets` |
| `RUN/xgb-followup` | The six-setting screen and the 31-leaf runs at seed 0 | one `train` run per setting |
| `RUN/xgb-followup/child10` | 15 leaves, 200 rounds, minimum child weight 10 | as above |
| `RUN/xgb-followup/faster15` | 15 leaves, 200 rounds, learning rate 0.1 | as above |
| `RUN/xgb-followup/longer15` | 15 leaves, 400 rounds | as above |
| `RUN/xgb-followup/slower15` | 15 leaves, 400 rounds, learning rate 0.025 | as above |
| `RUN/xgb-followup/leaves31` | 31 leaves, 200 rounds | as above |
| `RUN/xgb-followup/longer31` | 31 leaves, 400 rounds: the selected setting, and the source of the chosen bundle | `RUN/xgb-followup/longer31` |
| `RUN/seed-repeats/seed-<N>/pilot` | The old eight-model stack refitted with XGBoost at seed offset N. Its contact models built that seed's option lists. | `pilot --seed-offset N` in a new folder |
| `RUN/seed-repeats/seed-<N>/tightest-expanded7`, `RUN/seed-repeats/seed-<N>/tightest-expanded15` | The 7-leaf and 15-leaf selectors at seed offset N | a `train` run per seed |
| `RUN/xgb-repeats/seed-<N>/leaves31`, `RUN/xgb-repeats/seed-<N>/longer31` | The 31-leaf, 200-round and 400-round selectors at seed offset N | a `train` run per seed |
| `RUN/production/expanded31-400-depth1-e2b948d2` | The chosen model's production export: bundle and validation predictions | `RUN/production-31-leaves-400-rounds` |
| `RUN/production/timing-expanded31-400-depth1-e2b948d2` | The timing comparison's output | `RUN/timing-31-leaves-400-rounds` |
| `RUN/fps-motion-v1/seed-<N>`, `RUN/fps-velocity-v1/seed-<N>` | The frame-rate arms: all motion features scaled, and velocity only scaled | `RUN/fps/...`, one folder family per arm |
| `SWEEP/candidates/<configuration>/prepared` | Prepared inputs for one sweep configuration | `RUN/prepared` |
| `SWEEP/candidates/<configuration>/seed-<N>/{contacts-fit,pools,targets-five,selector,export}` | One sweep configuration's stages at seed offset N | `RUN/seed-<N>/...` |

The archived run records also keep the key names they were written with.
Commits `927cd260` and `de1b931a` renamed many of those keys in the code and in
the files here, and their diffs map old names to new.

## Checkouts and the timing script

The experiment ran from checkout `ddd2a7f2`. The production export used
`e2b948d2`. The timing script ran beside `e2b948d2` with md5
`e040d4041beaaf982c966fdd6ca9e214`. Commit `9a1cd0b6` later added it unchanged
as `benchmark.py`.

## Rebuilding and checking these files

The scripts in [`../analysis/`](../analysis/) rebuild the evidence files and
figures here and compare them with the tracked copies. JSON files must match
by content. Figures must match apart from Matplotlib's generated element ids.
The [analysis README](../analysis/README.md) lists each script's inputs and
the fields it copies rather than rebuilds.

| File | Rebuilt or checked by | Needs data outside the repo |
|---|---|---|
| `five-seed-evidence.json.gz`, `selector-settings-evidence.json.gz`, `previous-hgb-validation.json.gz` | `validation_evidence.py selectors` | no |
| `selected-production.json.gz` | `validation_evidence.py selectors` (the replay) and `ranking` (the review ranking) | no |
| `fps-normalisation-evidence.json.gz` | `validation_evidence.py frame-rate` | no |
| `input-preprocessing-evidence.json.gz`, `input-review-thresholds.json.gz` | `input_evidence.py`, with `input_selection.py rank` and `choose` for the choice of test configurations | no |
| `input-validation-review-rows.json.gz`, `input-test-review-rows.json.gz` | `input_selection.py rows` | yes: the local archive |
| `input-review-tradeoff.svg` | `plot_review_tradeoff.py` | no |
| `test-run/` | `test_run.py materialise`, then `check` loads all ten bundles | no |
| `screening-evidence.json.gz` | `screening_evidence.py` | yes: the local archive |
| `target-screen.svg` | `plot_target_screen.py` | no |
| `selected-profile.json.gz` | `profile_selected.py` makes a new profile; times differ from run to run | yes: prepared arrays |

The other files are saved records: the chosen settings, the timing run, the
stage costs, the model bundles and the prediction files. The scripts above
read them as inputs.

The local archive is `raw-data.tar.zst` in the campaign's local folder. Its
`remote/` folder holds the small records pulled from `RUN/`, with the same
folder names. So `remote/search-xgb/leaves-15` in
`screening-evidence.json.gz` is the run folder `RUN/search-xgb/leaves-15`.

## How the scores work

- A rally is fully correct when one predicted section contains its whole
  labelled contact sequence. Every contact must match within tolerance, with
  the correct player side and no extra contacts. Start and end frames need not
  match exactly.
- Correct-player F1 is twice the matched contacts with the correct player,
  divided by predicted plus labelled contacts.
- Tolerances of ±5 and ±10 frames are set at 30 fps and scaled to each video's
  frame rate.
- Counts are pooled across videos before each seed's F1 is computed. The mean
  F1 is the mean of the five per-seed scores.
- A paired difference always compares the same seed, tolerance and videos.

The validation comparisons change whole systems. They do not separate the
effects of the tree library, the option list or the removal of individual
models.
