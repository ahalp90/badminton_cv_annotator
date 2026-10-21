# Tree retune over preprocessing settings

This directory preserves the historical eight-model tuning results. Its HGB
fitting commands and their tests have been retired; reproducing those fits
requires the original repository revision. Use the
[current XGBoost guide](../tree_simplification/README.md) for new fits.

**Result: the shipped settings stay.**

- Tried: 48 combinations of preprocessing settings, mixing shipped values with
  the tuned heuristic's values. The trees were retrained for each one.
- Best on validation: `grid-017`.
- On test: +0.13 points of correct-player F1. That's less than the spread
  between seeds.

Full numbers: [RESULTS.md](RESULTS.md).

## Files

| File | What it does |
| --- | --- |
| `run.py` | Fits the trees for each of the 48 settings on the 32 training videos |
| `repeats.py` | Takes the top four settings plus shipped, fits four more seeds of each, scores all five seeds on validation and picks the winner |
| `final_test.py` | Scores the winner and shipped on the 46 test videos, five seeds each |
| `supervise.py` | Runs the phases one after another and pauses fits when memory runs low |
| `status.py` | Shows how far a run has got |
| `test_rerun.py` | Scores the saved 32-video and 40-video models on test under three rally and court conditions |
| [grid-48.json.gz](grid-48.json.gz) | The 48 settings |
| [results/](results/) | Scores, predictions and the saved models from the run |

The historical `tests/test_tree_hp_tune*.py` tests were retired with the fitting commands.

## Rerun saved models on test

- `test_rerun.py` compares the saved 32-video and 40-video
  contact-tree fits on all 46 test videos. It scores `grid-017` and shipped
  across five seeds, without refitting.
- Conditions: A discovers rallies with fast-robust scene courts; B uses known
  rallies ±1 second with video-robust courts; C uses the same known rallies
  with MonoTrack courts. A reuses the ten committed 40-video test runs.
- Paths: `TEST_PATHS.toml` needs `[paths]` entries for `test_inputs`,
  `test_courts`, `inpaint_root` and `pose_root`. Use the original benchmark
  inputs root for `test_inputs` to retain the ±1 second rally windows.
- Workers: at most `--workers` jobs run together, each with one numerical
  thread. The default is 28. Use the model-compatible environment described
  below.

```bash
python -m experiments.annotator.tree_hp_tune.test_rerun \
  --paths TEST_PATHS.toml --output RUN/test_rerun --workers 28
```

- Results: `RUN/test_rerun/<A|B|C>/<v32|final40>/<grid-017|shipped>/seed-<offset>/`
  contains per-video predictions and `scores.json.gz`.
- Summary: `summary.json.gz` and a printed table give the mean, min and max
  over five seeds at ±10 and ±5 frames. Counts are pooled across videos first.
  Production parity for shipped final40 seed 0 under B and C must pass.
- Resume: rerun the same command. Complete 46-video jobs are skipped; partial
  job directories are restarted. After changing code or inputs, use a new
  output folder. Job failures save tracebacks and stop new work while active
  jobs finish.
- Private files: exclude `RUN/test_rerun/configs/` from commits. Its generated
  TOMLs contain absolute input and bundle paths. Predictions and scores keep
  those paths out.

## What you need

- The full repository. The scripts read these files from outside this folder:
  - labels: `training/data/shuttleset/annotations/shots_master.csv`
  - training/validation split: `experiments/annotator/monotrack_benchmark/hitnet_retrain/shuttleset_development_split.json.gz`
  - training groups: `scratch/contact_det_full_ds_fit/records/training_video_score_groups.json`
  - test labels and video list: `experiments/annotator/monotrack_benchmark/benchmark_results/inputs/`
- Prepared shuttle tracks, poses and court evidence for every video. These
  aren't in the repository. The paths files point to them.
- Python 3.12 and scikit-learn 1.9.1. The saved models only load with that
  scikit-learn version.
- About 17 GiB of RAM per fit. Each fit takes one to six hours.

## Set up a run

1. Make a run folder. The commands below call it `RUN`.
2. Write `RUN/paths.toml` in the format of
   [paths.example.toml](../heuristic_only_tune/paths.example.toml). Fill in
   `shots_master`, `stages` and `development_courts`.
3. Write a second paths file for the test, called `TEST_PATHS.toml` below. It
   needs the same three entries plus `test_inputs`, `test_courts`,
   `inpaint_root` and `pose_root`.
4. Choose worker counts within these limits:
   - RAM: workers × 17 GiB must fit.
   - Cores: workers × threads at most 26 for `run.py`, 28 for the others.
   - Workers: at most 20 for `repeats.py`, `final_test.py` and `supervise.py`.

Run every command from the repository root.

## Run the phases

```bash
# 1. Fit all 48 settings
python -m experiments.annotator.tree_hp_tune.run \
  --paths RUN/paths.toml \
  --candidates experiments/annotator/tree_hp_tune/grid-48.json.gz \
  --output RUN/grid48 --workers 13 --threads 2 --hours inf

# 2. Check the shipped fit against the published one
python -c "from pathlib import Path; from experiments.annotator.tree_hp_tune import supervise; \
print(supervise.verify_baseline(Path('RUN/grid48'), Path('experiments/annotator/tree_hp_tune/results/grid48/trials/a322221a4890572e9065814a2e4e3e0ad033ba679e9b60a46ed38d9b639d7cd6/training_streams')))"

# 3. Refit the shortlist with more seeds and pick the winner
python -m experiments.annotator.tree_hp_tune.repeats --run-root RUN --workers 14 --threads 2

# 4. Score the winner and shipped on test
python -m experiments.annotator.tree_hp_tune.final_test --run-root RUN --test-paths TEST_PATHS.toml --workers 10 --threads 2

# Progress at any point
python -m experiments.annotator.tree_hp_tune.status --run-root RUN
```

Notes:

- Step 2 must pass before `repeats.py` will start.
- To time one fit first: add `--only shipped` to step 1.
- To resume after a stop: rerun the same command. Finished work is skipped.
- Resume checks training settings and seeds as well as inputs and environment.
  New grid snapshots and fit results record the settings used. Legacy grids use
  seed 0 from their frozen repeat plan as the available settings record. This
  does not independently verify the original grid fit settings. A legacy grid
  without that plan has unknown settings and requires a fresh run folder.
  Contact prediction rules are recorded explicitly in new snapshots and plans.
  Legacy comparisons read those rules from the saved model bundles.
- Accepted resumes preserve the historical snapshot and plans, including old
  source hashes. Source revision and dirty state are recorded when each worker
  or trial starts; they do not block resumes.
- After changing the algorithm or inputs: start a new run folder.

## Run unattended

`supervise.py` runs every phase after the grid, one after another. The
completed run used it.

What it does about memory:

- reserves room for each running fit's remaining memory growth
- pauses the fits with the most growth left when there isn't enough room
- resumes them as memory frees up

To use it, start the grid in the background and pass the supervisor its
process ID:

```bash
python -m experiments.annotator.tree_hp_tune.run --paths RUN/paths.toml \
  --candidates experiments/annotator/tree_hp_tune/grid-48.json.gz \
  --output RUN/grid48 --workers 16 --threads 1 --hours inf &
python -m experiments.annotator.tree_hp_tune.supervise --run-root RUN --repository . \
  --initial-pid $! \
  --previous-streams experiments/annotator/tree_hp_tune/results/grid48/trials/a322221a4890572e9065814a2e4e3e0ad033ba679e9b60a46ed38d9b639d7cd6/training_streams \
  --test-paths TEST_PATHS.toml --workers 14 --threads 2
```

Output:

- state: `RUN/automation.json`
- logs: `RUN/<phase>.log`

## Try different settings

1. Write a new grid file in the format of `grid-48.json.gz`. Each entry has a
   label, setting overrides at 30 fps and strategy choices. One entry must be
   labelled `shipped`.
2. Update the hard-coded values:
   - `repeats.py`: expects 48 fits and reads `grid-48.json.gz`
   - `supervise.py`: expects at least 26 fits from the first batch

## What's in results/

A trimmed copy of the run folder, with the same layout.

Path labels inside the records:

- `<run>`: the run folder
- `<stages>`, `<development_courts>`, `<test_inputs>`, `<test_courts>`,
  `<inpaint_root>`, `<pose_root>`: the matching paths-file entries
- anything else: relative to the repository root

| Path | Contents |
| --- | --- |
| `grid48/snapshot.json.gz` | Source revision, package versions, training settings, and input file paths, sizes and timestamps |
| `grid48/training.json.gz` | Training scores for all 48 settings, best first |
| `grid48/trials/<id>/result.json.gz` | One setting's training counts per video, fit details, run time and peak memory |
| `grid48/baseline-parity.json.gz` | Result of the step 2 check |
| `grid48/trials/a322…/training_streams/` | Shipped training predictions, used by step 2 |
| `grid48/trials/<id>/validation*` | Seed 0 validation scores and predictions for the shortlist |
| `repeats/seed-<offset>/trials/<id>/` | The other four seeds: training scores, seed settings, validation scores and predictions |
| `repeats/plan.json.gz`, `summary.json.gz`, `selected.json.gz` | Frozen shortlist and seeds, five-seed results, and the winner |
| `final_test/plan.json.gz`, `summary.json.gz` | Test plan, and every seed's test totals with paired differences |
| `grid48/trials/<id>/bundle_v32/`, `repeats/seed-<offset>/trials/<id>/bundle_v32/` | 32-video models for grid-017 and shipped, used for validation |
| `final_test/trials/<id>/seed-<offset>/` | Test scores, test predictions, contact refit record and the model |
| `test_rerun/<condition>/<stage>/<label>/seed-<offset>/` | Test scores and predictions from `test_rerun.py`. Condition is A, B or C; stage is `v32` or `final40` |
| `test_rerun/summary.json.gz` | Five-seed mean, min and max per condition, plus the production check |

Setting IDs:

- `grid-017`: `a521305d…`
- shipped: `a322221a…`

Load a test model: `annotator.models.load_models(<trial>/bundle_final40)`.

Not included:

- the other 54 models fitted during selection
- most training predictions
- cached player evidence

Use the commands above to recreate them.

Code version:

- The run used commit `2dda36ff`.
- Since then the scripts moved here from `scripts/` with updated imports, and
  an unused validation shortcut was removed from `run.py`.
