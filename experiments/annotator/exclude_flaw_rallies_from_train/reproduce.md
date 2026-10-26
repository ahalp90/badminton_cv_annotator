# Reproducing the exclusion experiment

The archive is self-contained with respect to code and scored results. It does
not require the experimental branch. Start from the v7 scoring-fix commit
`2c888619`; retain that commit's test scoring and contact model settings.

## Inspect the results or rebuild the exclusion list

`results/previous_summary.json.gz` and `refitted_summary.json.gz` contain test
means, ranges and five per-seed values under `summary[condition][tolerance]`.
`validation_comparison.json.gz` contains the previous and refitted models scored
on the same 347 rallies. `results/confidence/{1_1,2_3}/` contains all threshold
curves and per-detection judgements, plus exact selected cutoffs and their test
outcomes. `monotrack_selection.json.gz` contains all twelve five-seed candidates.

From the repository root:

```bash
python experiments/annotator/exclude_flaw_rallies_from_train/build_rally_exclusions.py
pytest experiments/annotator/exclude_flaw_rallies_from_train/test_build_rally_exclusions.py
```

The builder uses original annotations under `data/shuttleset/set/` and the
existing `training/data/shuttleset/annotations/shots_master.csv`. Its output is
`results/rally_exclusions.csv.gz` inside this experiment. Reasons overlap.

## Refit and evaluate in an isolated checkout

Set `ARCHIVE` to this directory's absolute path and `CHECKOUT` to a fresh worktree
path. Set `PY` to the Python executable for the training environment:

```bash
git worktree add --detach "$CHECKOUT" 2c888619
git -C "$CHECKOUT" apply "$ARCHIVE/implementation.patch"
cd "$CHECKOUT"
export PYTHONPATH="$CHECKOUT/src:$CHECKOUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

The patch installs the original exclusion builder, modified scoring/training
code and regression tests. It also installs three experiment runners in this
subtree: `prepare_reused_pools.py`, `prepare_reporting.py` and `score_results.py`.
They require the patched code, so they are included in the patch rather than
imported by the restored main pipeline.

For the same efficient refit, set `RUN` to a fresh absolute output directory,
`PREVIOUS_TRAINING` to the original prepared/contact-fit root, `SOURCE_POOLS` to
the original no-max training root, and `PATHS` to the benchmark path-config TOML.
Existing cached pools must use schema `tree-simplification-option-pools/1`, with
no maximum outward context and their referenced prepared files available.

```bash
E=experiments/annotator/exclude_flaw_rallies_from_train
W=experiments/annotator/monotrack_benchmark/writeup_scripts
"$PY" "$E/prepare_reused_pools.py" \
  --source "$SOURCE_POOLS" --output "$RUN/training" --workers 28
"$PY" "$W/retrain_annotator.py" \
  --paths "$PATHS" --output "$RUN/training" \
  --reuse-training-root "$PREVIOUS_TRAINING" \
  --device cuda --workers 28 --stages targets train export
"$PY" "$E/prepare_reporting.py" --run "$RUN"
"$PY" "$W/annotator_batch_runs.py" \
  --paths "$PATHS" --training-root "$RUN/reporting/training" \
  --models-40 "$RUN/reporting/models_40" \
  --out "$RUN/reporting/annotator_runs" --policies no_max --workers 14
```

This retains contact fits and candidate evidence, refreshes cached targets,
refits ten selector/review bundles, reuses forty validation streams and makes
five comparison bundles with the existing 40-video contact trees. Inference uses
14 two-thread workers; the other stages use at most 28 CPU workers.

For test scoring, `score_results.py --predictions "$RUN/reporting/annotator_runs/no_max"
--baseline BASELINE_PRIMARY --out "$RUN/primary" --workers 28` reuses unchanged
MonoTrack, court and trees-off results. `BASELINE_PRIMARY` is the original v7
primary result directory with JSON/CSV files decompressed, except
`rally_courts.csv.gz`. The committed original primary directory is
`experiments/annotator/monotrack_benchmark/benchmark_results/discovery_context/primary/`.

Use the patched `compare_discovery_context.py` with `--run previous` and
`--run excluded`, each followed by its `validation/32_videos` prediction directory,
plus `--output COMPARISON --workers 28`. Both sides then use the same exclusions.
Run `discovery_boundary_surveys.py --run excluded
"$RUN/reporting/annotator_runs/no_max" --out "$RUN/surveys" --workers 28`
to repeat the full validation threshold search and apply selected cutoffs to test.

The patched `benchmark_scripts/five_seed.py` supports `--config MONOTRACK_CONFIG
--rescore-validation SOURCE --output NEW --parallel 28 --selection-only`.
This rescored all sixty saved MonoTrack validation runs; their
unchanged winner made new MonoTrack test inference unnecessary.
