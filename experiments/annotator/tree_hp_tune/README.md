# Tree refits across a frozen preprocessing grid

The current run uses the automatic five-seed workflow below. The single-seed
commands later in this document describe the underlying grid runner.

## Automatic five-seed comparison

[`finish_tree_retune.py`](../../../scripts/finish_tree_retune.py) supervises the
existing first batch, checks its shipped baseline, finishes the grid and runs
the repeat and final-test scripts. Future batches use 16 single-thread workers.
The guard slows down below 96/64 GiB available and pauses all fits below the
10 GiB emergency floor. Paused fits retain memory; an all-paused state below
16 GiB requires intervention. No phase kills training at a runtime limit.

[`tree_retune_repeats.py`](../../../scripts/tree_retune_repeats.py) freezes the
training top four plus shipped, then adds four seed settings to each original
fit. Seed offsets are 0, 100, 200, 300 and 400 across all existing random seeds.
Mean attributed F1 at ±10 frames on the original eight validation videos
selects the configuration. Training has no weight after shortlisting.
Mean correct-rally count and mean F1 at ±5 break ties; an exact tie favours shipped.

[`tree_retune_final_test.py`](../../../scripts/tree_retune_final_test.py) compares
all five fits of the frozen winner and shipped configuration. It refits contact
on the original 40 videos and preserves each seed's downstream models trained
on 32. The fixed 46-video ShuttleSet22 cohort is read only after selection.
Results retain per-video counts, seed means, spread and paired differences.
No seed is chosen from its test score.

[`tree_retune_status.py`](../../../scripts/tree_retune_status.py) reports progress:

```bash
python scripts/tree_retune_status.py --run-root /path/to/run
```

The run directory contains `grid48/`, `repeats/`, `final_test/`, `automation.json`
and phase logs. Each repeat/final phase pins its inputs and skips completed
results on resume. Do not launch a second coordinator while one is alive.

## Underlying grid runner

This runner tests 48 preprocessing configurations with matched tree refits.
Smoothing stays fixed at the shipped value of three frames at 30 fps.
Every fit uses the original ShuttleSet 32 training videos. The eight original
validation videos select among the training shortlist. Test data never enters
the search.

Each configuration keeps the existing tree hyperparameters, seeds, group
membership, sampling order and fitting order. Its saved bundle includes the
preprocessing configuration used to produce its training evidence. The
original group record is tracked at
[`training_video_score_groups.json`](../../../scratch/contact_det_full_ds_fit/records/training_video_score_groups.json).

Supply a paths TOML using the existing
[`paths.example.toml`](../heuristic_only_tune/paths.example.toml) format. This
runner reads only `shots_master`, `stages` and `development_courts`. Relative
paths resolve from the repository root. Run commands from that root with the
project environment active.

Start with the shipped configuration to measure a complete refit:

```bash
python -m experiments.annotator.tree_hp_tune.run run \
  --paths /path/to/paths.toml \
  --candidates experiments/annotator/tree_hp_tune/grid-48.json.gz \
  --output /path/to/output \
  --workers 1 --threads 1 --hours 3 --only shipped
```

Resume the same output directory with the whole frozen grid:

```bash
python -m experiments.annotator.tree_hp_tune.run run \
  --paths /path/to/paths.toml \
  --candidates experiments/annotator/tree_hp_tune/grid-48.json.gz \
  --output /path/to/output \
  --workers 16 --threads 1 --hours inf
```

Workers run complete configurations in separate processes. Worker count times
threads must be at most 26. Use one or two threads per worker according to the
measured throughput. The hours limit stops new submissions; active refits may
finish afterwards. Completed configurations are skipped on resume. The pilot
still records all 48 planned configurations and cannot unlock validation.

The output snapshot pins the complete grid, source, package versions, original
group record and input file records. Changed source, inputs or candidates
require a fresh output directory. Sticky player evidence is cached within
that directory and shared across configurations.

After all 48 configurations complete, validate the frozen shortlist:

```bash
python -m experiments.annotator.tree_hp_tune.run validate \
  --paths /path/to/paths.toml \
  --candidates experiments/annotator/tree_hp_tune/grid-48.json.gz \
  --output /path/to/output \
  --workers 6 --threads 1 --hours 3
```

The shortlist contains the top five training results plus the shipped
configuration if needed. Ranking uses player-attributed contact F1 at ten
frames, then fully correct rallies at ten frames, then player-attributed F1
at five frames. Those matching tolerances are defined at 30 fps and scaled to
each video's frame rate. Counts are pooled across videos before rates are calculated.
The same ranking selects the validation winner. Validation uses one numerical
thread per worker. Streams, per-video counts, pooled scores and timings remain
in the output directory. No new configurations are added after validation.
