# Tree refits across a frozen preprocessing grid

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
  --workers 26 --threads 1 --hours 3
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
