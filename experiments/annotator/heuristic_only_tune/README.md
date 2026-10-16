# Tune the tree-disabled annotator

These scripts tune the complete heuristic annotator on original ShuttleSet.
They disable all eight annotator trees while retaining the saved shuttle,
pose and court inputs. Both the heuristic and full annotator discover rallies.
The scripts do not change production defaults.

## Search and selection

Use the frozen original ShuttleSet split: 32 training videos (2,691 rallies,
27,571 contacts) and eight validation videos (668 rallies, 5,696 contacts).
The [split](../monotrack_benchmark/hitnet_retrain/shuttleset_development_split.json.gz)
records the video IDs and frame rates. ShuttleSet22 is held out until settings
have been selected on validation.

1. Search for the highest pooled correct-player contact F1 at ±10 frames.
   Break ties with fully correct rallies, contact F1, correct-player F1 at ±5,
   then the configuration ID.
2. Follow with a local search for fully correct rallies. Swap the first two
   ranking terms and retain the contact winner as a separate choice.
3. Freeze a shortlist containing the top three settings for each objective,
   removing duplicates. Evaluate this fixed shortlist once on validation.
4. Select one setting per objective using the same ranking rules. If both
   objectives choose the same settings, run that configuration only once.
5. Compare the frozen choice or choices with the full annotator on the fixed
   46-video ShuttleSet22 population: 3,327 rallies and 37,184 contacts.

The initial 128 configurations include three tighter starting points, changes
to one parameter at a time around the earlier integer settings, and a seeded
sample of joint settings. The seed is `20261017`.
[search.py](search.py) records every range. Counts and speeds use the
annotator's base-30 convention and are scaled once for each video's frame rate.
The old direction-angle and minimum-contact-speed rules have been retired;
their historical settings are not parameters of the current impulse detector.

Refine around the best three configurations, changing individual parameters
and the coupled pairs listed in `COUPLED_AXES`. Complete each round before
choosing its successors. Stop contact refinement when a round leaves the
leader unchanged, or after two rounds. Then run one local round for fully
correct rallies. A winner at a range edge warrants a documented training-only
extension before freezing the shortlist. Do not extend ranges after validation.
Pass `refine --ranges additions.json.gz` to add recorded values to the original
ranges. This file maps parameter names to lists of extra values. Each generated
refinement file records the resulting ranges as well as every configuration.
Use `probe --parameter NAME --values ... --results ... --output ...` to test
all supplied values around the best three settings for each objective. This
covers a missed value directly, without requiring several adjacent steps.
The wrist-distance threshold is measured in player bounding-box heights and
is not scaled with frame rate. Its unchanged default is 1.4.

## Scores and limits

Report contact and correct-player precision, recall and F1 at ±5 and ±10
frames, scaled from 30 fps. Count contacts across the whole video, including
false positives outside labelled rallies. Pool integer counts before calculating
rates; a short video does not receive the same weight as a long video.

A fully correct rally must be one discovered sequence containing exactly one
complete labelled rally, with every contact and player correct. Splitting or
merging rallies fails this rule. This differs from the MonoTrack report's
fixed-window rally score, which supplies the true windows to the scorer.
Use fixed-window results only as a separately labelled diagnostic.

Inference failures produce empty predictions and stay in the denominator.
Missing inputs, corrupt caches and unfinished jobs stop the run. Saved output
includes predictions, integer counts, failures and runtime for every video and
configuration, so scoring can be inspected without repeating inference.

The heuristic keeps its existing scene-specific net geometry and fitted
alternating player order. The full annotator uses its fitted bundle's geometry
policy. The comparison therefore measures what the current learned system
adds to the existing heuristic system, including that architectural difference.
The full annotator's base configuration was chosen over an optional veto after
both validation and ShuttleSet22 scores had been inspected. ShuttleSet22 cases
also helped motivate that veto. Its contact tree was fitted on all 40 original
ShuttleSet videos, so this bundle is used only for the test comparison here.
See the [model-selection report](../reports/model_selection.md). This new
heuristic search keeps ShuttleSet22 out of parameter selection.

## Run

Run from the repository root in an annotator environment. Set `PYTHONPATH=src:.`
if the package is not installed. Copy [paths.example.toml](paths.example.toml)
to an untracked location and fill in the prepared input paths. Use scene-based
court inputs for both datasets; rally-bounded court files reveal the true
boundaries through their scene cuts.

```bash
export PYTHONPATH=src:.
search_dir=local_scratch/heuristic_only_tune
mkdir -p "$search_dir"
cp experiments/annotator/heuristic_only_tune/paths.example.toml "$search_dir/paths.toml"

python -m experiments.annotator.heuristic_only_tune.search coarse \
  --count 128 --output "$search_dir/coarse.json.gz"

python -m experiments.annotator.heuristic_only_tune.run \
  --paths "$search_dir/paths.toml" --candidates "$search_dir/coarse.json.gz" \
  --split train --output "$search_dir/coarse" \
  --workers 26 --workers-file "$search_dir/workers"
```

Before the full search, use `--only` with a training video and a separate output
directory to measure memory and runtime. Smoke-test results cannot select
settings. Repeat the exact command to resume an interrupted run. The runner
checks source, inputs, labels and settings before reusing completed jobs.

For the next round:

```bash
python -m experiments.annotator.heuristic_only_tune.search refine \
  --results "$search_dir/coarse/summary.json.gz" \
  --objective contacts --parents 3 --output "$search_dir/refine_contacts.json.gz"
```

Run that candidate file on `--split train` in a fresh output directory. Supply
all completed training summaries to later `refine` and `shortlist` commands.
Use `--objective rallies` for the follow-up. `shortlist --results ... --output ...`
creates the only candidate format accepted by `--split validation`.
`select --results VALIDATION_SUMMARY --output SELECTED` then creates the format
accepted by `--split test`. Keep the shortlist and selected settings with the
final results. Worker logs and machine-specific configuration remain untracked.

## Check progress or change the worker limit

```bash
python -m experiments.annotator.heuristic_only_tune.status \
  --output "$search_dir/coarse" --watch

printf '18\n' > "$search_dir/workers.tmp" &&
mv "$search_dir/workers.tmp" "$search_dir/workers"
```

The status view shows completed, active, queued and failed jobs. Each active
job shows its stage and elapsed time. Stages that publish a real counter also
have a progress bar. Omit `--watch` for one snapshot.

The worker limit can be changed between 0 and 26. Zero pauses new work.
Lower limits let running jobs finish; higher limits admit more work from the
shared queue. Each worker uses one numerical-library thread. The expensive
player-tracking loop is sequential, so extra threads within one job would not
help that part. The runner caches inputs within workers and reuses fixed player
evidence between configurations.
