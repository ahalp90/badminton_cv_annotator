# Reproducing the experiments

Saved-result checks, annotation runs, model fitting and regenerated vision
extracts each test a different part of the pipeline. The experiment reports identify which of these
produced each result.

The commands below run from the repository root. A clone contains reports, selected
models and small evidence bundles. Full source videos, pose/shuttle extracts
and some training caches require separate inputs.

## Available checks

| Goal | Recipe | Additional inputs |
|---|---|---|
| Annotate one video with the selected model | [Quickstart](../../docs/annotator/quickstart.md) | Saved metadata, court, shuttle and pose stages |
| Fit and evaluate a model on a new set of labelled videos | [General refit guide](../../docs/annotator/retuning.md) | Saved stages, complete contact labels and a grouped split manifest |
| Verify the court-repair result without videos or a GPU | [Saved-output checks](reports/court_repair.md#check-the-saved-results) | Project scoring dependencies; evidence is included |
| Repeat the selected model's base-versus-veto comparison | [New-court refit recipe](reports/model_selection.md#reproduction) | Recorded development/test extracts, labels and court results |
| Investigate historical refit differences | [Old-court runner](reports/refit_regression.md#reproduction) | Historical extracts, scores and fitted models named in the recipe |
| Test court fitting or sampling alternatives | [Court fitting](reports/court_fitting.md), [frame sampling](reports/court_sampling.md) | Images or videos and the dependencies for the chosen detector |

The refit comparison calls its optional unreliable-shuttle/no-player rejection
rule the **veto**. The selected **base** model leaves this rule off.

The specialist recipes retain the machine paths and environments used for the
original runs where these matter. Paths such as `/scratch/ahalperi/...`
refer to the original researcher's inputs and need local equivalents. Historical CourtKeyNet baselines also require the old detector
code and weights; the current production court detector has a separate
[guide](../../src/court_detector/README.md).

## Check the early rule-based measurements

The calibration runner uses the published
`shuttleset-annotator-heuristic-reference-v1` fixture release.
`ANNOTATOR_FIXTURES_ROOT` identifies the locally extracted fixture root:

```bash
export ANNOTATOR_FIXTURES_ROOT=/path/to/extracted-fixtures
PYTHONPATH=.:src uv run python \
  -m experiments.annotator.heuristic_tuning.run_cli \
  --out /tmp/annotator-calibration
```

This checks the early calibration chain. The normal annotation command runs
the current fitted system. The [July measurement](runs/20260730-041328/report.md)
records the eight original combinations of court source, video and shuttle
sampling. Its detector names and defaults describe that historical run.

The later [player-distance comparison](reports/player_assignment.md) uses the
same fixture release plus pinned pose extracts. The [boundary-search report](reports/boundary_search.md)
names the analyser and CSV needed to repeat its saved-output calculation.

## Record a new end-to-end measurement

`measurement.py` compares the fixed calibration configurations using a supplied
annotator model. It is useful when checking those fixtures; the general training
and evaluation commands are the route for a new video population.

```bash
uv sync --extra annotator-experiments
PYTHONPATH=.:src uv run python -m experiments.annotator.measurement \
  --manifest /path/to/input-manifest.json \
  --annotator-models /path/to/model-bundle \
  --court-python /path/to/court-environment/bin/python \
  --deeplsd-source /path/to/DeepLSD \
  --device cpu
```

`parse_input_manifest()` in [`measurement.py`](measurement.py) defines the
fixture manifest. Schema version 3 requires `videos`, `track_overrides`,
`inpaint_codes`, `deeplsd_weights` and `producers`. Each file record pins a
relative path, root (`fixtures` or `repo`) and MD5 digest. The manifest covers
exactly `sset_01`, `sset_15` and `sset_21`, plus the stride-1 shuttle variant for
`sset_01`.
The runner records success or failure under `runs/<UTC timestamp>/`. Successful
runs write `summary.json.gz` and `report.md`, then package the result files.
New arrays, JSON and CSV records use `.npy.xz`, `.json.gz` and `.csv.gz`.

To retry packaging an existing run:

```bash
PYTHONPATH=.:src uv run python -m experiments.annotator.records \
  experiments/annotator/runs/<YYYYMMDD-HHMMSS>
```

The cleaner backs up non-array files under
`local_scratch/annotator_experiment_backups/` before rewriting or removing them.
It uses temporary decompressed copies to inspect gzip text files. A cleaned
record may omit an intermediate file listed in its original manifest.

## Result records

The useful record of an experiment includes its question, fixed inputs, changed
setting, evaluation population, gained and lost rallies, and the reason for
retaining or rejecting the approach. [The development narrative](development.md)
connects that conclusion to the rest of the build.

Detailed reports hold rerun commands, input provenance and small result tables.
Model versions and training-example order matter: the refit investigation found
that the same examples and random seed can still produce different trees when
their order changes. Large caches and temporary logs remain outside Git.
