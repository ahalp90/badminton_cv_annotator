# Annotator experiments

The maintained annotation and retuning path is described in the
[auto-annotator guide](../../docs/annotator/README.md).

`heuristic_tuning/` retains earlier parameter sweeps and fixed-fixture scoring.
These compare heuristic variants; they are separate from normal model fitting.
`measurement.py` runs the fixed end-to-end court/annotation comparison and
requires `--annotator-models` pointing to a freshly fitted bundle.

The historical contact studies remain in `scratch/contact_det*` while the new
court inputs are retuned. Their selected feature, refinement and fitting code
now lives in `src/annotator`; their saved results describe the old court inputs.
They contain rejected variants and older model binaries, so begin with the
maintained guide when running or changing the annotator.

The [court geometry repair bundle](court_geometry_repair/README.md) contains
paired issue #148 evidence, saved-output checks and a pipeline reproduction recipe.

The fixed annotator CLI writes each successful or failed measurement to `runs/<UTC timestamp>/`.
Successful runs add `summary.json.gz` and `report.md`, then clean commit-candidate files in place.

New NumPy, JSON, and CSV artifacts are compressed as `.npy.xz`, `.json.gz`, and `.csv.gz`.
Git retains the complete cleaned run, so stage the timestamped directory without selecting files by hand.
Legacy uncompressed `.npy` artifacts remain ignored to prevent old large arrays from being staged accidentally.

To retry cleaning a completed run, install the operational tools and run:

```bash
uv sync --extra annotator-experiments
python -m experiments.annotator.records experiments/annotator/runs/<YYYYMMDD-HHMMSS>
```

An `rg` 15.1.0 executable already available on `PATH` also satisfies the ripgrep requirement.

The cleaner saves non-array files to `local_scratch/annotator_experiment_backups/` before any rewrite or deletion. It scans temporary decompressed copies of gzip text artifacts. A cleaned Git copy can omit a file which the historical manifest records as produced. Staging, committing and promotion remain manual.
