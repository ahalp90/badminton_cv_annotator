# Annotator experiment runs

The fixed annotator CLI writes each successful or failed measurement to `runs/<UTC timestamp>/`.
Successful runs add `summary.json.gz` and `report.md`, then clean commit-candidate files in place.

New NumPy, JSON, and CSV artifacts are compressed as `.npy.xz`, `.json.gz`, and `.csv.gz`.
Git retains the complete cleaned run, so stage the timestamped directory without selecting files by hand.
Legacy uncompressed `.npy` artifacts remain ignored to prevent old large arrays from being staged accidentally.

To retry cleaning a completed run, install the operational tools and run:

```bash
uv sync --extra annotator-experiments
python -m annotator.experiment_records experiments/annotator/runs/<YYYYMMDD-HHMMSS>
```

An `rg` 15.1.0 executable already available on `PATH` also satisfies the ripgrep requirement.

The cleaner saves non-array files to `local_scratch/annotator_experiment_backups/` before any rewrite or deletion. It scans temporary decompressed copies of gzip text artifacts. A cleaned Git copy can omit a file which the historical manifest records as produced. Staging, committing and promotion remain manual.
