# Annotator experiment runs

The fixed annotator CLI writes each successful or failed measurement to `runs/<UTC timestamp>/`.
Successful runs add `summary.json` and `report.md`, then clean small commit-candidate files in place.

NPY masks and arrays remain in the run directory but Git ignores them. Copy or archive them before a manual Git commit if they matter.

To retry cleaning a completed run, install the operational tools and run:

```bash
uv sync --extra annotator-experiments
python -m annotator.experiment_records experiments/annotator/runs/<YYYYMMDD-HHMMSS>
```

An `rg` 15.1.0 executable already available on `PATH` also satisfies the ripgrep requirement.

The cleaner saves non-NPY files to `local_scratch/annotator_experiment_backups/` before any rewrite or deletion. A cleaned Git copy can omit a file which the historical manifest records as produced. Staging, committing and promotion remain manual.
