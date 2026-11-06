# Historical ShuttleSet benchmarks

These runners reproduce pinned ShuttleSet feature and commentary comparisons.
They are separate from the current annotator training/evaluation command.

- `shuttleset_benchmark.py`: compare saved production rally records with labels.
- `shuttleset_features.py`: historical feature prototypes; production formulae
  are in `dataset_builder.features`.
- `shuttleset22_features.py`: historical feature comparison targeting obsolete
  court-evidence artefacts. Its pinned run is not runnable with current court
  loaders without rebuilding the evidence and updating the experiment inputs.
- `commentary_benchmark.py` and `commentary_benchmark_inputs.py`: commentary
  comparisons and their pinned corpus inputs.

Run modules from the repository root, for example:
`PYTHONPATH=src python -m experiments.annotator.shuttleset_benchmarks.commentary_benchmark --help`.
Production retimed-commentary loading is in `dataset_builder.commentary_inputs`.
