# ShuttleSet fixed-input production runbook

## Purpose

This path processes known local ShuttleSet videos without search or download.
It uses the production metadata, TrackNet, pose, court, annotation, commentary
pairing, primitive-projection, assembly, and reporting stages. The normal
search and download path is unchanged.

The tracked sample selects `sset_01` at 25 FPS and `sset_21` at 30 FPS. Edit
`fixed_sources.video_ids` and `run.max_videos` together for another bounded
run. Do not add a source until its exact identity is recorded in the versioned
source manifest.

## Inputs and command

Set `SHUTTLESET_SOURCE_ROOT` to the directory containing the exact MP4
basenames in
`configs/dataset_builder/shuttleset_sources_v1.toml`. The tracked ground truth
must remain under `training/data/shuttleset/annotations`. Configure the
TrackNet and pose interpreter environment variables required by
`configs/dataset_builder/shuttleset_fixed.toml`, then run:

```bash
PYTHONPATH=src python -m dataset_builder run \
  --config configs/dataset_builder/shuttleset_fixed.toml \
  --run-dir /absolute/path/to/run
```

The fixed-input CPU preflight completes before any GPU stage. It rejects
unknown or duplicate requested IDs, unavailable or ineligible sources,
missing or symlinked files, unsupported containers, digest changes,
ground-truth mapping changes, variable frame rate, and FPS or frame-count
mismatches.

## Outputs

The run directory contains `run_manifest.json.gz`, `rally_records.json.gz`,
`dataset_builder_report.json.gz`, and `selected_videos.csv.gz`. The fixed
acquisition snapshot is under `stages/download/fixed_acquisition.json.gz`.
Each video has production stage outputs below `stages/<stage>/<video_id>/` and
a reloadable index at:

```text
stages/artifact_index/<video_id>/video_artifact_index.json.gz
```

The index pins the source and manifest identity, canonical metadata, stage
configuration, interpreter and model identities, outcomes, reasons, output
integrity, annotations, masks, and primitive projections.

## Annotation replay

After Issue #96 finalizes the 2.5 annotation configuration, rerun annotation
and primitive projection from the pinned expensive vision artifacts with:

```bash
PYTHONPATH=src python -m dataset_builder replay \
  --config configs/dataset_builder/shuttleset_fixed.toml \
  --run-dir /absolute/path/to/the-existing-run
```

Replay uses the same production stage plans. It validates the fixed source,
run manifest, models, metadata, TrackNet input, shuttle, pose, court, and
per-video index before annotation starts. It does not run TrackNet, pose, or
court inference. A changed source commit or annotation configuration
invalidates annotation and its downstream stages.

## Resume and recovery

Rerunning `run` with the same directory resumes from validated artifacts.
Changed fingerprints or failed integrity checks invalidate the affected stage
and its dependants. A corrupt annotation or mask can be rebuilt with `replay`.
A missing or corrupt expensive vision artifact stops replay before annotation;
recover it by resuming the full `run` command. Optional unavailable stages are
reused unless `--retry-unavailable` is supplied. Partial failures retain the
completed per-video indexes, while any failed index or required publication
causes a nonzero command exit.
