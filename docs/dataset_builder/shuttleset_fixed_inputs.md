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

## Full-corpus production operation

Use the versioned source manifest as the only source-identity and eligibility
authority. A full-corpus configuration must list every manifest entry marked
eligible, preserve every entry marked ineligible, and set `run.max_videos` to
the eligible count. Do not add another exclusion because a stage fails.

Before the first launch, record the following values outside the clean source
checkout:

- the clean source commit and effective configuration identity;
- the source-manifest identity, eligible IDs, and all exclusions;
- each source path, digest, ground-truth mapping, FPS, and canonical frame
  count;
- every model, interpreter, wrapper, cache-file, and artifact-root identity;
- the host, GPU environment, available storage, and launch time.

Run the fixed-input preflight for the complete selection before starting a GPU
stage. It must validate all sources and ground-truth mappings in one call so
duplicate paths or identities cannot pass as independent checks. Confirm that
the exact launch environment also passes `DefaultPipelineRuntime.preflight()`.
Start the normal `dataset_builder run` command only after both gates pass.

Monitor the process, GPU, storage, run manifest, per-video stage records, and
production log. If a process is interrupted or a stage fails, keep the run
directory and rerun the same command with the same configuration. Resume
fingerprints and integrity checks decide which stages can be reused. Do not
copy artifacts between video directories, edit a stage record, or mark work
complete by hand. This preserves successful sibling artifacts when one video
fails.

After the run succeeds, run the same command once more as a no-op validation.
Reload every per-video artifact index with full source, model, configuration,
and output-integrity validation. Reconcile the index metadata and array lengths
for TrackNet input, shuttle outputs and masks, pose outputs, court masks,
annotations, commentary pairing, and primitive projections against the
canonical frame count.

The corpus handoff must state the exact artifact-index identities and artifact
paths that the downstream consumer may read. It must also retain every failed,
unavailable, and excluded video with its reason, plus stage counts, timings,
retries, and integrity results. Keep the run directory and expensive vision
artifacts pinned so `replay` can rebuild annotation and projection later.

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
