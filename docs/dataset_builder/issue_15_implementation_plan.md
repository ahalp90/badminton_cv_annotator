# Issue 15 Dataset Builder Implementation Plan

Status: planned

Prepared: 2026-08-08

Adversarial review: completed 2026-08-08; five findings incorporated

Issue: [#15, Wire the dataset builder and complete the end-to-end trial](https://github.com/ahalp90/badminton_cv_annotator/issues/15)

## Decision summary

Implement issue 15 as a thin, resumable coordinator around the existing
pipeline stages. Reuse the existing search, transcript, commentary, vision,
and annotator implementations. Add shared video metadata, whole-video
adapters, stage manifests, and a provisional rally-record assembler.

Visual selection and commentary availability are separate run contracts. A
missing transcript or unavailable commentary model must not remove a video
that the visual selection policy retained. Resume is allowed only when the
stage fingerprint, output integrity, and semantic validation all match.

The final engineered feature schema is outside this issue. Issues
[#18](https://github.com/ahalp90/badminton_cv_annotator/issues/18) and
[#22](https://github.com/ahalp90/badminton_cv_annotator/issues/22) own that
work.

## Current state

- The [README](../../README.md) says the
  search-to-dataset orchestrator is still being integrated.
- The general [rally CLI](../../src/annotator/rally/cli.py) calls `run_video`
  with `stop_after_segmentation=True`.
- The full [run_video](../../src/annotator/run_video.py) contract already
  accepts prepared shuttle, pose, timing, and court evidence.
- The [rally dataset contract](../rally_dataset_contract.md) assigns the
  assembler to section 2.3. It defers the physical format and final engineered
  feature table.
- Commentary pairing currently has a separate OpenCV FPS probe in
  [commentary_pairing.py](../../src/scraper/commentary_pairing.py).
- The validation overlay has the strictest current ffprobe implementation in
  [decode.py](../../src/annotator/validation_overlay/core/decode.py).
- The current shuttle CSV converter in
  [shuttle_extractor.py](../../src/bst_x/pipeline/shuttle_extractor.py) follows
  ShuttleSet clip naming and metadata assumptions.

## Workspace and baseline

- Branch: `issue-15-dataset-builder`
- Worktree: `cosc595/worktrees/issue-15-dataset-builder/`
- Starting commit: `ee6844f010cee76b701aa18eb43f84d05d05002a`
- Keep the implementation worklog outside the repository.
- Leave the existing untracked `annotation_videos/` directory unchanged.

The focused baseline passed 272 tests in 7.83 seconds. It covered FPS and
video metadata, scraper stages, commentary, court evidence, and `run_video`.

## Batch 1: Run contracts and canonical video metadata

Draft commit: `Add dataset-builder run contracts and video metadata`

### Files

- Add `src/dataset_builder/models.py`.
- Add `src/dataset_builder/manifest.py`.
- Add `src/annotator/video_metadata.py`.
- Add `tests/test_dataset_builder_manifest.py`.
- Update the current FPS and validation-overlay metadata tests.

### Changes

1. Extract the strict ffprobe logic from the validation-overlay namespace.
2. Keep `annotator.fps_constants.probe_fps` as a compatibility wrapper.
3. Record exact FPS, frame count, dimensions, and source path once per video.
4. Reject VFR and inconsistent metadata.
5. Create one immutable, non-empty `run_id` when a run starts. Reuse it only
   when resuming that run directory.
6. Add typed stage outcomes: `processed`, `skipped`, `excluded`, `failed`, and
   `unavailable`.
7. Define a stage fingerprint from the source commit, stage contract version,
   normalised effective configuration, resolved interpreter and version, model
   weight MD5 values, and persisted input MD5 values.
8. Record redacted commands, configuration, fingerprints, inputs, outputs,
   counts, elapsed time, semantic validation results, and failure reasons.
9. Record output MD5 values after atomic writes complete.
10. Update the run manifest atomically after every stage.
11. Reuse a stage only when its fingerprint matches, every output MD5 matches,
    and every semantic validation still passes. Otherwise rerun that stage and
    its dependants.

### Gate

- Existing FPS and validation-overlay behavior remains equal on current
  fixtures.
- Metadata rejects VFR, zero FPS, and conflicting frame counts.
- Manifest writes and resume-state reads round-trip exactly.
- An unchanged stage is reusable.
- Changing the source commit, effective configuration, interpreter, model
  weights, or input content invalidates the stage and its dependants.
- A missing, replaced, or corrupted output invalidates the stage even when its
  path and shape remain unchanged.

## Batch 2: Whole-video vision and full annotation

Draft commit: `Wire full-video extraction and annotation`

### Files

- Add `src/dataset_builder/vision.py`.
- Add `tests/test_dataset_builder_vision.py`.
- Extend `src/bst_x/pipeline/shuttle_extractor.py` with a whole-video adapter.

### Changes

1. Convert a whole-video TrackNet CSV into the shuttle array expected by the
   annotator.
2. Require the TrackNet `Frame` column to contain unique integers with exact
   coverage of `0..frame_count-1`.
3. Reindex validated TrackNet rows to frame order before creating the array.
   Reject gaps, duplicates, non-integral IDs, and out-of-range IDs.
4. Preserve string video IDs.
5. Run RTMLib pose extraction through its separate interpreter.
6. Build raw-cut and CourtKeyNet evidence with existing annotator functions.
7. Validate each frame-aligned array against the canonical frame count.
8. Reject mismatches instead of padding or truncating them.
9. Call the full `run_video` path with a caller-owned `RunCapture`.
10. Require both captured masks to be one-dimensional boolean arrays whose
    length equals `frame_count`.
11. Require `DeadMaskMode.REPLAY` for the version 0.1 dataset-builder path.
12. Persist the captured raw mask as `raw_replay_mask.npy.xz`. Commentary
    pairing consumes this mask because it applies its own duration filter.
13. Persist `definitive_exclusion_mask.npy.xz` as annotation provenance. Do not
    substitute it for the pairing replay mask.
14. Preserve every primitive `AnnotatorResult` field.
15. Store arrays as `.npy.xz` with `lzma.FORMAT_XZ` and preset 9.
16. Store structured results as `.json.gz`.
17. Keep TrackNet and RTMLib interpreter paths configurable.

### Gate

- Whole-video conversion supports non-numeric IDs.
- Shuffled TrackNet rows produce correctly reindexed output.
- Duplicate, missing, non-integral, and out-of-range TrackNet frame IDs fail.
- Shape, dtype, frame-index, and frame-count failures produce explicit stage
  outcomes.
- Full annotation reaches the same result contract as a direct `run_video`
  call on fixtures.
- Both captured masks survive compressed round-trip with exact values.
- A sustained replay inside a rally keeps the rally unpaired.
- Existing ShuttleSet conversion behavior remains covered.

## Batch 3: Primitive feature projection and record assembly

Draft commit: `Assemble provisional rally records`

### Files

- Add `src/dataset_builder/records.py`.
- Add `tests/test_dataset_builder_records.py`.
- Update commentary pairing to consume canonical video metadata.

### Changes

1. Derive stable primitives such as rally duration and accepted contact count.
2. Preserve raw and cleaned commentary with quality diagnostics.
3. Represent missing commentary as null with a stage outcome or reason.
4. Validate half-open rally intervals and composite-key uniqueness.
5. Emit one `rally-record/0.1` row per detected rally.
6. Write `rally_records.json.gz` and `run_manifest.json.gz`.
7. Reference large vision artefacts instead of embedding their arrays.
8. Keep assembly limited to validation and joining. It must not rerun or
   reinterpret producer stages.
9. Implement and test the complete logical mapping below. A missing source
   value remains null with its recorded provenance.

### Required field mapping

| Source | Required logical record content |
| --- | --- |
| Run manifest and canonical video metadata | `run_id`, source dataset, exact `video_id`, source reference, `fps`, `frame_count`, code version, configuration, integrity, and stage outcomes |
| `spans[rally_id]` | `rally_id`, `start_frame`, `end_frame`, `duration_frames`, and `duration_seconds` |
| `contacts` grouped by `rally_id` | Raw candidates with `contact_frame`, `proximity_ok`, `wrist_near`, and `suppressed` |
| `filtered_contacts` and `filtered_by_rally[rally_id]` | Accepted contacts and their zero-based `stroke_idx`; the two sources must agree exactly |
| `striker_halves[rally_id]` | Final-contact `striker_half` |
| `n_strokes_list[rally_id]` | `stroke_count`, equal to the accepted-contact count |
| `fitted_first_all[rally_id]` | Fitted first-stroke half, which is the server prediction |
| `next_servers[rally_id]` | Next-server half inferred from the following rally |
| `verdict_rows[rally_id]` | Verdict, verdict source, landing margin, line-margin flag, and net-margin flag |
| `landings[rally_id]` | Landing frame, normalised court position, court half, image-border flag, and net-ender flag |
| `geometric_verdict_rows[rally_id]` | Geometric verdict, geometric winner, agreement diagnostic, and mask-closed-window flag |
| `hit_height_by_frame` joined to accepted contacts | Per-contact ShuttleSet hit-height code |
| `hit_height_failures` grouped by `rally_id` | `stroke_idx`, contact frame, and failure reason |
| `RunCapture` mask artefacts | References to the raw replay and definitive exclusion masks with their integrity and stage configuration |
| Pair row and selected chunk sidecar | Nullable chunk ID and times, raw text, cleaned text, alternatives, cleaning diagnostics, provenance, and missing-commentary reason |

### Gate

- Conflicting FPS or frame counts stop assembly.
- Invalid spans and duplicate keys stop assembly.
- Rallies without commentary remain present.
- The assembled row count equals the detected rally count.
- Exact expected-record fixtures cover accepted and rejected contacts,
  unresolved striker and landing values, hit-height failures, and missing
  commentary.
- Tests assert every mapped primitive rather than checking only keys and row
  counts.

## Batch 4: One-command coordinator

Draft commit: `Add the end-to-end dataset-builder command`

### Files

- Add `src/dataset_builder/__init__.py`.
- Add `src/dataset_builder/__main__.py`.
- Add `src/dataset_builder/cli.py`.
- Add `src/dataset_builder/selection.py`.
- Add `tests/test_dataset_builder_cli.py`.
- Add `tests/test_dataset_builder_selection.py`.
- Update `src/scraper/search_index.py` to accept a per-run search count while
  preserving its current default.
- Update `src/scraper/download_scraped_videos.py` to accept an explicit set of
  selected video IDs while preserving its current `keep == 'True'` default.
- Add `configs/dataset_builder/trial.toml` without secrets or machine-specific
  paths.

### Command shape

```bash
PYTHONPATH=src uv run python -m dataset_builder run \
  --config configs/dataset_builder/trial.toml \
  --run-dir /path/to/run
```

### Stage order

1. Search with the configured terms and result count.
2. Attempt transcript acquisition.
3. Attempt relevance triage for candidates with transcripts.
4. Resolve and persist visual selection independently from commentary status.
5. Apply the video cap and download only selected videos.
6. Probe video metadata.
7. Clean available commentary.
8. Extract shuttle, pose, and court evidence.
9. Run full auto-annotation and capture both masks.
10. Pair available commentary with the raw replay mask.
11. Project primitive features.
12. Assemble records.
13. Write the final report.

### Visual selection contract

- Keep the existing `candidates.csv` schema and its commentary-oriented `keep`
  field unchanged.
- Write the run-scoped selection decision to `selected_videos.csv.gz`.
- Record `visual_selected`, `selection_source`, `selection_reason`, source
  order, and `commentary_status` for every candidate.
- Select `keep == 'True'` match candidates first in persisted candidate order.
- Never select a candidate with `keep == 'False'` through fallback.
- When a transcript or triage result is unavailable, allow fallback only for a
  match candidate whose doubles, duration, and upload-date suspicion flags are
  all false.
- Order fallback candidates with an acquired transcript first. Preserve
  candidate order inside each group.
- Apply `max_videos` to the resolved selection before creating download tasks.
- Pass selected IDs explicitly to the downloader. Do not rewrite `keep` to
  force a visual selection.
- Use commentary status and reason values that distinguish ineligible video,
  unavailable transcript, unavailable triage, no retained chunk, no pair, and
  failed commentary processing.

### Runtime behavior

- Preflight executables, model weights, and named environment variables for
  required stages. Never record secret values.
- Refuse an external run when tracked files differ from the recorded source
  commit. This keeps the code identity truthful.
- Treat missing optional commentary dependencies as `unavailable`. They must
  not fail visual selection, download, or later visual stages.
- Set `BADMINTON_SCRAPE_DIR` to the run workspace for scraper subprocesses.
- Apply the Batch 1 fingerprint and semantic-validation rules before skipping
  any stage.
- Stop assembly after a required-stage failure.
- Preserve individual video exclusions and continue with remaining videos.
- Let transcript acquisition, relevance triage, commentary cleaning, and
  commentary pairing report `unavailable` without dropping selected visual
  records.

### Gate

- Unit tests replace network, model, and subprocess boundaries with fixtures.
- A successful fixture run visits every stage in order.
- Required-stage failures stop later dependent stages.
- An unavailable transcript or triage endpoint still produces the bounded
  visual selection through the documented fallback.
- A triage-rejected candidate never enters the fallback selection.
- The selected-video cap is enforced before the downloader is called.
- Resume skips only outputs with matching fingerprints, output integrity, and
  semantic validation.

## Batch 5: External end-to-end trial

Draft commit: `Record the issue 15 end-to-end trial`

Run the trial only after local tests, lint, and type checks pass. Use a machine
with the required CUDA models, network access, and commentary credentials.

### Trial scope

- Use one professional-singles search term.
- Limit discovery to five results.
- Resolve visual selection with the documented triage-first and
  metadata-fallback policy.
- Process at most two selected videos. Enforce this before download.
- Prefer videos with available captions.
- Persist every search candidate, selection decision, and deterministic order.
- Record exclusions when fewer than two videos satisfy the requirements.
- Resume the completed run once to prove that every reusable stage is skipped
  for the recorded reason.

### Trial report

Add a concise tracked report containing:

- Source commit and exact command.
- Stage fingerprints, outcomes, and counts.
- Selected video IDs and canonical metadata.
- Rally and accepted-contact counts.
- Commentary coverage and missing reasons.
- Exclusions and failures.
- Paths and MD5 values for external artefacts.
- Raw replay-mask and definitive-exclusion-mask references.
- Resume results for every stage.
- Final acceptance result.

Keep downloaded videos and large array artefacts outside Git.

## Final acceptance gates

- Every selected video has finite CFR FPS and a positive frame count.
- Every TrackNet frame index is an integer with exact ordered coverage after
  validated reindexing.
- Every frame-aligned array matches that frame count.
- Both captured masks are boolean, frame-aligned, and integrity checked.
- Every rally interval is valid and uniquely keyed.
- Every detected rally produces one assembled record.
- Every required `AnnotatorResult` primitive is present or explicitly null in
  that record according to the field mapping.
- Every stage has an explicit outcome.
- Commentary unavailability does not remove a selected visual record.
- An unchanged resume preserves identical records and skips reusable stages.
- Changing an input, model, configuration, interpreter, or source commit
  invalidates the affected stage and its dependants.
- Ruff passes on changed Python files.
- Pyrefly passes for the configured project scope.
- The full pytest suite passes.
- The external trial reaches assembled records and produces its report.

## Risks

### High risk

- Frame alignment across independently produced vision artefacts.
- Passing complete court and pose evidence into the full annotator path.
- Resuming from stale or incompatible stage outputs.
- Joining commentary against incorrect timing metadata.

### Medium risk

- Coordinating separate Python environments.
- Distinguishing exclusions from failed stages.
- Keeping trial configuration reproducible without recording secrets.

### Low risk

- CLI presentation and report formatting.
- Adding the provisional compressed JSON output.

## Out of scope

- Existing ShuttleSet classifier pipeline behavior.
- Annotator tuning and precision rules from issue 16.
- Feature benchmarking from issue 17.
- Engineered feature formulas and schema v1 from issues 18 and 22.
- Parallel extraction optimization from issue 23.
- Amateur court detection from issue 24.
- VFR support.
- Mass scraping and dataset release from issue 19.
- Inpaint, serve, replay, and stroke-classification redesign.
- The existing untracked `annotation_videos/` directory.

## Proposed commit sequence

1. `Add dataset-builder run contracts and video metadata`
2. `Wire full-video extraction and annotation`
3. `Assemble provisional rally records`
4. `Add the end-to-end dataset-builder command`
5. `Record the issue 15 end-to-end trial`
