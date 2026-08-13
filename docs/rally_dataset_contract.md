# Provisional rally dataset contract

| Item | Value |
| --- | --- |
| Collection contract | `rally-record-collection/0.2` |
| Row contract | `rally-record/0.2` |
| Status | Provisional |
| Updated | 12 August 2026 |

This document defines the implemented boundary where the acquisition,
annotator, and commentary lanes meet. The dataset builder writes a compressed
JSON collection and validates it against an immutable input-manifest snapshot.

Version 0.2 fixes the provisional collection layout, record identity, timing
rules, required provenance, and primitive evidence. It does not freeze the
final engineered feature table or a stable version 1 storage format. Issue
[#18](https://github.com/ahalp90/badminton_cv_annotator/issues/18) owns the
final feature columns, column types, reliability metadata, and version 1
schema.

## Record identity

One logical record represents one detected rally. Its stable composite key is:

```text
(run_id, source_dataset, video_id, rally_id)
```

- `run_id` is a non-empty string that identifies the exact immutable
  extraction, annotation, and assembly run.
- `source_dataset` is the non-empty dataset label from `sources.toml`, or an
  equivalent explicit label for externally supplied inputs.
- `video_id` is a non-empty string. Readers and writers must preserve it
  exactly and must never coerce it through a numeric type. For example, `0012`
  and `12` are different identifiers.
- `rally_id` is a zero-based integer. It is the rally span's list position and
  is unique within one `(run_id, source_dataset, video_id)` group.

File basenames and paths are provenance. They are not record keys because a
file may be renamed or copied without changing the source video identity.
List-position rally IDs must not be used for cross-run joins because a changed
configuration or model can insert or remove an earlier detected span. A future
cross-run rally identity needs a separate matching rule.

## Time and interval rules

- Frame numbers are zero-based indices on the whole source video.
- Every frame interval is half-open: `[start_frame, end_frame)`. The start
  frame is included and the end frame is excluded.
- A rally must satisfy
  `0 <= start_frame < end_frame <= frame_count`.
- `frame_count` is the positive integer count from the same decoded video used
  to produce the frame-aligned evidence.
- `fps` is the finite, positive video frame rate used by the run. It must be
  stored per video, even when every video in a trial shares one value.
- Frame-aligned arrays for a video must have `frame_count` rows.
- Commentary chunk times are seconds on the source-video timeline. The
  pairing stage converts a rally end to seconds with `end_frame / fps`.

The assembler must reject conflicting frame counts or frame rates. It must not
silently rescale evidence from one timing basis onto another.

## Required provenance

Each assembled dataset must retain, directly or through a referenced run
manifest, enough information to trace every rally back to its source and
extraction run.

| Provenance | Required meaning | Current source |
| --- | --- | --- |
| Source dataset | Dataset label that namespaces video identifiers | Top-level `dataset` in `sources.toml` |
| Source video | Manifest basename, exact string `video_id`, title, URL, and commentary eligibility | Per-video entry in `sources.toml` |
| Video timing | Probed `fps` and `frame_count` for the decoded source used by every lane | Video probe and frame-aligned extraction outputs |
| Code version | Git commit used for extraction, annotation, pairing, and assembly | Collection envelope and input manifest |
| Run identity | Immutable non-empty `run_id` for the exact trial or extraction run | Collection envelope and input manifest |
| Stage configuration | Model, weights, resolution, mode, thresholds, and other settings that affect an output | Input manifest |
| Input artefacts | Paths or stable references for shuttle, pose, court, mask, transcript, and commentary inputs | Existing stage outputs |
| Integrity | Digest for persisted inputs and outputs when a digest is available | Input manifest |
| Stage outcome | Processed, skipped, excluded, failed, or unavailable, with a reason | Input manifest and batch report |

An input supplied outside the acquisition lane needs an equivalent source
reference. A local path alone is not enough provenance because it can change
or disappear.

## Persisted collection layout

`rally_records.json.gz` contains one `rally-record-collection/0.2` envelope.
The envelope stores `run_id`, `run_manifest`, `input_manifest_sha256`, the
complete immutable `input_manifest` snapshot, `code_version`, the redacted
assembly configuration, an ordered `sources` list, and `records`.

Each source occurs once. A source contains `source_dataset`, `video_id`, the
source reference, canonical video metadata, and the manifest stage that owns
the annotation masks. The referenced mask stage must have a reusable outcome
and both the raw-replay and definitive-exclusion mask outputs.

Each `rally-record/0.2` row contains only `schema`, `key`, `rally`, `contacts`,
`outcomes`, and `commentary`. Rows follow source-list order and rally IDs are
contiguous from zero within each source. The composite key retains `run_id`,
`source_dataset`, and `video_id`, so source identity does not depend on the
source-list position.

The loader verifies the manifest digest, code version, redaction, source and
row structure, ordering, mask references, and all primitive relationships. It
also requires the live run manifest to preserve and extend the embedded input
snapshot. The manifest remains the single source of truth for stage
configuration, integrity records, outcomes, and artifact paths.

## Primitive rally evidence

Version 0.2 preserves the values already exposed by `AnnotatorResult` and the
rally CSV boundary. These are primitive pipeline outputs. They remain
predictions or diagnostics unless a separate ground-truth source is joined.

### Rally span

Each record contains `start_frame` and `end_frame` under the half-open rule.
The rally CSV currently stores these with `video_id` and `rally_id`.

### Contacts

The record preserves two distinct contact views:

- raw contact candidates with `contact_frame`, nullable `proximity_ok`,
  nullable `wrist_near`, and nullable `suppressed` values;
- accepted contact frames after the current wrist, suppression, and definitive
  exclusion filters, in ascending order within the rally.

For accepted contacts, `stroke_idx` is the zero-based position within that
rally's accepted-contact list. Raw candidates and accepted contacts must not
be merged because the gate evidence explains why a candidate was retained or
removed.

### Rally-level derived values

The assembler must preserve these values when the annotator produces them:

- final-contact `striker_half`;
- `stroke_count`;
- fitted first-stroke half, which is the predicted server for this rally;
- next-server half inferred from the following rally;
- winner verdict, verdict source, landing margin, and line and net margin
  flags;
- landing frame, normalised court position, court half, image-border flag,
  and net-ender flag;
- geometric verdict, geometric winner, agreement diagnostic, and whether an
  exclusion mask closed the landing window;
- per-contact ShuttleSet hit-height code, plus the failure reason when the
  shuttle was not visible at the contact frame.

Court halves use the annotator's `Top` and `Bot` values. Verdicts use `won` and
`lost`, and verdict sources use `next_server`, `landing_geometry`, and
`net_rule`. Normalised landing coordinates use the doubles outline as the unit
square. Hit-height code `1` means above the net-band centre, while code `2`
means at or below it. The existing source enums remain the authority until
issue #18 freezes the external representation.

## Commentary evidence

Every validated rally remains in the dataset whether commentary is paired or
absent. When a chunk is paired, the logical record must preserve:

- `chunk_id`, commentary start, and commentary end in seconds;
- raw transcript `text`;
- cleaned `text_clean`;
- `alt_phrasings` when generated;
- `bert_f1` and `clean_pass` when scored;
- transcript and cleaning provenance, including the method and configuration.

The pairing stage currently writes the rally key, rally frame interval,
`chunk_id`, and commentary times. The assembler will join the selected chunk
sidecar to add its text and cleaning fields. A pair is a deterministic
time-window association. It does not assert that the commentary describes the
rally correctly.

## Missing values

The logical record uses a real null for an unavailable or unresolved value.
Current CSV stage boundaries use an empty field for that case. The future
assembler must normalise those empty fields to null.

- `false`, `0`, an empty list, and an empty string remain valid observed
  values. They must not stand in for missing data.
- Nullable contact gate values mean the gate was not measured. They do not
  mean that the gate failed.
- An empty accepted-contact list means the rally has no accepted contact. It
  differs from a contact stage that failed or did not run.
- Missing commentary must retain enough stage outcome information to
  distinguish ineligible video, unavailable transcript, no retained chunk,
  no time-window pair, and a failed commentary stage.
- A derived row or value that its source omits must remain null. Existing
  booleans emitted by the annotator keep their emitted values.

The assembler carries commentary outcome details beside each commentary value.
Other stage outcomes and failure reasons remain in the referenced manifest. An
absent stage must not become a negative label.

## Reliability

Reliability follows the origin of each value:

| Evidence group | Interpretation |
| --- | --- |
| Source identity and timing | Observed metadata after manifest and video validation |
| Rally spans and contacts | Heuristic predictions from frame-aligned vision evidence |
| Striker and server halves | Heuristic values derived from accepted contacts and player tracks |
| Landing, winner, and hit height | Experimental heuristic values; current measurement identifies these as weak |
| Raw commentary | Source transcript that may include transcription errors |
| Cleaned commentary and alternatives | Generated text derived from the raw transcript |
| `bert_f1` and `clean_pass` | Cleaning diagnostics, not calibrated truth probabilities |
| Rally-commentary pair | Mechanical timing association, not a semantic label |

Ground truth, manual corrections, and model predictions must carry distinct
provenance. A later stage may filter on a diagnostic, but it must preserve the
underlying primitive value and must not relabel a prediction as observed
truth.

Issue #18 will decide the final per-column reliability representation. Until
then, consumers must use this origin table and the recorded stage provenance.

## Stage ownership and assembly boundary

| Stage owner | Inputs | Outputs owned by the stage |
| --- | --- | --- |
| Acquisition and download | Search candidates and source metadata | Source video, `sources.toml`, commentary eligibility, download outcome |
| Vision extraction | Source video and model configuration | Frame-aligned shuttle, pose, court, and mask evidence, plus extraction provenance |
| Rally annotator | Vision evidence, `fps`, configuration | Rally spans, raw and accepted contacts, and rally-level derived values |
| Commentary triage and cleaning | Source transcript, video timing, cleaning configuration | Timestamped chunks, raw and cleaned text, alternatives, and cleaning diagnostics |
| Commentary pairing | Rally spans, chunks, replay mask, `fps`, source manifest | One pair row per rally, with nullable chunk and commentary times |
| Rally-record assembler | Validated outputs from all earlier stages and their run metadata | A normalized collection keyed by `(run_id, source_dataset, video_id, rally_id)` plus dataset-level provenance |

The assembler owns validation and joining. It must not rerun extraction,
change rally boundaries, choose different contacts, or reinterpret a stage's
missing value. It may report a failed validation and leave the run incomplete.

Existing producer stage files remain their current interfaces. Version 0.2 is
stored as compressed JSON. It remains provisional and has no compatibility
loader for earlier drafts.

## Deferred schema decisions

Issue [#18](https://github.com/ahalp90/badminton_cv_annotator/issues/18)
will define and freeze:

- engineered feature formulas and feature names;
- final flat or nested column layout and physical storage format;
- final column types and nullable encodings;
- feature-specific reliability fields;
- ShuttleSet feature retention decisions;
- model-ready tensor or sequence representations; and
- the first stable version 1 schema.

Those choices may add derived fields. They must keep the version 0.2 identity,
timing, provenance, and primitive evidence rules or record a new contract
version with an explicit migration.
