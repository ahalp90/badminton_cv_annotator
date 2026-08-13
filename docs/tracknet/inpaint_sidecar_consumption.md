# Inpaint sidecar: consumer state and open work

The fill-mask sidecar shipped in `9475036`. The producer contract is
`docs/tracknet/inpaint_sidecar.md`; that document is authoritative for the
JSON schema, span semantics, and settled boundary choices. This note
records what the annotator, scraper, and stroke-classifier consumers do
with the flag today, and what is still owed.

## What ships today

The writer runs inside every path that produces shuttle CSVs, so from
`9475036` onward every fresh TrackNetV3 extraction leaves a
`{video_stem}_stride{N}_inpaint_mask.json.gz` beside its `_ball.csv`. The
writer covers standalone `predict.py` and `batch_predict.py`, which is driven
by `src/bst_x/pipeline/shuttle_extractor.py`.

On the three whole-video reference tracks, the raw sidecar fill fraction
is **51.87%, 53.33%, and 45.96%** of frames (`docs/tracknet/inpaint_sidecar.md`
§ Verification record). The fabrication investigation graded a subset of
those frames as PROVEN invention (~34–37%); the sidecar's raw mask counts
every filled gap, provable or not. The sidecar figure is therefore the
top-line model-filled-content measure and provenance signal. The graded
subset is the narrower evidence for proven fabrication.

## Who reads the sidecar today

The dataset builder reads the sidecar through
`dataset_builder.shuttle_evidence`. It validates the producer identity and
expands the spans for quality measurement. It also derives recurrence-guard
codes from the final normalised track. Both forms of evidence are persisted
in the run manifest and checked again on resume.

The sidecar remains source provenance. Its fill mask does not drive event
rejection because every filled coordinate is not proven wrong. The derived
guard codes are the authoritative rejection evidence in the dataset builder.

## The event-mask boundary in the annotator

The annotator has an event-mask seam for frame-aligned evidence.

`annotator.calibration.gt_scoring.build_run_video_inputs` calls
`annotator.inpaint_guard.grade_track` on the loaded shuttle track and
passes the resulting per-frame codes into `run_video` as `inpaint_codes`
(`src/annotator/calibration/gt_scoring.py:409`,
`src/annotator/run_video.py:189`). `run_video._build_shuttle_hallucination_mask`
adapts those grades into a single boolean mask read by the downstream event
rules and enforces that `inpaint_codes` and an externally supplied mask
are mutually exclusive (`src/annotator/run_video.py:26-41`). The grades are
the codes described in `evidence/inpaint_fabrications_20260722/detector_options.md`
(0 clean, 1 fabricated / proven, 2 flat / suspect, 3 degraded); their
implementation is `src/annotator/inpaint_guard.py`.

The dataset builder now follows the same recurrence-grade boundary. Its
shuttle stage calls `grade_track`, persists the codes and diagnostics, and
passes the codes into production annotation as `inpaint_codes`. It also
writes `shuttle_quality.json.gz`, which records fill counts, guard-code
counts, and their intersection.

Other scraper and stroke-classifier `run_video` callers still receive no
event mask. They need their own evidence and compatibility gate before
adopting the dataset-builder policy.

## Open consumer work

- **Assess the remaining production lanes.** Scraper and stroke-classifier
  callers need separate compatibility evidence before they adopt recurrence
  grades. A raw sidecar fill mask must remain measurement evidence.
- **Old-cache regenerate versus adapt.** Existing whole-video reference
  npys pre-date the sidecar and have no companion JSON. Two options:
  regenerate the reference tracks under the sidecar-writing tip and
  re-pin any downstream reference; or adapt consumers to run the inpaint
  guard on the loaded track when a sidecar is absent (today's calibration
  behaviour). No ruling yet; option choice affects re-pin scope.
- **Per-rule consumer policy** for the contact gate, landing search, and
  lost-shuttle guard. The fabrications investigation sketched candidate
  policies (`evidence/inpaint_fabrications_20260722/inpaint_fabrications_investigation.md`
  § Proposed fix, Part 2); those sketches are historical proposals, not
  a shipped contract.

## Evidence pointers

- **Mechanism, measurements, and detector options:**
  `docs/tracknet/evidence/inpaint_fabrications_20260722/inpaint_fabrications_investigation.md`
  and `.../detector_options.md`.
- **Code-level source trace with citations:**
  `.../c11_landing_bisect/inpaint_source_findings.md`.
- **Historical write-out recipe (pre-shipping, superseded):**
  `.../c11_landing_bisect/inpaint_flag_writeout_recipe.md`. Its file:line
  anchors pin the pre-shipping tree; treat the recipe as design
  archaeology, not a current build guide.
- **Landing collapse plain-language report** (why the guard matters
  downstream): `.../c11_landing_bisect/c11_landing_report.md` and the
  companion `findings.txt` ledger.
- **Machine artefacts** (NumPy fill / recurrence masks per stride,
  generated ShuttleTrack CSV, sidecar manifest JSON):
  `.../stride1_retrack/`.

## Related tracked docs

- Producer contract: `docs/tracknet/inpaint_sidecar.md`
- Historical write-out recipe (evidence pack):
  `docs/tracknet/evidence/inpaint_fabrications_20260722/c11_landing_bisect/inpaint_flag_writeout_recipe.md`
