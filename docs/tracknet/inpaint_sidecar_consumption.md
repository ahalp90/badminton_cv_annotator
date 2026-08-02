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

**No production consumer reads the sidecar JSON.** Grepping
`inpaint_selected`, `inpaint_status`, and `inpaint_fill_mask` against
`src/annotator/`, `src/scraper/`, and `src/bst_x/` (excluding the
TrackNetV3 writer) returns no matches at the current tip. The
sidecar boundary choice
(`docs/tracknet/inpaint_sidecar.md` § Boundary choices) records this as
"writer only". The producer contract is stable; consumer wiring is the
open work.

## The event-mask boundary in the annotator

The annotator has an event-mask seam that the sidecar can feed into, but
does so today through an in-memory recurrence detector, not the sidecar.

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

**Only the calibration path constructs `inpaint_codes` today.** The
production scraper and stroke-classifier pipelines do not, so their
`run_video` invocations receive no event-mask, and downstream contact /
landing / lost-shuttle rules cannot distinguish invented frames. This is
the seam a future sidecar consumer plugs into: build a boolean mask from
the sidecar's `inpaint_selected` spans and pass it as the mutually
exclusive `shuttle_hallucination_mask` argument to `run_video`.

## Open consumer work

- **Wire the sidecar to production `run_video`.** Read
  `{video_stem}_stride{N}_inpaint_mask.json.gz`, expand
  `inpaint_selected` to a `(n_frames,) bool` array, pass it in as the
  `shuttle_hallucination_mask` (never together with `inpaint_codes`). Decide
  per lane whether the sidecar or the inpaint-guard codes are
  authoritative and document the choice.
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
