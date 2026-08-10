# Inpaint hallucination guard audit workset

This workset explains the live guard, records an exploratory motion audit, and
compares that audit with the producer's stride-8 inpaint sidecars. It contains
no production-code or detector-policy change. Aggregate heuristic results are
leads rather than hallucination labels. The issue-31 human review is the
bounded visual ground truth added after that review.

## TL;DR

The guard catches 66.93%, 79.98%, and 68.07% of the separate RANSAC candidate
frames in `sset_01`, `sset_15`, and `sset_21`. Those figures are agreement with
one heuristic, not hallucination recall. Curtis labelled all 18 selected
high-risk spans as visual hallucinations. The blind human review corrected one
provisional label that had mistaken marketing text for a resting shuttle. The
purposive result supports candidate ranking, not a population precision or
recall claim.

The event-union pass adds two context views. Union 1 combines the existing
uncaught-plus-sidecar evidence view with raw contact impulses. Union 2 combines
uncaught candidates, raw impulses and inductive TP rally-enders. Their totals
are evidence-view counts, not hallucination counts.

The separate provenance-coverage follow-up compares the current guard with an
exploratory `guard OR Union 2` tag over sidecar-selected material. It shows
more producer provenance covered, not more hallucinations caught.

## Contents

- [Start here](#start-here)
- [Executive overview](#executive-overview)
- [Current outputs](#current-outputs)
- [Why the formats are deliberate](#why-the-formats-are-deliberate)
- [Re-run the bounded analysis](#re-run-the-bounded-analysis)
- [View definitions](#view-definitions)
- [Validation and scope](#validation-and-scope)

## Start here

- [Guard operation trace](inpaint_hallucination_trace.md): source-grounded
  trace, plain-language explanation, and Mermaid charts.
- [Implementation infographic](inpaint_hallucination_infographic.png): how the
  live guard works.
- [Audit findings infographic](inpaint_hallucination_findings_infographic.png):
  what the bounded audit measured and what it cannot establish.
- [Guard versus RANSAC infographic](inpaint_hallucination_lenses_infographic.png):
  the different patterns each heuristic measures.
- [Next-step infographic](inpaint_hallucination_next_steps_infographic.png):
  the smallest useful video-review plan.
- [Event-union infographic](inpaint_hallucination_event_unions_infographic.png):
  what the two evidence unions contain.
- [Event-union percentage findings infographic](inpaint_hallucination_event_unions_findings_infographic.png):
  the same extension expressed as denominator-labelled percentages.
- [Inpaint provenance coverage infographic](inpaint_hallucination_provenance_coverage_infographic.png):
  how much sidecar-selected material receives a current or exploratory tag.
- [Follow-up report](ongoing_shuttle_hallucination_issues_20260731-094523.md):
  measured results, limitations, sidecar comparison, and next action.
- [Provenance coverage follow-up](inpaint_provenance_coverage_followup_20260731-192654.md):
  the separate frame- and span-level comparison behind the new infographic.
- [Issue-31 visual audit](shuttle_hallucination_visual_audit_20260809.md):
  18 reviewed spans, exact source ranges, visual labels, and remaining checks.

External review and planning records are kept outside this committed workset.

## Executive overview

The guard is useful as a broad screening heuristic. It catches structured
recurrence in the saved track and exposes inspectable non-zero grades. It does
not know whether a coordinate came from ordinary detector output, inpaint,
masking or real shuttle motion. The RANSAC comparison and sidecar overlap are
therefore leads, not recall or precision measurements.

The new unions add context around possible blind spots. Raw impulses follow
the current fps-resolved contact path. A TP rally-ender means shuttle events
that have closed a valid rally, did not overlap with another valid GT rally, and
are valid within our rally-ending ruleset. This encompasses the fact that
ShuttleSet's GT does not actually record the rally's final event, so we only
ever know it inductively. The GT dataset only ever records the final contact.
The accepted frame is a current-span close proxy, not direct GT truth.

The bounded human review is complete. All 18 selected spans are hallucinations,
but the challenge set has no real-shuttle controls. This supports using RANSAC
for ranking rather than automatic rejection. Nine confirmed hallucinations
have no sidecar overlap, so producer provenance cannot cover the residual
problem by itself. The challenge set does not justify a threshold change or
guard replacement.

## Current outputs

The compressed manifest and audit summary are
[raw_manifest.json.gz](raw_manifest.json.gz) and
[analysis/track_audit.json.gz](analysis/track_audit.json.gz).

The raw tracks are native NumPy float64 arrays wrapped in XZ level-9 streams:

- `raw/sset_01_track.npy.xz`
- `raw/sset_15_track.npy.xz`
- `raw/sset_21_track.npy.xz`

The derived arrays are `.npy.xz` files containing compact boolean or `uint8`
arrays:

- `analysis/*_guard_codes.npy.xz`
- `analysis/*_ransac_candidate.npy.xz`
- `analysis/*_uncaught_mask.npy.xz`
- `analysis/*_sidecar_inpaint_mask.npy.xz`
- `analysis/*_impulse_event_mask.npy.xz`
- `analysis/*_tp_rally_ender_mask.npy.xz`
- `analysis/*_event_source_codes.npy.xz`

Per-frame and per-chunk text outputs use gzip level 9:

- `analysis/*_frame_audit.csv.gz`
- `analysis/*_uncaught_chunks.csv.gz`
- `analysis/top_locations.csv.gz`
- `analysis/top_inpaint_locations.csv.gz`
- `analysis/top_unfiltered_inpaint_locations.csv.gz`
- `analysis/top_sequences.json.gz`
- `analysis/top_inpaint_sequences.json.gz`
- `analysis/top_unfiltered_inpaint_sequences.json.gz`
- `analysis/*_impulse_events.csv.gz`
- `analysis/*_tp_rally_ender_events.csv.gz`
- `analysis/*_visual_hallucination_audit.csv.gz`
- `analysis/human_visual_review_20260810.csv.gz`
- `analysis/event_union.json.gz`
- `analysis/inpaint_coverage.json.gz`
- `analysis/accepted_attractor_overlap.json.gz`

The ten image-space views are:

- [Top uncaught locations](plots/top_uncaught_locations.png)
- [Top uncaught sequence families](plots/top_uncaught_sequences.png)
- [Sidecar-selected inpaint locations](plots/top_inpaint_locations.png)
- [Sidecar-selected inpaint sequence families](plots/top_inpaint_sequences.png)
- [Union locations](plots/top_unfiltered_inpaint_locations.png)
- [Union sequence families](plots/top_unfiltered_inpaint_sequences.png)
- [Uncaught + inpaint + impulse locations](plots/top_uncaught_inpaint_impulse_locations.png)
- [Uncaught + inpaint + impulse sequence families](plots/top_uncaught_inpaint_impulse_sequences.png)
- [Uncaught + impulse + TP rally-ender locations](plots/top_uncaught_impulse_tp_rally_end_locations.png)
- [Uncaught + impulse + TP rally-ender sequence families](plots/top_uncaught_impulse_tp_rally_end_sequences.png)

## Why the formats are deliberate

The source tracks are float64. A trial conversion to float32 changed a small
number of RANSAC boundary decisions even though its pixel error was far below a
pixel. Compressed float64 `.npy.xz` preserves the exact audit decisions and is
only marginally larger than the float32 trial. Float16 was rejected because it
adds avoidable quantisation while saving little after LZMA compression. In
other words, float64 is the lowest tested precision that reproduced this
candidate set exactly.

`analysis/compressed_io.py` keeps the array and text format choices in one
small helper. `write_npy_xz` writes a native NumPy `.npy` stream through
`lzma.open(..., format=lzma.FORMAT_XZ, preset=9)`. `read_npy_xz` reloads it with
`np.load(..., allow_pickle=False)`. JSON and CSV use gzip level 9. The PNGs
were passed through pngquant and then oxipng with metadata stripped.
The location heatmaps use 16 colours, the sequence grids use 8 colours, and
the infographic uses 16 colours. At eight colours pngquant could not meet a
minimum quality of 40, so the selected runs use a maximum quality of 60 and
were checked visually.

## Re-run the bounded analysis

Run from the repository root:

```bash
~/.venvs/badminton-cicd/bin/python \
  docs/scraper_pipeline/inpaint_hallucination_fix/analysis/audit_tracks.py

MPLCONFIGDIR=/tmp/badminton-matplotlib \
  ~/.venvs/badminton-cicd/bin/python \
  docs/scraper_pipeline/inpaint_hallucination_fix/analysis/plot_recurrence_grids.py \
  --top-n 6 --view all
```

`audit_tracks.py` calls the live `annotator.inpaint_guard.grade_track`. Its
separate RANSAC lens fits a local quadratic in pixel coordinates over 16-frame
windows, uses a 3-pixel residual and 32 deterministic sample triples, and
steps windows by four frames. Any window containing exact `(0, 0)` masking is
excluded. A frame becomes a candidate when at least half of its eligible
windows vote it outside the model.

These settings generate leads only. Real acceleration can look like an
outlier, while a smooth fill can fit a quadratic and look like an inlier.
They are not production thresholds.

## View definitions

Here, **coordinate-valid** means that x and y are not both exactly `(0,0)`.
It does not mean that the detector was correct or that the visibility column
passed an independent test.

The uncaught view is `RANSAC candidate & guard_code == 0`. Its location plot
counts coordinate-valid candidate frames after rounding each point to an
integer pixel.
`#1`, `#2`, and so on are frequency ranks, not detector grades. Each legend
entry gives the rounded `(x, y)` pixel and the number of selected frames that
land there.

Its sequence plot uses coordinate-valid, guard-clean 16-frame windows
containing at least one uncaught candidate. The top 256 exact sequences are
then grouped with complete-linkage `fclusterdata` using its `distance`
criterion. This is a bounded descriptive sample. The
uncaught exact sequences are mostly singletons, so the grouping is more useful
than treating exact repetition as evidence, but it is not a second detector.

The inpaint view reads the producer's pinned `inpaint_selected` half-open frame
spans from the sidecar files under
`local_scratch/autograder_architecture/inpaint_sidecar/backfill_staging/out_stride8/`.
Locations use sidecar-selected coordinate-valid frames. Sequence windows
require all 16 frames to be sidecar-selected and coordinate-valid. Exact
`(0, 0)` coordinates are
reported separately and excluded from both image-space views.

The union view is the boolean union of the coordinate-valid uncaught mask and
the coordinate-valid sidecar mask. It is an evidence view, not a new detector.
Its locations count frames selected by either source. Its sequence windows
require all 16 frames to be selected by that union. The union can therefore
show a frame that is not sidecar-selected and a sidecar frame that the guard
already marked non-zero.

The first event view, `uncaught_inpaint_impulse`, extends that existing union
with the raw contact impulse mask. The second event view,
`uncaught_impulse_tp_rally_end`, combines the uncaught mask, raw impulse mask
and inductive TP rally-ender mask without the sidecar. Both views use the
coordinate-valid OR of their source masks. Their source-coded arrays retain
overlap information without double-counting frames.

For this audit, a TP rally-ender means shuttle events that have closed a valid
rally, did not overlap with another valid GT rally, and are valid within our
rally-ending ruleset. This encompasses the fact that ShuttleSet's GT does not
actually record the rally's final event, so we only ever know it inductively.
The GT dataset only ever records the final contact. The audit accepts only a
pre-video-end span containing one complete GT rally and no second overlapping GT
rally. It marks the last frame inside the span, so the result is an inductive
span-close proxy.

For both event views, location plots count coordinate-valid frames. Sequence
views require 16 valid coordinates and at least one selected union frame. They
do not require guard-clean status, and they scan every possible frame start.
The script clusters only the top 256 exact sequences. None of the fifteen
generated fixture/view combinations reaches the silhouette target of 0.5, so
the selected families remain descriptive leads.

The clustering threshold `t` is a sequence-level RMS pixel-distance threshold.
Each 16-frame sequence contributes 32 scalar values: x and y for each frame.
The distance is:

```text
sqrt(sum(dx_px**2 + dy_px**2) / 32)
```

So `t=96` means an RMS of at most 96 pixels across those 32 values for a
complete-linkage pair at the selected cut. It is not 96 pixels independently
added to every coordinate, and it is not `96 / 16 = 6` pixels of mean drift.
The same interpretation applies to `t=128`, with a 128-pixel sequence RMS
threshold. A few points can use much of the RMS budget while other points
remain close.
Silhouette 0.5 is a quality target, not a prescribed value of `t`. The script
tests a small fixed threshold grid, chooses the smallest threshold meeting 0.5,
and otherwise chooses the best tested silhouette. None of the fifteen generated
fixture/view combinations reaches 0.5, so the families remain descriptive
leads.

The report records the catch proportion as agreement with the RANSAC candidate
set. It is not recall, because the audit has no frame labels or independent
pre-inpaint track.

RANSAC is therefore a complementary audit lens, not a drop-in replacement for
the live guard. The guard detects recurring track patterns and nearby attractors;
RANSAC detects local departures from a fitted motion model. A replacement would
need a small visually reviewed sample before it could be judged fairly.

## Validation and scope

The scripts were re-run on the three requested fixtures. Native `.npy.xz`
arrays, JSON, CSV, sidecars, plots, and PNG outputs were reloaded or inspected
with bounded checks. The configured linter and `--help` checks passed for the
analysis scripts. The repository test suite was intentionally not run for this
exploratory workset close-out.
