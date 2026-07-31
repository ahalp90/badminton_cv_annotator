# Annotator regression measurement: current state

Status: current source of truth

Date: 2026-07-31

This document describes what the project can measure now, what the current
evidence shows, and the smallest justified next step. It covers the annotator
calibration and serve-prepend measurement lanes. It does not authorise a
production feature or a new test framework.

## TL;DR

The project currently measures the serve-prepend problem well enough to choose
the next measurement. The maintained calibration chain runs over 292 canonical
rallies, records per-rally raw and accepted contacts, captures replay-mask and
track-quality evidence, and writes reload-checked output. The recorded run
shows substantial clean evidence around many missed serves, but it does not yet
measure a serve-prepend candidate or its false-positive cost.

Do not build a standalone regression harness now. The current feature-specific
measurement lane is the right place for the next candidate experiment. If a
later annotator change needs causal stage isolation, the existing calibration
path has enough seams for a small GT-span/contact wrapper. That future wrapper
should reuse the current runner and scoring path. It should not become a new
package, experiment framework, or parallel production runner.

## Contents

- [TL;DR](#tldr)
- [Executive overview](#executive-overview)
- [Current measurement lane](#current-measurement-lane)
- [What the current evidence shows](#what-the-current-evidence-shows)
- [What the project does well](#what-the-project-does-well)
- [What the project still measures poorly](#what-the-project-still-measures-poorly)
- [Obvious next step](#obvious-next-step)
- [Future controlled comparisons](#future-controlled-comparisons)
- [Scope and minimum validation](#scope-and-minimum-validation)
- [Source map and uncertainty](#source-map-and-uncertainty)

## Executive overview

### How well are we measuring what we need?

We are measuring the current serve-prepend problem moderately well for
diagnosis. The current measurement runs the maintained annotator chain over the
three pinned fixtures. It records enough evidence to locate missed serves and
to decide what a candidate rule must test.

The measurement is not yet a complete feature decision. It observes the normal
detector output against GT. It does not control the pipeline with GT spans or
GT contacts. It therefore cannot attribute a future change cleanly to rally
segmentation, contact detection, or downstream scoring.

The regular calibration floor check protects `covered_fraction` and
`contact_f1`. It does not automatically reject changes in player, server,
landing, hit height, or winner fields, even though the scoring path records
those fields per rally.

### What are we doing well?

The current lane has several useful properties:

- it uses the maintained fixture loader and `build_run_video_inputs()` seam
- it runs the current `run_video()` path without changing production output
- it covers all three pinned fixtures and records the source Git SHA
- it keeps one row per GT rally rather than hiding movement in pooled totals
- it distinguishes raw contacts, accepted contacts, track visibility,
  inpaint state, court presence, and replay-mask state
- it reloads the compressed CSV, JSON, and joblib evidence it writes

These properties make the output useful for feature diagnosis and for choosing
the next experiment.

### What are we doing poorly?

The current lane does not yet measure a serve-prepend rule. It has no candidate
acceptance policy, no systematic false-positive verdict, and no comparison of
span expansion with contact injection. It also cannot hold segmentation or
contact detection fixed while measuring downstream behaviour.

The `no_replay` run is a sensitivity control, not a clean causal ablation. An
all-False raw replay mask changes segmentation before it changes contact
filtering. Its output cannot be read as the isolated benefit of accepting
contacts on replay frames.

### What is the obvious next step?

Keep the committed-mask run as the decision baseline. Add a recording-only,
feature-specific candidate measurement in the existing serve-prepend lane.
Reuse the raw contact, track, inpaint, pose, and mask evidence already
available. Give every candidate and negative case a verdict, including covered
rallies, junk evidence, replay stretches, and cases without usable track
support.

Only after that measurement should the project choose between expanding a
rally span and injecting a contact. Use the existing in-memory seams for that
comparison. Do not build a general regression harness for this feature.

## Current measurement lane

### Inputs and execution

The current script is
[`measure_serve_prepend_lookback.py`](serve_prepend_lookback/measure_serve_prepend_lookback.py).
It uses `FIXTURES` and `build_run_video_inputs()` to assemble the maintained
track, pose, detection, mask, court-presence, and scene-row inputs.

For each fixture and mask mode, `_measure_variant()` copies the input keyword
dictionary, optionally replaces `raw_exclusion_mask` with an all-False array,
adds a `RunCapture`, and calls `run_video()`
([script lines 199-216](serve_prepend_lookback/measure_serve_prepend_lookback.py#L199-L216)).

`build_run_video_inputs()` validates the length and Boolean dtype of the pinned
`court_present` array. It loads the pinned scene rows and converts them to the
homography-row input expected by `run_video()`
([`gt_scoring.py` lines 401-448](../../src/annotator/calibration/gt_scoring.py#L401-L448)).

### Ground-truth use

The script loads GT rallies after the normal `run_video()` call. It classifies
the natural spans and matches natural accepted contacts against GT
([script lines 216-240](serve_prepend_lookback/measure_serve_prepend_lookback.py#L216-L240)).

This is observation against GT. It is not GT injection. The script does not
pass `spans=` or `contacts=` to `run_video()`.

### Modes and outputs

The script supports two mask modes:

| Mode | Input change | Use |
| --- | --- | --- |
| `committed` | fixture-supplied raw replay mask | decision baseline |
| `no_replay` | all-False raw replay mask | mask-sensitivity control |

Each fixture and mode produces a gzip CSV with one row per GT rally and a
typed joblib/XZ evidence array. `summary.json.gz` records the fixtures, source
Git SHA, selected modes, output names, and pooled summaries. The script reloads
each output format after writing it.

The recorded evidence pack is
[`serve_prepend_lookback_20260731-040847`](serve_prepend_lookback/data/serve_prepend_lookback_20260731-040847/).
Its current-code orientation is
[`serve_prepend_lookback_20260731-091227.md`](serve_prepend_lookback/serve_prepend_lookback_20260731-091227.md).

## What the current evidence shows

The recorded summary covers 292 GT rallies:

| Fixture | FPS | GT rallies |
| --- | ---: | ---: |
| `sset_01` | 25 | 113 |
| `sset_15` | 25 | 104 |
| `sset_21` | 30 | 75 |

In the committed-mask variant, the summary records:

- 137 rallies where the GT serve was missed while at least one later GT
  stroke matched an accepted contact
- 164 rallies with an unmatched GT serve in total
- 114 of those cases with a clean visible run in the two-second serve-centred
  window
- 57 with a clean visible run in the pre-contact lookback
- 15 with a raw candidate within the contact-matching tolerance of the serve
- 1 with an accepted candidate within that tolerance
- 17 with the GT serve frame on the believed replay mask

These figures are evidence from the recorded run, not a production-feature
result. They show that the current chain has enough raw evidence to justify a
candidate measurement. They do not show that a candidate would improve
annotation quality.

The 15 raw and 1 accepted near-serve counts are video-wide nearest-contact
signals. The current script does not prove that those contacts belong to the
missed-serve rally. A future candidate ledger must scope this evidence to the
relevant rally or overlapping span.

The mask comparison also shows why the `no_replay` mode is not a clean
contact-filtering ablation:

| Fixture | Detected spans, committed | Detected spans, no replay | Raw contacts, committed | Raw contacts, no replay |
| --- | ---: | ---: | ---: | ---: |
| `sset_01` | 113 | 36 | 3,096 | 8,561 |
| `sset_15` | 142 | 94 | 2,210 | 7,179 |
| `sset_21` | 98 | 38 | 2,246 | 4,980 |

The all-False mask changes the segmentation and the number of raw contacts.
The feature orientation therefore treats it as sensitivity evidence.

## What the project does well

### It reuses the maintained execution path

The measurement uses the same fixture and `run_video()` seams as calibration.
It does not revive a retired driver or copy a second production chain. The
current fixture builder already loads the pinned arrays and pre-generated scene
and court inputs.

### It records evidence at the right granularity

The per-rally ledger keeps the GT serve frame, natural span identity, raw and
accepted contact distances, clean track fractions, inpaint categories, replay
mask state, court presence, and pose availability together. That detail makes
the 137 missed-serve cases inspectable instead of leaving only an aggregate
score.

The calibration scorer also records player, server, hit height, landing, and
winner fields in `RallyRow`. The automatic floor check does not yet protect
those fields, so their per-rally values need review when a change touches
them.

### It keeps the baseline and sensitivity control distinct

The committed-mask run is the decision baseline. The all-False run is labelled
as a sensitivity control. This makes the main policy change visible without
mistaking it for a clean feature ablation.

### It checks its own artefacts

The script reloads every gzip CSV and JSON file and checks the CSV schema and
row count. It reloads every joblib array and checks its type, dtype, and bytes.
That is a useful small integrity gate for an exploratory measurement pack.

## What the project still measures poorly

### Candidate quality

The current script does not run a serve-prepend trigger. It does not define
which raw evidence is sufficient to propose a contact, and it does not score
false positives across ordinary rallies, junk evidence, replay stretches, or
missing-track cases.

### Feature semantics

The current output does not decide whether a prepend should expand a rally span
or inject a contact while keeping the span stable. Those choices affect later
pairing and landing windows. They need a feature-specific counterfactual.

### Stage attribution

The current script uses natural spans and contacts. A future code change can
move the result because of segmentation, contact detection, mask policy, or
downstream scoring, and this measurement cannot separate those causes.

The existing `run_video()` signature already accepts optional half-open spans
and rally-indexed contacts
([`run_video.py` lines 199-231](../../src/annotator/run_video.py#L199-L231)).
The capability exists, but the current serve-lookback measurement does not use
it for GT control.

The current lookback anchor also needs careful interpretation. `_nearest()`
chooses the accepted contact closest to the GT serve. The stored anchor-source
label calls it the first assigned accepted contact. The 57 recorded lookback
runs are therefore diagnostic pre-anchor evidence, not a validated measure
anchored at the first accepted contact.

### Mask interpretation

The committed calibration input includes a raw exclusion mask. `run_video()`
derives the definitive mask before segmentation and contact filtering. Injected
contacts, if used later, would still pass through the ordinary exclusion and
hallucination policies. A GT contact would not automatically mean an
unfiltered contact.

## Obvious next step

Add a recording-only candidate rule to the serve-prepend measurement lane. The
rule should reuse the evidence already captured by the script:

1. select candidate frames from the available raw contact, track, inpaint,
   pose, and replay-mask evidence
2. record a reason for every accepted and rejected candidate
3. score candidates on covered rallies as well as missed serves
4. separate clean-track, no-track, fabricated/degraded, and believed-replay
   cases
5. scope near-serve evidence to the relevant rally or overlapping span
6. compare the smallest in-memory span and contact counterfactuals through the
   existing seams

Keep the committed-mask run as the baseline. Use the all-False mode only to
measure mask sensitivity. Do not change production output while this evidence
is being collected.

## Future controlled comparisons

The existing calibration path has enough capacity for a thin controlled
comparison if a later annotator change genuinely needs stage attribution. The
future comparison would reuse the same fixture inputs and ordinary scoring
path:

| Run | Controlled input | What remains live |
| --- | --- | --- |
| ordinary | no GT injection | segmentation, contact detection, and downstream stages |
| GT spans | half-open spans derived from GT contact extents | contact detection and downstream stages |
| GT spans plus contacts | GT spans and rally-indexed GT contact frames | downstream stages and their policies |

`load_gt_rallies()` exposes inclusive first and last stroke frames. A wrapper
would convert each extent to `(first_frame, last_frame + 1)` before passing it
as a half-open span
([`scoring.py` lines 34-77](../../src/annotator/calibration/scoring.py#L34-L77)).

This future comparison would not bypass every policy. Injected contacts would
still be subject to exclusion and hallucination filtering. The selected mask
policy would need to be recorded with the output.

The minimum useful implementation would be a mode selector and GT overlay
around the existing calibration path. It would reuse `score_video()` and the
existing calibration writers where their output contract fits. It would keep
mode identity in an output path or small manifest.

Do not add a new package, generic experiment-axis framework, resume system,
derived-array hashing, or shared CSV-schema field until an actual consumer
requires it. Do not extend the fixed
`src/annotator/e2e_court_annotator.py` measurement matrix for this purpose. It
has a separate job measuring static and detected court-evidence configurations.

## Scope and minimum validation

### Current scope

- current-chain evidence collection for serve-prepend
- per-rally inspection of natural spans and contacts against GT
- committed-mask baseline and all-False mask sensitivity
- feature-specific candidate measurement

### Deferred scope

- GT-controlled stage attribution for a concrete future annotator change
- comparison of span expansion with contact injection
- an independent CI or release gate

### Minimum validation when implementation is authorised

- every requested fixture and mode produces an output
- no requested mode is silently skipped
- GT span ends use the half-open convention
- injected contact-map keys match the injected span order
- every candidate receives a verdict, including negative cases
- one focused test covers the mode overlay or candidate boundary being changed

Do not add synthetic mutation orchestration or a general regression framework
without a demonstrated consumer. This is a student project. A small,
source-grounded check is more useful than a large reusable platform.

## Source map and uncertainty

Primary current evidence:

- [`measure_serve_prepend_lookback.py`](serve_prepend_lookback/measure_serve_prepend_lookback.py)
- [`summary.json.gz`](serve_prepend_lookback/data/serve_prepend_lookback_20260731-040847/summary.json.gz)
- [`serve_prepend_lookback_20260731-091227.md`](serve_prepend_lookback/serve_prepend_lookback_20260731-091227.md)

Relevant execution seams:

- `src/annotator/calibration/gt_scoring.py:401-448` for pinned fixture input assembly
- `src/annotator/calibration/gt_scoring.py:527-645` for existing calibration scoring
- `src/annotator/calibration/gt_scoring.py:690-702` for the current automatic floors
- `src/annotator/run_video.py:199-231,349-465` for ordinary and injected input paths
- `src/annotator/calibration/scoring.py:34-77` for GT rally extents

The recorded summary was generated at Git SHA
`63f40938a62f6612ca9a63b61127d24442a80865`, which matches the current `HEAD`.
The summary is evidence from one recorded run. This document did not rerun the
measurement script or the test suite.

PyCharm MCP symbol and text searches, direct source reads, and `rg` searches
confirmed the relevant definitions and usages. PyCharm's Python call-hierarchy
query did not resolve the callable FQNs, so the usage claims rely on those
searches and source reads.

This document is an audit lead and current working decision. Future data or a
concrete stage-isolation requirement can change the recommendation.
