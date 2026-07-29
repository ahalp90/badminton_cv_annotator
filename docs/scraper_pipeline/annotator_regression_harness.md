# Clean GT-injected regression harness: design

**STATUS: NOT BUILT.** This document turns the owner-ruled TODO entry (`TODO: Build
the clean GT-injected regression harness after cleanup`,
`local_scratch/autograder_architecture/TODO.md`, ruled 2026-07-29) into a concise
executable design. No harness code exists yet; nothing described here has landed. The
tracked calibration reference under `tests/data/annotator_calibration/reference/` is
today's evidence, captured through the existing production-chain command, not through
this harness.

## Why a harness, and why not the old ones

The auto-annotation heuristics still change. Isolating one change's effect needs a
harness built from first principles, around current production interfaces and
canonical ShuttleSet CSVs, with no revived shims, monkey patches, import rebinding,
two-tree package tricks, or duplicated historical implementations. `run_video`'s
`spans`/`contacts` injection seams
(`src/annotator/run_video.py:162`) and
`annotator.calibration.scoring.load_gt_rallies` are the starting point; the harness
reuses `gt_scoring`'s input loading, scoring, and reconciliation rather than building
a second chain. `end_to_end_yardstick.py`, `point_winner_pin.py`, `d5_winner_retest.py`,
and the retired S28 driver remain historical evidence — design archaeology, not code
to repair.

## The three isolation modes

Each mode calls `annotator.run_video.run_video` with the same fixture arrays
(`annotator.calibration.gt_scoring.build_run_video_inputs(fixture)` already assembles
the invariant positional/keyword arguments), differing only in what is injected.

### Mode 1: full production chain

Call `run_video(*inputs.positional, **inputs.keyword)` with no `spans` or `contacts`
override — the ordinary path `run_fixture` already takes
(`src/annotator/calibration/gt_scoring.py:705`). This measures the behaviour users
receive: rally segmentation, contact detection, attribution, landing, hit height,
server, and point winner are all live.

### Mode 2: GT rally spans only

Load `shots_master.csv` with pandas through
`annotator.calibration.scoring.load_gt_rallies(shots_master, vid)`, which groups
strokes by `(vid, set_id, rally)` into `GtRally` rows ordered by `(set_id, rally)`
(`src/annotator/calibration/scoring.py:56`). Each `GtRally.extent` is the inclusive
`(first_stroke_frame, last_stroke_frame)`. Convert every extent to the half-open span
`run_video(spans=...)` requires: `(first_stroke_frame, last_stroke_frame + 1)`. Inject
only `spans=<converted list>`; leave `contacts` unset. This removes rally segmentation
from the comparison while leaving contact detection and every downstream stage live.

The injected extents are GT contact extents, not a claim that the GT first and last
contacts are the rally's true visual boundaries. The final shuttle event of a rally
(the landing) happens after the final contact by necessity, and a real rally clip
needs context before the first contact — see the "Rally extents measured by exact
ShuttleSet GT frames are flawed" entry in `TODO.md`. Mode 2's spans measure contact
detection and downstream logic against GT contact frames; they are not a rally-clipping
reference.

### Mode 3: GT rally spans plus GT contacts

Inject the same converted `spans` and `contacts={rally_index: list(stroke_frames)}`,
keyed by each span's index in the injected list (the same order `load_gt_rallies`
returns, so `rally_index` lines up 1:1 with `spans[rally_index]`). This removes both
rally segmentation and contact detection from the comparison, isolating striker
attribution, landing, hit height, server, and point-winner logic against the actual
GT rows. Per-set winner and landing labels come from the existing reconciliation code
(`annotator.calibration.gt_scoring.reconcile_sets`,
`src/annotator/calibration/gt_scoring.py:460`), which already reads the canonical
per-set CSVs.

`run_video` rejects `serve_start` combined with injected `spans`
(`src/annotator/run_video.py:199`); the calibration chain does not pass `serve_start`,
so this does not conflict with modes 2 or 3.

## Expected causal pattern

| Changed stage | Full chain | GT spans | GT spans + contacts |
|---|---:|---:|---:|
| Rally segmentation only | moves | stable | stable |
| Contact detection only | moves | moves | stable |
| Attribution, landing, hit height, server, or winner | moves | moves | moves |

A different pattern is evidence that the change crosses the assumed stage boundary,
or that the harness is wired incorrectly.

## Comparison and re-pinning

Every mode runs through the same scoring and output path the production chain already
uses: `score_video` builds a `VideoScoring`, `flatten_metrics` produces the structured
aggregate row, and `write_rallies_csv`/`write_geometric_verdicts_csv` produce the
stable per-rally rows (`src/annotator/calibration/gt_scoring.py:527`, `:648`, `:724`,
`:731`). Compare reference and candidate values directly, reporting deltas by fixture,
rally, field, and mode, so a change localises to the earliest moving stage.

Exact semantic equality is the default for deterministic outputs. A numeric tolerance
is an explicit, named policy for a particular metric, never an implicit blanket — the
existing frame tolerance for contact matching,
`annotator.calibration.gt_scoring.canonical_tolerance(fps)`
(`src/annotator/calibration/gt_scoring.py:372`), is the model for how a tolerance
should be named and scoped to the one metric that needs it.

Capture and acceptance stay separate. A comparison command never overwrites its own
reference. Re-pinning — replacing a tracked reference with a new capture — is an
explicit, reviewed action separate from running a comparison, recorded with the
old/new delta and a short causal reason. The source commit and fixture input hashes
recorded alongside a reference are provenance only; a recorded pin is not required to
equal the current HEAD.

## Acceptance for the future build

- One current implementation path, with no compatibility adapter.
- All three isolation modes run on the same fixture/input manifest.
- A controlled synthetic change at each stage first moves the expected mode (the
  causal-pattern table above).
- Equal-and-opposite per-rally movements cannot hide behind unchanged aggregates.
- Direct CSV/pandas GT loading is deterministic and fails loudly on missing or
  ambiguous GT rows.
- Focused tests, full project gates, and one clean pre/post real-fixture
  demonstration are recorded.

**This harness remains a TODO.** See `local_scratch/autograder_architecture/TODO.md`
for the owner ruling and current status.
