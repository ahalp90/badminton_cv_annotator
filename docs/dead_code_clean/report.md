# Dead and duplicate code: what the audit found

This is the readable summary. The full evidence lives in findings.md and the
raw sweep returns under findings/. Nothing has been changed; every line below
is a finding or a recommendation, not an action taken. Where a claim was
checked only by a sweep agent and not re-checked by hand, it says so.

## The short version

The codebase is in better shape than the duplication suggests. Most of the
copying between bst_x, the scraper and src/shared turned out to be deliberate
and documented, done to stop packages depending on each other's internals.
The real rot is smaller and more specific: one dead vendor file, one dead
player tracker, a shared library that was never fully adopted, a crop of
functions kept alive only by their tests, and forty finished-pass scripts
that belong in an archive folder. The duplicates genuinely worth merging are
few, small, and all have a clear winner.

## Theme 1: the bric TrackNet copy is a working duplicate, and part of it is dead

bric carries its own copy of TrackNetV3 under perception/_vendor. The eight
Python files that do the actual work are byte-identical to the bst_x copy.
The divergences are batch_predict.py (only the bst_x side has the
--large_video option) plus comment-level drift in the README and
requirements. And bric's copy of batch_predict.py has no caller at all: bric
drives predict.py through a subprocess, and the API imports two functions
from the package. So today the mirror buys namespace isolation and nothing
else, at the price of two trees to patch.

If you want one tree, the work is: retarget bric's subprocess path and the
API imports at the bst_x copy, keep --large_video, delete the bric tree. The
files being identical means no behaviour reconciliation is needed, but the
path and import changes still deserve a runtime check before anyone calls
the job small. If you keep the mirror, delete bric's batch_predict.py
either way, since it is dead on both sides of the argument.

## Theme 2: the bst_x / scraper / shared "redundancy" is mostly on purpose

This was the audit's surprise. The suspicious pairs (two yt-dlp downloaders,
two court-maths modules, two player mappers, two clip-bound calculators, two
taxonomy registries) are documented mirrors, not accidents. For the shared
copies, src/shared/README.md spells out the trade: BRIC must stay
self-contained rather than import bst_x internals, so shared carries copies
and accepts drift. The bst_x downloader states its own duplication in a
comment. The annotator, for what it is worth, already imports bst_x's
sticky-anchor heuristics directly, so the wall only ever applied to BRIC.
Killing these copies would re-couple what the mirrors keep apart. You could
revisit that design, but it is a decision, not an oversight.

The drift the mirrors accepted has, however, started to happen. Two live
divergences deserve a deliberate ruling rather than silence:

- shared's compute_clip_bounds lacks the max(0, start_f) clamp the bst_x
  version has, so a stroke near the start of a video produces different clip
  bounds on the two sides.
- the two taxonomy registries disagree about driven_flight: bst_x maps it to
  drive, the shared legacy map sends it to unknown.

## Theme 3: src/shared is a shared library nobody finished moving into

shared was built as the neutral ground, but adoption stopped partway, and a
chunk of it is now dead or kept alive only by tests: the legacy court
projection pair, the whole temporal module, video_io's readers and thumbnail
writer, the import-time SPLITS_V2 table (its one importer never uses the
name), and two taxonomy constants whose live twins sit in bst_x's own
config. All of these were re-checked by hand. The honest options per module
are: name a production user, or delete it and shrink the README.

## Theme 4: functions kept alive only by their tests

A repeating pattern, mostly in the annotator: production moved to a newer
function and something older stayed behind because a test still calls it.
They fall into three groups:

- Superseded wrappers, re-checked by hand: pick_landing (production uses
  pick_landing_to_end), detect_contacts (production uses
  detect_contact_flags), fetch_span (production streams frames), the
  court-inputs builder, the legacy winner-config loader, and courtkeynet's
  single-frame detect (production uses detect_batch). Fix: point the test at
  the live function, delete the wrapper.
- Test helpers living in production files, sweep-verified only: the
  calibration floors assertion, score_stage8, two selection functions, and
  court_scale_boxes. Each needs its own look; some may just move into the
  test file.
- Probes nothing consults, sweep-verified only: the API's registry and
  model liveness checks (is_available, available_splits, _live_splits,
  _summary_live).

Related maintenance hazard from the same sweep: run_video repeats the same
resolved span options across five branches, so a new option needs five
synchronised edits. The suggested fix is a local variable, not new
machinery.

## Theme 5: stale knobs from the fixed-25fps era

Four config aliases (BEST_CONFIG_THRESHOLDS, COURT_ABSENT_WINDOW,
SUSTAINED_LOSS_FRAMES, MIN_DESCEND_SAMPLES) predate the fps-scaling work and
now shadow the values production actually resolves (sweep-verified, each
with its live replacement named in the ledger). Two scraper download knobs
(CONCURRENT_FRAGMENTS, DOWNLOAD_WORKERS) have zero consumers because the
downloader hardcodes the same numbers; that pair was re-checked by hand. All
six mislead whoever goes looking for the tuning point.

## Theme 6: half the scripts are finished checks, not tools

The census walked all 77 files across scripts/ and the validation folders.
None is unreachable, but 40 are completed one-off checks from finished work:
the rtmlib migration checks (14 files), the refactoring equivalence checks
(8), the mmpose investigation scripts (6), most of scripts/plots (7 charts
pinned to historical run IDs), and five singles including the mmpose zeroing
equivalence check. They did their job and the docs cite their results.
Moving them under an archive folder, as scripts/archive already models,
would make the live tool set visible at a glance. The scripts/archive
boundary itself is clean: nothing live imports it.

## Theme 7: the duplicates worth merging, all small, all with a clear winner

Re-checked by hand:

- The confusion-matrix renderer exists twice; shared/eval_plots is the
  caller path, the scripts/plots copy should become a thin command-line
  front over it.
- Two identical clip-path indexers in bst_x; keep pipeline/clip_index.
- bric/eval duplicates bric/train's _select_device byte for byte, while
  already importing other train helpers.
- The API re-declares bric's RGB normalisation constants, with a comment
  saying it mirrors them.
- hparam_sweep hand-rolls mean arithmetic next to the reducers it should
  call.

Sweep-verified only, needs a closer look before acting:

- _find_rally_spans recomputes scaffolding _rally_regions already returns.
- inout_verdict repeats geometry landing_margins already computes.

And one genuinely dead block with no argument for keeping it, re-checked by
hand: bric's ByteTrack-based detect_and_track player tracker (~200 lines),
which nothing has called since the frame-detection path took over.

## Decisions this leaves with you

1. Keep or collapse the bric TrackNet mirror (theme 1). The evidence says
   the working files are identical today; the cost sits in retargeting the
   subprocess and imports, and wants a runtime check.
2. Rule on the two live mirror divergences (theme 2): the clip-bound clamp
   and the driven_flight mapping.
3. Per shared module with no production caller: adopt or delete (theme 3).
4. Three bric/api items that anticipate features which never arrived, all
   sweep-verified only: the model-selection plumbing the routes never
   dispatch on; the court-enabled deployment knob that cannot work because
   the API never supplies the court tensors; and the dataset building
   tensors for disabled model lanes. Related: the advertised cache-repair
   path is bypassed on ordinary reruns, and several player/shuttle cache
   fields are written but never read. That last one changes an on-disk
   format, so it needs its own careful pass before anyone acts.

## The numbers

Eight read-only sweeps returned 161 findings. After merging, the ledger
(findings.md) carries them in eight sections; the counts below are ledger
rows, and a few items appear in two sweeps' returns because the test sweep
corroborated the package sweeps:

- 3 unreachable code blocks (the shared court pair, the shared temporal
  helpers, bric's player tracker), all re-checked by hand.
- 8 symbols with no production caller (dead vendor entry point, scraper
  wrapper, legacy winner-config loader and similar), re-checked by hand.
- About 18 functions or constants kept alive only by tests (theme 4 plus
  the shared library's test-only surface).
- About 14 unused-surface items inside live code (stale knobs, write-only
  fields, dead parameters), most sweep-verified only.
- 7 duplicates recommended for merging (theme 7), 5 re-checked by hand.
- 40 of 77 census files classified as finished checks to archive,
  classification per file cited in findings/wp7.md.
- Around 20 deliberate mirrors recorded with their reasons, left alone.

Verification method: every "re-checked by hand" claim above was confirmed
against the full tracked repository, including the documented entry points
(CI, the served API, and every python -m command the docs mention). Two
sweep claims failed that check and were thrown out; two were corrected.
Items marked sweep-verified rest on the sweep's file-and-line evidence and
should be re-confirmed at refactor time. The ledger tags every row one way
or the other.
