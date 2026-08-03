# Dead and duplicate code audit

This report summarises the audit scope, findings, and final refactor rulings.
The merged evidence is in `findings.md`. The read-only sweep returns are under
`findings/`. The audit changed no source code.

The audit used eight read-only sweeps, each covering a separate area. The
results were merged against a root manifest of tracked entry points.
Load-bearing claims were checked first-hand against all tracked content. An
independent second check reviewed the final report and the R0-R9 rulings.
Hand-run entry points required separate attention because ordinary caller
tracing does not expose them.

## The short version

Most duplication between bst_x, the scraper and src/shared was deliberate and
documented. The copies keep packages from depending on each other's internals.
The confirmed dead or superseded surface is narrower: one dead vendor file,
one dead player tracker, partly adopted shared modules, functions kept alive
only by tests, and a small set of duplicates with a clear canonical version.
The audit classified 40 finished-pass scripts as archive candidates. R9 later
ruled that all 40 stay in their existing locations.

## Theme 1: the bric TrackNet copy is a working duplicate, and part of it is dead

bric carries its own copy of TrackNetV3 under perception/_vendor. The eight
Python files that do the actual work are byte-identical to the bst_x copy.
The divergences are batch_predict.py (only the bst_x side has the
--large_video option) plus comment-level drift in the README and
requirements. And bric's copy of batch_predict.py has no caller at all: bric
drives predict.py through a subprocess, and the API imports two functions
from the package. So today the mirror buys namespace isolation and nothing
else, at the price of two trees to patch.

R1 settles the mirror question. Keep one tree with bst_x's content at
`src/shared/tracknetv3/`. Retarget the four consumers and all documented
commands. Preserve `--large_video`, update literal tooling exclusions, and
delete both old trees after the move. The working files are identical, so no
behaviour reconciliation is needed. The path and import changes still require
runtime checks.

## Theme 2: the bst_x / scraper / shared "redundancy" is mostly on purpose

The suspicious pairs (two yt-dlp downloaders,
two court-maths modules, two player mappers, two clip-bound calculators, two
taxonomy registries) are documented mirrors, not accidents. For the shared
copies, src/shared/README.md spells out the trade: BRIC must stay
self-contained rather than import bst_x internals, so shared carries copies
and accepts drift. The bst_x downloader states its own duplication in a
comment. The annotator, for what it is worth, already imports bst_x's
sticky-anchor heuristics directly, so the wall only ever applied to BRIC.
Removing these copies without a replacement boundary would re-couple the
packages. R2 and R3 now define the replacement boundaries.

The mirrors had two live divergences. R2 settles both:

- `shared.dataset.compute_clip_bounds` lacks the `max(0, start_f)` clamp in the
  bst_x version. R2 adopts bst_x semantics, including the clamp. This is an
  accepted behaviour change for early-frame clip bounds.
- The taxonomy registries disagree about `driven_flight`. bst_x maps it to
  `drive`, while the shared legacy map sends it to `unknown`. R2 makes the
  bst_x mapping authoritative and corrects every merge map, including legacy
  entries.

## Theme 3: src/shared is a shared library nobody finished moving into

shared was built as neutral ground, but adoption stopped partway. A chunk of
the package is dead or kept alive only by tests: the legacy court
projection pair, the whole temporal module, video_io's readers and thumbnail
writer, the import-time SPLITS_V2 table (its one importer never uses the
name), and two taxonomy constants whose live twins sit in bst_x's own
config. All of these were re-checked by hand. R2 deletes the unused surface,
moves classifier-only modules to `src/classifier_shared/`, keeps annotator or
scraper dependencies in `src/shared/`, and rewrites `shared/README.md` for the
split.

## Theme 4: functions kept alive only by their tests

A repeating pattern, mostly in the annotator: production moved to a newer
function and something older stayed behind because a test still calls it.
They fall into three groups:

- Superseded wrappers, re-checked by hand: pick_landing (production uses
  pick_landing_to_end), detect_contacts (production uses
  detect_contact_flags), fetch_span (production streams frames), the
  court-inputs builder, the legacy winner-config loader, and courtkeynet's
  single-frame detect (production uses detect_batch). R5 deletes each wrapper
  and updates or removes its test as specified in the ruling.
- Test helpers living in production files, reported by the read-only sweep:
  calibration floors assertion, score_stage8, two selection functions, and
  court_scale_boxes. R5 keeps these in scope but requires first-hand
  verification before deletion. Each ruling includes its test action.
- Probes nothing consults, reported by the read-only sweep: the API's registry and
  model liveness checks (is_available, available_splits, _live_splits,
  _summary_live). R8 deletes all four probes and the two direct tests at
  `tests/test_api.py:125-136`.

`run_video` also repeats the same resolved span options across five branches.
R5 hoists those options into one local mapping.

## Theme 5: stale knobs from the fixed-25fps era

Four config aliases (`BEST_CONFIG_THRESHOLDS`, `COURT_ABSENT_WINDOW`,
`SUSTAINED_LOSS_FRAMES`, `MIN_DESCEND_SAMPLES`) predate the fps-scaling work.
They now shadow the values production actually resolves. Each live replacement
is named in the ledger. Two scraper download knobs, `CONCURRENT_FRAGMENTS` and
`DOWNLOAD_WORKERS`, had zero consumers because the downloader hardcoded the
same numbers. That pair was re-checked by hand. R3 keeps the constants and
makes the downloader read them, along with the other named yt-dlp settings.
R5 deletes the four stale fixed-25fps aliases.

## Theme 6: half the scripts are finished checks, not tools

The census walked all 77 files across scripts/ and the validation folders.
None is unreachable, but 40 are completed one-off checks from finished work:
the rtmlib migration checks (14 files), the refactoring equivalence checks
(8), the mmpose investigation scripts (6), most of scripts/plots (7 charts
pinned to historical run IDs), and five singles including the mmpose zeroing
equivalence check. They did their job and the docs cite their results.
The `scripts/archive` boundary itself is clean: nothing live imports it. R9
rules against archiving any of the 40 candidates because the larger groups
already sit in named subtrees. The five loose single files also stay in place.

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

Reported by the read-only sweep and requiring a first-hand check before acting:

- _find_rally_spans recomputes scaffolding _rally_regions already returns.
- inout_verdict repeats geometry landing_margins already computes.

One dead block was re-checked by hand: bric's ByteTrack-based
`detect_and_track` player tracker, ~200 lines, has had no caller since the
frame-detection path took over. R8 deletes the block while retaining
`DEFAULT_YOLO_WEIGHTS`.

## What was decided

The rulings in `decisions.md` are the contract for the later refactor:

1. R0 keeps bric working but frozen. Only the mechanical retargets,
   consolidations, and deletions named in R1, R2 and R8 are allowed.
2. R1 moves the authoritative TrackNet tree to `src/shared/tracknetv3/` and
   retargets four consumers. Hand-run commands and literal tooling exclusions
   must also move.
3. R2 splits shared code by consumer. Annotator or scraper dependencies stay
   in `src/shared/`; classifier-only code moves to `src/classifier_shared/`.
   The ruling adopts bst_x court, clip-bound and taxonomy semantics while
   preserving every listed compatibility and stored-data contract.
4. R3 promotes the scraper downloader. A bst_x adapter, an explicit video-only
   mode, dual filename support, completed-output handling, a separate
   resolution metadata module, and config-driven yt-dlp settings are required.
5. R4 leaves the remaining lane-specific mirrors and both manifest readers
   separate. It also keeps the live bst_x confusion-matrix renderer.
6. R5 approves the annotator and CourtKeyNet deletions and three small dedups.
   The four sweep-only helpers require first-hand verification before deletion.
7. R6 approves the two test-helper dedups and two test deletions. It leaves the
   sticky-anchor mirror and legacy call-shape test under their stated conditions.
8. R7 leaves bst_x internals alone except for WP5-1 and WP5-6. The hparam sweep
   readers must retain their fixed-five completeness validation.
9. R8 approves the named bric and API deletions and dedups. It rules that
   WP6-2/3/4/5/11 and the WP6 comparison rows stay unchanged.
10. R9 keeps all 40 script archive candidates in place.

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
  fields, dead parameters), most marked REPORTED.
- 7 duplicates recommended for merging (theme 7), 5 re-checked by hand.
- 40 of 77 census files classified as finished checks to archive,
  classification per file cited in findings/wp7.md.
- Around 20 deliberate mirrors recorded with their reasons, left alone.

Every "re-checked by hand" claim above was confirmed against the full tracked
repository. The check included CI, the served API, and every documented
`python -m` entry point. Two reported claims failed that check and were
refuted. Two were amended. Items marked REPORTED rest on file-and-line evidence
from an automated read-only sweep and must be re-confirmed at refactor time.
