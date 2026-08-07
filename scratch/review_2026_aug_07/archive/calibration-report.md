> ARCHIVED 2026-08-07: superseded by ../REVIEW.md, which folds this calibration's
> additions (C-1, the serve-setup assessment, the F-8 and facade nuances, section 6
> questions) into the merged final positions. Paths: see ARCHIVE_MAP.md.

# Independent calibration of the maintainability review — main @ 8c67569 (2026-08-07)

Method: I sampled the codebase independently before reading `report.md` closely, and
recorded my positions first (`calibration-independent-positions.md`, beside this file).
Evidence came from my own reads of the core annotator modules plus six Luna Max evidence
workers (five commissioned before reading the report, one after; packets in
`evidence/calibration-jobs/`). Every classification below names the evidence that
supports it.

## 1. Calibration verdict

The first report is accurate and can be trusted. Every P1/P2 finding I re-derived or
adversarially checked held up, several of them findings I had already reached
independently before opening the report (F-2, F-3, F-6, F-7). The refactor-quality,
FPS-SSOT, failure-semantics and do-not-refactor conclusions all survive challenge.

Two gaps, both additive rather than corrective:

- the report missed one live instance of its own headline failure class
  (recorded-vs-executed configuration): the `quiet_start_window` lane silently
  overrides `span_open` (C-1 below);
- its test audit never examined the second-largest annotator cluster, the
  serve-setup tests (62 collected cases) — see section 5.

Per area:

| First-report conclusion | Calibration |
|---|---|
| Annotator maintainability: good shape | FIRST REVIEW HOLDS |
| Scraper maintainability: solid, four deliberate failure policies | FIRST REVIEW HOLDS |
| Refactor quality: genuine, not cosmetic | FIRST REVIEW HOLDS |
| Silent-failure risk: two real gaps (F-1, F-3), rest deliberate | FIRST REVIEW HOLDS, one addition (C-1) |
| Test-suite value: mostly justified, ~10 prunable cases | HOLDS IN DIRECTION; serve-setup cluster INSUFFICIENTLY INVESTIGATED (section 5) |
| Do-not-refactor judgments | FIRST REVIEW HOLDS (all six re-challenged) |

## 2. Independent evidence sample

Chosen before reading the report, by structural importance and change frequency:

- Full reads: run_video.py, rally_segmentation.py, rally/spans.py, rally/cli.py,
  video_outcomes.py, batch_report.py, config.py, resolve.py, fps_constants.py,
  scraper/commentary_pairing.py; targeted reads of calibration/sweep.py,
  gt_scoring.py, e2e_court_annotator.py, rally/serve.py, rally/evidence.py.
- Delegated evidence (Luna Max, read-only, Serena-enabled): facade importer census;
  scraper stage artefact/failure map; FPS-policy duplication census; e2e failure
  handling; namespace-migration test inventory; serve-setup test inventory.
- Test-suite shape: collected counts per file; direct reads of the serve-setup,
  fixtures and taxonomy test surfaces; churn history since July.
- Overlap with the first report's sample was deliberate for the four seams the
  calibration brief names; the serve-setup cluster, the quiet-start lane and the
  rally/evidence.py import mechanics were sampled only by me.

My pre-report positions matched the report on: refactor genuinely good; FPS SSOT real;
manifest landing-options duplication (its F-2); pairing sidecar fallbacks (its F-3);
stale spec anchors incl. the `shuttle_extractor.py:244-249` rot (its F-6); the
`replay_mask=` naming lag (its F-7). I had under-weighted what became F-1: my notes
recorded "failures become recorded skips" without following through to the
all-skipped + exit-0 + header-only-CSV terminal state. The report's F-1 is the better
analysis; that disagreement resolved in the report's favour.

## 3. Original conclusions that survived challenge

Conclusions I actively tried to falsify, with the evidence that ended the attempt:

- **F-1 (P1, all-skipped batch exits 0).** Verified the full mechanism: per-video
  `except Exception` (rally/cli.py:228), only `all_excluded_error` gates the CSV write
  (cli.py:249-251), header-only CSVs then feed pairing as "no rallies". Both pinning
  tests exist as claimed (test_batch_report.py:129-145; test_fps_cli_and_tracknet_modes.py:58-103).
  Falsification attempt: "the batch report makes it loud enough". It fails because the
  report file and stdout are only read by a human; the exit code is what a wrapper
  script sees, and the team's own all-excluded guard (ba4e750) already establishes that
  empty outputs deserve refusal. P1 stands, with the caveat that severity rests on
  unattended/scripted runs of the two-stage chain.
- **F-2 (P1, manifest retypes landing options).** I found this independently before
  reading the report (`_configuration_values` restates LANDING_OPTIONS' five values and
  hard-codes `"dead_mask_mode": "replay"`; e2e_court_annotator.py:632-650) and confirmed
  c32c492 fixed the identical class for court policy with a pinning test while landing
  fields have none.
- **F-3 (P2, pairing sidecar fallbacks).** Independently confirmed: `_load_chunks` and
  `_load_replay_mask` return empty values with no log line (commentary_pairing.py:219-236),
  unlike the rally CLI loaders which log INFO. The P2 (not P1) grading is right:
  docstring-documented, malformed-present files raise, per-video pair counts are logged.
- **F-4 through F-10.** Each verified mechanically: the four-slot accumulators
  (gt_scoring.py:576-582), the plot script reading renamed-away `boundary_crowns.csv`
  against `best_config_comparison.csv` (sweep.py:464), the sanitiser's silent `continue`
  on malformed records (experiment_records.py:221-223), the gitignored
  `local_scratch/.../scraper_spec.md` anchors, `replay_mask=` receiving the definitive
  exclusion mask (run_video.py:325-333, 417-426), the three retyped extension sets, the
  inline count-form F1 (selection.py:53-57), and the red local pyrefly gate (reproduced:
  3 errors, 2 in scope).
- **FPS SSOT.** My duplication census specifically hunted for independent fps formulas
  in src/annotator and src/scraper and found none; every fps-dependent value routes
  through `scale_for_fps`/`ScalingKind`. This is the strongest-verified claim in the
  report.
- **Do-not-refactor list, all six entries.** run_video's breadth: exactly four
  production callers, all sweep/measurement-facing — a config object would add a layer
  and remove nothing. The facade: my full importer census (Serena references + text +
  AST scan, dynamic surfaces included) found 14 test files, five production modules
  (run_video, replay_mask, resolve, sweep, types type-only) and three runnable scripts
  under docs/ importing it — it genuinely is the import seam. One nuance the report did
  not state: about half of the ~70 re-exports (census: 35 names, e.g.
  CONTACT_DEDUP_RADIUS_FRAMES, BODY_UNIT_HALF_WINDOW, QUIET_START_REST_FRACTION) have
  no executable user outside the facade. Their carrying cost is near zero, so this
  changes nothing about the keep verdict; it is only worth knowing if the import block
  is ever tidied. thresholds=None opt-out: equivalence-pinned; removal is ~30 edits for
  readability only. Scraper's four failure policies: the stage map confirmed each is
  distinct and deliberate (transcript >50% + mid-batch check, triage circuit-break,
  cleaning all-attempted-failed, downloader post-hoc 0.5). SHARED_FILES swap: commented,
  try/finally-guarded, single-process harness. Double resolve: deterministic and cheap.
- **Dead loops.** `court_scale_slots` parked-by-ruling with #33 as live prospect —
  reasonable on both grounds; `_pick_one_frame` alias delete — verified test-only.
- **test_namespace_migration KEEP.** I challenged this beyond the report's
  "already deliberately pruned" history argument: the current 38 cases are artefact/
  schema pins (T6/T8/T10) plus five legacy-name text scans (T11) with allow-lists.
  The allow-lists have needed only two touches since introduction, so the maintenance
  cost the KEEP trades against is real but small. KEEP survives on present-day
  protection, not just history.
- **Live-issue table (§7).** Independently re-derived from `gh issue list`; the six
  selections and the friction mappings (#38→F-7, #32→F-4, #33→court_scale_slots) are
  the same ones I reached.
- **WIP collision.** Confirmed: the only unmerged branch
  (feature/annotator-unification, 2026-07-12) touches courtkeynet validation scripts
  only. All findings UNAFFECTED, including the new C-1.

## 4. Conclusions that should change

### C-1 [P2] quiet_start_window silently overrides span_open; the conflict the repo validates elsewhere is unguarded here

**Confidence:** High
**Files/symbols:** rally/spans.py:362-369 (precedence chain); run_video.py:191-194
(validates serve_start conflicts only); config.py:130 (`span_open=SpanOpen.BACK_FILL`
default); calibration/sweep.py `_base_and_serve` (sets span_open only when swept).
**Relevant callers/consumers:** any quiet-start candidate spec routed through
`run_candidates`/`production_candidate_runner`; `segment_video`.

**Original position:** the first report does not mention the interaction; its §8 entry
cites quiet_start_window as a live sweep dimension while dropping F-2-class vigilance at
this spot.

**Independent evidence:** `find_rally_spans` dispatches serve_start > quiet_start >
span_open; the quiet-start finder does not take span_open at all. `BaseAnnotatorConfig`
defaults `span_open=BACK_FILL`, and `_base_and_serve` leaves that default in place for
any candidate that sweeps `quiet_start_window` without a span_open strategy — so such a
candidate's config carries BACK_FILL (and its winner delta, which omits unswept keys,
resolves to BACK_FILL on any reading) while execution runs quiet-open semantics. The sibling conflicts are all loud: serve×REGION_START and
serve.close×BACK_FILL raise in `segment_video`, serve×quiet raises in `run_video`
(pinned by test_annotator_config_strategies.py:127-135). This pair is the one silent
member of the family.

**Strongest counterargument:** no pinned grid sweeps `quiet_start_window`
(BOUNDARY_KEYS/CONTACT_KEYS are threshold-only), so nothing currently executes the
combination; the winner round-trip is also internally consistent (delta-recording, same
routing on reload), so no persisted artefact misstates what a rerun would execute.

**Revised position:** add the missing guard — raise in `segment_video` (or
`_validate_run_inputs`) when `quiet_start_window` is set and `span_open` is not None,
matching the three existing lane-conflict errors. One test mirroring
`test_quiet_start_and_serve_start_fail_in_run_video`. Latent today, but it sits on the
exact class (recorded-vs-executed, silent lane precedence) the report rates P1 elsewhere,
and the fix is a two-line convention-following change. Fold into the report's PR 3
(small-cleanups sweep).

Illustrative, beside run_video.py:193:

```python
if resolved.quiet_start_window is not None and resolved.span_open is not None:
    raise ValueError('quiet_start_window cannot be combined with span_open')
```

**Classification:** INSUFFICIENTLY INVESTIGATED (an addition; nothing the report states
is wrong).

### C-2 [note] F-8's FINE_BATCH_SIZE item conflates two knobs

**Original position:** F-8 lists "`FINE_BATCH_SIZE = 16` defined but `_score_chunks`
hardcodes `batch_size=16` (commentary_cleaning.py:169)" as one drift-risk pair.

**Independent evidence:** `FINE_BATCH_SIZE` feeds WhisperX transcription
(commentary_cleaning.py:369); line 169 is the BERTScore scorer's batch size. Same
number, different consumers with independent tuning pressure.

**Strongest counterargument:** both are "fine-pass batch size" in spirit, and either
naming fix removes the magic number.

**Revised position:** still fix it, but by naming a separate scorer constant, not by
pointing both at FINE_BATCH_SIZE — sharing one constant would create the false shared
concept the report elsewhere warns against.

**Classification:** FIRST REVIEW HOLDS with a corrected fix shape.

### C-3 [note] rally/evidence.py's sys.path mutation was not examined

**Original position:** not mentioned anywhere in report.md or worklog.md.

**Independent evidence:** rally/evidence.py:14-18 inserts `src/bst_x` into `sys.path`
at import time to reach `preparing_data.heuristics.sticky_anchor`, with a comment
declaring the single-implementation intent. Importing any annotator module that touches
the rally package therefore mutates global import state and makes `preparing_data` and
`pipeline` importable top-level everywhere in the process — the same dual-module-identity
class the report's F-10 flags for `src.`-prefixed test imports.

**Strongest counterargument:** it is commented, deliberate, and keeps the anchor picker
single-sourced; a packaging fix (importing bst_x as a proper package) is blocked by
bst_x's internal flat-import convention, so the smallest real change is a package
restructure that is out of proportion today.

**Revised position:** leave the code; record it as a known import-order hazard rather
than a work item (see section 6). The calibration point is only that the report's
maintainability sweep did not look at it.

**Classification:** INSUFFICIENTLY INVESTIGATED (minor; no action recommended).

## 5. Test-suite calibration

The report's direction — "mostly justified, ~10 prunable cases, no campaign" — held for
every cluster it examined; my spot-checks of its KEEPs (run_video behavioural tests,
fixtures/taxonomy provenance pins, calibration schema pins, namespace-migration) all
agreed. Its audit, however, deep-read three suspect files and never assessed
tests/test_annotator_serve_setup.py + _b2.py — 62 collected cases, the second-largest
annotator cluster, with a visible validator-matrix shape.

### Test cluster: serve-setup gate tests

**Tests / parameter family:** tests/test_annotator_serve_setup.py (38 collected),
tests/test_annotator_serve_setup_b2.py (24 collected).
**Collected cases:** 62 across 31 functions.
**Production symbols:** rally/serve.py — serve_setup_still, series_drift,
build_serve_setup_inputs, ServeSetupInputs.validate, _resolve_serve_gate,
_sticky_serve_setup_before.
**Maintained callers:** run_video.build_serve_options → build_serve_setup_inputs
(run_video.py:48-66); the serve gate inside span finding (spans.py:276-299,
serve.py:247-317).
**Why independently selected:** second-largest annotator cluster by collected count;
visible validator-matrix shape; absent from the first report's audit.

**Observable behaviours:** three distinct families. (a) ~15 cases of gate semantics —
lane routing by standing count, per-slot distance/height pairing, burst-frame
exclusion, inclusive claimed frame, drift split rules. (b) ~13 cases of fail-closed
semantics on production-shaped sentinel states (NaN ankles, missing analysed coverage,
zero body unit, clipped windows) — the sticky producer genuinely emits these sentinels
(evidence.py:153-172). (c) ~34 collected checks of input-validation matrices.

**Existing overlap:** none — no other file tests the serve gate; run_video tests
exercise only the builder hand-off.

**Historical evidence:** the gate landed test-first (305b3ad: "building blocks land,
not yet called by anything"), and two real bug fixes are pinned by these tests:
fail-open on an unmeasurable player (305b3ad) and double normalisation plus
wrong-player distance/height pairing (cc02b62; b2:56 pins the pairing directly).
Groups (a) and (b) protect exactly the failure modes that have actually occurred.

**Evidence for keeping:** (a)+(b) fully; and two (c) families that guard live
boundaries: the builder's resolution validator (run_video validates only *presence*
of resolution — run_video.py:216-230 — so caller-supplied zero/NaN/wrong-shape
resolution really reaches it) and the cross-field threshold check (a malformed
ServeStartConfig is only typed, not value-checked, at types.py:69-75).

**Evidence for pruning:** the remaining (c) matrices reject states unreachable from
the production path: series_drift shape/dtype (4 cases — its only production caller
passes builder-validated `(t, 2)` float arrays), serve_setup_still's
window/claimed-frame/threshold/slots matrices (17 cases — production supplies
FPS-derived windows, span-detected frames and literal slot tuples through
`_resolve_serve_gate`, which already validates at serve.py:369-386), and
ServeSetupInputs dtype/shape matrices (~8 cases — the sticky producer allocates fixed
shapes and dtypes). These pin defensive rejections of caller garbage no maintained
caller can produce; one representative case per family preserves the "validator still
fires" signal.

**Assessment:** `INSUFFICIENTLY INVESTIGATED` (first report; the cluster is absent
from its audit) → on my evidence, mostly KEEP with a bounded RESHAPE.

**Action:** RESHAPE — collapse each synthetic-only rejection matrix to one
representative case; keep (a), (b), the resolution family and the cross-field family
untouched.

**Estimated effect:** collected cases removed ~20 (62 → ~42); test functions removed
0 (parametrisation shrink only); production code removable: none.

**Protection after change:** every validator still exercised once; all measured gate
semantics and fail-closed behaviour retain full coverage; the two live boundary
validators retain their full matrices.

**WIP collision:** UNAFFECTED (only unmerged branch touches courtkeynet scripts).

This adjusts the first report's "~10 prunable collected cases" to roughly 30 if the
team wants the reshape, without changing its central verdict: the suite is healthy and
no pruning campaign is warranted. The reshape is optional — these matrices cost
nothing per run and churn only when the validators churn; the case for collapsing them
is legibility of the cluster, not runtime or maintenance relief.

One naming residue the report missed in its rename sweep: `test_annotator_serve_setup_b2.py`
keeps a legacy batch label ("b2") that the responsibility-names pass (#51/#55) eliminated
everywhere else; the report caught the analogous "stage8" residues but not this one.
Rename-only, alongside its two misnamed-file renames.

## 6. Unresolved questions

Evidence gaps where neither KEEP nor CHANGE is justified; recorded, not work items:

- **F-1 severity rests on usage.** Whether the rally CLI actually runs unattended
  (wrapper scripts, remote batches) decides P1 vs P2. The one-line guard is justified
  either way, so the answer changes labelling, not action.
- **rally/evidence.py sys.path hazard (C-3).** Whether any realistic in-process
  combination imports `preparing_data`/`pipeline` via two paths (bst_x tools plus
  annotator) was not established; nothing in-repo does so today under pytest collection.
- **Whether quiet-start should ever compose with BACK_FILL.** C-1 proposes rejecting
  the combination; if the team instead intends quiet-start as a refinement *of*
  back-fill opening, the right fix is different (compose, not reject). The intent
  question is the team's to answer; the silent status quo is the only wrong option.
- **contact_frames.csv** has no in-repo reader (both my stage map and the first
  report's delegates agree). The report marked it "likely external data product —
  INVESTIGATE"; that remains the honest state.
