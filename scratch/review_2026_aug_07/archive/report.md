> ARCHIVED 2026-08-07: superseded by ../REVIEW.md, which merges this report with
> calibration-report.md into final positions. Every finding here carries into
> REVIEW.md (calibration adjustments applied). Paths: see ARCHIVE_MAP.md.

# Maintainability and test-pruning review — main @ 8c67569 (2026-08-07)

Evidence: worklog.md (process + all candidate findings), evidence/jobs/*/result_extracted.md
(10 Luna delegate census/verification results). Every kept finding was inspected first-hand;
the three P1/P2-critical candidates were additionally adversarially verified by independent
read-only delegates instructed to refute them.

## 1. Executive verdict

**Annotator:** in good shape. The July–August refactors (#55, #58, #59, #60, #62, #63) did what
they set out to do: naming is now responsibility-based with almost no residue, FPS scaling has a
genuine single source of truth, run_video is staged and behaviour-preserving (zero test-file
changes in #59), and the rally split put real logic in focused modules behind a facade that earns
its keep as the import and test seam. No area REGRESSED. The two problems worth fixing promptly
are failure semantics, not structure: an all-skipped batch exits 0 with legitimate-looking empty
CSVs (F-1), and the e2e measurement manifest retypes landing options instead of recording the
executed object (F-2) — the same recorded-vs-executed class the team already fixed for court
policy.

**Scraper:** solid. The four batch stages have four *different* mass-failure policies, and on
inspection each is deliberate and mostly documented; the remaining gap is that missing sidecars
at the pairing stage degrade with only an undifferentiated INFO line (F-3, P2).

**Refactor progress:** genuine, not cosmetic. Prior-review items are FIXED or IMPROVED except a
cluster of calibration/e2e internals (positional score_video contracts, SHARED_FILES swap,
sanitiser skip-then-delete) that were documented rather than restructured — acceptable for now,
but issue #32 will collide with the positional contracts (F-4).

**Test suite:** the 1394 tests are mostly justified. The suspicious clusters (mock-heavy
measurement tests, patched-decomposition run_video tests) survived inspection: the seams feed
assertions on observable outputs, and several "smelly" surfaces are deliberately pinned
(thresholds=None equivalence, persisted CSV schemas). The honest prunable total is ~10 collected
cases plus two file renames — there is no bloat problem worth a campaign.

## 2. Refactor scorecard

Baseline: scratch/swarm_review/readability_refactor_scoping_handoff.md (anchored 1afc86a, 2026-08-04).

| Prior-review area | Verdict | Evidence |
|---|---|---|
| Mask filename chains (dead vs replay) | FIXED | verified both producer→consumer chains consistent; helper renamed (`_load_dead_mask`, rally/cli.py:42) |
| Mirrored config values in run metadata | FIXED | c32c492: e2e imports court_evidence constants; test pins manifest == executable court policy |
| Shared frame-rate scaling rules | IMPROVED | fps_constants base-30 table + resolve() is a clean SSOT; residue: gt_scoring.canonical_tolerance (:403) re-scales 5.0 outside CONTACT_TOLERANCES_BASE30; replay_mask fns re-derive scale_for_fps internally (:72, :212) |
| run_video orchestration | IMPROVED | #59 staged helpers read coherently; PR changed zero test files (behaviour-preserving); ~30-param breadth remains but is the sweepable composition root (deliberate — see §8) |
| rally_segmentation split | IMPROVED | #58 moved real logic to rally/*; facade remains the import surface (14 test files + resolve.py:10, replay_mask.py:34) and the monkeypatch seam — keep it (§8) |
| Historical/pipeline naming | IMPROVED | renames thorough (delegate 06: zero old-name matches in code); residue: stale doc anchors (finding F-6), plot_stage8_tradeoffs.py name+input (F-5), `replay_mask` param at the segmentation seam (F-7), one test name "stage8" |
| score_video positional accumulators | STILL A PROBLEM | four-slot lists + positional RallyRow build remain (documented only) |
| e2e SHARED_FILES module-global swap | STILL A PROBLEM | e2e:1156-1162, commented NOT THREADSAFE |
| experiment_records sanitiser skip/delete | STILL A PROBLEM | silently skips malformed record; scanner deletes leaf after backup |
| run_clean _score_pending lifecycle | STILL A PROBLEM (readability only) | traced: behaviourally correct |
| scraper _download_one seam | IMPROVED | _verify_existing extracted; explicit unlink policy |

## 3. Prioritised findings

Ten findings kept, four recorded as DEFER (end of this section). Dropped outright after
verification: run_video API breadth (deliberate sweep design); thresholds=None dual path
(guarded by an equivalence test); "crowns" naming (deliberate rename — the residual consumer is
F-5); scraper config re-export (documented cross-pipeline contract).

### [P1] F-1 — an all-skipped annotator batch reports success

**Confidence:** High (adversarially verified, delegate 09; my own read agrees)
**Files/symbols:** rally/cli.py:201-234 (skip paths), :248-263 (only `all_excluded_error` checked);
batch_report.py:39-53 (no processed-count branch); config.py:27-28 (output CSVs).

Evidence: a batch where every video is skipped — FPS-CSV id mismatch, or `run_video` raising per
video — logs each skip, prints "batch completed: 0 of N", writes both output CSVs as header-only
files, and returns normally (exit 0). The contrast is explicit in the same file: an all-EXCLUDED
batch (doubles filter) raises and refuses to write CSVs (cli.py:141-145, :248-263; commit ba4e750
calls this the "refusal to write empty outputs").
Why it matters: header-only `rally_spans.csv` is a legitimate-looking artefact that downstream
pairing will consume as "no rallies", so one bad FPS CSV can silently blank the pipeline's main
data product. In unattended batch runs the exit code is the only signal anyone checks.
Best counterargument: two tests pin the current behaviour (test_batch_report.py:129-145 asserts
"batch completed: 0 of 1" completes normally; test_fps_cli_and_tracknet_modes.py:58-103 likewise),
and the per-video log-and-skip comment (cli.py:228) is deliberate. Per-video continuation is
clearly intended — but nothing suggests the *all*-skipped terminal state was separately considered,
and the all-excluded guard shows the team already decided empty outputs deserve refusal.
Verdict: KEEP FINDING.
Smallest change: after the loop, mirror the all-excluded guard — if no video was processed, raise
(keeping the report so the skip reasons still print). Illustrative shape, beside cli.py:141-145's
existing guard:

```python
if processed_count == 0 and all_excluded_error is None:
    raise RuntimeError('batch processed 0 of %d videos; refusing to write empty outputs' % video_count)
```

Smallest verification: retarget the two pinning tests to expect the raise; keep the partial-skip
tests unchanged.

### [P1] F-2 — e2e manifest retypes landing options instead of recording the executed object

**Confidence:** High (adversarially verified, delegate 08)
**Files/symbols:** e2e_court_annotator.py:79-85 (LANDING_OPTIONS), :633-650 (`_configuration_values`),
:909/:1124 (manifest insertion), :984-997 (executed call); calibration/gt_scoring.py:464-472
(second construction of the same five values); point_winner.py:355-363 (required fields).

Evidence: the five landing-filter values (7, 0.004, 5, 7, 0.75) are typed in three places —
e2e's LANDING_OPTIONS, gt_scoring's build_run_video_inputs, and again as literals inside
`_configuration_values`, the function that writes the run manifest. `_configuration_values` does
not reference LANDING_OPTIONS at all, and no test compares the manifest to the executed options
(the existing manifest test covers court policy only, test_annotator_measurement.py:58-62). The
manifest's `"dead_mask_mode": "replay"` is likewise a raw string that happens to equal the
resolved default enum value, with no assertion linking them. `ref_err_px` = 3.5 exists as both an
e2e constant and an independent run_video default.
Why it matters: the measurement harness's entire value is that the manifest states what actually
ran. Editing LANDING_OPTIONS (or the resolved dead-mask default) without touching
`_configuration_values` produces manifests that misreport executed configuration — the exact
recorded-vs-executed failure class commit c32c492 fixed for court policy, with a pin test. Landing
options were simply missed.
Best counterargument: `_configuration_values` already derives `ref_err_px` from REF_ERR_PX and
`landing_horizons_s` from LANDING_HORIZONS (:641, :650), so the pattern exists and the values are
currently consistent; LandingFilterOptions' five fields are required keywords, so the shape at
least cannot drift silently.
Verdict: KEEP FINDING.
Smallest change: (1) make `_configuration_values` read the landing fields from LANDING_OPTIONS
and the dead-mask string from the resolved config's enum value; (2) give the five values one home
— gt_scoring imports e2e's LANDING_OPTIONS or both import a shared constant; (3) extend the
existing court-policy manifest test to the landing fields. Illustrative:

```python
'landing_filter_options': dataclasses.asdict(LANDING_OPTIONS),
'dead_mask_mode': resolved.dead_mask_mode.value,
```

Smallest verification: the extended manifest test plus one e2e smoke case.

### [P2] F-3 — eligible videos with missing sidecars pair silently (documented fallback, no distinct signal)

**Confidence:** High (adversarially verified, delegate 10; severity revised down from P1)
**Files/symbols:** commentary_pairing.py:219-236 (loaders), :151-185 (None-mask bypass),
:341-364 (loop + unconditional final write); download_scraped_videos.py:260-273
(`commentary_eligible` manifest flag); relevance_triage.py:236-261 (chunks written by a different stage).

Evidence: eligibility comes from `sources.toml` plus spans/FPS, not from the chunk sidecars, so an
eligible video whose triage chunks or replay mask never landed is possible. Absent chunks → `[]`
(zero pairs); absent mask → replay filtering silently disabled (pairs produced unmasked). Both
absent-file branches return without logging. There is no batch-level guard; the final pairs CSV is
always written.
Why this is P2 and not P1: the fallbacks are documented in the loader docstrings, `pair_video`
types the mask as nullable, malformed *present* files raise loudly, and every video logs an INFO
line "N rallies, M paired" — so the information is in the log, just not distinguished from the
legitimate "commentary matched nothing" case.
Best counterargument: the mask genuinely is optional by design (a test asserts pairing works with
`None`), and commentary pairing is a best-effort enrichment product, not a correctness gate.
Verdict: KEEP FINDING at P2.
Smallest change: one WARNING in the batch loop when a manifest-eligible video has no chunks
sidecar (and optionally when it has no replay mask), so "sidecar missing" reads differently from
"nothing matched". No new machinery.
Smallest verification: one test asserting the warning fires for an eligible video without chunks.

### [P2] F-4 — score_video's positional contracts make GT-schema change expensive

**Confidence:** High
**Files/symbols:** calibration/gt_scoring.py — four-slot positional accumulators (:576-582),
vestigial one-iteration loop (:624), positional 27-field RallyRow build (:669-674), positional
VideoScoring build (:677-682).
**Callers:** e2e `_score_configurations` → `score_video`; sweep scoring path.

Evidence: each metric family is accumulated in bare 4-slot lists (`br = [0, 0, 0, 0]`) whose slot
meaning exists only in the reader's head, then unloaded positionally into a 27-field RallyRow.
The count-gate assertion at :675 reconciles two independent recounts — that part is a deliberate
guard and should stay.
Why it matters: issue #32 (real rally start/end GT) lands exactly here; adding one GT column means
threading positional edits through every accumulator and both positional constructor calls.
Best counterargument: PR #63 already documented these contracts, the code is correct today, and a
keyword/dataclass rewrite is pure churn until a schema change is actually scheduled.
Verdict: KEEP FINDING — but time the fix to #32, not before.
Smallest change: when #32 starts, convert RallyRow/VideoScoring construction to keyword arguments
and name the accumulator slots (a small NamedTuple or four named counters); keep the count-gate.
Smallest verification: existing calibration schema tests plus one targeted score_video test.

### [P2] F-5 — plot script still reads the renamed-away sweep artefact

**Confidence:** High
**Files/symbols:** scripts/plots/plot_stage8_tradeoffs.py:63-93 reads `boundary_crowns.csv`;
sweep writes `best_config_comparison.csv` (calibration/sweep.py:464).

Evidence: the crowns→best_config rename (deliberate) was not propagated to this consumer; on any
fresh sweep the script raises FileNotFoundError. The script name also keeps the retired "stage8"
label. Why it matters: the plot silently rotted out of the toolchain; next sweep analysis session
hits a dead script. Best counterargument: the script may be considered retired with the old
artefact — in which case delete it rather than update it.
Verdict: KEEP FINDING. Smallest change: update the filename (and ideally rename the script), or
delete the script if it is retired. Smallest verification: run it against an existing sweep output
directory.

### [P2] F-6 — live docstrings anchor to an untracked scratch spec

**Confidence:** High
**Files/symbols:** rally_segmentation.py:1, replay_mask.py:1, config.py:4 (and "spec s6" line
comments, e.g. config.py:28), scraper/config.py:10, scraper/__init__.py:3, commentary_pairing.py:1;
rally_segmentation's docstring also cites `shuttle_extractor.py:244-249` (file moved, line-rotted).

Evidence: the docs cite `local_scratch/autograder_architecture/scraper_spec.md`, which is
untracked and stale by the owner's own ruling. Why it matters: a new reader (or the second team
member) follows the anchor and finds nothing, or worse, finds stale intent. Best counterargument:
the anchors are historical provenance and harmless if everyone knows to ignore them.
Verdict: KEEP FINDING. Smallest change: delete the citations or repoint them at the tracked
contracts docs from PR #63/75edb3e. Smallest verification: grep for `local_scratch` and `spec s`
in src/ returning nothing.

### [P2] F-7 — `segment_video(replay_mask=...)` now receives the definitive dead mask

**Confidence:** High
**Files/symbols:** rally_segmentation.py:136 (parameter), rally/cli.py:42 (`_load_dead_mask`
feeding it), run_video.py call sites (:325, :417).

Evidence: PR #63 renamed the CLI loader to `_load_dead_mask` but the API parameter it feeds is
still `replay_mask`, though the value is the definitive exclusion mask including composition
cuts. Why it matters: issue #38 will add a third mask producer; a parameter named for one specific
producer actively misleads at the exact seam new masks join. Best counterargument: the rename
touches a public-ish parameter used by tests and callers — churn now, and the docstring already
explains the semantics. Verdict: KEEP FINDING — cheap, and best done before #38 lands.
Smallest change: rename the parameter (e.g. `exclusion_mask` or `dead_mask`) across the facade,
run_video call sites and tests in one commit. Smallest verification: whole-project pyrefly check
plus grep for the old kwarg.

### [P2] F-8 — small single-source-of-truth drift risks (grouped)

**Confidence:** High (each verified individually; grouped because each is small)
The category-1 (shared concept, can drift) items from the duplication census:

- Video extension set `{'.mp4','.mkv','.webm','.avi','.mov'}` retyped in 3 scraper modules.
- `FINE_BATCH_SIZE = 16` defined but `_score_chunks` hardcodes `batch_size=16`
  (commentary_cleaning.py:169).
- e2e retypes `(512, 288)` beside court_evidence.DETECTOR_RESOLUTION `(512.0, 288.0)` (e2e:503-509).
- Downloader mass-failure threshold `0.5` is an unnamed literal (download_scraped_videos.py:582-589)
  while the other three stage policies are named/documented.
- selection.py:53-57 inlines count-form F1 maths beside the shared `safe_f1`.
- gt_scoring.canonical_tolerance (:403) re-scales 5.0 independently of CONTACT_TOLERANCES_BASE30.

Why it matters: each is a one-edit-forgets-the-other bug waiting; all are one-line fixes.
Best counterargument: none has bitten yet, and some (selection F1) are arguably clearer inline.
Verdict: KEEP FINDING as one batched cleanup commit; drop any item that resists a one-line fix.
Smallest verification: pyrefly check + the existing tests that already cover each site.

### [P2] F-9 — experiment_records sanitiser can skip-then-delete a malformed record

**Confidence:** High
**Files/symbols:** experiment_records.py — `_planned_json_changes` silently skips malformed
configuration records (:221-223) that `build_summary` would raise on; `_scanner_findings` then
lets `clean_run` delete the unsanitised leaf after a tar backup (:336).

Evidence: a malformed record is exempted from sanitisation but not from deletion; the delete is
printed and tar-backed, so recovery exists but nothing flags the inconsistency. Why it matters:
the one file most likely to need inspection (the malformed one) is the one that leaves the tree
unsanitised. Best counterargument: deletion happens only after a backup, the action is printed,
and malformed records are rare; the pipeline is operator-driven. Verdict: KEEP FINDING at P2.
Smallest change: make the sanitiser fail loudly (or explicitly skip-and-report) on a malformed
record instead of silently passing it to the scanner. Smallest verification: one test with a
malformed record asserting the loud path.

### [P2] F-10 — local pyrefly gate is red on two test imports

**Confidence:** High (verified first-hand)
**Files/symbols:** tests/test_fps_constants.py:11,14 (`from src.annotator....`); pyproject.toml:166-177
and ci.yml:34-37 (CI narrows pyrefly to unknown-name, so CI stays green).

Evidence: whole-project `pyrefly check` — the stated pre-commit gate — reports these two errors on
every run, training the team to ignore a red gate. The `src.` prefix also imports a second module
object (`src.annotator.config` vs `annotator.config`), so identity-based assertions can silently
diverge. Best counterargument: CI is green by deliberate, documented narrowing; tests pass.
Verdict: KEEP FINDING. Smallest change: drop the `src.` prefix on both lines (and, in passing, in
test_fps_cli_and_tracknet_modes.py:200,252,312 for consistency). Smallest verification:
`pyrefly check` drops to the single out-of-scope bric error; pytest on the two files.

### DEFER (recorded, no action recommended now)

- **BaseAnnotatorConfig vs ResolvedAnnotatorConfig duplicated defaults** (span_open,
  rejected_grades) — drift possible but both sit in config.py within sight of each other.
- **rally_segmentation.py:205-215 dead identical if/else** — both branches make the same call
  (`sticky_distances` is already None in the else arm); fold into F-8's cleanup commit if touched.
- **e2e SHARED_FILES module-global swap and positional pin indexing** — see §8.
- **run_clean `_score_pending` lifecycle readability** — traced correct; per-video/per-chunk
  asymmetry is documented and deliberate.

## 4. Test-pruning audit

Overall verdict: the 1394-test suite mostly earns its keep. The three suspect files I deep-read
(test_annotator_measurement, test_annotator_run_video, test_fps_cli_and_tracknet_modes) are
largely behavioural; the monkeypatching serves failure-semantics and provenance contracts rather
than pinning private decomposition. The honest prunable total is small (~10 collected cases).
No large-scale pruning is justified.

### Cluster: retired-option CLI tests — DELETE

**Tests:** `test_rally_segmentation_main_rejects_retired_options` (test_fps_cli_and_tracknet_modes.py:144-161), 6 parametrised cases.
**Production symbols:** none — grep finds no retired-option handling in rally/cli.py or rally_segmentation.py.

Behaviour protected: that argparse rejects `--gate-dir`, `--pose-dir`, `--homography-csv`,
`--resolution-csv`, `--court-box-csv`, `--thresholds`. That is stock argparse behaviour for any
unknown flag; there is no production code behind these cases. The realistic regression (someone
re-adds a retired option deliberately) would be an intentional change, not a bug this test catches.
- Collected cases removed: 6. Test functions removed: 1. Production code removable: none.
- Protection remaining: argparse rejects unknown args universally; the retirement is documented in commit 3f6e34d.

### Cluster: sticky_anchor `_pick_one_frame` alias — DELETE WITH PRODUCTION CODE

**Tests:** identity assertion at tests/test_sticky_anchor.py:459 (plus alias uses in that file).
**Production symbols:** `_pick_one_frame = pick_one_frame` (src/bst_x/preparing_data/heuristics/sticky_anchor.py:321).

Production uses `pick_one_frame`; the underscore alias exists only for tests, and one test asserts
the alias is the same object — a helper justified solely by its own test. (bst_x sits outside the
review's src scope, but the test file is in scope.)
- Collected cases removed: 1 (identity test); remaining tests re-point at `pick_one_frame`.
- Production code removable: the alias line.

### Cluster: `src.`-prefixed imports in tests — RESHAPE (2-line fix)

tests/test_fps_constants.py:11,14 import `src.annotator.*` while line 12 imports `annotator.*`.
This is the local pyrefly gate's only in-scope failure (2 of 3 errors) and creates dual module
identity at runtime (`src.annotator.config` and `annotator.config` are distinct module objects, so
constants pinned by identity can diverge). CI stays green because it narrows pyrefly to
unknown-name only (pyproject.toml:166-177). Same `src.` family appears in
test_fps_cli_and_tracknet_modes.py:200,252,312 (`src.bric`, `src.bst_x`). Fix imports; delete nothing.

### Cluster: constant-echo matrix test — optional DELETE

`test_fixed_matrix_and_parent_order_are_deterministic` (test_annotator_measurement.py:112-124)
restates the CASES/PARENTS literals. Any deliberate matrix change edits the test in lockstep; the
only regression caught is an accidental reorder, which the per-configuration manifests would also
surface. Marginal — delete or leave (1 case).

### Cluster: misnamed test files — RESHAPE (rename only)

test_scraper_doubles_flag.py and test_scraper_composition_mask.py both target src/annotator
modules. Rename to match; no content change.

### Kept after inspection (highest-suspicion clusters that survived)

- **test_annotator_run_video.py (38 tests): KEEP.** Spy tests call through to the real builders and
  assert real data handoffs (original-track-before-replay-mask, tracker-segments handoff, sticky
  built once) with a justifying docstring at :217-221. Mode-validation matrices (15 cases) are
  cheap contracts for a ~30-param composition root used by sweep, e2e and the CLI.
- **test_annotator_measurement.py (remaining 20): KEEP.** The seams feed assertions on observable
  outputs — exit codes, manifest statuses, artefact md5s, failure-isolation scope — which is the
  measurement harness's provenance job.
- **thresholds=None surface: KEEP** (see §8). `test_thresholds_none_matches_explicit_shipped_preset`
  pins the module-globals path bit-for-bit against SHIPPED_THRESHOLDS, so drift is loud.
- **test_calibration_schemas: KEEP** — pins persisted CSV/JSON contracts.
- **test_namespace_migration: KEEP** — already deliberately pruned (2026-06-16, T1-T12 → 4 families).
- **validation_overlay tests: KEEP** — real-ffmpeg media tests; issue #31 makes the tool live tooling.
- **Scraper batch tests: KEEP** — patch the per-video worker (triage_video, acquire_transcript,
  _clean_once, _download_one) which is the right seam for batch-policy tests.

## 5. Confirmed dead test/function loops

Only after bidirectional tracing (production callers → test callers → history):

1. **`court_scale_slots` (rally/evidence.py:42)** — 0 production callers, 1 facade re-export, 1 test.
   Sibling `court_scale_boxes` was deleted in 51b2977; slots survived by explicit ruling (W2.4).
   Counterargument, and it is a real one: open issue #33 (court-space distance normalisation) is a
   plausible near-term consumer of exactly this helper. Verdict: dead loop today, but retention was
   an explicit ruling and #33 gives it a live prospect — leave until #33 is decided.
2. **`_pick_one_frame` alias + identity test** — dead loop, delete both (§4).
3. **Checked and NOT dead:** `test_first_last_stroke_buffered_search` → scripts analysis tool
   (explicit keep ruling, 2026-08-02 R9); `convert_landing_options` and
   `corner_error_band_from_corners` (real production callers); the scraper/config re-export block
   (scraper/config.py:21) — `CONTACT_FRAMES_CSV` has no scraper consumer and an identity test
   (test_scraper_search_index.py:24-27), but the block deliberately documents the cross-pipeline
   artefact paths; harmless, keep.

## 6. Silent-failure map

Verified paths; error → outcome:

| Path | Trigger | Outcome | Loud or silent |
|---|---|---|---|
| rally/cli batch | doubles filter excludes every video | raises, refuses to write CSVs | LOUD (deliberate guard) |
| rally/cli batch | every video skipped (fps CSV id mismatch, or run_video raises per video) | header-only CSVs written, "batch completed: 0 of N", normal exit 0 | SILENT for automation (0/N visible only to a human reading the report) — finding F-1 |
| rally/cli per video | bad track / invalid mask | log-and-skip, recorded in batch report | LOUD enough (recorded) |
| replay_mask CLI | missing court mask / homography / track sidecar | that signal all-False, INFO log, mask still written | DELIBERATE fallback; mask records nothing about which signals ran |
| commentary_pairing | eligible video missing chunks sidecar or replay mask | blank pairings (chunks=[]) or unmasked pairings (mask=None); INFO "N rallies, 0 paired" only | SILENT-ish (docstring-documented fallback; absent→fallback but malformed→raise) — finding F-3 |
| commentary_pairing | missing fps for a video | warning + skip; empty pairs CSV still written if all skipped | mixed |
| commentary_pairing | manifest missing/invalid entry for fps-bearing video | raises | LOUD |
| transcript_acquisition | corrupt caption JSON | uncaught exception kills batch | LOUD (but inconsistent with per-video skip policy) |
| transcript_acquisition | >50% failures past floor | raises mid-run | LOUD (deliberate) |
| relevance_triage | corrupt transcript sidecar | uncaught, kills batch | LOUD |
| download stage | ≥50% of attempted downloads fail | post-hoc guard: exit 2 (download_scraped_videos.py:582-589; threshold is an unnamed literal) | LOUD (deliberate; policy differs per stage) |
| commentary_cleaning | all attempted LLM calls fail | raises | LOUD (deliberate) |
| experiment_records clean_run | malformed configuration record | sanitiser skips it; scanner may DELETE the unsanitised manifest (after tar backup) | SILENT-ish (printed; backup exists) |
| e2e measurement | per-configuration failure | failure.json + terminal manifest, run continues | LOUD (well-engineered) |
| sweep plots | fresh sweep output + plot_stage8_tradeoffs.py | FileNotFoundError (reads renamed-away boundary_crowns.csv) | LOUD but orphaned consumer |

## 7. Future-change pressure check (live issues)

Six open issues place near-term change pressure on the reviewed code. (#50 "Break up the
monoliths" is itself a refactor request, so it is context only — though its stated worry, that
VLM on/off-ramps will stress the current seams, matches what #38 shows below.)

| Issue | Likely change shape | Current landing point | Local today? | Architectural friction |
|---|---|---|---|---|
| #38 VLM scene filtering | new mask/evidence producer feeding exclusion | new sidecar + mask concept → dead_mask dispatcher → run_video `raw_exclusion_mask` | Mostly | The mask pipeline composes cleanly (per-concept masks, one dispatcher). Friction is naming: `segment_video(replay_mask=...)` already receives the definitive dead mask (F-7); a third mask source makes that lag actively confusing. |
| #33 court-space distances | parallel court-space positions for sticky attribution | rally/evidence.py (sticky build) + point_winner attribution | Mostly | Contained in the evidence/attribution seam. Note: dead-today `court_scale_slots` (§5) is a plausible building block — a reason not to delete it yet. |
| #32 real rally start/end GT | new GT columns + new scoring dimension | calibration/gt_scoring.py + rally/serve.py | No | Directly stresses the positional 27-field RallyRow build and four-slot accumulators (F-4): adding one GT column means threading positional edits through score_video. The serve-start seam itself is ready (sweepable config). |
| #40 compressed e2e artefacts | writers emit .xz/.gz; loaders accept both | e2e writer helpers + FilePin md5 verification | Yes | Writers are centralised (`_write_rows` etc.) and manifests pin md5-of-bytes, so compression lands locally. Decide md5-of-compressed vs md5-of-raw once, in FilePin. |
| #31 hallucination guards | investigation now; later small extra guard masks | inpaint_guard / `shuttle_hallucination_mask` threading | Yes | Already well-supported: run_video threads inpaint_codes and the hallucination mask, and tests pin the threading (test_run_video_threads_event_mask_to_dead_mask_builder). |
| #12 fps-normalise uploads | pre-pipeline normalisation step | fps_constants/resolve | Yes | Well-supported: the base-30 SSOT means a normalised input needs no threshold changes anywhere. |

**Already well-supported:** #31, #12, #40 (mostly), the serve-start half of #32.
**Current friction confirmed by issues:** #38 → F-7 (mask param naming); #32 → F-4 (positional
score_video contracts); #33 → keep `court_scale_slots` parked.
**Do not pre-build:** no VLM on/off-ramp scaffolding before #38's trial decides the tool and its
outputs; no court-space parallel array plumbing before #33 is actually scheduled; no generic
"mask registry" — a third mask source can be added the same way the second was.

## 8. Do-not-refactor list

Imperfect but currently cheaper to leave alone:

- **run_video's ~30-parameter API.** It is the sweepable composition root: serve_start,
  quiet_start_window and span-open variants are live sweep dimensions (sweep.py:269-271, 414-416).
  Narrowing the API means inventing a config object that sweep, e2e, gt_scoring and the CLI all
  re-plumb — another place to look, no knowledge removed.
- **The rally_segmentation facade (~70 re-exports).** It is the import surface for 14 test files
  and 2 production modules, and the monkeypatch seam the run_video tests rely on. Removing it is
  churn with no behaviour gain.
- **The thresholds=None module-globals opt-out.** Test-only convenience, pinned bit-for-bit
  against SHIPPED_THRESHOLDS by an equivalence test. Removing it costs ~20 test edits and ~10
  production edits for readability only.
- **Four distinct scraper mass-failure policies.** Each stage's policy is deliberate and mostly
  documented (transcripts >50%, triage all-fail+floor, clean all-attempted-failed, download ≥0.5
  post-hoc). Unifying them into shared machinery would blur genuinely different intents. (Naming
  the downloader's 0.5 literal is worth doing in passing — F-8.)
- **e2e's double config resolve.** run_video re-resolves what e2e already resolved; deterministic,
  harmless, and a resolved-config parameter would widen the API for nothing.
- **Frame/time conversion variants.** Display vs scoring vs decode contexts differ deliberately
  (delegate 07 census); consolidation would encode a false shared concept.
- **e2e SHARED_FILES module-global swap.** Ugly but commented ("NB NOT THREADSAFE"), guarded by
  try/finally, and the harness is single-process by design. A parameter-threading refactor would
  touch many call sites for a hazard that cannot currently occur. DEFER, revisit only if the
  harness ever parallelises.

## 9. Next PRs (max 3)

1. **Refuse empty batch outputs (F-1, F-3).** One guard in rally/cli.py mirroring the existing
   all-excluded refusal when zero videos processed; one WARNING in commentary_pairing when a
   manifest-eligible video lacks its chunks sidecar. Retarget the two tests that pin all-skipped
   success; add one warning test. Small, and it closes the only P1-severity silent path.
2. **Record executed configuration in the e2e manifest (F-2).** Derive `_configuration_values`'s
   landing fields from LANDING_OPTIONS and the dead-mask string from the resolved enum; give the
   five landing values one home shared with gt_scoring; extend the existing court-policy manifest
   test to cover them. This finishes what c32c492 started.
3. **Small-cleanups sweep (F-5, F-6, F-7, F-8, F-10 + §4 prunes).** Fix the plot artefact name,
   delete/repoint stale doc anchors, rename `segment_video(replay_mask=...)` before #38 lands,
   name the six drifting literals, drop the `src.` import prefixes, delete the retired-option
   tests and the `_pick_one_frame` alias, rename the two misnamed test files, fold in the dead
   if/else. All mechanical; one reviewable commit each or one batched PR. Gate: whole-project
   pyrefly check plus targeted pytest.

F-4 (positional score_video contracts) deliberately gets no PR now — do it as the first commit of
the #32 work, when the schema change pays for it.
