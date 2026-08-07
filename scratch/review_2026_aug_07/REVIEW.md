# Maintainability review of main @ 8c67569 (2026-08-07)

This is the final maintainability and test-pruning review for `main @ 8c67569`. It combines the completed maintainability pass and its independent calibration into one set of decisions. The supporting reports and evidence are under `archive/`; see [Provenance](#11-provenance).

A few labels used below:

- **P1**: fix promptly.
- **P2**: worth fixing, with timing stated in the finding.
- Actions are **fix now**, **fix when touching this area**, **leave alone**, or **recheck later**.
- File and line references are for commit `8c67569`.

## 1. Overall judgment

The refactor worked. The repo is in good shape and is reasonably maintainable now. The July–August work cleaned up responsibilities and naming, and no area regressed. FPS scaling now has a real single source of truth, `run_video` is staged without changing behaviour, and the rally split moved logic into focused modules while keeping a useful import/test facade.

The remaining work is small and mostly about **silent or misleading outcomes**, not architecture:

- **F-1 (P1):** an annotator batch can skip every video, write header-only CSVs, and exit `0`.
- **F-2 (P1):** the e2e manifest can record landing settings separately from the object that actually ran.
- **F-3 (P2):** commentary pairing can quietly produce no pairs when an eligible video is missing sidecars.
- **C-1 (P2):** `quiet_start_window` can silently override `span_open` even though related strategy conflicts fail loudly.

Those are the things to fix. Do not turn this into another structural cleanup pass.

Several awkward-looking areas are fine as-is for now: the ~30-argument `run_video` API, the `rally_segmentation` facade, `thresholds=None`, the different scraper failure policies, e2e's `SHARED_FILES` swap, the frame/time conversion variants, and e2e's double config resolve. They were checked specifically and do not justify work today. The `rally/evidence.py` `sys.path` insertion is a known hazard, but the proportionate fix would be a larger `bst_x` package restructure, so there is no work item for it now.

The **1394-test suite is not a maintainability problem**. It mostly earns its keep. The review found only about 10 collected cases worth pruning at all, plus three file renames; one of those cases is explicitly marginal. Separate from that, the serve-setup validator matrices can optionally be reduced by about 20 more cases, but only to make that cluster easier to read. That optional reshape is not needed for runtime, safety, or test speed. There is no case for a broad pruning campaign.

Next work should be three small PRs:

1. Refuse empty annotator outputs and warn on missing commentary sidecars (**F-1, F-3**).
2. Make the e2e manifest record the configuration that actually ran (**F-2**).
3. Do the mechanical cleanups (**F-5, F-6, F-7, F-8, F-10, C-1**, plus the firm test prunes and renames).

**F-4 waits for issue #32.** The positional scoring code is awkward, but it is correct today; #32 is the change that makes the refactor worth doing. **F-9 waits until `experiment_records` is touched again.**

Four points are still genuinely uncertain: whether unattended use makes F-1 P1 rather than P2; whether quiet-start should reject or compose with `span_open`; whether the `rally/evidence.py` `sys.path` hazard can occur in realistic in-process use; and whether `contact_frames.csv` has an external reader. None changes the immediate plan except the exact C-1 implementation choice.

**Contents**

- [2. Refactor status](#2-refactor-status)
- [3. Findings](#3-findings)
- [4. Tests: mostly keep them](#4-tests-mostly-keep-them)
- [5. Silent-failure map](#5-silent-failure-map)
- [6. Near-term issue pressure](#6-near-term-issue-pressure)
- [7. Dead-looking code: delete one thing, keep the rest](#7-dead-looking-code-delete-one-thing-keep-the-rest)
- [8. Leave these alone](#8-leave-these-alone)
- [9. Next PRs](#9-next-prs)
- [10. Open questions](#10-open-questions)
- [11. Provenance](#11-provenance)

## 2. Refactor status

Baseline: `scratch/swarm_review/readability_refactor_scoping_handoff.md` at `1afc86a` (2026-08-04).

| Area | Status | What matters now |
|---|---|---|
| Mask filename chains (dead vs replay) | **Fixed** | Producer/consumer chains are consistent; helper is `_load_dead_mask` (`rally/cli.py:42`). |
| Mirrored config values in run metadata | **Fixed** | `c32c492` made e2e use `court_evidence` constants and added a manifest-vs-execution test for court policy. F-2 is the remaining landing-options version of the same problem. |
| Shared frame-rate scaling rules | **Improved** | `fps_constants` base-30 table + `resolve()` is the SSOT. Small residue remains in `gt_scoring.canonical_tolerance` (`:403`) and `replay_mask` (`:72`, `:212`); see F-8. |
| `run_video` orchestration | **Improved** | PR #59 split it into readable stages without changing tests. Its broad API is deliberate; leave it alone. |
| `rally_segmentation` split | **Improved** | PR #58 moved real logic into `rally/*`. The facade is still a real import and monkeypatch seam; leave it alone. |
| Historical/pipeline naming | **Improved** | Old names are largely gone. Remaining residue is F-5, F-6, F-7 and a few test/file names. |
| `score_video` positional accumulators | **Still awkward** | Correct today. Fix with #32, not before it (F-4). |
| e2e `SHARED_FILES` module-global swap | **Still awkward** | Deliberate single-process harness behaviour; leave it alone unless the harness parallelises. |
| `experiment_records` sanitiser skip/delete | **Still a problem** | Malformed records can be skipped by sanitisation then deleted after backup; fix when this area is next touched (F-9). |
| `run_clean._score_pending` lifecycle | **Readability only** | Traced as behaviourally correct. Leave it alone. |
| scraper `_download_one` seam | **Improved** | `_verify_existing` is extracted and unlink policy is explicit. |

## 3. Findings

### [P1] F-1: all-skipped annotator batches report success — **fix now** (PR 1)

**Files:** `rally/cli.py:201-234` (skip paths), `:248-263` (only `all_excluded_error` checked); `batch_report.py:39-53` (no processed-count branch); `config.py:27-28` (output CSVs).

If every video is skipped—for example because FPS CSV ids do not match, or because `run_video` raises per video—the batch logs the skips, prints `batch completed: 0 of N`, writes header-only output CSVs, and exits `0`. `rally_spans.csv` then looks like a legitimate “no rallies” result, so downstream pairing can silently lose the pipeline's main data product. In scripted runs the exit code is the only signal a wrapper sees.

That is different from the all-EXCLUDED path: when the doubles filter excludes everything, the same CLI raises and refuses to write the CSVs (`rally/cli.py:141-145`, `:248-263`; commit `ba4e750`). Per-video log-and-skip is deliberate (`cli.py:228`); the problem is only the terminal state where nothing was processed.

**Change:** after the loop, mirror the all-excluded guard. Keep the batch report so the skip reasons still print, but raise before writing empty outputs.

```python
if processed_count == 0 and all_excluded_error is None:
    raise RuntimeError(
        'batch processed 0 of %d videos; refusing to write empty outputs'
        % video_count
    )
```

Retarget `test_batch_report.py:129-145` and `test_fps_cli_and_tracknet_modes.py:58-103` to expect the raise. Leave the partial-skip tests unchanged.

The fix is justified either way. The remaining uncertainty is only severity: if this CLI runs unattended, P1 is right; if not, it is closer to P2.

### [P1] F-2: e2e manifest can disagree with the configuration that ran — **fix now** (PR 2)

**Files:** `e2e_court_annotator.py:79-85` (`LANDING_OPTIONS`), `:633-650` (`_configuration_values`), `:909/:1124` (manifest insertion), `:984-997` (executed call); `calibration/gt_scoring.py:464-472` (second construction of the same five values); `point_winner.py:355-363` (required fields).

The five landing-filter values `(7, 0.004, 5, 7, 0.75)` exist in three places: e2e's `LANDING_OPTIONS`, `gt_scoring.build_run_video_inputs`, and literal values inside `_configuration_values`, which writes the run manifest. `_configuration_values` does not read `LANDING_OPTIONS`, and the current manifest test covers court policy but not these landing fields (`test_annotator_measurement.py:58-62`).

The same pattern appears in smaller form with `"dead_mask_mode": "replay"`, which is a raw string rather than the resolved enum value. `ref_err_px = 3.5` also exists as both an e2e constant and an independent `run_video` default.

The e2e manifest is useful only if it says what actually ran. Today, changing `LANDING_OPTIONS` without changing `_configuration_values` can make that false. This is the same recorded-vs-executed problem already fixed for court policy in `c32c492`.

**Change:**

1. Read landing fields in `_configuration_values` from `LANDING_OPTIONS`.
2. Read the dead-mask string from the resolved enum.
3. Give the five landing values one home shared with `gt_scoring`.
4. Extend the existing manifest test to cover the landing fields.

```python
'landing_filter_options': dataclasses.asdict(LANDING_OPTIONS),
'dead_mask_mode': resolved.dead_mask_mode.value,
```

Verify with the extended manifest test and one e2e smoke case.

### [P2] F-3: eligible commentary videos can miss sidecars silently — **fix now** (PR 1)

**Files:** `commentary_pairing.py:219-236` (loaders), `:151-185` (`None`-mask bypass), `:341-364` (loop and unconditional final write); `download_scraped_videos.py:260-273` (`commentary_eligible` flag); `relevance_triage.py:236-261` (chunks written by a different stage).

A video can be manifest-eligible even if its triage chunks or replay-mask sidecar never arrived. Missing chunks produce zero pairs; a missing mask disables replay filtering. Both missing-file branches return without logging, and the final pairs CSV is still written.

This stays P2 rather than P1. The fallbacks are documented, malformed files do raise, `pair_video` deliberately accepts a nullable mask, and each video logs `N rallies, M paired`. The problem is that a missing sidecar looks the same as “commentary matched nothing”.

**Change:** add one `WARNING` in the batch loop when a manifest-eligible video has no chunks sidecar. A warning for a missing replay mask is reasonable but optional. Do not add a new subsystem or batch policy.

Add one test that asserts the warning fires.

### [P2] C-1: `quiet_start_window` silently overrides `span_open` — **fix now**, implementation choice still open (PR 3)

**Files:** `rally/spans.py:362-369` (precedence); `run_video.py:191-194` (serve-start conflict validation); `config.py:130` (`span_open=SpanOpen.BACK_FILL` default); `calibration/sweep.py::_base_and_serve` (sets `span_open` only when swept).

`find_rally_spans` dispatches `serve_start`, then quiet-start, then `span_open`. The quiet-start finder does not take `span_open`, so a config can carry the default `BACK_FILL` while execution actually uses quiet-open semantics. Related conflicts already fail loudly: serve with `REGION_START`, serve-close with `BACK_FILL`, and serve with quiet-start all raise. This pair is the silent exception.

The interaction is latent today. No pinned grid sweeps `quiet_start_window`; `BOUNDARY_KEYS` and `CONTACT_KEYS` are threshold-only, and the winner round-trip is internally consistent. The lane is **unmeasured**, not measured and rejected.

The current proposed guard is:

```python
if resolved.quiet_start_window is not None and resolved.span_open is not None:
    raise ValueError('quiet_start_window cannot be combined with span_open')
```

A matching test should mirror `test_quiet_start_and_serve_start_fail_in_run_video`.

**Do not pretend the product decision is settled.** Rejecting the combination is right only if quiet-start and back-fill are meant to be mutually exclusive. If quiet-start is supposed to refine back-fill, compose them instead. The silent status quo is the part that needs to go.

### [P2] F-4: `score_video` positional contracts make the next GT schema change expensive — **fix with #32, not before**

**Files:** `calibration/gt_scoring.py:576-582` (four-slot positional accumulators), `:624` (one-iteration loop), `:669-674` (positional 27-field `RallyRow`), `:677-682` (positional `VideoScoring`). Callers: e2e `_score_configurations` through `score_video`, and the sweep scoring path.

Each metric family is stored in a bare four-slot list whose slot meanings are implicit, then unpacked positionally into a 27-field `RallyRow`. Issue #32 adds real rally start/end GT in exactly this area, so that change will otherwise require carefully threading new positions through every accumulator and constructor call.

The code is correct today, and PR #63 already documented the contracts. Refactoring it now would be churn.

**When #32 starts:** switch `RallyRow`/`VideoScoring` construction to keyword arguments and give the accumulator slots names (for example a small `NamedTuple` or four named counters). Keep the count-gate assertion at `:675`; it reconciles two independent recounts and is deliberate.

Verify with the existing calibration schema tests plus one targeted `score_video` test.

### [P2] F-5: plot script still reads the retired sweep artefact — **fix now** (PR 3)

**Files:** `scripts/plots/plot_stage8_tradeoffs.py:63-93` reads `boundary_crowns.csv`; the sweep writes `best_config_comparison.csv` (`calibration/sweep.py:464`).

The crowns-to-`best_config` rename was not propagated to this script. A fresh sweep therefore makes the script fail with `FileNotFoundError`; the script name also keeps the retired `stage8` label.

If the script is still live, update the input filename and preferably rename the script. If it was meant to retire with the old artefact, delete it instead. That ownership decision is still open.

Verify by running it against an existing sweep output directory.

### [P2] F-6: live docstrings point at an untracked, stale scratch spec — **fix now** (PR 3)

**Files:** `rally_segmentation.py:1`, `replay_mask.py:1`, `config.py:4` and `spec s6` comments such as `config.py:28`, `scraper/config.py:10`, `scraper/__init__.py:3`, `commentary_pairing.py:1`; `rally_segmentation` also cites moved/line-rotted `shuttle_extractor.py:244-249`.

These docs point at `local_scratch/autograder_architecture/scraper_spec.md`, which is untracked and stale. That is worse than no link because a new maintainer can follow it into missing or obsolete intent.

Delete those anchors or repoint them to the tracked contracts docs from PR #63 (`75edb3e`). Verify that `grep` for `local_scratch` and `spec s` under `src/` returns nothing.

### [P2] F-7: `segment_video(replay_mask=...)` is now named for the wrong thing — **fix now, before #38** (PR 3)

**Files:** `rally_segmentation.py:136` (parameter), `rally/cli.py:42` (`_load_dead_mask`), `run_video.py:325,417` (call sites).

The argument called `replay_mask` now receives the definitive exclusion/dead mask, including composition cuts. PR #63 already renamed the CLI loader to `_load_dead_mask`, but this seam kept the older producer-specific name. Issue #38 is expected to add another mask source, which will make `replay_mask=` actively misleading.

Rename the parameter to `exclusion_mask` or `dead_mask` across the facade, `run_video` call sites, callers and tests in one commit. Verify with whole-project `pyrefly check` and a grep for the old keyword.

### [P2] F-8: small single-source-of-truth drifts — **fix now, but keep each fix small** (PR 3)

These are separate small cases of “edit one place and forget the other”:

- Video extensions `{'.mp4','.mkv','.webm','.avi','.mov'}` are repeated in three scraper modules.
- `FINE_BATCH_SIZE = 16` controls WhisperX transcription (`commentary_cleaning.py:369`), while BERTScore separately hardcodes `batch_size=16` (`commentary_cleaning.py:169`). They are independent knobs: give BERTScore its own named constant; do **not** point it at `FINE_BATCH_SIZE`.
- e2e repeats `(512, 288)` beside `court_evidence.DETECTOR_RESOLUTION = (512.0, 288.0)` (`e2e:503-509`).
- The download mass-failure threshold `0.5` is an unnamed literal (`download_scraped_videos.py:582-589`); the other stage policies are named and documented.
- `selection.py:53-57` repeats count-form F1 maths beside shared `safe_f1`.
- `gt_scoring.canonical_tolerance` (`:403`) rescales `5.0` separately from `CONTACT_TOLERANCES_BASE30`.

Batch these into the cleanup PR. If any item stops being a one-line/small local fix, drop it from this PR rather than expanding scope. Verify with `pyrefly check` and the existing tests covering each site.

### [P2] F-9: `experiment_records` can skip sanitising a malformed record and then delete it — **fix when touching this area**

**Files:** `experiment_records.py:_planned_json_changes` silently skips malformed configuration records (`:221-223`) that `build_summary` would reject; `_scanner_findings` then allows `clean_run` to delete the unsanitised leaf after tar backup (`:336`).

A malformed record is exempt from sanitisation but not deletion. The delete is printed and recoverable from the tar backup, so this is not being scheduled as an immediate PR. Still, the inconsistency is real: the file most likely to need inspection can disappear from the working tree without having been sanitised.

When this area is next touched, make the sanitiser fail loudly or explicitly skip-and-report the malformed record. Add one malformed-record test for the chosen loud path.

### [P2] F-10: local `pyrefly check` is red because of test imports — **fix now** (PR 3)

**Files:** `tests/test_fps_constants.py:11,14` (`from src.annotator...`); `pyproject.toml:166-177`; `ci.yml:34-37` (CI narrows pyrefly to unknown-name, so CI remains green).

Whole-project `pyrefly check`, which is the stated local pre-commit gate, reports these imports every time. That makes a red local gate normal. The `src.` prefix can also load a second module object (`src.annotator.config` vs `annotator.config`), which can make identity-based assertions diverge.

Drop the `src.` prefix on those two imports and, for consistency, in `test_fps_cli_and_tracknet_modes.py:200,252,312`. Verify that whole-project `pyrefly check` drops to the single out-of-scope `bric` error, then run pytest on the two affected test files.

### Deferred details — **no separate work item**

- `BaseAnnotatorConfig` and `ResolvedAnnotatorConfig` duplicate defaults such as `span_open` and `rejected_grades`. Drift is possible, but both definitions are together in `config.py`; do nothing now.
- `rally_segmentation.py:205-215` has an identical `if/else` call because `sticky_distances` is already `None` in the else arm. Fold it into the F-8 cleanup if touched.
- e2e `SHARED_FILES` and positional pin indexing are covered by the leave-alone section below.
- `run_clean._score_pending` is asymmetrical between per-video and per-chunk handling, but the lifecycle was traced as correct and deliberate. Leave it alone.

## 4. Tests: mostly keep them

The 1394 tests mostly protect real behaviour. The most suspicious clusters were checked in detail rather than assumed good: `test_annotator_measurement`, `test_annotator_run_video`, `test_fps_cli_and_tracknet_modes`, and the serve-setup pair. Their mocks and monkeypatches mainly protect observable failure semantics, persisted artefacts and public seams; they are not just pinning private implementation structure.

The required cleanup is small. The review found only about 10 collected cases worth pruning at all, plus renames; the constant-echo case below is explicitly marginal. **Do not add the optional serve-setup reshape to that number.** Those extra ~20 cases are separate and are optional compression for readability only.

### Delete or rename now

- **Delete the retired-option CLI test:** `test_rally_segmentation_main_rejects_retired_options` (`test_fps_cli_and_tracknet_modes.py:144-161`), six parametrised cases. It only checks stock argparse rejection of flags that no longer exist. Reintroducing one of those flags would be an intentional product change. Retirement is documented in `3f6e34d`.
- **Delete `_pick_one_frame` and its identity test together:** the alias at `src/bst_x/preparing_data/heuristics/sticky_anchor.py:321` exists only for tests, and `tests/test_sticky_anchor.py:459` merely asserts that it is the same object. Point the remaining tests at `pick_one_frame`. (The alias sits in `bst_x`, outside the review's `src` scope; the test file is in scope.)
- **Constant-echo matrix test: optional delete:** `test_fixed_matrix_and_parent_order_are_deterministic` (`test_annotator_measurement.py:112-124`) restates `CASES`/`PARENTS`. Its only protection is against accidental reorder, which per-configuration manifests would also expose. Deleting or keeping this one case is both defensible.
- **Fix `src.`-prefixed imports, do not delete tests:** F-10.
- **Rename only:** `test_scraper_doubles_flag.py` and `test_scraper_composition_mask.py` actually target `src/annotator`; `test_annotator_serve_setup_b2.py` keeps a retired batch label.

### Optional: make the serve-setup cluster easier to read

The 62 collected cases in `test_annotator_serve_setup.py` and `_b2.py` break down roughly as:

- ~15 gate-semantics cases;
- ~13 fail-closed cases for production-shaped sentinel states emitted by `evidence.py:153-172`;
- ~34 input-validation matrix cases.

Keep the first two groups. They protect real behaviour and two previously shipped bugs: fail-open on an unmeasurable player (`305b3ad`), and double normalisation/wrong-player pairing (`cc02b62`).

Also keep the validation families that protect live boundaries:

- the builder's resolution validation, because `run_video.py:216-230` validates only that resolution is present;
- the cross-field threshold check, because `ServeStartConfig` is typed but not value-checked at `types.py:69-75`.

The remaining matrices cover states no maintained caller can produce: `series_drift` shape/dtype, `serve_setup_still` window/claimed-frame/threshold/slots combinations, and `ServeSetupInputs` dtype/shape checks. Those can each collapse to one representative case, taking the cluster from about 62 to about 42.

Again: this reshape is **optional and for legibility only**. It is not needed for runtime, speed, or safety.

### Keep these

- **`test_annotator_run_video.py` (38 tests): keep.** The spy tests call through real builders and assert real hand-offs. The 15 mode-validation cases are cheap contracts for a broad composition root used by sweep, e2e and CLI.
- **`test_annotator_measurement.py` (remaining 20): keep.** These tests assert observable outputs: exit codes, manifest status, artefact md5s and failure-isolation scope. That is exactly what the measurement harness needs to protect.
- **`thresholds=None`: keep.** `test_thresholds_none_matches_explicit_shipped_preset` checks the module-global path bit-for-bit against `SHIPPED_THRESHOLDS`.
- **`test_calibration_schemas`: keep.** It pins persisted CSV/JSON contracts.
- **`test_namespace_migration`: keep.** It was already deliberately reduced on 2026-06-16 from T1–T12 to four families. The current 38 cases still pin artefacts/schemas plus five legacy-name scans; their allow-lists have needed only two changes since introduction.
- **`validation_overlay` tests: keep.** They use real ffmpeg media and issue #31 keeps the tool live.
- **Scraper batch tests: keep.** Patching `triage_video`, `acquire_transcript`, `_clean_once` and `_download_one` is the right seam for testing batch policies.

## 5. Silent-failure map

This is the useful comparison: which paths already fail loudly, which deliberate fallbacks are fine, and which silent cases need work.

| Path | Trigger | Result | Judgment |
|---|---|---|---|
| `rally/cli` batch | doubles filter excludes every video | raises; writes no CSVs | **Good: deliberate loud guard** |
| `rally/cli` batch | every video skipped (FPS id mismatch or per-video `run_video` failure) | header-only CSVs; `0 of N`; exit `0` | **Fix: silent for automation (F-1)** |
| `rally/cli` per video | bad track / invalid mask | log-and-skip; batch report records it | **Fine** |
| `replay_mask` CLI | missing court mask / homography / track sidecar | that signal becomes all-False; INFO log; mask still written | **Deliberate fallback**; mask itself does not record which signals ran |
| `commentary_pairing` | eligible video missing chunks sidecar or replay mask | blank or unmasked pairing; only `N rallies, 0 paired` INFO | **Fix visibility (F-3)** |
| `commentary_pairing` | missing FPS | warning and skip; all-skipped batch can still write empty pairs CSV | Mixed; existing warning is loud enough per video |
| `commentary_pairing` | missing/invalid manifest entry for FPS-bearing video | raises | **Fine** |
| `run_video` config | `quiet_start_window` plus default `span_open` | quiet-open executes while config records `BACK_FILL` | **Fix: silent and latent (C-1)** |
| `transcript_acquisition` | corrupt caption JSON | uncaught exception kills batch | Loud, though inconsistent with its per-video skip policy |
| `transcript_acquisition` | >50% failures past floor | raises mid-run | **Deliberate** |
| `relevance_triage` | corrupt transcript sidecar | uncaught exception kills batch | Loud |
| download stage | >=50% attempted downloads fail | post-hoc guard exits `2` (`download_scraped_videos.py:582-589`) | **Deliberate**; name the `0.5` literal under F-8 |
| `commentary_cleaning` | all attempted LLM calls fail | raises | **Deliberate** |
| `experiment_records.clean_run` | malformed configuration record | sanitiser skips; scanner may delete unsanitised leaf after backup | **Fix when touching (F-9)** |
| e2e measurement | one configuration fails | `failure.json` + terminal manifest; run continues | **Good** |
| sweep plots | fresh output + `plot_stage8_tradeoffs.py` | `FileNotFoundError` | Loud but stale consumer (F-5) |

## 6. Near-term issue pressure

These open issues touch the reviewed areas. The useful question is whether today's seams are ready—not whether we can pre-build abstractions for them. Issue #50 (“Break up the monoliths”) is context only; it does not change the leave-alone decisions below.

| Issue | Likely change | Where it lands | Ready today? | Current friction |
|---|---|---|---|---|
| #38 VLM scene filtering | another mask/evidence producer | new sidecar/mask concept, dead-mask dispatcher, `run_video.raw_exclusion_mask` | Mostly | Mask composition is fine; F-7's `replay_mask=` name will be misleading once a third source exists. |
| #33 court-space distances | parallel court-space positions for sticky attribution | `rally/evidence.py` + point-winner attribution | Mostly | Contained by the current seam. Dead-today `court_scale_slots` may become useful, so keep it until #33 is decided. |
| #32 real rally start/end GT | new GT columns and scoring dimension | `calibration/gt_scoring.py` + `rally/serve.py` | No | Directly stresses F-4's positional accumulators/builds. Fix F-4 as the first #32 commit. |
| #40 compressed e2e artefacts | compressed writers/loaders | e2e writer helpers + `FilePin` md5 verification | Yes | Writers are centralised. Decide md5-of-compressed vs md5-of-raw once in `FilePin`. |
| #31 hallucination guards | investigation, then small guard masks | `inpaint_guard` / `shuttle_hallucination_mask` threading | Yes | Existing seams and tests already support this. |
| #12 FPS-normalise uploads | pre-pipeline normalisation | `fps_constants` / `resolve` | Yes | The base-30 SSOT means normalised inputs need no threshold changes. |

Do not pre-build VLM on/off-ramp scaffolding before #38 decides the tool and outputs. Do not add court-space parallel arrays before #33 is scheduled. Do not invent a generic mask registry: a third mask source can be added the same way the second was.

## 7. Dead-looking code: delete one thing, keep the rest

- **`court_scale_slots` (`rally/evidence.py:42`): leave alone until #33 is decided.** It has zero production callers, one facade re-export and one test. Its sibling `court_scale_boxes` was deleted in `51b2977`, but `court_scale_slots` survived an explicit ruling (W2.4), and #33 is a plausible near-term consumer.
- **`_pick_one_frame` alias + identity test: delete both.** This is a real dead production/test loop; see the test section.
- **These are not dead:** `test_first_last_stroke_buffered_search` is a live scripts-analysis contract with an explicit keep ruling (2026-08-02 R9); `convert_landing_options` and `corner_error_band_from_corners` have production callers; the `scraper/config.py:21` re-export block deliberately documents cross-pipeline artefact paths. `CONTACT_FRAMES_CSV` has no scraper consumer and does have an identity test (`test_scraper_search_index.py:24-27`), but the re-export block is harmless and intentional. Keep it.

## 8. Leave these alone

These are not invitations for a later cleanup backlog. They are current **do-not-refactor** decisions. Revisit only when the stated trigger changes.

- **`run_video`'s ~30-parameter API: leave it alone.** It is the sweepable composition root. `serve_start`, `quiet_start_window` and span-open variants are live sweep dimensions (`sweep.py:269-271`, `414-416`), and its four production callers are all sweep/measurement-facing. Wrapping this in another config object would move the same knowledge somewhere else and force sweep, e2e, `gt_scoring` and CLI to re-plumb it.
- **`rally_segmentation` facade (~70 re-exports): leave it alone.** The importer census found 14 test files, five production modules and three runnable docs scripts using it. It is a real import seam and the monkeypatch seam used by `run_video` tests. Roughly 35 re-exports have no executable user outside the facade, but their carrying cost is negligible; that does not create a cleanup task.
- **`thresholds=None`: leave it alone.** It is a test convenience, and an equivalence test pins it bit-for-bit to `SHIPPED_THRESHOLDS`. Removing it would cost about 30 edits for readability only.
- **Different scraper mass-failure policies: leave them different.** Transcript acquisition, triage, cleaning and download have deliberately different failure policies. Unifying them would erase real differences. Only name the download stage's `0.5` literal under F-8.
- **e2e double config resolve: leave it alone.** `run_video` re-resolves a config e2e already resolved. It is deterministic and harmless; accepting resolved config would widen the API for no benefit.
- **Frame/time conversion variants: leave them alone.** Display, scoring and decode have different semantics. Combining them would create a false abstraction.
- **e2e `SHARED_FILES` module-global swap: leave it alone while the harness is single-process.** It is ugly, explicitly marked `NB NOT THREADSAFE`, and protected by `try/finally`. Revisit only if the harness parallelises.
- **`rally/evidence.py:14-18` `sys.path` insertion: record the risk, do not refactor now.** It inserts `src/bst_x` so the sticky-anchor picker stays single-sourced, which makes `preparing_data` and `pipeline` importable as top-level modules process-wide. That creates an import-order/dual-module-identity hazard similar to F-10, but no in-repo pytest path currently demonstrates it. A real fix is a larger `bst_x` package restructure, which is out of proportion today.

## 9. Next PRs

### PR 1 — make empty/missing-input outcomes visible

- **F-1:** refuse to write annotator outputs when zero videos were processed.
- **F-3:** warn when a manifest-eligible commentary video has no chunks sidecar; optionally warn for a missing replay mask.
- Retarget the two tests that currently pin all-skipped success and add one warning test.

### PR 2 — make e2e manifests truthful

- **F-2:** derive manifest landing fields from `LANDING_OPTIONS` and dead-mask mode from the resolved enum.
- Give the five landing values one shared home with `gt_scoring`.
- Extend the existing court-policy manifest test to cover landing fields and run one e2e smoke case.

### PR 3 — mechanical cleanup only

Include **F-5, F-6, F-7, F-8, F-10 and C-1**, plus the firm test prunes/renames. Fold in the identical `rally_segmentation.py:205-215` branch if it stays trivial. Keep the scope mechanical; if an F-8 item grows beyond a local fix, leave it out.

Gate this PR with whole-project `pyrefly check` plus targeted pytest.

**Not in these PRs:** F-4 waits for #32. F-9 waits until `experiment_records` is next touched. The serve-setup ~20-case reshape is optional and can ride with PR 3 only if it genuinely makes the tests easier to read.

## 10. Open questions

These are evidence gaps, not hidden tasks.

- **Is the rally CLI used unattended?** This changes F-1's P1/P2 label, not the fix itself.
- **Should quiet-start reject or compose with `BACK_FILL`?** C-1 currently proposes rejection. If quiet-start is intended to refine back-fill, composition is the right fix instead. The silent override is the only clearly wrong state. The lane is unmeasured, so removing it entirely is also still a possible product decision.
- **Can the `rally/evidence.py` `sys.path` hazard occur in a realistic process?** No in-repo pytest collection path currently imports `preparing_data`/`pipeline` through both module identities. That does not prove external or future code cannot.
- **Who reads `contact_frames.csv`?** Both passes found no in-repo reader. It is likely an external data product, but that has not been established. Recheck before changing its writer.

## 11. Provenance

This review covers `main @ 8c67569` on 2026-08-07 and combines two completed passes.

The first maintainability/test-pruning pass inspected every retained finding directly. The three highest-risk candidates were then given adversarial read-only checks designed to refute them. Its report is `archive/report.md`, and the 10 census/verification packets are under `archive/evidence/jobs/`.

The second pass was an independent adversarial calibration. It recorded its positions before reading the first report, then used six more evidence workers under `archive/evidence/calibration-jobs/`. Its report is `archive/calibration-report.md`. It confirmed the first report wherever it challenged it, added C-1, and supplied the serve-setup test assessment plus qualifications folded into F-8 and the leave-alone list.

Measured counts such as importer censuses, collected-test totals and stage maps come from those evidence packets. The readable worker output is `result_extracted.md` in each job directory. The F-1/F-2/F-3 adversarial checks are:

- `archive/evidence/jobs/09_verify_cf8_batch_exit0`
- `archive/evidence/jobs/08_verify_cf7_landing_options`
- `archive/evidence/jobs/10_verify_cf11_silent_sidecars`

The six calibration jobs cover the facade census, serve-setup tests, FPS SSOT, e2e failure handling, scraper stage map and namespace-migration tests.