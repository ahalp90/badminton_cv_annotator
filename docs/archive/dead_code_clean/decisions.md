# Refactor decisions

## Audit limits and execution cautions

1. **Reference tracing.** Semantic navigation used pyrefly references. The call
   hierarchy did not resolve Python callables in this project, so every finding
   recommended for action was also checked with direct grep of tracked files.
   Re-exports and dynamic dispatch can escape a single reference method.
2. **Hand-run entry points.** Grep and caller tracing do not reveal every
   operational entry point. Config comments and runbooks that describe hand-run
   chains are live surface and must be checked before deleting code.
3. **Scope.** `docs/**/*.py` and `scripts/archive` were outside the audit scope.
   They are evidence artefacts and already archived code. The audited code scope
   was `src/` and `scripts/`.
4. **Audit only.** The audit changed documentation only. The refactor is a
   separate later pass governed by R0-R9 below.
5. **Review.** An independent plan review raised ten points, all of which were
   reflected in plan revision 2. An independent final review raised fourteen
   points in revision 1, then four carried points and two new points in revision
   2. Every final-review point was adopted.

## Refactor rulings (2026-08-02)

Contract for the dead-code and duplication refactor. Sources: `findings.md`,
the maintainer's rulings, and three automated read-only checks from 2026-08-02:
D1 downloader compatibility, D2 shared-swap blast radius, and D3 verification.
Planning only: no code changes were made during the audit. Execution is a later
pass. The D1/D2/D3 raw returns are archived as
docs/archive/dead_code_clean/findings/d1_downloader.md, d2_shared_swap.md and
d3_verification.md; their consumer and contract tables are normative for
the retargets below.

### R0. The bric maintenance freeze, defined

bric is kept working but not developed. For this refactor the freeze
means: no new bric features, no rework of bric-only pipelines, no
behaviour tuning. It does NOT bar mechanical retargets of bric imports,
deletion of dead bric code, or consolidation that replaces a bric-local
copy with the canonical shared one; those are the point of this pass.
Approved touches: R1 (vendor path retarget), R2 (import retargets, the
clip-bound clamp behaviour change, taxonomy callsite updates), R8 (dead
code deletion and two dedups). Anything beyond that list stays frozen
(the parked WP6 rows in R8).

### R1. One TrackNet tree in a neutral home
(corrected 2026-08-02)

Point of confusion: tracked callers made TrackNet look classifier-only.
Resolution: the scrape lane's shuttle tracks come
from hand-running the bst_x tree today: scraper/config.py:154-158
records "the current hand-run s29 chain is the scrape-lane consumer"
and pins SCRAPE_TRACKNET_STRIDE = 8 and SCRAPE_TRACKNET_LARGE_VIDEO =
True for the promoted wrapper's subprocess boundary. The annotator then
consumes the resulting (t, 3) track arrays (rally_segmentation.py,
validation_overlay/overlays/shuttle_track.py). The WP1 caller trace counted
tracked callers only, so it did not expose the hand-run chain.

Keep one TrackNetV3 tree with bst_x's content (it carries the
--large_video path that pipeline/shuttle_extractor.py:149 and the
scrape lane both use). The destination is pinned:
src/shared/tracknetv3/: genuinely neutral ground, per R2's own rule
that anything the annotator or scraper consumes lives under shared/,
and flat by the naming ruling (no underscore-prefixed dirs, no
containing-folder nesting for a single vendored tree). (Rev 3 said
classifier_shared/_vendor; that contradicted the scrape-lane
consumption and is superseded.) Changing the destination requires a
new owner ruling.

Retarget four consumers: bric's subprocess call
(bric/perception/shuttle.py), the api's load_models / predict_video
imports (api/bric_inference.py), bst_x's batch_predict invocation
(pipeline/shuttle_extractor.py, whose PROFILE_DEFAULTS 'scrape' profile
at :39 already wires the SCRAPE_TRACKNET_* values and IS the scrape
lane's tracked wrapper), and the hand-run s29 chain; any doc or runbook
that spells out its batch_predict / write_inpaint_metadata commands
updates to the new path. tests/test_inpaint_sidecar.py parses both old
trees by fixed path (around :515-543) and retargets to the new tree.
pyproject.toml's literal lint exclusions (Ruff excludes
"src/bst_x/TrackNetV3/" at :151 and the bric vendor dir at :152)
retarget to "src/shared/tracknetv3/" so the moved vendor code does not
enter the lint gate. The **/_vendor/** and **/TrackNetV3/** globs at
:177 do NOT cover the flat lowercase home, so every exclusion list
that relied on them (:177's tool included) must name the new path
explicitly; verify with a whole-project lint and pyrefly run after the
move. Delete
src/bric/perception/_vendor/tracknetv3/ including its caller-less
batch_predict.py (WP1-2), and the old src/bst_x/TrackNetV3/ once moved.
Working files are byte-identical today (WP1-1), so no behaviour
reconciliation is needed. Smoke the bric subprocess path, the api
import path, and the batch invocation including --large_video (the
flag the scrape lane depends on).

Standing caution from this miss: the gap was hand-run ENTRY-POINT
discovery, not caller-less constants (shuttle_extractor's scrape
profile does import the SCRAPE_TRACKNET_* values). Config comments
naming a hand-run chain are live operational surface; no future
dead-code pass may treat a documented hand-run path as dead on a
tracked-callers-only trace.

### R2. Split shared/ into shared/ and classifier_shared/

Ruling: unify the mirrors with bst_x as source of truth. A module stays
in src/shared/ only when the annotator or scraper genuinely consumes it;
classifier-only material moves to a new src/classifier_shared/. For
every surface that moves or disappears, the consumer table in
d2_shared_swap.md section 2 is the normative retarget checklist; every
listed consumer (production, diagnostics, scripts, tests, and indirect
re-exports such as preparing_data/heuristics/current.py) gets its import
updated; "retired" always means the old module is deleted and all
listed consumers point at the replacement, with no re-export shim left
behind.

- Court maths stays in src/shared/ (the annotator imports it in
  calibration/fixtures.py, calibration/gt_scoring.py, court_evidence.py
  and point_winner.py). One implementation with the union surface: the
  mirrored maths at bst_x semantics, shared's REF_COURT_M /
  REF_COURT_CORNERS_M / load_all_court_info, and bst_x's
  build_all_court_info plus its live to_court_coordinate /
  check_pos_in_court (these replace the dead shared copies from WP4-1).
  bst_x/pipeline/court_utils.py is retired; all D2-listed callers,
  including the kept validation scripts, retarget to shared.court.
- Player mapping: single copy in classifier_shared/ (the bodies are
  already equivalent; ZH_TO_EN comes from the unified taxonomy module).
  bst_x/pipeline/player_mapping.py is retired; clip_generator,
  scripts/build_shots_master.py and tests/test_player_mapping.py
  retarget.
- Flaw parsing and clip bounds: single copies in classifier_shared/ with
  bst_x semantics, including the max(0, start_f) clamp. Data paths become
  explicit parameters (the two sides use different roots). Accepted
  behaviour change, allowed under R0: early-frame clip bounds shift for
  bric/api consumers and for regenerated shots_master metadata (D2
  section 3). The path and split exports bric uses (HOMOGRAPHY_CSV_PATH,
  VIDEO_METADATA_PATH, SPLITS_V2_PATH, SPLITS_BST_BASELINE) survive the
  move; SPLITS_V2 and _load_splits_v2 are still deleted per WP4-2.
- Taxonomy: one module in classifier_shared/; bst_x's taxonomies are
  authoritative, ruled 2026-08-02. The bric-side
  driven_flight -> unknown mapping was a historical accident (bst-x made
  and fixed the same one) and is corrected to driven_flight -> drive in
  every merge map, legacy entries included. The correction changes
  dataset-build and eval behaviour only; it cannot change deployed
  decoding (below). bst_x's Taxonomy dataclass becomes the one API;
  bric's callsites on the shared-only fields (bric/dataset.py,
  network.py, train.py, plus tests/test_network.py and
  bric/smoke_test.py) update mechanically.
  Legacy registry keys are NOT plain aliases; the legacy contracts
  cannot be rebuilt through bst_x's class-list builder logic (independent
  review finding 1). Retained legacy entries are stored as pinned literals:
  the canonical dataclass keeps bst_x's explicit `classes` field and
  gains one minimal addition, an optional excluded-from-training
  marker, so the class_list() vs trainable_class_list() distinction
  bric's callers use survives without a second dataclass.
  Dispositions were settled by a 2026-08-02 reference sweep (`git grep` over
  src/, tests/, scripts/, runtime/, docs/models_registry.yaml,
  frontend/):
  - une_merge_v1_nosides: RETAINED, pinned verbatim: 15 full classes
    including unknown, 14 trainable, orderings unchanged (the deployed
    manifest's stored 14-class list is the trainable view); the merge
    map carries the driven_flight correction. It is the deployed bric
    contract (runtime/deployed/bric/*/manifest.yaml:7, eval sidecars,
    docs/models_registry.yaml:139) and bric's default.
  - raw_35: RETAINED, pinned (tests/test_network.py:11 imports
    TAXONOMY_RAW_35; it defines the raw label space).
  - merged_25, une_merge_v1, une_collapsed_v1_nosides: RETIRED with
    their module constants. No live registry lookup exists; the
    merged_25 strings in scripts/plots/trajectory_chart_macro_and_min.py
    are local chart labels and tests/test_namespace_migration.py:297
    embeds it in a run-name literal; neither imports the registry.
    Re-evaluating an old 25-class bric checkpoint would need the entry
    restored from git history; accepted under R0.
  - DEFAULT_TAXONOMY stays 'une_merge_v1_nosides' (bric/dataset.py:72,
    network.py:244 and train.py:648 default to it; changing it would
    change bric behaviour, barred by R0). TAXONOMY_UNE_MERGE_V1_NOSIDES
    is retained (bric/smoke_test.py:192).
  Hard constraints for the deployed bric model: _class_list stays
  exactly equal to the manifest's stored config.classes
  (api/bric_inference.py:180 reads the manifest, not the registry, so
  registry changes cannot alter decoding); the une_merge_v1_nosides key
  must still construct a 14-output BRICNetwork head that loads the
  deployed checkpoint (api/bric_inference.py:184); registry and sidecar
  name strings stay literal. docs/models_registry.yaml values stay
  unchanged; the api returns its literal taxonomy field and never
  imports the taxonomy module, so this is a separate requirement from
  the runtime key table.
- eval_plots moves to classifier_shared/ (bric is the only production
  caller). Absorb the script renderer per WP2-7: optional output-path
  and figsize parameters; scripts/plots/confusion_matrix.py becomes a
  thin CLI over it.
- video_io moves to classifier_shared/, trimmed per WP4-6/7: delete
  read_frames / iter_frames and read_frame_at / write_frame_thumbnail
  with their tests and the stale API-contract docstring claim; keep
  get_video_info (live in bric).
- temporal.py is deleted entirely (WP4-4/5); no production adopter
  named.
- shared/taxonomy.py's unused PLAYERS / UNPREFIXED_TYPES are deleted
  (WP4-3) during the taxonomy move.
- shared/README.md is rewritten for the split. The "BRIC must not import
  bst_x" wall is retired in favour of both classifiers importing
  classifier_shared/.

### R3. Promote the scraper downloader (WP2-1, WP4-8, WP4-9)

The scraper downloader becomes the sole yt-dlp downloader, per D1's
plan. Module dispositions first, so nothing surviving calls something
deleted (independent review finding 2):

- build_resolution_csv and its report move to a new metadata-only module
  bst_x/pipeline/video_metadata.py, whose CLI is resolution-only (the
  old --skip-download behaviour becomes the only behaviour).
- The match.csv -> candidates adapter (id -> video_id, video -> title,
  keep = True, EXCLUDED_VIDEOS filter applied before task creation)
  lives in bst_x/pipeline/download_adapter.py and calls the scraper's
  download_all_videos WITH the video-only mode explicitly enabled;
  the default audio-gated behaviour would reject exactly the URLs the
  mode exists for (independent review, revision 2).
- build_dataset.py calls the adapter, then video_metadata; the whole old
  bst_x/pipeline/download_videos.py module (download_all_videos,
  download_video, its CLI) is then deleted.

Scraper-side changes:

- An explicit video-only mode is added to the scraper downloader (the
  H.264 video-only selector plus the allow_missing_audio path, recording
  commentary_eligible = false). The current format string requires
  audio, so the mode protects URLs for which only video-only H.264 is
  available; whether any ShuttleSet URL actually needs it was not
  verified (D1 NOT CHECKED) and does not change the requirement.
- Filenames: scraper naming (<id>.mp4) is the contract for new
  downloads, but existing spaced-name videos ({id} {title}.mp4) stay
  readable; bst_x's clip glob and video_metadata's stem parsing accept
  BOTH forms, and fail loudly on an ambiguous double match. The
  downloader's completed-output detection must ALSO recognise a legacy
  spaced file as already-downloaded and skip it; otherwise a normal
  rerun downloads <id>.mp4 beside the legacy file and manufactures the
  very ambiguity the readers reject (independent review finding 8, revision 2).
- build_dataset's raw-video guard tightens to video extensions, because
  the scraper writes sources.toml into the video directory even when
  there is nothing to download.
- The downloader reads YTDLP_BIN, YTDLP_RETRIES, SLEEP_INTERVAL_S,
  MAX_SLEEP_INTERVAL_S, SLEEP_REQUESTS_S, LIMIT_RATE,
  CONCURRENT_FRAGMENTS and DOWNLOAD_WORKERS from scraper.config instead
  of hardcoding them (WP4-8, flipped from delete-constants to
  wire-config; values are identical today, so no behaviour change).
- The test-only download_video wrapper and its direct tests are deleted
  (WP4-9).

### R4. WP2 rows left as deliberate mirrors

WP2-6/9/10 stay as-is: bst_x-vs-bric and bst_x-vs-scraper lane pairs
with different contracts, and reworking those bric pipelines is barred
by R0. WP2-11 is overtaken by R3 (the bst_x downloader is deleted; the
scraper side defers to config). WP2-8 stays: result_utils.
plot_confusion_matrix has a live caller (bst_x_train.py:944, behind
show_confusion_matrix; D3 confirmed). WP4-10's two manifest readers stay
separate: one initialises a missing manifest and validates for
write-back, the other is read-only and must-exist, and the overlap is
about eight lines.

### R5. Annotator + courtkeynet: the WP3 rows, as corroborated by WP8

(The WP8-only rows WP8-2/3/15/16 are parked in R6, not approved here.)

Deletions, each with its documented test action (independent review finding 13):

- pick_landing: delete; tests call landing_window + pick_landing_to_end
  (WP3-12).
- detect_contacts: delete; test uses detect_contact_flags (WP3-7).
- fetch_span: delete; tests stack iter_span_frames results (WP3-8).
- build_detected_court_inputs: delete; test inspects
  build_detected_court_evidence(...).inputs (WP3-6).
- assert_floors: delete from production; the floor assertion moves into
  tests/test_annotator_scoring.py (WP3-11).
- load_winner_config + _load_winner_document: delete with their tests;
  load_boundary_winner is the contract (WP3-9).
- CourtKeyNetDetector.detect: delete; tests call detect_batch([f])[0]
  (WP3-13).
- RenderPlan.frames: delete outright; nothing reads it, tests included
  (WP3-10).
- Stale fixed-25fps aliases: delete BEST_CONFIG_THRESHOLDS,
  COURT_ABSENT_WINDOW, SUSTAINED_LOSS_FRAMES, MIN_DESCEND_SAMPLES;
  update their tests and docstrings to the live resolved values
  (WP3-3/4/5).

Dedups: WP3-1 (_find_rally_spans consumes _rally_regions' result),
WP3-2 (inout_verdict classifies landing_margins' result), WP3-17 (hoist
run_video's repeated span options into one local mapping).

The REPORTED-only rows are in scope but must be re-verified
first-hand before deletion; on confirmation, each has its own test
action (WP8 rows):

- score_stage8: delete with its shape test
  (tests/test_calibration_scoring.py:150-155).
- select_best_config: delete with its test
  (tests/test_calibration_selection.py:143-148).
- select_contact_live_winners: delete with its stability test
  (tests/test_calibration_selection.py:121-127).
- court_scale_boxes: delete with its two smoke assertions
  (tests/test_point_winner.py:648-655,
  tests/test_scraper_stage8.py:748-755).

Deliberate mirrors stay: static vs detected court evidence builders,
composition vs replay dead-mask policies, and the two rolling-mean modes
(zero-fill vs NaN-ignoring, selected by SmoothingMode).

### R6. Tests (WP8)

The circular test-keeps-dead-function pairs die with their functions
(covered by R5 and R8). WP8-14's legacy call-shape test retires together
with the Stage 7 _rest_mask retirement, not before. WP8-1's mirrored
setup stays: the two suites import the code under different package
spellings (src.bst_x... vs flat preparing_data...), which creates
distinct class identities in Python, so a shared conftest helper would
break isinstance checks.

Ruled 2026-08-02:

- WP8-2: approved dedup. The ServeSetupInputs defaults move to
  tests/conftest.py; the B1/B2 files keep only their count/wrist
  overrides local.
- WP8-3: approved dedup. One conftest writer for complete three-column
  doubles-flag rows; the blank-rally adaptation stays at its callsite.
- WP8-15: approved delete. _assert_ndet_fits_int8 and its four calls
  go; the full-path dtype and n_max=128 boundary tests stay.
- WP8-16: approved delete. test_environment.py goes; real test imports
  cover the dependency check.

### R7. bst_x internals (WP5)

Leave alone, except: WP5-1 (raw_extract.build_stem_to_path absorbed into
pipeline/clip_index.build_clip_path_index) and WP5-6 (the hparam_sweep
readers call the cumulative_mean / per_class_mean reducers for the
arithmetic while KEEPING their own fixed-five completeness validation;
the dedup must not silently accept an incomplete sweep; independent review
finding 9).

### R8. bric + api (WP6)

- WP6-1: delete detect_and_track, PlayerTrack and the private helper
  chain (~200 LoC including the ByteTrack wiring); keep
  DEFAULT_YOLO_WEIGHTS (two live importers).
- WP6-9: delete bric/eval's _select_device copy; eval calls the train
  helper it already imports beside.
- WP6-10: delete the api's _RGB_MEAN_VAL / _RGB_STD_VAL literals; import
  bric/dataset's constants (the api already imports that module).
- WP6-6/7/8: delete _summary_live, _live_splits, is_available,
  available_splits, the two tests at tests/test_api.py:125-136, and the
  stale retention comments around registry.py:473-477. Confirmed dead
  first-hand and by D3 across routes, dynamic dispatch, docs and the
  frontend tree.
- WP6-2/3/4/5/11 and the WP6 C rows: RULED leave alone, 2026-08-02.
  No code change and no added test
  coverage; the refactor pass does not touch them.
- api_contract.md match-filter drift: RULED fix, and done in-session
  (2026-08-02): the phantom `match` query param was removed from the
  /clips and /stats endpoint descriptions after confirming the routes
  at src/api/registry.py:399-434 and :387-397 expose no such filter.

### R9. Scripts census (WP7)

No archiving. The 40 finished-check candidates already sit in named
subtrees (validation_scripts/rtmlib_migration/, .../refactoring/,
.../mmpose_heuristic_investigation/, scripts/plots/), which satisfies
the ruling that organised subtrees are fine. The five loose singles
(calibration_ece.py, collation_fulldiff.py, compute_clip_length_stats.py,
failsafe_bst_mmpose_zeroing_check_equivalence.py,
analyse_first_last_stroke_buffered_search.py) stay in place too
(confirmed by the 2026-08-02 ruling).

### Verification provenance for this ruling set

A first-hand check covered WP2-8's live caller, WP6-6/7/8 dead status,
WP4-8's config usage, the WP4-9/10 contracts, WP8-1's import spellings,
the WP7 subtree layout, and the manifest-vs-registry decode source at
api/bric_inference.py:180-184. Automated read-only checks using pyrefly
and grep supplied D1's downloader contract map, D2's consumer table and
stored-name check, and D3's renderer history and probe sweep. Rows marked
REPORTED in `findings.md` keep their re-verify-before-acting condition at
execution time. An independent review covered revision 1 with fourteen
findings. Revision 2 carried four findings and added two. Every finding was
adopted. The taxonomy key dispositions rest on a first-hand `git grep`
reference sweep run 2026-08-02.

Point of confusion: tracked callers made TrackNet look classifier-only.
Evidence: the scrape lane hand-runs TrackNet through the s29 chain at
scraper/config.py:154-158. Resolution: R1 changed the destination from
classifier_shared/_vendor to shared/tracknetv3, kept the destination flat,
and expanded the consumer count from three to four. Hand-run contract values
documented in config comments are live surface that caller tracing cannot see.
