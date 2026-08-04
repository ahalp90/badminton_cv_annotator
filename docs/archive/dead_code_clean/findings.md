# Merged findings ledger

Sources: eight automated read-only sweeps, with raw returns under
`findings/wp1.md` through `findings/wp8.md` (161 ledger rows). This file is the
deduplicated and checked view. The raw returns carry the full evidence chains.
Categories are defined in `audit_plan.md`.

Verification legend:

- **VERIFIED**: confirmed by a first-hand `git grep` check of all tracked
  content, including `.md`, `.sh`, and `.yml` references, against the root
  manifest. Semantic call hierarchy did not resolve Python callables in this
  project. Each row therefore rests on direct grep plus independent pyrefly
  references.
- **REPORTED**: found by an automated read-only sweep but not independently
  re-verified. Re-verify before acting. This label replaces the former
  **DELEGATE** label.
- **AMENDED**: the first-hand check changed part of the reported claim.
- **REFUTED**: the first-hand check overturned the reported claim.

## 1. The vendored TrackNet mirror (bric vs bst_x)

- WP1-1 | C | The two trees are near-identical: all eight Python files that do
  the work (predict, dataset, model, inference_utils, utils/general,
  write_inpaint_metadata) are byte-identical; only batch_predict.py diverges
  (bst_x adds --large_video forwarding) plus README/requirements comment drift.
  git log shows bst_x is the maintained side. VERIFIED (WP1 ran cmp -s per
  file; a first-hand check confirmed the consumed surfaces below).
- WP1-2 | D-prod | bric's _vendor/tracknetv3/batch_predict.py has no caller:
  bric invokes predict.py via subprocess; the api imports load_models /
  predict_video; only bst_x's copy of batch_predict.py is invoked
  (shuttle_extractor.py:149). VERIFIED.
- Consolidation picture: bric consumes predict.py (subprocess) and the api
  imports from the vendor package; bst_x consumes batch_predict.py. Because the
  working files are byte-identical today, one tree can serve both consumers if
  the refactor retargets bric's subprocess path and the api imports, keeping
  bst_x's --large_video behaviour. The mirror's only current benefit is
  sys.path/namespace isolation. R1 in `decisions.md` gives the final ruling.

## 2. Cross-package mirrors (bst_x/pipeline vs scraper vs shared)

The big surprise: most of this duplication is documented and deliberate.
src/shared/README.md records the isolation trade-off (bric and annotator must
not import bst_x internals); download_videos.py:73-75 states its duplication
on purpose. WP2 therefore filed comparison notes, not supersede findings:

- WP2-1 | P | the two yt-dlp downloaders share worker shape but different
  output contracts (resolution CSV vs audio-gated TOML manifest). REPORTED.
- WP2-2/3/4/5 | C | court maths, player mapping, clip-bound maths and taxonomy
  registries all mirrored between bst_x/pipeline and src/shared. REPORTED, with
  two live divergences worth a deliberate ruling:
  - shared/dataset.compute_clip_bounds omits bst_x's max(0, start_f) clamp
    (shared/dataset.py:260-277): negative clip starts behave differently.
  - the taxonomy registries map driven_flight differently (bst_x: -> drive;
    shared legacy: -> unknown) and use different registry names.
- WP2-6/9/10/11 | C | shuttle extraction, video-metadata scanners, clip
  generation and throttle constants: same problem class, different contracts
  per lane. REPORTED.
- WP2-7 | S | scripts/plots/confusion_matrix.py duplicates the renderer in
  shared/eval_plots.py (annotate_cells / render_panel); absorb into
  eval_plots with output-path and figsize params, keep the script as a thin
  CLI. VERIFIED (eval_plots' own docstring references the script; bric/eval.py
  is the shared caller).
- WP2-8 | C | bst_x/result_utils.plot_confusion_matrix is a separate legacy
  renderer with different ordering/zero-division semantics. VERIFIED as
  separate (bric/eval.py:187 calls the shared one, not result_utils).

## 3. src/shared dead surface (WP4)

- WP4-1 | D-unreach | shared/court.py to_court_coordinate + check_pos_in_court:
  zero callers of the shared copies; the live copies are bst_x's own
  court_utils (used by heuristics). VERIFIED. Delete, trim README.
- WP4-4 | D-unreach | shared/temporal.py clip_window_seconds/clip_window_frames:
  README-only references. VERIFIED. Delete.
- WP4-5 | T | shared/temporal.py subsample_indices: tests only. VERIFIED.
  With WP4-4 this kills the whole module unless a production adopter is named.
- WP4-6 | D-prod (amended from D-unreach) | shared/video_io.py
  read_frames + iter_frames: tests reference them, nothing else; iter_frames
  is also read_frames' internal dependency. VERIFIED (courtkeynet's
  score_hand_corners has its own local read_frames, unrelated). Delete pair
  + tests.
- WP4-7 | T | video_io.read_frame_at / write_frame_thumbnail: test-only
  (read_frame_at's one non-test caller is write_frame_thumbnail itself,
  video_io.py:146); the "API contract" claim in the docstring is stale.
  VERIFIED.
- WP4-2 | U, AMENDED | shared/dataset.SPLITS_V2 + _load_splits_v2: computed at
  import, never consumed: but build_shots_master.py:11 imports the name
  without using it, so the fix is delete constant + loader + stale import.
  VERIFIED.
- WP4-3 | U | shared/taxonomy PLAYERS / UNPREFIXED_TYPES: zero callers of the
  shared copies (the live PLAYERS is bst_x pipeline/config's own). VERIFIED.
- WP4-8 | U | scraper/config CONCURRENT_FRAGMENTS / DOWNLOAD_WORKERS: zero
  consumers; downloader hardcodes the values. VERIFIED. (WP2-11 note: the
  downloader's hardcoding is itself documented as deliberate.)
- WP4-9 | D-prod | scraper download_video wrapper: CLI uses
  download_all_videos; only tests call the wrapper. VERIFIED. Delete with its
  tests.
- WP4-10 | C | stage11 vs downloader manifest readers: deliberate. REPORTED.

## 4. annotator + courtkeynet (WP3, corroborated by WP8)

Test-only wrappers (T): WP3 and WP8 found these independently. All were
VERIFIED by direct grep:
- point_winner.pick_landing (production uses pick_landing_to_end)
- rally_segmentation.detect_contacts (production uses detect_contact_flags)
- validation_overlay/core/decode.fetch_span (production streams
  iter_span_frames)
- court_evidence.build_detected_court_inputs
- calibration/gt_scoring.assert_floors
- calibration/sweep.load_winner_config + _load_winner_document (D-prod;
  live path is load_boundary_winner)
- courtkeynet wrapper.CourtKeyNetDetector.detect (production uses detect_batch)

WP8-only additions (REPORTED): calibration/scoring.score_stage8,
calibration/selection.select_best_config + select_contact_live_winners,
rally_segmentation.court_scale_boxes.

Stale config aliases (U, REPORTED, all with named live replacements):
- config.BEST_CONFIG_THRESHOLDS (live: SHIPPED_THRESHOLDS)
- config.COURT_ABSENT_WINDOW (live: scale_for_fps(fps).court_absent_window)
- point_winner.SUSTAINED_LOSS_FRAMES / MIN_DESCEND_SAMPLES (live:
  FpsConstants fields)
- validation_overlay cli RenderPlan.frames property (D-prod, unread)

Supersedable duplication inside the package (S, REPORTED):
- WP3-1: _find_rally_spans recomputes scaffolding _rally_regions already
  returns.
- WP3-2: point_winner.inout_verdict repeats landing_margins' geometry.

One O finding with a stated cost (WP3-17, REPORTED): run_video repeats the
resolved span-finder options across five branches; a new knob needs
synchronised edits. Fix is a local hoist, no new abstraction.

Deliberate mirrors left alone (C): static vs detected court evidence
builders; composition vs replay dead-mask policies; the two rolling-mean
modes.

## 5. bst_x internals (WP5)

- WP5-1 | S | raw_extract.build_stem_to_path duplicates
  pipeline/clip_index.build_clip_path_index (functionally identical glob).
  VERIFIED. Absorb into clip_index.
- WP5-6 | S | hparam_sweep read_run_mean/read_run_per_class_mean duplicate
  cumulative_mean/per_class_mean arithmetic. VERIFIED (both readers hand-roll
  the sums at :300-324; reducers at :355-371 per WP5's refs).
- WP5-2 | U | prepare_train loads full _failed.npy payloads where only lengths
  are used. REPORTED.
- WP5-10 | U | aim_backfill._derive_tags dead serial_no param + per-serial
  recompute. REPORTED.
- WP5-11 | U | bst_x_infer.infer unpacks labels it never reads. REPORTED.
- WP5-12 | T | sticky_anchor compatibility aliases: tests + archive only;
  source already marks them for Stage 7 removal. VERIFIED via WP5+WP8 rows.
- C rows (deliberate, left alone): current.apply parity oracle vs
  detect_players_2d; NumPy/Torch bone parity pair; tempose vs bst attention
  (unification explicitly declined in a past refactor); data_access vs
  collation clip readers; the run/reporting module family; hparam_sweep vs
  collation_runner.
- WP5 outward notes REFUTED by first-hand checks: scraper does not call
  bst_x download_videos (no such import at download_scraped_videos.py:499);
  bric/eval.py:187 calls shared eval_plots, not result_utils.

## 6. bric + api (WP6)

- WP6-1 | D-unreach | bric/perception/players.py detect_and_track + PlayerTrack
  + helper chain: zero callers anywhere (live paths use frame-detection).
  VERIFIED. Keep DEFAULT_YOLO_WEIGHTS (two live importers). Biggest single
  dead block found (~200 LoC incl. ByteTrack wiring).
- WP6-9 | S | bric/eval._select_device is byte-identical to
  bric/train._select_device, and eval already imports train helpers. VERIFIED.
- WP6-10 | S | api/bric_inference duplicates the RGB mean/std constants and
  formula from bric/dataset; the api even comments "Mirror
  src/bric/dataset.py::_build_rgb normalization exactly". VERIFIED.
- WP6-5 | U | api model_id plumbing validated and stored but never dispatched
  on (fallback reads first registry model). REPORTED.
- WP6-6/7/8 | U/T | registry._summary_live, registry._live_splits,
  bst_x_inference.is_available / available_splits: dead or test-only probe
  surface. REPORTED.
- WP6-2/3/4 | U | dataset builds disabled-lane tensors; preprocess cache-repair
  branch unreachable on reruns; write-only cache fields (vid, video_path,
  n_frames, fps, top_conf, bottom_conf, dense frame). REPORTED: the
  write-only-fields row alters an on-disk cache contract; verify hard before
  acting.
- WP6-11 | U | the bric api accepts a court-enabled deployment manifest but
  never reads court_encoder or supplies the three court tensors BRICNetwork
  requires; the knob is not end-to-end (deployed manifest has
  use_court: false). REPORTED, medium confidence: an api contract question,
  verify before refactoring.
- WP6-14 | C | the api inference triplet is a dispatcher over three different
  contracts (stub / BRIC upload / BST library), not a supersedable triple.
  REPORTED, matches WP2's exclusion.
- WP6-16, WP6-12, WP6-13, WP6-15 | C | deliberate lifecycle mirrors. REPORTED.
- Outward: docs/api_contract.md advertises a `match` filter the registry does
  not expose. This is documentation drift, not code. REPORTED.

## 7. Census: scripts + validation trees (WP7, 77 files)

No file in the census is unreachable; every one traced to a root or a
documented one-shot purpose. 40 of the 77 files are archive candidates
(D-prod, REPORTED unless noted):
- rtmlib_migration/: all 14 files, completed migration gate family.
- validation_scripts/refactoring/: all 8 smokes/gates, completed passes.
- mmpose_heuristic_investigation/: 6 of its files (finished investigation).
- scripts/plots/: 7 presentation charts pinned to historical run IDs
  (confusion_matrix.py also carries the WP2-7 absorb).
- singles: calibration_ece.py, collation_fulldiff.py,
  compute_clip_length_stats.py, analyse_first_last_stroke_buffered_search.py,
  failsafe_bst_mmpose_zeroing_check_equivalence.py.
- scripts/archive boundary: CLEAN: no live import/path/subprocess into it
  (only doc mentions and the pyproject ruff exclusions). VERIFIED via WP7's
  explicit sweep.
- Everything else classified LIVE with the documenting root cited per row
  (see findings/wp7.md).

## 8. Tests (WP8)

- S: serve-setup default builders duplicated across the B1/B2 test files;
  doubles-flag CSV writer duplicated in two files. Conftest candidates.
  REPORTED.
- T-list: corroborates the annotator/calibration wrappers in section 4 and
  api available_splits in section 6.
- WP8-14: a test freezes a call shape production marks legacy
  (Stage 7 retirement): retire together. REPORTED.
- O: _assert_ndet_fits_int8 re-proves a bounded cast; test_environment.py
  imports six heavy deps to assert a boolean. REPORTED.
- WP8-1 kept as C: the sticky-anchor vs doubles-overcount setup mirror crosses
  namespace spellings deliberately.

## Refuted / amended row summary

- REFUTED: WP5 outward "scraper calls bst_x downloader"; WP5 outward
  "bric/eval reuses result_utils".
- AMENDED: WP4-2 (stale import also needs removing); WP4-6 (D-prod, not
  D-unreach: tests reference the pair).
- All other rows stand as filed.
