## 1. Question investigated

Read-only census of duplicated constants, formulas, filename conventions, validation guards and enum-like strings under `src/annotator/**` and `src/scraper/**` on `main`.

The scope contained 52 Python files and 15,945 Python lines.

## 2. Files/symbols inspected

Used Serena symbol overviews, `find_symbol`, `find_referencing_symbols`, `find_declaration`, and scoped text searches.

Key symbols inspected:

- `annotator.fps_constants`: `ScalingKind.scale`, `scale_for_fps`, `probe_fps`
- `annotator.config`, `resolve`, `types`
- Rally modules: `segmentation`, `spans`, `contacts`, `serve`, `evidence`, `trajectory`
- `replay_mask`, `composition_mask`, `dead_mask`, `court_evidence`, `point_winner`, `run_video`
- Calibration: `scoring`, `gt_scoring`, `selection`, `sweep`, `schemas`
- `e2e_court_annotator`, `experiment_records`, `broadcast_timeline_labels`
- Scraper: `config`, `search_index`, `relevance_triage`, `transcript_acquisition`, `download_scraped_videos`, `commentary_pairing`, `commentary_cleaning`

## 3. Concrete evidence

### Numeric constants and conventions

- FPS scaling is centralised in `fps_constants.py`: `BASE_FPS = 30.0`, base speed values, and the frame-count/per-frame-speed formulas are at `src/annotator/fps_constants.py:18-37`. The same positive-finite FPS guard appears in `ScalingKind.scale` at `:32-33` and `scale_for_fps` at `:79-80`.

- `scale_for_fps` contains repeated base-30 values for separate fields: `15.0` for `court_absent_window`, `replay_mask_min_frames`, `serve_stillness_window_frames` and `composition_min_scene_len` at `src/annotator/fps_constants.py:92-99`. The dataclass comments identify replay masking and composition scene length as distinct concepts at `:51-65`.

- The legacy 25-FPS surface is named `_AT_25FPS` at `src/annotator/config.py:33-42`. The contact module independently calls `scale_for_fps(25.0)` three times at `src/annotator/rally/contacts.py:13-18`. No separate numeric `FPS_25` constant was found.

- `25.0` also appears as expected fixture metadata at `src/annotator/calibration/fixtures.py:299,319` and `src/annotator/e2e_court_annotator.py:122-124`. The expected 30-FPS fixture appears at `e2e_court_annotator.py:125`.

- The literal `0.5` is used for separate named policies:
  - `COMPOSITION_KEEP_VOTE` and `DOUBLES_SPAN_FRACTION`: `src/annotator/config.py:99,109`
  - `VISIBILITY_REST_FRAC`: `src/annotator/rally/spans.py:31`
  - `PLAYER_PRESENT_MIN_FRAC`: `src/annotator/rally/serve.py:89`
  - `SCENE_VALID_MIN_FRACTION`: `src/annotator/court_evidence.py:46`
  - `NET_COURT_Y`: `src/annotator/point_winner.py:92`
  - net-band geometry retypes `0.5` at `src/annotator/court_evidence.py:312-313`
  - scraper failure threshold: `src/scraper/config.py:128` and raw downloader comparison at `src/scraper/download_scraped_videos.py:586`

- The failure comparisons diverge: transcript acquisition uses `>` against `TRANSCRIPT_FAIL_FRACTION_BLOCK` at `src/scraper/transcript_acquisition.py:318,338`; downloading uses `>= 0.5` directly at `src/scraper/download_scraped_videos.py:586`.

- `0.15` occurs under different names: `PROXIMITY_MAX` and `SLOWMO_SPEED_FRAC` at `src/annotator/config.py:43,92`; `DENSITY_MIN_PER_MIN` at `src/scraper/config.py:142`.

- `0.05` occurs as `PERSPECTIVE_SHIFT_THRESHOLD` at `src/annotator/config.py:87` and as the default `reentry_guard_buffer` at `src/annotator/config.py:131-133`.

- The central table uses `12.0` for several frame fields at `src/annotator/fps_constants.py:94,98,100,104`. A separate compatibility default retypes `12` as `BODY_UNIT_HALF_WINDOW` at `src/annotator/rally/evidence.py:38-39`, with the resolved value passed explicitly at `:249-255`.

- Base-30 `5` is present in the central contact tolerance tuple at `src/annotator/calibration/scoring.py:16`; strict scoring retypes defaults `(5, 10)` at `:272-292`; `canonical_tolerance` independently scales `5.0` at `src/annotator/calibration/gt_scoring.py:403-405`.

- The wide-edge scoring window retypes base-30 `90` at `src/annotator/calibration/scoring.py:354`, while the FPS table uses `90.0` for `end_rest_frames` at `src/annotator/fps_constants.py:92` and the sweep grid includes `90.0` at `src/annotator/calibration/sweep.py:65-70`.

- Landing settings are repeated as identical raw values:
  - `LANDING_OPTIONS`: `src/annotator/e2e_court_annotator.py:79-85`
  - GT scoring construction: `src/annotator/calibration/gt_scoring.py:464-472`
  - manifest serialisation: `src/annotator/e2e_court_annotator.py:633-640`
  No shared landing-options definition was found between these modules.

- Detector dimensions are named centrally as `DETECTOR_RESOLUTION = (512.0, 288.0)` at `src/annotator/court_evidence.py:43`; the E2E metadata guard retypes `(512, 288)` at `src/annotator/e2e_court_annotator.py:503-509`.

- `FINE_BATCH_SIZE = 16` is defined at `src/scraper/commentary_cleaning.py:51-53`, but `_score_chunks` passes `batch_size=16` directly at `:160-170`. `annotator.inpaint_guard` has a separate `DEFAULT_WINDOW = 16` at `src/annotator/inpaint_guard.py:21`.

### Repeated formulas

- Frame-to-time and time-to-frame conversions are implemented with different rounding rules:
  - `int(start_s * fps)`: `src/scraper/commentary_pairing.py:91-94`
  - `end_frame / fps`: `src/scraper/commentary_pairing.py:172-173`
  - half-up frame conversion using `floor(... + 0.5)`: `src/annotator/video_outcomes.py:348-350`
  - exact `Fraction` conversions with half-frame bounds: `src/annotator/validation_overlay/core/decode.py:189-193`
  - `int(exact_seconds * fps)`: `src/annotator/validation_overlay/core/timeline.py:110-118`
  - direct display conversion: `src/annotator/manual_broadcast_timeline_annotator.py:433-436`
  `ScalingKind.FRAME_COUNT` centralises base-30 frame scaling, not these generic timestamp conversions.

- F1 has a shared implementation at `src/annotator/calibration/scoring.py:57-60`. `_prf` calls it at `:439-443`; GT metric flattening calls it at `src/annotator/calibration/gt_scoring.py:706-709`; E2E metrics recompute precision and recall before calling it at `src/annotator/e2e_court_annotator.py:592-602`.

- Boundary selection independently inlines the count form `2 * TP / (2 * TP + FN + FP)` at `src/annotator/calibration/selection.py:53-57`. This is mathematically equivalent to the shared F1 calculation for those counts, with different surrounding inputs.

- The shared shuttle-speed formula is `np.diff` plus `np.linalg.norm` in `src/annotator/types.py:105-112`. Other direct norm calculations include:
  - contact impulse: `src/annotator/rally/contacts.py:31-57`
  - player proximity: `src/annotator/rally/contacts.py:144-152`
  - serve drift: `src/annotator/rally/serve.py:62-85`
  - sticky wrist distance: `src/annotator/rally/evidence.py:211-233`
  - landing wrist and ankle distances: `src/annotator/point_winner.py:403-420`
  - court-out and corner-error distances: `src/annotator/point_winner.py:611-627,643-651`
  These use different inputs, masks and unit normalisers.

- Rolling means are implemented twice in one module. `_rolling_mean` uses convolution sums/counts at `src/annotator/rally/trajectory.py:8-21`; `_nan_rolling_mean` repeats the same structure with a validity mask at `:24-37`.

- No active `iou` match was found in the scoped source search.

### Filename, glob and suffix conventions

- `sources.toml` has a central name at `src/scraper/config.py:30-35`. The downloader constructs the path through that constant at `src/scraper/download_scraped_videos.py:486`; pairing reads the same constant at `src/scraper/commentary_pairing.py:337`.

- Transcript sidecars use the repeated construction `TRANSCRIPTS_DIR / f'{video_id}.json'` at `src/scraper/transcript_acquisition.py:301` and `src/scraper/relevance_triage.py:53,238`.

- Chunk sidecars use `CHUNKS_DIR / f'{video_id}.json'` at `src/scraper/relevance_triage.py:258`, `src/scraper/commentary_pairing.py:219-225`, and `src/scraper/commentary_cleaning.py:196,414`.

- Replay masks use the same `_replay.npy` suffix at the writer `src/annotator/replay_mask.py:343-345` and scraper reader `src/scraper/commentary_pairing.py:228-236`. No shared suffix constant was found.

- Dead masks use `_dead_mask.npy` at the writer `src/annotator/composition_mask.py:164-167`, annotator reader `src/annotator/rally/cli.py:42-54`, and fixture path builder `src/annotator/calibration/fixtures.py:89`.

- `manifest.json` is retyped in the E2E runner at `src/annotator/e2e_court_annotator.py:868,920,1300` and in experiment-record reads at `src/annotator/experiment_records.py:74-84,204-206`. No shared manifest filename constant was found.

- Calibration has a central filename-to-schema mapping at `src/annotator/calibration/schemas.py:123-132`; the sweep uses that mapping while constructing files at `src/annotator/calibration/sweep.py:462-521`. `WINNER_FILENAME = "config_winner.json"` is centralised at `sweep.py:53-56` and used at `:451,531,541`.

- `.csv.gz` handling is local to broadcast labels: reading at `src/annotator/broadcast_timeline_labels.py:210-217`, suffix validation and temporary-name construction at `:261-277`. No `.json.gz` or `.npy.xz` occurrence was found in the scoped source.

- The same video extension set is independently defined at:
  - `src/scraper/download_scraped_videos.py:32`
  - `src/scraper/commentary_pairing.py:41-44`
  - `src/scraper/commentary_cleaning.py:46-47`
  All three contain `{'.mp4', '.mkv', '.webm', '.avi', '.mov'}`.

- `video_fps.csv` is locally defined only in pairing at `src/scraper/commentary_pairing.py:41-44`; it is not in `scraper.config`.

### Validation and guard logic

- FPS validation is repeated with identical logic inside `src/annotator/fps_constants.py:30-37,77-80`. `VideoMetadata` independently checks finite positive FPS at `src/annotator/broadcast_timeline_labels.py:46-52`. `probe_fps` adds invalid-rate and VFR checks at `fps_constants.py:122-125`.

- Full manifest validation occurs in `src/scraper/download_scraped_videos.py:147-179`. Pairing repeats dataset, video-table and entry-dictionary checks at `src/scraper/commentary_pairing.py:239-256`, but does not repeat the scalar and field-type checks.

- Frame-aligned boolean-array validation appears at:
  - `src/annotator/replay_mask.py:151-160`
  - `src/annotator/replay_mask.py:246-260`
  - `src/annotator/dead_mask.py:25-42`
  - `src/annotator/rally/evidence.py:74-79`
  - `src/scraper/commentary_pairing.py:228-236`
  The pairing check validates dimensionality and dtype but not mask length.

- The serve presence floor is applied in both median-count branches at `src/annotator/rally/serve.py:275-280,291-296`, using the same `PLAYER_PRESENT_MIN_FRAC` constant.

### Enum-like strings

- Central enum definitions are in:
  - `DeadMaskMode`, `SmoothingMode`, `SpanOpen`, `ReentryGuardVariant`: `src/annotator/types.py:21-56`
  - `Half`, `Verdict`, `VerdictSource`: `src/annotator/point_winner.py:58-81`
  - `SceneTruth`: `src/annotator/broadcast_timeline_labels.py:28-35`
  - scraper substreams: `src/scraper/config.py:62-63`

- Raw enum-like strings remain at `src/annotator/e2e_court_annotator.py:636` (`"dead_mask_mode": "replay"`).

- External CSV side values are converted from raw `"Top"`/`"Bot"` strings in `src/annotator/calibration/gt_scoring.py:407-409,621,630`, then compared as `Half` enum members at `:642-659`.

- `DeadMaskMode(mode)` performs string-to-enum dispatch at `src/annotator/dead_mask.py:71-83`. Calibration strategy names are dynamically converted with `SpanOpen[name]`, `ServeStartMode[name]` and `ServeStartClose[name]` at `src/annotator/calibration/sweep.py:245-272`.

- Scraper boolean CSV values are raw strings. The convention is documented at `src/scraper/config.py:40-45`; consumers compare to `'True'` at `src/scraper/download_scraped_videos.py:457` and `src/scraper/commentary_cleaning.py:187,404`.

## 4. Callers/consumers found

- Serena reference tracing found `scale_for_fps` consumers in config, rally segmentation, contacts, replay masking, composition masking, court evidence, calibration sweep and scraper pairing: `src/annotator/config.py:36`, `rally_segmentation.py:52,117-127`, `rally/contacts.py:13-18`, `replay_mask.py:72`, `composition_mask.py:158-160`, `court_evidence.py:186-190`, `calibration/sweep.py:291-299`, `src/scraper/commentary_pairing.py:26,150`.

- `compute_speed` is consumed by replay masking and landing kinematics at `src/annotator/replay_mask.py:193` and `src/annotator/point_winner.py:420`.

- `safe_f1` is consumed by scoring, GT flattening, selection and E2E metrics at `src/annotator/calibration/scoring.py:439-443`, `gt_scoring.py:706-709`, `selection.py:113-123`, and `e2e_court_annotator.py:600-602`.

- CLI entry points are present in scraper downloader and annotator E2E runner at `src/scraper/download_scraped_videos.py:542-592` and `src/annotator/e2e_court_annotator.py:1332-1348`.

## 5. Counterevidence / surprises

- `annotator.config` builds `SHIPPED_THRESHOLDS` from named values at `src/annotator/config.py:69-79`.

- `scraper.config` explicitly declares shared output paths, columns, sidecar layout and rate limits as a single source at `src/scraper/config.py:1-7`; `read_candidates` and `write_candidates` are central helpers at `:247-273`.

- `rally_segmentation.py` documents `compute_speed`, `true_runs` and `rolling_nanmedian` as shared primitives re-exported from `annotator.types` at its module docstring and imports.

- `CONTACT_TOLERANCES_BASE30`, `WINNER_FILENAME` and `CSV_COLUMNS_BY_FILENAME` already have central definitions, although some consumers still retype individual values or filenames.

- No files were changed. No diff exists for the requested scope.

## 6. Unresolved or dynamic surfaces

- `getattr` is used for dynamic field access in `src/annotator/e2e_court_annotator.py:272,585-586`, `src/annotator/calibration/gt_scoring.py:702`, `src/annotator/calibration/sweep.py:296,299,314,321`, and `src/annotator/court_evidence.py:144,224`.

- CLI and serialized configuration values can enter through enum conversions and raw strings. Static text search cannot enumerate external callers or user-provided values beyond the conversion sites above.

- Assumption: comments and docstrings were reported only when they documented an active convention; they were not counted as active constants.

- Checks run: scoped `rg`, `nl -ba`, `wc`, `git status`, `git diff --stat`, and Serena structural queries. Read-only commands completed without source changes. Tests, lint and type checks were not run because this was an evidence-only census.

- Current status: `main...origin/main`; pre-existing changes reported in `.gitignore`, `.serena/` and `scratch/review_2026_aug_07/`. No files outside the requested scope were altered.