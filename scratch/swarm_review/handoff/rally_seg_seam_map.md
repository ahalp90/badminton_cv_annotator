# Verified split-seam map: `rally_segmentation.py`

Produced 2026-08-04 by gpt-5.6-luna (reasoning effort max), read-only source
analysis at main commit 1afc86a, commissioned by the review orchestrator.

All line ranges below are inclusive. The three candidate owners are **S** for serve setup and span selection, **C** for contact detection, gating, and whole-stage composition, and **B** for CLI and batch handling. **X** marks shared support or a boundary façade that does not belong cleanly to one of those three owners.

Verification used an AST inventory, full line-numbered source inspection, and source-level reference search. The optional semantic-navigation interfaces were not available in this session.

## 1. Top-level definitions and classes in source order

| Candidate owner | Definition or class | Lines |
|---|---|---:|
| X shared support | `scale_thresholds` | 106-123 |
| X shared support | `_rolling_mean` | 126-139 |
| X shared support | `_nan_rolling_mean` | 142-155 |
| S serve/setup and span selection | `CourtGeo` | 161-174 |
| S serve/setup and span selection | `ServeSetupInputs` | 177-222 |
| S serve/setup and span selection | `series_drift` | 225-248 |
| S serve/setup and span selection | `serve_setup_still` | 255-293 |
| S serve/setup and span selection | `build_serve_setup_inputs` | 296-330 |
| S serve/setup and span selection | `ServeStartMode` | 333-343 |
| S serve/setup and span selection | `ServeStartClose` | 346-357 |
| S serve/setup and span selection | `ServeStartOptions` | 360-392 |
| S serve/setup and span selection | `_gap_is_high_shot_oob` | 402-413 |
| S serve/setup and span selection | `_gap_passes_reentry_guard` | 416-433 |
| S serve/setup and span selection | `_gap_state_rest_mask` | 436-457 |
| S serve/setup and span selection | `_rest_mask` | 460-486 |
| S serve/setup and span selection | `_rally_regions` | 489-517 |
| S serve/setup and span selection | `_find_rally_spans` | 520-552 |
| S serve/setup and span selection | `_find_rally_spans_span_open` | 555-580 |
| S serve/setup and span selection | `_find_rally_spans_quiet_start` | 583-598 |
| S serve/setup and span selection | `court_scale_slots` | 601-620 |
| S serve/setup and span selection | `_serve_distance_ratio_passes` | 623-635 |
| S serve/setup and span selection | `_last_rest_close` | 638-653 |
| S serve/setup and span selection | `_valid_serve_window` | 656-660 |
| S serve/setup and span selection | `_valid_serve_threshold` | 663-667 |
| S serve/setup and span selection | `_sticky_serve_setup_before` | 670-746 |
| S serve/setup and span selection | `_serve_start_find_rally_spans` | 749-862 |
| C/X boundary support | `apply_replay_mask` | 868-901 |
| C contact detection and gating | `span_impulses` | 915-941 |
| C contact detection and gating | `rolling_floor` | 944-961 |
| C contact detection and gating | `impulse_cell_candidates` | 964-1003 |
| C contact detection and gating | `detect_contact_flags` | 1006-1011 |
| C contact detection and gating | `contact_proximity_ok` | 1014-1036 |
| C contact detection and gating | `wrist_contact_near` | 1039-1050 |
| C contact detection and gating | `suppress_contact_flags` | 1053-1068 |
| S/C sticky-evidence bridge | `tracker_segments` | 1071-1117 |
| S/C sticky-evidence bridge | `build_sticky_result` | 1120-1225 |
| S serve/span façade | `find_rally_spans` | 1228-1256 |
| C contact façade | `assemble_contacts` | 1259-1280 |
| C/X whole-stage façade | `segment_video` | 1283-1367 |
| B CLI/batch | `_format_bool` | 1376-1383 |
| B CLI/batch | `_load_positions` | 1386-1394 |
| B CLI/batch | `_load_replay_mask` | 1397-1409 |
| B CLI/batch | `_read_string_id_table` | 1412-1424 |
| B CLI/batch | `_read_fps_table` | 1427-1432 |
| B CLI/batch | `main` | 1435-1582 |

The module-level choice table `_SPAN_OPEN_CHOICES` is at lines 1373-1373, and the module entry-point guard is at lines 1584-1586. Neither is a top-level definition or class.

## 2. Claimed regions and corrected boundaries

The claimed ranges are not three contiguous semantic blocks; the top-level inventory spans lines 106-1586 with the islands shown below [rally_segmentation.py:106-1586].

- **S, serve setup and span selection:** the main block is lines 161-862, including the serve-setup types, serve gate, rest-state logic, and span-opening rules. Its public span façade is separated at lines 1228-1256. The sticky-evidence producer at lines 1071-1225 is a bridge because its result supplies both serve-setup inputs and contact-gate inputs.

- **C, contact detection, gating, and whole-stage composition:** the contact primitives are lines 915-1068, `assemble_contacts` is the separated façade at lines 1259-1280, and `segment_video` is the whole-stage façade at lines 1283-1367. `apply_replay_mask` at lines 868-901 precedes the contact block but is input preprocessing for the whole stage, not contact detection.

- **B, CLI and batch:** the CLI support block begins with `_SPAN_OPEN_CHOICES` at line 1373, continues through `main` at lines 1435-1582, and ends with the module guard at lines 1584-1586.

The top-level definitions that do not fit cleanly into one of the three named lanes are `scale_thresholds` [106-123], `_rolling_mean` [126-139], `_nan_rolling_mean` [142-155], `apply_replay_mask` [868-901], `tracker_segments` [1071-1117], `build_sticky_result` [1120-1225], and `segment_video` [1283-1367]. `_nan_rolling_mean` is use-wise contact support, but its source position is in the shared prelude. `find_rally_spans` [1228-1256] and `assemble_contacts` [1259-1280] are displaced from their primary blocks but fit S and C respectively; `segment_video` fits C only as a public whole-stage façade and remains a C/X boundary.

The prior boundary anchors fall as follows:

| Claimed anchor | What the line actually contains | Correct owner |
|---:|---|---|
| 225 | The first line of `series_drift`, which runs through line 248 | S [225-248] |
| 760 | The docstring of `_serve_start_find_rally_spans`, which runs from line 749 to line 862 | S [749-862] |
| 900 | The assignment inside `apply_replay_mask`; the function runs from line 868 to line 901 | X [868-901] |
| 1290 | The `spans` parameter inside `segment_video`; the function runs from line 1283 to line 1367 | X [1283-1367] |
| 1400 | The docstring of `_load_replay_mask`; the function runs from line 1397 to line 1409 | B [1397-1409] |

## 3. Cross-region coupling

The following list names local helpers, constants, types, and data contracts that cross the candidate owners. Imported names that already belong to `.config`, `.fps_constants`, or `.types` are separated in section 4.

### S: serve setup and span selection

- S uses the shared `_rolling_mean` in `_rest_mask` at line 484. The same helper is used by C's `span_impulses` at lines 933-934, so `_rolling_mean` cannot be owned solely by S.
- S consumes the sticky-result contract in `build_serve_setup_inputs`: the input is typed as `StickyResult` at line 297, and the function reads `standing_count`, `wrist_dist_px`, `analysed`, ankle positions, and box heights at lines 306-323. X constructs those arrays in `build_sticky_result` at lines 1146-1154 and returns them at lines 1222-1225.
- S uses the shared `SpanOpen` type in `_find_rally_spans_span_open` at line 556 and `_serve_start_find_rally_spans` at line 751. C's whole-stage façade and B's CLI also use `SpanOpen` at lines 1287 and 1373-1466, so this type must not be private to S.

### C: contact detection and gating

- C uses shared smoothing helpers in `span_impulses`: `_rolling_mean` at lines 933-934 and `_nan_rolling_mean` at lines 938-939. `_rolling_mean` is also used by S at line 484.
- C consumes sticky evidence through `assemble_contacts`: its `sticky_distances` input is declared at line 1261, and the wrist gate reads it at line 1273. X creates per-slot body-unit distances at lines 1152-1153 and fills them at lines 1199-1215 before returning the `StickyResult` at lines 1222-1225.
- C/X's public composition uses S's `ServeStartOptions` type at line 1286 and S's `find_rally_spans` façade at lines 1350-1355. It also takes the shared `SpanOpen` type at line 1287.
- `assemble_contacts` is called by the C/X `segment_video` façade at lines 1359-1365. The contact primitives remain local to C, while the public composition crosses into X.

### B: CLI and batch

- B uses the shared `SpanOpen` type in `_SPAN_OPEN_CHOICES` at line 1373, in the parser at lines 1451-1454, and when resolving the option at line 1466. S also uses that type at lines 556 and 751, and C/X uses it in the `segment_video` signature at line 1287.
- B has no direct use site for a local S helper or a local C helper. `main` imports `run_video` at lines 1474-1475 and calls that external module function at line 1522; it does not call this module's `segment_video` directly.

### X: shared and boundary façades

- `segment_video` uses `apply_replay_mask` at line 1347, the S façade `find_rally_spans` at lines 1350-1355, and the C façade `assemble_contacts` at lines 1359-1365. Its `serve_start` parameter also exposes the S type `ServeStartOptions` at line 1286.
- `find_rally_spans` dispatches to the S rest mask and span finders at lines 1242-1256. This is why the function belongs with span selection even though it is located after the contact and sticky blocks.
- `build_sticky_result` uses the constant `BODY_UNIT_HALF_WINDOW` at line 1125. That constant is declared beside the S-side option types at line 396, while the builder is in X at lines 1120-1225.

## 4. Placement of module-level constants and types

### Local constants

| Name and definition | Use sites | Placement implied by the current uses |
|---|---|---|
| `VISIBILITY_REST_FRAC` [93-93] | `_rest_mask` [485] | S |
| `QUIET_START_REST_FRACTION` [94-94] | `_find_rally_spans_quiet_start` [594] | S |
| `IMPULSE_FLOOR_HALF_WINDOW_FRAMES` [98-98] | `rolling_floor` [946], `impulse_cell_candidates` [982] | C |
| `CONTACT_DEDUP_RADIUS_FRAMES` [99-99] | `impulse_cell_candidates` [983] | C |
| `CONTACT_IMPULSE_MULTIPLE` [100-100] | `impulse_cell_candidates` [985] | C |
| `FLOOR_EPS` [101-101] | `impulse_cell_candidates` [986] | C |
| `BODY_UNIT_WRIST_THRESHOLD` [102-102] | `wrist_contact_near` [1050] | C |
| `CONTACT_SUPPRESSION_RADIUS_FRAMES` [103-103] | `suppress_contact_flags` [1055], `assemble_contacts` [1274] | C |
| `PLAYER_PRESENT_MIN_FRAC` [252-252] | `_sticky_serve_setup_before` [702, 717] | S |
| `BODY_UNIT_HALF_WINDOW` [396-396] | `build_sticky_result` [1125] | X sticky-evidence bridge |
| `_SPAN_OPEN_CHOICES` [1373-1373] | `main` [1451, 1466] | B |

`_BST_X_ROOT` [80-82] and the `sticky_anchor`/`ClipContext`/`RawClip` imports [84-85] are sticky-builder support and belong with the X sticky-evidence bridge. The module logger `log` [87] is used by the B loaders and batch loop at lines 1392, 1407, 1488-1493, 1512, 1540, 1553, and 1571, so logger setup belongs with B.

`_rolling_mean` [126-139] is the shared local helper because S uses it at line 484 and C uses it at lines 933-934. `_nan_rolling_mean` [142-155] is only called by C at lines 938-939, so it can move with C even though it is physically in the prelude. `scale_thresholds` [106-123] is a threshold-resolution utility rather than a span, contact, or CLI definition, so it belongs with shared threshold support.

The imported configuration constants already have an external home in `.config` [47-59]. The span lane reads `END_REST_FRAMES`, `REST_SPEED`, `REST_WINDOW`, `START_MIN_FRAMES`, and `START_SPEED` at lines 479-507. The contact lane reads `SMOOTH_WINDOW` at line 927 and `PROXIMITY_MAX` at line 1036. The CLI reads `RALLY_SPANS_CSV` and `CONTACT_FRAMES_CSV` at lines 1455-1456 and constructs `BaseAnnotatorConfig` at line 1525. Those imported constants should remain in `.config`, not be copied into a split file.

### Types

The locally declared types `CourtGeo` [161-174], `ServeSetupInputs` [177-222], `ServeStartMode` [333-343], `ServeStartClose` [346-357], and `ServeStartOptions` [360-392] belong with S. `ServeStartOptions` is also part of the X façade contract at line 1286. `ContactCandidate` is imported at lines 62-70 and used by C at line 1264. `SmoothingMode` is imported at lines 62-70 and used by C at lines 916-918 and 1263. `StickyResult`, `Slot`, and the ankle and wrist keypoint constants are imported at lines 62-75 and are used by the X sticky builder at lines 1120-1225 and by S's setup-input builder at lines 296-330.

`SpanOpen` is the domain type used across all three split surfaces: S uses it at lines 556 and 751, C/X exposes it at line 1287, and B maps CLI strings to it at lines 1373 and 1466. It is already imported from `.types` at lines 62-70, so `.types` is the shared home after the split. `Stage8Thresholds` is shared by threshold support and S/C, as shown by its import at line 58 and uses in S at lines 461, 490, 521, and 750 and C at lines 916 and 965; B does not reference that type directly.

`FpsConstants` and `ReentryGuardVariant` are imported from `.fps_constants` and `.types` at lines 61-69. They are used by S's gap and span rules at lines 402-419 and 461-464, and by X's span façade at lines 1231-1233 and 1293-1296. They should remain in their existing shared modules.

## 5. Both-ends mask anchor check

### Stage 8 loader in `rally_segmentation.py`

The loader explicitly documents and constructs the dead-mask filename:

> `src/annotator/rally_segmentation.py:1397-1405`
> `def _load_replay_mask(mask_dir: Path | None, video_id: str) -> np.ndarray | None:`
> `"""Load \`<video_id>_dead_mask.npy\` from mask_dir if present, else None.`
> `mask_path = mask_dir / f'{video_id}_dead_mask.npy'`

The CLI calls this loader at line 1521 and passes the result as `raw_exclusion_mask` at lines 1522-1529 [rally_segmentation.py:1521-1529].

### Stage 9 writer in `replay_mask.py`

The replay-mask CLI writes the `_replay.npy` filename:

> `src/annotator/replay_mask.py:343-345`
> `args.out_dir.mkdir(parents=True, exist_ok=True)`
> `out_path = args.out_dir / f'{args.video_id}_replay.npy'`
> `np.save(out_path, mask)`

The module docstring states the same output convention at lines 3-6 [src/annotator/replay_mask.py:3-6].

### Stage 11 reader in `stage11_pairing.py`

Stage 11 reads `_replay.npy`, not `_dead_mask.npy`:

> `src/scraper/stage11_pairing.py:228-235`
> `def _load_replay_mask(masks_dir: Path, video_id: str) -> np.ndarray | None:`
> `"""Load a one-dimensional boolean \`<video_id>_replay.npy\`, or None if absent."""`
> `mask_path = masks_dir / f'{video_id}_replay.npy'`
> `replay_mask = np.load(mask_path)`

Its CLI invokes that reader at line 346 [src/scraper/stage11_pairing.py:346-351]. The verified filename chain is therefore `rally_segmentation.py` loader: `<id>_dead_mask.npy`; `replay_mask.py` CLI writer: `<id>_replay.npy`; `stage11_pairing.py` reader: `<id>_replay.npy` [rally_segmentation.py:1398-1405; replay_mask.py:343-345; stage11_pairing.py:228-235].

## Summary

The three-way split is not clean as three contiguous files because S, C, and B are separated by shared smoothing support, a dual-use sticky-evidence block, and the whole-stage `segment_video` façade [rally_segmentation.py:126-155, 1071-1225, 1283-1367]. The single heaviest coupling is `segment_video`, which applies the mask, calls span selection, and calls contact assembly at lines 1347-1365 while exposing the S-side `ServeStartOptions` type at line 1286 [rally_segmentation.py:1283-1367].
