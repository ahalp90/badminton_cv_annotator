# D2 raw return (automated read-only check, 2026-08-02): blast radius of replacing src/shared mirrors with the bst_x versions

## 1. SUMMARY

- Court maths: **NEEDS CARE**: the pure maths is mirrored, but the shared file provides `REF_COURT_*` and `load_all_court_info`, while BST provides `build_all_court_info`, so a whole-file move needs a compatibility surface. (`src/shared/court.py:33-46,149-152`; `src/bst_x/pipeline/court_utils.py:11,170-180`)
- Player mapping: **SAFE MECHANICAL**: the mapping functions are equivalent; only the `ZH_TO_EN` import path differs. (`src/shared/player_mapping.py:21,28-114`; `src/bst_x/pipeline/player_mapping.py:15,22-129`)
- Flaw parsing and clip bounds: **NEEDS CARE**: parsing and temporal-bound logic match, but the clip helper clamps `start_f` to zero in BST and not in shared, and the modules use different data roots. (`src/shared/dataset.py:63-71,86-139,217-277`; `src/bst_x/pipeline/config.py:15-27,266-313`; `src/bst_x/pipeline/clip_generator.py:37-76,113-143`)
- Taxonomy registries: **HARD BLOCKER**: the dataclass APIs, registry names, class ordering and merge semantics differ, while deployed BRIC metadata uses the shared name and class list. (`src/shared/taxonomy.py:117-177,180-235`; `src/bst_x/pipeline/config.py:145-180,205-260`; `runtime/deployed/bric/20260518_013238_rgb_shuttle-tcn-outgoing_only_une_merge_v1_nosides_42/manifest.yaml:7-22`)

## 2. CONSUMER TABLE

The table lists direct imports or calls. Pair 3 is symbol-scoped because the BST implementation is split across `config.py` and `clip_generator.py`.

### Court maths

| Surface | Consumers |
|---|---|
| `src/shared/court.py` | `src/bric/dataset.py:17-22`; `src/bric/perception/players.py:42`; `src/bric/preprocessing/preprocess_videos.py:71-76`; `src/bric/diagnostics/debug_court_bias.py:27-33`; `src/bric/diagnostics/validate_court_positions.py:30-35`; `src/bric/diagnostics/evaluate_players.py:30-35`; `src/bric/smoke_test.py:198-201`; `src/annotator/calibration/fixtures.py:14`; `src/annotator/calibration/gt_scoring.py:26`; `src/annotator/point_winner.py:37-44`; `src/annotator/court_evidence.py:19`; `src/api/bric_inference.py:40-46`. |
| `src/bst_x/pipeline/court_utils.py` | `src/bst_x/preparing_data/prepare_train_on_shuttleset.py:64-66`; `src/bst_x/preparing_data/apply_heuristic.py:34-35`; `src/bst_x/preparing_data/heuristics/sticky_anchor.py:49-50`; indirect re-export consumer `src/bst_x/preparing_data/heuristics/current.py:59-62`; `src/bst_x/validation_scripts/mmpose_heuristic_investigation/diagnose_top_k_capture.py:35`; `render_sticky_anchor_overlays.py:50`; `render_detection_overlays.py:52`; `src/bst_x/validation_scripts/render_anchor_and_dets_overlay.py:73`; `src/bst_x/validation_scripts/refactoring/smoke_prepare_2d_bit_exact.py:132`; `src/bst_x/validation_scripts/rtmlib_migration/_common.py:196`; `gate_dtype_parity.py:47`. |

### Player mapping

| Surface | Consumers |
|---|---|
| `src/shared/player_mapping.py` | `scripts/build_shots_master.py:63`; `tests/test_player_mapping.py:24-28`. |
| `src/bst_x/pipeline/player_mapping.py` | `src/bst_x/pipeline/clip_generator.py:25,271`. |

### Flaw parsing and clip bounds

| Surface | Consumers |
|---|---|
| `src/shared/dataset.py` | `src/bric/dataset.py:23`; `src/bric/preprocessing/preprocess_videos.py:77`; `src/bric/preprocessing/slice_rallies.py:49`; `src/bric/diagnostics/debug_court_bias.py:34`; `validate_court_positions.py:36`; `evaluate_players.py:36`; `src/api/bric_inference.py:39`; `scripts/build_shots_master.py:52-62`; `tests/test_player_mapping.py:23`. |
| BST helpers | `parse_flaw_records` is invoked by its module at `src/bst_x/pipeline/config.py:300`; `compute_temporal_bounds` and `_compute_clip_bounds` are invoked inside `src/bst_x/pipeline/clip_generator.py:197,282`; the containing clip module is imported by `src/bst_x/pipeline/build_dataset.py:30`. |

### Taxonomy

| Surface | Consumers |
|---|---|
| `src/shared/taxonomy.py` | `src/shared/player_mapping.py:21`; `src/bric/dataset.py:24-28`; `src/bric/network.py:13`; `src/bric/train.py:33`; `src/bric/smoke_test.py:191-195`; `src/api/bric_inference.py:38`; `scripts/build_shots_master.py:64`; `tests/test_network.py:11`; `tests/test_player_mapping.py:29`. |
| BST `Taxonomy/TAXONOMIES` API | `src/bst_x/pipeline/build_dataset.py:23-27,41-42,284-297`; `src/bst_x/pipeline/clip_generator.py:18-23,29,302-304`; `src/bst_x/pipeline/verify.py:14-22,94-105`; `src/bst_x/pipeline/data_access.py:96-103,248-271,318-325,509-516`; `src/bst_x/bst_x_train.py:39-45,1321`; `src/bst_x/bst_x_infer.py:35-40,201-225`; `src/bst_x/bst_x_common.py:18,169-216`; `src/bst_x/bst_x_reporting.py:15,146-155`; `src/bst_x/model/bst.py:431-432`; `src/bst_x/preparing_data/prepare_train_on_shuttleset.py:50-62,908-918`; validation consumers at `find_busted_clips.py:58,114-136`, `zeroed_frames_class_audit.py:92,100-127`, `fail_rate_per_class.py:41,112-116`, `validate_zeroed_frames.py:55,1256-1291`, `smoke_b1_validate_gpu.py:37-39`, `smoke_infer_bit_exact.py:64,82`, and `smoke_b7_seeded_train.py:54,116`; tests at `tests/test_integration.py:56,93`, `test_data_access.py:19-24,90-177`, `test_taxonomy.py:35-50,80-87,142-221`, `test_train_surface.py:35-43`, `test_inference_smoke.py:26,59-64`, and `test_remote_preflight.py:44,83-184`. |

Relevant transitive entry points are `src/api/main.py:273-282` for live BRIC inference and annotator tests such as `tests/test_court_evidence.py:8-11`, `tests/test_point_winner.py:11`, `tests/test_annotator_run_video.py:8-10`, and `tests/test_annotator_fixtures.py:12,25`.

## 3. DIVERGENCE BLAST RADIUS

### Court API surface

The projection and normalisation functions are mirrored, but shared consumers require `REF_COURT_M`, `REF_COURT_CORNERS_M`, and `load_all_court_info`. (`src/shared/court.py:33-46,85-98,149-152`)

BST preparation code requires `build_all_court_info`, which keys results from the resolution dataframe rather than every homography row. (`src/bst_x/pipeline/court_utils.py:170-180`)

A whole-file replacement therefore causes import failures in BRIC and annotator unless the moved module preserves the union of both APIs. No model or wire-format contract is involved. (`src/bric/dataset.py:17-23`; `src/annotator/calibration/gt_scoring.py:26`; `src/bst_x/preparing_data/prepare_train_on_shuttleset.py:64-66`)

### Player mapping

The function bodies and outputs are equivalent. The only pipeline-local dependency is `ZH_TO_EN`: BST imports it from `pipeline.config`, while shared imports it from `shared.taxonomy`. (`src/shared/player_mapping.py:21,28-43,65-114`; `src/bst_x/pipeline/player_mapping.py:15,22-40,70-129`)

The impact is limited to import rewrites or a compatibility wrapper. The resulting player and stroke columns feed `shots_master.csv`, but the function has no stored taxonomy-name contract. (`scripts/build_shots_master.py:111-121`)

### Flaw parsing and clip bounds

The parser and temporal-bound calculations are mirrored, but their default paths differ. Shared points at `training/data/shuttleset/annotations`, while BST points at `data/shuttleset`. (`src/shared/dataset.py:60-71,86-117`; `src/bst_x/pipeline/config.py:15-27,266-293`)

BST clamps the clip start to zero; shared does not. (`src/bst_x/pipeline/clip_generator.py:135-143`; `src/shared/dataset.py:272-277`)

The API stores the per-stroke bounds and separately clamps the rally union start, so early-frame behaviour can differ without necessarily preventing decoding. (`src/api/bric_inference.py:266-277`)

The shot-building script writes the computed bounds as `shuttle_start_f` and `shuttle_end_f`, so regenerated metadata can differ at early frames. (`scripts/build_shots_master.py:115-121`)

A whole `shared.dataset` replacement would also remove shared-only path and split exports used by BRIC, including `HOMOGRAPHY_CSV_PATH`, `VIDEO_METADATA_PATH`, `SPLITS_V2_PATH`, and `SPLITS_BST_BASELINE`. (`src/shared/dataset.py:63-71,146-157`; `src/bric/dataset.py:23`)

### Taxonomy registries

The dataclass interfaces are incompatible. Shared exposes `base_types`, `standalone_types`, `unknown_first`, `class_list()`, and `trainable_class_list()`, while BST exposes explicit `classes`, `has_sides`, and `excluded_base_stroke_types`. (`src/shared/taxonomy.py:121-177`; `src/bst_x/pipeline/config.py:160-180`)

BRIC directly uses the shared-only fields and methods. (`src/bric/dataset.py:78-89,125-140`; `src/bric/network.py:242-245`; `src/bric/train.py:508`)

The 25-class registries also differ semantically: shared maps `driven_flight` to `unknown`, while BST maps it to `drive`. (`src/shared/taxonomy.py:63-70,180-186`; `src/bst_x/pipeline/config.py:91-98,205-211`)

Shared’s merged 25-class list places `unknown` first; BST’s sided class builder appends `unknown`. (`src/shared/taxonomy.py:139-162,180-186`; `src/bst_x/pipeline/config.py:182-196,205-210`)

The 14-class no-sides lists are both 14-class lists, but they still use different registry names and object APIs: `une_merge_v1_nosides` versus `une_v1_14`. (`src/shared/taxonomy.py:98-103,199-205,229-235`; `src/bst_x/pipeline/config.py:77-82,229-235,254-260`)

## 4. STORED-NAME CHECK

| Artefact or interface | Finding |
|---|---|
| BRIC checkpoints | The BRIC checkpoint writer stores both `taxonomy` and `classes`. (`src/bric/train.py:582-598`) The live API loads only `model_state_dict` from the checkpoint, but constructs the model from the manifest taxonomy and class count. (`src/api/bric_inference.py:170-192`; `src/bric/network.py:242-245,289`) |
| Deployed BRIC manifest | The deployed manifest stores `une_merge_v1_nosides`, its ordered 14-class list, and `num_classes: 14`. (`runtime/deployed/bric/20260518_013238_rgb_shuttle-tcn-outgoing_only_une_merge_v1_nosides_42/manifest.yaml:7-22,70`) |
| BRIC label decoding | The live API indexes `_class_list` for predicted and top-k labels, so changing the stored order changes response labels even when logits still load. (`src/api/bric_inference.py:180-192,530-546`) |
| API registry | The registry module does not import the taxonomy module; it reads manifest class lists and returns the registry’s literal `taxonomy` field. (`src/api/registry.py:17,141-156,198-228`) Therefore a shared-registry swap alone does not break the Tier 1 registry endpoint. Changing the manifest or registry values would change the API’s display contract. (`docs/models_registry.yaml:5-14,127-145`) |
| Stub API responses | The non-live inference path decodes labels from sidecar `class_list` or legacy `active_class_list`, not from a taxonomy registry. (`src/api/inference.py:33-50,130-170`) |
| BST training and inference artefacts | BST manifests store `classes`; `taxonomy.name` drives collated-directory naming and inference lookup. (`src/bst_x/bst_x_train.py:1233-1253,1319-1323`; `src/bst_x/bst_x_infer.py:201-225`) Replacing BST registry names with shared names would break lookup for names such as `bst_25` and `une_v1_14`, and could redirect paths to different class spaces. (`src/bst_x/pipeline/config.py:205-260`; `docs/models_registry.yaml:25-125`) |

The taxonomy swap is therefore a stored-data and label-contract risk, not just a mechanical import rename.

## 5. REVERSE IMPORTS

- `court_utils.py` imports `HOMOGRAPHY_RESOLUTION` from `pipeline.config`. (`src/bst_x/pipeline/court_utils.py:11`) The shared module already defines that value locally. (`src/shared/court.py:28-33`)
- `player_mapping.py` imports `ZH_TO_EN` from `pipeline.config`. (`src/bst_x/pipeline/player_mapping.py:15`) The shared taxonomy already owns the same mapping. (`src/shared/taxonomy.py:32-57`)
- `clip_generator.py` imports pipeline paths, stroke lists, curation sets, taxonomy objects, and `collect_shots` from `pipeline.player_mapping`. (`src/bst_x/pipeline/clip_generator.py:14-29`) Moving only the three pure helper functions avoids dragging in MoviePy, pipeline paths, and player collection.
- `parse_flaw_records` defaults to `config.FLAW_RECORDS_PATH` and runs at config import time. (`src/bst_x/pipeline/config.py:26,266-313`) A shared version needs an explicit path or a path adapter because shared uses a different annotation root. (`src/shared/dataset.py:60-71,120-139`)
- The BST taxonomy callers depend on the BST-specific fields and helpers, including `taxonomy_lookup` and `derive_class_index`. (`src/bst_x/pipeline/config.py:338-380`) Moving only the registry data requires either adapters exposing those fields or callsite changes across the BST consumers listed above.

## 6. MINIMAL ALTERNATIVES

- **Option, court:** keep `shared.court` as the compatibility facade, move only the common maths, retain `REF_COURT_*` and `load_all_court_info`, and expose `build_all_court_info` through the BST-facing path. (`src/shared/court.py:33-46,149-152`; `src/bst_x/pipeline/court_utils.py:170-180`)
- **Option, player mapping:** make the shared implementation authoritative, change its mapping import to the shared taxonomy, and leave `pipeline.player_mapping` as a thin compatibility wrapper. (`src/shared/player_mapping.py:21,28-114`; `src/bst_x/pipeline/player_mapping.py:15,22-129`)
- **Option, flaw and clip bounds:** share the parser and temporal helper with explicit data paths, but retain separate public clip-bound wrappers so BST keeps the zero clamp and shared callers retain their current contract. (`src/shared/dataset.py:86-117,217-277`; `src/bst_x/pipeline/clip_generator.py:113-143`)
- **Option, taxonomy:** keep both registry name sets and both compatibility APIs in one implementation file, preserving the stored names and exact class lists for BRIC and BST. (`src/shared/taxonomy.py:117-177,229-235`; `src/bst_x/pipeline/config.py:145-180,254-260`)

## 7. NOT CHECKED

- `.pt`, `.npz`, and `.npy` payloads were not opened. The checkpoint and NPZ schemas were checked from source writers only. (`src/bric/train.py:582-598`; `src/bst_x/bst_x_common.py:206-216`)
- `experiments/`, `data/`, `scripts/archive/`, and `docs/**/*.py` were excluded as requested.
- Ruff, whole-project Pyrefly, and pytest were not run because this was a read-only dependency audit.
