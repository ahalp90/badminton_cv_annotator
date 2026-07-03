# mmpose → rtmlib migration: findings & scope

> **Status (2026-07-03):** Adapter (`rtmlib_pose.py`), the CPU + GPU gate harness
> (`validation_scripts/rtmlib_migration/`), and the `raw_extract.py` /
> `detect_players_2d` migrations are written in the working tree and green on the
> CPU gates (G1–G6, raw-schema); the 2026-07-02 adversarial review (Batches 0–3)
> is folded (see `04_adversarial_review.md` + the worklog). The tree is
> **uncommitted**, staged for commits C0–C3 per `01_runbook.md`. Pending: the GPU
> handoff (G7 self-variance → G8 parity → G9 Phase-A) on Bourbaki, then Batch 4
> (deps manifests + docs). Line numbers are a read-time snapshot; locate by symbol.

## Goal

Replace the skeleton keypoint extraction backend from the pinned, source-built,
numpy<2, separate-venv OpenMMLab stack (`mmpose 1.3.2` + `mmcv 2.1.0` + `mmdet`
+ `mmengine`) with **rtmlib** (`github.com/Tau-J/rtmlib`, onnxruntime-based,
numpy-2 compatible), running the same model family (RTMDet-nano person detector
+ RTMPose-L COCO-17, updated `body7` weights), with byte-identical downstream
behaviour and verified extraction parity.

## TL;DR verdict

- **Small, well-bounded surface.** Only two files import mmpose:
  `raw_extract.py` (production extractor) and `prepare_train_on_shuttleset.py`
  (legacy `detect_players_2d` + dormant `detect_players_3d`, lazy imports).
- **The whole API / web-app / real-time inference path is mmpose-free already**
  (BRIC = YOLO11 + TrackNetV3; BST-X "live" = precomputed tensors). Out of scope.
- **Downstream is model-agnostic**: it depends only on the COCO-17 schema +
  bbox-diagonal normalisation, never on values being "from mmpose". So the swap
  is a genuine drop-in *if* the adapter reproduces the per-person contract.
- **Migrate, do not delete.** Under usage uncertainty, migrating the 2D paths is
  zero-regret (preserves any manual/external use *and* removes mmpose). 3D is
  quarantined (left byte-identical on an optional lazy mmpose import), not ported
  (unverifiable) and not deleted (the user's caution).
- **Model is updated**, so raw keypoints will not byte-match the old extract.
  Acceptance is staged: CPU downstream byte-identity + GPU extraction parity →
  decision gate → full re-extract + retrain + F1 compare.

## 1. Scope

### In scope (migrate to a shared rtmlib adapter)

| Target | File | Why |
|--------|------|-----|
| Production raw extractor | `src/bst_x/preparing_data/raw_extract.py` | Builds the deployed `ShuttleSet_keypoints_raw`; the real integration point. Consumes all four person fields. |
| Legacy 2D live path | `src/bst_x/preparing_data/prepare_train_on_shuttleset.py` `detect_players_2d` + `prepare_2d_dataset_npy_from_raw_video` (`:160`, `:429`) | Documented Step-2 CLI + the `smoke_prepare_2d_bit_exact` gate target. Migrating makes that gate CPU-runnable. |
| Dependency manifests | `requirements.txt`, `src/bst_x/preparing_data/requirements.txt`, `pyproject.toml` comments | Add rtmlib + onnxruntime; move the mmpose pins to an optional legacy file. |
| Gates to re-point | `smoke_prepare_2d_bit_exact.py`, `failsafe_bst_mmpose_zeroing_check_equivalence.py` | Keep working; add the new parity gate. |

### Quarantined (preserve untouched, do NOT port, do NOT delete)

- `detect_players_3d` + `prepare_3d_dataset_npy_from_raw_video`
  (`prepare_train_on_shuttleset.py:224`, `:469`) and the `--use-3d-pose` flag
  (`:973`). Doubly dormant: reachable only via `--use-3d-pose` (never set in any
  deployed config), no gate, no 3D data artifact anywhere on the pool. Its 3D
  lifting uses a *second, different* model (`MMPoseInferencer(pose3d="human3d")`)
  that rtmlib would serve via `RTMPose3d` — a different output shape with no way
  to verify the port. Left exactly as-is (already a lazy import); mmpose becomes
  an **optional** install needed only to run this path.

### Out of scope (mmpose-free already; do not touch)

- The entire `src/api/` layer, BRIC (`bric_inference.py`, `perception/players.py`
  → ultralytics YOLO), BST-X live inference (precomputed tensors), TrackNetV3
  shuttle detection, all downstream heuristics / collation / training / data
  access. None import mmpose.

### Delete nothing

The only behavioural micro-change: drop `torch.cuda.empty_cache()` from
`raw_extract` (irrelevant once onnxruntime replaces torch for extraction), done
explicitly with a note, keeping a harmless `gc.collect()`.

## 2. The mmpose touch-point map (verified)

Only two files import mmpose:

- `raw_extract.py:30` — `from mmpose.apis import MMPoseInferencer` (hard, top-level).
- `prepare_train_on_shuttleset.py:62` (TYPE_CHECKING), `:254`, `:452`, `:486`
  (lazy inside the three pose functions).

Model instantiations: `raw_extract.py:256`,`:262`; `prepare_train...:453` (2D),
`:487` (3D's 2D model), `:255` (`pose3d="human3d"`, per-clip).

**Result-dict fields consumed** (the rtmlib output contract to reproduce):

| Consumer | keypoints | bbox | bbox_score | keypoint_scores |
|----------|:---:|:---:|:---:|:---:|
| `raw_extract.extract_raw_frame` (`:77-81`) | ✓ | ✓ (`bbox[0]`) | ✓ | ✓ |
| `detect_players_2d` (`:186`,`:201`) | ✓ | ✓ | – | – |
| `detect_players_3d` (`:261`,`:268`) | ✓ (2D+3D) | – | – | – |

Downstream consumption of the raw arrays: `sticky_anchor` reads
`ndet`/`scores`(=bbox_score)/`bboxes`/`kps`; `current` reads `ndet`/`kps`/`bboxes`.
**`_raw_kp_scores` (keypoint_scores) is loaded into `RawClip` but read by no
heuristic** — schema-only past extraction (verified against
`heuristics/sticky_anchor.py:169-181` and `current.py:61-98`). It must still be
written to keep `apply_heuristic._load_raw_clip` (`:114`) intact.

## 3. Production vs legacy (evidence chain)

The **active** pipeline (per `docs/architecture_notes/mmpose_heuristic/mmpose_heuristic.md:39-54`,
the operational reference) is two-stage decoupled:

1. `raw_extract.py` (GPU) → `ShuttleSet_keypoints_raw/` (32,203 stems).
2. `apply_heuristic.py --heuristic sticky_anchor` (CPU) →
   `ShuttleSet_keypoints_clean_sticky_anchor/`, which `$BST_X_MMPOSE_NPY_DIR`
   points at (`.env.example` HPC block; every training manifest attributes the
   data to sticky_anchor).

`detect_players_2d` is the **legacy** path: still wired into
`prepare_train_on_shuttleset.main()` Step 2 (`:1184`) and the
`smoke_prepare_2d_bit_exact` gate, and documented as Step 2 in
`data_pipeline_to_model_train.md:156`, but not used to build the deployed
dataset (Step 2 is skipped in the sticky_anchor workflow). Its logic is the
reference twin of the `current` heuristic. `detect_players_3d` is doubly dormant
(see §1). **Cannot fully rule out manual/external use of Step 2** → migrate,
don't delete.

## 4. rtmlib feasibility + two critical findings

rtmlib repo `Tau-J/rtmlib` (Apache-2.0, PyPI 0.0.15 2026-02, repo `main` newer —
**pin the version**). onnxruntime backend; CPU + CUDA execution providers.

- **COCO-17 with a hash-identical detector is available.** The rtmlib-loadable
  `rtmdet-nano-person` ONNX (`...-05d8511e.zip`) is byte-identical to the
  detector inside `MMPoseInferencer("human")`; `rtmpose-l_simcc-body7-256x192`
  is exactly "RTMPose-L COCO-17, updated (7-dataset) weights".

- **Finding 1 — solutions return only `(keypoints, scores)`; no bbox / no
  detection score.** Low-level `RTMDet` returns boxes but *discards* the
  detection score in `postprocess`. `raw_extract` needs `bbox` and `bbox_score`,
  and `sticky_anchor` consumes `bbox_score`. → The adapter drives `RTMDet` +
  `RTMPose` at the low level and recovers the detection score (subclass
  `RTMDet.postprocess` to also return the kept scores — it already computes them).

- **Finding 2 — undocumented BGR/RGB bug in rtmlib's RTMPose preprocessing.** It
  uses an RGB-order mean but never converts BGR→RGB, effectively swapping R/B vs
  training. Left as-is, outputs would not be comparable to mmpose. Fix: feed
  RTMPose an RGB crop while feeding RTMDet BGR (warpAffine is colour-agnostic).
  **Validate empirically** against a few mmpose reference frames before trusting.

- **Empty-frame trap:** rtmlib's RTMPose falls back to a full-image bbox on zero
  detections → one spurious full-frame "person". The adapter must guard: zero
  detections ⇒ `ndet=0`, all-NaN frame (matching mmpose).

- **Determinism:** onnxruntime CPU EP is deterministic run-to-run for fixed
  input+model+thread-count; CUDA EP is not, and CPU≠CUDA bit-for-bit. Only
  matters for inference-running gates, not the pure-numpy downstream CPU gate.

## 5. Model + dependency decisions (locked)

- **Pose:** `rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504`
  (RTMPose-L, COCO-17, 256×192).
- **Detector:** `rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e` (320×320,
  NMS baked in).
- **Deps:** add `rtmlib` (pinned) + `onnxruntime` (CPU) to the main requirements;
  document `onnxruntime-gpu` for the GPU box. Pin the two ONNX URLs + verify
  their SHA256 (rtmlib does no hash check by default). rtmlib is numpy-2 clean;
  extraction can now run in the **main env — no separate venv**.
- **Legacy 3D:** move `mmengine/mmdet/mmpose/mmcv` pins to
  `requirements-legacy-3d.txt`, documented as "only needed for `--use-3d-pose`".

## 6. OUT-list (hard do-not-touch, with reasons)

1. `heuristics/current.py`, `heuristics/sticky_anchor.py`, `heuristics/base.py` —
   byte-identity of the deployed extract depends on them; the adapter feeds them, never edits them.
2. `apply_heuristic.py`, `collate_npy` (in `prepare_train_on_shuttleset.py`),
   `pipeline/court_utils.py`, `normalize_joints`/`normalize_shuttlecock` — downstream contract, unchanged.
3. The committed pool dirs `ShuttleSet_keypoints_raw` and
   `..._clean_sticky_anchor` (+ `_unknown` siblings) — never written to; they are the baselines.
4. The raw 5-array schema: shapes, `float32`/`int8` dtypes, **NaN** padding,
   `N_max=16`, top-`N_max`-by-`bbox_score` truncation, COCO-17 joint order,
   `_raw_ndet` saved-last resume marker, and the `_raw_kp_scores` slot (written though unused).
5. `detect_players_3d` / `prepare_3d` / `--use-3d-pose` — quarantined, left byte-identical.
6. Vendored TrackNetV3 (`src/bric/perception/_vendor/`), BRIC/YOLO, the whole `src/api/` layer.
7. The `_order_two_on_court` helper and the `current`↔`detect_players_2d`
   equivalence contract (B5 invariants in `function_invariants/detect_players.md`): strict-`>` y-flip, `<2` short-circuit before `check_pos_in_court`, `dtype=float` (float64) appends.

## 7. Data & fixtures inventory (already local — no SSH needed for CPU gate)

`/srv/mergerfs/main_pool/320_cosc594_data-bourbaki/`:

| Artifact | Content | Use |
|----------|---------|-----|
| `ShuttleSet/clips/{split}/{class}/*.mp4` | 33,481 clips | GPU extraction inputs |
| `ShuttleSet_keypoints_raw/` | 32,203 clips × 5 raw arrays (N_max=16) | CPU byte-equality fixtures; mmpose parity baseline |
| `ShuttleSet_keypoints_clean_sticky_anchor/` | 32,203 clips × 3 clean arrays | deployed baseline (never overwrite) |
| `ShuttleSet_keypoints_raw_provenance/` | `_smoke50.txt` (50 stems), `stems_to_extract.txt` (30,487), `shard_00/01` | ready-made smoke + shard sets |
| `ShuttleSet/shuttle_csv`, `shuttle_npy_flat` | TrackNetV3 (not mmpose) | collation input (unchanged) |
| `ShuttleSet_data_{bst_25,une_v1_14,...}` | collated training datasets | retrain baselines |

In-repo CSVs the CPU path needs all exist: `data/shuttleset/set/homography.csv`,
`data/shuttleset/my_raw_video_resolution.csv`, `notebooks/clips_master.csv`.

Local host: CPU-only (6 cores, 46 GB, numpy 2.2.4). Runs the raw→clean CPU gate
and rtmlib CPU inference. GPU extraction + retrain run remotely (Bourbaki); user
executes those (no SSH from here).

## 8. Acceptance criteria — staged (parity → retrain gate)

**Phase A (parity, cheap):** rtmlib extract on `smoke50` + 1–2 shards; compare
against the committed mmpose raw arrays — per-clip frame counts, `ndet`
distribution vs the captured baseline, keypoint agreement on matched
high-confidence detections, and post-`sticky_anchor` failed-frame rate. CPU
downstream byte-identity holds throughout (fixed inputs). **Decision gate.**

**Phase B (full, if Phase A clears):** full 32,203-clip re-extract → re-run
`sticky_anchor` → re-collate → retrain BST-X → compare macro-F1 / min-class-F1
against the committed baseline. Mirrors the prior mmpose-heuristic Phase-1→Phase-2.

## 9. Decisions log

| # | Question | Decision | Date |
|---|----------|----------|------|
| D1 | Legacy `detect_players_2d/3d` scope | Migrate 2D (raw_extract + detect_players_2d) to shared adapter; quarantine 3D untouched; delete nothing | 2026-07-01 |
| D2 | Acceptance bar | Staged: parity → retrain decision gate | 2026-07-01 |
| D3 | Pose model | RTMPose-L body7 @ 256×192; detector rtmdet-nano-person | 2026-07-01 |
| D4 | Doc location | In-repo, `docs/architecture_notes/rtmlib_migration/` | 2026-07-01 |
| D5 | Branch name | `migrate-mmpose-to-rtmlib` | 2026-07-01 |

### Open (to confirm before/at execution)

- O1: RESOLVED — pin `rtmlib==0.0.15` (prototype ran on it); re-diff on install.
- O2: RESOLVED — `onnxruntime` (CPU) goes in main `requirements.txt`; extraction
  is now venv-free. `onnxruntime-gpu` documented for the GPU box.
- O3: RESOLVED — concrete Phase-A thresholds in `03_verification.md` G9 (frame
  count 100% exact; failed-rate |Δ|≤2pp agg / ≤5pp clip; keypoint L2 median≤5px
  / p90≤12px), all set above the G7 CUDA self-variance floor.
- O4: whether to also expose an `--input-size 384x288` option for a later
  accuracy pass (not now; 256×192 is the baseline).

**Adversarial plan-review (round 1)** — see `04_adversarial_review.md`. Parity
claim CONFIRMED across 20 diverse+hard clips; gate review found real holes (no
self-certifying keypoint-value gate; repro unpinned; float32/float64 dtype drift
on the `detect_players_2d` path), all folded into `03_verification.md`. One
code-level carry into execution: the **G3 dtype-parity** fix.
