# rtmlib migration: runbook

> Executable plan. Batches are ordered; each names files, change, gate, and
> dependency. Line numbers are point-in-time — re-grep and locate by symbol at
> execution. OUT-list is in `00_findings_and_scope.md §6`; do not touch it.
> Nothing is committed by the assistant; the user commits with the drafted
> messages after approval.

## Risk buckets

- **A — new isolated code** (adapter module, gate scripts). No existing behaviour
  touched; risk is self-contained and unit-testable on CPU.
- **B — backend swap on a live path** (`raw_extract`, `prepare_2d`). Behaviour
  *intentionally* changes (updated model); the invariant to preserve is the raw
  **schema/contract**, not the values. Gated by CPU schema/contract tests +
  the GPU parity handoff.
- **C — manifests/docs**. Mechanical; gated by ruff + pytest + doc-consistency.

## Gate ladder

- **Tier 0** (every batch): `ruff check` (pinned `ruff==0.15.12`, the version CI runs) + `pytest` + targeted re-grep of touched symbols. CI (`ci.yml`) enforces `ruff check .` and `pytest`; it does **not** run `ruff format --check`, and the repo is hand-formatted (10/11 `preparing_data` files fail `ruff format --check` on `main`), so new files match the surrounding hand style rather than ruff-format defaults.
- **Tier 1** (adapter + swaps): Tier 0 + the **adapter contract test** (CPU): the adapter emits `{keypoints (n,17,2) f32, bbox (n,4) f32, bbox_score (n,) f32, keypoint_scores (n,17) f32}` in COCO-17 order, with the N_max truncation, NaN padding, empty-frame guard, and dtypes matching the raw schema.
- **Tier 2 — CPU downstream byte-equality** (raw→clean invariance): feed a fixed set of committed mmpose raw arrays through the *unchanged* `apply_heuristic`(`current`/`sticky_anchor`) + `collate_npy`, before and after the branch's edits; assert byte-identical (`_failed` exact, `_pos`/`_joints` atol 1e-5). Proves the migration touched nothing downstream.
- **Tier 3 — GPU extraction parity** (HALT-AND-HANDOFF, Bourbaki): user runs the rtmlib extractor on `smoke50` (+ shards), then the parity gate compares vs the committed mmpose raw arrays: per-clip frame count, `ndet` distribution vs baseline, IoU-matched keypoint agreement, post-`sticky_anchor` failed-rate. **Never self-certified — different environment.**

## Batches

### Batch 0 — verification harness (Bucket A, no src edit)

Gate specs are authoritative in `03_verification.md` (G1–G10); this builds the
scripts. **Every script imports the shipped `raw_extract`/adapter, never the
scratchpad prototype** (adversarial finding H2).

- **Files (new, under `src/bst_x/validation_scripts/rtmlib_migration/`):**
  - `download_and_verify_models.py` — fetch the two ONNX; **assert** SHA256 vs a committed constant, exit non-zero on mismatch; vendor them to the pool (H3).
  - `gate_keypoint_value.py` — **G1**, the critical value gate: committed mmpose kp reference + IoU-matched pixel L2 (median ≤5px / p90 ≤12px) + anatomical order-sanity backstop.
  - `adapter_contract_test.py` — **G2**: schema/dtypes + synthetic 20-box (truncation), no-person (empty guard), partial-frame (NaN padding).
  - `gate_dtype_parity.py` — **G3**: `detect_players_2d` float64 parity.
  - `gate_cpu_determinism.py` — **G4**: adapter twice, `OMP_NUM_THREADS=1` pinned.
  - `gate_cpu_downstream_byteeq.py` — **G5**: dual-invocation precondition (fixed mmpose raw → unchanged downstream).
  - `gate_deployed_parity.py` — **G6**: sticky_anchor(rtmlib) vs committed clean (bbox-path; from the validated prototype).
  - `gate_gpu_parity.py`, `gate_cuda_selfvariance.py`, `phase_a_decision.py` — **G7–G9**, authored here, *run* on Bourbaki in Batches 2/5.
- **Gate:** CPU scripts run against the pool fixtures; G5 self-verifies 0-diff (capture twice on `main`, confirm identical); SHA-assert passes.
- **Depends on:** nothing (G1/G2/G4/G6 need the Batch-1 adapter to gate against — author the harness here, wire to the adapter in Batch 1).
- **Handoff:** none (CPU). G7–G9 run in the GPU handoffs.
- **Commit msg:** C0.

### Batch 1 — shared rtmlib adapter module (Bucket A)

- **File (new):** `src/bst_x/preparing_data/rtmlib_pose.py` — `RTMDetScored` (recovers detection score for the NMS-baked person ONNX), RGB-fixed `RTMPose` driver, and `extract_frame(frame_bgr) -> per-person dict/arrays` reproducing the mmpose per-person contract. Module hard-imports rtmlib; consumers import it lazily (mirrors the current lazy-mmpose pattern so `prepare_train_on_shuttleset` still imports without the dep).
- **Gate:** Tier 1 (adapter contract test) + Tier 0.
- **Depends on:** Batch 0.
- **Commit msg:** see C1.

### Batch 2 — migrate `raw_extract.py` to the adapter (Bucket B)

- **File:** `src/bst_x/preparing_data/raw_extract.py`. Replace the `MMPoseInferencer` import + `("human")` instantiations + `extract_raw_frame` field extraction with the adapter. Preserve exactly: 5-array schema, `N_max` top-by-`bbox_score` truncation, NaN padding, int8 `ndet` saved-last resume marker, `--inspect-result`/`--dry-run`. Drop `torch.cuda.empty_cache()` (onnxruntime-managed; keep `gc.collect()`), with an inline note.
- **Gate:** Tier 0 + Tier 1 (adapter path) locally. **Then Tier 3 HALT-AND-HANDOFF:** user runs `raw_extract` (rtmlib) on `smoke50` on Bourbaki; `gate_gpu_parity.py` vs the committed mmpose raw arrays.
- **Depends on:** Batch 1.
- **Commit msg:** C2.

### Batch 3 — migrate `detect_players_2d` / `prepare_2d` to the adapter (Bucket B)

- **File:** `prepare_train_on_shuttleset.py`. `prepare_2d_dataset_npy_from_raw_video` builds the adapter (lazy import) instead of `MMPoseInferencer("human")`; `detect_players_2d` consumes `keypoints`+`bbox` from the adapter's per-person output. `_order_two_on_court` and the failed-frame/append logic are untouched (OUT-list §7). Leave `detect_players_3d`/`prepare_3d`/`--use-3d-pose` exactly as-is (quarantined lazy mmpose). **dtype (G3, adversarial finding):** cast the adapter's keypoints/bbox to float64 on this 2D path so `np.array([...])` stays float64 as it was under mmpose (rtmlib returns float32 → else `normalize_joints`/projection drift at the atol boundary).
- **Gate:** Tier 0 + **G3** (dtype-parity) + **G1** (keypoint-value) on the 2D path + re-point `smoke_prepare_2d_bit_exact.py` to run on the rtmlib path (now CPU-capable); document that byte-exact-vs-old-mmpose is **not** expected (model changed) — the smoke now gates future rtmlib-path refactors (rtmlib-main vs rtmlib-branch).
- **Depends on:** Batch 1.
- **Commit msg:** C3.

### Batch 4 — dependency manifests + docs (Bucket C)

- **Files:** `requirements.txt` (+`rtmlib` pinned, `onnxruntime`), new `requirements-legacy-3d.txt` (the moved mmcv/mmdet/mmpose/mmengine pins, documented "only for `--use-3d-pose`"), `src/bst_x/preparing_data/requirements.txt` (rewrite to rtmlib extraction deps + point at the legacy file for 3D), `pyproject.toml` comment block. Docs: `mmpose_heuristic.md`, `data_pipeline_to_model_train.md`, `keypoints_schema.md`, `raw_extract.py`/`prepare_train` docstrings — swap "MMPose" wording for rtmlib where it describes the *extractor* (keep historical/investigation docs as-is).
- **Gate:** Tier 0 + doc-consistency (grep no stale "MMPoseInferencer" claims on the migrated paths) + `scripts/gen-requirements.sh --check` if applicable.
- **Depends on:** Batches 2–3.
- **Commit msg:** C4.

### Batch 5 — Phase-A parity run (HALT-AND-HANDOFF, Bourbaki)

- User runs rtmlib `raw_extract` on `smoke50` + 1–2 shards on Bourbaki (GPU or CPU); `gate_gpu_parity.py` + a CPU `apply_heuristic sticky_anchor` on the new raw → failed-rate vs baseline. **Decision gate** (thresholds O3). No src edit.

### Batch 6 — Phase-B full re-extract + retrain (HALT-AND-HANDOFF, if Phase A clears)

- Full 32,203-clip re-extract → `sticky_anchor` → collate → retrain BST-X → macro/min-F1 vs committed. No src edit; consumes the migrated code.

## Pre-approved commit messages (draft — approve before use; no `Co-Authored-By: Claude` trailer)

> Refreshed 2026-07-03 to match the as-built code (adversarial-review fold: G1's
> RGB counterfactual + L/R-mirror-robust tail, G9 nan-aware/per-clip checks, the
> --n-max default, and C4's actual file locations). The planning-doc suite
> (`docs/architecture_notes/rtmlib_migration/`) + the tidied worklog are a
> separate docs commit, not part of C0–C4. All five map to the current uncommitted
> tree; the tree carries Batches 0–4 mixed, so stage per-batch.

**C0**
```
Add rtmlib migration verification harness

Gate scripts under validation_scripts/rtmlib_migration/: a keypoint-value gate
(committed mmpose reference, pixel L2, a byte-exact RGB-feed counterfactual, and
an L/R-mirror-robust confident tail), a contract test with synthetic
truncation/empty/partial frames, dtype-parity and determinism checks, the
dual-invocation downstream byte-equality precondition, deployed-output parity
(with a directional failed-frame split), and the GPU parity / CUDA self-variance
/ Phase-A scripts for the Bourbaki handoff. Model download asserts SHA256 and
vendors the ONNX to the pool.
```

**C1**
```
Add rtmlib pose adapter reproducing the mmpose per-person contract

New rtmlib_pose.py drives RTMDet (person, with recovered detection score) plus
an RGB-corrected RTMPose-L body7 COCO-17 estimator, emitting per-person
keypoints/bbox/bbox_score/keypoint_scores in the exact schema the raw extract
and detect_players_2d consume. onnxruntime backend; no mmpose, numpy-2 clean.
Consumers import it lazily so the pipeline still imports without the dep.
```

**C2**
```
Extract raw keypoints via rtmlib instead of mmpose

raw_extract now runs the rtmlib adapter per frame; the five-array raw schema
(N_max=16 top-by-bbox_score, NaN padding, int8 ndet resume marker) is unchanged
so apply_heuristic and every downstream stage are untouched. Drops the torch
CUDA cache clear (onnxruntime manages its own memory) and sets the --n-max CLI
default to 16 to match the committed schema. Keypoints come from the updated
body7 RTMPose-L, so values differ from the old extract by design; gated by the
GPU parity check, not byte-equality.
```

**C3**
```
Route the 2D pose path through the rtmlib adapter

prepare_2d_dataset_npy_from_raw_video builds the rtmlib adapter in place of
MMPoseInferencer("human"). detect_players_2d now reads per-frame keypoints/bbox
from the adapter's arrays (cast to float64 so the projection chain keeps the
dtype it had under mmpose's np.array-of-lists); _order_two_on_court and the
failed-frame handling are untouched. detect_players_3d stays on its lazy mmpose
import, quarantined and unmodified. Adds --device (cuda default) for the 2D path.
```

**C4**
```
Migrate the pose-extraction dependencies from mmpose to rtmlib

Rewrite preparing_data/requirements.txt to the rtmlib extraction set (rtmlib,
onnxruntime, numpy 2 — the gate-validated pins) and move the mmcv/mmdet/mmpose/
mmengine stack to preparing_data/requirements-legacy-3d.txt, needed only for the
dormant --use-3d-pose path. Extraction still runs in its own dataset-build venv
but no longer needs the numpy<2 / source-built-mmcv isolation. Update the
pyproject comments and the rtmlib/torch lazy-import notes, swap the 2D-extractor
wording mmpose->rtmlib, and gitignore the G7/G8 gate JSON dumps.
```
(The top-level requirements.txt is uv.lock-generated and carries no mmpose, so it
is not touched here — the mmpose pins and the rtmlib deps both live in the
isolated preparing_data extraction venv.)

## Diagnostics (fold in as they surface)

- `PYTHONUNBUFFERED=1` for long redirected GPU runs (or `tee` shows nothing until end).
- rtmlib caches models to `~/.cache/rtmlib/hub/checkpoints/` (override `XDG_CACHE_HOME`); no SHA check by default — pin + verify in Batch 0.
- onnxruntime sets thread count = physical cores by default; rtmlib exposes no `SessionOptions`, so set `OMP_NUM_THREADS=1` in a gate's run command when it needs a fixed thread count (G4).
- For GPU: install `onnxruntime-gpu` (rtmlib pulls only CPU `onnxruntime`); confirm `CUDAExecutionProvider` present.
- Pool env: local `python3` lacks pandas; the CPU raw→clean gate needs a venv with numpy + pandas. (Originally cautioned pandas<3 for the codebase's inplace/iloc patterns; **G5 then verified byte-identical downstream on pandas 3.0.3** over 50 clips, so the extraction venv pins 3.0.3.)
