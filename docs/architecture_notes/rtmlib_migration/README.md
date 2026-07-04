# rtmlib migration and detector restoration

The 2D pose-extraction path (`raw_extract`, `detect_players_2d`) runs on
rtmlib (onnxruntime, numpy-2 clean, no source builds) instead of the pinned
mmpose/mmcv/mmdet/mmengine stack. Models: **RTMDet-M person @640x640**
(checkpoint 235e8209, the detector `MMPoseInferencer("human")` actually
resolved at the pinned mmpose 1.3.2) with a strict `DET_SCORE_THR = 0.3` cut,
and **RTMPose-L body7 COCO-17 @256x192** (a deliberate upgrade over the mmpose
alias's RTMPose-M body7). The five-array raw schema, `sticky_anchor` player
selection, and collation are byte-preserved; keypoint values are gated by
parity against the committed mmpose extraction. The dormant 3D path stays on
mmpose behind lazy imports (`requirements-legacy-3d.txt`).

## The story, short

1. **Migration (2026-07-01..03).** Adapter (`preparing_data/rtmlib_pose.py`)
   reproduces the mmpose per-person contract over onnxruntime, with two rtmlib
   quirks fixed (detection-score recovery; a BGR/RGB normalisation bug). A
   staged gate ladder (CPU byte-equality and parity gates, then GPU
   self-variance/parity/decision on the HPC box) validated it against the
   committed mmpose extraction. Initial detector: RTMDet-nano@320, believed
   identical to mmpose's.
2. **The gates kept complaining.** A one-directional frame-loss bias (rtmlib
   dropping players mmpose kept) forced a keep-threshold recalibration from
   0.3 to 0.15 to pass Phase-A.
3. **Root cause (2026-07-04).** Primary sources (mmpose 1.3.2
   `default_det_models`) showed the pre-migration detector was RTMDet-M@640,
   not nano@320 — the identity claim was wrong, and the 0.15 crutch was
   compensation for an unnoticed model downgrade.
4. **Restoration.** Detector swapped to the RTMDet-M ONNX (same mmdeploy
   export batch as the nano, drop-in), threshold returned to mmpose's 0.3,
   SHA re-pinned, docs corrected, full ladder re-run. Verdict: GO with no
   residuals. Details and receipts: `07_detector_restoration.md`.

Authoritative 200-clip comparison vs the committed mmpose extraction (same
stems all three runs):

| run | verdict | rtmlib-only : mmpose-only failed frames | dropped players | kp med/p90 |
|---|---|---|---|---|
| nano @0.3 | NO-GO | 50:7 | 5 clips | 1.74 / 4.58px |
| nano @0.15 | GO + accepted residual | 15:20 | 0 | 1.93 / 4.21px |
| **M @0.3 (shipped)** | **GO, no residuals** | **0:3** | **0** | **1.72 / 3.93px** |

## Benchmarks (A100; `bench_detector_pose_configs.py`)

Pose cost scales with crops kept, so the nano+0.15 combination admits ~40
crowd boxes/frame and pays for each:

| config | total ms/frame | fps | kp px vs deployed |
|---|---|---|---|
| nano + L256 @0.15 | 243 | 4.1 | 1.96 |
| nano + L256 @0.30 (lossy tail) | 45 | 22.1 | 2.00 |
| **M + L256 @0.30 (shipped)** | **81** | **12.3** | **1.62** |
| M + L384 @0.30 | 109 | 9.2 | 1.84 |

The restoration is also the faster production config: ~3x quicker than the
nano@0.15 setup that would otherwise have shipped.

## Reproducing the verification

- Models: `validation_scripts/rtmlib_migration/download_and_verify_models.py`
  downloads, SHA-verifies against committed pins, and vendors both ONNX files.
- CPU gates (dev box, from the repo root, `PYTHONPATH=src/bst_x:src`, a venv
  per `preparing_data/requirements.txt`): `gate_raw_schema`,
  `adapter_contract_test`, `gate_cpu_determinism` (OMP_NUM_THREADS=1),
  `gate_dtype_parity`, `gate_keypoint_value` (G1), `gate_cpu_downstream_byteeq`
  (G5), `gate_deployed_parity` (G6).
- GPU gates (CUDA box; env recipe in `preparing_data/requirements.txt`):
  `gate_cuda_selfvariance` (G7) then `gate_gpu_parity` (G8) then
  `phase_a_decision` (G9); paths/knobs via `RTMLIB_GATE_*` env vars documented
  in each script's docstring.
- Timed config benchmark: `bench_detector_pose_configs.py` (CPU or
  `RTMLIB_GATE_DEVICE=cuda`).

## History

The nano-era planning and decision docs (findings/scope, runbook, adapter
design, verification plan, adversarial review, GPU handoff, and the superseded
Phase-A decision) were retired from this directory once the restoration
landed; they remain in git history at this file's parent commits, recorded
with correction banners. `07_detector_restoration.md` is the surviving
decision record and carries the primary-source receipts.
