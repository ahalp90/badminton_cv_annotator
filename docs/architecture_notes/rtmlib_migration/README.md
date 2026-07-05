# rtmlib pose extraction

The 2D pose-extraction path (`raw_extract`, `detect_players_2d`) runs on
rtmlib (onnxruntime, numpy-2 clean, no source builds) instead of the pinned
mmpose/mmcv/mmdet/mmengine stack it replaced.

Models:

- **Detector: RTMDet-M person @640x640**, the checkpoint mmpose 1.3.2's
  `MMPoseInferencer("human")` resolves (`default_det_models.py`:
  `rtmdet_m_8xb32-100e_coco-obj365-person-235e8209`), loaded as the ONNX
  export from the OpenMMLab `onnx_sdk` bucket. Keep-filter: strict
  `score > 0.3`, matching the mmpose inferencer's cut (the committed raw's
  minimum bbox score is 0.30008).
- **Pose: RTMPose-L body7 COCO-17 @256x192.** The mmpose alias's own pose
  model is RTMPose-M body7; L is a deliberate capacity upgrade.

The five-array raw schema, `sticky_anchor` player selection, and collation are
byte-preserved; keypoint values are gated by parity against the committed
mmpose extraction. The 3D pose stream was removed upstream;
`requirements-legacy-3d.txt` remains only as the env for its parked revival
design.

## Detector selection

Both RTMDet person detectors were evaluated against the committed extraction
over the same stratified 200-clip sample, pose model held fixed:

| config | frames failed only by rtmlib : only by mmpose | dropped players | keypoint median / p90 |
|---|---|---|---|
| nano@320, keep > 0.3 | 50 : 7 | 5 clips | 1.74 / 4.58 px |
| nano@320, keep > 0.15 | 15 : 20 | 0 | 1.93 / 4.21 px |
| **M@640, keep > 0.3 (shipped)** | **0 : 3** | **0** | **1.72 / 3.93 px** |

nano passes only with its keep-threshold lowered to 0.15 and still loses
frames on hard contact clips. M at the standard cut loses none: the three
disagreeing frames are frames mmpose zeroed and rtmlib keeps. Deployed-output
parity for the shipped config reaches failed-frame agreement 1.000 on the
sample, with every clip inside the keypoint bounds.

Timed benchmarks (A100, same clips, medians; `bench_detector_pose_configs.py`):

| config | total ms/frame | fps | kp px vs deployed |
|---|---|---|---|
| nano + L256, keep > 0.15 | 243 | 4.1 | 1.96 |
| nano + L256, keep > 0.3 | 45 | 22.1 | 2.00 |
| **M + L256, keep > 0.3 (shipped)** | **81** | **12.3** | **1.62** |
| M + L384, keep > 0.3 | 109 | 9.2 | 1.84 |

Pose cost scales with the crops kept: at 0.15, nano admits ~40 crowd boxes per
frame and pays pose on each, so the shipped M configuration is both the
closest match to the deployed extraction and ~3x faster than nano at 0.15.
nano at 0.3 is the fast option, at the price of the dropped players above.

## Known differences vs the committed extraction

- mmpose ran a second NMS at IoU 0.3 after its score cut; the ONNX bakes IoU
  0.5 and the adapter adds none, so box pairs with IoU in (0.3, 0.5] survive
  here that mmpose suppressed. The excess is crowd, which `sticky_anchor`
  geometry-rejects.
- mmpose substituted a full-frame person when zero detections survived its
  cut; the adapter returns none by design. 4 of 160,940 sampled committed
  detections are that fabricated person.

## Verifying

- `validation_scripts/rtmlib_migration/download_and_verify_models.py`
  downloads both ONNX files, checks their SHA-256 against committed pins, and
  vendors them to the pool.
- CPU checks (repo root, `PYTHONPATH=src/bst_x:src`, venv per
  `preparing_data/requirements.txt`): `gate_raw_schema`,
  `adapter_contract_test`, `gate_cpu_determinism` (`OMP_NUM_THREADS=1`),
  `gate_dtype_parity`, `gate_keypoint_value`, `gate_cpu_downstream_byteeq`,
  `gate_deployed_parity`.
- GPU checks (CUDA box; env recipe in `preparing_data/requirements.txt`):
  `gate_cuda_selfvariance`, then `gate_gpu_parity`, then `phase_a_decision`
  over their JSON outputs. Paths and knobs are `RTMLIB_GATE_*` env vars,
  documented in each script's docstring.
- Timed config benchmark: `bench_detector_pose_configs.py` (CPU, or
  `RTMLIB_GATE_DEVICE=cuda`).
