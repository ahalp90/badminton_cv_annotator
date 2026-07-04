# Detector restoration: rtmdet-nano@320 back to RTMDet-M@640

> **Status (2026-07-04):** code restored (adapter constants + SHA pin) and all
> CPU gates green, including the G1/G6 parity reruns — G6 hit fmatch 1.000 on
> all 20 clips with a 0:0 directional split. Only the Bourbaki GPU ladder
> (G7/G8/G9, `_m` JSONs) remains. Until G9 re-adjudicates, the nano-era
> Phase-A GO (06) authorises nothing.

## What happened

The 2026-07-01 migration to rtmlib chose the `rtmdet-nano-person` ONNX
(`05d8511e`) at 320x320 on the claim it was "byte-identical to the detector
inside `MMPoseInferencer("human")`" (00 §4). That claim carried no recorded
evidence anywhere in docs 00-06, and it is false. The mis-identification then
drove real cost: nano@320 under-scored players mmpose kept, producing the
one-directional frame-loss bias, the G-4 NO-GO, and the `DET_SCORE_THR`
0.3→0.15 recalibration (06) — all compensation for an unintended detector
downgrade. The docs even recorded the contradiction without seeing it: "mmpose
ran the RTMDet `.pth` at a larger test size" (02, 06) cannot coexist with
byte-identity to a fixed-320 ONNX.

## Ground truth (primary sources, verified 2026-07-04)

mmpose is pinned at 1.3.2 (`requirements-legacy-3d.txt`; mmdet 3.2.0,
mmengine 0.10.4, mmcv 2.1.0). At tag v1.3.2 — and identically at v1.3.0/1.3.1,
with RTMDet-M already the choice at v1.2.0 —
`mmpose/apis/inferencers/utils/default_det_models.py` resolves `human` (and
`body`, `wholebody`) to:

```python
human=dict(
    model=osp.join(mmpose_path, '.mim', 'demo/mmdetection_cfg/'
        'rtmdet_m_640-8xb32_coco-person.py'),
    weights='https://download.openmmlab.com/mmpose/v1/projects/'
    'rtmposev1/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
    cat_ids=(0, )),
```

- Det config (`rtmdet_m_640-8xb32_coco-person.py`, mmpose v1.3.2, inheriting
  mmdet v3.2.0 `rtmdet_m_8xb32-300e_coco.py`): test pipeline
  `Resize(scale=(640, 640), keep_ratio=True)` + `Pad(640, pad_val=114)`;
  test_cfg `score_thr=0.05`, NMS IoU 0.6, `max_per_img=100`.
- Inferencer defaults (`pose2d_inferencer.py` v1.3.2 `preprocess_single`):
  keep `scores > bbox_thr` with `bbox_thr=0.3` (strict), then NMS IoU 0.3;
  when zero boxes survive, a full-frame bbox with score 1.0 is substituted.
- The `human` alias *pose* model (the only metafile line defining the alias,
  `configs/body_2d_keypoint/rtmpose/body8/rtmpose_body8-coco.yml`):
  RTMPose-M body7, checkpoint `rtmpose-m_simcc-body7_..._e48f03d0` @256x192 —
  not RTMPose-L as `mmpose_changes.md` recorded.
- The nano checkpoint `05d8511e` exists in mmpose only as the
  `projects/rtmpose` demo detector (also rtmlib's zoo default); nothing in
  `mmpose/apis/inferencers` references it at any tag checked.
- Data corroboration (committed raw, 1,500-clip sample): min `bbox_score`
  0.30008 (the strict 0.3 cut); zero `ndet==0` frames with 4/160,940
  detections at score exactly 1.0 (the full-frame fallback, leaked rarely);
  ndet mean 6.63, max at the N_max=16 cap.

## The restoration

Scope decisions (2026-07-04, with the user): **detector only** — RTMPose-L
body7 stays, now documented as a deliberate upgrade over the alias's
RTMPose-M body7 rather than "the same model". Ship `DET_SCORE_THR = 0.3`
(mmpose's cut; the adapter's keep-filter was already strict `>`), validate
through the existing gate ladder.

Code delta (one commit): `rtmlib_pose.py` `DET_URL` →
`rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.zip` (same
`rtmposev1/onnx_sdk/` bucket, uploaded 7 seconds after the nano zip — same
mmdeploy export batch, same NMS-baked `dets (1,N,5)` + `labels` output, so
`RTMDetScored` is unchanged), `DET_INPUT_SIZE` → (640, 640), `DET_SCORE_THR`
→ 0.3. New SHA-256 pin (extracted `.onnx`, identical across two independent
downloads):
`4f4d7e07350b1753299111d1ae500fd64447a5b0e38e4bacbefab6573c742d30`.

Known, accepted semantic gaps vs mmpose (measured, not patched):

- **NMS:** mmpose ran a second NMS at IoU 0.3 after its score cut; the ONNX
  bakes IoU 0.5 and the adapter adds none. Box pairs with IoU in (0.3, 0.5]
  survive here that mmpose suppressed → a small positive ndet excess
  (crowd), which `sticky_anchor` geometry-rejects. Quantified by the G8 ndet
  distribution; an adapter-side NMS is added only if parity gates demand it.
- **Empty frames:** mmpose fabricated a full-frame person; the adapter
  returns m=0 by design. Affects the handful of zero-survivor frames
  (4/160,940 mmpose-side); shows up as tiny ndet 0-vs-1 deltas.

## Evidence so far (2026-07-04, dev box CPU)

- ONNX graph: input `[1,3,640,640]` static; outputs `dets (1,N,5)` +
  `labels`. Blank-frame guard still returns m=0.
- Prototype clip `11_1_10_2` vs committed mmpose raw: 62/62 frames; kp L2
  median **2.17px** (nano era: 2.54), confident-joints median **1.43px**;
  ndet mean 8.21 / max 13 (mmpose 6.63 / 10 — the expected NMS-gap crowd
  excess); frames with fewer than 2 detections: **0**. ~0.49 s/frame CPU.
- Gates: G2 adapter contract PASS (score floor auto-tracked to 0.3), G4 CPU
  determinism bit-identical (20 frames), G3 dtype parity PASS, raw-schema
  PASS.
- G1/G6 reruns (user, dev CPU, same day): G1 PASS on all 5 clips with medians
  1.17-2.17px and swap-fraction at most 5%; G6 PASS on all 20 clips at fmatch
  1.000, posMed 0.0000, jntMed at most 0.0043, and **zero** directional frame
  loss in either direction — the deployed failed-frame decisions reproduce
  exactly on this sample, where the nano era managed mean 0.98 / min 0.86 with
  a one-directional bias.

## Rerun ladder and status

| Gate | Scope | Env | Status |
|---|---|---|---|
| G2/G3/G4/raw-schema | contract/dtype/determinism/schema | dev CPU | **green** (2026-07-04) |
| G1 keypoint value | vs committed raw | dev CPU (user) | **green** (2026-07-04): 5 clips, medians 1.17-2.17px [gate 5], conf p90 2.48-5.34 [gate 12], L/R-swap 0-5% [cap 20%], RGB counterfactual byte-exact |
| G6 deployed parity | vs committed clean | dev CPU (user) | **green** (2026-07-04): 20 clips, fmatch 1.000 on every clip, dF=0, posMed 0.0000, jntMed max 0.0043 [gate 0.03], directional loss 0:0 |
| G7 self-variance | fresh CUDA floors | Bourbaki (user) | pending — write `g7_selfvariance_m.json` |
| G8 parity smoke50 + authoritative | vs committed raw | Bourbaki (user) | pending — write `g8_parity_m_*.json` |
| G9 Phase-A re-decision | verdict | Bourbaki run, adjudicated | pending |

Nano-era artifacts (`g7_selfvariance.json`, `g8_parity_*.json`, tee logs, the
vendored nano `.onnx`) are evidence — kept, never overwritten; all M runs
write `_m` names. Expected shape of the rerun: the directional frame-loss
split collapses toward symmetric noise, the 06 residual `2_1_10_2` and the
`4_1_10_x` dropped-player family resolve, and over-detection (>16) warnings
stay near the baseline rate (mmpose itself hit the 16 cap).

Phase-B consequences: collation tag moves off `rtmlib_015` (0.15 is no longer
the shipped threshold); the external `phase_b_run.sh` must drop
`--skip-trajectory` (main removed the step) and point at fresh output dirs so
`_raw_ndet.npy` resume markers cannot silently reuse nano-era arrays.

## Corrected records

`rtmlib_pose.py` prose, `mmpose_changes.md` (resolved-models row + speed
note), `requirements.txt` header, gate narrative strings
(`gate_deployed_parity`, `phase_a_decision`, `gate_gpu_parity`),
`diag_g4_fails.py` (marked historical), `frontend/src/utils/adapters.js`
(detector name + ~64M perception-stack params), `raw_ndet_stats_outputs/baseline_2026-04-29.md`
and `mmpose_heuristic/phase1_vs_phase2_2026-04-29.md` (dated corrections),
banners on 00/02/03/05/06/README. Docs 00-06 stay whole as the audit trail of
how the error happened and what it cost.
