# rtmlib migration — overview

Moves the 2D pose-extraction path (`raw_extract`, `detect_players_2d`) off the pinned
mmpose / mmcv / mmdet / mmengine stack onto **rtmlib** (onnxruntime, numpy-2 clean, no
source builds), keeping the **same model family** — RTMDet-nano person 320×320 +
RTMPose-L body7 COCO-17. The detector ONNX is byte-identical to mmpose's; the pose model
is the updated body7 RTMPose-L.

## Outcome

**Phase-A: GO.** Over a stratified all-courts sample (200 clips, all 40 extractable
video-ids), rtmlib **matches or beats** the mmpose baseline on 199/200 clips — aggregate
keypoint median 1.9px (gate 5px), failed-rate parity 0.01pp (gate 2pp), zero dropped
players. One hard motion-blur clip (`2_1_10_2`) is an accepted, documented residual.
Result + reproduction: **`06_phase_a_decision.md`**.

## Key decisions (metrics)

- **Detector keep-threshold 0.3 → 0.15.** The fixed-320 RTMDet under-scores players
  (0.10–0.30, median 0.18) that mmpose — run at a larger test size — scored above ~0.3,
  so 0.3 dropped them on hard/contact frames (authoritative run: 5 clips lost a player,
  per-clip loss to 18.75pp). 0.15 recovers them; `sticky_anchor` geometry-rejects the
  extra crowd. A **post-inference score filter — model, weights, SHA pins and the 320
  input are unchanged.**
- **Joints delta is report-only.** The bbox-normalised `jntMed` is confounded by the
  known body7-vs-old-RTMPose-L drift (keypoint median ~1.4px, partly L/R relabeling).
  Keypoint fidelity is gated in **raw pixels** (`kp_med`/`kp_p90`), which pass with margin.
- **Accepted residual `2_1_10_2`.** One clip loses 5 frames to sub-0.15 under-scoring —
  1/200, aggregate parity intact; chasing it to 0.10 trades crowd for marginal gain.
- **Two gate-metric notes** (logged for Phase-B, not defects): the per-clip failed-rate
  gate is symmetric, so it flags clips where rtmlib *beats* mmpose (`33_1_12_2`,
  `43_1_10_2`); and G8's confident-p90 lacks the L/R-swap correction G1 applies
  (`5_1_10_1`).

## What this means for you

- The 2D extraction path now runs rtmlib — install via `preparing_data/requirements.txt`
  plus the GPU recipe in `05_gpu_handoff.md`. The dormant 3D path still needs the legacy
  OpenMMLab stack (`requirements-legacy-3d.txt`).
- The **committed keypoint dataset is unchanged** until the Phase-B re-extract (32,203
  clips at 0.15) + retrain, which measures macro/min-F1 against the baseline.

## Docs

| | |
|---|---|
| `00_findings_and_scope.md` | touch-point map + scope |
| `01_runbook.md` | env + run notes |
| `02_adapter_design.md` | the adapter (raw-array parity, RGB fix, `ndet` analysis) |
| `03_verification.md` | the gate ladder (byte-equal vs parity) |
| `04_adversarial_review.md` | gate-review findings |
| `05_gpu_handoff.md` | the Bourbaki GPU run loop (G7–G9) |
| `06_phase_a_decision.md` | **the authoritative Phase-A decision — start here for the result** |
