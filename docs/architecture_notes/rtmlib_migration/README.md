# rtmlib migration: overview

> **Detector corrected (2026-07-04):** the migration first shipped RTMDet-nano
> @320 on a false "byte-identical to mmpose's" claim; mmpose actually ran
> RTMDet-M @640. Restored, with the 0.3 keep-threshold, in
> [07_detector_restoration.md](07_detector_restoration.md) — the nano-era
> numbers below are the superseded audit trail.

Moves the 2D pose-extraction path (`raw_extract`, `detect_players_2d`) off the pinned
mmpose / mmcv / mmdet / mmengine stack onto rtmlib (onnxruntime, numpy-2 clean, no source
builds). Models as restored: RTMDet-M person 640x640 (the detector
`MMPoseInferencer("human")` actually resolved) plus RTMPose-L body7 COCO-17 (a deliberate
upgrade over the alias's RTMPose-M body7).

## Outcome

Phase-A: GO. Over a stratified all-courts sample (200 clips, all 40 extractable
video-ids), rtmlib matches or beats the mmpose extraction on the Phase-A parity metrics
(aggregate keypoint median 1.9px [gate 5px], failed-rate parity 0.01pp [gate 2pp], zero
dropped players) on 199/200 clips. Downstream classification F1 is not measured here; that
is Phase B. One hard motion-blur clip (`2_1_10_2`) is an accepted, documented residual.
Result and reproduction: `06_phase_a_decision.md`.

## Key decisions (metrics)

- Detector keep-threshold 0.3 to 0.15. The fixed-320 RTMDet under-scores players
  (0.10-0.30, median 0.18) that mmpose, run at a larger test size, scored above ~0.3, so
  0.3 dropped them on hard/contact frames (authoritative run: 5 clips lost a player,
  per-clip loss to 18.75pp). 0.15 recovers them; `sticky_anchor` geometry-rejects the
  extra crowd. It is a post-inference score filter; the model, weights, SHA pins and the
  320 input are unchanged.
- Joints delta is report-only. The bbox-normalised `jntMed` is confounded by the known
  body7-vs-old-RTMPose-L drift (keypoint median ~1.4px, partly L/R relabeling). Keypoint
  fidelity is gated in raw pixels (`kp_med`/`kp_p90`), which pass with margin.
- Accepted residual `2_1_10_2`. One clip loses 5 frames to sub-0.15 under-scoring: 1/200,
  aggregate parity intact; chasing it to 0.10 trades crowd for marginal gain.
- Two gate-metric notes (logged for Phase-B, not defects): the per-clip failed-rate gate
  is symmetric, so it flags clips where rtmlib beats mmpose (`33_1_12_2`, `43_1_10_2`); and
  G8's confident-p90 lacks the L/R-swap correction G1 applies (`5_1_10_1`).

## What this means for you

- The 2D extraction path now runs rtmlib; install via `preparing_data/requirements.txt`
  plus the GPU recipe in `05_gpu_handoff.md`. The dormant 3D path still needs the legacy
  OpenMMLab stack (`requirements-legacy-3d.txt`).
- The committed keypoint dataset is unchanged until the Phase-B 30,487-stem re-extract at
  0.15 (the mmpose baseline covered 32,203 clips) and retrain, which measures macro/min-F1
  against the baseline.

## Docs

| | |
|---|---|
| `00_findings_and_scope.md` | touch-point map + scope |
| `01_runbook.md` | env + run notes |
| `02_adapter_design.md` | the adapter (raw-array parity, RGB fix, `ndet` analysis) |
| `03_verification.md` | the gate ladder (byte-equal vs parity) |
| `04_adversarial_review.md` | gate-review findings |
| `05_gpu_handoff.md` | the Bourbaki GPU run loop (G7-G9) |
| `06_phase_a_decision.md` | the nano-era Phase-A decision (superseded audit trail) |
| `07_detector_restoration.md` | the detector correction: nano to RTMDet-M@640, threshold back to 0.3 — start here for current state |
