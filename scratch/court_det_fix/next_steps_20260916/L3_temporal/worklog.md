# L3 temporal pilot worklog

## Resume block

Next action: hand the completed L3 packet to W3. Do not rerun the scorer. The current
result is a conditional sampled-view opportunity with a ranking failure. Verified outputs
are `result.md`, `W3_PACKET.md`, `source_manifest.json`, `score_matrix.json`,
`score_matrix.csv`, `view_alignment.json` and six overlays. The runbook is the L3 prompt
at `scratch/court_det_fix/worklog/webui_further_followups_16092026/prompts/L3_cached_temporal_opportunity.md`.

## Concerns and observations

- [B1] The user expanded the scoring sample beyond the prompt's original three-frame cap.
  The packet names this as a seven-frame exploratory extension.
- [B1] Registration is sampled evidence only. Moving people and captions create large
  held-out outliers even though the static background aligns.
- [B1] References were joined only for locked selected and existing contrary courts.
  They were not used to choose the panel, frame set or winner.

## Module state

- `run_l3_temporal.py` imports the documented frozen helper roots and uses the existing
  target-frame `evaluate_pool` path. It does no candidate generation.
- The output matrix has one fixed 30-court panel, seven target frames, 210 `ok` scores,
  exact origin diagonals and a common subset of 30.
- The registration JSON records target-to-anchor homographies, grid displacement,
  inlier coverage, holdout residuals and overlay paths.

## Initial readiness

The cached GX pack and local copy had matching MD5 values. All seven images were present
at 1920x1080. The two origin records were complete all-camera pools using the same stage.
No dirty user files existed at the initial checkpoint.

## Execution log

### Batch 1 — cached temporal scoring and view diagnostic

- Files: `run_l3_temporal.py` and all files in this directory.
- Change: score the fixed origin panel on the seven user-authorised GX frames and record
  the one static-feature registration diagnostic.
- Gate: Python compile green; scoped Ruff green; Serena/Pyrefly diagnostics `{}`; one
  process completed with panel 30, common 30, frames 7, six registration pairs `ok`.
- Commit: close-out commit to be recorded in Git history.
