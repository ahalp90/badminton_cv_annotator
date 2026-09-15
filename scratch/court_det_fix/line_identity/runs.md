# Run record

Exact commands, hashes, timings and exit codes. Local commands run from this folder with `~/.venvs/badminton-cicd/bin/python`; every input resolves inside the repository through `common.py`. `R` is the remote experiment root and `L` the frozen helper tree (`scratch/court_det_fix/worklog/checks/independent/player_guided/20260914`), both mapped in the gitignored `paths.local.sh`.

## Inputs read

- The nine frozen input packs and native frames the direction experiment used (`common.PACKS`, `common.frame_path`).
- Saved baseline direction records: `L/vp_pruning/coverage/results/<case>.json.gz`.
- Direction-experiment records: `scratch/court_det_fix/direction_agreement/runs/direction_agreement_20260915_144900/{e0,e2,e3}/<case>.json.gz`.
- Courts-before-the-gate records (run `pregate_20260915_115949`, gitignored): `<follow-ups>/pregate_loss/records/new/<arm>/results/<case>.json.gz` where `<follow-ups>` is `scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS`. MD5 in that folder's `manifest.json`.
- Baseline generation records for GX0 and Amateur-3 frame 0, pulled from `R/automatic_axes_20260914/results/` into the cruft folder (gitignored): MD5 recorded in `evidence.md`.

## Local stages

| Stage | Command | When (AEST) | Exit | Notes |
| --- | --- | --- | ---: | --- |
| axis replay | `axis_replay.py --pregate-records <follow-ups>/pregate_loss/records/new --baseline-records <cruft>/baseline_records --output runs/axis_replay --uncapped-combination --pairs gxBQ_window_00_frame_0:M:best am3_window_00_frame_0:R:best am3_window_00_frame_0:R:nearest am3_window_00_frame_0:B:best gxBQ_window_00_frame_0:B:nearest gxBQ_window_00_frame_0:B:best` | 08:05, rerun 08:20 with the uncapped combination | 0 | six pairs; gates in `runs/axis_replay/run.log` |
| paint profiles | `paint_profiles.py --output runs/paint_profiles` | 08:35, rerun 08:50 with two-sided flanks | 0 | nine views, 5,370 fragments |

## Prior checks carried in

`prior_checks/` holds the scripts, tables, notes, gate outputs and job receipts of the three follow-up checks this series continues (cap loss, person-box mask replay, courts before the player gate), copied from the gitignored follow-ups folder with the private paths replaced by `R` and `<follow-ups>`. Their inputs (the packs, the saved direction records and the pulled generation records) stay in the gitignored trees named above; the two copied scripts that resolve the repository root from their own location were adjusted for the new depth and nothing else.
