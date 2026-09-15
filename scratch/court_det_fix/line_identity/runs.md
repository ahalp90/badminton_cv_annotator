# Run record

Exact commands, hashes, timings and exit codes. Local commands run from this folder with `~/.venvs/badminton-cicd/bin/python`; inputs resolve inside the repository checkout through `shared.py`, some of them in gitignored trees (the follow-ups folder and the cruft folder named below). `R` is the remote experiment root, `L` the frozen helper tree (`scratch/court_det_fix/worklog/checks/independent/player_guided/20260914`) and `<cruft>` this session's untracked working folder (`scratch/court_det_fix/worklog/claude_session_16092026`); `paths.local.sh` (gitignored) maps `R` and the cruft folder to real locations.

## Inputs read

- The nine frozen input packs and native frames the direction experiment used (`shared.PACKS`, `shared.frame_path`).
- Saved baseline direction records: `L/vp_pruning/coverage/results/<case>.json.gz`.
- Direction-experiment records: `scratch/court_det_fix/direction_agreement/runs/direction_agreement_20260915_144900/{e0,e2,e3}/<case>.json.gz`.
- Courts-before-the-gate records (run `pregate_20260915_115949`, gitignored): `<follow-ups>/pregate_loss/records/new/<arm>/results/<case>.json.gz` where `<follow-ups>` is `scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS`. MD5 in that folder's `manifest.json`.
- Baseline generation records for GX0 and Amateur-3 frame 0, pulled from `R/automatic_axes_20260914/results/` into the cruft folder (gitignored): MD5 recorded in `evidence.md`.

## Local checks

Run from this folder; rerun after every code change.

```text
~/.venvs/badminton-cicd/bin/ruff check . --exclude geometry_certificate.py --exclude prior_checks   # exit 0
PYTHONDONTWRITEBYTECODE=1 ~/.venvs/badminton-cicd/bin/pytest tests -q -p no:cacheprovider          # 4 passed
```

`geometry_certificate.py` is the WebUI direction audit's file, kept byte-identical to its source in the follow-ups folder (`gx_directions/`), and `prior_checks/` holds copies of earlier scripts; neither is restyled.

## Local stages

| Stage | Command | When (AEST) | Exit | Notes |
| --- | --- | --- | ---: | --- |
| axis replay | `axis_replay.py --pregate-records <follow-ups>/pregate_loss/records/new --baseline-records <cruft>/baseline_records --output runs/axis_replay --uncapped-combination --pairs gxBQ_window_00_frame_0:M:best am3_window_00_frame_0:R:best am3_window_00_frame_0:R:nearest am3_window_00_frame_0:B:best gxBQ_window_00_frame_0:B:nearest gxBQ_window_00_frame_0:B:best` | 08:00, 08:05 with the uncapped combination, 08:22 with the per-pair fourth gate | 0 | six pairs; gates in `runs/axis_replay/run.log` |
| paint profiles | `paint_profiles.py --output runs/paint_profiles` | 08:07, 08:10 with two-sided flanks, 08:19 with two flank windows and the marking label | 0 | nine views, 5,370 fragments |
| filter replay | `filter_replay.py --output runs/filter_replay --uncapped-combination` | 08:22 (first), 08:29 with the marking columns, 08:45 with the observation-only arms and reproducible inputs | 0 | nine views, six direction arms, six axis arms; writes `inputs/<arm>/` |
| tables | `tabulate.py` | after the replay | 0 | Markdown tables for evidence.md and results.md |

## Prior checks carried in

`prior_checks/` holds the scripts, tables, notes, gate outputs and job receipts of the three follow-up checks this series continues (cap loss, person-box mask replay, courts before the player gate), copied from the gitignored follow-ups folder with the private paths replaced by `R` and `<follow-ups>`. Their inputs (the packs, the saved direction records and the pulled generation records) stay in the gitignored trees named above; the two copied scripts that resolve the repository root from their own location were adjusted for the new depth and nothing else. `pregate_loss/gate1.py` and `analyse.py` read `records/` beside themselves and `../cap_loss/records/saved/`, which exist only in the original follow-ups folder; rerun them there.
